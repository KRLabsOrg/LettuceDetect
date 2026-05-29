"""Post-processing: expand source-cache context so every function call is backed by evidence.

For each cached instance this script runs two passes:

  Pass A — internal imports
    Resolves relative AND absolute-internal imports in source_files, fetches
    the referenced definitions from the bare git repo, stores them in
    ``dependency_files``.  Covers functions defined inside the same repo.

  Pass B — external library docs (Context7)
    Extracts the specific names imported from external libraries (numpy,
    pandas, sklearn, …) and fetches targeted per-function docs from Context7.
    Stores results in ``external_docs``.

After both passes the script rebuilds every sample's prompt so that every
function call in the answer is backed by either its source definition or its
API docs — no more phantom function calls that look like hallucinations.

Usage::

    python scripts/expand_context.py [--dry-run] [--limit N] [--no-context7]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.code_hallucination.config import (  # noqa: E402
    DATA_DIR,
    DATASET_PATH,
    INSTANCES_PATH,
    METADATA_PATH,
    REPOS_DIR,
    SOURCE_CACHE_DIR,
)
from scripts.code_hallucination.context7_docs import get_documentation_for_instance  # noqa: E402
from scripts.code_hallucination.sample_assembler import build_prompt  # noqa: E402
from scripts.code_hallucination.source_fetcher import fetch_import_dependencies  # noqa: E402

warnings.filterwarnings("ignore", category=SyntaxWarning)

_LIGHT_INDEX_PATH = DATA_DIR / "swebench_index_light.json"
EXPANDED_DATASET_PATH = DATA_DIR / "code_hallucination_data_expanded.json"
STATS_REPORT_PATH = DATA_DIR / "expand_context_stats.txt"


# ── Index helpers ──────────────────────────────────────────────────────────────


def _load_swebench_index(instances_path: Path) -> dict[str, dict]:
    if _LIGHT_INDEX_PATH.exists():
        with open(_LIGHT_INDEX_PATH) as f:
            return json.load(f)

    if not instances_path.exists():
        print(f"WARNING: instances file not found at {instances_path}")
        return {}

    print("  Building lightweight index from full instances file (one-time)...")
    with open(instances_path) as f:
        instances = json.load(f)
    index = {
        i["instance_id"]: {"repo": i["repo"], "base_commit": i["base_commit"]}
        for i in instances
    }
    with open(_LIGHT_INDEX_PATH, "w") as f:
        json.dump(index, f)
    print(f"  Saved to {_LIGHT_INDEX_PATH}")
    return index


def _load_existing_dataset(
    dataset_path: Path, metadata_path: Path
) -> tuple[list[dict], list[dict]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    with open(dataset_path) as f:
        samples = json.load(f)
    with open(metadata_path) as f:
        metadata = json.load(f)
    return samples, metadata


# ── Main processing loop ───────────────────────────────────────────────────────


def process_instances(
    metadata: list[dict],
    swebench_index: dict[str, dict],
    dry_run: bool,
    limit: int | None,
    use_context7: bool,
) -> dict:
    """Expand source-cache entries with import deps (Pass A) and external docs (Pass B)."""
    stats = {
        "unique_instances": 0,
        "instances_with_repo": 0,
        "instances_skipped_no_repo": 0,
        "pass_a_expanded": 0,
        "pass_a_defs_found": 0,
        "pass_b_expanded": 0,
        "pass_b_docs_fetched": 0,
    }

    # Deduplicate by instance_id
    seen: set[str] = set()
    instance_order: list[str] = []
    for meta in metadata:
        iid = meta.get("instance_id", "")
        if iid and iid not in seen:
            seen.add(iid)
            instance_order.append(iid)

    stats["unique_instances"] = len(instance_order)
    if limit is not None:
        instance_order = instance_order[:limit]

    print(f"Processing {len(instance_order)} unique instances "
          f"({'dry-run' if dry_run else 'writing'}, context7={'on' if use_context7 else 'off'})...")

    for i, instance_id in enumerate(instance_order, 1):
        if i % 500 == 0 or i == len(instance_order):
            print(f"  [{i}/{len(instance_order)}] "
                  f"pass_a={stats['pass_a_expanded']} pass_b={stats['pass_b_expanded']}")

        swe = swebench_index.get(instance_id)
        if not swe:
            continue

        repo = swe.get("repo", "")
        commit = swe.get("base_commit", "")
        if not repo or not commit:
            continue

        repo_dir = REPOS_DIR / repo.replace("/", "__")
        if not repo_dir.exists():
            stats["instances_skipped_no_repo"] += 1
            continue

        stats["instances_with_repo"] += 1

        cache_path = SOURCE_CACHE_DIR / f"{instance_id}.json"
        if not cache_path.exists():
            continue

        with open(cache_path) as f:
            entry = json.load(f)

        source_files: dict[str, str] = entry.get("source_files", {})
        if not source_files:
            continue

        dirty = False

        # ── Pass A: internal imports ───────────────────────────────────────
        if not entry.get("dependency_files"):
            try:
                deps = fetch_import_dependencies(source_files, repo_dir, commit)
            except Exception as e:
                print(f"  WARNING Pass A {instance_id}: {e}")
                deps = {}
            if deps:
                stats["pass_a_expanded"] += 1
                stats["pass_a_defs_found"] += len(deps)
                if not dry_run:
                    entry["dependency_files"] = deps
                    dirty = True

        # ── Pass B: external library docs via Context7 ─────────────────────
        if use_context7 and not entry.get("external_docs"):
            try:
                ext_docs = get_documentation_for_instance(
                    changed_files=entry.get("changed_files", []),
                    patch="",
                    problem_statement="",
                    repo=repo,
                    source_files=source_files,
                )
            except Exception as e:
                print(f"  WARNING Pass B {instance_id}: {e}")
                ext_docs = {}
            if ext_docs:
                stats["pass_b_expanded"] += 1
                stats["pass_b_docs_fetched"] += sum(len(v) for v in ext_docs.values())
                if not dry_run:
                    entry["external_docs"] = ext_docs
                    dirty = True

        if dirty:
            with open(cache_path, "w") as f:
                json.dump(entry, f)

    return stats


# ── Dataset rebuild ────────────────────────────────────────────────────────────


def rebuild_expanded_dataset(
    samples: list[dict],
    metadata: list[dict],
) -> list[dict]:
    """Rebuild every sample's prompt using the updated source-cache entries."""
    updated = []

    for sample, meta in zip(samples, metadata):
        instance_id = meta.get("instance_id", "")
        cache_path = SOURCE_CACHE_DIR / f"{instance_id}.json"

        if not cache_path.exists():
            updated.append(sample)
            continue

        with open(cache_path) as f:
            entry = json.load(f)

        source_files = entry.get("source_files", {})

        # Merge internal deps (both relative and absolute-internal)
        dep_files: dict = dict(entry.get("dependency_files", {}))

        # Merge external docs into the documentation dict for build_prompt
        ext_docs: dict[str, str] = entry.get("external_docs", {})

        # Extract user query and any existing doc blocks from original prompt
        original_prompt = sample.get("prompt", "")
        user_match = re.search(r"User request:\s*(.*?)$", original_prompt, re.DOTALL)
        user_query = user_match.group(1).strip() if user_match else ""

        doc_blocks: dict[str, str] = dict(ext_docs)
        for m in re.finditer(
            r"Documentation for ([^\n:]+):\n(.*?)(?=\n\nFile:|\n\nUser request:|\n\nReferenced definitions:|$)",
            original_prompt,
            re.DOTALL,
        ):
            key = m.group(1).strip()
            if key not in doc_blocks:
                doc_blocks[key] = m.group(2).strip()

        new_prompt = build_prompt(
            source_files,
            doc_blocks,
            user_query,
            dependency_files=dep_files or None,
        )

        s = dict(sample)
        s["prompt"] = new_prompt
        updated.append(s)

    return updated


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand context: internal import deps + targeted external docs."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Process but do not write any files.")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Process at most N unique instances.")
    parser.add_argument("--no-context7", action="store_true",
                        help="Skip Context7 external docs (Pass B).")
    args = parser.parse_args()

    print("Loading SWE-bench index...")
    swebench_index = _load_swebench_index(INSTANCES_PATH)
    print(f"  {len(swebench_index)} instances loaded.")

    print("Loading existing dataset...")
    samples, metadata = _load_existing_dataset(DATASET_PATH, METADATA_PATH)
    print(f"  {len(samples)} samples loaded.")

    stats = process_instances(
        metadata, swebench_index,
        dry_run=args.dry_run,
        limit=args.limit,
        use_context7=not args.no_context7,
    )

    if not args.dry_run:
        print("\nRebuilding expanded dataset...")
        updated_samples = rebuild_expanded_dataset(samples, metadata)
        EXPANDED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EXPANDED_DATASET_PATH, "w") as f:
            json.dump(updated_samples, f, indent=2)
        print(f"  Written: {EXPANDED_DATASET_PATH}")

    report_lines = [
        "=== expand_context.py stats ===",
        f"  Unique instances:           {stats['unique_instances']}",
        f"  Instances with local repo:  {stats['instances_with_repo']}",
        f"  Instances skipped (no repo):{stats['instances_skipped_no_repo']}",
        f"  Pass A expanded:            {stats['pass_a_expanded']}",
        f"  Pass A definitions found:   {stats['pass_a_defs_found']}",
        f"  Pass B expanded:            {stats['pass_b_expanded']}",
        f"  Pass B docs chars fetched:  {stats['pass_b_docs_fetched']}",
        f"  Dry-run:                    {args.dry_run}",
    ]
    report = "\n".join(report_lines)
    print("\n" + report)

    if not args.dry_run:
        with open(STATS_REPORT_PATH, "w") as f:
            f.write(report + "\n")
        print(f"\nStats written to {STATS_REPORT_PATH}")


if __name__ == "__main__":
    main()
