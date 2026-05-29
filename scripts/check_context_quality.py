"""Audit context quality for 100 instances.

For each instance:
  1. Run Pass A (fetch_import_dependencies) to get dependency definitions.
  2. Build the full prompt.
  3. Extract bare function calls from the answer.
  4. Check which calls have evidence in the prompt (import stmt OR definition).
  5. Print per-instance verdict: CLEAN / GAPS.

Usage::
    python scripts/check_context_quality.py [--limit N] [--clean-only] [--show-prompt]
"""

from __future__ import annotations

import argparse
import ast
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
    METADATA_PATH,
    REPOS_DIR,
    SOURCE_CACHE_DIR,
)
from scripts.code_hallucination.sample_assembler import build_prompt  # noqa: E402
from scripts.code_hallucination.source_fetcher import fetch_import_dependencies  # noqa: E402

warnings.filterwarnings("ignore", category=SyntaxWarning)

_LIGHT_INDEX_PATH = DATA_DIR / "swebench_index_light.json"

# Standard-library builtins that don't need context evidence
_BUILTINS = frozenset(
    dir(__builtins__) if isinstance(__builtins__, dict) else dir(__builtins__)
) | {
    "print",
    "len",
    "range",
    "enumerate",
    "zip",
    "map",
    "filter",
    "sorted",
    "list",
    "dict",
    "set",
    "tuple",
    "str",
    "int",
    "float",
    "bool",
    "type",
    "isinstance",
    "issubclass",
    "hasattr",
    "getattr",
    "setattr",
    "delattr",
    "open",
    "super",
    "next",
    "iter",
    "any",
    "all",
    "sum",
    "min",
    "max",
    "abs",
    "round",
    "hex",
    "oct",
    "bin",
    "ord",
    "chr",
    "repr",
    "hash",
    "id",
    "vars",
    "dir",
    "help",
    "input",
    "format",
    "staticmethod",
    "classmethod",
    "property",
    "object",
    "Exception",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "NotImplementedError",
    "RuntimeError",
    "StopIteration",
    "GeneratorExit",
    "AssertionError",
    "ImportError",
    "OSError",
    "IOError",
    "FileNotFoundError",
}

# Patterns that are clearly attribute calls — skip them
_ATTR_CALL_RE = re.compile(r"\b\w+\.\w+\s*\(")


def extract_bare_calls(code: str) -> set[str]:
    """Extract bare function call names from code (no attribute prefix, not builtins)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except SyntaxError:
        # Fall back to regex
        calls = set(re.findall(r"\b([a-zA-Z_]\w*)\s*\(", code))
        return calls - _BUILTINS

    calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Only bare Name calls (not attribute calls like obj.method())
        if isinstance(func, ast.Name):
            calls.add(func.id)
    return calls - _BUILTINS


def is_evidenced(name: str, prompt: str) -> bool:
    """Return True if *name* has some form of evidence in *prompt*.

    Evidence = import statement OR def/class definition present.
    """
    # import statement: "import name" or "from X import name"
    if re.search(rf"\bimport\b.*\b{re.escape(name)}\b", prompt):
        return True
    # definition: "def name" or "class name"
    if re.search(rf"\b(?:def|class)\s+{re.escape(name)}\b", prompt):
        return True
    # decorator reference: @name
    if re.search(rf"@{re.escape(name)}\b", prompt):
        return True
    return False


def load_index() -> dict[str, dict]:
    if _LIGHT_INDEX_PATH.exists():
        with open(_LIGHT_INDEX_PATH) as f:
            return json.load(f)
    print(f"WARNING: light index not found at {_LIGHT_INDEX_PATH}")
    return {}


def audit_instance(
    sample: dict,
    meta: dict,
    swebench_index: dict[str, dict],
) -> dict:
    """Run Pass A and check coverage for a single instance."""
    instance_id = meta.get("instance_id", "")
    answer = sample.get("answer", "")
    is_hallucinated = meta.get("is_hallucinated", False)

    cache_path = SOURCE_CACHE_DIR / f"{instance_id}.json"
    if not cache_path.exists():
        return {"instance_id": instance_id, "skip": "no_cache"}

    with open(cache_path) as f:
        entry = json.load(f)

    source_files: dict[str, str] = entry.get("source_files", {})
    if not source_files:
        return {"instance_id": instance_id, "skip": "no_source_files"}

    # Run Pass A (don't write results, just compute)
    swe = swebench_index.get(instance_id)
    dep_files: dict[str, str] = {}
    if swe:
        repo = swe.get("repo", "")
        commit = swe.get("base_commit", "")
        repo_dir = REPOS_DIR / repo.replace("/", "__") if repo else None
        if repo_dir and repo_dir.exists() and commit:
            try:
                dep_files = fetch_import_dependencies(source_files, repo_dir, commit)
            except Exception:
                dep_files = {}

    # Build the full prompt with dependency definitions
    original_prompt = sample.get("prompt", "")
    user_match = re.search(r"User request:\s*(.*?)$", original_prompt, re.DOTALL)
    user_query = user_match.group(1).strip() if user_match else ""

    prompt = build_prompt(
        source_files,
        {},  # No external docs for now
        user_query,
        dependency_files=dep_files or None,
    )

    # Extract bare calls from the answer (strip code fences first)
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", answer, re.DOTALL)
    code_to_check = "\n".join(code_blocks) if code_blocks else answer
    bare_calls = extract_bare_calls(code_to_check)

    # Check each call for evidence in the prompt
    evidenced = {name for name in bare_calls if is_evidenced(name, prompt)}
    missing = bare_calls - evidenced

    return {
        "instance_id": instance_id,
        "repo": meta.get("repo", ""),
        "is_hallucinated": is_hallucinated,
        "dep_files_found": len(dep_files),
        "bare_calls": sorted(bare_calls),
        "evidenced": sorted(evidenced),
        "missing": sorted(missing),
        "coverage_pct": round(100 * len(evidenced) / max(len(bare_calls), 1)),
        "prompt_chars": len(prompt),
        "prompt": prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit context quality for N instances.")
    parser.add_argument("--limit", type=int, default=100, metavar="N")
    parser.add_argument(
        "--clean-only", action="store_true", help="Only check clean (non-hallucinated) samples."
    )
    parser.add_argument(
        "--show-prompt", action="store_true", help="Print full prompt for each instance."
    )
    parser.add_argument(
        "--gaps-only", action="store_true", help="Only print instances with missing evidence."
    )
    args = parser.parse_args()

    print("Loading index and dataset...")
    index = load_index()
    with open(DATASET_PATH) as f:
        samples = json.load(f)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    print(f"  {len(samples)} samples, {len(index)} index entries")

    # Filter and limit
    pairs = list(zip(samples, metadata))
    if args.clean_only:
        pairs = [(s, m) for s, m in pairs if not m.get("is_hallucinated")]
    pairs = pairs[: args.limit]
    print(f"  Auditing {len(pairs)} instances...\n")

    results = []
    for i, (sample, meta) in enumerate(pairs, 1):
        r = audit_instance(sample, meta, index)
        results.append(r)
        if r.get("skip"):
            continue

        missing = r["missing"]
        status = "GAPS" if missing else "CLEAN"
        pct = r["coverage_pct"]
        dep = r["dep_files_found"]
        calls_total = len(r["bare_calls"])

        if args.gaps_only and not missing:
            continue

        sep = "─" * 70
        print(sep)
        print(f"[{i:3d}] {r['instance_id']}")
        print(f"      repo={r['repo']}  hallucinated={r['is_hallucinated']}")
        print(
            f"      dep_files_fetched={dep}  bare_calls={calls_total}  coverage={pct}%  → {status}"
        )

        if missing:
            print(f"      MISSING evidence for: {missing}")

        if r["evidenced"]:
            print(
                f"      evidenced: {r['evidenced'][:10]}{'...' if len(r['evidenced']) > 10 else ''}"
            )

        if args.show_prompt:
            print(f"\n--- PROMPT ({r['prompt_chars']} chars) ---")
            print(r["prompt"][:3000])
            if r["prompt_chars"] > 3000:
                print(f"... [{r['prompt_chars'] - 3000} chars truncated]")
            print("--- END PROMPT ---\n")

    # Summary stats
    audited = [r for r in results if not r.get("skip")]
    skipped = len(results) - len(audited)
    clean_instances = [r for r in audited if not r["is_hallucinated"]]
    hall_instances = [r for r in audited if r["is_hallucinated"]]

    def _stats(group: list[dict], label: str):
        if not group:
            return
        fully_covered = sum(1 for r in group if not r["missing"])
        avg_cov = sum(r["coverage_pct"] for r in group) / len(group)
        gap_counts = [len(r["missing"]) for r in group if r["missing"]]
        avg_gaps = sum(gap_counts) / max(len(gap_counts), 1)
        print(
            f"  {label}: {len(group)} instances | fully_covered={fully_covered} ({100 * fully_covered // len(group)}%) | avg_coverage={avg_cov:.1f}% | instances_with_gaps={len(gap_counts)} | avg_gaps_when_missing={avg_gaps:.1f}"
        )

    print("\n" + "═" * 70)
    print("SUMMARY")
    print("═" * 70)
    print(f"  Total audited: {len(audited)}  (skipped={skipped})")
    _stats(clean_instances, "Clean  ")
    _stats(hall_instances, "Halluc.")


if __name__ == "__main__":
    main()
