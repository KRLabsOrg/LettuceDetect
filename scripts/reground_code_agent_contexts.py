#!/usr/bin/env python3
"""Repair grounding in already-generated code-agent samples, in place.

For every sample whose answer references names absent from its context, resolve
those references at the base commit (same tiers as generation-time grounding)
and prepend the missing ``Referenced definitions`` block to the sample's
``context`` and ``prompt``. Labels, answers, and spans are never touched.

Usage:
    GITHUB_TOKEN=... python scripts/reground_code_agent_contexts.py \
        --data data/v2/code_agent [--splits train dev test] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.code_hallucination.answer_grounding import (  # noqa: E402
    remaining_ungrounded,
    render_definitions,
    resolve_definitions,
)

DATA = PROJECT_ROOT / "data" / "code_hallucination"
MAX_CONTEXT_CHARS = 28000


def _blank_labels(sample: dict) -> str:
    """Return the answer with labeled spans blanked, so injected text isn't grounded."""
    answer = sample["answer"]
    for label in sorted(sample["labels"], key=lambda x: -x["start"]):
        answer = (
            answer[: label["start"]]
            + " " * (label["end"] - label["start"])
            + answer[label["end"] :]
        )
    return answer


def _load_repo_meta() -> dict[str, tuple[str | None, str | None]]:
    instances = json.loads((DATA / "swebench_instances.json").read_text())
    instances = instances if isinstance(instances, list) else list(instances.values())
    return {i["instance_id"]: (i.get("repo"), i.get("base_commit")) for i in instances}


def _reground(sample: dict, repo_meta: dict) -> str | None:
    """Return the grounded-context replacement for one sample, or None if unchanged."""
    answer = _blank_labels(sample)
    context = sample["context"]
    if not remaining_ungrounded(answer, context):
        return None
    instance_id = sample["metadata"]["instance_id"]
    repo, commit = repo_meta.get(instance_id, (None, None))
    if not repo or not commit:
        return None
    cache_path = DATA / "source_cache" / f"{instance_id}.json"
    sc = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    defs = resolve_definitions(
        answer,
        context,
        repo=repo,
        commit=commit,
        changed_files=sc.get("changed_files", []),
        modified_functions=sc.get("modified_functions", []),
    )
    block = render_definitions(defs)
    if not block:
        return None
    budget = max(MAX_CONTEXT_CHARS - len(block) - 4, 0)
    return block + "\n\n" + context[:budget]


def repair_split(path: Path, repo_meta: dict, workers: int) -> tuple[int, int]:
    """Reground every sample in one split file in place; return (changed, total)."""
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        new_contexts = list(pool.map(lambda s: _reground(s, repo_meta), samples))
    changed = 0
    for sample, new_context in zip(samples, new_contexts):
        if new_context is None:
            continue
        old_context = sample["context"]
        if old_context in sample["prompt"]:
            sample["prompt"] = sample["prompt"].replace(old_context, new_context, 1)
        sample["context"] = new_context
        changed += 1
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return changed, len(samples)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/v2/code_agent"))
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    repo_meta = _load_repo_meta()
    for split in args.splits:
        path = args.data / f"{split}.jsonl"
        if not path.exists():
            continue
        changed, total = repair_split(path, repo_meta, args.workers)
        print(f"{split}: regrounded {changed}/{total}")


if __name__ == "__main__":
    main()
