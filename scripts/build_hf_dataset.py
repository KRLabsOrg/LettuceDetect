#!/usr/bin/env python3
r"""Assemble one or more ``data/v2`` sources into a Hugging Face dataset and push it.

Every source directory holds ``{split}.jsonl`` files in the unified
``HallucinationSample`` schema (see :mod:`lettucedetect.preprocess.apply_taxonomy`
and the generation adapters). This merges any set of sources into a single
``DatasetDict`` and uploads it, differentiating sources by the ``dataset`` field.

It is the one place that does the HF-specific normalization:

- ``dev`` split is renamed to ``validation``;
- ``metadata`` (a dict whose keys vary per source) is serialized to a JSON
  **string**, so the Hub schema stays consistent across sources;
- ``labels`` are reduced to the fixed key set, defensively.

Usage::

    # combine the prose sources and publish
    python scripts/build_hf_dataset.py \
        --source data/v2/psiloqa --source data/v2/ragtruth \
        --repo-id KRLabsOrg/lettucedetect-prose-hallucination

    # inspect what would be uploaded without pushing
    python scripts/build_hf_dataset.py --source data/v2/code_hallucination --dry-run
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Top-level fields kept on every uploaded record, in order.
RECORD_FIELDS = (
    "prompt",
    "answer",
    "labels",
    "split",
    "task_type",
    "dataset",
    "language",
    "context_modality",
    "category",
    "subcategory",
    "metadata",
)
LABEL_FIELDS = ("start", "end", "label", "category", "subcategory")
SPLIT_RENAME = {"dev": "validation"}


def _norm_label(label: dict) -> dict:
    return {k: label.get(k) for k in LABEL_FIELDS}


def _norm_record(rec: dict) -> dict:
    """Normalize one record for the Hub (split name, metadata string, label keys)."""
    out = {k: rec.get(k) for k in RECORD_FIELDS}
    out["split"] = SPLIT_RENAME.get(out["split"], out["split"])
    out["labels"] = [_norm_label(label) for label in (rec.get("labels") or [])]
    meta = rec.get("metadata")
    out["metadata"] = meta if isinstance(meta, str) else json.dumps(meta or {})
    return out


def load_source(source_dir: Path) -> list[dict]:
    """Read all split JSONL files in a source dir (skipping failure logs)."""
    records: list[dict] = []
    for path in sorted(source_dir.glob("*.jsonl")):
        if path.name.endswith(".failures.jsonl"):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(_norm_record(json.loads(line)))
    return records


def summarize(records: list[dict]) -> None:
    """Print per-split, per-source, per-language counts."""
    by_split: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)
    for split in ("train", "validation", "test"):
        rows = by_split.get(split, [])
        if not rows:
            continue
        n_hall = sum(1 for r in rows if r["labels"])
        srcs = collections.Counter(r["dataset"] for r in rows)
        langs = collections.Counter(r["language"] for r in rows)
        print(f"  {split}: {len(rows)} samples ({n_hall} hallucinated)")
        print(f"      sources: {dict(srcs)}")
        print(f"      languages: {dict(langs.most_common())}")
    print(f"  TOTAL: {len(records)} samples")


def build_and_push(
    source_dirs: list[Path],
    repo_id: str | None,
    *,
    private: bool,
    dry_run: bool,
) -> None:
    """Merge the sources into a DatasetDict and (unless dry-run) push it."""
    from datasets import Dataset, DatasetDict

    records: list[dict] = []
    for d in source_dirs:
        src_records = load_source(d)
        print(f"loaded {len(src_records)} from {d}")
        records.extend(src_records)

    summarize(records)

    by_split: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)
    dd = DatasetDict(
        {
            split: Dataset.from_list(by_split[split])
            for split in ("train", "validation", "test")
            if by_split.get(split)
        }
    )

    if dry_run:
        print("\n[dry-run] not pushing. Schema:")
        print(next(iter(dd.values())).features)
        return
    if not repo_id:
        raise SystemExit("--repo-id is required unless --dry-run")

    print(f"\nPushing to {repo_id} (private={private}) ...")
    dd.push_to_hub(repo_id, private=private)
    print("Done.")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Assemble v2 sources into an HF dataset.")
    ap.add_argument(
        "--source",
        action="append",
        required=True,
        help="A data/v2 source directory (repeatable).",
    )
    ap.add_argument("--repo-id", help="Target HF dataset repo id.")
    ap.add_argument("--private", action="store_true", help="Create a private repo.")
    ap.add_argument("--dry-run", action="store_true", help="Summarize without pushing.")
    args = ap.parse_args()

    build_and_push(
        [Path(s) for s in args.source],
        args.repo_id,
        private=args.private,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
