#!/usr/bin/env python3
"""Extract (context, answer, span, category, subcategory) tuples for the taxonomy head.

The taxonomy head is a cascade Stage-B: it types spans that the binary detector
already found. So the training data is one row per GOLD hallucinated span, carrying
the grounding context, the answer, the span offsets+text, and its category +
subcategory (None -> "unspecified"). Detection is handled elsewhere; there are no
"supported" rows here.

    python scripts/build_taxonomy_data.py \
        --dataset KRLabsOrg/lettucedetect-code-hallucination \
        --dataset KRLabsOrg/lettucedetect-prose-hallucination \
        --out data/taxonomy
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

SPLIT_ALIASES = {"dev": "validation"}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", action="append", default=[], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    args.out.mkdir(parents=True, exist_ok=True)
    writers: dict[str, object] = {}
    counts: dict[str, int] = collections.Counter()
    cat_sub: dict[tuple, int] = collections.Counter()

    for name in args.dataset:
        dd = load_dataset(name)
        for split in dd:
            out_split = SPLIT_ALIASES.get(split, split)
            f = writers.get(out_split) or (args.out / f"{out_split}.jsonl").open("w")
            writers[out_split] = f
            for i, row in enumerate(dd[split]):
                if args.limit and i >= args.limit:
                    break
                answer = row["answer"]
                for lab in row.get("labels") or []:
                    start, end = lab["start"], lab["end"]
                    text = answer[start:end]
                    if not text.strip():
                        continue
                    category = lab.get("category")
                    subcategory = lab.get("subcategory") or "unspecified"
                    rec = {
                        "context": row["context"] or row["prompt"],
                        "answer": answer,
                        "start": start,
                        "end": end,
                        "span_text": text,
                        "category": category,
                        "subcategory": subcategory,
                        "dataset": row.get("dataset", name),
                        "language": row.get("language", "en"),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counts[out_split] += 1
                    cat_sub[(category, subcategory)] += 1

    for f in writers.values():
        f.close()
    print("spans per split:", dict(counts))
    print("\n(category, subcategory) counts:")
    for (c, s), n in sorted(cat_sub.items(), key=lambda kv: -kv[1]):
        print(f"  {c:>22} / {s:<14} {n}")


if __name__ == "__main__":
    main()
