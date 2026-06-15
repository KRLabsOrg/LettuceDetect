#!/usr/bin/env python3
"""Build chat-format SFT data for the generative span detector from the HF datasets.

Target = the span-TEXT JSON the detector must emit (verbatim quotes, never offsets):
    {"hallucinated_spans": [{"text": ..., "category": ..., "explanation": ...}], ...}
Clean samples emit {"hallucinated_spans": []}. Writes {split}.jsonl with a `messages`
list usable directly by Unsloth/TRL chat-template SFT (train on the assistant turn).

Usage:
    python scripts/build_generative_sft.py \
        --dataset KRLabsOrg/lettucedetect-code-hallucination \
        --dataset KRLabsOrg/lettucedetect-prose-hallucination \
        --out data/generative_sft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SPLIT_ALIASES = {"dev": "validation"}
SYSTEM = (
    "You verify a generated answer against its context and list the hallucinated spans. "
    "Quote each unsupported span verbatim from the answer and label it. "
    'Reply with JSON: {"hallucinated_spans": [{"text": "...", "category": "...", '
    '"explanation": "..."}]}. If nothing is unsupported, reply {"hallucinated_spans": []}.'
)


def to_messages(row: dict) -> dict | None:
    """One HF row -> {'messages': [...], 'split': ...} or None if degenerate."""
    answer = row["answer"]
    spans = []
    for lab in row.get("labels") or []:
        text = answer[lab["start"] : lab["end"]]
        if not text.strip():
            continue
        span = {"text": text, "category": lab.get("category")}
        if lab.get("explanation"):
            span["explanation"] = lab["explanation"]
        spans.append(span)
    user = f"Context:\n{row['context'] or row['prompt']}\n\nAnswer to verify:\n{answer}"
    assistant = json.dumps({"hallucinated_spans": spans}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "split": SPLIT_ALIASES.get(row.get("split", "train"), row.get("split", "train")),
    }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Build span-text SFT data from HF datasets.")
    ap.add_argument("--dataset", action="append", default=[], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    args.out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    writers: dict[str, object] = {}
    for name in args.dataset:
        dd = load_dataset(name)
        for split in dd:
            out_split = SPLIT_ALIASES.get(split, split)
            f = writers.get(out_split) or (args.out / f"{out_split}.jsonl").open("w")
            writers[out_split] = f
            for i, row in enumerate(dd[split]):
                if args.limit and i >= args.limit:
                    break
                rec = to_messages(row)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[out_split] = counts.get(out_split, 0) + 1
    for f in writers.values():
        f.close()
    print(counts)


if __name__ == "__main__":
    main()
