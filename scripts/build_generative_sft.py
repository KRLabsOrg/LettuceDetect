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
import hashlib
import json
from pathlib import Path

SPLIT_ALIASES = {"dev": "validation"}
# Two prompt variants form an explanation switch: rows whose spans carry an
# explanation get SYSTEM_EXPL (explanation is a requested field), the rest get
# SYSTEM_BASE. The model learns to emit explanations only when asked, and the
# field can be toggled at inference by picking the prompt.
SYSTEM_BASE = (
    "You verify a generated answer against its context and list the hallucinated spans. "
    "Quote each unsupported span verbatim from the answer and label it with a category and "
    "subcategory. "
    'Reply with JSON: {"hallucinated_spans": [{"text": "...", "category": "...", '
    '"subcategory": "..."}]}. If nothing is unsupported, reply {"hallucinated_spans": []}.'
)
SYSTEM_EXPL = (
    "You verify a generated answer against its context and list the hallucinated spans. "
    "Quote each unsupported span verbatim from the answer, label it with a category and "
    "subcategory, and explain why it is unsupported. "
    'Reply with JSON: {"hallucinated_spans": [{"text": "...", "category": "...", '
    '"subcategory": "...", "explanation": "..."}]}. '
    'If nothing is unsupported, reply {"hallucinated_spans": []}.'
)


def _expl_key(answer: str, start: int, end: int) -> str:
    """Stable key joining an explanation to a span (md5 of answer + offsets)."""
    digest = hashlib.md5(answer.encode(), usedforsecurity=False).hexdigest()[:16]
    return f"{digest}|{start}|{end}"


def to_messages(row: dict, expl_map: dict | None = None) -> dict | None:
    """One HF row -> {'messages': [...], 'split': ...} or None if degenerate."""
    answer = row["answer"]
    spans = []
    for lab in row.get("labels") or []:
        start, end = lab["start"], lab["end"]
        text = answer[start:end]
        if not text.strip():
            continue
        span = {"text": text, "category": lab.get("category")}
        if lab.get("subcategory"):
            span["subcategory"] = lab["subcategory"]
        explanation = lab.get("explanation")
        if not explanation and expl_map:
            explanation = expl_map.get(_expl_key(answer, start, end))
        if explanation:
            span["explanation"] = explanation
        spans.append(span)
    system = SYSTEM_EXPL if any("explanation" in s for s in spans) else SYSTEM_BASE
    user = f"Context:\n{row['context'] or row['prompt']}\n\nAnswer to verify:\n{answer}"
    assistant = json.dumps({"hallucinated_spans": spans}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": system},
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
    ap.add_argument("--explanations-map", type=Path, help="JSON map md5(answer)|start|end -> explanation.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    expl_map = json.loads(args.explanations_map.read_text()) if args.explanations_map else None
    matched = 0

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
                rec = to_messages(row, expl_map)
                matched += sum(
                    1 for m in rec["messages"] if m["role"] == "assistant" and '"explanation"' in m["content"]
                )
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[out_split] = counts.get(out_split, 0) + 1
    for f in writers.values():
        f.close()
    print(counts)
    if expl_map:
        print(f"rows carrying >=1 explanation: {matched}")


if __name__ == "__main__":
    main()
