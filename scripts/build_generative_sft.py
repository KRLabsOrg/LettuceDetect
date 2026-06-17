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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import SYSTEM_BASE, SYSTEM_EXPL

SPLIT_ALIASES = {"dev": "validation"}
# Prompt is chosen by source (see to_messages): code-agent uses SYSTEM_EXPL
# (explanation requested), other sources use SYSTEM_BASE. Both come from the
# shared taxonomy module so the label space + task definition never drift.


def _expl_key(answer: str, start: int, end: int) -> str:
    """Stable key joining an explanation to a span (md5 of answer + offsets)."""
    digest = hashlib.md5(answer.encode(), usedforsecurity=False).hexdigest()[:16]
    return f"{digest}|{start}|{end}"


def to_messages(
    row: dict, expl_map: dict | None = None, explain_sources: set | None = None
) -> dict | None:
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
    # Prompt is chosen by SOURCE, not by span-presence: a source whose answers
    # carry explanations uses SYSTEM_EXPL for BOTH clean and hallucinated rows,
    # so the model still has to discriminate (clean rows -> []). Keying on
    # span-presence would make the prompt a label leak (EXPL <=> hallucinated).
    use_expl = bool(explain_sources) and row.get("dataset") in explain_sources
    system = SYSTEM_EXPL if use_expl else SYSTEM_BASE
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
    ap.add_argument(
        "--explain-source",
        action="append",
        default=["lettucedetect-code-agent"],
        help="Dataset name(s) whose rows (clean AND hallucinated) use the explanation prompt.",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    expl_map = json.loads(args.explanations_map.read_text()) if args.explanations_map else None
    explain_sources = set(args.explain_source)
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
                rec = to_messages(row, expl_map, explain_sources)
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
