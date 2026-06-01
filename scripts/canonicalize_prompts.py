#!/usr/bin/env python3
r"""Separate baked prompts into context+question and move the question to the front.

Our generation adapters used to bake ``{context}\n\nUser request: {question}``
into ``prompt`` — request **last**, where ``truncation="only_first"`` clips it on
long prompts. This transform, run in place over the ``data/v2`` sources we
generated:

1. recovers ``context`` and ``question`` from the baked prompt;
2. stores them as separate fields;
3. rebuilds ``prompt`` with the question at the **front**, so it is never
   truncated away.

It is idempotent (re-running rebuilds from the stored ``context``/``question``)
and never deletes data — each file is rewritten in place. RAGTruth is skipped:
its native prompt already puts the instruction first and has no ``User request:``
delimiter to split on.

Usage::

    python scripts/canonicalize_prompts.py --root data/v2 \
        --source psiloqa --source code_hallucination --source squeez \
        --source acl --source readme --source wikipedia
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DELIM = "User request: "
DEFAULT_SOURCES = ("psiloqa", "code_hallucination", "squeez", "acl", "readme", "wikipedia")


def split_prompt(prompt: str) -> tuple[str, str | None]:
    """Recover ``(context, question)`` from a baked ``User request:`` prompt."""
    idx = prompt.rfind(DELIM)
    if idx == -1:
        return prompt, None
    context = prompt[:idx].rstrip()
    question = prompt[idx + len(DELIM) :].strip()
    return context, (question or None)


def build_prompt(context: str, question: str | None) -> str:
    """Build the model input with the question at the front (truncation-safe)."""
    if question:
        return f"User request: {question}\n\n{context}"
    return context


def transform_record(rec: dict) -> dict:
    """Add/refresh ``context``/``question`` and rebuild the question-first ``prompt``."""
    if rec.get("context"):
        context, question = rec["context"], rec.get("question")
    else:
        context, question = split_prompt(rec["prompt"])
    rec["context"] = context
    rec["question"] = question
    rec["prompt"] = build_prompt(context, question)
    return rec


def transform_file(path: Path) -> tuple[int, int]:
    """Rewrite one JSONL file in place. Returns (n_records, n_with_question)."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    n_q = 0
    for rec in records:
        transform_record(rec)
        n_q += rec["question"] is not None
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return len(records), n_q


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Canonicalize baked prompts in place.")
    ap.add_argument("--root", default="data/v2", help="Root dir holding source subdirs.")
    ap.add_argument(
        "--source",
        action="append",
        help=f"Source subdir to transform (repeatable). Default: {DEFAULT_SOURCES}",
    )
    args = ap.parse_args()

    root = Path(args.root)
    sources = args.source or list(DEFAULT_SOURCES)
    for src in sources:
        src_dir = root / src
        files = [p for p in sorted(src_dir.glob("*.jsonl")) if not p.name.endswith(".failures.jsonl")]
        if not files:
            print(f"{src}: (no files)")
            continue
        for path in files:
            n, n_q = transform_file(path)
            print(f"  {path}: {n} records ({n_q} with question)")


if __name__ == "__main__":
    main()
