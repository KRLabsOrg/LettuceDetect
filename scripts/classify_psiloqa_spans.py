#!/usr/bin/env python3
r"""Fold PsiloQA into the unified taxonomy by classifying its spans with an LLM.

Source: ``s-nlp/PsiloQA``. Unlike our other sources, PsiloQA hallucinations are
**natural** — produced by real LLMs answering Wikipedia-grounded questions, not
injected — and annotated only as binary character-offset spans, with no error
type. We therefore do not generate or inject anything here; we keep the original
answer and spans verbatim and only assign each span a ``(category, subcategory)``
from the unified taxonomy using :func:`lettucedetect.generation.classify`.

All 14 languages and the original train/validation/test splits are preserved.
Output is the v2 HallucinationSample schema, ``dataset="psiloqa"``,
``context_modality="prose"``.

Usage::

    API_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \\
        MODEL=Qwen/Qwen3.6-35B-A3B \\
        python scripts/classify_psiloqa_spans.py --limit 50 --out data/v2/psiloqa
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI  # noqa: E402

from lettucedetect.generation.classify import classify_span_async  # noqa: E402
from lettucedetect.generation.runner import Outcome, run_batched_sync  # noqa: E402

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.6-35B-A3B")
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "32000"))
CONTEXT_MODALITY = "prose"

# PsiloQA's native split names already match our schema.
SPLITS = ("train", "validation", "test")


def build_prompt(passage: str, question: str) -> str:
    """Sample context: the Wikipedia passage plus the user question."""
    return f"{passage[:MAX_CONTEXT_CHARS]}\n\nUser request: {question}"


def to_item(row: dict, split: str) -> dict | None:
    """Normalize a PsiloQA row into a work item, or None if unusable.

    ``labels`` is a list of ``[start, end]`` char offsets into ``llm_answer``;
    we keep only the in-range ones. Rows with no labels are clean samples.
    """
    answer = row.get("llm_answer") or ""
    passage = row.get("wiki_passage") or ""
    question = row.get("question") or ""
    if not answer or not passage or not question:
        return None
    spans = [
        [int(s), int(e)]
        for s, e in (row.get("labels") or [])
        if 0 <= int(s) < int(e) <= len(answer)
    ]
    return {
        "id": row["id"],
        "lang": row.get("lang") or "en",
        "split": split,
        "passage": passage,
        "question": question,
        "answer": answer,
        "spans": spans,
    }


def _item_key(item: dict) -> str:
    return f"{item['split']}::{item['id']}"


def _record_key(record: dict) -> str:
    return f"{record['split']}::{record['metadata']['id']}"


def make_sample(item: dict, labels: list[dict]) -> dict:
    """Build one v2 HallucinationSample dict from classified labels."""
    is_hall = bool(labels)
    cat = labels[0]["category"] if labels else None
    sub = labels[0].get("subcategory") if labels else None
    return {
        "prompt": build_prompt(item["passage"], item["question"]),
        "answer": item["answer"],
        "labels": labels,
        "split": item["split"],
        "task_type": "qa",
        "dataset": "psiloqa",
        "language": item["lang"],
        "context_modality": CONTEXT_MODALITY,
        "category": cat,
        "subcategory": sub,
        "metadata": {
            "id": item["id"],
            "is_hallucinated": is_hall,
            "classifier_model": MODEL if is_hall else None,
        },
    }


def _make_process(aclient: AsyncOpenAI) -> Callable[[dict], Awaitable[Outcome]]:
    async def process(item: dict) -> Outcome:
        key = _item_key(item)
        answer, passage = item["answer"], item["passage"]

        # Clean sample: no spans to classify.
        if not item["spans"]:
            return Outcome(key=key, ok=True, record=make_sample(item, []))

        # Classify every annotated span; one failure retries the whole item on
        # rerun rather than persisting a guessed label.
        verdicts = await asyncio.gather(
            *(
                classify_span_async(
                    aclient,
                    MODEL,
                    context=passage,
                    answer=answer,
                    span_text=answer[s:e],
                    start=s,
                    end=e,
                )
                for s, e in item["spans"]
            )
        )
        if any(v is None for v in verdicts):
            return Outcome(key=key, ok=False, reason="classify_failed")

        labels = [
            {"start": s, "end": e, "label": cat, "category": cat, "subcategory": sub}
            for (s, e), (cat, sub) in zip(item["spans"], verdicts)
        ]
        return Outcome(key=key, ok=True, record=make_sample(item, labels))

    return process


def main() -> None:
    """Classify PsiloQA spans into the unified taxonomy via the batched runner."""
    ap = argparse.ArgumentParser(description="Fold PsiloQA into the unified taxonomy.")
    ap.add_argument("--limit", type=int, default=0, help="Max rows per split (0=all).")
    ap.add_argument("--out", type=str, default="data/v2/psiloqa")
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"LLM: {MODEL} @ {API_BASE_URL}  (batch_size={args.batch_size})")
    aclient = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    process = _make_process(aclient)

    for split in SPLITS:
        ds = load_dataset("s-nlp/PsiloQA", split=split)
        items = [it for it in (to_item(r, split) for r in ds) if it is not None]
        if args.limit:
            items = items[: args.limit]
        if not items:
            continue

        out_path = out_dir / f"{split}.jsonl"
        stats = run_batched_sync(
            items,
            process,
            out_path=out_path,
            failures_path=out_dir / f"{split}.failures.jsonl",
            key_of=_item_key,
            record_key=_record_key,
            batch_size=args.batch_size,
            on_progress=lambda s: print(f"  {split}: {s}"),
        )
        print(f"{split}: {stats} -> {out_path}")


if __name__ == "__main__":
    main()
