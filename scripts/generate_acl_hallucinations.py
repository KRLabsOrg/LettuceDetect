#!/usr/bin/env python3
r"""Generate markdown hallucination samples from ACL paper retrieval data.

Source: ``KRLabsOrg/acl-verbatim-spans`` (canonical). Each question already has
its top-k retrieved chunks (markdown passages from ACL papers), so we skip
question generation and retrieval. Per question we:

1. assemble the top-5 retrieved chunks (distractors included) as the context;
2. generate a correct answer grounded in them (markdown modality);
3. inject a localized, paper-specific hallucination (NUMERICAL / ENTITY /
   RELATIONAL / METHODOLOGICAL / CITATIONAL) detectable against the excerpts,
   using menu-mode injection (the model picks the 1-3 types that fit).

Output is the v2 HallucinationSample schema, ``dataset="lettucedetect-acl"``,
``context_modality="markdown"``.

Usage::

    API_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \\
        MODEL=Qwen/Qwen3.6-35B-A3B \\
        python scripts/generate_acl_hallucinations.py --limit 20 --out data/v2/acl
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from collections.abc import Awaitable, Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI  # noqa: E402

from lettucedetect.generation.answers import generate_grounded_answer_async  # noqa: E402
from lettucedetect.generation.doc_source import hash_split  # noqa: E402
from lettucedetect.generation.injection import InjectionResult, inject_menu_async  # noqa: E402
from lettucedetect.generation.runner import Outcome, run_batched_sync  # noqa: E402

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.6-35B-A3B")
HALLUCINATION_RATIO = float(os.environ.get("HALLUCINATION_RATIO", "0.4"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "32000"))
TOP_K = 5
CONTEXT_MODALITY = "markdown"

# Academic-paper injection prompt (menu mode). Adapted from the acl-verbatim paper
# injector: here the thing edited is the generated ANSWER, and the provided paper
# excerpts are the source of truth.
PAPER_INJECTION_PROMPT = """\
You are a factual hallucination injector for an academic-paper QA dataset.

You are given a CORRECT answer to a question, grounded in the provided paper
excerpts, plus those excerpts (the source of truth). Return ONLY a small set of
localized replacement edits that turn the answer into a subtly hallucinated one.
The pipeline applies your edits; outside them the answer must stay identical.

CRITICAL grounding rule:
- Every error MUST be detectable by comparing the answer against the excerpts.
- Use real entities, numbers, methods, and datasets that appear in the excerpts,
  placed in the wrong role/value. Do NOT invent names the excerpts never mention.
- The error must contradict something stated in the excerpts (a number, a
  comparison, a method, an attribution), not require outside knowledge.

Hallucination types (pick the 1-3 that fit; one per edit):
- NUMERICAL: change a reported number (score, dataset size, year, percentage) to
  a value that contradicts the excerpts.
- ENTITY: swap a named entity (model, dataset, benchmark, method, author) for a
  different one the excerpts do not use in this role.
- RELATIONAL: flip a comparison/relationship (outperforms<->underperforms,
  higher<->lower, increases<->decreases) so it contradicts the excerpts.
- METHODOLOGICAL: change a procedural detail (fine-tuned<->pre-trained,
  supervised<->unsupervised, encoder<->decoder) against what the excerpts describe.
- CITATIONAL: misattribute a claim or change a cited year/author the excerpts make
  clear.

Rules:
- 1-3 DISTINCT edits, in different sentences.
- Each replacement span is 8-120 characters, as small as possible.
- Total changed text < 30% of the answer.
- PLAUSIBLE and SUBTLE, not obviously broken.
- Do NOT add words like wrong, incorrect, error, hallucination, fix.
- "original" must be an exact substring of the answer, appearing exactly once.
- If a substring contains markdown tokens (*, _, `, |, $, [, ], (, ), #),
  the replacement must keep the same tokens in the same positions.

Respond in this exact JSON format (no markdown):
{
  "changes": [
    {
      "original": "exact substring from the answer",
      "hallucinated": "replacement text",
      "hallucination_type": "NUMERICAL | ENTITY | RELATIONAL | METHODOLOGICAL | CITATIONAL",
      "explanation": "what the excerpts actually state that this contradicts"
    }
  ]
}
If you cannot find a good edit, return {"changes": []}.
"""


def group_questions(rows: list[dict]) -> list[dict]:
    """Group rows into per-question items with their top-k retrieved chunks.

    The split is assigned per item by hashing the paper id, so the (tiny) source
    test config is ignored and train/dev/test are paper-separated and sensibly
    sized.
    """
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_q[r["question"]].append(r)

    items = []
    for question, qrows in by_q.items():
        # rank>=1 covers both retrieved (train/val) and gold-ranked (test) chunks
        ranked = sorted(
            (r for r in qrows if r.get("chunk") and r.get("retrieval_rank", 0) >= 1),
            key=lambda r: r["retrieval_rank"],
        )[:TOP_K]
        if not ranked:
            continue
        # need at least one chunk that actually answers the question
        if not any(r.get("answerable") for r in qrows):
            continue
        paper_id = qrows[0].get("gold_paper") or ranked[0].get("paper_id", "")
        items.append(
            {
                "question": question,
                "paper_id": paper_id,
                "chunks": [r["chunk"] for r in ranked],
                "split": hash_split(paper_id, dev_pct=8, test_pct=8),
            }
        )
    return items


def build_context(chunks: list[str]) -> str:
    """Join the retrieved chunks into a single labelled markdown context."""
    parts = [f"Excerpt {i + 1}:\n{c}" for i, c in enumerate(chunks)]
    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


def build_prompt(context: str, question: str) -> str:
    """Sample context: the excerpts plus the user question."""
    return f"{context}\n\nUser request: {question}"


def _item_key(item: dict) -> str:
    return f"{item['split']}::{item['paper_id']}::{item['question'][:100]}"


def _record_key(record: dict) -> str:
    m = record["metadata"]
    return f"{record['split']}::{m['paper_id']}::{m['question'][:100]}"


def make_sample(item: dict, context: str, answer: str, hall: InjectionResult | None) -> dict:
    """Build one v2 HallucinationSample dict (clean or hallucinated)."""
    is_hall = hall is not None and hall.ok
    if is_hall:
        final_answer, labels = hall.hallucinated_answer, hall.labels
        cat = labels[0]["category"] if labels else None
        sub = labels[0].get("subcategory") if labels else None
    else:
        final_answer, labels, cat, sub = answer, [], None, None
    return {
        "prompt": build_prompt(context, item["question"]),
        "answer": final_answer,
        "labels": labels,
        "split": item["split"],
        "task_type": "qa",
        "dataset": "lettucedetect-acl",
        "language": "en",
        "context_modality": CONTEXT_MODALITY,
        "category": cat,
        "subcategory": sub,
        "metadata": {
            "paper_id": item["paper_id"],
            "question": item["question"],
            "is_hallucinated": is_hall,
            "injector_model": MODEL if is_hall else None,
        },
    }


def _make_process(
    aclient: AsyncOpenAI, hall_keys: set[str]
) -> Callable[[dict], Awaitable[Outcome]]:
    async def process(item: dict) -> Outcome:
        key = _item_key(item)
        context = build_context(item["chunks"])
        answer = await generate_grounded_answer_async(
            aclient,
            MODEL,
            question=item["question"],
            evidence=context,
            temperature=0.7,
            completion_kwargs={"max_tokens": 400},
        )
        if not answer:
            return Outcome(key=key, ok=False, reason="answer_generation_failed")

        hall = None
        if key in hall_keys:
            hall = await inject_menu_async(
                aclient,
                MODEL,
                context=context,
                clean_answer=answer,
                system_prompt=PAPER_INJECTION_PROMPT,
                source="paper",
                temperature=0.8,
                completion_kwargs={"max_tokens": 600},
            )
        sample = make_sample(item, context, answer, hall)
        extra = {"inject_failed": hall.reason} if (hall and not hall.ok) else {}
        return Outcome(key=key, ok=True, record=sample, extra=extra)

    return process


def main() -> None:
    """Generate ACL markdown hallucination samples per split via the batched runner."""
    import random

    ap = argparse.ArgumentParser(description="Generate ACL paper hallucination samples.")
    ap.add_argument("--limit", type=int, default=20, help="Max questions total (0=all).")
    ap.add_argument("--out", type=str, default="data/v2/acl")
    ap.add_argument("--ratio", type=float, default=HALLUCINATION_RATIO)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"LLM: {MODEL} @ {API_BASE_URL}  (batch_size={args.batch_size})")
    aclient = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    ds = load_dataset("KRLabsOrg/acl-verbatim-spans", "canonical")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)  # noqa: S311 - seeded reproducibility, not crypto

    # Pool all source rows; split is assigned per question by paper hash.
    all_rows = [r for split in ds for r in ds[split]]
    items = group_questions(all_rows)
    if args.limit:
        items = items[: args.limit]

    for split in ("train", "dev", "test"):
        split_items = [it for it in items if it["split"] == split]
        if not split_items:
            continue
        n_targets = int(len(split_items) * args.ratio)
        hall_keys = {
            _item_key(it) for it in rng.sample(split_items, min(n_targets, len(split_items)))
        }
        process = _make_process(aclient, hall_keys)

        out_path = out_dir / f"{split}.jsonl"
        stats = run_batched_sync(
            split_items,
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
