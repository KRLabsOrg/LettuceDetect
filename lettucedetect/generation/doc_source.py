"""Generate hallucination samples from a corpus of markdown documents.

Shared flow for document-based sources (READMEs, Wikipedia): chunk each document
by heading, generate a typed question answerable from the chunk, generate a
grounded answer, inject a source-specific hallucination, and assemble v2 samples
— run batched and resumable. Sources differ only in the corpus, the question-type
subset, the injection prompt, and the native-type map; everything else is here.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from lettucedetect.generation.answers import generate_grounded_answer_async
from lettucedetect.generation.injection import InjectionResult, inject_menu_async
from lettucedetect.generation.questions import generate_question_async, sample_question_type
from lettucedetect.generation.runner import Outcome, run_batched_sync

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# Generic factual injection prompt for markdown documents (READMEs, wiki). Content
# is heterogeneous, so the model picks whichever factual error type fits the
# passage rather than being forced into a domain-specific schema.
GENERIC_MARKDOWN_INJECTION_PROMPT = """\
You are a hallucination injector for a document QA dataset.

You are given a CORRECT answer to a question, grounded in a document excerpt,
plus that excerpt (the source of truth). Return ONLY localized replacement edits
that make the answer subtly wrong. The pipeline applies your edits; outside them
the answer must stay identical.

CRITICAL grounding rule:
- Every error MUST be detectable by comparing the answer to the excerpt.
- Use real names, numbers, and facts from the excerpt placed in a wrong
  value/role. Do NOT invent things the excerpt never mentions (except FABRICATED,
  which references a plausible named element absent from the excerpt).

Hallucination types (pick the 1-3 that fit; one per edit):
- NUMERICAL: change a number, quantity, or version to one the excerpt contradicts.
- TEMPORAL: change a date, year, or ordering against the excerpt.
- ENTITY: swap a named entity (tool, project, person, place, format) for a
  different one the excerpt does not use in this role.
- RELATIONAL: flip a comparison or relationship (supports<->lacks,
  more<->less, before<->after) so it contradicts the excerpt.
- FABRICATED: reference a named element (feature, file, option, section) that does
  NOT appear in the excerpt.
- CLAIM: add a factual claim the excerpt neither states nor implies.

Rules:
- 1-3 DISTINCT edits, in different sentences.
- Each replacement span is 8-120 characters, as small as possible.
- Total changed text < 30% of the answer.
- PLAUSIBLE and SUBTLE, not obviously broken or absurd.
- Do NOT add words like wrong, incorrect, error, hallucination, fix.
- "original" must be an exact substring of the answer, appearing exactly once.
- If a substring contains markdown tokens (`, *, _, |, [, ], (, ), #), keep the
  same tokens in the same positions in the replacement.

Respond in this exact JSON format (no markdown):
{
  "changes": [
    {
      "original": "exact substring from the answer",
      "hallucinated": "replacement text",
      "hallucination_type": "NUMERICAL | TEMPORAL | ENTITY | RELATIONAL | FABRICATED | CLAIM",
      "explanation": "what the excerpt actually states that this contradicts"
    }
  ]
}
If you cannot find a good edit, return {"changes": []}.
"""


def hash_split(key: str, dev_pct: int = 5, test_pct: int = 5) -> str:
    """Deterministic document-level split, so all chunks of a doc share a split."""
    bucket = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + dev_pct:
        return "dev"
    return "train"


def chunk_by_heading(text: str, min_chars: int = 300, max_chars: int = 4000) -> list[str]:
    """Split markdown into heading-delimited sections within a size band.

    Each section starts at a ``#`` heading and runs to the next one. Sections
    shorter than ``min_chars`` are skipped; longer ones are truncated to
    ``max_chars`` so a chunk is a focused, bounded context.
    """
    lines = text.splitlines()
    sections: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.lstrip().startswith("#") and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)

    chunks = []
    for sec in sections:
        body = "\n".join(sec).strip()
        if len(body) >= min_chars:
            chunks.append(body[:max_chars])
    return chunks


@dataclass
class DocSourceConfig:
    """Per-source configuration for :func:`generate_doc_source`.

    Defaults to the generic factual markdown injection (suits READMEs and wiki);
    a source can override ``injection_prompt`` / ``source_key`` for a tailored
    schema.
    """

    dataset_name: str  # e.g. "lettucedetect-readme"
    source_key: str = "markdown"  # taxonomy map key for the injected types
    injection_prompt: str = GENERIC_MARKDOWN_INJECTION_PROMPT
    question_types: list[str] = field(default_factory=list)  # subset; empty = all
    modality: str = "markdown"
    task_type: str = "qa"
    max_chunks_per_doc: int = 3
    min_chunk_chars: int = 300
    max_chunk_chars: int = 4000


def _make_sample(
    item: dict, question: str, answer: str, hall: InjectionResult | None, cfg: DocSourceConfig
) -> dict:
    is_hall = hall is not None and hall.ok
    if is_hall:
        final_answer, labels = hall.hallucinated_answer, hall.labels
        cat = labels[0]["category"] if labels else None
        sub = labels[0].get("subcategory") if labels else None
    else:
        final_answer, labels, cat, sub = answer, [], None, None
    return {
        "prompt": f"{item['chunk']}\n\nUser request: {question}",
        "answer": final_answer,
        "labels": labels,
        "split": item["split"],
        "task_type": cfg.task_type,
        "dataset": cfg.dataset_name,
        "language": "en",
        "context_modality": cfg.modality,
        "category": cat,
        "subcategory": sub,
        "metadata": {
            "doc_id": item["doc_id"],
            "question_type": item["q_type"],
            "is_hallucinated": is_hall,
            "injector_model": item["model"] if is_hall else None,
        },
    }


def _item_key(item: dict) -> str:
    return f"{item['split']}::{item['doc_id']}::{item['chunk_index']}"


def _record_key(record: dict) -> str:
    m = record["metadata"]
    return f"{record['split']}::{m['doc_id']}::{m.get('chunk_index', 0)}"


def _build_items(docs: Iterable[dict], cfg: DocSourceConfig, rng: random.Random) -> list[dict]:
    """Chunk each doc and emit work items (one per kept chunk)."""
    items = []
    for doc in docs:
        chunks = chunk_by_heading(doc["text"], cfg.min_chunk_chars, cfg.max_chunk_chars)
        for ci, chunk in enumerate(chunks[: cfg.max_chunks_per_doc]):
            items.append(
                {
                    "doc_id": doc["id"],
                    "chunk_index": ci,
                    "chunk": chunk,
                    "split": doc["split"],
                    "q_type": sample_question_type(rng, cfg.question_types or None),
                }
            )
    return items


def _make_process(
    aclient: AsyncOpenAI, model: str, cfg: DocSourceConfig, hall_keys: set[str]
) -> Callable[[dict], Awaitable[Outcome]]:
    async def process(item: dict) -> Outcome:
        key = _item_key(item)
        item["model"] = model
        question = await generate_question_async(
            aclient, model, doc=item["chunk"], q_type=item["q_type"]
        )
        if not question:
            return Outcome(key=key, ok=False, reason="question_generation_failed")
        answer = await generate_grounded_answer_async(
            aclient, model, question=question, evidence=item["chunk"],
            completion_kwargs={"max_tokens": 400},
        )
        if not answer:
            return Outcome(key=key, ok=False, reason="answer_generation_failed")

        hall = None
        if key in hall_keys:
            hall = await inject_menu_async(
                aclient, model, context=item["chunk"], clean_answer=answer,
                system_prompt=cfg.injection_prompt, source=cfg.source_key,
                completion_kwargs={"max_tokens": 600},
            )
        sample = _make_sample(item, question, answer, hall, cfg)
        extra = {"inject_failed": hall.reason} if (hall and not hall.ok) else {}
        return Outcome(key=key, ok=True, record=sample, extra=extra)

    return process


def generate_doc_source(
    docs: Iterable[dict],
    cfg: DocSourceConfig,
    *,
    aclient: AsyncOpenAI,
    model: str,
    out_path: str | Path,
    failures_path: str | Path | None = None,
    ratio: float = 0.4,
    seed: int = 42,
    batch_size: int = 16,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Run the doc-source pipeline over ``docs`` (each ``{id, text, split}``)."""
    rng = random.Random(seed)  # noqa: S311 - seeded reproducibility, not crypto
    items = _build_items(docs, cfg, rng)
    n_targets = int(len(items) * ratio)
    hall_keys = {_item_key(it) for it in rng.sample(items, min(n_targets, len(items)))}
    process = _make_process(aclient, model, cfg, hall_keys)
    return run_batched_sync(
        items,
        process,
        out_path=out_path,
        failures_path=failures_path,
        key_of=_item_key,
        record_key=_record_key,
        batch_size=batch_size,
        on_progress=on_progress,
    )
