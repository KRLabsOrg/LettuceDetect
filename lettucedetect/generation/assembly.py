"""Assemble generated samples into upload-ready splits.

Small, reusable helpers applied when turning a source's raw generation output
into the final dataset: balance the hallucinated/clean ratio and load/group
per-split records. Used by upload flows so no per-source munging scripts are
needed.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def format_prompt(context: str, question: str | None) -> str:
    """Build a sample's model-input prompt with the question at the front.

    Placing the request before the context keeps it from being truncated away
    when a long context is clipped (``truncation="only_first"`` cuts the tail).
    Sources build their own ``context`` string; this is the shared final step.
    """
    if question:
        return f"User request: {question}\n\n{context}"
    return context


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of records."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def balance_hallucination_ratio(
    records: list[dict],
    ratio: float,
    rng: random.Random,
) -> list[dict]:
    """Trim clean samples so hallucinated records make up ``ratio`` of the result.

    Keeps every hallucinated record (a label-bearing sample) and randomly drops
    clean ones until ``hallucinated / total == ratio``. Returns a new shuffled
    list; if there are already too few clean samples, returns all records.
    """
    hall = [r for r in records if r.get("labels")]
    clean = [r for r in records if not r.get("labels")]
    if not hall or ratio <= 0:
        return list(records)

    keep_clean = round(len(hall) * (1 - ratio) / ratio)
    if keep_clean < len(clean):
        clean = clean[:]
        rng.shuffle(clean)
        clean = clean[:keep_clean]

    result = hall + clean
    rng.shuffle(result)
    return result
