"""Classify an untyped hallucination span into the unified taxonomy with an LLM.

Some sources ship spans without a native label — most notably PsiloQA, whose
hallucinations are natural (model-produced, not injected) and annotated only as
binary char-offset spans. To fold these into the unified taxonomy we cannot use
the mechanical :func:`lettucedetect.datasets.taxonomy.map_label`, because there
is no native label to map. Instead we ask an LLM to read the context and the
answer with the span marked, and assign one ``(category, subcategory)``.

This is a classifier, not a generator: it never edits text and never invents
spans. It only labels a span that an annotator already marked as unsupported,
so the category space excludes ``supported``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from lettucedetect.datasets.taxonomy import (
    CATEGORY_DEFINITIONS,
    SUBCATEGORIES,
    SUBCATEGORY_DEFINITIONS,
)
from lettucedetect.generation._completion import complete, complete_async

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

# The span is known to be a hallucination, so "supported" is not a valid target.
INJECTABLE_CATEGORIES: tuple[str, ...] = tuple(CATEGORY_DEFINITIONS)

# Markers placed around the span inside the answer so the model sees exactly
# which substring it must classify, even when the same text repeats elsewhere.
_OPEN, _CLOSE = "«", "»"


def _taxonomy_block() -> str:
    """Render the category + subcategory definitions for the system prompt."""
    lines: list[str] = ["Categories (pick exactly one):"]
    for cat in INJECTABLE_CATEGORIES:
        lines.append(f"- {cat}: {CATEGORY_DEFINITIONS[cat]}")
        subs = SUBCATEGORIES.get(cat, [])
        if subs:
            rendered = "; ".join(f"{s} ({SUBCATEGORY_DEFINITIONS.get(s, s)})" for s in subs)
            lines.append(f"    subcategories: {rendered}")
    return "\n".join(lines)


_SYSTEM_PROMPT = (
    "You classify a single hallucinated span in an answer against its grounding "
    "context, using a fixed taxonomy. The span is already known to be "
    "unsupported; your job is only to say WHAT KIND of error it is.\n\n"
    + _taxonomy_block()
    + "\n\nReturn ONLY a JSON object: "
    '{"category": "<one of the categories>", "subcategory": "<one listed '
    'subcategory, or null>"}. Choose the subcategory that best fits, or null if '
    "none clearly applies. No prose, no code fence."
)


def _mark_span(answer: str, span_text: str, start: int | None, end: int | None) -> str:
    """Return the answer with the target span wrapped in markers.

    Uses offsets when given (exact, handles repeated substrings); otherwise falls
    back to the first textual occurrence.
    """
    if start is not None and end is not None and 0 <= start < end <= len(answer):
        return answer[:start] + _OPEN + answer[start:end] + _CLOSE + answer[end:]
    idx = answer.find(span_text)
    if idx == -1:
        return answer
    return answer[:idx] + _OPEN + span_text + _CLOSE + answer[idx + len(span_text) :]


def _user_msg(
    context: str,
    answer: str,
    span_text: str,
    start: int | None,
    end: int | None,
    context_chars: int,
) -> str:
    marked = _mark_span(answer, span_text, start, end)
    return (
        f"Context:\n{context[:context_chars]}\n\n"
        f"Answer (the span to classify is wrapped in {_OPEN}{_CLOSE}):\n{marked}\n\n"
        f"Span: {span_text}\n\n"
        "Classify the marked span."
    )


def _parse(content: str | None) -> tuple[str, str | None] | None:
    """Parse and validate the JSON verdict, or None to reject and retry."""
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    category = obj.get("category")
    if category not in INJECTABLE_CATEGORIES:
        return None
    # Accept either "subcategory" (string) or "subcategories" (a list the model
    # sometimes emits for long, multi-faceted spans); keep the first valid one.
    # An out-of-set or missing subcategory just drops to None rather than
    # rejecting the whole verdict.
    valid_subs = SUBCATEGORIES.get(category, [])
    sub = obj.get("subcategory", obj.get("subcategories"))
    if isinstance(sub, list):
        sub = next((s for s in sub if s in valid_subs), None)
    subcategory = sub if sub in valid_subs else None
    return category, subcategory


def classify_span(
    client: OpenAI,
    model: str,
    *,
    context: str,
    answer: str,
    span_text: str,
    start: int | None = None,
    end: int | None = None,
    temperature: float = 0.0,
    completion_kwargs: dict | None = None,
    context_chars: int = 8000,
    max_retries: int = 3,
) -> tuple[str, str | None] | None:
    """Classify ``span_text`` into ``(category, subcategory)`` (or None on failure)."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _user_msg(context, answer, span_text, start, end, context_chars),
        },
    ]
    return complete(
        client,
        model,
        messages,
        transform=_parse,
        temperature=temperature,
        completion_kwargs=completion_kwargs or {"max_tokens": 128},
        max_retries=max_retries,
    )


async def classify_span_async(
    aclient: AsyncOpenAI,
    model: str,
    *,
    context: str,
    answer: str,
    span_text: str,
    start: int | None = None,
    end: int | None = None,
    temperature: float = 0.0,
    completion_kwargs: dict | None = None,
    context_chars: int = 8000,
    max_retries: int = 3,
) -> tuple[str, str | None] | None:
    """Async twin of :func:`classify_span` for batched throughput."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _user_msg(context, answer, span_text, start, end, context_chars),
        },
    ]
    return await complete_async(
        aclient,
        model,
        messages,
        transform=_parse,
        temperature=temperature,
        completion_kwargs=completion_kwargs or {"max_tokens": 128},
        max_retries=max_retries,
    )
