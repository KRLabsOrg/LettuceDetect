"""Steps shared by the LLM-backed detectors.

Everything except :func:`complete` is a pure function: no client, no cache and no thread.
"""

from __future__ import annotations

import logging
import numbers
import re
import threading
from typing import Any

from lettucedetect.datasets.taxonomy import CATEGORY_DEFINITIONS, SUBCATEGORY_DEFINITIONS
from lettucedetect.detectors.errors import UpstreamError
from lettucedetect.detectors.llm_client import LLMClient
from lettucedetect.detectors.types import DetectedSpan

# The only logger that may carry customer text, the raw reply included.
content_logger = logging.getLogger("lettucedetect.content")

Labels = dict[str, Any]
"""Label names mapped to their descriptions (``None`` or empty for a label without one)."""


def resolve_taxonomy(
    include_taxonomy: bool | list[str] | dict[str, Any],
) -> tuple[Labels | None, Labels | None]:
    """Turn the four ``include_taxonomy`` forms into category and subcategory labels.

    :param include_taxonomy: ``True`` for the unified taxonomy, a list of category names, a flat
        ``{category: description}`` dict, or ``{"categories": {...}, "subcategories": {...}}``.
    :returns: the categories and the subcategories, each ``None`` when there are none.
    """
    if isinstance(include_taxonomy, bool):
        if not include_taxonomy:
            return None, None
        return dict(CATEGORY_DEFINITIONS), dict(SUBCATEGORY_DEFINITIONS)
    if isinstance(include_taxonomy, dict):
        if (
            include_taxonomy
            and set(include_taxonomy) <= {"categories", "subcategories"}
            and all(isinstance(value, dict) for value in include_taxonomy.values())
        ):
            return (
                dict(include_taxonomy.get("categories") or {}) or None,
                dict(include_taxonomy.get("subcategories") or {}) or None,
            )
        return dict(include_taxonomy) or None, None
    return {name: CATEGORY_DEFINITIONS.get(name) for name in include_taxonomy} or None, None


def unparseable_response(what: str) -> UpstreamError:
    """Build the error for a reply that does not have the expected JSON shape.

    :param what: the shape that was expected, without any text of the reply.
    """
    return UpstreamError(
        f"The LLM reply is not {what}. Retry, or use a model that follows the response schema.",
        code="unparseable_response",
        service="llm",
    )


def span_score(item: dict[str, Any]) -> float | None:
    """Return the model's ``confidence`` of an item, or ``None`` if it is not a number in [0, 1]."""
    confidence = item.get("confidence")
    if (
        isinstance(confidence, numbers.Real)
        and not isinstance(confidence, bool)
        and 0.0 <= confidence <= 1.0
    ):
        return float(confidence)
    return None


def locate_items(
    items: list[Any], answer: str, min_score: float = 0.0
) -> tuple[list[dict[str, Any]], int, int]:
    """Locate the items of a reply in the answer.

    An item is a string, or an object with ``text`` and optionally ``confidence``, ``reasoning``,
    ``category``, ``subcategory`` and ``is_hallucination``. Each item takes the first verbatim
    match that overlaps no earlier item. An item the model rejected (``is_hallucination`` is
    false) or scored below ``min_score`` still takes its match, so a later repeated item keeps
    its place, and is then left out.

    :param items: the items of the reply, in answer order.
    :param answer: the answer the items were copied from.
    :param min_score: leave out items whose ``confidence`` is below this value.
    :returns: the located spans as dicts with ``start``, ``end``, ``text`` and the optional keys
        the item had; the number of items without a free match; the number of malformed items.
    """
    spans: list[dict[str, Any]] = []
    used: list[tuple[int, int]] = []
    not_found = 0
    malformed = 0
    for item in items:
        text = item.get("text") if isinstance(item, dict) else item
        if not isinstance(text, str) or not text:
            malformed += 1
            continue
        match = next(
            (
                match
                for match in re.finditer(re.escape(text), answer)
                if not any(match.start() < end and start < match.end() for start, end in used)
            ),
            None,
        )
        if match is None:
            not_found += 1
            continue
        used.append((match.start(), match.end()))
        span: dict[str, Any] = {"start": match.start(), "end": match.end(), "text": text}
        if isinstance(item, dict):
            if item.get("is_hallucination") is False:
                continue
            score = span_score(item)
            if score is not None and score < min_score:
                continue
            for key in ("confidence", "reasoning", "category", "subcategory"):
                if key in item:
                    span[key] = item[key]
        spans.append(span)
    return spans, not_found, malformed


def to_detected_spans(spans: list[dict[str, Any]]) -> tuple[list[DetectedSpan], bool]:
    """Turn located spans into ``DetectedSpan`` objects.

    :param spans: spans as :func:`locate_items` returns them.
    :returns: the spans, and whether any of them had no usable ``confidence`` and scores 1.0.
    """
    detected = []
    synthetic = False
    for span in spans:
        score = span_score(span)
        synthetic = synthetic or score is None
        labels = {
            key: span[key] if isinstance(span.get(key), str) else None
            for key in ("category", "subcategory", "reasoning")
        }
        detected.append(
            DetectedSpan(
                start=span["start"],
                end=span["end"],
                text=span["text"],
                score=1.0 if score is None else score,
                **labels,
            )
        )
    return detected, synthetic


def complete(
    client: LLMClient,
    semaphore: threading.BoundedSemaphore,
    *,
    system: str,
    user: str,
    model: str,
    temperature: float,
    schema: dict[str, Any],
) -> str:
    """Call the client while holding the detector's semaphore, and return the raw reply.

    The semaphore is held only for the client call and is released when the call returns or
    raises. Errors of the client propagate unwrapped.
    """
    with semaphore:
        reply = client.complete(
            system=system, user=user, model=model, temperature=temperature, schema=schema
        )
    content_logger.debug("LLM reply: %s", reply)
    return reply
