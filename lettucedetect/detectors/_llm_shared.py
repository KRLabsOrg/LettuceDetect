"""Steps shared by the LLM-backed detectors.

Only :func:`fetch_parsed` touches a client or a cache, and :func:`verify_spans` calls it. The
other functions are pure.
"""

from __future__ import annotations

import json
import logging
import numbers
import re
import threading
from collections.abc import Callable
from string import Template
from typing import Any, TypeVar

from lettucedetect.datasets.taxonomy import CATEGORY_DEFINITIONS, SUBCATEGORY_DEFINITIONS
from lettucedetect.detectors.errors import UpstreamError
from lettucedetect.detectors.llm_client import LLMClient, build_verification_schema
from lettucedetect.detectors.prompt_utils import PROMPT_DIR, PromptUtils
from lettucedetect.detectors.response_cache import ResponseCache, make_cache_key
from lettucedetect.detectors.types import DetectedSpan

logger = logging.getLogger(__name__)

# The only logger that may carry customer text, the raw reply included.
content_logger = logging.getLogger("lettucedetect.content")

T = TypeVar("T")

VERIFY_SYSTEM = (
    "You are a strict fact-checker confirming whether flagged spans are truly unsupported."
)

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


def fetch_parsed(
    client: LLMClient,
    semaphore: threading.BoundedSemaphore,
    cache: ResponseCache | None,
    key: str,
    parse: Callable[[str], T],
    units: dict[str, int],
    *,
    system: str,
    user: str,
    model: str,
    temperature: float,
    schema: dict[str, Any],
) -> T:
    """Get the reply of one LLM call, from the cache or the client, and parse it.

    This is the only place a client or a cache is touched. The semaphore is held only for the
    client call and is released when the call returns or raises. A reply is written to the cache
    only after it parsed, and a cached reply that does not parse is deleted. The cache's
    synchronous methods are used, so the function runs as it is in a worker thread.

    :param client: the LLM client. Its errors propagate unwrapped.
    :param semaphore: the detector's bound on client calls in flight.
    :param cache: the reply cache, ``None`` for no caching.
    :param key: the cache key of the call.
    :param parse: turns the raw reply into the result, or raises ``UpstreamError``.
    :param units: the usage counts of the detect call; ``cache_hits`` or ``llm_calls`` is
        incremented.
    :returns: what ``parse`` returned.
    :raises UpstreamError: whatever ``parse`` raised.
    """
    reply = cache.get(key) if cache is not None else None
    from_cache = reply is not None
    if from_cache:
        units["cache_hits"] += 1
    else:
        units["llm_calls"] += 1
        with semaphore:
            reply = client.complete(
                system=system, user=user, model=model, temperature=temperature, schema=schema
            )
    content_logger.debug("LLM reply: %s", reply)
    try:
        parsed = parse(reply)
    except UpstreamError:
        if from_cache:
            cache.delete(key)
        raise
    if cache is not None and not from_cache:
        cache.set(key, reply)
    return parsed


def load_verify_template() -> Template:
    """Load the packaged prompt template of the verify pass."""
    return Template((PROMPT_DIR / "hallucination_verification.txt").read_text(encoding="utf-8"))


def build_verify_prompt(
    template: Template, lang: str, context: str, answer: str, texts: list[str]
) -> str:
    """Fill the verify prompt with the flagged span texts as candidates.

    :param template: the verify prompt template.
    :param lang: language code of the sources and the answer.
    :param context: the formatted sources, or the prompt.
    :param answer: the answer the spans are in.
    :param texts: the text of each flagged span.
    :returns: the user prompt of the verify call.
    """
    return template.substitute(
        lang=PromptUtils.get_full_language_name(lang),
        context=context,
        answer=answer,
        candidates="\n".join(f"- {text}" for text in texts),
    )


def parse_rejected(reply: str) -> set[str]:
    """Parse a verify reply into the span texts the verifier rejected.

    :param reply: the raw reply of the client.
    :returns: the ``text`` of every verdict whose ``is_hallucination`` is false.
    :raises UpstreamError: (``unparseable_response``) the reply is not JSON, has no
        ``verifications`` or has one that is not a list.
    """
    try:
        verdicts = json.loads(reply)["verifications"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise unparseable_response("a JSON object with verifications") from exc
    if not isinstance(verdicts, list):
        raise unparseable_response("a JSON object whose verifications is a list")
    return {
        verdict["text"]
        for verdict in verdicts
        if isinstance(verdict, dict)
        and verdict.get("is_hallucination") is False
        and isinstance(verdict.get("text"), str)
    }


def verify_spans(
    client: LLMClient,
    semaphore: threading.BoundedSemaphore,
    cache: ResponseCache | None,
    units: dict[str, int],
    template: Template,
    spans: list[dict[str, Any]],
    *,
    lang: str,
    context: str,
    answer: str,
    model: str,
    model_version: str | None,
    temperature: float,
) -> tuple[list[dict[str, Any]], bool]:
    """Re-judge the flagged spans with one strict LLM call and drop the rejected ones.

    A span the verifier does not mention is kept. When the verify call fails with an
    ``UpstreamError`` (its reply can't be parsed, or the client reports the service failed), the
    spans are returned unchanged and ``fallback_used`` is logged with the code
    ``verification_skipped``. Any other error propagates.

    :param spans: the first-pass spans, dicts with ``text``.
    :returns: the kept spans, and whether the verification was skipped.
    """
    user = build_verify_prompt(template, lang, context, answer, [span["text"] for span in spans])
    try:
        rejected = fetch_parsed(
            client,
            semaphore,
            cache,
            make_cache_key(user, model, model_version, temperature, "verify"),
            parse_rejected,
            units,
            system=VERIFY_SYSTEM,
            user=user,
            model=model,
            temperature=temperature,
            schema=build_verification_schema(),
        )
    except UpstreamError as exc:
        logger.warning(
            "verification_skipped: the verify call failed (%s), keeping %d first-pass spans",
            exc.code,
            len(spans),
            extra={"event": "fallback_used"},
        )
        return spans, True
    return [span for span in spans if span["text"] not in rejected], False
