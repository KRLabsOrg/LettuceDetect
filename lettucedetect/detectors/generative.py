"""Detector that runs a fine-tuned generative span detector through an LLM client."""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lettucedetect.detectors._llm_shared import (
    dropped_item_warnings,
    fetch_parsed,
    load_verify_template,
    locate_items,
    request_context,
    resolve_taxonomy,
    to_detected_spans,
    unparseable_response,
    verify_spans,
)
from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.errors import ConfigurationError
from lettucedetect.detectors.llm_client import LLMClient, build_generative_schema
from lettucedetect.detectors.prompt_utils import LANG_TO_PASSAGE
from lettucedetect.detectors.response_cache import ResponseCache, make_cache_key
from lettucedetect.detectors.types import Detection, DetectRequest, ResultWarning, Usage
from lettucedetect.prompts import generative as gen

logger = logging.getLogger(__name__)

__all__ = ["GenerativeConfig", "GenerativeDetector"]


def parse_hallucinated_spans(reply: str) -> list[Any]:
    """Parse a reply of a generative detector into its list of items.

    Code fences around the JSON object and text around it are tolerated.

    :param reply: the raw reply of the client.
    :returns: the items of ``hallucinated_spans``.
    :raises UpstreamError: (``unparseable_response``) the reply is not JSON, has no
        ``hallucinated_spans`` or has one that is not a list.
    """
    text = reply.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            embedded = re.search(r"\{.*\}", text, re.DOTALL)
            if embedded is None:
                raise
            payload = json.loads(embedded.group(0))
        items = payload["hallucinated_spans"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise unparseable_response("a JSON object with hallucinated_spans") from exc
    if not isinstance(items, list):
        raise unparseable_response("a JSON object whose hallucinated_spans is a list")
    return items


def to_items(spans: list[Any]) -> list[Any]:
    """Rename the keys of a generative reply to the ones :func:`locate_items` reads.

    ``explanation`` becomes ``reasoning``. Only ``text``, ``category``, ``subcategory`` and
    ``explanation`` are read, so no item carries a confidence or a verdict of its own.

    :param spans: the items of ``hallucinated_spans``.
    :returns: the items, objects with the renamed keys and anything else unchanged.
    """
    items = []
    for span in spans:
        if isinstance(span, dict):
            item = {key: span[key] for key in ("text", "category", "subcategory") if key in span}
            if "explanation" in span:
                item["reasoning"] = span["explanation"]
            span = item
        items.append(span)
    return items


class GenerativeConfig(BaseModel):
    """Options of a :class:`GenerativeDetector`.

    An invalid or unknown option raises :class:`ConfigurationError` with ``param`` set to
    ``config.<option>`` and the code ``invalid_config`` or ``invalid_language`` (``lang``). The
    prompt is fixed, so the judge's ``zero_shot``, ``fewshot_path`` and ``prompt_path`` are
    unknown options here.

    :param temperature: sampling temperature of the LLM calls.
    :param lang: language of the sources and the answer, one of the supported languages. It
        labels the passages and is named in the verify prompt.
    :param include_reasoning: ask for an explanation per span, returned as ``reasoning``.
    :param include_taxonomy: ``False`` and ``True`` keep the labels the models were trained on.
        A list of category names, a flat ``{category: description}`` dict or
        ``{"categories": {...}, "subcategories": {...}}`` replaces them in the prompt; without
        subcategories the only subcategory is ``unspecified``.
    :param verify: run a second, strict LLM call that re-judges the flagged spans.
    :param max_concurrency: the most LLM calls in flight at once on one detector, the verify
        call included; at least 1.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    temperature: float = 0.0
    lang: str = "en"
    include_reasoning: bool = False
    include_taxonomy: bool | list[str] | dict[str, Any] = False
    verify: bool = False
    max_concurrency: int = Field(default=30, ge=1, strict=True)

    @model_validator(mode="wrap")
    @classmethod
    def _raise_configuration_error(cls, data: Any, handler: Any) -> GenerativeConfig:  # noqa: ANN401
        try:
            config = handler(data)
        except ValidationError as exc:
            first = exc.errors(include_input=False, include_url=False)[0]
            option = str(first["loc"][0]) if first["loc"] else None
            raise ConfigurationError(
                f"Invalid GenerativeConfig option {option!r}; see the GenerativeConfig docstring.",
                code="invalid_config",
                param=f"config.{option}" if option else None,
            ) from exc
        if config.lang not in LANG_TO_PASSAGE:
            raise ConfigurationError(
                f"Invalid language; choose from {', '.join(LANG_TO_PASSAGE)}.",
                code="invalid_language",
                param="config.lang",
            )
        return config


class GenerativeDetector(BaseDetector):
    """Detect hallucinations with a fine-tuned generative span detector served behind an LLM client.

    The models answer one fixed prompt (:mod:`lettucedetect.prompts.generative`) with typed
    spans. The caller passes the client, so construction builds no client, reads no environment
    variable and writes no file. Each item of the reply is located in the answer by its first
    verbatim match that overlaps no earlier item.

    Every span has ``category`` and ``subcategory`` from the reply, and ``reasoning`` from the
    reply's ``explanation``, which the model gives with ``include_reasoning=True``. The models
    give no confidence: every ``score`` is 1.0, "flagged without a score", and a result with
    spans carries one ``synthetic_score`` warning. Other warnings: ``span_not_found`` (items
    that are not in the answer were dropped), ``item_malformed`` (items that are neither a
    string nor an object with ``text`` were dropped) and ``verification_skipped`` (see below).

    With ``verify=True`` a second call re-judges the flagged spans and the rejected ones are
    dropped. When that call fails with an ``UpstreamError``, the first-pass spans are returned
    with the ``verification_skipped`` warning.

    ``usage`` has one entry with the units ``llm_calls`` (calls actually made, the verify call
    included), ``cache_hits``, ``source_chars`` and ``answer_chars``.

    Example::

        detector = GenerativeDetector(
            OpenAIClient(api_key="<key>", base_url="http://localhost:8000/v1"),
            "KRLabsOrg/lettucedect-v2-qwen-2b",
        )
        result = detector.detect(
            SourcesInput(
                question="What is the capital of France?",
                sources=[Chunk(id="doc-1", text="The capital of France is Paris.")],
                answer="The capital of France is Lyon.",
            )
        )

    :param client: the LLM client every call goes through. Its errors propagate unwrapped.
    :param model_id: the model name passed to the client; also the ``model_id`` in ``usage``.
    :param config: options, ``GenerativeConfig()`` when ``None``.
    :param cache: a cache for the replies; ``None`` means no caching and no file. A reply is
        stored only after it parsed, under ``make_cache_key`` with the kinds ``"native"`` and
        ``"verify"``.
    :param model_version: version reported in ``usage`` and part of the cache key.
    """

    supports_tokens = False
    supports_detail = False
    model_id: str | None = None
    model_version: str | None = None

    def __init__(
        self,
        client: LLMClient,
        model_id: str,
        *,
        config: GenerativeConfig | None = None,
        cache: ResponseCache | None = None,
        model_version: str | None = None,
    ) -> None:
        """Build the detector; see the class docstring for the arguments."""
        self.config = config if config is not None else GenerativeConfig()
        self.client = client
        self.model_id = model_id
        self.model_version = model_version
        self.cache = cache
        explain = self.config.include_reasoning
        categories, subcategories = resolve_taxonomy(self.config.include_taxonomy)
        if isinstance(self.config.include_taxonomy, bool) or not categories:
            self._system = gen.SYSTEM_EXPL if explain else gen.SYSTEM_BASE
            self._schema = build_generative_schema(explain=explain)
        else:
            # Custom labels replace the trained ones as name and description text only, so the
            # prompt keeps its trained structure.
            subcategories = subcategories or {
                "unspecified": gen.SUBCATEGORY_DESCRIPTIONS["unspecified"]
            }
            self._system = gen.build_system_prompt(
                explain=explain, categories=categories, subcategories=subcategories
            )
            self._schema = build_generative_schema(
                explain=explain, categories=list(categories), subcategories=list(subcategories)
            )
        self._verify_template = load_verify_template()
        # Held only around a client call, so every entry point shares the bound.
        self._semaphore = threading.BoundedSemaphore(self.config.max_concurrency)

    def _detect(
        self, request: DetectRequest, *, include_tokens: bool, include_detail: bool
    ) -> Detection:
        """Ask the model for the hallucinated spans of the answer and locate them.

        :param request: the sources (or prompt) and the answer to check.
        :param include_tokens: unused, ``detect`` rejects it (``supports_tokens`` is ``False``).
        :param include_detail: unused, ``detect`` rejects it (``supports_detail`` is ``False``).
        :returns: the detection result.
        :raises UpstreamError: (``unparseable_response``) the reply is not JSON, has no
            ``hallucinated_spans`` or has one that is not a list. Such a reply is not cached,
            and a cached one is deleted.
        """
        config = self.config
        answer = request.answer
        context, units = request_context(request, config.lang)
        logger.debug("detect started: kind=%s model_id=%s", request.kind, self.model_id)

        user = gen.build_user_message(context, answer)
        items = fetch_parsed(
            self.client,
            self._semaphore,
            self.cache,
            make_cache_key(
                self._system + user, self.model_id, self.model_version, config.temperature, "native"
            ),
            parse_hallucinated_spans,
            units,
            system=self._system,
            user=user,
            model=self.model_id,
            temperature=config.temperature,
            schema=self._schema,
        )
        located, not_found, malformed = locate_items(to_items(items), answer)
        warnings = dropped_item_warnings(not_found, malformed, len(items))

        if config.verify and located:
            located, skipped = verify_spans(
                self.client,
                self._semaphore,
                self.cache,
                units,
                self._verify_template,
                located,
                lang=config.lang,
                context=context,
                answer=answer,
                model=self.model_id,
                model_version=self.model_version,
                temperature=config.temperature,
            )
            if skipped:
                warnings.append(
                    ResultWarning(
                        code="verification_skipped",
                        message="The verify call failed; the spans are unverified.",
                    )
                )

        spans, synthetic = to_detected_spans(located)
        if synthetic:
            warnings.append(
                ResultWarning(
                    code="synthetic_score",
                    message="The model gives no confidence; every span scores 1.0.",
                )
            )
            logger.debug("synthetic_score: the model gives no confidence, every span scores 1.0")
        logger.debug("detect finished: items=%d spans=%d", len(items), len(spans))
        return Detection(
            spans=spans,
            usage=[Usage(model_id=self.model_id, model_version=self.model_version, units=units)],
            warnings=warnings,
        )
