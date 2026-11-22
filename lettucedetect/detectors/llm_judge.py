"""Detector that asks an LLM, as a zero-shot or few-shot judge, which parts of an answer are hallucinated."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from string import Template
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from lettucedetect.detectors._llm_shared import (
    Labels,
    complete,
    locate_items,
    resolve_taxonomy,
    to_detected_spans,
    unparseable_response,
)
from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.errors import ConfigurationError
from lettucedetect.detectors.llm_client import LLMClient, build_hallucination_schema
from lettucedetect.detectors.prompt_utils import LANG_TO_PASSAGE, PROMPT_DIR, PromptUtils
from lettucedetect.detectors.response_cache import ResponseCache
from lettucedetect.detectors.types import (
    Detection,
    DetectRequest,
    ResultWarning,
    SourcesInput,
    Usage,
)

logger = logging.getLogger(__name__)

__all__ = ["LLMJudgeConfig", "LLMJudgeDetector"]

_SYSTEM = "You are an expert in detecting hallucinations in LLM outputs."

_RESPONSE_FORMAT = """**Return** a JSON object following *exactly* this schema
   (no extra keys, no markdown, no code-block fences):

   `{"hallucination_list": ["substring1", "substring2", …]}`

   Each substring must be copied **verbatim** from the answer — an exact,
   character-for-character match. List substrings in answer order and include a
   repeated substring at most once per distinct occurrence. If none are found,
   return `{"hallucination_list": []}`."""


def fewshot_block(fewshot: list[dict[str, Any]]) -> str:
    """Render the few-shot examples of the judge prompt.

    :param fewshot: examples with ``source``, ``answer`` and ``hallucination_list``.
    :returns: the examples block, empty when there are no examples.
    """
    lines = []
    for i, ex in enumerate(fewshot, 1):
        lines.append(
            f"""<example{i}>
<source>{ex["source"]}</source>
<answer>{ex["answer"]}</answer>
<target>{{"hallucination_list": {json.dumps(ex["hallucination_list"], ensure_ascii=False)} }}</target>
</example{i}>"""
        )
    return "\n".join(lines)


def _label_lines(labels: Labels) -> str:
    return "\n".join(
        f'     - "{name}": {desc}' if desc else f'     - "{name}"' for name, desc in labels.items()
    )


def response_format_block(
    include_reasoning: bool, categories: Labels | None, subcategories: Labels | None
) -> str:
    """Compose the response-format instructions of the judge prompt for the enabled options.

    :param include_reasoning: whether the reply carries an analysis and per-span reasoning,
        confidence and verdict.
    :param categories: category labels, ``None`` for an untyped reply.
    :param subcategories: subcategory labels, ``None`` for none.
    :returns: the response-format block for the prompt template.
    """
    if not include_reasoning and not categories:
        return _RESPONSE_FORMAT

    example: dict = {"text": "substring1"}
    notes = [
        '- "text" must be an exact substring of the answer and items must be listed in answer order.',
        "- For repeated substrings, return at most one item per distinct occurrence.",
    ]
    if include_reasoning:
        example["reasoning"] = "compare the span against the source before judging it"
        notes.append(
            '- "reasoning" comes first: weigh the span against the source before scoring or judging it.'
        )
    if categories:
        example["category"] = next(iter(categories))
        notes.append(
            '- "category" classifies the hallucination as one of:\n' + _label_lines(categories)
        )
    if subcategories:
        example["subcategory"] = next(iter(subcategories))
        notes.append(
            '- "subcategory" further classifies the span as one of:\n' + _label_lines(subcategories)
        )
    if include_reasoning:
        example["confidence"] = 0.95
        example["is_hallucination"] = True
        notes.append(
            '- "confidence" is your confidence between 0 and 1 that the span is hallucinated.'
        )
        notes.append(
            '- "is_hallucination" is your final verdict. You may list a span you suspect and, after '
            "reasoning, set this to false if you conclude it is actually supported; such spans are "
            "discarded. Only spans you confirm should be true."
        )
    notes.append(
        "- Examples above may show only the hallucinated substrings; still return full objects."
    )

    example_json = json.dumps(example, ensure_ascii=False)
    notes_block = "\n   ".join(notes)
    if include_reasoning:
        schema_line = f'`{{"analysis": "...", "hallucination_list": [{example_json}, …]}}`'
        notes_block = (
            '- "analysis" (written first) is a brief claim-by-claim comparison of the answer '
            "against the source; reason here before listing any spans.\n   " + notes_block
        )
        empty_line = 'return `{"analysis": "...", "hallucination_list": []}`'
    else:
        schema_line = f'`{{"hallucination_list": [{example_json}, …]}}`'
        empty_line = 'return `{"hallucination_list": []}`'
    return (
        "**Return** a JSON object following *exactly* this schema\n"
        "   (no extra keys, no markdown, no code-block fences):\n\n"
        f"   {schema_line}\n\n"
        f"   {notes_block}\n\n"
        f"   If none are found, {empty_line}."
    )


def parse_hallucination_list(reply: str) -> list[Any]:
    """Parse a judge reply into its list of items.

    A ``hallucination_list`` that is a JSON-encoded string is decoded.

    :param reply: the raw reply of the client.
    :returns: the items of ``hallucination_list``.
    :raises UpstreamError: (``unparseable_response``) the reply is not JSON, has no
        ``hallucination_list`` or has one that is not a list.
    """
    try:
        items = json.loads(reply)["hallucination_list"]
        if isinstance(items, str):
            items = json.loads(items)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise unparseable_response("a JSON object with a hallucination_list") from exc
    if not isinstance(items, list):
        raise unparseable_response("a JSON object whose hallucination_list is a list")
    return items


class LLMJudgeConfig(BaseModel):
    """Options of a :class:`LLMJudgeDetector`.

    An invalid option raises :class:`ConfigurationError` with ``param`` set to
    ``config.<option>`` and the code ``invalid_config``, ``invalid_language`` (``lang``) or
    ``fewshot_not_found`` (a ``fewshot_path`` that does not exist).

    :param temperature: sampling temperature of the LLM calls.
    :param lang: language of the sources and the answer, one of the supported languages.
    :param zero_shot: leave the few-shot examples out of the prompt.
    :param fewshot_path: JSON file of few-shot examples; the packaged examples of ``lang`` when
        ``None``.
    :param prompt_path: prompt template file; the packaged judge prompt when ``None``.
    :param include_reasoning: ask for a reasoning and a confidence per span. The confidence
        becomes the span's ``score``.
    :param include_taxonomy: type each span. ``True`` uses the unified taxonomy; a list of
        category names, a flat ``{category: description}`` dict or
        ``{"categories": {...}, "subcategories": {...}}`` sets the labels.
    :param min_score: drop spans the model scored below this value, in ``[0, 1]``, before the
        verify pass. ``min_score`` of ``detect`` filters the final spans on top of it.
    :param verify: run a second, strict LLM call that re-judges the flagged spans.
    :param max_concurrency: the most LLM calls in flight at once on one detector, at least 1.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    temperature: float = 0.0
    lang: str = "en"
    zero_shot: bool = False
    fewshot_path: str | Path | None = None
    prompt_path: str | Path | None = None
    include_reasoning: bool = False
    include_taxonomy: bool | list[str] | dict[str, Any] = False
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    verify: bool = False
    max_concurrency: int = Field(default=30, ge=1, strict=True)

    @model_validator(mode="wrap")
    @classmethod
    def _raise_configuration_error(cls, data: Any, handler: Any) -> LLMJudgeConfig:  # noqa: ANN401
        try:
            config = handler(data)
        except ValidationError as exc:
            first = exc.errors(include_input=False, include_url=False)[0]
            option = str(first["loc"][0]) if first["loc"] else None
            raise ConfigurationError(
                f"Invalid LLMJudgeConfig option {option!r}; see the LLMJudgeConfig docstring.",
                code="invalid_config",
                param=f"config.{option}" if option else None,
            ) from exc
        if config.lang not in LANG_TO_PASSAGE:
            raise ConfigurationError(
                f"Invalid language; choose from {', '.join(LANG_TO_PASSAGE)}.",
                code="invalid_language",
                param="config.lang",
            )
        if config.fewshot_path is not None and not Path(config.fewshot_path).is_file():
            raise ConfigurationError(
                "The few-shot examples file does not exist; pass the path of a JSON file, "
                "or leave fewshot_path unset for the packaged examples.",
                code="fewshot_not_found",
                param="config.fewshot_path",
            )
        if config.prompt_path is not None and not Path(config.prompt_path).is_file():
            raise ConfigurationError(
                "The prompt template file does not exist; pass the path of a template file, "
                "or leave prompt_path unset for the packaged prompt.",
                code="invalid_config",
                param="config.prompt_path",
            )
        return config


class LLMJudgeDetector(BaseDetector):
    """Detect hallucinations by asking an LLM which parts of the answer the sources don't support.

    The caller passes the client, so construction builds no client, reads no environment variable
    and writes no file. Each item of the reply is located in the answer by its first verbatim
    match that overlaps no earlier item.

    ``score`` is the model's own ``confidence``, which it gives with ``include_reasoning=True``.
    A span without one scores 1.0, "flagged without a score", and the result then carries one
    ``synthetic_score`` warning. Other warnings: ``span_not_found`` (items that are not in the
    answer were dropped) and ``item_malformed`` (items that are neither a string nor an object
    with ``text`` were dropped).

    Example::

        detector = LLMJudgeDetector(
            OpenAIClient(api_key="<key>"),
            "gpt-4.1-mini",
            config=LLMJudgeConfig(include_reasoning=True),
        )
        result = detector.detect(
            SourcesInput(
                question="What is the capital of France?",
                sources=[Chunk(id="doc-1", text="The capital of France is Paris.")],
                answer="The capital of France is Lyon.",
            )
        )

    :param client: the LLM client every call goes through. Its errors propagate unwrapped.
    :param model: the model name passed to the client; also the ``model_id`` in ``usage``.
    :param config: options, ``LLMJudgeConfig()`` when ``None``.
    :param cache: a cache for the replies; ``None`` means no caching.
    :param model_version: version reported in ``usage``.
    """

    supports_tokens = False
    supports_detail = False
    model_id: str | None = None
    model_version: str | None = None

    def __init__(
        self,
        client: LLMClient,
        model: str,
        *,
        config: LLMJudgeConfig | None = None,
        cache: ResponseCache | None = None,
        model_version: str | None = None,
    ) -> None:
        """Build the detector; see the class docstring for the arguments."""
        self.config = config if config is not None else LLMJudgeConfig()
        self.client = client
        self.model_id = model
        self.model_version = model_version
        self.cache = cache
        self.categories, self.subcategories = resolve_taxonomy(self.config.include_taxonomy)
        self._schema = build_hallucination_schema(
            include_reasoning=self.config.include_reasoning,
            categories=list(self.categories) if self.categories else None,
            subcategories=list(self.subcategories) if self.subcategories else None,
        )
        fewshot: list[dict[str, Any]] = []
        if not self.config.zero_shot:
            fewshot_path = (
                self.config.fewshot_path or PROMPT_DIR / f"examples_{self.config.lang}.json"
            )
            fewshot = json.loads(Path(fewshot_path).read_text(encoding="utf-8"))
        self._fewshot_block = fewshot_block(fewshot)
        self._response_format_block = response_format_block(
            self.config.include_reasoning, self.categories, self.subcategories
        )
        prompt_path = self.config.prompt_path or PROMPT_DIR / "hallucination_detection.txt"
        self._template = Template(Path(prompt_path).read_text(encoding="utf-8"))
        # Held only around a client call, so every entry point shares the bound.
        self._semaphore = threading.BoundedSemaphore(self.config.max_concurrency)

    def _build_prompt(self, context: str, answer: str) -> str:
        """Fill the judge prompt template for one context and answer."""
        return self._template.substitute(
            lang=PromptUtils.get_full_language_name(self.config.lang),
            context=context,
            answer=answer,
            fewshot_block=self._fewshot_block,
            response_format_block=self._response_format_block,
        )

    def _detect(
        self, request: DetectRequest, *, include_tokens: bool, include_detail: bool
    ) -> Detection:
        """Ask the LLM for the hallucinated parts of the answer and locate them.

        :param request: the sources (or prompt) and the answer to check.
        :param include_tokens: unused, ``detect`` rejects it (``supports_tokens`` is ``False``).
        :param include_detail: unused, ``detect`` rejects it (``supports_detail`` is ``False``).
        :returns: the detection result.
        :raises UpstreamError: (``unparseable_response``) the reply is not JSON, has no
            ``hallucination_list`` or has one that is not a list.
        """
        config = self.config
        answer = request.answer
        if isinstance(request, SourcesInput):
            context = PromptUtils.format_context(
                [chunk.text for chunk in request.sources], request.question, config.lang
            )
        else:
            context = request.prompt
        logger.debug("detect started: kind=%s model_id=%s", request.kind, self.model_id)

        reply = complete(
            self.client,
            self._semaphore,
            system=_SYSTEM,
            user=self._build_prompt(context, answer),
            model=self.model_id,
            temperature=config.temperature,
            schema=self._schema,
        )
        items = parse_hallucination_list(reply)
        located, not_found, malformed = locate_items(items, answer, config.min_score)

        warnings = []
        if not_found:
            warnings.append(
                ResultWarning(
                    code="span_not_found",
                    message=f"{not_found} of {len(items)} items are not in the answer and were dropped.",
                )
            )
        if malformed:
            warnings.append(
                ResultWarning(
                    code="item_malformed",
                    message=(
                        f"{malformed} of {len(items)} items are neither a string nor an object "
                        "with text and were dropped."
                    ),
                )
            )
        if warnings:
            logger.warning(
                "items dropped: span_not_found=%d item_malformed=%d items=%d",
                not_found,
                malformed,
                len(items),
            )

        spans, synthetic = to_detected_spans(located)
        if synthetic:
            warnings.append(
                ResultWarning(
                    code="synthetic_score",
                    message="Spans without a confidence from the model score 1.0.",
                )
            )
            logger.debug("synthetic_score: spans without a model confidence score 1.0")
        logger.debug("detect finished: items=%d spans=%d", len(items), len(spans))
        return Detection(
            spans=spans,
            usage=[Usage(model_id=self.model_id, model_version=self.model_version, units={})],
            warnings=warnings,
        )
