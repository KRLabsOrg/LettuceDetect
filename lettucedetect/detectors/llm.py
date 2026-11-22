from __future__ import annotations

import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from lettucedetect.detectors._deprecation import (
    _UNSET,
    _message,
    _warn,
    deprecated,
    warn_if_passed,
)
from lettucedetect.detectors._llm_shared import (
    load_verify_template,
    locate_items,
    resolve_taxonomy,
)
from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.errors import ConfigurationError, InvalidInputError
from lettucedetect.detectors.file_cache import FileCache
from lettucedetect.detectors.generative import GenerativeConfig, GenerativeDetector
from lettucedetect.detectors.llm_client import (
    LLMClient,
    build_hallucination_schema,
    make_llm_client,
)
from lettucedetect.detectors.llm_judge import (
    LLMJudgeConfig,
    LLMJudgeDetector,
    response_format_block,
)
from lettucedetect.detectors.prompt_utils import Lang, PromptUtils
from lettucedetect.detectors.types import BatchResult, Detection, DetectRequest, PromptInput


def _looks_native(model: str) -> bool:
    """Heuristically decide whether ``model`` is one of our fine-tuned generative detectors.

    True for the ``lettucedect-v2`` generative models (Qwen/LFM), which expect the
    frozen :mod:`lettucedetect.prompts.generative` prompt and emit ``hallucinated_spans``.
    Excludes the encoder detectors (those go through ``TransformerDetector``). Served
    under a non-standard name? Pass ``native=True`` explicitly.
    """
    m = model.lower()
    if any(enc in m for enc in ("bert", "modernbert", "eurobert")):
        return False
    return "lettucede" in m and any(k in m for k in ("qwen", "lfm", "generative"))


# Token-level probabilities reported when the LLM gave us no per-span confidence
# (i.e. include_reasoning=False). Same fixed-constant idea as RAGFactCheckerDetector.
_HALLUCINATED_TOKEN_PROB = 0.9
_SUPPORTED_TOKEN_PROB = 0.1


@deprecated("LLMJudgeDetector or GenerativeDetector")
class LLMDetector(BaseDetector):
    """LLM-powered hallucination detector.

    .. deprecated:: 0.3.0
        Use :class:`LLMJudgeDetector` or :class:`GenerativeDetector` and pass them a client.
        ``LLMDetector`` will be removed in 0.5.0. It builds one of the two, chosen by the model
        name or by ``native=``, and its ``provider``, ``client_kwargs`` and ``cache_file``
        arguments and the omitted ``client`` are deprecated with it. ``detect`` and ``adetect``
        are those of the detector it built.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        lang: Lang = "en",
        zero_shot: bool = False,
        fewshot_path: str | None = None,
        prompt_path: str | None = None,
        cache_file: str | None = None,
        provider: str = _UNSET,
        client: LLMClient | None = None,
        include_reasoning: bool = False,
        include_taxonomy: bool | list[str] | dict[str, str] = False,
        min_confidence: float = 0.0,
        verify: bool = False,
        native: bool | None = None,
        **client_kwargs,
    ):
        """Initialize the LLMDetector.

        :param model: The model to use for hallucination detection.
        :param temperature: The temperature to use for hallucination detection.
        :param lang: The language to use for hallucination detection.
        :param zero_shot: Whether to use zero-shot hallucination detection.
        :param fewshot_path: The path to the few-shot examples.
        :param prompt_path: The path to the prompt.
        :param cache_file: The path to the cache file. Deprecated: without it nothing is cached
            and no file is written.
        :param provider: Backend provider, ``"openai"`` (the default) or ``"bedrock"``. Ignored if
            ``client`` is given. Deprecated.
        :param client: An explicit :class:`LLMClient` instance; overrides ``provider``. Leaving it
            out is deprecated.
        :param include_reasoning: If True, ask the LLM for a confidence score and a short
            reasoning per span and include them as ``confidence`` and ``reasoning`` keys
            in the returned spans.
        :param include_taxonomy: If True, ask the LLM to classify each span into one of
            the unified taxonomy categories defined in
            :data:`lettucedetect.datasets.taxonomy.CATEGORY_DEFINITIONS`. Pass a
            ``{category: description}`` dict to use a custom taxonomy with your own
            definitions, or a list of category names (definitions are reused from the
            unified taxonomy where the name matches, otherwise omitted). The
            classification is included as a ``category`` key in the returned spans.
        :param min_confidence: Drop spans whose ``confidence`` is below this threshold.
            Only meaningful together with ``include_reasoning`` (which produces the score).
        :param verify: If True, run a second adversarial pass that re-judges each flagged
            span against the source and discards the ones it cannot confirm.
        :param native: ``True`` runs :class:`GenerativeDetector`, ``False``
            :class:`LLMJudgeDetector`; ``None`` chooses by the model name.
        :param client_kwargs: Extra keyword arguments forwarded to the provider client
            constructor. Deprecated.
        :raises ConfigurationError: ``native_options_conflict`` for the native path with
            ``zero_shot``, ``fewshot_path`` or ``prompt_path``; the codes of
            :class:`LLMJudgeConfig` and :class:`GenerativeConfig` for an invalid option, such as
            ``invalid_language`` and ``fewshot_not_found``.
        """
        warn_if_passed(provider, "provider", "the 'client' argument")
        warn_if_passed(client_kwargs or _UNSET, "client_kwargs", "the 'client' argument")
        if cache_file is not None:
            warn_if_passed(
                cache_file,
                "cache_file",
                "cache=FileCache(path) on LLMJudgeDetector or GenerativeDetector",
            )

        self.model = model
        self.temperature = temperature
        self.lang = lang
        self.zero_shot = zero_shot
        self.include_reasoning = include_reasoning
        self.min_confidence = min_confidence
        self.verify = verify
        # Native mode: serve our fine-tuned generative span detectors (lettucedect-v2-*)
        # with their frozen training prompt + hallucinated_spans contract, instead of
        # the zero-shot judge prompt. Auto-detected from the model id; override explicitly.
        self.native = _looks_native(model) if native is None else native
        conflicts = {
            "zero_shot": zero_shot,
            "fewshot_path": fewshot_path,
            "prompt_path": prompt_path,
        }
        conflict = next((name for name, value in conflicts.items() if value), None)
        if self.native and conflict:
            raise ConfigurationError(
                "zero_shot, fewshot_path, and prompt_path have no effect on the native "
                "generative detectors (they use their frozen training prompt); pass "
                "native=False to prompt this model as a generic LLM judge instead.",
                code="native_options_conflict",
                param=conflict,
            )
        config = (
            GenerativeConfig(
                temperature=temperature,
                lang=lang,
                include_reasoning=include_reasoning,
                include_taxonomy=include_taxonomy,
                verify=verify,
            )
            if self.native
            else LLMJudgeConfig(
                temperature=temperature,
                lang=lang,
                zero_shot=zero_shot,
                fewshot_path=fewshot_path,
                prompt_path=prompt_path,
                include_reasoning=include_reasoning,
                include_taxonomy=include_taxonomy,
                min_score=min_confidence,
                verify=verify,
            )
        )
        self.categories, self.subcategories = resolve_taxonomy(include_taxonomy)
        self.schema = build_hallucination_schema(
            include_reasoning=include_reasoning,
            categories=list(self.categories) if self.categories else None,
            subcategories=list(self.subcategories) if self.subcategories else None,
        )
        if client is None:
            _warn(
                _message(
                    "LLMDetector without the 'client' argument",
                    "client=OpenAIClient(...) or client=BedrockClient(...)",
                ),
                2,
            )
            client = make_llm_client("openai" if provider is _UNSET else provider, **client_kwargs)
        self.client = client
        self.verify_template = load_verify_template()

        # A cache exists only for a cache_file the caller passed.
        self.cache = None if cache_file is None else FileCache(cache_file)

        # The native path runs through GenerativeDetector, the judge path through LLMJudgeDetector.
        self._generative = (
            GenerativeDetector(self.client, model, config=config, cache=self.cache)
            if self.native
            else None
        )
        self._judge = (
            None
            if self.native
            else LLMJudgeDetector(self.client, model, config=config, cache=self.cache)
        )
        self._delegate = self._generative if self.native else self._judge
        self.model_id = self._delegate.model_id

    def detect(
        self,
        request: DetectRequest,
        *,
        min_score: float = 0.0,
        include_tokens: bool = False,
        include_detail: bool = False,
    ) -> Detection:
        """Detect hallucinated spans with the judge or the generative detector that was built.

        :param request: the sources (or prompt) and the answer to check.
        :param min_score: drop spans scored below this value, in ``[0, 1]``.
        :param include_tokens: not supported, raises ``InvalidInputError``.
        :param include_detail: not supported, raises ``InvalidInputError``.
        :returns: the detection result of that detector.
        """
        return self._delegate.detect(
            request,
            min_score=min_score,
            include_tokens=include_tokens,
            include_detail=include_detail,
        )

    async def adetect(
        self,
        request: DetectRequest,
        *,
        min_score: float = 0.0,
        include_tokens: bool = False,
        include_detail: bool = False,
    ) -> Detection:
        """Async version of :meth:`detect`, forwarded to the same detector.

        :param request: the sources (or prompt) and the answer to check.
        :param min_score: drop spans scored below this value, in ``[0, 1]``.
        :param include_tokens: not supported, raises ``InvalidInputError``.
        :param include_detail: not supported, raises ``InvalidInputError``.
        :returns: the detection result of that detector.
        """
        return await self._delegate.adetect(
            request,
            min_score=min_score,
            include_tokens=include_tokens,
            include_detail=include_detail,
        )

    def detect_batch(
        self,
        requests: Sequence[DetectRequest],
        *,
        min_score: float = 0.0,
        include_tokens: bool = False,
        include_detail: bool = False,
    ) -> list[BatchResult]:
        """Detect hallucinated spans for many requests with the detector that was built.

        :param requests: the requests, any mix of ``SourcesInput`` and ``PromptInput``.
        :param min_score: drop spans scored below this value, in ``[0, 1]``.
        :param include_tokens: not supported, raises ``InvalidInputError``.
        :param include_detail: not supported, raises ``InvalidInputError``.
        :returns: one result per request, in input order, from that detector.
        """
        return self._delegate.detect_batch(
            requests,
            min_score=min_score,
            include_tokens=include_tokens,
            include_detail=include_detail,
        )

    def _response_format_block(self) -> str:
        """Compose the response-format instructions for the enabled options.

        :return: The response-format block for the prompt template.
        """
        return response_format_block(self.include_reasoning, self.categories, self.subcategories)

    def _build_prompt(self, context: str, answer: str) -> str:
        """Fill the judge template with runtime values, inserting few-shot examples.

        :param context: The context string.
        :param answer: The answer string.
        :return: The filled template.
        """
        return self._judge._build_prompt(context, answer)

    @staticmethod
    def _to_spans(items: list[str | dict], answer: str, min_confidence: float = 0.0) -> list[dict]:
        """Convert hallucinated items to a list of spans.

        Items the model self-rejected (``is_hallucination`` is False) or scored
        below ``min_confidence`` are dropped; the ``is_hallucination`` verdict is
        not carried into the output since every returned span is a hallucination.

        :param items: List of hallucinated substrings, or objects with a ``text`` key
            plus ``reasoning``/``category``/``confidence``/``is_hallucination`` keys
            depending on the enabled options.
        :param answer: The answer string.
        :param min_confidence: Drop spans whose ``confidence`` is below this value.
        :returns: List of spans.
        """
        return locate_items(items, answer, min_confidence)[0]

    @staticmethod
    def _convert_to_tokens(answer: str, spans: list[dict]) -> list[dict]:
        """Project predicted character spans onto the whitespace tokens of ``answer``.

        Emits one ``{"token", "pred", "prob"}`` dict per whitespace-delimited token,
        matching the token format the other detectors produce
        (``TransformerDetector._predict_single``, ``RAGFactCheckerDetector._convert_to_tokens``).
        A token is flagged (``pred=1``) when its character range overlaps any predicted
        span. ``prob`` uses the overlapping span's ``confidence`` when present (only set
        with ``include_reasoning=True``), otherwise the fixed constants above.

        :param answer: The answer the spans were matched against.
        :param spans: Character spans as returned by :meth:`_to_spans`.
        :returns: One token dict per whitespace token in ``answer``.
        """
        tokens = []
        for match in re.finditer(r"\S+", answer):
            hit = next(
                (s for s in spans if s["start"] < match.end() and match.start() < s["end"]),
                None,
            )
            if hit is None:
                prob = _SUPPORTED_TOKEN_PROB
            else:
                prob = hit.get("confidence")
                if prob is None:
                    prob = _HALLUCINATED_TOKEN_PROB
            tokens.append({"token": match.group(), "pred": 0 if hit is None else 1, "prob": prob})
        return tokens

    def _predict(
        self, prompt: str, answer: str, output_format: str = "spans", min_confidence: float = 0.0
    ) -> list:
        """Single (prompt, answer) pair → hallucination spans or tokens.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :param min_confidence: Drop ``"spans"`` below this confidence threshold.
        :returns: List of spans, or per-token dicts when ``output_format="tokens"``.
        """
        spans = self._detect_spans(prompt, answer, output_format, min_confidence)
        if output_format == "tokens":
            return self._convert_to_tokens(answer, spans)
        return spans

    def _detect_spans(
        self, prompt: str, answer: str, output_format: str, min_confidence: float
    ) -> list[dict]:
        """Run the generative or the judge detector and turn its result into span dicts.

        ``confidence`` is the span's ``score``. For ``"tokens"`` a span the model gave no
        confidence for has no ``confidence`` key, so its tokens get the fixed probability.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :param min_confidence: Passed as ``min_score`` for ``"spans"``; tokens are not filtered.
        :returns: List of spans.
        :raises UpstreamError: (``unparseable_response``) the reply can't be parsed.
        """
        for_tokens = output_format == "tokens"
        detection = self._delegate.detect(
            PromptInput(prompt=prompt, answer=answer),
            min_score=0.0 if for_tokens else min_confidence,
        )
        # With a synthetic_score warning, a score of 1.0 marks a span without a model confidence.
        synthetic = any(warning.code == "synthetic_score" for warning in detection.warnings)
        spans = []
        for span in detection.spans:
            item = {"start": span.start, "end": span.end, "text": span.text}
            if not (for_tokens and synthetic and span.score == 1.0):
                item["confidence"] = span.score
            for key in ("reasoning", "category", "subcategory"):
                value = getattr(span, key)
                if value is not None:
                    item[key] = value
            spans.append(item)
        return spans

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "spans",
        min_confidence: float = 0.0,
    ) -> list:
        """Predict hallucination spans from the provided context, answer, and question.

        :param context: List of passages that were supplied to the LLM / user.
        :param answer: Model-generated answer to inspect.
        :param question: Original question (``None`` for summarisation).
        :param output_format: ``"tokens"`` for per-token dicts, ``"spans"`` for character spans.
        :param min_confidence: Drop spans whose ``confidence`` is below this threshold
            (in ``[0, 1]``). Applied on top of the constructor-level ``min_confidence``;
            ``0.0`` keeps every span. Only affects ``output_format="spans"`` — token
            output is returned unfiltered.
        :returns: List of spans, or per-token dicts when ``output_format="tokens"``.
        :raises InvalidInputError: ``unsupported_output_format`` or ``invalid_min_score``.
        """
        self._check_legacy_arguments(output_format, min_confidence)
        # Use PromptUtils to format the context and question
        full_prompt = PromptUtils.format_context(context, question, self.lang)
        return self._predict(full_prompt, answer, output_format, min_confidence)

    def predict_prompt(
        self, prompt: str, answer: str, output_format: str = "spans", min_confidence: float = 0.0
    ) -> list:
        """Predict hallucination spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: ``"tokens"`` for per-token dicts, ``"spans"`` for character spans.
        :param min_confidence: Drop spans below this confidence threshold (``[0, 1]``); applied
            on top of the constructor-level ``min_confidence``. Ignored when
            ``output_format="tokens"``.
        :returns: List of spans, or per-token dicts when ``output_format="tokens"``.
        :raises InvalidInputError: ``unsupported_output_format`` or ``invalid_min_score``.
        """
        self._check_legacy_arguments(output_format, min_confidence)
        return self._predict(prompt, answer, output_format, min_confidence)

    def predict_prompt_batch(
        self,
        prompts: list[str],
        answers: list[str],
        output_format: str = "spans",
        min_confidence: float = 0.0,
    ) -> list:
        """Predict hallucination spans from the provided prompts and answers.

        :param prompts: List of prompt strings.
        :param answers: List of answer strings.
        :param output_format: ``"tokens"`` for per-token dicts, ``"spans"`` for character spans.
        :param min_confidence: Drop spans below this confidence threshold (``[0, 1]``); applied
            on top of the constructor-level ``min_confidence``. Not applied to token output.
        :returns: One result per input pair: spans, or per-token dicts when
            ``output_format="tokens"``.
        :raises InvalidInputError: ``length_mismatch`` if ``len(prompts) != len(answers)``,
            ``unsupported_output_format`` or ``invalid_min_score``.
        """
        if len(prompts) != len(answers):
            raise InvalidInputError(
                "Number of prompts must match number of answers; pass lists of the same length.",
                code="length_mismatch",
                param="answers",
            )
        self._check_legacy_arguments(output_format, min_confidence)

        with ThreadPoolExecutor(max_workers=30) as pool:
            futs = [
                pool.submit(self._predict, p, a, output_format, min_confidence)
                for p, a in zip(prompts, answers)
            ]
            return [fut.result() for fut in futs]
