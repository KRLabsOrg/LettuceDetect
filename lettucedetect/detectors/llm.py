from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lettucedetect.detectors._llm_shared import (
    load_verify_template,
    locate_items,
    resolve_taxonomy,
)
from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.cache import CacheManager
from lettucedetect.detectors.errors import UpstreamError
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
from lettucedetect.detectors.prompt_utils import LANG_TO_PASSAGE, Lang, PromptUtils
from lettucedetect.detectors.types import PromptInput

logger = logging.getLogger(__name__)


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


class LLMDetector(BaseDetector):
    """LLM-powered hallucination detector."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        lang: Lang = "en",
        zero_shot: bool = False,
        fewshot_path: str | None = None,
        prompt_path: str | None = None,
        cache_file: str | None = None,
        provider: str = "openai",
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
        :param cache_file: The path to the cache file.
        :param provider: Backend provider, ``"openai"`` or ``"bedrock"``. Ignored if ``client`` is given.
        :param client: An explicit :class:`LLMClient` instance; overrides ``provider``.
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
        :param client_kwargs: Extra keyword arguments forwarded to the provider client constructor.
        """
        if lang not in LANG_TO_PASSAGE:
            raise ValueError(f"Invalid language. Use one of: {', '.join(LANG_TO_PASSAGE.keys())}")

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
        if self.native and (zero_shot or fewshot_path or prompt_path):
            raise ValueError(
                "zero_shot, fewshot_path, and prompt_path have no effect on the native "
                "generative detectors (they use their frozen training prompt); pass "
                "native=False to prompt this model as a generic LLM judge instead."
            )
        self.categories, self.subcategories = resolve_taxonomy(include_taxonomy)
        self.schema = build_hallucination_schema(
            include_reasoning=include_reasoning,
            categories=list(self.categories) if self.categories else None,
            subcategories=list(self.subcategories) if self.subcategories else None,
        )
        self.client = client or make_llm_client(provider, **client_kwargs)

        if fewshot_path is not None and not Path(fewshot_path).exists():
            logger.warning("Few-shot examples file not found at %s", fewshot_path)
            # A missing file means a prompt without examples, which is the zero-shot prompt.
            fewshot_path, zero_shot = None, True
        if prompt_path is not None and not Path(prompt_path).exists():
            raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
        self.verify_template = load_verify_template()

        # Set up cache
        if cache_file is None:
            cache_file = (
                Path(__file__).parent.parent
                / "cache"
                / f"cache_{model.replace(':', '_')}_{lang}.json"
            )
            logger.info("Using default cache file: %s", cache_file)
        else:
            logger.info("Using provided cache file: %s", cache_file)

        self.cache = CacheManager(cache_file)

        # The native path runs through GenerativeDetector, the judge path through LLMJudgeDetector.
        self._generative = (
            GenerativeDetector(
                self.client,
                model,
                config=GenerativeConfig(
                    temperature=temperature,
                    lang=lang,
                    include_reasoning=include_reasoning,
                    include_taxonomy=include_taxonomy,
                    verify=verify,
                ),
                cache=self.cache,
            )
            if self.native
            else None
        )
        self._judge = (
            None
            if self.native
            else LLMJudgeDetector(
                self.client,
                model,
                config=LLMJudgeConfig(
                    temperature=temperature,
                    lang=lang,
                    zero_shot=zero_shot,
                    fewshot_path=fewshot_path,
                    prompt_path=prompt_path,
                    include_reasoning=include_reasoning,
                    include_taxonomy=include_taxonomy,
                    min_score=min_confidence,
                    verify=verify,
                ),
                cache=self.cache,
            )
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
        """
        for_tokens = output_format == "tokens"
        detector = self._generative if self.native else self._judge
        try:
            detection = detector.detect(
                PromptInput(prompt=prompt, answer=answer),
                min_score=0.0 if for_tokens else min_confidence,
            )
        except UpstreamError as exc:
            if exc.code != "unparseable_response":
                raise
            # These methods keep returning no spans for a reply that can't be parsed.
            return []
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
        """
        if output_format not in ["tokens", "spans"]:
            raise ValueError(
                f"LLMDetector doesn't support '{output_format}' format. Use 'tokens' or 'spans'"
            )
        self._validate_min_confidence(min_confidence)
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
        """
        if output_format not in ["tokens", "spans"]:
            raise ValueError(
                f"LLMDetector doesn't support '{output_format}' format. Use 'tokens' or 'spans'"
            )
        self._validate_min_confidence(min_confidence)
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
        :raises ValueError: If ``len(prompts) != len(answers)``.
        """
        if len(prompts) != len(answers):
            raise ValueError("Number of prompts must match number of answers")
        if output_format not in ["tokens", "spans"]:
            raise ValueError(
                f"LLMDetector doesn't support '{output_format}' format. Use 'tokens' or 'spans'"
            )
        self._validate_min_confidence(min_confidence)

        with ThreadPoolExecutor(max_workers=30) as pool:
            futs = [
                pool.submit(self._predict, p, a, output_format, min_confidence)
                for p, a in zip(prompts, answers)
            ]
            return [fut.result() for fut in futs]
