from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template

from lettucedetect.datasets.taxonomy import CATEGORY_DEFINITIONS, SUBCATEGORY_DEFINITIONS
from lettucedetect.detectors.cache import CacheManager
from lettucedetect.detectors.llm_client import (
    LLMClient,
    build_generative_schema,
    build_hallucination_schema,
    build_verification_schema,
    make_llm_client,
)
from lettucedetect.detectors.prompt_utils import LANG_TO_PASSAGE, Lang, PromptUtils
from lettucedetect.prompts import generative as gen

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


_RESPONSE_FORMAT = """**Return** a JSON object following *exactly* this schema
   (no extra keys, no markdown, no code-block fences):

   `{"hallucination_list": ["substring1", "substring2", …]}`

   Each substring must be copied **verbatim** from the answer — an exact,
   character-for-character match. If none are found, return `{"hallucination_list": []}`."""

_VERIFY_SYSTEM = (
    "You are a strict fact-checker confirming whether flagged spans are truly unsupported."
)


class LLMDetector:
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
        if isinstance(include_taxonomy, bool):
            self.categories = dict(CATEGORY_DEFINITIONS) if include_taxonomy else None
            # The unified taxonomy is category + subcategory; enabling it types both.
            self.subcategories = dict(SUBCATEGORY_DEFINITIONS) if include_taxonomy else None
        elif isinstance(include_taxonomy, dict):
            self.categories = dict(include_taxonomy) or None
            self.subcategories = None  # custom category set: no defined subcategories
        else:
            self.categories = {
                name: CATEGORY_DEFINITIONS.get(name) for name in include_taxonomy
            } or None
            self.subcategories = None
        self.schema = build_hallucination_schema(
            include_reasoning=include_reasoning,
            categories=list(self.categories) if self.categories else None,
            subcategories=list(self.subcategories) if self.subcategories else None,
        )
        self.client = client or make_llm_client(provider, **client_kwargs)

        # Load few-shot examples
        if fewshot_path is None:
            fewshot_path = (
                Path(__file__).parent.parent / "prompts" / f"examples_{lang.lower()}.json"
            )
        path = Path(fewshot_path)
        if not path.exists():
            logger.warning("Few-shot examples file not found at %s", path)
        self.fewshot = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

        # Load hallucination detection template
        if prompt_path is None:
            prompt_path = Path(__file__).parent.parent / "prompts" / "hallucination_detection.txt"
        template_path = Path(prompt_path)
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found at {template_path}")
        self.template = Template(template_path.read_text(encoding="utf-8"))

        # Load verification template
        verify_template_path = (
            Path(__file__).parent.parent / "prompts" / "hallucination_verification.txt"
        )
        if not verify_template_path.exists():
            raise FileNotFoundError(f"Verification template not found at {verify_template_path}")
        self.verify_template = Template(verify_template_path.read_text(encoding="utf-8"))

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

    def _fewshot_block(self) -> str:
        if self.zero_shot or not self.fewshot:
            return ""
        lines = []
        for i, ex in enumerate(self.fewshot, 1):
            lines.append(
                f"""<example{i}>
<source>{ex["source"]}</source>
<answer>{ex["answer"]}</answer>
<target>{{"hallucination_list": {json.dumps(ex["hallucination_list"], ensure_ascii=False)} }}</target>
</example{i}>"""
            )
        return "\n".join(lines)

    def _response_format_block(self) -> str:
        """Compose the response-format instructions for the enabled options.

        :return: The response-format block for the prompt template.
        """
        if not self.include_reasoning and not self.categories:
            return _RESPONSE_FORMAT

        example: dict = {"text": "substring1"}
        notes = ['- "text" must be an exact substring of the answer.']
        if self.include_reasoning:
            example["reasoning"] = "compare the span against the source before judging it"
            notes.append(
                '- "reasoning" comes first: weigh the span against the source before scoring or judging it.'
            )
        if self.categories:
            example["category"] = next(iter(self.categories))
            category_lines = "\n".join(
                f'     - "{name}": {desc}' if desc else f'     - "{name}"'
                for name, desc in self.categories.items()
            )
            notes.append('- "category" classifies the hallucination as one of:\n' + category_lines)
        if self.subcategories:
            example["subcategory"] = next(iter(self.subcategories))
            subcategory_lines = "\n".join(
                f'     - "{name}": {desc}' if desc else f'     - "{name}"'
                for name, desc in self.subcategories.items()
            )
            notes.append(
                '- "subcategory" further classifies the span as one of:\n' + subcategory_lines
            )
        if self.include_reasoning:
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
        if self.include_reasoning:
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

    def _build_prompt(self, context: str, answer: str) -> str:
        """Fill the template with runtime values, inserting few-shot examples.

        :param context: The context string.
        :param answer: The answer string.
        :return: The filled template.
        """
        language_name = PromptUtils.get_full_language_name(self.lang)

        return self.template.substitute(
            lang=language_name,
            context=context,
            answer=answer,
            fewshot_block=self._fewshot_block(),
            response_format_block=self._response_format_block(),
        )

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
        spans = []
        for item in items:
            if not isinstance(item, (str, dict)):
                logger.warning("Skipping malformed hallucination item: %r", item)
                continue
            sub = item.get("text") if isinstance(item, dict) else item
            if not sub:
                continue
            if isinstance(item, dict):
                if item.get("is_hallucination") is False:
                    continue
                confidence = item.get("confidence")
                if confidence is not None and confidence < min_confidence:
                    continue
            # Use regex for more reliable matching
            match = re.search(re.escape(sub), answer)
            if not match:
                logger.debug("Dropping span not found verbatim in answer: %r", sub)
                continue
            span = {"start": match.start(), "end": match.end(), "text": sub}
            if isinstance(item, dict):
                for key in ("confidence", "reasoning", "category", "subcategory"):
                    if key in item:
                        span[key] = item[key]
            spans.append(span)
        return spans

    @staticmethod
    def _parse_response(raw: str) -> list:
        """Parse and validate the raw LLM response into a hallucination list.

        Recovers the case where the model returns ``hallucination_list`` as a
        JSON-encoded string instead of an array.

        :param raw: Raw JSON string returned by the client.
        :returns: The validated hallucination list.
        :raises ValueError: If ``hallucination_list`` is not a list.
        """
        payload = json.loads(raw)
        items = payload["hallucination_list"]
        if isinstance(items, str):
            items = json.loads(items)
        if not isinstance(items, list):
            raise ValueError(f"hallucination_list is {type(items).__name__}, expected list")
        return items

    def _predict(self, prompt: str, answer: str) -> list[dict]:
        """Single (prompt, answer) pair → hallucination spans.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :returns: List of spans.
        """
        if self.native:
            return self._predict_native(prompt, answer)
        # Build the full LLM prompt using the template
        llm_prompt = self._build_prompt(prompt, answer)

        # Use the full LLM prompt for cache key calculation
        cache_key = self.cache._hash(llm_prompt, self.model, str(self.temperature))

        cached = self.cache.get(cache_key)
        from_cache = cached is not None
        if not from_cache:
            cached = self.client.complete(
                system="You are an expert in detecting hallucinations in LLM outputs.",
                user=llm_prompt,
                model=self.model,
                temperature=self.temperature,
                schema=self.schema,
            )

        try:
            items = self._parse_response(cached)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error("Error parsing LLM response: %s", e)
            logger.debug("Raw response: %s", cached)
            if from_cache:
                self.cache.delete(cache_key)
            return []

        if not from_cache:
            self.cache.set(cache_key, cached)
        spans = self._to_spans(items, answer, self.min_confidence)
        if self.verify and spans:
            spans = self._verify(prompt, answer, spans)
        return spans

    def _predict_native(self, context: str, answer: str) -> list[dict]:
        """Native path: serve a fine-tuned generative span detector (lettucedect-v2-*).

        Sends the frozen training prompt and parses the model's ``hallucinated_spans``
        JSON (typed with category/subcategory, plus explanation when ``include_reasoning``),
        recovering character offsets by matching each span verbatim in the answer.

        :param context: The formatted source/context block (already includes the request).
        :param answer: The answer string.
        :returns: List of typed spans.
        """
        system = gen.SYSTEM_EXPL if self.include_reasoning else gen.SYSTEM_BASE
        user = gen.build_user_message(context, answer)
        schema = build_generative_schema(explain=self.include_reasoning)

        cache_key = self.cache._hash(system + user, self.model, str(self.temperature), "native")
        cached = self.cache.get(cache_key)
        if cached is None:
            cached = self.client.complete(
                system=system,
                user=user,
                model=self.model,
                temperature=self.temperature,
                schema=schema,
            )
            self.cache.set(cache_key, cached)
        spans = gen.spans_to_offsets(answer, gen.parse_spans(cached or ""))
        for span in spans:  # unify the rationale key with the judge path
            if "explanation" in span:
                span["reasoning"] = span.pop("explanation")
        if self.verify and spans:
            spans = self._verify(context, answer, spans)
        return spans

    def _verify(self, context: str, answer: str, spans: list[dict]) -> list[dict]:
        """Re-judge each flagged span against the source and drop the unconfirmed ones.

        Runs one additional, deliberately strict LLM call. Spans the verifier marks
        ``is_hallucination`` false are removed; spans it does not return are kept, so a
        malformed or partial verdict never silently discards confirmed detections.

        :param context: The formatted source/context block.
        :param answer: The answer string.
        :param spans: Spans produced by the first pass.
        :returns: The subset of spans the verifier confirms.
        """
        candidates = "\n".join(f"- {span['text']}" for span in spans)
        user = self.verify_template.substitute(
            lang=PromptUtils.get_full_language_name(self.lang),
            context=context,
            answer=answer,
            candidates=candidates,
        )
        cache_key = self.cache._hash(user, self.model, str(self.temperature), "verify")
        cached = self.cache.get(cache_key)
        from_cache = cached is not None
        if not from_cache:
            cached = self.client.complete(
                system=_VERIFY_SYSTEM,
                user=user,
                model=self.model,
                temperature=self.temperature,
                schema=build_verification_schema(),
            )
        try:
            verdicts = json.loads(cached)["verifications"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error("Error parsing verification response: %s", e)
            if from_cache:
                self.cache.delete(cache_key)
            return spans
        if not from_cache:
            self.cache.set(cache_key, cached)
        rejected = {
            v["text"]
            for v in verdicts
            if isinstance(v, dict) and v.get("is_hallucination") is False and "text" in v
        }
        return [span for span in spans if span["text"] not in rejected]

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
        :param output_format: ``"spans"`` for character spans.
        :param min_confidence: Drop spans whose ``confidence`` is below this threshold
            (in ``[0, 1]``). Applied on top of the constructor-level ``min_confidence``;
            ``0.0`` keeps every span.
        :returns: List of spans.
        """
        if output_format not in ["tokens", "spans"]:
            raise ValueError(
                f"LLMDetector doesn't support '{output_format}' format. Use 'tokens' or 'spans'"
            )
        self._validate_min_confidence(min_confidence)
        # Use PromptUtils to format the context and question
        full_prompt = PromptUtils.format_context(context, question, self.lang)
        spans = self._predict(full_prompt, answer)
        return self._filter_spans_by_confidence(spans, output_format, min_confidence)

    def predict_prompt(
        self, prompt: str, answer: str, output_format: str = "spans", min_confidence: float = 0.0
    ) -> list:
        """Predict hallucination spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: ``"spans"`` for character spans.
        :param min_confidence: Drop spans below this confidence threshold (``[0, 1]``); applied
            on top of the constructor-level ``min_confidence``.
        :returns: List of spans.
        """
        if output_format not in ["tokens", "spans"]:
            raise ValueError(
                f"LLMDetector doesn't support '{output_format}' format. Use 'tokens' or 'spans'"
            )
        self._validate_min_confidence(min_confidence)
        spans = self._predict(prompt, answer)
        return self._filter_spans_by_confidence(spans, output_format, min_confidence)

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
        :param output_format: ``"spans"`` for character spans.
        :param min_confidence: Drop spans below this confidence threshold (``[0, 1]``); applied
            on top of the constructor-level ``min_confidence``.
        :returns: List of spans.
        """
        if output_format not in ["tokens", "spans"]:
            raise ValueError(
                f"LLMDetector doesn't support '{output_format}' format. Use 'tokens' or 'spans'"
            )
        self._validate_min_confidence(min_confidence)

        with ThreadPoolExecutor(max_workers=30) as pool:
            futs = [pool.submit(self._predict, p, a) for p, a in zip(prompts, answers)]
            return [
                self._filter_spans_by_confidence(f.result(), output_format, min_confidence)
                for f in futs
            ]
