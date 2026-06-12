from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template

from lettucedetect.detectors.cache import CacheManager
from lettucedetect.detectors.llm_client import (
    HALLUCINATION_JSON_SCHEMA,
    HALLUCINATION_REASONING_JSON_SCHEMA,
    LLMClient,
    make_llm_client,
)
from lettucedetect.detectors.prompt_utils import LANG_TO_PASSAGE, Lang, PromptUtils

logger = logging.getLogger(__name__)

_RESPONSE_FORMAT = """**Return** a JSON object following *exactly* this schema
   (no extra keys, no markdown, no code-block fences):

   `{"hallucination_list": ["substring1", "substring2", …]}`

   If none are found, return `{"hallucination_list": []}`."""

_RESPONSE_FORMAT_REASONING = """**Return** a JSON object following *exactly* this schema
   (no extra keys, no markdown, no code-block fences):

   `{"hallucination_list": [{"text": "substring1", "confidence": 0.95, "reasoning": "why it is unsupported"}, …]}`

   - "text" must be an exact substring of the answer.
   - "confidence" is your confidence between 0 and 1 that the span is hallucinated.
   - "reasoning" is a brief explanation of why the span contradicts or is unsupported by the source.
   - Examples above may show only the hallucinated substrings; still return full objects.

   If none are found, return `{"hallucination_list": []}`."""


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
        :param client_kwargs: Extra keyword arguments forwarded to the provider client constructor.
        """
        if lang not in LANG_TO_PASSAGE:
            raise ValueError(f"Invalid language. Use one of: {', '.join(LANG_TO_PASSAGE.keys())}")

        self.model = model
        self.temperature = temperature
        self.lang = lang
        self.zero_shot = zero_shot
        self.include_reasoning = include_reasoning
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
            response_format_block=(
                _RESPONSE_FORMAT_REASONING if self.include_reasoning else _RESPONSE_FORMAT
            ),
        )

    @staticmethod
    def _to_spans(items: list[str | dict], answer: str) -> list[dict]:
        """Convert hallucinated items to a list of spans.

        :param items: List of hallucinated substrings, or objects with ``text``,
            ``confidence``, and ``reasoning`` keys when reasoning is enabled.
        :param answer: The answer string.
        :returns: List of spans.
        """
        spans = []
        for item in items:
            sub = item["text"] if isinstance(item, dict) else item
            if not sub:
                continue
            # Use regex for more reliable matching
            match = re.search(re.escape(sub), answer)
            if not match:
                continue
            span = {"start": match.start(), "end": match.end(), "text": sub}
            if isinstance(item, dict):
                span["confidence"] = item.get("confidence")
                span["reasoning"] = item.get("reasoning")
            spans.append(span)
        return spans

    def _predict(self, prompt: str, answer: str) -> list[dict]:
        """Single (prompt, answer) pair → hallucination spans.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :returns: List of spans.
        """
        # Build the full LLM prompt using the template
        llm_prompt = self._build_prompt(prompt, answer)

        # Use the full LLM prompt for cache key calculation
        cache_key = self.cache._hash(llm_prompt, self.model, str(self.temperature))

        cached = self.cache.get(cache_key)
        if cached is None:
            cached = self.client.complete(
                system="You are an expert in detecting hallucinations in LLM outputs.",
                user=llm_prompt,
                model=self.model,
                temperature=self.temperature,
                schema=(
                    HALLUCINATION_REASONING_JSON_SCHEMA
                    if self.include_reasoning
                    else HALLUCINATION_JSON_SCHEMA
                ),
            )
            if cached:
                self.cache.set(cache_key, cached)

        try:
            payload = json.loads(cached)
            return self._to_spans(payload["hallucination_list"], answer)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Error parsing LLM response: %s", e)
            logger.debug("Raw response: %s", cached)
            return []

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "spans",
    ) -> list:
        """Predict hallucination spans from the provided context, answer, and question.

        :param context: List of passages that were supplied to the LLM / user.
        :param answer: Model-generated answer to inspect.
        :param question: Original question (``None`` for summarisation).
        :param output_format: ``"spans"`` for character spans.
        :returns: List of spans.
        """
        if output_format not in ["tokens", "spans"]:
            raise ValueError(
                f"LLMDetector doesn't support '{output_format}' format. Use 'tokens' or 'spans'"
            )
        # Use PromptUtils to format the context and question
        full_prompt = PromptUtils.format_context(context, question, self.lang)
        return self._predict(full_prompt, answer)

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "spans") -> list:
        """Predict hallucination spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: ``"spans"`` for character spans.
        :returns: List of spans.
        """
        if output_format not in ["tokens", "spans"]:
            raise ValueError(
                f"LLMDetector doesn't support '{output_format}' format. Use 'tokens' or 'spans'"
            )
        return self._predict(prompt, answer)

    def predict_prompt_batch(
        self, prompts: list[str], answers: list[str], output_format: str = "spans"
    ) -> list:
        """Predict hallucination spans from the provided prompts and answers.

        :param prompts: List of prompt strings.
        :param answers: List of answer strings.
        :param output_format: ``"spans"`` for character spans.
        :returns: List of spans.
        """
        if output_format not in ["tokens", "spans"]:
            raise ValueError(
                f"LLMDetector doesn't support '{output_format}' format. Use 'tokens' or 'spans'"
            )

        with ThreadPoolExecutor(max_workers=30) as pool:
            futs = [pool.submit(self._predict, p, a) for p, a in zip(prompts, answers)]
            return [f.result() for f in futs]
