from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template

from openai import OpenAI

from lettucedetect.detectors.cache import CacheManager
from lettucedetect.detectors.prompt_utils import LANG_TO_PASSAGE, Lang, PromptUtils

ANNOTATE_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "annotate",
            "description": "Return hallucinated substrings from the answer relative to the source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hallucination_list": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["hallucination_list"],
            },
        },
    }
]


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
    ):
        """Initialize the LLMDetector.

        :param model: The model to use for hallucination detection.
        :param temperature: The temperature to use for hallucination detection.
        :param lang: The language to use for hallucination detection.
        :param zero_shot: Whether to use zero-shot hallucination detection.
        :param fewshot_path: The path to the few-shot examples.
        :param prompt_path: The path to the prompt.
        :param cache_file: The path to the cache file.
        """
        if lang not in LANG_TO_PASSAGE:
            raise ValueError(f"Invalid language. Use one of: {', '.join(LANG_TO_PASSAGE.keys())}")

        self.model = model
        self.temperature = temperature
        self.lang = lang
        self.zero_shot = zero_shot

        # Load few-shot examples
        if fewshot_path is None:
            fewshot_path = (
                Path(__file__).parent.parent / "prompts" / f"examples_{lang.lower()}.json"
            )
        path = Path(fewshot_path)
        if not path.exists():
            print(f"Warning: Few-shot examples file not found at {path}")
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
            print(f"Using default cache file: {cache_file}")
        else:
            print(f"Using provided cache file: {cache_file}")

        self.cache = CacheManager(cache_file)

    def _openai(self) -> OpenAI:
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
            base_url=os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1",
        )

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
        )

    @staticmethod
    def _to_spans(substrs: list[str], answer: str) -> list[dict]:
        """Convert a list of substrings to a list of spans.

        :param substrs: List of substrings.
        :param answer: The answer string.
        :returns: List of spans.
        """
        spans = []
        for sub in substrs:
            if not sub:
                continue
            # Use regex for more reliable matching
            match = re.search(re.escape(sub), answer)
            if match:
                spans.append({"start": match.start(), "end": match.end(), "text": sub})
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
            resp = self._openai().chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in detecting hallucinations in LLM outputs.",
                    },
                    # Use the full LLM prompt here, not the raw context
                    {"role": "user", "content": llm_prompt},
                ],
                tools=ANNOTATE_SCHEMA,
                tool_choice={"type": "function", "function": {"name": "annotate"}},
                temperature=self.temperature,
            )
            cached = resp.choices[0].message.tool_calls[0].function.arguments
            self.cache.set(cache_key, cached)

        try:
            payload = json.loads(cached)
            return self._to_spans(payload["hallucination_list"], answer)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {cached}")
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
        :param answer: Model‑generated answer to inspect.
        :param question: Original question (``None`` for summarisation).
        :param output_format: ``"spans"`` for character spans.
        :returns: List of spans.
        """
        if output_format != "spans":
            raise ValueError("LLMDetector only supports 'spans' output_format.")
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
        if output_format != "spans":
            raise ValueError("LLMDetector only supports 'spans' output_format.")
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
        if output_format != "spans":
            raise ValueError("LLMDetector only supports 'spans' output_format.")

        with ThreadPoolExecutor(max_workers=30) as pool:
            futs = [pool.submit(self._predict, p, a) for p, a in zip(prompts, answers)]
            return [f.result() for f in futs]
