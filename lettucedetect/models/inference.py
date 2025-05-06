import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from string import Template
from typing import Literal

import torch
from openai import OpenAI
from transformers import AutoModelForTokenClassification, AutoTokenizer

from lettucedetect.datasets.hallucination_dataset import (
    HallucinationDataset,
)

LANG_TO_PASSAGE = {
    "en": "passage",
    "de": "Passage",
    "fr": "passage",
    "es": "pasaje",
    "it": "brano",
    "pl": "fragment",
}


# ==== Base class for all detectors ====
class BaseDetector(ABC):
    @abstractmethod
    def predict(self, context: str, answer: str, output_format: str = "tokens") -> list:
        """Given a context and an answer, returns predictions.

        :param context: The context string.
        :param answer: The answer string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        pass


# ==== Transformer-based detector ====
class TransformerDetector(BaseDetector):
    def __init__(
        self,
        model_path: str,
        max_length: int = 4096,
        device=None,
        lang: Literal["en", "de", "fr", "es", "it", "pl"] = "en",
        **kwargs,
    ):
        """Initialize the TransformerDetector.

        :param model_path: The path to the model.
        :param max_length: The maximum length of the input sequence.
        :param device: The device to run the model on.
        :param lang: The language of the model.
        """
        if lang not in LANG_TO_PASSAGE:
            raise ValueError(f"Invalid language. Use one of: {', '.join(LANG_TO_PASSAGE.keys())}")

        self.lang = lang
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path, **kwargs)
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        prompt_path = Path(__file__).parent.parent / "prompts" / f"qa_prompt_{lang.lower()}.txt"
        self.prompt_qa = Template(prompt_path.read_text(encoding="utf-8"))
        prompt_path = (
            Path(__file__).parent.parent / "prompts" / f"summary_prompt_{lang.lower()}.txt"
        )
        self.prompt_summary = Template(prompt_path.read_text(encoding="utf-8"))

    def _form_prompt(self, context: list[str], question: str | None) -> str:
        """Form a prompt from the provided context and question. We use different prompts for summary and QA tasks.

        :param context: A list of context strings.
        :param question: The question string.
        :return: The formatted prompt.
        """
        context_str = "\n".join(
            [
                f"{LANG_TO_PASSAGE[self.lang]} {i + 1}: {passage}"
                for i, passage in enumerate(context)
            ]
        )
        if question is None:
            return self.prompt_summary.substitute(text=context_str)
        else:
            return self.prompt_qa.substitute(
                question=question, num_passages=len(context), context=context_str
            )

    def _predict(self, context: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination tokens or spans from the provided context and answer.

        :param context: The context string.
        :param answer: The answer string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        # Use the shared tokenization logic from RagTruthDataset
        encoding, labels, offsets, answer_start_token = (
            HallucinationDataset.prepare_tokenized_input(
                self.tokenizer, context, answer, self.max_length
            )
        )

        # Create a label tensor: mark tokens before answer as -100 (ignored) and answer tokens as 0.
        labels = torch.full_like(encoding.input_ids[0], -100, device=self.device)
        labels[answer_start_token:] = 0
        # Move encoding to the device
        encoding = {
            key: value.to(self.device)
            for key, value in encoding.items()
            if key in ["input_ids", "attention_mask", "labels"]
        }

        # Run model inference
        with torch.no_grad():
            outputs = self.model(**encoding)
        logits = outputs.logits
        token_preds = torch.argmax(logits, dim=-1)[0]
        probabilities = torch.softmax(logits, dim=-1)[0]

        # Mask out predictions for context tokens.
        token_preds = torch.where(labels == -100, labels, token_preds)

        if output_format == "tokens":
            # return token probabilities for each token (with the tokens as well, if not -100)
            token_probs = []
            input_ids = encoding["input_ids"][0]  # Get the input_ids tensor from the encoding dict
            for i, (token, pred, prob) in enumerate(zip(input_ids, token_preds, probabilities)):
                if not labels[i].item() == -100:
                    token_probs.append(
                        {
                            "token": self.tokenizer.decode([token]),
                            "pred": pred.item(),
                            "prob": prob[1].item(),  # Get probability for class 1 (hallucination)
                        }
                    )
            return token_probs
        elif output_format == "spans":
            # Compute the answer's character offset (the first token of the answer).
            if answer_start_token < offsets.size(0):
                answer_char_offset = offsets[answer_start_token][0].item()
            else:
                answer_char_offset = 0

            spans: list[dict] = []
            current_span: dict | None = None

            # Iterate over tokens in the answer region.
            for i in range(answer_start_token, token_preds.size(0)):
                # Skip tokens marked as ignored.
                if labels[i].item() == -100:
                    continue

                token_start, token_end = offsets[i].tolist()
                # Skip special tokens with zero length.
                if token_start == token_end:
                    continue

                # Adjust offsets relative to the answer text.
                rel_start = token_start - answer_char_offset
                rel_end = token_end - answer_char_offset

                is_hallucination = (
                    token_preds[i].item() == 1
                )  # assuming class 1 indicates hallucination.
                confidence = probabilities[i, 1].item() if is_hallucination else 0.0

                if is_hallucination:
                    if current_span is None:
                        current_span = {
                            "start": rel_start,
                            "end": rel_end,
                            "confidence": confidence,
                        }
                    else:
                        # Extend the current span.
                        current_span["end"] = rel_end
                        current_span["confidence"] = max(current_span["confidence"], confidence)
                else:
                    # If we were building a hallucination span, finalize it.
                    if current_span is not None:
                        # Extract the hallucinated text from the answer.
                        span_text = answer[current_span["start"] : current_span["end"]]
                        current_span["text"] = span_text
                        spans.append(current_span)
                        current_span = None

            # Append any span still in progress.
            if current_span is not None:
                span_text = answer[current_span["start"] : current_span["end"]]
                current_span["text"] = span_text
                spans.append(current_span)

            return spans
        else:
            raise ValueError("Invalid output_format. Use 'tokens' or 'spans'.")

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        return self._predict(prompt, answer, output_format)

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
    ) -> list:
        """Predict hallucination tokens or spans from the provided context, answer, and question.
        This is a useful interface when we don't want to predict a specific prompt, but rather we have a list of contexts, answers, and questions. Useful to interface with RAG systems.

        :param context: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        prompt = self._form_prompt(context, question)
        return self._predict(prompt, answer, output_format)


# ==== LLM-based detector ====
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


class LLMDetector(BaseDetector):
    """LLM-powered hallucination detector using function calling and a prompt template."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: int = 0,
        lang: Literal["en", "de", "fr", "es", "it", "pl"] = "en",
        zero_shot: bool = False,
        fewshot_path: str | None = None,
        prompt_path: str | None = None,
        cache_file: str | None = None,
    ):
        """Initialize the LLMDetector.

        :param model: OpenAI model.
        :param temperature: model temperature.
        :param lang: language of the examples.
        :param zero_shot: whether to use zero-shot prompting.
        :param fewshot_path: path to the fewshot examples.
        :param prompt_path: path to the prompt template.
        :param cache_file: path to the cache file.
        """
        self.model = model
        self.temperature = temperature

        if lang not in LANG_TO_PASSAGE:
            raise ValueError(f"Invalid language. Use one of: {', '.join(LANG_TO_PASSAGE.keys())}")

        self.lang = lang
        self.zero_shot = zero_shot
        if fewshot_path is None:
            print(
                f"No fewshot path provided, using default path: {Path(__file__).parent.parent / 'prompts' / f'examples_{lang.lower()}.json'}"
            )
            fewshot_path = (
                Path(__file__).parent.parent / "prompts" / f"examples_{lang.lower()}.json"
            )

            if not fewshot_path.exists():
                raise FileNotFoundError(f"Fewshot file not found at {fewshot_path}")
        else:
            fewshot_path = Path(fewshot_path)

        if prompt_path is None:
            print(
                f"No prompt path provided, using default path: {Path(__file__).parent.parent / 'prompts' / 'hallucination_detection.txt'}"
            )
            template_path = Path(__file__).parent.parent / "prompts" / "hallucination_detection.txt"
        else:
            template_path = Path(prompt_path)

        prompt_qa_path = Path(__file__).parent.parent / "prompts" / f"qa_prompt_{lang.lower()}.txt"
        prompt_summary_path = (
            Path(__file__).parent.parent / "prompts" / f"summary_prompt_{lang.lower()}.txt"
        )

        self.template = Template(template_path.read_text(encoding="utf-8"))
        self.prompt_qa = Template(prompt_qa_path.read_text(encoding="utf-8"))
        self.prompt_summary = Template(prompt_summary_path.read_text(encoding="utf-8"))

        self.fewshot = json.loads(fewshot_path.read_text(encoding="utf-8"))
        self.cache_path = cache_file

        if cache_file is None:
            self.cache_path = (
                Path(__file__).parent.parent / "cache" / f"cache_{self.model}_{self.lang}.json"
            )
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Read in cache
            if self.cache_path.exists():
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            else:
                self.cache = {}

            print(f"Cache file not provided, using default path: {self.cache_path}")
        else:
            self.cache_path = Path(cache_file)
            if not self.cache_path.exists():
                raise FileNotFoundError(f"Cache file not found at {self.cache_path}")
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _build_prompt(
        self,
        context: str,
        answer: str,
    ) -> str:
        """Fill the template with runtime values, inserting zero or many few‑shot examples.
        Uses `${placeholder}` tokens in the .txt file.
        """
        fewshot_block = ""
        if self.fewshot and not self.zero_shot:
            lines: list[str] = []
            for idx, ex in enumerate(self.fewshot, 1):
                lines.append(
                    f"""<example{idx}>
<source>{ex["source"]}</source>
<answer>{ex["answer"]}</answer>
<target>{{"hallucination_list": {json.dumps(ex["hallucination_list"], ensure_ascii=False)} }}</target>
</example{idx}>"""
                )
            fewshot_block = "\n".join(lines)

        filled = self.template.substitute(
            lang=self.lang,
            context=context,
            answer=answer,
            fewshot_block=fewshot_block,
        )
        return filled

    def _form_context(self, context: list[str], question: str | None) -> str:
        """Form a prompt from the provided context and question. We use different prompts for summary and QA tasks.
        :param context: A list of context strings.
        :param question: The question string.
        :return: The formatted prompt.
        """
        context_str = "\n".join(
            [f"passage {i + 1}: {passage}" for i, passage in enumerate(context)]
        )
        if question is None:
            return self.prompt_summary.substitute(text=context_str)
        else:
            return self.prompt_qa.substitute(
                question=question, num_passages=len(context), context=context_str
            )

    def _get_openai_client(self) -> OpenAI:
        """Get OpenAI client configured from environment variables.

        :return: Configured OpenAI client
        :raises ValueError: If API key is not set
        """
        api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
        api_base = os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"

        return OpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    def _hash(self, prompt: str) -> str:
        """Hash the prompt."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _call_openai(self, prompt: str) -> str:
        """Call the OpenAI API.

        :param prompt: The prompt to call the OpenAI API with.
        :return: The response from the OpenAI API.
        """
        client = self._get_openai_client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in detecting hallucinations in LLM outputs.",
                },
                {"role": "user", "content": prompt},
            ],
            tools=ANNOTATE_SCHEMA,
            tool_choice={"type": "function", "function": {"name": "annotate"}},
            temperature=self.temperature,
        )

        return resp.choices[0].message.tool_calls[0].function.arguments

    def _save_cache(self):
        """Save the cache to the cache file."""
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")

    def _to_spans(self, subs: list[str], answer: str) -> list[dict]:
        """Convert a list of substrings to a list of spans.

        :param subs: A list of substrings.
        :param answer: The answer string.
        :return: A list of spans.
        """
        spans = []
        for s in subs:
            m = re.search(re.escape(s), answer)
            if m:
                spans.append({"start": m.start(), "end": m.end(), "text": s})
        return spans

    def _predict(self, context: str, answer: str, output_format: str = "spans") -> list:
        """Prompts the ChatGPT model to predict hallucination spans from the provided context and answer.

        :param context: The context string.
        :param answer: The answer string.
        :param output_format: works only for "spans" and returns grouped spans.
        """
        if output_format == "spans":
            llm_prompt = self._build_prompt(context, answer)

            key = self._hash("||".join([llm_prompt, self.model, str(self.temperature)]))

            # Check if the response is cached
            cached_response = self.cache.get(key)
            if cached_response is None:
                cached_response = self._call_openai(llm_prompt)
                self.cache[key] = cached_response
                self._save_cache()

            payload = json.loads(cached_response)
            return self._to_spans(payload["hallucination_list"], answer)
        else:
            raise ValueError(
                "Invalid output_format. This model can only predict hallucination spans. Use spans."
            )

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: "spans" to return grouped spans.
        """
        return self._predict(prompt, answer, output_format)

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "spans",
    ) -> list:
        """Predict hallucination spans from the provided context, answer, and question.
        This is a useful interface when we don't want to predict a specific prompt, but rather we have a list of contexts, answers, and questions. Useful to interface with RAG systems.

        :param context: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.
        :param output_format: "spans" to return grouped spans.
        """
        prompt = self._form_context(context, question)
        return self._predict(prompt, answer, output_format=output_format)


class HallucinationDetector:
    def __init__(self, method: str = "transformer", **kwargs):
        """Facade for the hallucination detector.

        :param method: "transformer" for the model-based approach.
        :param kwargs: Additional keyword arguments passed to the underlying detector.
        """
        if method == "transformer":
            self.detector = TransformerDetector(**kwargs)
        elif method == "llm":
            self.detector = LLMDetector(**kwargs)
        else:
            raise ValueError("Unsupported method. Choose 'transformer' or 'llm'.")

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
    ) -> list:
        """Predict hallucination tokens or spans from the provided context, answer, and question.
        This is a useful interface when we don't want to predict a specific prompt, but rather we have a list of contexts, answers, and questions. Useful to interface with RAG systems.

        :param context: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.
        """
        return self.detector.predict(context, answer, question, output_format)

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        """
        return self.detector.predict_prompt(prompt, answer, output_format)
