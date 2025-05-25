"""SentenceTransformer‑based hallucination detector."""

from __future__ import annotations

from typing import Literal

import nltk
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from lettucedetect.datasets.hallucination_dataset import HallucinationDataset
from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.prompt_utils import LANG_TO_PASSAGE, Lang, PromptUtils
from lettucedetect.models.sentece_model import SentenceModel

__all__ = ["SentenceTransformer"]


class SentenceTransformer(BaseDetector):
    """Detect hallucinations with a fine‑tuned sentence classifier."""

    def __init__(
        self,
        model_path: str,
        max_length: int = 4096,
        device=None,
        lang: Literal["en", "de", "fr", "es", "it", "pl"] = "en",
        threshold: float = 0.5,
        **kwargs,
    ):
        """Initialize the SentenceTransformer.
        :param model_path: The path to the model.
        :param max_length: The maximum length of the input sequence.
        :param device: The device to run the model on.
        :param lang: The language of the model.
        :param threshold: Confidence threshold for considering a span relevant (0.0-1.0)
        """

        self.lang = lang
        self.model = SentenceModel.from_pretrained(model_path, **kwargs)
        base_model = getattr(self.model.config, "model_name", "answerdotai/ModernBERT-base")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        self.model.to(self.device).eval()

    def _predict(self, context: str, answer: str, output_format: str):
        """Predict hallucination tokens or spans from the provided context and answer.

        :param context: The context string.
        :param answer: The answer string.
        """

        if output_format == "spans":
            sentences = nltk.sent_tokenize(answer)
            # Use the shared tokenization logic from HallucinationDataset
            (
                input_ids,
                attention_mask,
                offset_mapping,
                sentence_boundaries,
                sentence_offset_mappings,
            ) = HallucinationDataset.encode_context_and_sentences_with_offset(
                self.tokenizer, context, sentences, self.max_length
            )

            input_ids = input_ids.unsqueeze(0).to(self.device)
            attention_mask = attention_mask.unsqueeze(0).to(self.device)

            # Run model inference
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, [sentence_boundaries])

            # Extract hallucinated sentences
            hallucinated_sentences = []

            if len(outputs) > 0 and len(outputs[0]) > 0:
                sentence_preds = torch.nn.functional.softmax(outputs[0], dim=1)
                for i, pred in enumerate(sentence_preds):
                    if i < len(sentences) and pred[1] > self.threshold:
                        hallucinated_sentences.append(sentences[i])

            return hallucinated_sentences
        else:
            raise ValueError(
                "Invalid output_format. This model can only predict hallucination sentences. Use spans."
            )

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "spans") -> list:
        """Predict hallucination tokens or spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: "spans" for sentences,
        """
        return self._predict(prompt, answer, output_format)

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "spans",
    ) -> list:
        """Predict hallucination tokens or spans from the provided context, answer, and question.
        This is a useful interface when we don't want to predict a specific prompt, but rather we have a list of contexts, answers, and questions. Useful to interface with RAG systems.

        :param context: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.
        :param output_format: "spans" to return sentences.
        """
        formatted_prompt = PromptUtils.format_context(context, question, self.lang)
        return self._predict(formatted_prompt, answer, output_format)

    def predict_prompt(self, prompt, answer, output_format="tokens") -> list:
        """Predict hallucination sentences from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        return self._predict(prompt, answer, output_format)

    def predict_prompt_batch(self, prompts, answers, output_format="tokens") -> list:
        """Predict hallucination sentences from the provided prompts and answers.

        :param prompts: List of prompt strings.
        :param answers: List of answer strings.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        return [self._predict(p, a, output_format) for p, a in zip(prompts, answers)]
