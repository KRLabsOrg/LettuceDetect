"""Transformer‑based hallucination detector."""

from __future__ import annotations

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from lettucedetect.datasets.hallucination_dataset import HallucinationDataset
from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.prompt_utils import LANG_TO_PASSAGE, Lang, PromptUtils

__all__ = ["TransformerDetector"]


class TransformerDetector(BaseDetector):
    """Detect hallucinations with a fine‑tuned token classifier."""

    def __init__(
        self, model_path: str, max_length: int = 4096, device=None, lang: Lang = "en", **tok_kwargs
    ):
        """Initialize the transformer detector.

        :param model_path: Path to the pre-trained model.
        :param max_length: Maximum length of the input sequence.
        :param device: Device to use for inference.
        :param lang: Language of the model.
        :param tok_kwargs: Additional keyword arguments for the tokenizer.
        """
        if lang not in LANG_TO_PASSAGE:
            raise ValueError(f"Invalid language. Choose from {', '.join(LANG_TO_PASSAGE)}")
        self.lang, self.max_length = lang, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path, **tok_kwargs)
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device).eval()

    def _predict(self, prompt: str, answer: str, output_format: str) -> list:
        """Predict hallucination tokens or spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        # Use the shared tokenization logic from HallucinationDataset
        encoding, _, offsets, answer_start_token = HallucinationDataset.prepare_tokenized_input(
            self.tokenizer, prompt, answer, self.max_length
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

    def predict(self, context, answer, question=None, output_format="tokens") -> list:
        """Predict hallucination tokens or spans from the provided context, answer, and question.

        :param context: List of passages that were supplied to the LLM / user.
        :param answer: Model‑generated answer to inspect.
        :param question: Original question (``None`` for summarisation).
        :param output_format: ``"tokens"`` for token‑level dicts, ``"spans"`` for character spans.
        :returns: List of predictions in requested format.
        """
        formatted_prompt = PromptUtils.format_context(context, question, self.lang)
        return self._predict(formatted_prompt, answer, output_format)

    def predict_prompt(self, prompt, answer, output_format="tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompt and answer.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        return self._predict(prompt, answer, output_format)

    def predict_prompt_batch(self, prompts, answers, output_format="tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompts and answers.

        :param prompts: List of prompt strings.
        :param answers: List of answer strings.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        """
        return [self._predict(p, a, output_format) for p, a in zip(prompts, answers)]
