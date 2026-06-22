"""Transformer-based hallucination detector."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from lettucedetect.datasets.hallucination_dataset import HallucinationDataset
from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.prompt_utils import LANG_TO_PASSAGE, Lang, PromptUtils

logger = logging.getLogger(__name__)

__all__ = ["TransformerDetector"]


class TransformerDetector(BaseDetector):
    """Detect hallucinations with a fine-tuned token classifier.

    When the combined context + answer exceeds ``max_length`` tokens, the
    context is automatically split into chunks.  Each chunk is scored
    independently and the per-token hallucination probability is aggregated
    across chunks with ``max()``.
    """

    def __init__(
        self,
        model_path: str,
        max_length: int = 4096,
        device: torch.device | str | None = None,
        lang: Lang = "en",
        taxonomy_head: str | None = None,
        **tok_kwargs: object,
    ) -> None:
        """Initialize the transformer detector.

        :param model_path: Path to the pre-trained model.
        :param max_length: Maximum length of the input sequence.
        :param device: Device to use for inference.
        :param lang: Language of the model.
        :param taxonomy_head: Optional path/HF id of a label-conditioned typing head. When set,
            ``"spans"`` predictions are typed with a ``category``/``subcategory`` (cascade mode).
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
        self.typer = None
        if taxonomy_head is not None:
            from lettucedetect.detectors.taxonomy_head import TaxonomyTyper

            self.typer = TaxonomyTyper(taxonomy_head, device=self.device)

    # ------------------------------------------------------------------
    # Chunking helpers
    # ------------------------------------------------------------------

    def _group_passages_into_chunks(
        self,
        context: list[str],
        question: str | None,
        answer: str,
    ) -> list[list[str]]:
        """Group passages into chunks so each chunk, wrapped in the full instruction template, fits in ``max_length``.

        Preserves the instruction template (question, "Bear in mind...", etc.)
        in every chunk by working at the passage level instead of slicing raw
        tokens.

        :param context: List of passage strings.
        :param question: Original question (``None`` for summarisation).
        :param answer: The answer string (token budget is reserved).
        :returns: List of passage groups.  Each group will be formatted into a
            complete prompt via ``PromptUtils.format_context``.
        """
        answer_tokens = self.tokenizer(answer, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].shape[1]
        # Total prompt-token budget: max_length minus answer tokens minus 3 special tokens
        total_budget = self.max_length - answer_tokens - 3

        # Fast path: check whether all passages fit in one prompt.
        full_prompt = PromptUtils.format_context(context, question, self.lang)
        full_prompt_tokens = self.tokenizer(
            full_prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].shape[1]

        if full_prompt_tokens <= total_budget:
            return [context]

        # Measure instruction overhead (everything except the passage content).
        minimal_prompt = PromptUtils.format_context([""], question, self.lang)
        instruction_overhead = self.tokenizer(
            minimal_prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].shape[1]

        passage_budget = total_budget - instruction_overhead
        if passage_budget <= 0:
            # Instructions + answer alone exceed max_length; single group, will truncate.
            return [context]

        # Tokenize each formatted passage line to get its token count.
        p_word = LANG_TO_PASSAGE[self.lang]
        passage_token_counts: list[int] = []
        for i, passage in enumerate(context):
            line = f"{p_word} {i + 1}: {passage}"
            tok_count = self.tokenizer(line, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].shape[1]
            passage_token_counts.append(tok_count)

        # Greedily group passages into buckets that fit within the budget.
        groups: list[list[str]] = []
        current_group: list[str] = []
        current_tokens = 0

        for passage, tok_count in zip(context, passage_token_counts):
            # +1 for the newline separator between passages
            effective = tok_count + (1 if current_group else 0)
            if current_tokens + effective > passage_budget and current_group:
                groups.append(current_group)
                current_group = []
                current_tokens = 0
                effective = tok_count
            current_group.append(passage)
            current_tokens += effective

        if current_group:
            groups.append(current_group)

        return groups if groups else [context]

    # ------------------------------------------------------------------
    # Single-chunk prediction (the original _predict logic)
    # ------------------------------------------------------------------

    def _predict_single(self, prompt: str, answer: str, output_format: str) -> list:
        """Run prediction on a single (prompt, answer) pair that fits in ``max_length``.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: ``"tokens"`` or ``"spans"``.
        """
        encoding, _, offsets, answer_start_token = HallucinationDataset.prepare_tokenized_input(
            self.tokenizer, prompt, answer, self.max_length
        )

        labels = torch.full_like(encoding.input_ids[0], -100, device=self.device)
        labels[answer_start_token:] = 0

        encoding = {
            key: value.to(self.device)
            for key, value in encoding.items()
            if key in ["input_ids", "attention_mask", "labels"]
        }

        with torch.no_grad():
            outputs = self.model(**encoding)
        logits = outputs.logits
        token_preds = torch.argmax(logits, dim=-1)[0]
        probabilities = torch.softmax(logits, dim=-1)[0]

        token_preds = torch.where(labels == -100, labels, token_preds)

        if output_format == "tokens":
            token_probs: list[dict] = []
            input_ids = encoding["input_ids"][0]
            for i, (token, pred, prob) in enumerate(zip(input_ids, token_preds, probabilities)):
                if labels[i].item() != -100:
                    token_probs.append(
                        {
                            "token": self.tokenizer.decode([token]),
                            "pred": pred.item(),
                            "prob": prob[1].item(),
                        }
                    )
            return token_probs

        # output_format == "spans"
        if answer_start_token < offsets.size(0):
            answer_char_offset = offsets[answer_start_token][0].item()
        else:
            answer_char_offset = 0

        spans: list[dict] = []
        current_span: dict | None = None

        for i in range(answer_start_token, token_preds.size(0)):
            if labels[i].item() == -100:
                continue

            token_start, token_end = offsets[i].tolist()
            if token_start == token_end:
                continue

            rel_start = token_start - answer_char_offset
            rel_end = token_end - answer_char_offset

            is_hallucination = token_preds[i].item() == 1
            confidence = probabilities[i, 1].item() if is_hallucination else 0.0

            if is_hallucination:
                if current_span is None:
                    current_span = {
                        "start": rel_start,
                        "end": rel_end,
                        "confidence": confidence,
                    }
                else:
                    current_span["end"] = rel_end
                    current_span["confidence"] = max(current_span["confidence"], confidence)
            else:
                if current_span is not None:
                    current_span["text"] = answer[current_span["start"] : current_span["end"]]
                    spans.append(current_span)
                    current_span = None

        if current_span is not None:
            current_span["text"] = answer[current_span["start"] : current_span["end"]]
            spans.append(current_span)

        return spans

    # ------------------------------------------------------------------
    # Multi-chunk prediction with max() aggregation
    # ------------------------------------------------------------------

    def _predict_chunked(self, chunk_prompts: list[str], answer: str, output_format: str) -> list:
        """Run prediction over multiple context chunks and aggregate with ``max()``.

        For each answer token, the hallucination probability is the maximum
        across all chunks.  This is conservative: a token is only considered
        supported if *every* chunk considers it supported.

        :param chunk_prompts: Context chunk strings.
        :param answer: The answer string (same for all chunks).
        :param output_format: ``"tokens"`` or ``"spans"``.
        """
        all_token_results: list[list[dict]] = []
        for chunk in chunk_prompts:
            tokens = self._predict_single(chunk, answer, output_format="tokens")
            all_token_results.append(tokens)

        n_tokens = len(all_token_results[0])

        # Aggregate: max hallucination probability across chunks.
        aggregated: list[dict] = []
        for tok_idx in range(n_tokens):
            max_prob = max(
                chunk_result[tok_idx]["prob"]
                for chunk_result in all_token_results
                if tok_idx < len(chunk_result)
            )
            aggregated.append(
                {
                    "token": all_token_results[0][tok_idx]["token"],
                    "pred": 1 if max_prob >= 0.5 else 0,
                    "prob": max_prob,
                }
            )

        if output_format == "tokens":
            return aggregated

        # For "spans", build character spans from the aggregated token predictions.
        # Use offset mapping from the first chunk (answer offsets are identical
        # across chunks because BERT tokenizers process segments independently).
        return self._build_spans_from_tokens(aggregated, chunk_prompts[0], answer)

    def _build_spans_from_tokens(
        self, token_results: list[dict], prompt: str, answer: str
    ) -> list[dict]:
        """Convert aggregated token predictions into character-level spans.

        Uses the tokenizer offset mapping from encoding *(prompt, answer)* to
        get precise character positions within the answer.

        :param token_results: Per-answer-token dicts with ``token``, ``pred``, ``prob``.
        :param prompt: A context prompt string (used to get answer offset mapping).
        :param answer: The original answer string.
        :returns: List of span dicts with ``start``, ``end``, ``confidence``, ``text``.
        """
        _, _, offsets, answer_start_token = HallucinationDataset.prepare_tokenized_input(
            self.tokenizer, prompt, answer, self.max_length
        )

        if answer_start_token < offsets.size(0):
            answer_char_offset = offsets[answer_start_token][0].item()
        else:
            answer_char_offset = 0

        # Collect answer-region offsets (skip special tokens with zero length).
        answer_offsets: list[tuple[int, int]] = []
        for i in range(answer_start_token, offsets.size(0)):
            s, e = offsets[i].tolist()
            if s == e:
                continue
            answer_offsets.append((s - answer_char_offset, e - answer_char_offset))

        spans: list[dict] = []
        current_span: dict | None = None

        for tok_idx, tok in enumerate(token_results):
            if tok_idx >= len(answer_offsets):
                break
            rel_start, rel_end = answer_offsets[tok_idx]
            is_hall = tok["pred"] == 1
            confidence = tok["prob"] if is_hall else 0.0

            if is_hall:
                if current_span is None:
                    current_span = {
                        "start": rel_start,
                        "end": rel_end,
                        "confidence": confidence,
                    }
                else:
                    current_span["end"] = rel_end
                    current_span["confidence"] = max(current_span["confidence"], confidence)
            else:
                if current_span is not None:
                    current_span["text"] = answer[current_span["start"] : current_span["end"]]
                    spans.append(current_span)
                    current_span = None

        if current_span is not None:
            current_span["text"] = answer[current_span["start"] : current_span["end"]]
            spans.append(current_span)

        return spans

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
    ) -> list:
        """Predict hallucination tokens or spans from the provided context, answer, and question.

        :param context: List of passages that were supplied to the LLM / user.
        :param answer: Model-generated answer to inspect.
        :param question: Original question (``None`` for summarisation).
        :param output_format: ``"tokens"`` for token-level dicts, ``"spans"`` for character spans.
        :returns: List of predictions in requested format.
        """
        if output_format not in ("tokens", "spans"):
            raise ValueError(
                f"TransformerDetector doesn't support '{output_format}' format. "
                "Use 'tokens' or 'spans'"
            )

        groups = self._group_passages_into_chunks(context, question, answer)

        if len(groups) == 1:
            prompt = PromptUtils.format_context(groups[0], question, self.lang)
            result = self._predict_single(prompt, answer, output_format)
        else:
            chunk_prompts = [
                PromptUtils.format_context(group, question, self.lang) for group in groups
            ]
            result = self._predict_chunked(chunk_prompts, answer, output_format)

        if output_format == "spans" and self.typer is not None:
            result = self.typer.type_spans(answer, "\n".join(context), result)
        return result

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucination tokens or spans from the provided prompt and answer.

        Note: unlike :meth:`predict`, this method does **not** chunk the prompt
        automatically.  If the prompt + answer exceed ``max_length``, the prompt
        will be truncated.  Use :meth:`predict` with structured passages for
        automatic chunking.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :returns: List of predictions in requested format.
        """
        if output_format not in ("tokens", "spans"):
            raise ValueError(
                f"TransformerDetector doesn't support '{output_format}' format. "
                "Use 'tokens' or 'spans'"
            )
        # Warn if the input will be truncated.
        total_tokens = self.tokenizer(prompt, answer, add_special_tokens=True, return_tensors="pt")[
            "input_ids"
        ].shape[1]
        if total_tokens > self.max_length:
            logger.warning(
                "predict_prompt: input (%d tokens) exceeds max_length (%d). "
                "The prompt will be truncated. Use predict() with structured "
                "passages for automatic chunking.",
                total_tokens,
                self.max_length,
            )
        spans = self._predict_single(prompt, answer, output_format)
        if output_format == "spans" and self.typer is not None:
            spans = self.typer.type_spans(answer, prompt, spans)
        return spans

    def predict_prompt_batch(
        self, prompts: list[str], answers: list[str], output_format: str = "tokens"
    ) -> list:
        """Predict hallucination tokens or spans from the provided prompts and answers.

        :param prompts: List of prompt strings.
        :param answers: List of answer strings.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :returns: List of prediction lists, one per input pair.
        """
        return [self.predict_prompt(p, a, output_format) for p, a in zip(prompts, answers)]
