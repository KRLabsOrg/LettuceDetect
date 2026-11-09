"""Transformer-based hallucination detector."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from lettucedetect.detectors._deprecation import deprecated
from lettucedetect.detectors.encoder import (
    EncoderConfig,
    EncoderDetector,
    _finish_load,
    _log_model_loaded,
    _resolve_taxonomy,
    _resolved_commit,
)
from lettucedetect.detectors.prompt_utils import LANG_TO_PASSAGE, Lang, PromptUtils

logger = logging.getLogger(__name__)

__all__ = ["TransformerDetector"]


@deprecated("EncoderDetector.from_pretrained")
class TransformerDetector(EncoderDetector):
    """Detect hallucinations with a fine-tuned token classifier.

    .. deprecated:: 0.3.0
        Use :meth:`EncoderDetector.from_pretrained`. ``TransformerDetector`` will be removed
        in 0.5.0. It is an :class:`EncoderDetector`, so ``detect`` and ``adetect`` work too.

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
        include_taxonomy: bool | list | dict = True,
        batch_size: int = 16,
        **tok_kwargs: object,
    ) -> None:
        """Initialize the transformer detector.

        :param model_path: Path to the pre-trained model.
        :param max_length: Maximum length of the input sequence.
        :param device: Device to use for inference.
        :param lang: Language of the model.
        :param taxonomy_head: Optional path/HF id of a label-conditioned typing head. When set,
            ``"spans"`` predictions are typed with a ``category``/``subcategory`` (cascade mode).
        :param include_taxonomy: Label set for the typing head. ``True`` uses the trained
            taxonomy; a ``{name: description}`` dict REPLACES the categories (zero-shot: labels
            enter only as text); ``{"categories": {...}, "subcategories": {...}}`` controls both
            sets; a list of names selects a subset of the trained categories. Only meaningful
            together with ``taxonomy_head``.
        :param batch_size: Default number of (prompt, answer) pairs scored together in one pass
            by :meth:`predict_prompt_batch`. Must be >= 1.
        :param tok_kwargs: Additional keyword arguments for the tokenizer.
        """
        if lang not in LANG_TO_PASSAGE:
            raise ValueError(f"Invalid language. Choose from {', '.join(LANG_TO_PASSAGE)}")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        started = time.perf_counter()
        categories, subcategories = _resolve_taxonomy(taxonomy_head, include_taxonomy)
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
        model = AutoModelForTokenClassification.from_pretrained(model_path, **tok_kwargs)
        device, typer = _finish_load(
            model, device, taxonomy_head, categories, subcategories, tok_kwargs
        )
        version = None if Path(model_path).is_dir() else _resolved_commit(model_path, tok_kwargs)
        super().__init__(
            model,
            tokenizer,
            typer,
            config=EncoderConfig(max_length=max_length, batch_size=batch_size, lang=lang),
            model_id=model_path,
            model_version=version,
        )
        self.device = device
        _log_model_loaded(model_path, version, device, started)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
        min_confidence: float = 0.0,
    ) -> list:
        """Predict hallucination tokens or spans from the provided context, answer, and question.

        :param context: List of passages that were supplied to the LLM / user.
        :param answer: Model-generated answer to inspect.
        :param question: Original question (``None`` for summarisation).
        :param output_format: ``"tokens"`` for token-level dicts, ``"spans"`` for character spans.
        :param min_confidence: Drop ``"spans"`` whose ``confidence`` is below this threshold
            (in ``[0, 1]``; ``0.0`` keeps every span). Ignored for ``"tokens"`` output.
        :returns: List of predictions in requested format.
        """
        if output_format not in ("tokens", "spans"):
            raise ValueError(
                f"TransformerDetector doesn't support '{output_format}' format. "
                "Use 'tokens' or 'spans'"
            )
        self._validate_min_confidence(min_confidence)

        # The tokenizer and model are shared state, so scoring takes the same lock as detect().
        with self._get_scoring_lock():
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
        return self._filter_spans_by_confidence(result, output_format, min_confidence)

    def predict_prompt(
        self, prompt: str, answer: str, output_format: str = "tokens", min_confidence: float = 0.0
    ) -> list:
        """Predict hallucination tokens or spans from the provided prompt and answer.

        Note: unlike :meth:`predict`, this method does **not** chunk the prompt
        automatically.  If the prompt + answer exceed ``max_length``, the prompt
        will be truncated.  Use :meth:`predict` with structured passages for
        automatic chunking.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :param min_confidence: Drop ``"spans"`` below this confidence threshold (``[0, 1]``).
        :returns: List of predictions in requested format.
        """
        if output_format not in ("tokens", "spans"):
            raise ValueError(
                f"TransformerDetector doesn't support '{output_format}' format. "
                "Use 'tokens' or 'spans'"
            )
        self._validate_min_confidence(min_confidence)
        with self._get_scoring_lock():
            # Warn if the input will be truncated.
            total_tokens = self.tokenizer(
                prompt, answer, add_special_tokens=True, return_tensors="pt"
            )["input_ids"].shape[1]
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
        return self._filter_spans_by_confidence(spans, output_format, min_confidence)

    def predict_prompt_batch(
        self,
        prompts: list[str],
        answers: list[str],
        output_format: str = "tokens",
        min_confidence: float = 0.0,
        batch_size: int | None = None,
    ) -> list:
        """Predict hallucination tokens or spans from the provided prompts and answers.

        :param prompts: List of prompt strings.
        :param answers: List of answer strings.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :param min_confidence: Drop ``"spans"`` below this confidence threshold (``[0, 1]``).
        :param batch_size: Max number of pairs tokenized and scored together in one forward
            pass. Defaults to ``self.batch_size`` or the whole input list if that is also
            unset. Every sample in a batch is padded to the longest sequence in that batch,
            so memory scales with ``batch_size x longest_sequence_in_batch``.
        :returns: List of prediction lists, one per input pair.
        :raises ValueError: If ``len(prompts) != len(answers)``.
        """
        if len(prompts) != len(answers):
            raise ValueError("Number of prompts must match number of answers")
        if not prompts:
            return []
        if output_format not in ("tokens", "spans"):
            raise ValueError(
                f"TransformerDetector doesn't support '{output_format}' format."
                " Use 'tokens' or 'spans'"
            )
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._validate_min_confidence(min_confidence)

        effective_batch_size = self.batch_size if batch_size is None else batch_size

        results: list[list] = []
        for start in range(0, len(prompts), effective_batch_size):
            end = start + effective_batch_size
            results.extend(
                self._predict_batch(prompts[start:end], answers[start:end], output_format)
            )

        if output_format == "spans" and self.typer is not None:
            for i in range(len(results)):
                results[i] = self.typer.type_spans(answers[i], prompts[i], results[i])

        return [
            self._filter_spans_by_confidence(result, output_format, min_confidence)
            for result in results
        ]
