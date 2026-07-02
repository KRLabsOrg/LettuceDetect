"""Abstract base class for hallucination detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """All hallucination detectors implement a common interface."""

    @abstractmethod
    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
        min_confidence: float = 0.0,
    ) -> list:
        """Predict hallucination tokens or spans given passages and an answer.

        :param context: List of passages that were supplied to the LLM / user.
        :param answer: Model-generated answer to inspect.
        :param question: Original question (``None`` for summarisation).
        :param output_format: ``"tokens"`` for token-level dicts, ``"spans"`` for character spans.
        :param min_confidence: Drop ``"spans"`` whose ``confidence`` is below this threshold
            (in ``[0, 1]``; ``0.0`` keeps every span). Ignored for ``"tokens"`` output.
        :returns: List of predictions in requested format.
        """
        pass

    @abstractmethod
    def predict_prompt(
        self, prompt: str, answer: str, output_format: str = "tokens", min_confidence: float = 0.0
    ) -> list:
        """Predict hallucinations from a pre-built prompt string.

        :param prompt: Full prompt (context + question already concatenated).
        :param answer: Model-generated answer to inspect.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :param min_confidence: Drop ``"spans"`` below this confidence threshold (``[0, 1]``).
        :returns: List of predictions in requested format.
        """
        pass

    @abstractmethod
    def predict_prompt_batch(
        self,
        prompts: list[str],
        answers: list[str],
        output_format: str = "tokens",
        min_confidence: float = 0.0,
    ) -> list:
        """Batch version of :meth:`predict_prompt`.

        :param prompts: List of full prompt strings.
        :param answers: List of answers to inspect.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :param min_confidence: Drop ``"spans"`` below this confidence threshold (``[0, 1]``).
        :returns: List of prediction lists, one per input pair.
        """
        pass

    @staticmethod
    def _validate_min_confidence(min_confidence: float) -> None:
        """Validate that ``min_confidence`` is a probability in ``[0, 1]``.

        :param min_confidence: The threshold to validate.
        :raises ValueError: If ``min_confidence`` is outside ``[0, 1]``.
        """
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")

    @staticmethod
    def _filter_spans_by_confidence(
        result: list, output_format: str, min_confidence: float
    ) -> list:
        """Drop spans scored below ``min_confidence`` from a ``"spans"`` result.

        Only ``"spans"`` output is filtered, and only when ``min_confidence > 0``;
        token-level output is returned unchanged so callers can still inspect
        low-confidence tokens. Spans without a ``confidence`` key are kept, mirroring
        the existing behaviour of the LLM detector's ``_to_spans``.

        :param result: The prediction list returned by a detector.
        :param output_format: The format the result was produced in.
        :param min_confidence: Minimum span confidence to keep (``[0, 1]``).
        :returns: The (possibly filtered) result list.
        """
        if output_format != "spans" or min_confidence <= 0.0:
            return result
        return [
            span
            for span in result
            if span.get("confidence") is None or span["confidence"] >= min_confidence
        ]
