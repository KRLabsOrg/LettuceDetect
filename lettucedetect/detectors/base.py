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
    ) -> list:
        """Predict hallucination tokens or spans given passages and an answer.

        :param context: List of passages that were supplied to the LLM / user.
        :param answer: Model‑generated answer to inspect.
        :param question: Original question (``None`` for summarisation).
        :param output_format: ``"tokens"`` for token‑level dicts, ``"spans"`` for character spans.
        :returns: List of predictions in requested format.
        """
        pass

    @abstractmethod
    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucinations when you already have a *single* full prompt string."""
        pass

    @abstractmethod
    def predict_prompt_batch(
        self, prompts: list[str], answers: list[str], output_format: str = "tokens"
    ) -> list:
        """Batch version of `predict_prompt`."""
        pass
