"""Factory function for creating detector instances."""

from __future__ import annotations

from lettucedetect.detectors.base import BaseDetector

__all__ = ["make_detector"]


def make_detector(method: str, **kwargs) -> BaseDetector:
    """Create a detector of the requested type with the given parameters.

    :param method: One of "transformer" or "llm".
    :param kwargs: Passed to the concrete detector constructor.
    :return: A concrete detector instance.
    :raises ValueError: If method is not one of "transformer" or "llm".
    """
    if method == "transformer":
        from lettucedetect.detectors.transformer import TransformerDetector

        return TransformerDetector(**kwargs)
    elif method == "llm":
        from lettucedetect.detectors.llm import LLMDetector

        return LLMDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector method: {method}. Use one of: transformer, llm")
