"""Factory function for creating detector instances."""

from __future__ import annotations

from lettucedetect.detectors._deprecation import deprecated
from lettucedetect.detectors.base import BaseDetector

__all__ = ["make_detector"]


@deprecated("load_detector")
def make_detector(method: str, **kwargs) -> BaseDetector:
    """Create a detector of the requested type with the given parameters.

    .. deprecated:: 0.3.0
        Use :func:`~lettucedetect.detectors.load_detector` for a KR Labs model id, or build
        the detector class directly. ``make_detector`` will be removed in 0.5.0.

    :param method: One of "transformer", "llm", or "rag_fact_checker".
    :param kwargs: Passed to the concrete detector constructor.
    :return: A concrete detector instance.
    :raises ValueError: If method is not supported.
    """
    if method == "transformer":
        from lettucedetect.detectors.transformer import TransformerDetector

        return TransformerDetector(**kwargs)
    elif method == "llm":
        from lettucedetect.detectors.llm import LLMDetector

        return LLMDetector(**kwargs)
    elif method == "rag_fact_checker":
        from lettucedetect.detectors.rag_fact_checker import RAGFactCheckerDetector

        return RAGFactCheckerDetector(**kwargs)
    else:
        raise ValueError(
            f"Unknown detector method: {method}. Use one of: transformer, llm, rag_fact_checker"
        )
