from __future__ import annotations

from lettucedetect.detectors._deprecation import LettuceDetectDeprecationWarning
from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.errors import (
    ConfigurationError,
    InvalidInputError,
    LettuceDetectError,
    UpstreamError,
)
from lettucedetect.detectors.factory import make_detector as _make_detector
from lettucedetect.detectors.llm import LLMDetector
from lettucedetect.detectors.rag_fact_checker import RAGFactCheckerDetector
from lettucedetect.detectors.transformer import TransformerDetector
from lettucedetect.detectors.types import (
    Chunk,
    DetailedResult,
    DetectedSpan,
    Detection,
    DetectRequest,
    PromptInput,
    ResultWarning,
    SourcesInput,
    Span,
    TokenScore,
    Usage,
)

__all__ = [
    "BaseDetector",
    "Chunk",
    "ConfigurationError",
    "DetailedResult",
    "DetectRequest",
    "DetectedSpan",
    "Detection",
    "InvalidInputError",
    "LLMDetector",
    "LettuceDetectError",
    "PromptInput",
    "LettuceDetectDeprecationWarning",
    "RAGFactCheckerDetector",
    "ResultWarning",
    "SourcesInput",
    "Span",
    "TokenScore",
    "TransformerDetector",
    "UpstreamError",
    "Usage",
    "_make_detector",
]
