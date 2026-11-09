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
from lettucedetect.detectors.file_cache import FileCache
from lettucedetect.detectors.llm import LLMDetector
from lettucedetect.detectors.rag_fact_checker import RAGFactCheckerDetector
from lettucedetect.detectors.response_cache import (
    InMemoryCache,
    ResponseCache,
    SyncCacheBase,
    make_cache_key,
)
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
    "FileCache",
    "InMemoryCache",
    "InvalidInputError",
    "LLMDetector",
    "LettuceDetectDeprecationWarning",
    "LettuceDetectError",
    "PromptInput",
    "RAGFactCheckerDetector",
    "ResponseCache",
    "ResultWarning",
    "SourcesInput",
    "Span",
    "SyncCacheBase",
    "TokenScore",
    "TransformerDetector",
    "UpstreamError",
    "Usage",
    "_make_detector",
    "make_cache_key",
]
