"""LettuceDetect: Hallucination detection and generation for RAG systems."""

# Main detection interface
# Core data structures
from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
    HallucinationDataset,
    HallucinationSample,
)

# Generation interface
from lettucedetect.models.generation import HallucinationGenerator
from lettucedetect.models.inference import HallucinationDetector

# Direct RAGFactChecker access for advanced users
from lettucedetect.ragfactchecker import RAGFactChecker

__version__ = "0.2.0"

__all__ = [
    "HallucinationData",
    "HallucinationDataset",
    "HallucinationDetector",
    "HallucinationGenerator",
    "HallucinationSample",
    "RAGFactChecker",  # Direct access to triplet functionality
]
