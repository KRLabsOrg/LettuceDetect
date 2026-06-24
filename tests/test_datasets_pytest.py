"""Pytest tests for the datasets module."""

from lettucedetect.datasets.hallucination_dataset import HallucinationSample


def test_from_json_with_minimal_keys():
    """Test that from_json handles data missing optional dataset/language keys."""
    sample = HallucinationSample.from_json(
        {
            "prompt": "User request: What is AI?\n\nAI is artificial intelligence.",
            "answer": "AI stands for Artificial Intelligence.",
            "labels": [],
            "split": "test",
            "task_type": "qa",
        }
    )
    assert sample.prompt == "User request: What is AI?\n\nAI is artificial intelligence."
    assert sample.answer == "AI stands for Artificial Intelligence."
    assert sample.labels == []
    assert sample.split == "test"
    assert sample.task_type == "qa"
    assert sample.dataset == "unknown"
    assert sample.language == "en"


def test_from_json_explicit_dataset_language():
    """Test that explicit dataset and language keys are preserved."""
    sample = HallucinationSample.from_json(
        {
            "prompt": "Test prompt",
            "answer": "Test answer",
            "labels": [{"start": 0, "end": 4, "label": "intrinsic"}],
            "split": "train",
            "task_type": "summarization",
            "dataset": "cnn_dailymail",
            "language": "de",
        }
    )
    assert sample.dataset == "cnn_dailymail"
    assert sample.language == "de"
