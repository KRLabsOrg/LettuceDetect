"""Pytest tests for RAGFactCheckerDetector batch prediction."""

from unittest.mock import patch

import pytest

from lettucedetect.detectors.rag_fact_checker import RAGFactCheckerDetector

HALLUCINATED_RESULT = {"hallucinated_triplets": [["Paris", "is capital of", "Germany"]]}
CLEAN_RESULT = {"hallucinated_triplets": []}


@pytest.fixture
def detector():
    """RAGFactCheckerDetector with a mock backend. The tests need no API key and no network."""
    with patch("lettucedetect.ragfactchecker.RAGFactChecker"):
        return RAGFactCheckerDetector()


class TestPredictPromptBatchRAG:
    """predict_prompt_batch() keeps one result for each (prompt, answer) pair."""

    @pytest.mark.parametrize(
        ("prompts", "answers"),
        [
            (["p1", "p2"], ["a1"]),
            (["p1"], ["a1", "a2"]),
            (["p1"], []),
            ([], ["a1"]),
        ],
    )
    def test_mismatched_lengths_raise_value_error(self, detector, prompts, answers):
        """A length mismatch raises a ValueError before the backend gets a request."""
        with pytest.raises(ValueError, match="Number of prompts must match number of answers"):
            detector.predict_prompt_batch(prompts, answers)
        detector.rag.detect_hallucinations_batch.assert_not_called()

    def test_matched_lengths_return_spans_in_input_order(self, detector):
        """Equal lengths give one spans result for each pair, in input order."""
        detector.rag.detect_hallucinations_batch.return_value = [HALLUCINATED_RESULT, CLEAN_RESULT]
        answers = ["Paris is the capital of Germany.", "Berlin is the capital of Germany."]

        results = detector.predict_prompt_batch(["c1", "c2"], answers, output_format="spans")

        assert len(results) == 2
        assert [span["text"] for span in results[0]] == ["Germany"]
        assert results[1] == []

    def test_matched_lengths_return_tokens_in_input_order(self, detector):
        """Equal lengths give one tokens result for each pair, in input order."""
        detector.rag.detect_hallucinations_batch.return_value = [HALLUCINATED_RESULT, CLEAN_RESULT]
        answers = ["Paris is the capital of Germany.", "Berlin is the capital of Germany."]

        results = detector.predict_prompt_batch(["c1", "c2"], answers, output_format="tokens")

        assert [len(tokens) for tokens in results] == [6, 6]
        assert any(token["pred"] == 1 for token in results[0])
        assert all(token["pred"] == 0 for token in results[1])

    def test_short_backend_result_raises_instead_of_truncating(self, detector):
        """The call raises a ValueError when the backend returns fewer results than answers."""
        detector.rag.detect_hallucinations_batch.return_value = [CLEAN_RESULT]

        with pytest.raises(ValueError, match="shorter"):
            detector.predict_prompt_batch(["c1", "c2"], ["a1", "a2"], output_format="spans")
