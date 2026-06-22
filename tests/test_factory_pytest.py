"""Pytest tests for the detector factory (``make_detector``)."""

from unittest.mock import MagicMock, patch

import pytest

from lettucedetect.detectors.factory import make_detector


class TestMakeDetector:
    """Tests for the ``make_detector`` factory function."""

    def test_transformer_method_returns_transformer_detector(self):
        """``method="transformer"`` builds a TransformerDetector with the given kwargs."""
        mock_detector = MagicMock()
        with patch(
            "lettucedetect.detectors.transformer.TransformerDetector",
            return_value=mock_detector,
        ) as mock_cls:
            result = make_detector("transformer", model_path="dummy_path")
            mock_cls.assert_called_once_with(model_path="dummy_path")
            assert result is mock_detector

    def test_llm_method_returns_llm_detector(self):
        """``method="llm"`` builds an LLMDetector with the given kwargs."""
        mock_detector = MagicMock()
        with patch(
            "lettucedetect.detectors.llm.LLMDetector",
            return_value=mock_detector,
        ) as mock_cls:
            result = make_detector("llm", model="gpt-4.1-mini")
            mock_cls.assert_called_once_with(model="gpt-4.1-mini")
            assert result is mock_detector

    def test_rag_fact_checker_method_returns_rag_detector(self):
        """``method="rag_fact_checker"`` builds a RAGFactCheckerDetector."""
        mock_detector = MagicMock()
        with patch(
            "lettucedetect.detectors.rag_fact_checker.RAGFactCheckerDetector",
            return_value=mock_detector,
        ) as mock_cls:
            result = make_detector("rag_fact_checker")
            mock_cls.assert_called_once_with()
            assert result is mock_detector

    def test_unknown_method_raises_value_error(self):
        """An unsupported method raises ``ValueError`` listing the valid options."""
        with pytest.raises(ValueError) as exc_info:
            make_detector("bogus")
        message = str(exc_info.value)
        assert "bogus" in message
        assert "transformer" in message
        assert "llm" in message
        assert "rag_fact_checker" in message
