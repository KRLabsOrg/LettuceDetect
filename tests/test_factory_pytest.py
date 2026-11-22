"""Pytest tests for the detector factory (``make_detector``)."""

import sys
import warnings
from unittest.mock import MagicMock, patch

import pytest

from lettucedetect.detectors import _deprecation as dep
from lettucedetect.detectors._deprecation import LettuceDetectDeprecationWarning
from lettucedetect.detectors.factory import make_detector
from lettucedetect.models.inference import HallucinationDetector

pytestmark = pytest.mark.filterwarnings(
    "ignore::lettucedetect.detectors._deprecation.LettuceDetectDeprecationWarning"
)


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


class TestDeprecation:
    """``make_detector`` warns outside the package and names ``load_detector``."""

    MESSAGE = (
        f"make_detector is deprecated since {dep.DEPRECATED_SINCE} and will be removed in "
        f"{dep.REMOVED_IN}. Use load_detector."
    )

    def test_call_from_a_test_module_warns_at_the_caller(self):
        """The warning has the standard message and points at the calling line."""
        with (
            patch("lettucedetect.detectors.transformer.TransformerDetector") as mock_cls,
            pytest.warns(LettuceDetectDeprecationWarning) as caught,
        ):
            line = sys._getframe().f_lineno + 1
            result = make_detector("transformer", model_path="dummy_path")
        assert result is mock_cls.return_value
        assert [str(w.message) for w in caught] == [self.MESSAGE]
        assert (caught[0].filename, caught[0].lineno) == (__file__, line)

    def test_unknown_method_warns_and_raises(self):
        """An unknown method still raises ``ValueError`` after the warning."""
        with pytest.warns(LettuceDetectDeprecationWarning), pytest.raises(ValueError):
            make_detector("bogus")

    def test_call_from_inside_the_package_is_silent(self):
        """A caller in ``lettucedetect.*``, such as the facade, sees no warning."""
        namespace = {"__name__": "lettucedetect.fake_internal", "make_detector": make_detector}
        with (
            patch("lettucedetect.detectors.transformer.TransformerDetector") as mock_cls,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error")
            exec("made = make_detector('transformer', model_path='dummy_path')", namespace)  # noqa: S102
            facade = HallucinationDetector(method="transformer", model_path="dummy_path")
        assert namespace["made"] is mock_cls.return_value
        assert facade.detector is mock_cls.return_value

    def test_is_marked_and_documented(self):
        """The function carries ``__deprecated__`` and the docstring note."""
        assert make_detector.__deprecated__ == self.MESSAGE
        assert ".. deprecated:: 0.3.0" in make_detector.__doc__
        assert "load_detector" in make_detector.__doc__
        assert dep.REMOVED_IN in make_detector.__doc__


class TestDetectorHierarchy:
    """Every concrete detector must subclass BaseDetector (``predict`` relies on its helpers)."""

    def test_all_detectors_inherit_base_detector(self):
        """LLMDetector, TransformerDetector and RAGFactCheckerDetector subclass BaseDetector."""
        from lettucedetect.detectors.base import BaseDetector
        from lettucedetect.detectors.llm import LLMDetector
        from lettucedetect.detectors.rag_fact_checker import RAGFactCheckerDetector
        from lettucedetect.detectors.transformer import TransformerDetector

        for cls in (LLMDetector, TransformerDetector, RAGFactCheckerDetector):
            assert issubclass(cls, BaseDetector)
