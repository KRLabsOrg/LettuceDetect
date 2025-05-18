"""Pytest tests for the inference module."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from lettucedetect.detectors.prompt_utils import PromptUtils
from lettucedetect.detectors.transformer import TransformerDetector
from lettucedetect.models.inference import HallucinationDetector


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [101, 102, 103, 104, 105]
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]])
    model.return_value = mock_output
    return model


class TestHallucinationDetector:
    """Tests for the HallucinationDetector class."""

    def test_init_with_transformer_method(self):
        """Test initialization with transformer method."""
        with patch("lettucedetect.detectors.transformer.TransformerDetector") as mock_transformer:
            detector = HallucinationDetector(method="transformer", model_path="dummy_path")
            mock_transformer.assert_called_once_with(model_path="dummy_path")
            assert isinstance(detector.detector, MagicMock)

    def test_init_with_invalid_method(self):
        """Test initialization with invalid method."""
        with pytest.raises(ValueError):
            HallucinationDetector(method="invalid_method")

    def test_predict(self):
        """Test predict method."""
        # Create a mock detector with the predict method
        mock_detector = MagicMock()
        mock_detector.predict.return_value = []

        with patch(
            "lettucedetect.detectors.transformer.TransformerDetector", return_value=mock_detector
        ):
            detector = HallucinationDetector(method="transformer")
            context = ["This is a test context."]
            answer = "This is a test answer."
            question = "What is the test?"

            result = detector.predict(context, answer, question)

            # Check that the mock detector's predict method was called with the correct arguments
            mock_detector.predict.assert_called_once()
            call_args = mock_detector.predict.call_args[0]
            assert call_args[0] == context
            assert call_args[1] == answer
            assert call_args[2] == question
            assert call_args[3] == "tokens"

    def test_predict_prompt(self):
        """Test predict_prompt method."""
        # Create a mock detector with the predict_prompt method
        mock_detector = MagicMock()
        mock_detector.predict_prompt.return_value = []

        with patch(
            "lettucedetect.detectors.transformer.TransformerDetector", return_value=mock_detector
        ):
            detector = HallucinationDetector(method="transformer")
            prompt = "This is a test prompt."
            answer = "This is a test answer."

            result = detector.predict_prompt(prompt, answer)

            # Check that the mock detector's predict_prompt method was called with the correct arguments
            mock_detector.predict_prompt.assert_called_once()
            call_args = mock_detector.predict_prompt.call_args[0]
            assert call_args[0] == prompt
            assert call_args[1] == answer
            assert call_args[2] == "tokens"


class TestTransformerDetector:
    """Tests for the TransformerDetector class."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_tokenizer, mock_model):
        """Set up test fixtures."""
        self.mock_tokenizer = mock_tokenizer
        self.mock_model = mock_model

        # Patch the AutoTokenizer and AutoModelForTokenClassification
        self.tokenizer_patcher = patch(
            "lettucedetect.detectors.transformer.AutoTokenizer.from_pretrained",
            return_value=self.mock_tokenizer,
        )
        self.model_patcher = patch(
            "lettucedetect.detectors.transformer.AutoModelForTokenClassification.from_pretrained",
            return_value=self.mock_model,
        )

        self.mock_tokenizer_cls = self.tokenizer_patcher.start()
        self.mock_model_cls = self.model_patcher.start()

        yield

        self.tokenizer_patcher.stop()
        self.model_patcher.stop()

    def test_init(self):
        """Test initialization."""
        detector = TransformerDetector(model_path="dummy_path")

        self.mock_tokenizer_cls.assert_called_once_with("dummy_path")
        self.mock_model_cls.assert_called_once_with("dummy_path")
        assert detector.tokenizer == self.mock_tokenizer
        assert detector.model == self.mock_model
        assert detector.max_length == 4096

    def test_predict(self):
        """Test predict method."""

        # Create a proper mock encoding with input_ids as a tensor attribute
        class MockEncoding:
            def __init__(self):
                self.input_ids = torch.tensor([[101, 102, 103]])

        mock_encoding = MockEncoding()
        mock_labels = torch.tensor([0, 0, 0])
        mock_offsets = torch.tensor([[0, 0], [0, 1], [1, 2]])
        mock_answer_start = 1

        # Patch the _predict method to avoid the actual implementation
        with patch.object(TransformerDetector, "_predict", return_value=[]):
            detector = TransformerDetector(model_path="dummy_path")
            context = ["This is a test context."]
            answer = "This is a test answer."
            question = "What is the test?"

            result = detector.predict(context, answer, question)

            # Verify the result
            assert isinstance(result, list)

    def test_form_prompt_with_question(self):
        """Test _form_prompt method with a question."""
        detector = TransformerDetector(model_path="dummy_path")
        context = ["This is passage 1.", "This is passage 2."]
        question = "What is the test?"

        prompt = PromptUtils.format_context(context, question, "en")

        # Check that the prompt contains the question and passages
        assert question in prompt
        assert "passage 1: This is passage 1." in prompt
        assert "passage 2: This is passage 2." in prompt

    def test_form_prompt_without_question(self):
        """Test _form_prompt method without a question (summary task)."""
        detector = TransformerDetector(model_path="dummy_path")
        context = ["This is a text to summarize."]

        prompt = PromptUtils.format_context(context, None, "en")

        # Check that the prompt contains the text to summarize
        assert "This is a text to summarize." in prompt
        assert "Summarize" in prompt
