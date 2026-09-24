"""Pytest tests for the inference module."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from lettucedetect.datasets.hallucination_dataset import HallucinationDataset
from lettucedetect.detectors.base import BaseDetector
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
        # Patch internals to avoid actual model inference
        with (
            patch.object(
                TransformerDetector,
                "_group_passages_into_chunks",
                return_value=[["This is a test context."]],
            ),
            patch.object(TransformerDetector, "_predict_single", return_value=[]),
        ):
            detector = TransformerDetector(model_path="dummy_path")
            context = ["This is a test context."]
            answer = "This is a test answer."
            question = "What is the test?"

            result = detector.predict(context, answer, question)

            # Verify the result
            assert isinstance(result, list)

    @pytest.mark.parametrize("output_format", ["tokens", "spans"])
    def test_predict_accepts_valid_output_formats(self, output_format):
        """predict() accepts both supported output formats."""
        with (
            patch.object(
                TransformerDetector,
                "_group_passages_into_chunks",
                return_value=[["This is a test context."]],
            ),
            patch.object(TransformerDetector, "_predict_single", return_value=[]) as mock_single,
        ):
            detector = TransformerDetector(model_path="dummy_path")

            result = detector.predict(
                ["This is a test context."],
                "This is a test answer.",
                "What is the test?",
                output_format=output_format,
            )

        assert result == []
        assert mock_single.call_args.args[2] == output_format

    @pytest.mark.parametrize("output_format", ["tokens", "spans"])
    def test_predict_prompt_accepts_valid_output_formats(self, output_format):
        """predict_prompt() accepts both supported output formats."""
        self.mock_tokenizer.return_value = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
        with patch.object(TransformerDetector, "_predict_single", return_value=[]) as mock_single:
            detector = TransformerDetector(model_path="dummy_path")

            result = detector.predict_prompt(
                "This is a test prompt.",
                "This is a test answer.",
                output_format=output_format,
            )

        assert result == []
        assert mock_single.call_args.args[2] == output_format

    @pytest.mark.parametrize(
        ("method_name", "args"),
        [
            ("predict", (["This is a test context."], "This is a test answer.", "What is it?")),
            ("predict_prompt", ("This is a test prompt.", "This is a test answer.")),
        ],
    )
    def test_invalid_output_format_raises_value_error(self, method_name, args):
        """Public transformer entry points reject unsupported output formats."""
        detector = TransformerDetector(model_path="dummy_path")

        with pytest.raises(ValueError) as exc_info:
            getattr(detector, method_name)(*args, output_format="json")

        message = str(exc_info.value)
        assert "json" in message
        assert "tokens" in message
        assert "spans" in message

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


class TestTransformerPromptBatch:
    """Tests for padded transformer batch inference."""

    @pytest.fixture(autouse=True)
    def setup(self, local_wordpiece_tokenizer):
        """Build a detector around a deterministic model stub."""
        self.detector = TransformerDetector.__new__(TransformerDetector)
        self.detector.tokenizer = local_wordpiece_tokenizer
        self.detector.max_length = 32
        self.detector.device = torch.device("cpu")
        self.detector.typer = None

        hallucinated_ids = {
            local_wordpiece_tokenizer.convert_tokens_to_ids("paris"),
            local_wordpiece_tokenizer.convert_tokens_to_ids("answer"),
        }

        def forward(input_ids, attention_mask, labels=None):
            del attention_mask, labels
            logits = torch.empty((*input_ids.shape, 2), dtype=torch.float)
            logits[..., 0] = 4.0
            logits[..., 1] = 0.0
            for token_id in hallucinated_ids:
                mask = input_ids == token_id
                logits[mask, 0] = 0.0
                logits[mask, 1] = 4.0
            output = MagicMock()
            output.logits = logits
            return output

        self.detector.model = MagicMock(side_effect=forward)

    def test_batch_matches_single_predictions_and_calls_model_once(self):
        """Uneven inputs retain order and use one model forward pass."""
        prompts = ["The capital of France is", "short"]
        answers = ["paris.", "short answer"]

        expected = [
            self.detector.predict_prompt(prompt, answer, output_format="tokens")
            for prompt, answer in zip(prompts, answers)
        ]
        self.detector.model.reset_mock()

        actual = self.detector.predict_prompt_batch(prompts, answers, output_format="tokens")

        assert actual == expected
        self.detector.model.assert_called_once()
        assert [item[0]["token"].strip() for item in actual] == ["paris", "short"]
        assert all(
            item["token"] != "[PAD]"  # noqa: S105 - tokenizer special token
            for result in actual
            for item in result
        )

    def test_single_item_batch_uses_one_forward_call(self):
        """A batch of size one still uses the padded batch path."""
        result = self.detector.predict_prompt_batch(["short"], ["paris."], "tokens")

        assert result[0][0]["pred"] == 1
        self.detector.model.assert_called_once()

    def test_spans_confidence_filtering_and_taxonomy_typing(self):
        """Span output is filtered after optional taxonomy typing."""
        typer = MagicMock()
        typer.type_spans.side_effect = lambda answer, prompt, spans: [
            {**span, "category": f"{prompt}:{answer}"} for span in spans
        ]
        self.detector.typer = typer

        results = self.detector.predict_prompt_batch(
            ["capital", "short"],
            ["paris.", "short answer"],
            output_format="spans",
            min_confidence=0.9,
        )

        assert [[span["text"] for span in result] for result in results] == [
            ["paris"],
            ["answer"],
        ]
        assert results[0][0]["category"] == "capital:paris."
        assert results[1][0]["category"] == "short:short answer"
        assert typer.type_spans.call_count == 2

        filtered = self.detector.predict_prompt_batch(
            ["capital"], ["paris."], output_format="spans", min_confidence=0.99
        )
        assert filtered == [[]]

    def test_empty_batch_skips_tokenizer_and_model(self):
        """An empty input produces an empty output without inference."""
        tokenizer = MagicMock(wraps=self.detector.tokenizer)
        self.detector.tokenizer = tokenizer

        assert self.detector.predict_prompt_batch([], []) == []
        tokenizer.assert_not_called()
        self.detector.model.assert_not_called()

    def test_mismatched_lengths_raise_before_inference(self):
        """Trailing prompts or answers are rejected instead of truncated by zip."""
        with pytest.raises(ValueError, match="same number"):
            self.detector.predict_prompt_batch(["one", "two"], ["answer"])

        self.detector.model.assert_not_called()


class TestChunking:
    """Tests for automatic context chunking when input exceeds max_length."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up a TransformerDetector with mocked model/tokenizer."""
        with (
            patch(
                "lettucedetect.detectors.transformer.AutoTokenizer.from_pretrained",
            ) as tok_cls,
            patch(
                "lettucedetect.detectors.transformer.AutoModelForTokenClassification.from_pretrained",
            ) as model_cls,
        ):
            # Set up a realistic tokenizer mock
            self.mock_tokenizer = MagicMock()
            self.mock_model = MagicMock()
            tok_cls.return_value = self.mock_tokenizer
            model_cls.return_value = self.mock_model

            self.detector = TransformerDetector(model_path="dummy", max_length=32)
            yield

    def test_aggregation_uses_max(self):
        """Max aggregation: highest hallucination prob across chunks wins."""
        # Simulate two chunks with different probabilities for the same tokens
        chunk1_tokens = [
            {"token": "The", "pred": 0, "prob": 0.1},
            {"token": "answer", "pred": 0, "prob": 0.3},
            {"token": "is", "pred": 1, "prob": 0.8},
        ]
        chunk2_tokens = [
            {"token": "The", "pred": 0, "prob": 0.2},
            {"token": "answer", "pred": 1, "prob": 0.6},
            {"token": "is", "pred": 0, "prob": 0.4},
        ]

        with patch.object(self.detector, "_predict_single") as mock_single:
            mock_single.side_effect = [chunk1_tokens, chunk2_tokens]

            # Mock _build_spans_from_tokens for the spans path
            with patch.object(self.detector, "_build_spans_from_tokens", return_value=[]):
                result = self.detector._predict_chunked(
                    ["chunk1", "chunk2"], "The answer is", output_format="tokens"
                )

        assert len(result) == 3
        # Token 0: max(0.1, 0.2) = 0.2 → pred 0
        assert result[0]["prob"] == 0.2
        assert result[0]["pred"] == 0
        # Token 1: max(0.3, 0.6) = 0.6 → pred 1 (≥ 0.5)
        assert result[1]["prob"] == 0.6
        assert result[1]["pred"] == 1
        # Token 2: max(0.8, 0.4) = 0.8 → pred 1
        assert result[2]["prob"] == 0.8
        assert result[2]["pred"] == 1

    def test_group_passages_single_group(self):
        """When all passages fit, _group_passages_into_chunks returns one group."""

        def tokenizer_side_effect(text, **kwargs):
            # Approximate: 1 token per word
            words = len(text.split()) if text else 1
            return {"input_ids": torch.zeros(1, words, dtype=torch.long)}

        self.mock_tokenizer.side_effect = tokenizer_side_effect

        with patch(
            "lettucedetect.detectors.transformer.PromptUtils.format_context",
            return_value="short prompt with passages",
        ):
            groups = self.detector._group_passages_into_chunks(
                ["passage one", "passage two"], "What?", "short answer"
            )
        assert len(groups) == 1
        assert groups[0] == ["passage one", "passage two"]

    def test_group_passages_multiple_groups(self):
        """When passages exceed budget, they should be split into multiple groups."""
        # max_length=32, answer=5 tokens → total_budget = 32-5-3 = 24
        # instruction_overhead = 10 tokens → passage_budget = 24-10 = 14
        # Each passage = 10 tokens → 2 passages won't fit (10+1+10=21 > 14), so 1 per group

        def tokenizer_side_effect(text, **kwargs):
            if text == "short answer":
                return {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
            if text.startswith("passage"):
                return {"input_ids": torch.zeros(1, 10, dtype=torch.long)}
            # "full_prompt" → 50 tokens (exceeds budget, triggers chunking)
            if "full_prompt" in text:
                return {"input_ids": torch.zeros(1, 50, dtype=torch.long)}
            # "minimal" → 10 tokens (instruction overhead)
            return {"input_ids": torch.zeros(1, 10, dtype=torch.long)}

        self.mock_tokenizer.side_effect = tokenizer_side_effect

        format_calls = [0]

        def format_side_effect(ctx, q, lang):
            format_calls[0] += 1
            if format_calls[0] == 1:
                return "full_prompt_long"  # First call: all passages → triggers chunking
            return "minimal"  # Second call: [""] → measures instruction overhead

        with patch(
            "lettucedetect.detectors.transformer.PromptUtils.format_context",
            side_effect=format_side_effect,
        ):
            groups = self.detector._group_passages_into_chunks(
                ["passage A", "passage B", "passage C"], "What?", "short answer"
            )

        # With passage_budget = 14 and each passage = 10 tokens,
        # each group holds 1 passage (10 < 14, but 10+1+10 = 21 > 14)
        assert len(groups) == 3
        assert groups[0] == ["passage A"]
        assert groups[1] == ["passage B"]
        assert groups[2] == ["passage C"]

    def test_predict_uses_passage_chunking(self):
        """predict() should use _group_passages_into_chunks for chunking."""
        with (
            patch.object(
                self.detector,
                "_group_passages_into_chunks",
                return_value=[["p1"], ["p2"]],
            ) as mock_group,
            patch.object(
                self.detector,
                "_predict_chunked",
                return_value=[{"token": "x", "pred": 0, "prob": 0.1}],
            ) as mock_chunked,
            patch(
                "lettucedetect.detectors.transformer.PromptUtils.format_context",
                side_effect=lambda ctx, q, lang: f"prompt_{ctx[0]}",
            ),
        ):
            self.detector.predict(["p1", "p2"], "answer", "What?", "tokens")
            mock_group.assert_called_once()
            mock_chunked.assert_called_once_with(["prompt_p1", "prompt_p2"], "answer", "tokens")


class TestAnswerStartToken:
    """Tests for the answer_start_token fix in prepare_tokenized_input."""

    def test_answer_start_token_basic(self, local_wordpiece_tokenizer):
        """answer_start_token should point to the first answer token."""
        context = "The capital of France is Paris."
        answer = "Paris is the capital."

        _, _, offsets, answer_start = HallucinationDataset.prepare_tokenized_input(
            local_wordpiece_tokenizer, context, answer, max_length=512
        )

        # answer_start should be within bounds
        assert answer_start > 0
        assert answer_start < offsets.size(0)
        # The offset at answer_start should be non-zero (actual token, not special)
        assert offsets[answer_start][1].item() > offsets[answer_start][0].item()

    def test_answer_start_token_with_truncation(self, local_wordpiece_tokenizer):
        """answer_start_token should be correct even when context is truncated."""
        # Create a very long context that will be truncated at max_length=32
        context = "word " * 200  # ~200 tokens
        answer = "short answer"

        encoding, _, offsets, answer_start = HallucinationDataset.prepare_tokenized_input(
            local_wordpiece_tokenizer, context, answer, max_length=32
        )

        total_len = encoding["input_ids"].shape[1]
        assert total_len == 32  # Should be truncated to max_length

        # answer_start should be within the actual sequence
        assert answer_start > 0
        assert answer_start < total_len

        # The answer tokens should be at the end (before trailing [SEP])
        # Verify by checking that the text at answer_start offset is non-empty
        assert offsets[answer_start][1].item() > offsets[answer_start][0].item()


class TestMinConfidenceFilter:
    """Unit tests for the shared min_confidence helpers on BaseDetector."""

    @pytest.mark.parametrize("value", [-0.1, -1.0, 1.1, 2.0])
    def test_validate_rejects_out_of_range(self, value):
        """Values outside [0, 1] raise a ValueError mentioning min_confidence."""
        with pytest.raises(ValueError, match="min_confidence"):
            BaseDetector._validate_min_confidence(value)

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_validate_accepts_in_range(self, value):
        """Values within [0, 1] are accepted (no exception)."""
        BaseDetector._validate_min_confidence(value)

    def test_filter_drops_low_confidence_spans(self):
        """Spans below the threshold are dropped from a spans result."""
        spans = [
            {"start": 0, "end": 3, "confidence": 0.55, "text": "foo"},
            {"start": 4, "end": 7, "confidence": 0.85, "text": "bar"},
        ]
        filtered = BaseDetector._filter_spans_by_confidence(spans, "spans", 0.7)
        assert filtered == [{"start": 4, "end": 7, "confidence": 0.85, "text": "bar"}]

    def test_filter_zero_threshold_is_noop(self):
        """A 0.0 threshold returns the spans unchanged (backward compatible)."""
        spans = [{"start": 0, "end": 3, "confidence": 0.1, "text": "foo"}]
        assert BaseDetector._filter_spans_by_confidence(spans, "spans", 0.0) == spans

    def test_filter_ignores_token_output(self):
        """Token-level output is never filtered, even at a high threshold."""
        tokens = [{"token": "foo", "pred": 1, "prob": 0.2}]
        assert BaseDetector._filter_spans_by_confidence(tokens, "tokens", 0.9) == tokens

    def test_filter_keeps_spans_without_confidence(self):
        """Spans lacking a confidence key are kept (e.g. LLM without reasoning)."""
        spans = [{"start": 0, "end": 3, "text": "foo"}]
        assert BaseDetector._filter_spans_by_confidence(spans, "spans", 0.9) == spans


class TestTransformerMinConfidence:
    """min_confidence behaviour wired through TransformerDetector."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Build a TransformerDetector with mocked model/tokenizer."""
        with (
            patch(
                "lettucedetect.detectors.transformer.AutoTokenizer.from_pretrained",
                return_value=MagicMock(),
            ),
            patch(
                "lettucedetect.detectors.transformer.AutoModelForTokenClassification.from_pretrained",
                return_value=MagicMock(),
            ),
        ):
            self.detector = TransformerDetector(model_path="dummy_path")
            yield

    def test_predict_spans_filtered_by_min_confidence(self):
        """predict(output_format='spans') drops spans below the threshold."""
        spans = [
            {"start": 0, "end": 3, "confidence": 0.55, "text": "foo"},
            {"start": 4, "end": 7, "confidence": 0.85, "text": "bar"},
        ]
        with (
            patch.object(
                TransformerDetector, "_group_passages_into_chunks", return_value=[["ctx"]]
            ),
            patch.object(TransformerDetector, "_predict_single", return_value=spans),
        ):
            result = self.detector.predict(
                ["ctx"], "foo bar", output_format="spans", min_confidence=0.7
            )
        assert result == [{"start": 4, "end": 7, "confidence": 0.85, "text": "bar"}]

    def test_predict_tokens_ignore_min_confidence(self):
        """predict(output_format='tokens') keeps every token regardless of threshold."""
        tokens = [
            {"token": "foo", "pred": 1, "prob": 0.55},
            {"token": "bar", "pred": 0, "prob": 0.10},
        ]
        with (
            patch.object(
                TransformerDetector, "_group_passages_into_chunks", return_value=[["ctx"]]
            ),
            patch.object(TransformerDetector, "_predict_single", return_value=tokens),
        ):
            result = self.detector.predict(
                ["ctx"], "foo bar", output_format="tokens", min_confidence=0.9
            )
        assert result == tokens

    @pytest.mark.parametrize("bad_value", [-0.1, 1.5])
    @pytest.mark.parametrize("method_name", ["predict", "predict_prompt"])
    def test_invalid_min_confidence_raises(self, method_name, bad_value):
        """Out-of-range min_confidence raises ValueError before any inference."""
        args = (["ctx"], "ans", "q") if method_name == "predict" else ("prompt", "ans")
        with pytest.raises(ValueError, match="min_confidence"):
            getattr(self.detector, method_name)(
                *args, output_format="spans", min_confidence=bad_value
            )

    def test_predict_prompt_batch_respects_min_confidence(self, local_wordpiece_tokenizer):
        """The batch path applies the threshold to each item's spans."""
        spans = [
            {"start": 0, "end": 3, "confidence": 0.40, "text": "foo"},
            {"start": 4, "end": 7, "confidence": 0.90, "text": "bar"},
        ]
        self.detector.tokenizer = local_wordpiece_tokenizer

        def forward(**kwargs):
            output = MagicMock()
            output.logits = torch.zeros((*kwargs["input_ids"].shape, 2))
            return output

        self.detector.model.side_effect = forward
        with patch.object(TransformerDetector, "_format_prediction", return_value=spans):
            results = self.detector.predict_prompt_batch(
                ["p1"], ["foo bar"], output_format="spans", min_confidence=0.5
            )
        assert results == [[{"start": 4, "end": 7, "confidence": 0.90, "text": "bar"}]]


class TestFacadeMinConfidence:
    """The HallucinationDetector facade threads min_confidence to the detector."""

    @staticmethod
    def _facade_with_mock() -> tuple[HallucinationDetector, MagicMock]:
        """Build a facade whose underlying detector is a MagicMock."""
        mock_detector = MagicMock()
        with patch(
            "lettucedetect.detectors.transformer.TransformerDetector", return_value=mock_detector
        ):
            detector = HallucinationDetector(method="transformer")
        return detector, mock_detector

    def test_predict_passes_min_confidence(self):
        """predict() forwards min_confidence to the detector as a keyword."""
        detector, mock_detector = self._facade_with_mock()
        mock_detector.predict.return_value = []
        detector.predict(["ctx"], "ans", "q", output_format="spans", min_confidence=0.8)
        assert mock_detector.predict.call_args.kwargs["min_confidence"] == 0.8

    def test_predict_prompt_passes_min_confidence(self):
        """predict_prompt() forwards min_confidence to the detector as a keyword."""
        detector, mock_detector = self._facade_with_mock()
        mock_detector.predict_prompt.return_value = []
        detector.predict_prompt("prompt", "ans", output_format="spans", min_confidence=0.6)
        assert mock_detector.predict_prompt.call_args.kwargs["min_confidence"] == 0.6

    def test_predict_prompt_batch_passes_min_confidence(self):
        """predict_prompt_batch() forwards min_confidence to the detector as a keyword."""
        detector, mock_detector = self._facade_with_mock()
        mock_detector.predict_prompt_batch.return_value = []
        detector.predict_prompt_batch(["p"], ["a"], output_format="spans", min_confidence=0.3)
        assert mock_detector.predict_prompt_batch.call_args.kwargs["min_confidence"] == 0.3
