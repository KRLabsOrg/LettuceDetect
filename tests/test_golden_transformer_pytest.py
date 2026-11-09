"""Golden (characterization) tests pinning today's ``TransformerDetector`` output.

The model is a stub whose logits are a closed-form function of the token id and the tokenizer is
the local WordPiece one from ``conftest.py``, so the tests are deterministic and network-free.
Recorded output lives in ``tests/fixtures/golden/transformer_*.json``.

Today's quirks are recorded as-is: ``"tokens"`` output ends with the trailing ``[SEP]`` token, and
the multi-group path flags a token at ``prob >= 0.5`` while the single-group path uses argmax.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from lettucedetect.detectors.transformer import TransformerDetector
from tests.golden import assert_golden

pytestmark = pytest.mark.filterwarnings(
    "ignore::lettucedetect.detectors._deprecation.LettuceDetectDeprecationWarning"
)

FORMATS = ["tokens", "spans"]
MULTI_MAX_LENGTH = 90

# (context passages, answer, question); question None means summarisation.
PREDICT_INPUTS = [
    (
        ["the capital of france is paris ."],
        "the capital of france is paris",
        "the capital of france ?",
    ),
    (["the capital of france is paris ."], "paris is the capital .", None),
    (["short answer word ."], "word answer short", "short ?"),
]
PROMPTS = [
    "the capital of france is paris .",
    "short answer word",
    "the capital of france is paris . the capital of france",
]
ANSWERS = ["paris is the capital", "word", "the capital of france is paris and paris"]

MULTI_CONTEXT = [
    "the capital of france is paris . the capital of france is paris .",
    "short answer word . short answer word . short answer word .",
    "the capital of france is paris . the capital of france is paris .",
    "word word word word word word word word word word word word word word .",
]
MULTI_QUESTION = "the capital of france ?"
MULTI_ANSWER = "paris is the capital of france and the answer is word"


def _stub_forward(input_ids=None, attention_mask=None, **kwargs):
    """Logits depend only on the token id: P(hallucination) = sigmoid((id*7) % 11 - 5)."""
    score = ((input_ids * 7) % 11 - 5).float() / 2
    logits = torch.stack([-score, score], dim=-1)
    output = MagicMock()
    output.logits = logits
    return output


def _build(tokenizer, max_length: int) -> TransformerDetector:
    model = MagicMock(side_effect=_stub_forward)
    with (
        patch(
            "lettucedetect.detectors.transformer.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ),
        patch(
            "lettucedetect.detectors.transformer.AutoModelForTokenClassification.from_pretrained",
            return_value=model,
        ),
    ):
        return TransformerDetector(model_path="dummy_path", max_length=max_length)


class _StubTyper:
    """Types every span from its own text and the context string it was handed."""

    def type_spans(self, answer, context, spans):
        for span in spans:
            span["category"] = f"CAT[{span['text']}]"
            span["subcategory"] = f"SUB[{len(context)}]"
        return spans


@pytest.fixture
def detector(local_wordpiece_tokenizer):
    return _build(local_wordpiece_tokenizer, 512)


@pytest.fixture
def multi_detector(local_wordpiece_tokenizer):
    return _build(local_wordpiece_tokenizer, MULTI_MAX_LENGTH)


def _predict_all(det, output_format):
    return [det.predict(c, a, q, output_format=output_format) for c, a, q in PREDICT_INPUTS]


def _predict_prompt_all(det, output_format):
    return [det.predict_prompt(p, a, output_format=output_format) for p, a in zip(PROMPTS, ANSWERS)]


def _predict_prompt_batch(det, output_format):
    return det.predict_prompt_batch(PROMPTS, ANSWERS, output_format=output_format)


@pytest.mark.parametrize("output_format", FORMATS)
class TestTransformerGolden:
    """Every public entry point in both output formats."""

    def test_predict(self, detector, output_format):
        """predict() on the small inputs."""
        assert_golden(f"transformer_predict_{output_format}", _predict_all(detector, output_format))

    def test_predict_prompt(self, detector, output_format):
        """predict_prompt() on the small inputs."""
        assert_golden(
            f"transformer_predict_prompt_{output_format}",
            _predict_prompt_all(detector, output_format),
        )

    def test_predict_prompt_batch(self, detector, output_format):
        """predict_prompt_batch() on the small inputs."""
        assert_golden(
            f"transformer_predict_prompt_batch_{output_format}",
            _predict_prompt_batch(detector, output_format),
        )

    def test_multi_group(self, multi_detector, output_format):
        """A context needing several passage groups, aggregated with max()."""
        groups = multi_detector._group_passages_into_chunks(
            MULTI_CONTEXT, MULTI_QUESTION, MULTI_ANSWER
        )
        assert len(groups) > 1, "the multi-group fixture must really need several groups"
        output = multi_detector.predict(
            MULTI_CONTEXT, MULTI_ANSWER, MULTI_QUESTION, output_format=output_format
        )
        assert_golden(
            f"transformer_predict_multi_group_{output_format}",
            {"groups": groups, "output": output},
        )


class TestTransformerTypedSpansGolden:
    """Spans typed by a stub typer."""

    def test_typed_spans(self, detector):
        """The typer sees joined passages in predict and the full prompt elsewhere."""
        detector.typer = _StubTyper()
        assert_golden(
            "transformer_typed_spans",
            {
                "predict": _predict_all(detector, "spans"),
                "predict_prompt": _predict_prompt_all(detector, "spans"),
                "predict_prompt_batch": _predict_prompt_batch(detector, "spans"),
            },
        )
