"""Tests for the detect request and result types."""

import pytest
from pydantic import TypeAdapter, ValidationError

from lettucedetect.detectors import (
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
from lettucedetect.detectors.types import Triplets

ANSWER = "The capital of Atlantis is Poseidonis."
SOURCE = "Atlantis is a legendary island."

REQUEST_ADAPTER = TypeAdapter(DetectRequest)


def _detail() -> DetailedResult:
    return DetailedResult(
        triplets=Triplets(
            answer=[["Atlantis", "capital", "Poseidonis"]],
            context=[["Barcelona", "north", "coast", "Madrid"]],  # triplets may have 4 items
            hallucinated=[["Atlantis", "capital", "Poseidonis"]],
        ),
        fact_check_results={0: False},
        span_triplets=[["Atlantis", "capital", "Poseidonis"], None],
    )


def _detection() -> Detection:
    return Detection(
        spans=[
            DetectedSpan(
                start=28, end=38, text="Poseidonis", score=0.8, category="fabricated", reasoning="r"
            )
        ],
        tokens=[TokenScore(token="Poseidonis", start=28, end=38, score=0.8, flagged=True)],  # noqa: S106
        detail=_detail(),
        usage=[Usage(model_id="m", model_version=None, units={"sources": 1})],
        warnings=[ResultWarning(code="input_truncated")],
    )


INSTANCES = [
    Chunk(id="a", text=SOURCE, metadata={"k": 1}, source_start=0, source_end=10),
    Span(start=0, end=3, text="The", score=0.5),
    DetectedSpan(start=0, end=3, text="The", score=0.5),
    TokenScore(token="The", start=0, end=3, score=0.5, flagged=False),  # noqa: S106
    Usage(model_id=None, model_version=None, units={}),
    _detail(),
    SourcesInput(sources=[Chunk(id="a", text=SOURCE)], answer=ANSWER, question="q"),
    PromptInput(prompt="p", answer=ANSWER),
    ResultWarning(code="x", message="m"),
    _detection(),
]


@pytest.mark.parametrize("obj", INSTANCES, ids=lambda o: type(o).__name__)
def test_json_round_trip(obj):
    # Every type survives model_dump_json -> model_validate_json unchanged.
    assert type(obj).model_validate_json(obj.model_dump_json()) == obj


def test_detect_request_validates_by_kind():
    # A DetectRequest becomes a SourcesInput or a PromptInput according to `kind`.
    sources = SourcesInput(sources=[Chunk(id="a", text=SOURCE)], answer=ANSWER)
    prompt = PromptInput(prompt="p", answer=ANSWER)
    assert REQUEST_ADAPTER.validate_json(sources.model_dump_json()) == sources
    assert isinstance(REQUEST_ADAPTER.validate_json(prompt.model_dump_json()), PromptInput)
    with pytest.raises(ValidationError):
        REQUEST_ADAPTER.validate_python({"kind": "other", "answer": "a"})


@pytest.mark.parametrize("obj", INSTANCES, ids=lambda o: type(o).__name__)
def test_types_are_frozen(obj):
    # Assigning to a field raises.
    field = next(iter(type(obj).model_fields))
    with pytest.raises(ValidationError):
        setattr(obj, field, getattr(obj, field))


def test_validation_error_hides_customer_text():
    # A bad SourcesInput reports the problem without echoing the answer or a source text.
    with pytest.raises(ValidationError) as exc:
        SourcesInput.model_validate(
            {"sources": [{"id": "", "text": SOURCE}, {"id": "b", "text": 5}], "answer": 7}
        )
    message = str(exc.value)
    assert ANSWER not in message
    assert SOURCE not in message
    with pytest.raises(ValidationError) as exc:
        Chunk(id="a", text=SOURCE, source_start=5, source_end=1)
    assert SOURCE not in str(exc.value)


def test_detection_defaults():
    # tokens and detail default to None; usage and warnings to separate empty lists.
    first, second = Detection(), Detection()
    assert first.spans == []
    assert first.tokens is None
    assert first.detail is None
    assert first.usage == []
    assert first.warnings == []
    assert first.usage is not second.usage
    assert first.warnings is not second.warnings


def test_span_validation():
    # Offsets are ordered, the text fits the offsets and the score is in [0, 1].
    with pytest.raises(ValidationError):
        Span(start=3, end=3, text="", score=0.5)
    with pytest.raises(ValidationError):
        Span(start=0, end=5, text="abc", score=0.5)
    with pytest.raises(ValidationError):
        Span(start=0, end=3, text="abc", score=1.5)
    with pytest.raises(ValidationError):
        Span(start=-1, end=2, text="abc", score=0.5)


def test_chunk_validation():
    # The id is non-empty and the source offsets are ordered.
    with pytest.raises(ValidationError):
        Chunk(id="", text="t")
    with pytest.raises(ValidationError):
        Chunk(id="a", text="t", source_start=5, source_end=1)


def test_detailed_result_accepts_json_string_keys():
    # fact_check_results has int keys, which JSON turns into strings.
    restored = DetailedResult.model_validate_json(_detail().model_dump_json())
    assert restored.fact_check_results == {0: False}
    assert restored.triplets.context == [["Barcelona", "north", "coast", "Madrid"]]
