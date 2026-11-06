"""Tests for the bridge between the old predict* and the new detect interfaces."""

from __future__ import annotations

import asyncio
import threading
import warnings
from typing import ClassVar

import pytest

from lettucedetect.detectors import LettuceDetectDeprecationWarning
from lettucedetect.detectors import base as base_module
from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.errors import InvalidInputError
from lettucedetect.detectors.types import (
    Chunk,
    DetectedSpan,
    Detection,
    PromptInput,
    SourcesInput,
    TokenScore,
)

SOURCES = SourcesInput(
    sources=[Chunk(id="a", text="s1"), Chunk(id="b", text="s2")], answer="abcdef", question="q?"
)
PROMPT = PromptInput(prompt="the prompt", answer="abcdef")


class NewStyle(BaseDetector):
    """Implements only _detect."""

    supports_tokens = True

    def __init__(self) -> None:
        """Create the detector."""
        self.requests = []

    def _detect(self, request, *, include_tokens, include_detail):
        self.requests.append(request)
        tokens = [TokenScore(token="abc", start=0, end=3, score=0.8, flagged=True)]  # noqa: S106
        return Detection(
            spans=[
                DetectedSpan(
                    start=0,
                    end=3,
                    text="abc",
                    score=0.8,
                    category="c",
                    subcategory="s",
                    reasoning="r",
                ),
                DetectedSpan(start=4, end=6, text="ef", score=0.3),
            ],
            tokens=tokens if include_tokens else None,
        )


class LegacyStyle(BaseDetector):
    """Implements only the old methods."""

    model_id = "legacy-model"
    spans: ClassVar[list[dict]] = [
        {"start": 0, "end": 3, "text": "abc", "confidence": 0.7, "category": "c", "reasoning": "r"},
        {"start": 4, "end": 6, "text": "ef"},
    ]

    def __init__(self) -> None:
        """Create the detector."""
        self.calls = []
        self.thread_ids = []

    def predict(self, context, answer, question=None, output_format="tokens", min_confidence=0.0):  # noqa: D102
        self.calls.append(("predict", context, answer, question, output_format, min_confidence))
        self.thread_ids.append(threading.get_ident())
        return list(self.spans)

    def predict_prompt(self, prompt, answer, output_format="tokens", min_confidence=0.0):  # noqa: D102
        self.calls.append(("predict_prompt", prompt, answer, output_format, min_confidence))
        self.thread_ids.append(threading.get_ident())
        return list(self.spans)


@pytest.fixture(autouse=True)
def _forget_warned_classes():
    """Start every test with no class marked as already warned."""
    base_module._reset_legacy_warnings()


# --- detect-only subclass answers predict* ---------------------------------------------------


def test_predict_spans_maps_score_to_confidence():
    """A detect-only subclass answers predict(output_format="spans") with confidence from score."""
    d = NewStyle()
    out = d.predict(["s1", "s2"], "abcdef", "q?", "spans")
    assert out == [
        {
            "start": 0,
            "end": 3,
            "text": "abc",
            "confidence": 0.8,
            "category": "c",
            "subcategory": "s",
            "reasoning": "r",
        },
        {"start": 4, "end": 6, "text": "ef", "confidence": 0.3},
    ]
    assert d.requests[0] == SourcesInput(
        sources=[Chunk(id="0", text="s1"), Chunk(id="1", text="s2")], answer="abcdef", question="q?"
    )


def test_predict_tokens_maps_score_to_prob():
    """The predict method returns token dicts with pred from flagged and prob from score."""
    out = NewStyle().predict(["s1"], "abcdef")
    assert out == [{"token": "abc", "pred": 1, "prob": 0.8}]


def test_predict_min_confidence_filters_spans():
    """min_confidence is passed on as min_score and drops low spans."""
    out = NewStyle().predict(["s1"], "abcdef", output_format="spans", min_confidence=0.5)
    assert [s["text"] for s in out] == ["abc"]


def test_predict_prompt_and_batch():
    """predict_prompt builds a PromptInput; the batch runs one pair at a time, in order."""
    d = NewStyle()
    assert d.predict_prompt("p", "abcdef", "spans")[0]["confidence"] == 0.8
    assert d.requests[-1] == PromptInput(prompt="p", answer="abcdef")
    batch = d.predict_prompt_batch(["p1", "p2"], ["abcdef", "abcdef"], "tokens")
    assert batch == [[{"token": "abc", "pred": 1, "prob": 0.8}]] * 2
    assert [r.prompt for r in d.requests[-2:]] == ["p1", "p2"]


@pytest.mark.parametrize(
    ("call", "code", "param"),
    [
        (
            lambda d: d.predict(["s"], "a", output_format="words"),
            "unsupported_output_format",
            "output_format",
        ),
        (
            lambda d: d.predict_prompt("p", "a", output_format="words"),
            "unsupported_output_format",
            "output_format",
        ),
        (
            lambda d: d.predict(["s"], "a", min_confidence=1.5),
            "invalid_min_score",
            "min_confidence",
        ),
        (
            lambda d: d.predict_prompt("p", "a", min_confidence=-1),
            "invalid_min_score",
            "min_confidence",
        ),
        (lambda d: d.predict_prompt_batch(["p"], ["a", "b"]), "length_mismatch", "answers"),
        (
            lambda d: d.predict_prompt_batch(["p"], ["a"], "words"),
            "unsupported_output_format",
            "output_format",
        ),
        (
            lambda d: d.predict_prompt_batch(["p"], ["a"], min_confidence=2),
            "invalid_min_score",
            "min_confidence",
        ),
    ],
)
def test_predict_argument_errors(call, code, param):
    """The predict* methods raise InvalidInputError with the documented code and param."""
    d = NewStyle()
    with pytest.raises(InvalidInputError) as exc:
        call(d)
    assert exc.value.code == code
    assert exc.value.param == param
    assert d.requests == []


# --- predict-only subclass answers detect ----------------------------------------------------


def test_legacy_detect_sources_and_prompt():
    """The detect method routes sources to predict and a prompt to predict_prompt, and maps the fields."""
    d = LegacyStyle()
    with pytest.warns(LettuceDetectDeprecationWarning):
        by_sources = d.detect(SOURCES)
    by_prompt = d.detect(PROMPT)
    assert d.calls[0] == ("predict", ["s1", "s2"], "abcdef", "q?", "spans", 0.0)
    assert d.calls[1] == ("predict_prompt", "the prompt", "abcdef", "spans", 0.0)
    for result in (by_sources, by_prompt):
        first, second = result.spans
        assert (first.score, first.category, first.reasoning) == (0.7, "c", "r")
        assert second.score == 1.0
        assert [w.code for w in result.warnings] == ["synthetic_score"]
        assert [(u.model_id, u.units) for u in result.usage] == [("legacy-model", {})]


def test_legacy_without_synthetic_span_has_no_warning_entry():
    """A result whose spans all carry a confidence has no synthetic_score warning."""

    class Scored(LegacyStyle):
        spans: ClassVar[list[dict]] = [{"start": 0, "end": 3, "text": "abc", "confidence": 0.7}]

    with pytest.warns(LettuceDetectDeprecationWarning):
        assert Scored().detect(SOURCES).warnings == []


def test_legacy_min_score_filters_after_mapping():
    """min_score applies to the mapped score, so the synthetic 1.0 passes and 0.7 does not."""
    with pytest.warns(LettuceDetectDeprecationWarning):
        out = LegacyStyle().detect(SOURCES, min_score=0.8)
    assert [s.text for s in out.spans] == ["ef"]


def test_legacy_detect_rejects_tokens_and_detail():
    """include_tokens / include_detail raise on a legacy subclass."""
    for kwargs, code in (
        ({"include_tokens": True}, "tokens_not_supported"),
        ({"include_detail": True}, "detail_not_supported"),
    ):
        with pytest.raises(InvalidInputError) as exc:
            LegacyStyle().detect(SOURCES, **kwargs)
        assert exc.value.code == code


def test_legacy_prompt_request_without_predict_prompt_raises_instead_of_recursing():
    """A predict-only class given a prompt request raises, rather than predict_prompt <-> detect looping."""

    class OnlyPredict(BaseDetector):
        def predict(
            self, context, answer, question=None, output_format="tokens", min_confidence=0.0
        ):
            return []

    with pytest.warns(LettuceDetectDeprecationWarning), pytest.raises(InvalidInputError) as exc:
        OnlyPredict().detect(PROMPT)
    assert exc.value.code == "prompt_input_not_supported"


# --- adetect on a legacy subclass ------------------------------------------------------------


async def test_legacy_adetect_both_kinds_equal_detect_and_off_loop_thread():
    """The adetect method works on a predict-only subclass, returns what detect returns, off the loop thread."""
    d = LegacyStyle()
    with pytest.warns(LettuceDetectDeprecationWarning):
        awaited = [await d.adetect(request) for request in (SOURCES, PROMPT)]
    assert d.thread_ids
    assert threading.get_ident() not in d.thread_ids
    assert awaited == [d.detect(SOURCES), d.detect(PROMPT)]


async def test_legacy_warning_once_per_class_across_sync_async_and_concurrent_calls():
    """One warning per class, however many sync, awaited or concurrent calls follow."""

    class Fresh(LegacyStyle):
        pass

    d = Fresh()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d.detect(SOURCES)
        d.detect(PROMPT)
        await d.adetect(SOURCES)
        await asyncio.gather(*(d.adetect(PROMPT) for _ in range(8)))
        Fresh().detect(SOURCES)
    mine = [w for w in caught if issubclass(w.category, LettuceDetectDeprecationWarning)]
    assert len(mine) == 1
    assert "Fresh" in str(mine[0].message)


def test_each_legacy_class_warns_once():
    """Two different legacy classes warn once each."""

    class One(LegacyStyle):
        pass

    class Two(LegacyStyle):
        pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cls in (One, Two, One, Two):
            cls().detect(SOURCES)
    assert len([w for w in caught if issubclass(w.category, LettuceDetectDeprecationWarning)]) == 2


# --- defining subclasses ---------------------------------------------------------------------


def test_subclass_overriding_nothing_raises_type_error():
    """A class that overrides none of detect, _detect, predict fails at class definition."""
    with pytest.raises(TypeError, match="Neither"):

        class Neither(BaseDetector):
            pass


def test_defining_valid_subclasses_emits_no_warning():
    """Defining subclasses that override detect, _detect or predict emits no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class A(BaseDetector):
            def _detect(self, request, *, include_tokens, include_detail):
                raise NotImplementedError

        class B(BaseDetector):
            def detect(self, request, **kwargs):
                raise NotImplementedError

        class C(BaseDetector):
            def predict(self, *args: object, **kwargs: object):
                raise NotImplementedError

        class D(A):
            pass


def test_reset_makes_a_warned_class_warn_again():
    """After the reset helper, a class that already warned warns on its next detect call."""

    class Again(LegacyStyle):
        pass

    with pytest.warns(LettuceDetectDeprecationWarning):
        Again().detect(SOURCES)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Again().detect(SOURCES)
    base_module._reset_legacy_warnings()
    with pytest.warns(LettuceDetectDeprecationWarning):
        Again().detect(SOURCES)


def test_subclass_of_a_warned_class_warns_on_its_own():
    """The flag is per class: a subclass of a warned class still gets its own warning."""

    class Parent(LegacyStyle):
        pass

    class Child(Parent):
        pass

    with pytest.warns(LettuceDetectDeprecationWarning, match="Parent"):
        Parent().detect(SOURCES)
    with pytest.warns(LettuceDetectDeprecationWarning, match="Child"):
        Child().detect(SOURCES)
