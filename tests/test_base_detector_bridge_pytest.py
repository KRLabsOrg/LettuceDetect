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


# --- detect goes through a predict* override of a detect-capable class -------------------------


class FiltersShort(NewStyle):
    """Overrides predict below the class that implements _detect, and filters its spans."""

    def predict(self, context, answer, question=None, output_format="tokens", min_confidence=0.0):  # noqa: D102
        spans = super().predict(context, answer, question, output_format, min_confidence)
        return [span for span in spans if len(span["text"]) < 3]


def test_detect_goes_through_an_overridden_predict():
    """A predict override below _detect also decides what detect returns for sources."""
    d = FiltersShort()
    with pytest.warns(LettuceDetectDeprecationWarning, match="FiltersShort"):
        result = d.detect(SOURCES)
    assert [s.text for s in result.spans] == ["ef"]
    assert [s["text"] for s in d.predict(["s1", "s2"], "abcdef", "q?", "spans")] == ["ef"]
    assert [(u.model_id, u.units) for u in result.usage] == [(None, {})]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert [s.text for s in d.detect(SOURCES, min_score=0.5).spans] == []
        # predict_prompt is not overridden, so a prompt request is answered by _detect.
        assert [s.text for s in d.detect(PROMPT).spans] == ["abc", "ef"]


def test_detect_goes_through_an_overridden_predict_prompt():
    """A predict_prompt override answers a prompt request; sources stay on _detect."""

    class PromptOnly(NewStyle):
        def predict_prompt(self, prompt, answer, output_format="tokens", min_confidence=0.0):
            return [{"start": 0, "end": 1, "text": "a", "confidence": 0.5}]

    d = PromptOnly()
    with pytest.warns(LettuceDetectDeprecationWarning, match="PromptOnly"):
        assert [s.text for s in d.detect(PROMPT).spans] == ["a"]
    assert [s.text for s in d.detect(SOURCES).spans] == ["abc", "ef"]


def test_tokens_request_on_a_predict_override_is_answered_by_detect():
    """A tokens request keeps the _detect result, because predict returns no token offsets."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = FiltersShort().detect(SOURCES, include_tokens=True)
    assert result.tokens
    assert [s.text for s in result.spans] == ["abc", "ef"]


def test_predict_override_that_calls_detect_does_not_recurse():
    """An override built on detect reaches _detect instead of itself."""

    class OnDetect(NewStyle):
        def predict(
            self, context, answer, question=None, output_format="tokens", min_confidence=0.0
        ):
            spans = self.detect(SOURCES).spans
            return [{"start": s.start, "end": s.end, "text": s.text} for s in spans[:1]]

    d = OnDetect()
    with pytest.warns(LettuceDetectDeprecationWarning):
        assert [s.text for s in d.detect(SOURCES).spans] == ["abc"]
    assert [s.text for s in d.detect(SOURCES).spans] == ["abc"]
    assert d.predict(["s1"], "abcdef") == [{"start": 0, "end": 3, "text": "abc"}]


def test_class_that_defines_detect_and_predict_together_is_not_bridged():
    """A predict defined beside _detect is an adapter over it, so detect stays on _detect."""

    class Both(BaseDetector):
        def _detect(self, request, *, include_tokens, include_detail):
            return Detection(spans=[DetectedSpan(start=0, end=3, text="abc", score=0.8)])

        def predict(
            self, context, answer, question=None, output_format="tokens", min_confidence=0.0
        ):
            raise AssertionError("detect must not call predict")

    class Child(Both):
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for cls in (Both, Child):
            assert [s.text for s in cls().detect(SOURCES).spans] == ["abc"]


def test_none_of_the_library_detectors_is_bridged():
    """None of the library's own detectors answers detect through its predict* methods."""
    from lettucedetect.detectors import (
        EncoderDetector,
        GenerativeDetector,
        LLMDetector,
        LLMJudgeDetector,
        RAGFactCheckerDetector,
        TransformerDetector,
        VelaDetector,
    )

    for cls in (
        EncoderDetector,
        TransformerDetector,
        LLMJudgeDetector,
        GenerativeDetector,
        LLMDetector,
        RAGFactCheckerDetector,
        VelaDetector,
    ):
        detector = object.__new__(cls)  # no constructor: only the class layout matters
        for request in (SOURCES, PROMPT):
            assert not detector._answers_through_predict(request, False, False), cls.__name__


def test_transformer_subclass_with_a_predict_override_gives_the_same_spans_from_detect(
    local_wordpiece_tokenizer,
):
    """A TransformerDetector subclass whose predict post-filters spans is not bypassed by detect."""
    import torch

    from lettucedetect.detectors.transformer import TransformerDetector

    def forward(input_ids=None, attention_mask=None, **kwargs):
        score = ((input_ids * 7) % 11 - 5).float() / 2
        return type("Output", (), {"logits": torch.stack([-score, score], dim=-1)})()

    with warnings.catch_warnings():
        # Subclassing and building the deprecated class warn; that is not what is tested here.
        warnings.simplefilter("ignore", LettuceDetectDeprecationWarning)

        class ShortSpansOnly(TransformerDetector):
            def predict(
                self, context, answer, question=None, output_format="tokens", min_confidence=0.0
            ):
                spans = super().predict(context, answer, question, output_format, min_confidence)
                return [span for span in spans if len(span["text"]) <= 5]

        plain = TransformerDetector.__new__(TransformerDetector)
        filtering = ShortSpansOnly.__new__(ShortSpansOnly)
    for detector in (plain, filtering):
        detector.model, detector.tokenizer = forward, local_wordpiece_tokenizer
        detector.device, detector.max_length, detector.lang = torch.device("cpu"), 512, "en"
    context = ["the capital of france is paris ."]
    answer = "paris is the capital of france and the answer is a short word"
    request = SourcesInput(sources=[Chunk(id="0", text=context[0])], answer=answer)

    predicted = filtering.predict(context, answer, output_format="spans")
    unfiltered = plain.predict(context, answer, output_format="spans")
    assert predicted
    assert len(predicted) < len(unfiltered), "the override must really drop a span"
    with pytest.warns(LettuceDetectDeprecationWarning, match="ShortSpansOnly"):
        detected = filtering.detect(request)
    assert [(s.start, s.end, s.text, s.score) for s in detected.spans] == [
        (s["start"], s["end"], s["text"], s["confidence"]) for s in predicted
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert [(s.start, s.end) for s in plain.detect(request).spans] == [
            (s["start"], s["end"]) for s in unfiltered
        ]
