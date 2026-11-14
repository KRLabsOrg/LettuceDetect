"""Tests for ``VelaDetector`` and ``VelaConfig``: offline, with a fake SystemOne model."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from lettucedetect.detectors import VelaConfig, VelaDetector
from lettucedetect.detectors.errors import ConfigurationError, InvalidInputError, UpstreamError
from lettucedetect.detectors.types import Chunk, PromptInput, SourcesInput
from tests.detector_contract import DetectorContract

ANSWER = "The capital of France is Berlin, and about 3 million people live there."
SOURCES = SourcesInput(
    sources=[
        Chunk(id="s1", text="Paris is the capital of France."),
        Chunk(id="s2", text="Paris has two million people."),
    ],
    answer=ANSWER,
    question="What is the capital of France?",
)
PROMPT = PromptInput(
    prompt="Answer from the context: Paris is the capital of France.", answer=ANSWER
)

# (word, probability): the fake flags each word found in the answer.
FLAGGED = [("France", 0.3), ("Berlin", 0.9), ("3 million", 0.6)]


class FakeVela:
    """Stand-in for a Vela model: flags fixed words of the answer like ``system_one`` does."""

    def __init__(
        self,
        on_call: Callable[[], None] | None = None,
        input_tokens: int | None = 120,
        reply: dict[str, Any] | None = None,
    ) -> None:
        """Keep the hook, the reported token count and an optional fixed reply."""
        self.on_call = on_call
        self.input_tokens = input_tokens
        self.reply = reply
        self.calls: list[tuple[dict, dict]] = []
        self._active = 0
        self._guard = threading.Lock()
        self.overlapped = False

    def system_one(self, state: dict, questions: dict) -> dict:
        """Answer the span question over ``state["answer"]``."""
        with self._guard:
            self._active += 1
            self.overlapped = self.overlapped or self._active > 1
        try:
            self.calls.append((state, questions))
            if self.on_call:
                self.on_call()
            if self.reply is not None:
                return self.reply
            answer = state["answer"]
            spans = []
            for word, probability in FLAGGED:
                start = answer.find(word)
                if start >= 0:
                    spans.append(
                        {
                            "label": "unsupported",
                            "start": start,
                            "end": start + len(word),
                            "text": word,
                            "probability": probability,
                        }
                    )
            reply: dict[str, Any] = {
                "model": "vllm-sr/Vela-2.0-0.3B",
                "answers": {
                    "halu": {"type": "noul", "noul": max([0.0] + [s["probability"] for s in spans])}
                },
                "spans": {"halu": spans},
                "thresholds": {"halu": 0.55},
            }
            if self.input_tokens is not None:
                reply["usage"] = {"input_tokens": self.input_tokens, "output_tokens": 0}
            return reply
        finally:
            with self._guard:
                self._active -= 1


def _detector(model: FakeVela | None = None, **config: object) -> VelaDetector:
    return VelaDetector(
        model or FakeVela(),
        config=VelaConfig(**config) if config else None,
        model_id="vllm-sr/Vela-2.0-0.3B",
        model_version="abc123",
    )


class TestVelaDetectorContract(DetectorContract):
    """The shared detector contract holds for ``VelaDetector``."""

    sources_request = SOURCES
    prompt_request = PROMPT

    def make_detector(self, on_call=None):  # noqa: D102
        return _detector(FakeVela(on_call=on_call))


# -- config -------------------------------------------------------------------------------


def test_config_defaults_are_velas_documented_question():
    config = VelaConfig()
    assert config.instructions == "Which spans of the answer are not supported by the context?"
    assert config.criteria == {"unsupported": "a claim not supported by the context"}
    assert config.threshold is None
    assert config.max_length == 8192


def test_config_is_frozen():
    with pytest.raises(ValidationError):
        VelaConfig().threshold = 0.5


@pytest.mark.parametrize(
    ("kwargs", "param"),
    [
        ({"threshold": 1.5}, "threshold"),
        ({"threshold": -0.1}, "threshold"),
        ({"max_length": 0}, "max_length"),
        ({"instructions": ""}, "instructions"),
        ({"instructions": "   "}, "instructions"),
        ({"criteria": {}}, "criteria"),
        ({"criteria": {"unsupported": " "}}, "criteria"),
        ({"criteria": {" ": "a claim"}}, "criteria"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_config_errors_carry_code_and_param(kwargs, param):
    with pytest.raises(ConfigurationError) as exc:
        VelaConfig(**kwargs)
    assert exc.value.code == "invalid_config"
    assert exc.value.param == param


# -- request mapping ----------------------------------------------------------------------


def test_sources_with_a_question_are_sent_as_request_source_and_answer():
    model = FakeVela()
    _detector(model).detect(SOURCES)
    state, questions = model.calls[0]
    assert state == {
        "request": "What is the capital of France?",
        "source": "Paris is the capital of France.\n\nParis has two million people.",
        "answer": ANSWER,
    }
    assert questions == {
        "halu": {
            "type": "span",
            "instructions": "Which spans of the answer are not supported by the context?",
            "criteria": {"unsupported": "a claim not supported by the context"},
        }
    }


def test_sources_without_a_question_have_no_request():
    model = FakeVela()
    request = SourcesInput(
        sources=[Chunk(id="s1", text="Ships Friday."), Chunk(id="s2", text=" ")],
        answer="Ships Monday.",
        question="  ",
    )
    _detector(model).detect(request)
    assert model.calls[0][0] == {"source": "Ships Friday.", "answer": "Ships Monday."}


def test_prompt_is_sent_as_the_context():
    model = FakeVela()
    _detector(model).detect(PROMPT)
    assert model.calls[0][0] == {"source": PROMPT.prompt, "answer": ANSWER}


def test_threshold_and_custom_question_are_passed_only_when_set():
    model = FakeVela()
    _detector(
        model,
        instructions="Which claims are made up?",
        criteria={"made_up": "invented"},
        threshold=0.4,
    ).detect(PROMPT)
    assert model.calls[0][1] == {
        "halu": {
            "type": "span",
            "instructions": "Which claims are made up?",
            "criteria": {"made_up": "invented"},
            "threshold": 0.4,
        }
    }


# -- span mapping -------------------------------------------------------------------------


def test_spans_map_offsets_probability_and_answer_text():
    spans = _detector().detect(SOURCES).spans
    assert [(s.start, s.end, s.text, s.score) for s in spans] == [
        (ANSWER.index("France"), ANSWER.index("France") + 6, "France", 0.3),
        (ANSWER.index("Berlin"), ANSWER.index("Berlin") + 6, "Berlin", 0.9),
        (ANSWER.index("3 million"), ANSWER.index("3 million") + 9, "3 million", 0.6),
    ]
    assert all(s.category is None and s.subcategory is None for s in spans)


def test_span_text_comes_from_the_answer_and_numpy_numbers_are_accepted():
    reply = {
        "spans": {
            "halu": [
                {
                    "label": "unsupported",
                    "start": np.int64(25),
                    "end": np.int64(31),
                    "text": "something else",
                    "probability": np.float32(0.75),
                }
            ]
        }
    }
    span = _detector(FakeVela(reply=reply)).detect(SOURCES).spans[0]
    assert (span.start, span.end, span.text) == (25, 31, "Berlin")
    assert span.score == pytest.approx(0.75)
    assert type(span.start) is int


def test_min_score_filters_velas_spans():
    spans = _detector().detect(SOURCES, min_score=0.5).spans
    assert [s.text for s in spans] == ["Berlin", "3 million"]


def _span_item(**overrides: object) -> dict:
    item = {"label": "unsupported", "start": 0, "end": 3, "text": "The", "probability": 0.9}
    return {**item, **overrides}


@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"spans": None},
        {"spans": {}},
        {"spans": {"other": []}},
        {"spans": {"halu": "Berlin"}},
        {"spans": {"halu": ["Berlin"]}},
        {"spans": {"halu": [_span_item(start=None)]}},
        {"spans": {"halu": [_span_item(start=True, end=3)]}},
        {"spans": {"halu": [_span_item(start=3, end=3)]}},
        {"spans": {"halu": [_span_item(start=-1)]}},
        {"spans": {"halu": [_span_item(end=len(ANSWER) + 1)]}},
        {"spans": {"halu": [_span_item(start=0.0)]}},
        {"spans": {"halu": [_span_item(probability=1.2)]}},
        {"spans": {"halu": [_span_item(probability=float("nan"))]}},
        {"spans": {"halu": [_span_item(probability="0.9")]}},
        {"spans": {"halu": []}, "usage": {"input_tokens": -1}},
        {"spans": {"halu": []}, "usage": {"input_tokens": "12"}},
    ],
)
def test_malformed_replies_raise_upstream_error(reply):
    with pytest.raises(UpstreamError) as exc:
        _detector(FakeVela(reply=reply)).detect(SOURCES)
    assert exc.value.code == "invalid_model_output"
    assert exc.value.service == "vela"
    assert ANSWER not in exc.value.message


def test_a_reply_that_is_not_an_object_raises_upstream_error():
    with pytest.raises(UpstreamError) as exc:
        _detector(FakeVela(reply=["not", "a", "dict"])).detect(SOURCES)  # type: ignore[arg-type]
    assert exc.value.code == "invalid_model_output"


# -- blank input --------------------------------------------------------------------------


@pytest.mark.parametrize("answer", ["", "   \n"])
def test_blank_answer_returns_no_spans_without_calling_the_model(answer):
    model = FakeVela()
    result = _detector(model).detect(SourcesInput(sources=SOURCES.sources, answer=answer))
    assert model.calls == []
    assert result.spans == []
    assert result.usage[0].units == {"answer_chars": len(answer), "sources": 2, "source_chars": 60}


@pytest.mark.parametrize(
    "request_",
    [
        SourcesInput(sources=[], answer=ANSWER),
        SourcesInput(sources=[Chunk(id="s1", text="  ")], answer=ANSWER),
        PromptInput(prompt=" \n ", answer=ANSWER),
    ],
)
def test_blank_context_raises_invalid_input_error(request_):
    model = FakeVela()
    with pytest.raises(InvalidInputError) as exc:
        _detector(model).detect(request_)
    assert (exc.value.code, exc.value.param) == ("empty_context", "request")
    assert model.calls == []


@pytest.mark.parametrize("sources", [[], [Chunk(id="s1", text="  ")]])
def test_question_without_sources_is_sent_as_request_and_answer(sources):
    model = FakeVela()
    _detector(model).detect(SourcesInput(sources=sources, answer=ANSWER, question="Where?"))
    assert model.calls[0][0] == {"request": "Where?", "answer": ANSWER}


class _SystemOneError(ValueError):
    """Stand-in for Vela's ``SystemOneError``, which carries a ``type``."""

    def __init__(self, type_: str) -> None:
        super().__init__("questions: the questions alone do not fit the 8192-token window")
        self.type = type_


def _raise(error: Exception) -> Callable[[], None]:
    def hook() -> None:
        raise error

    return hook


def test_input_that_does_not_fit_raises_invalid_input_error():
    error = _SystemOneError("too_long")
    with pytest.raises(InvalidInputError) as exc:
        _detector(FakeVela(on_call=_raise(error))).detect(SOURCES)
    assert (exc.value.code, exc.value.param) == ("input_too_long", "request")
    assert exc.value.__cause__ is error


def test_other_model_errors_are_not_translated():
    with pytest.raises(_SystemOneError):
        _detector(FakeVela(on_call=_raise(_SystemOneError("value_error")))).detect(SOURCES)


# -- usage and warnings -------------------------------------------------------------------


def test_usage_units_for_sources():
    usage = _detector().detect(SOURCES).usage
    assert len(usage) == 1
    assert usage[0].model_id == "vllm-sr/Vela-2.0-0.3B"
    assert usage[0].model_version == "abc123"
    assert usage[0].units == {
        "answer_chars": len(ANSWER),
        "sources": 2,
        "source_chars": 60,
        "input_tokens": 120,
    }


def test_usage_units_for_prompt_have_no_sources():
    units = _detector().detect(PROMPT).usage[0].units
    assert units == {
        "answer_chars": len(ANSWER),
        "source_chars": len(PROMPT.prompt),
        "input_tokens": 120,
    }


def test_usage_has_no_input_tokens_when_the_model_does_not_report_them():
    units = _detector(FakeVela(input_tokens=None)).detect(PROMPT).usage[0].units
    assert "input_tokens" not in units


@pytest.mark.parametrize(("input_tokens", "truncated"), [(8191, False), (8192, True), (9500, True)])
def test_input_truncated_when_the_model_used_its_whole_budget(input_tokens, truncated, caplog):
    result = _detector(FakeVela(input_tokens=input_tokens)).detect(SOURCES)
    assert [w.code for w in result.warnings] == (["input_truncated"] if truncated else [])
    records = [r for r in caplog.records if r.getMessage().startswith("input_truncated")]
    assert len(records) == (1 if truncated else 0)
    assert ANSWER not in caplog.text


def test_input_truncated_follows_the_configured_budget():
    result = _detector(FakeVela(input_tokens=600), max_length=512).detect(SOURCES)
    assert [w.code for w in result.warnings] == ["input_truncated"]


def test_no_warning_when_the_model_does_not_report_input_tokens():
    assert _detector(FakeVela(input_tokens=None)).detect(SOURCES).warnings == []


# -- side effects and concurrency ---------------------------------------------------------


class _RecordingEnviron(dict):
    """A stand-in for ``os.environ`` that records every key read."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.read: list[str] = []

    def __getitem__(self, key: str) -> str:
        self.read.append(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        self.read.append(key)
        return super().get(key, default)

    def __contains__(self, key: object) -> bool:
        self.read.append(str(key))
        return super().__contains__(key)


def test_a_call_reads_no_environment_variable(monkeypatch):
    detector = _detector()
    environ = _RecordingEnviron(os.environ)
    monkeypatch.setattr(os, "environ", environ)
    detector.detect(SOURCES)
    detector.detect(PROMPT)
    assert environ.read == []


async def test_concurrent_detect_and_adetect_calls_match_sequential_results():
    model = FakeVela(on_call=lambda: time.sleep(0.002))
    detector = _detector(model)
    requests = []
    for i in range(12):
        answer = ["Berlin it is.", "About 3 million.", "France, Berlin, 3 million.", "Nothing."][
            i % 4
        ]
        if i % 2:
            requests.append(PromptInput(prompt=f"Context number {i}.", answer=answer))
        else:
            requests.append(
                SourcesInput(sources=[Chunk(id=str(i), text=f"Source {i}.")], answer=answer)
            )
    expected = [detector.detect(r) for r in requests]
    loop = asyncio.get_running_loop()
    sync_results = [loop.run_in_executor(None, detector.detect, r) for r in requests[::2]]
    async_results = [detector.adetect(r) for r in requests[1::2]]
    results = await asyncio.gather(*sync_results, *async_results)
    assert list(results) == expected[::2] + expected[1::2]
    assert not model.overlapped


def test_same_input_gives_the_same_result_after_other_input():
    detector = _detector()
    first = detector.detect(SOURCES)
    detector.detect(PromptInput(prompt="Something else entirely.", answer="Berlin."))
    assert detector.detect(SOURCES) == first


@pytest.mark.parametrize(
    ("flag", "code"),
    [("include_tokens", "tokens_not_supported"), ("include_detail", "detail_not_supported")],
)
def test_tokens_and_detail_are_not_supported(flag, code):
    model = FakeVela()
    with pytest.raises(InvalidInputError) as exc:
        _detector(model).detect(SOURCES, **{flag: True})
    assert (exc.value.code, exc.value.param) == (code, flag)
    assert model.calls == []


def test_predict_returns_legacy_spans():
    spans = _detector().predict(
        ["Paris is the capital of France."],
        ANSWER,
        "What?",
        output_format="spans",
        min_confidence=0.5,
    )
    assert [s["text"] for s in spans] == ["Berlin", "3 million"]
    assert spans[0]["confidence"] == 0.9
