"""Run the detector contract suite against stand-ins, good and deliberately broken."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.types import (
    Chunk,
    DetectedSpan,
    Detection,
    PromptInput,
    SourcesInput,
    Usage,
)
from tests.detector_contract import CONTENT_LOGGER, DetectorContract, run_check

ANSWER = "The capital of France is Berlin, and about 3 million people live there."
SOURCES = SourcesInput(
    sources=[Chunk(id="s1", text="Paris is the capital of France and has two million people.")],
    answer=ANSWER,
    question="What is the capital of France?",
)
PROMPT = PromptInput(
    prompt="Answer from the context: Paris is the capital of France.", answer=ANSWER
)

# (word, score): three flagged words with three different scores.
FLAGGED = [("France", 0.3), ("3 million", 0.6), ("Berlin", 0.9)]


def _raw_spans(answer: str, shift: int = 0) -> list[dict]:
    spans = []
    for word, score in FLAGGED:
        start = answer.index(word)
        spans.append(
            {
                "start": start + shift,
                "end": start + shift + len(word),
                "text": word,
                "confidence": score,
            }
        )
    return sorted(spans, key=lambda s: s["start"])


def _span(raw: dict) -> DetectedSpan:
    return DetectedSpan(
        start=raw["start"], end=raw["end"], text=raw["text"], score=raw["confidence"]
    )


class DetectOnly(BaseDetector):
    """Stand-in that implements only the new ``_detect`` hook."""

    model_id = "stand-in"

    def __init__(self, on_call: Callable[[], None] | None = None) -> None:
        """Keep the hook the suite wants called."""
        self.on_call = on_call

    def _spans(self, answer: str) -> list[DetectedSpan]:
        return [_span(s) for s in _raw_spans(answer)]

    def _detect(self, request, *, include_tokens, include_detail):
        if self.on_call:
            self.on_call()
        return Detection(
            spans=self._spans(request.answer),
            usage=[Usage(model_id=self.model_id, model_version="1", units={})],
        )


class PredictOnly(BaseDetector):
    """Old-style stand-in that implements only the legacy ``predict*`` methods."""

    model_id = "legacy-stand-in"

    def __init__(self, on_call: Callable[[], None] | None = None) -> None:
        """Keep the hook the suite wants called."""
        self.on_call = on_call

    def predict(self, context, answer, question=None, output_format="tokens", min_confidence=0.0):  # noqa: D102
        if self.on_call:
            self.on_call()
        return _raw_spans(answer)

    def predict_prompt(self, prompt, answer, output_format="tokens", min_confidence=0.0):  # noqa: D102
        if self.on_call:
            self.on_call()
        return _raw_spans(answer)


class TestDetectOnlyContract(DetectorContract):
    """The contract holds for a detector that implements ``detect`` only."""

    sources_request = SOURCES
    prompt_request = PROMPT

    def make_detector(self, on_call=None):  # noqa: D102
        return DetectOnly(on_call)


@pytest.mark.filterwarnings(
    "ignore::lettucedetect.detectors._deprecation.LettuceDetectDeprecationWarning"
)
class TestPredictOnlyContract(DetectorContract):
    """The contract holds for an old-style detector that implements ``predict`` only."""

    sources_request = SOURCES
    prompt_request = PROMPT

    def make_detector(self, on_call=None):  # noqa: D102
        return PredictOnly(on_call)


# -- deliberately broken stand-ins: each must fail the matching check ---------------------------


class WrongOffsets(DetectOnly):
    """Spans point one character to the right of their text."""

    def _spans(self, answer):
        return [_span(s) for s in _raw_spans(answer, shift=1)]


class Prints(DetectOnly):
    """Prints the answer."""

    def _detect(self, request, *, include_tokens, include_detail):
        print("checking", request.answer)
        return super()._detect(
            request, include_tokens=include_tokens, include_detail=include_detail
        )


class LogsAnswer(DetectOnly):
    """Logs the answer on an ordinary module logger."""

    def _detect(self, request, *, include_tokens, include_detail):
        logging.getLogger("lettucedetect.detectors.broken").info("answer: %s", request.answer)
        return super()._detect(
            request, include_tokens=include_tokens, include_detail=include_detail
        )


class LogsAnswerOnContentLogger(DetectOnly):
    """Logs the answer on the one logger that may carry it; this one is allowed."""

    def _detect(self, request, *, include_tokens, include_detail):
        logging.getLogger(CONTENT_LOGGER).debug("answer: %s", request.answer)
        return super()._detect(
            request, include_tokens=include_tokens, include_detail=include_detail
        )


class WritesFileToCwd(DetectOnly):
    """Writes a file in the working directory."""

    def _detect(self, request, *, include_tokens, include_detail):
        Path("scratch.txt").write_text("x")
        return super()._detect(
            request, include_tokens=include_tokens, include_detail=include_detail
        )


class WritesFileToHome(DetectOnly):
    """Writes a cache file below HOME."""

    def _detect(self, request, *, include_tokens, include_detail):
        (Path.home() / ".cache-of-broken-detector").write_text("x")
        return super()._detect(
            request, include_tokens=include_tokens, include_detail=include_detail
        )


class BlocksTheLoop(DetectOnly):
    """Its ``adetect`` runs the detection on the event loop's own thread."""

    async def adetect(self, request, *, min_score=0.0, include_tokens=False, include_detail=False):  # noqa: D102
        return self.detect(
            request,
            min_score=min_score,
            include_tokens=include_tokens,
            include_detail=include_detail,
        )


class AdetectDiffers(DetectOnly):
    """Its ``adetect`` ignores ``min_score``."""

    async def adetect(self, request, *, min_score=0.0, include_tokens=False, include_detail=False):  # noqa: D102
        return await super().adetect(
            request, min_score=0.0, include_tokens=include_tokens, include_detail=include_detail
        )


class Unrepeatable(DetectOnly):
    """Gives a different score on every call."""

    calls = 0

    def _detect(self, request, *, include_tokens, include_detail):
        type(self).calls += 1
        result = super()._detect(
            request, include_tokens=include_tokens, include_detail=include_detail
        )
        bump = 0.001 * type(self).calls
        spans = [s.model_copy(update={"score": min(1.0, s.score / 2 + bump)}) for s in result.spans]
        return result.model_copy(update={"spans": spans})


class WrongModelId(DetectOnly):
    """Reports usage for a model other than its own."""

    def _detect(self, request, *, include_tokens, include_detail):
        result = super()._detect(
            request, include_tokens=include_tokens, include_detail=include_detail
        )
        return result.model_copy(
            update={"usage": [Usage(model_id="someone-else", model_version=None, units={})]}
        )


class BrokenContract(DetectorContract):
    """A contract bound to one broken stand-in."""

    sources_request = SOURCES
    prompt_request = PROMPT
    detector_class: type[BaseDetector]

    def make_detector(self, on_call=None):  # noqa: D102
        return self.detector_class(on_call)


def _contract_for(detector_class: type[BaseDetector]) -> BrokenContract:
    contract = BrokenContract()
    contract.detector_class = detector_class
    return contract


BROKEN = [
    (WrongOffsets, "test_span_text_equals_answer_between_offsets"),
    (Prints, "test_a_call_prints_nothing_and_logs_no_text_outside_the_content_logger"),
    (LogsAnswer, "test_a_call_prints_nothing_and_logs_no_text_outside_the_content_logger"),
    (WritesFileToCwd, "test_a_call_leaves_an_empty_cwd_and_home_empty"),
    (WritesFileToHome, "test_a_call_leaves_an_empty_cwd_and_home_empty"),
    (BlocksTheLoop, "test_adetect_runs_off_the_loop_thread_and_keeps_the_loop_free"),
    (AdetectDiffers, "test_adetect_equals_detect"),
    (Unrepeatable, "test_same_input_gives_the_same_result_also_after_other_input"),
    (WrongModelId, "test_usage_has_an_entry_with_the_model_id_and_a_model_version_field"),
]


@pytest.mark.parametrize(
    ("detector_class", "check"), BROKEN, ids=lambda v: getattr(v, "__name__", v)
)
def test_broken_stand_in_fails_the_matching_check(detector_class, check):
    """A deliberately broken stand-in fails the check that is about its defect."""
    with pytest.raises(AssertionError):
        run_check(_contract_for(detector_class), check)


def test_logging_on_the_content_logger_is_allowed():
    """A detector that logs the answer on lettucedetect.content passes the logging check."""
    run_check(
        _contract_for(LogsAnswerOnContentLogger),
        "test_a_call_prints_nothing_and_logs_no_text_outside_the_content_logger",
    )


def test_every_check_of_the_contract_is_covered_by_a_stand_in_in_this_file():
    """Each contract check is run by both good stand-ins; the list of checks is not empty."""
    checks = [n for n, _ in inspect.getmembers(DetectorContract) if n.startswith("test_")]
    assert len(checks) >= 12
    for suite in (TestDetectOnlyContract, TestPredictOnlyContract):
        for name in checks:
            assert hasattr(suite, name)


async def test_cancelled_thread_keeps_running_until_it_finishes():
    """After Task.cancel() the worker thread is still inside the detector, and it finishes once released."""
    import threading

    entered, gate, finished = threading.Event(), threading.Event(), threading.Event()

    def blocked() -> None:
        entered.set()
        gate.wait(10)
        finished.set()

    task = asyncio.create_task(DetectOnly(blocked).adetect(SOURCES))
    while not entered.is_set():
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.05)
    assert not finished.is_set()  # the thread was not stopped by the cancel
    gate.set()
    deadline = time.monotonic() + 10
    while not finished.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
    assert finished.is_set()
