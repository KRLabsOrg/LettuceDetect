"""Tests for BaseDetector.detect and BaseDetector.adetect."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time

import pytest

from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.errors import InvalidInputError
from lettucedetect.detectors.types import (
    DetectedSpan,
    Detection,
    SourcesInput,
    TokenScore,
)

REQUEST = SourcesInput(sources=[], answer="abcdefghij")


def _result() -> Detection:
    return Detection(
        spans=[
            DetectedSpan(start=0, end=2, text="ab", score=0.2),
            DetectedSpan(start=3, end=5, text="de", score=0.5),
            DetectedSpan(start=6, end=8, text="gh", score=0.9),
        ],
        tokens=[TokenScore(token="ab", start=0, end=2, score=0.2, flagged=False)],  # noqa: S106
    )


class FakeDetector(BaseDetector):
    """Detector with a recording, configurable hook."""

    supports_tokens = True
    supports_detail = True

    def __init__(self, result=None, delay=0.0, error=None) -> None:
        """Create the fake."""
        self.result = result or _result()
        self.delay = delay
        self.error = error
        self.calls = []
        self.thread_ids = []

    def predict(self, *a: object, **k: object) -> list:  # noqa: D102
        return []

    def predict_prompt(self, *a: object, **k: object) -> list:  # noqa: D102
        return []

    def predict_prompt_batch(self, *a: object, **k: object) -> list:  # noqa: D102
        return []

    def _detect(self, request, *, include_tokens, include_detail):
        self.calls.append((request, include_tokens, include_detail))
        self.thread_ids.append(threading.get_ident())
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


class PlainDetector(FakeDetector):
    """Detector without token or detail support."""

    supports_tokens = False
    supports_detail = False


def test_detect_returns_hook_result_and_defaults():
    """The detect method returns what the hook returns; defaults pass no flags to the hook; model_id is None."""
    d = FakeDetector()
    assert d.detect(REQUEST) == _result()
    assert d.calls == [(REQUEST, False, False)]
    assert d.model_id is None


def test_flags_are_passed_to_hook_when_supported():
    """include_tokens / include_detail reach the hook when the detector supports them."""
    d = FakeDetector()
    d.detect(REQUEST, include_tokens=True, include_detail=True)
    assert d.calls == [(REQUEST, True, True)]


def test_min_score_filters_spans_not_tokens():
    """Spans scoring below min_score are dropped (equal is kept); tokens are not filtered."""
    out = FakeDetector().detect(REQUEST, min_score=0.5)
    assert [s.score for s in out.spans] == [0.5, 0.9]
    assert out.tokens == _result().tokens


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
@pytest.mark.parametrize("is_async", [False, True])
async def test_invalid_min_score(value, is_async):
    """min_score outside [0, 1] (or NaN) raises invalid_min_score before the hook runs, from detect and from await adetect."""
    d = FakeDetector()
    with pytest.raises(InvalidInputError) as exc:
        if is_async:
            await d.adetect(REQUEST, min_score=value)
        else:
            d.detect(REQUEST, min_score=value)
    assert exc.value.code == "invalid_min_score"
    assert exc.value.param == "min_score"
    assert d.calls == []


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_min_score_boundaries_accepted(value):
    """The bounds 0.0 and 1.0 are valid min_score values."""
    FakeDetector().detect(REQUEST, min_score=value)


@pytest.mark.parametrize(
    ("kwargs", "code", "param"),
    [
        ({"include_tokens": True}, "tokens_not_supported", "include_tokens"),
        ({"include_detail": True}, "detail_not_supported", "include_detail"),
    ],
)
@pytest.mark.parametrize("is_async", [False, True])
async def test_unsupported_flags(kwargs, code, param, is_async):
    """Unsupported include_tokens / include_detail raise their own code and param before the hook runs, sync and async."""
    d = PlainDetector()
    with pytest.raises(InvalidInputError) as exc:
        if is_async:
            await d.adetect(REQUEST, **kwargs)
        else:
            d.detect(REQUEST, **kwargs)
    assert exc.value.code == code
    assert exc.value.param == param
    assert d.calls == []


async def test_adetect_is_coroutine_function_and_equal_result():
    """The adetect method is a coroutine function and returns a result equal to detect."""
    assert inspect.iscoroutinefunction(BaseDetector.adetect)
    d = FakeDetector()
    assert await d.adetect(REQUEST, min_score=0.5) == d.detect(REQUEST, min_score=0.5)


async def test_hook_runs_off_the_loop_thread_and_loop_stays_responsive():
    """The hook runs on a thread other than the event loop's, and the loop keeps ticking while a slow hook sleeps."""
    d = FakeDetector(delay=0.3)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    task = asyncio.create_task(ticker())
    await d.adetect(REQUEST)
    task.cancel()
    assert d.thread_ids[0] != threading.get_ident()
    assert ticks >= 5


async def test_gather_keeps_call_order():
    """Concurrent adetect calls finish out of order (different delays) but gather returns them in call order, each equal to its detect result."""
    detectors = [FakeDetector(delay=0.05 * (3 - i)) for i in range(3)]
    scores = [0.1, 0.4, 0.7]
    results = await asyncio.gather(
        *(d.adetect(REQUEST, min_score=s) for d, s in zip(detectors, scores, strict=True))
    )
    for d, s, r in zip(detectors, scores, results, strict=True):
        assert r == d.detect(REQUEST, min_score=s)


@pytest.mark.parametrize("is_async", [False, True])
async def test_hook_exception_propagates_unwrapped(is_async):
    """An exception from the hook is raised as the very same object, with __cause__ intact, from detect and await adetect."""
    cause = RuntimeError("root")
    err = KeyError("boom")
    err.__cause__ = cause
    d = FakeDetector(error=err)
    with pytest.raises(KeyError) as exc:
        if is_async:
            await d.adetect(REQUEST)
        else:
            d.detect(REQUEST)
    assert exc.value is err
    assert exc.value.__cause__ is cause


def test_adetect_docstring_states_the_caveats():
    """The adetect docstring states the default executor and its size, that it is shared, that cancelling stops the wait only, and that there is no executor argument."""
    doc = BaseDetector.adetect.__doc__
    for text in ("default executor", "min(32", "shared", "Cancelling", "no ``executor`` argument"):
        assert doc is not None
        assert text in doc


def test_default_hook_raises_not_implemented():
    """The base class itself, which has no hook and no legacy predict, raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        BaseDetector().detect(REQUEST)
