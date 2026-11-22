"""Tests for ``detect_batch`` and ``adetect_batch``: the result type, order and concurrency."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

import lettucedetect.detectors as detectors
from lettucedetect.detectors import (
    BatchResult,
    GenerativeConfig,
    GenerativeDetector,
    LLMDetector,
    LLMJudgeConfig,
    LLMJudgeDetector,
    RAGFactCheckerDetector,
)
from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.errors import InvalidInputError, UpstreamError
from lettucedetect.detectors.types import DetectedSpan, Detection, PromptInput
from tests.test_generative_detector_pytest import _numbered as _numbered_generative
from tests.test_llm_judge_detector_pytest import FakeClient
from tests.test_llm_judge_detector_pytest import _numbered as _numbered_judge
from tests.test_rag_fact_checker_detector_pytest import FakeFactChecker

BATCH_THREAD = "lettucedetect-batch"


def _words(results: list[BatchResult]) -> list[list[str]]:
    return [[span.text for span in result.get_or_raise().spans] for result in results]


def _expected_words(count: int) -> list[list[str]]:
    return [[f"word-{chr(ord('a') + i)}"] for i in range(count)]


def _judge(count: int, max_concurrency: int, on_call=None):
    requests, replies = _numbered_judge(count)
    client = FakeClient(replies, on_call)
    config = LLMJudgeConfig(max_concurrency=max_concurrency)
    return LLMJudgeDetector(client, "judge-model", config=config), client, requests


def _generative(count: int, max_concurrency: int, on_call=None):
    requests, replies = _numbered_generative(count)
    client = FakeClient(replies, on_call)
    config = GenerativeConfig(max_concurrency=max_concurrency)
    return GenerativeDetector(client, "generative-model", config=config), client, requests


LLM_BACKED = pytest.mark.parametrize("build", [_judge, _generative], ids=["judge", "generative"])


class Echo(BaseDetector):
    """Flags the first word of the answer; sleeps longer for earlier requests when asked to."""

    def __init__(self, max_concurrency: int | None = None, delays: dict[str, float] | None = None):
        """Keep the limit, when one is given, and the delay of each answer."""
        if max_concurrency is not None:
            self.max_concurrency = max_concurrency
        self.delays = delays or {}
        self.threads: list[int] = []
        self.order: list[str] = []

    def _detect(self, request, *, include_tokens, include_detail):
        self.threads.append(threading.get_ident())
        time.sleep(self.delays.get(request.answer, 0.0))
        self.order.append(request.answer)
        if request.answer.startswith("fail"):
            raise UpstreamError("the stub failed", code="upstream_unavailable", service="stub")
        if request.answer.startswith("bug"):
            raise KeyError("a bug")
        word = request.answer.split()[0]
        return Detection(spans=[DetectedSpan(start=0, end=len(word), text=word, score=0.5)])


def _prompts(*answers: str) -> list[PromptInput]:
    return [PromptInput(prompt="p", answer=answer) for answer in answers]


# -- BatchResult ------------------------------------------------------------------------------


def test_batch_result_holds_a_detection_or_an_error():
    detection = Detection()
    error = UpstreamError("failed", code="upstream_unavailable", service="stub")
    good, bad = BatchResult(detection=detection), BatchResult(error=error)
    assert (good.ok, good.detection, good.error) == (True, detection, None)
    assert good.get_or_raise() is detection
    assert (bad.ok, bad.detection, bad.error) == (False, None, error)
    with pytest.raises(UpstreamError) as exc:
        bad.get_or_raise()
    assert exc.value is error
    assert detectors.BatchResult is BatchResult
    assert "BatchResult" in detectors.__all__


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"detection": Detection(), "error": InvalidInputError("bad")},
        {"error": ValueError("not a library error")},
    ],
)
def test_batch_result_needs_exactly_one_of_detection_and_typed_error(fields):
    with pytest.raises(ValidationError):
        BatchResult(**fields)


def test_batch_result_is_frozen():
    result = BatchResult(detection=Detection())
    with pytest.raises(ValidationError):
        result.detection = None
    with pytest.raises(ValidationError):
        result.ok = False


# -- the default implementation ---------------------------------------------------------------


def test_default_runs_requests_one_after_the_other_on_the_calling_thread():
    """Without a ``max_concurrency`` of its own a detector is not called concurrently."""
    detector = Echo()
    requests = _prompts("one a", "two b", "three c")
    assert BaseDetector.max_concurrency == 1
    assert _words(detector.detect_batch(requests)) == [["one"], ["two"], ["three"]]
    assert detector.order == ["one a", "two b", "three c"]
    assert set(detector.threads) == {threading.get_ident()}


def test_concurrent_requests_finish_out_of_order_and_return_in_input_order():
    answers = ["one a", "two b", "three c", "four d"]
    delays = dict(zip(answers, [0.2, 0.1, 0.05, 0.0], strict=True))
    detector = Echo(max_concurrency=4, delays=delays)
    results = detector.detect_batch(_prompts(*answers))
    assert _words(results) == [["one"], ["two"], ["three"], ["four"]]
    assert detector.order == answers[::-1]
    assert len(set(detector.threads)) == 4


@pytest.mark.parametrize("max_concurrency", [1, 4])
def test_failures_keep_their_position_and_min_score_applies_to_the_rest(max_concurrency):
    detector = Echo(max_concurrency=max_concurrency)
    results = detector.detect_batch(_prompts("one a", "fail b", "three c", "fail d"))
    assert [result.ok for result in results] == [True, False, True, False]
    assert [result.error.code for result in results if not result.ok] == [
        "upstream_unavailable"
    ] * 2
    assert results[1].error is not results[3].error
    filtered = detector.detect_batch(_prompts("one a", "fail b"), min_score=0.6)
    assert filtered[0].get_or_raise().spans == []
    assert not filtered[1].ok


@pytest.mark.parametrize("max_concurrency", [1, 4])
async def test_bug_in_one_request_fails_the_whole_batch(max_concurrency):
    detector = Echo(max_concurrency=max_concurrency)
    requests = _prompts("one a", "bug b", "fail c")
    with pytest.raises(KeyError):
        detector.detect_batch(requests)
    with pytest.raises(KeyError):
        await detector.adetect_batch(requests)


def test_invalid_item_is_named_by_position_and_nothing_runs():
    detector = Echo()
    secret = "customer text that must stay out of the message"  # noqa: S105
    with pytest.raises(InvalidInputError) as exc:
        detector.detect_batch([*_prompts("one a", "two b"), {"prompt": "p", "answer": secret}])
    assert (exc.value.code, exc.value.param) == ("invalid_request", "requests")
    assert "requests[2]" in str(exc.value)
    assert secret not in str(exc.value)
    assert detector.order == []


# -- max_concurrency on the LLM-backed detectors ------------------------------------------------


@LLM_BACKED
def test_many_more_requests_than_max_concurrency_complete_in_order(build):
    """100 requests with ``max_concurrency=2``: every result is right and nothing deadlocks."""
    detector, client, requests = build(26, 2, lambda: time.sleep(0.002))
    batch = (requests * 4)[:100]
    results = detector.detect_batch(batch)
    assert _words(results) == (_expected_words(26) * 4)[:100]
    assert len(client.calls) == 100
    assert client.max_in_flight == 2


@LLM_BACKED
async def test_max_concurrency_bounds_client_calls_of_a_batch(build):
    detector, client, requests = build(12, 3, lambda: time.sleep(0.03))
    assert _words(detector.detect_batch(requests)) == _expected_words(12)
    assert client.max_in_flight == 3
    assert _words(await detector.adetect_batch(requests)) == _expected_words(12)
    assert client.max_in_flight == 3


@LLM_BACKED
async def test_max_concurrency_bounds_two_gathered_batches_together(build):
    detector, client, requests = build(12, 2, lambda: time.sleep(0.03))
    first, second = await asyncio.gather(
        detector.adetect_batch(requests[:6]), detector.adetect_batch(requests[6:])
    )
    assert _words(first) + _words(second) == _expected_words(12)
    assert client.max_in_flight == 2
    assert len(client.calls) == 12


@LLM_BACKED
async def test_max_concurrency_bounds_a_batch_next_to_single_calls(build):
    detector, client, requests = build(16, 2, lambda: time.sleep(0.03))
    with ThreadPoolExecutor(max_workers=5) as pool:
        batch_future = pool.submit(detector.detect_batch, requests[:4])
        single_futures = [pool.submit(detector.detect, request) for request in requests[4:8]]
        awaited_batch, *awaited = await asyncio.gather(
            detector.adetect_batch(requests[8:12]),
            *(detector.adetect(request) for request in requests[12:]),
        )
        results = (
            batch_future.result()
            + [BatchResult(detection=future.result()) for future in single_futures]
            + awaited_batch
            + [BatchResult(detection=detection) for detection in awaited]
        )
    assert _words(results) == _expected_words(16)
    assert client.max_in_flight == 2
    assert len(client.calls) == 16


@LLM_BACKED
def test_one_failing_client_call_does_not_discard_the_other_results(build):
    detector, client, requests = build(6, 2)
    del client.replies[requests[2].answer]
    client.replies[requests[2].answer] = "not json"
    results = detector.detect_batch(requests)
    assert [result.ok for result in results] == [True, True, False, True, True, True]
    assert isinstance(results[2].error, UpstreamError)
    assert results[2].error.code == "unparseable_response"
    expected = _expected_words(6)
    assert _words(results[:2] + results[3:]) == expected[:2] + expected[3:]


async def test_cancelled_batch_finishes_its_requests_and_shuts_its_pool_down():
    """Cancelling stops the wait only; the pool ends with the batch and the detector stays usable."""
    entered, gate = threading.Event(), threading.Event()
    loop_errors: list[dict] = []
    asyncio.get_running_loop().set_exception_handler(lambda _loop, ctx: loop_errors.append(ctx))

    def blocked() -> None:
        entered.set()
        gate.wait(10)

    detector, client, requests = _judge(8, 2, blocked)
    task = asyncio.create_task(detector.adetect_batch(requests))
    while not entered.is_set():
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.05)
    assert client.max_in_flight == 2
    assert len(client.calls) == 2  # the calls in flight are still blocked
    gate.set()
    deadline = time.monotonic() + 10
    while any(thread.name.startswith(BATCH_THREAD) for thread in threading.enumerate()):
        assert time.monotonic() < deadline, "the pool of the cancelled batch never shut down"
        await asyncio.sleep(0.005)
    assert len(client.calls) == 8
    assert _words(detector.detect_batch(requests)) == _expected_words(8)
    assert _words(await detector.adetect_batch(requests)) == _expected_words(8)
    assert client.max_in_flight == 2
    assert loop_errors == []


# -- RAGFactCheckerDetector ---------------------------------------------------------------------


def _rag(max_concurrency: int):
    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    def on_call() -> None:
        with guard:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.03)
        with guard:
            state["active"] -= 1

    answers = [f"Answer number {i} mentions Madrid." for i in range(12)]
    results = {
        answer: {"hallucinated_triplets": [["Madrid", "is", f"city {i}"]]}
        for i, answer in enumerate(answers)
    }
    detector = RAGFactCheckerDetector(
        fact_checker=FakeFactChecker(results, on_call), max_concurrency=max_concurrency
    )
    return detector, state, _prompts(*answers)


async def test_rag_fact_checker_runs_a_batch_concurrently_within_max_concurrency():
    detector, state, requests = _rag(3)
    expected = [detector.detect(request, include_detail=True) for request in requests]
    state["max"] = 0
    results = detector.detect_batch(requests, include_detail=True)
    assert [result.get_or_raise() for result in results] == expected
    assert state["max"] == 3
    assert [r.get_or_raise() for r in await detector.adetect_batch(requests)] == [
        detector.detect(request) for request in requests
    ]
    assert state["max"] == 3


async def test_rag_fact_checker_bound_holds_across_batches_and_single_calls():
    detector, state, requests = _rag(2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(detector.detect, request) for request in requests[:4]]
        first, second = await asyncio.gather(
            detector.adetect_batch(requests[4:8]), detector.adetect_batch(requests[8:])
        )
        single = [future.result() for future in futures]
    assert single + [result.get_or_raise() for result in first + second] == [
        detector.detect(request) for request in requests
    ]
    assert state["max"] == 2


# -- LLMDetector ----------------------------------------------------------------------------------


@pytest.mark.filterwarnings(
    "ignore::lettucedetect.detectors._deprecation.LettuceDetectDeprecationWarning"
)
async def test_llm_detector_forwards_batches_to_the_detector_it_built():
    requests, replies = _numbered_judge(4)
    detector = LLMDetector(model="judge-model", client=FakeClient(replies))
    assert _words(detector.detect_batch(requests)) == _expected_words(4)
    assert _words(await detector.adetect_batch(requests)) == _expected_words(4)
    with pytest.raises(InvalidInputError) as exc:
        detector.detect_batch(requests, include_tokens=True)
    assert exc.value.code == "tokens_not_supported"
