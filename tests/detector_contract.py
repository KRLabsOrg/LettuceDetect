"""Reusable contract suite that every detector must pass.

The suite checks the invariants of ``detect`` and ``adetect`` and nothing else, so it does not
care how a detector is built. It reads the public result and the public errors only.

How a detector task plugs in
============================

1. Subclass :class:`DetectorContract` inside a ``tests/test_*_pytest.py`` file. The base class is
   not collected by pytest itself, only your ``Test...`` subclass is::

       from tests.detector_contract import DetectorContract

       class TestMyDetectorContract(DetectorContract):
           sources_request = SourcesInput(...)
           prompt_request = PromptInput(...)

           def make_detector(self, on_call=None):
               return MyDetector(model=FakeModel(on_call))

2. Set ``sources_request`` and ``prompt_request``. The answers must be long enough that no
   short fragment of them also appears in an unrelated log message, and the detector must flag
   spans with at least two different scores for the same request (the ``min_score`` and
   ordering checks need that).
3. Implement ``make_detector(on_call)``. It returns a fresh detector that works offline and
   needs no GPU. When ``on_call`` is not ``None``, the detector must call ``on_call()`` once per
   ``detect`` call, on the thread that runs the detection, after it started and before it
   returns. Put it where the slow work happens (the model call or the HTTP call). The suite uses
   it to make a detector slow, or to block it on an event.
4. Detectors that predate ``detect`` pass too. A ``predict``-only subclass is wrapped by
   ``BaseDetector`` and is checked through ``detect`` and ``adetect``. Its test class must
   silence ``LettuceDetectDeprecationWarning`` (``pytest.mark.filterwarnings("ignore::...")``).
5. Detectors that do not support tokens or detail leave ``supports_tokens`` and
   ``supports_detail`` at ``False``. The suite then expects ``InvalidInputError``.

Every check is a plain method without pytest fixtures, so ``tests/test_detector_contract_pytest.py``
can also run a check against a deliberately broken detector and assert that it fails.

About cancelling ``adetect``
============================

``adetect`` runs ``detect`` in a worker thread. Cancelling the awaiting task, or timing out with
``asyncio.wait_for``, stops the wait only. The thread keeps running until the detector returns,
because Python cannot stop a thread. The cancel checks document that: the detector is still busy
right after the cancel, the thread is let go afterwards, and only then do the checks run.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import logging
import math
import os
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Generator
from pathlib import Path

import pytest

from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.errors import InvalidInputError
from lettucedetect.detectors.types import Detection, DetectRequest, PromptInput, SourcesInput

CONTENT_LOGGER = "lettucedetect.content"
"""The only logger allowed to carry answer or source text (CONTRIBUTING, "Errors and logging")."""

_SLOW_SECONDS = 0.3
_WAIT_SECONDS = 10.0


@contextlib.contextmanager
def _isolated_environment() -> Generator[tuple[Path, Path]]:
    """Run with an empty temporary directory as cwd and as ``HOME``; yield both paths."""
    with tempfile.TemporaryDirectory() as root:
        cwd = Path(root) / "cwd"
        home = Path(root) / "home"
        cwd.mkdir()
        home.mkdir()
        old_cwd = Path.cwd()
        saved = {key: os.environ.get(key) for key in ("HOME", "USERPROFILE")}
        os.chdir(cwd)
        os.environ["HOME"] = os.environ["USERPROFILE"] = str(home)
        try:
            yield cwd, home
        finally:
            os.chdir(old_cwd)
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class _Recorder(logging.Handler):
    """Collect every log record that reaches the root logger."""

    def __init__(self) -> None:
        """Create an empty recorder."""
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the record."""
        self.records.append(record)


def _is_content_logger(name: str) -> bool:
    return name == CONTENT_LOGGER or name.startswith(CONTENT_LOGGER + ".")


def _texts_of(request: DetectRequest) -> list[str]:
    """Return every piece of customer text in a request."""
    if isinstance(request, SourcesInput):
        texts = [chunk.text for chunk in request.sources]
        if request.question:
            texts.append(request.question)
    else:
        texts = [request.prompt]
    texts.append(request.answer)
    return [text for text in texts if text]


def _assert_empty(directory: Path, what: str) -> None:
    left = sorted(path.name for path in directory.rglob("*"))
    assert not left, f"a call left files in the temporary {what}: {left}"


class DetectorContract:
    """The checks every detector must pass. Subclass it and set the three members below."""

    sources_request: SourcesInput
    prompt_request: PromptInput

    def make_detector(self, on_call: Callable[[], None] | None = None) -> BaseDetector:
        """Return a fresh detector that calls ``on_call()`` once per ``detect`` call."""
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------------------------

    def _requests(self) -> list[DetectRequest]:
        return [self.sources_request, self.prompt_request]

    def _distinct_scores(self, request: DetectRequest) -> list[float]:
        scores = sorted({span.score for span in self.make_detector().detect(request).spans})
        assert len(scores) >= 2, "the sample request must produce spans with two different scores"
        return scores

    # -- result shape ----------------------------------------------------------------------

    def test_span_text_equals_answer_between_offsets(self) -> None:
        """Every span's text equals the answer between its offsets, for both request kinds."""
        detector = self.make_detector()
        for request in self._requests():
            spans = detector.detect(request).spans
            assert spans, "the sample request must produce at least one span"
            for span in spans:
                assert request.answer[span.start : span.end] == span.text

    def test_every_span_has_a_numeric_score_and_min_score_filters_on_it(self) -> None:
        """Every span has a numeric score in [0, 1], and min_score keeps exactly the spans at or above it."""
        detector = self.make_detector()
        for request in self._requests():
            everything = detector.detect(request).spans
            for span in everything:
                assert isinstance(span.score, int | float)
                assert not isinstance(span.score, bool)
                assert 0.0 <= span.score <= 1.0
            for threshold in [0.0, *self._distinct_scores(request), 1.0]:
                kept = detector.detect(request, min_score=threshold).spans
                assert kept == [s for s in everything if s.score >= threshold]

    def test_usage_has_an_entry_with_the_model_id_and_a_model_version_field(self) -> None:
        """The usage list has an entry whose model_id is the detector's and that has a model_version field."""
        detector = self.make_detector()
        for request in self._requests():
            usage = detector.detect(request).usage
            matching = [u for u in usage if u.model_id == detector.model_id]
            assert matching, f"no usage entry for model_id {detector.model_id!r}"
            assert "model_version" in type(matching[0]).model_fields
            assert hasattr(matching[0], "model_version")

    def test_tokens_and_detail_raise_where_unsupported(self) -> None:
        """include_tokens / include_detail raise InvalidInputError where unsupported, and return data where supported."""
        detector = self.make_detector()
        for request in self._requests():
            if detector.supports_tokens:
                assert detector.detect(request, include_tokens=True).tokens is not None
                assert detector.detect(request).tokens is None
            else:
                with pytest.raises(InvalidInputError) as exc:
                    detector.detect(request, include_tokens=True)
                assert exc.value.param == "include_tokens"
            if detector.supports_detail:
                assert detector.detect(request, include_detail=True).detail is not None
                assert detector.detect(request).detail is None
            else:
                with pytest.raises(InvalidInputError) as exc:
                    detector.detect(request, include_detail=True)
                assert exc.value.param == "include_detail"

    # -- repeatability and side effects ----------------------------------------------------

    def test_same_input_gives_the_same_result_also_after_other_input(self) -> None:
        """A second call with the same input returns the same result, also after a call with other input."""
        detector = self.make_detector()
        for request, other in [
            (self.sources_request, self.prompt_request),
            (self.prompt_request, self.sources_request),
        ]:
            first = detector.detect(request)
            assert detector.detect(request) == first
            detector.detect(other)
            assert detector.detect(request) == first

    def test_a_call_leaves_an_empty_cwd_and_home_empty(self) -> None:
        """With an empty temporary directory as cwd and HOME, detect and adetect leave both empty."""
        detector = self.make_detector()
        with _isolated_environment() as (cwd, home):
            for request in self._requests():
                detector.detect(request)
                asyncio.run(detector.adetect(request))
            _assert_empty(cwd, "cwd")
            _assert_empty(home, "HOME")

    def test_a_call_prints_nothing_and_logs_no_text_outside_the_content_logger(self) -> None:
        """A call prints nothing, and no log record outside lettucedetect.content holds answer, source or prompt text."""
        detector = self.make_detector()
        recorder = _Recorder()
        root = logging.getLogger()
        old_level = root.level
        root.addHandler(recorder)
        root.setLevel(logging.DEBUG)
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                for request in self._requests():
                    detector.detect(request)
                    asyncio.run(detector.adetect(request))
        finally:
            root.removeHandler(recorder)
            root.setLevel(old_level)
        assert out.getvalue() == "", "a call printed to stdout"
        assert err.getvalue() == "", "a call printed to stderr"
        texts = [text for request in self._requests() for text in _texts_of(request)]
        for record in recorder.records:
            if _is_content_logger(record.name):
                continue
            rendered = record.getMessage() + (record.exc_text or "")
            for text in texts:
                assert text not in rendered, f"logger {record.name!r} logged customer text"

    # -- adetect ---------------------------------------------------------------------------

    async def test_adetect_equals_detect(self) -> None:
        """Adetect returns the same result as detect, for both request kinds and with min_score."""
        detector = self.make_detector()
        for request in self._requests():
            assert await detector.adetect(request) == detector.detect(request)
            threshold = self._distinct_scores(request)[-1]
            assert await detector.adetect(request, min_score=threshold) == detector.detect(
                request, min_score=threshold
            )

    async def test_adetect_runs_off_the_loop_thread_and_keeps_the_loop_free(self) -> None:
        """The detector runs on a thread other than the loop's, and a loop ticker keeps ticking while it is slow."""
        thread_ids: list[int] = []

        def slow() -> None:
            thread_ids.append(threading.get_ident())
            time.sleep(_SLOW_SECONDS)

        detector = self.make_detector(on_call=slow)
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        task = asyncio.create_task(ticker())
        try:
            await detector.adetect(self.sources_request)
        finally:
            task.cancel()
        assert thread_ids, "the detector never called on_call"
        assert thread_ids[0] != threading.get_ident(), "the detector ran on the event loop thread"
        assert ticks >= 5, f"the loop was blocked while the detector waited (ticks={ticks})"

    async def test_concurrent_adetect_returns_results_in_call_order(self) -> None:
        """Concurrent adetect calls under asyncio.gather finish out of order but return results in call order."""
        scores = self._distinct_scores(self.sources_request)
        thresholds = [0.0, scores[0], scores[-1]]
        delays = iter([0.15, 0.08, 0.0])

        def make_slow(delay: float) -> BaseDetector:
            return self.make_detector(on_call=lambda: time.sleep(delay))

        detectors = [make_slow(next(delays)) for _ in thresholds]
        expected = [
            self.make_detector().detect(self.sources_request, min_score=t) for t in thresholds
        ]
        assert len({len(e.spans) for e in expected}) > 1, (
            "the thresholds must give different results"
        )
        results = await asyncio.gather(
            *(
                d.adetect(self.sources_request, min_score=t)
                for d, t in zip(detectors, thresholds, strict=True)
            )
        )
        assert list(results) == expected

    async def test_adetect_errors_match_detect_errors(self) -> None:
        """Out-of-range min_score and unsupported flags come out of the await with the same class, code and param as from detect."""
        detector = self.make_detector()
        cases: list[dict] = [
            {"min_score": -0.1},
            {"min_score": 1.1},
            {"min_score": math.nan},
        ]
        if not detector.supports_tokens:
            cases.append({"include_tokens": True})
        if not detector.supports_detail:
            cases.append({"include_detail": True})
        for kwargs in cases:
            with pytest.raises(InvalidInputError) as sync_exc:
                detector.detect(self.sources_request, **kwargs)
            with pytest.raises(InvalidInputError) as async_exc:
                await detector.adetect(self.sources_request, **kwargs)
            assert type(async_exc.value) is type(sync_exc.value)
            assert async_exc.value.code == sync_exc.value.code
            assert async_exc.value.param == sync_exc.value.param
            assert async_exc.value.param is not None

    # -- cancelling adetect ----------------------------------------------------------------

    async def _cancel_and_check(
        self, cancel: Callable[[asyncio.Task[Detection]], Awaitable[None]]
    ) -> None:
        """Cancel a blocked ``adetect`` with ``cancel`` and check the detector stays usable."""
        entered, gate, returned = threading.Event(), threading.Event(), threading.Event()
        loop_errors: list[dict] = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))

        def blocked() -> None:
            entered.set()
            gate.wait(_WAIT_SECONDS)
            returned.set()

        detector = self.make_detector(on_call=blocked)
        request = self.sources_request
        gate.set()
        baseline = detector.detect(request)
        gate.clear()
        entered.clear()
        returned.clear()

        with _isolated_environment() as (cwd, home):
            task = asyncio.create_task(detector.adetect(request))
            deadline = time.monotonic() + _WAIT_SECONDS
            while not entered.is_set():
                assert time.monotonic() < deadline, "the detector never started"
                await asyncio.sleep(0.005)
            await cancel(task)
            # The wait is over, the thread is not: it keeps running until the detector returns.
            assert not returned.is_set(), "the detector stopped when the wait was cancelled"
            gate.set()
            while not returned.is_set():
                assert time.monotonic() < deadline, "the abandoned thread never finished"
                await asyncio.sleep(0.005)
            await asyncio.sleep(0.05)  # let the loop see anything the thread would report

            assert detector.detect(request) == baseline
            assert await detector.adetect(request) == baseline
            assert loop_errors == [], f"the abandoned thread reported to the loop: {loop_errors}"
            _assert_empty(cwd, "cwd")
            _assert_empty(home, "HOME")

    async def test_task_cancel_leaves_the_detector_usable(self) -> None:
        """After Task.cancel() on a busy adetect the thread keeps running; once it finishes the detector is valid."""

        async def cancel(task: asyncio.Task[Detection]) -> None:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        await self._cancel_and_check(cancel)

    async def test_wait_for_timeout_leaves_the_detector_usable(self) -> None:
        """After asyncio.wait_for times out on a busy adetect the thread keeps running; once it finishes the detector is valid."""

        async def cancel(task: asyncio.Task[Detection]) -> None:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(task, timeout=0.05)

        await self._cancel_and_check(cancel)


def run_check(contract: DetectorContract, name: str) -> None:
    """Run one check of a contract instance, awaiting it when it is a coroutine function."""
    method = getattr(contract, name)
    if inspect.iscoroutinefunction(method):
        asyncio.run(method())
    else:
        method()
