"""Tests for ``RAGFactCheckerDetector.detect`` with an injected, network-free fact checker."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from unittest.mock import patch

import pytest

from lettucedetect.detectors.errors import ConfigurationError, InvalidInputError, UpstreamError
from lettucedetect.detectors.rag_fact_checker import RAGFactCheckerDetector
from lettucedetect.detectors.types import (
    Chunk,
    DetailedResult,
    PromptInput,
    SourcesInput,
    Triplets,
)
from tests.detector_contract import DetectorContract
from tests.test_golden_rag_fact_checker_pytest import ANSWER_4, CASES

RAG_FACT_CHECKER = "lettucedetect.ragfactchecker.RAGFactChecker"

ANSWER = (
    "The population of France is 69 million according to the report, "
    "and Madrid is mentioned as the largest city in the region."
)
SOURCES = SourcesInput(
    sources=[
        Chunk(id="s1", text="France has a population of about 67 million people."),
        Chunk(id="s2", text="Paris is the capital and largest city of France."),
    ],
    answer=ANSWER,
    question="How many people live in France?",
)
PROMPT = PromptInput(
    prompt="Answer from the context: France has a population of about 67 million.", answer=ANSWER
)
# A phrase match ("69 million", 0.9) and a single-word fallback match ("Madrid", 0.7).
RESULT = {
    "answer_triplets": [["France", "population", "69 million"]],
    "context_triplets": [["France", "population", "67 million"]],
    "hallucinated_triplets": [
        ["France", "population", "69 million"],
        ["Barcelona", "north", "coast", "Madrid"],
    ],
    "fact_check_results": {0: False},
}


class FakeFactChecker:
    """Returns a fixed result per answer and records every call; never touches the network."""

    def __init__(self, results: dict[str, dict], on_call: Callable[[], None] | None = None) -> None:
        """Keep the results and the hook to call once per fact check."""
        self.results = results
        self.on_call = on_call
        self.calls: list[tuple[list[str], str, str | None]] = []
        self.thread_ids: list[int] = []

    def detect_hallucinations(self, context, answer, question=None) -> dict:
        """Return the fixed result for ``answer``."""
        self.calls.append((list(context), answer, question))
        self.thread_ids.append(threading.get_ident())
        if self.on_call:
            self.on_call()
        return self.results[answer]

    def detect_hallucinations_batch(self, contexts, answers) -> list[dict]:
        """Return the fixed results for ``answers``, in order."""
        return [self.results[answer] for answer in answers]


def _detector(results: dict[str, dict] | None = None, **kwargs) -> RAGFactCheckerDetector:
    fake = FakeFactChecker(results if results is not None else {ANSWER: RESULT})
    return RAGFactCheckerDetector(fact_checker=fake, **kwargs)


class TestRAGFactCheckerDetectorContract(DetectorContract):
    """The detector contract holds with a fake fact checker."""

    sources_request = SOURCES
    prompt_request = PROMPT

    def make_detector(self, on_call=None):  # noqa: D102
        return RAGFactCheckerDetector(fact_checker=FakeFactChecker({ANSWER: RESULT}, on_call))


# -- constructor -----------------------------------------------------------------------------


def test_injected_fact_checker_builds_nothing_and_reads_no_environment(monkeypatch):
    """With ``fact_checker`` passed, RAGFactChecker is not built and no API key is needed."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with patch(RAG_FACT_CHECKER, side_effect=AssertionError("RAGFactChecker was built")):
        detector = _detector()
    assert detector.detect(SOURCES).spans


def test_without_fact_checker_builds_rag_fact_checker_from_the_old_arguments():
    """Without ``fact_checker``, RAGFactChecker is built from the old arguments, as before."""
    with patch(RAG_FACT_CHECKER) as built:
        detector = RAGFactCheckerDetector("key", "gpt-4o-mini", "http://local/v1", 0.2)
    built.assert_called_once_with(
        openai_api_key="key", model="gpt-4o-mini", base_url="http://local/v1", temperature=0.2
    )
    assert detector.rag is built.return_value


def test_model_id_is_model():
    """``model_id`` is the ``model`` argument."""
    assert _detector().model_id == "gpt-4o"
    assert _detector(model="gpt-4o-mini").model_id == "gpt-4o-mini"


def test_max_concurrency_defaults_to_30():
    """The default bound on concurrent ``detect`` calls is 30."""
    assert _detector().max_concurrency == 30


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_invalid_max_concurrency_raises_configuration_error(value):
    """``max_concurrency`` below 1 (or not an integer) raises ``ConfigurationError``."""
    with patch(RAG_FACT_CHECKER, side_effect=AssertionError("built before validating")):
        with pytest.raises(ConfigurationError) as exc:
            RAGFactCheckerDetector(max_concurrency=value)
    assert exc.value.code == "invalid_config"
    assert exc.value.param == "max_concurrency"


# -- detect ----------------------------------------------------------------------------------


def test_detect_passes_context_question_and_answer_to_the_fact_checker():
    """Sources become the context list; a prompt becomes a one-item context without question."""
    detector = _detector()
    detector.detect(SOURCES)
    detector.detect(PROMPT)
    assert detector.rag.calls == [
        ([c.text for c in SOURCES.sources], ANSWER, SOURCES.question),
        ([PROMPT.prompt], ANSWER, None),
    ]


def test_scores_are_0_9_for_a_full_match_and_0_7_for_a_partial_match():
    """The fixed scores of today's ``confidence`` are kept."""
    spans = _detector().detect(SOURCES).spans
    assert {(s.text, s.score) for s in spans} == {("69 million", 0.9), ("Madrid", 0.7)}


def test_merged_span_text_equals_the_answer_between_its_offsets():
    """Overlapping matches merge into one span whose text covers the merged offsets."""
    detector = _detector(dict(CASES["merge_overlap"]))
    spans = detector.detect(PromptInput(prompt="p", answer=ANSWER_4)).spans
    assert len(spans) == 1
    assert spans[0].text == ANSWER_4[spans[0].start : spans[0].end] == "France is 69 million"


def test_one_synthetic_score_warning_when_there_are_spans():
    """A result with spans has exactly one ``synthetic_score`` warning."""
    warnings = _detector().detect(SOURCES).warnings
    assert [w.code for w in warnings] == ["synthetic_score"]


def test_no_warning_when_there_are_no_spans():
    """A result without spans has no ``synthetic_score`` warning."""
    clean = {**RESULT, "hallucinated_triplets": []}
    detection = _detector({ANSWER: clean}).detect(SOURCES)
    assert detection.spans == []
    assert detection.warnings == []


def test_min_score_filtering_keeps_the_warning_from_the_unfiltered_result():
    """``min_score`` drops low spans; the warning reports the scoring, so it stays."""
    detection = _detector().detect(SOURCES, min_score=0.8)
    assert [s.score for s in detection.spans] == [0.9]
    assert [w.code for w in detection.warnings] == ["synthetic_score"]


def test_usage_has_one_entry_with_sources_and_char_counts():
    """``usage`` has one entry: model id, no version, sources and character counts."""
    detector = _detector()
    (usage,) = detector.detect(SOURCES).usage
    assert usage.model_id == "gpt-4o"
    assert usage.model_version is None
    assert usage.units == {
        "sources": 2,
        "source_chars": sum(len(c.text) for c in SOURCES.sources),
        "answer_chars": len(ANSWER),
    }
    (prompt_usage,) = detector.detect(PROMPT).usage
    assert prompt_usage.units == {
        "sources": 1,
        "source_chars": len(PROMPT.prompt),
        "answer_chars": len(ANSWER),
    }


def test_include_detail_returns_triplets_results_and_a_triplet_per_span():
    """``include_detail=True`` returns the evidence; ``span_triplets`` is parallel to the spans."""
    detection = _detector().detect(SOURCES, include_detail=True)
    assert detection.detail == DetailedResult(
        triplets=Triplets(
            answer=RESULT["answer_triplets"],
            context=RESULT["context_triplets"],
            hallucinated=RESULT["hallucinated_triplets"],
        ),
        fact_check_results={0: False},
        span_triplets=[
            ["France", "population", "69 million"],
            ["Barcelona", "north", "coast", "Madrid"],
        ],
    )
    assert [s.text for s in detection.spans] == ["69 million", "Madrid"]


def test_without_include_detail_detail_is_none():
    """``detail`` is ``None`` unless asked for."""
    assert _detector().detect(SOURCES).detail is None


def test_min_score_keeps_span_triplets_parallel_to_spans():
    """Spans dropped by ``min_score`` take their source triplet with them."""
    detection = _detector().detect(SOURCES, min_score=0.8, include_detail=True)
    assert [s.text for s in detection.spans] == ["69 million"]
    assert detection.detail.span_triplets == [["France", "population", "69 million"]]


def test_malformed_fact_check_results_raise_upstream_error_only_with_detail():
    """A reply that does not fit ``DetailedResult`` raises ``UpstreamError``; spans still work."""
    malformed = {**RESULT, "fact_check_results": {"0": "contradiction"}}
    detector = _detector({ANSWER: malformed})
    with pytest.raises(UpstreamError) as exc:
        detector.detect(SOURCES, include_detail=True)
    assert exc.value.code == "invalid_model_output"
    assert detector.detect(SOURCES).spans


def test_include_tokens_raises_tokens_not_supported():
    """The detector has no token scores."""
    with pytest.raises(InvalidInputError) as exc:
        _detector().detect(SOURCES, include_tokens=True)
    assert exc.value.code == "tokens_not_supported"
    assert exc.value.param == "include_tokens"


# -- legacy predict* methods -----------------------------------------------------------------


@pytest.mark.filterwarnings(
    "ignore::lettucedetect.detectors._deprecation.LettuceDetectDeprecationWarning"
)
@pytest.mark.parametrize("output_format", ["tokens", "spans", "detailed"])
def test_predict_and_predict_prompt_go_through_detect(output_format):
    """``predict`` and ``predict_prompt`` call ``detect`` once per call."""
    detector = _detector()
    with patch.object(detector, "detect", wraps=detector.detect) as detect:
        detector.predict(["context"], ANSWER, output_format=output_format)
        detector.predict_prompt("prompt", ANSWER, output_format=output_format)
    assert detect.call_count == 2


@pytest.mark.filterwarnings(
    "ignore::lettucedetect.detectors._deprecation.LettuceDetectDeprecationWarning"
)
def test_predict_spans_carry_the_source_triplet():
    """Legacy spans keep their ``triplet`` key."""
    spans = _detector().predict(["context"], ANSWER, output_format="spans")
    assert [(s["text"], s["triplet"]) for s in spans] == [
        ("69 million", ["France", "population", "69 million"]),
        ("Madrid", ["Barcelona", "north", "coast", "Madrid"]),
    ]


@pytest.mark.filterwarnings(
    "ignore::lettucedetect.detectors._deprecation.LettuceDetectDeprecationWarning"
)
@pytest.mark.parametrize(
    ("kwargs", "code", "param"),
    [
        ({"output_format": "bad"}, "unsupported_output_format", "output_format"),
        ({"min_confidence": 1.5}, "invalid_min_score", "min_confidence"),
        ({"min_confidence": -0.1}, "invalid_min_score", "min_confidence"),
    ],
)
def test_predict_and_predict_prompt_raise_invalid_input_error(kwargs, code, param):
    """Bad legacy arguments raise ``InvalidInputError`` before the fact checker runs."""
    detector = _detector()
    for call in (
        lambda: detector.predict(["context"], ANSWER, **kwargs),
        lambda: detector.predict_prompt("prompt", ANSWER, **kwargs),
    ):
        with pytest.raises(InvalidInputError) as exc:
            call()
        assert (exc.value.code, exc.value.param) == (code, param)
    assert detector.rag.calls == []


@pytest.mark.filterwarnings(
    "ignore::lettucedetect.detectors._deprecation.LettuceDetectDeprecationWarning"
)
@pytest.mark.parametrize(
    ("args", "kwargs", "code"),
    [
        ((["p", "p"], [ANSWER]), {}, "length_mismatch"),
        ((["p"], [ANSWER]), {"min_confidence": 2.0}, "invalid_min_score"),
        ((["p"], [ANSWER]), {"output_format": "detailed"}, "unsupported_output_format"),
    ],
)
def test_predict_prompt_batch_raises_invalid_input_error(args, kwargs, code):
    """Bad batch arguments raise ``InvalidInputError`` before the fact checker runs."""
    detector = _detector()
    with patch.object(detector.rag, "detect_hallucinations_batch") as batch:
        with pytest.raises(InvalidInputError) as exc:
            detector.predict_prompt_batch(*args, **kwargs)
    assert exc.value.code == code
    batch.assert_not_called()


@pytest.mark.filterwarnings(
    "ignore::lettucedetect.detectors._deprecation.LettuceDetectDeprecationWarning"
)
def test_max_concurrency_bounds_concurrent_predict_calls():
    """``predict`` and ``predict_prompt`` share the bound through ``detect``."""
    lock = threading.Lock()
    active = peak = 0

    def count() -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1

    detector = RAGFactCheckerDetector(
        fact_checker=FakeFactChecker({ANSWER: RESULT}, on_call=count), max_concurrency=1
    )
    threads = [
        threading.Thread(target=detector.predict, args=(["context"], ANSWER)) for _ in range(4)
    ] + [
        threading.Thread(target=detector.predict_prompt, args=("prompt", ANSWER)) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(detector.rag.calls) == 8
    assert peak == 1


# -- adetect and concurrency -----------------------------------------------------------------


async def test_adetect_equals_detect_and_calls_the_fact_checker_from_a_worker_thread():
    """``adetect`` returns what ``detect`` returns, and runs the fact checker off the caller's thread."""
    detector = _detector()
    result = await detector.adetect(SOURCES)
    assert detector.rag.thread_ids[-1] != threading.get_ident()
    assert result == detector.detect(SOURCES)


def test_a_failing_fact_check_frees_its_concurrency_slot():
    """An error from the fact checker releases the slot, so the next call still runs."""
    calls = 0

    def fail_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("fact checker failed")

    detector = RAGFactCheckerDetector(
        fact_checker=FakeFactChecker({ANSWER: RESULT}, on_call=fail_once), max_concurrency=1
    )
    with pytest.raises(RuntimeError):
        detector.detect(SOURCES)
    assert detector.detect(SOURCES).spans


async def test_max_concurrency_bounds_concurrent_detect_and_adetect_calls():
    """With ``max_concurrency=2``, the fact checker never runs more than twice at once."""
    lock = threading.Lock()
    active = peak = 0

    def count() -> None:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1

    detector = RAGFactCheckerDetector(
        fact_checker=FakeFactChecker({ANSWER: RESULT}, on_call=count), max_concurrency=2
    )
    threads = [threading.Thread(target=detector.detect, args=(SOURCES,)) for _ in range(8)]
    for thread in threads:
        thread.start()
    await asyncio.gather(*(detector.adetect(PROMPT) for _ in range(8)))
    for thread in threads:
        thread.join()
    assert len(detector.rag.calls) == 16
    assert peak == 2
