"""Tests for ``RAGFactCheckerDetector.detect`` with an injected, network-free fact checker."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from unittest.mock import patch

import pytest

from lettucedetect.detectors.errors import ConfigurationError, InvalidInputError
from lettucedetect.detectors.rag_fact_checker import RAGFactCheckerDetector
from lettucedetect.detectors.types import (
    Chunk,
    DetectedSpan,
    PromptInput,
    SourcesInput,
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
    "fact_check_results": {"0": False},
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


def _legacy_as_spans(raw: list[dict]) -> list[DetectedSpan]:
    return [
        DetectedSpan(start=s["start"], end=s["end"], text=s["text"], score=s["confidence"])
        for s in raw
    ]


@pytest.mark.parametrize("case", list(CASES))
def test_detect_spans_equal_legacy_spans_with_score_as_confidence(case):
    """``detect`` returns today's ``predict(output_format="spans")`` spans, score = confidence."""
    detector = _detector(dict(CASES[case]))
    for answer, _ in CASES[case]:
        legacy = _legacy_as_spans(detector.predict(["context"], answer, "q", output_format="spans"))
        sources = SourcesInput(sources=[Chunk(id="0", text="context")], answer=answer, question="q")
        assert detector.detect(sources).spans == legacy
        legacy_prompt = _legacy_as_spans(
            detector.predict_prompt("prompt", answer, output_format="spans")
        )
        assert detector.detect(PromptInput(prompt="prompt", answer=answer)).spans == legacy_prompt


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


def test_include_tokens_raises_tokens_not_supported():
    """The detector has no token scores."""
    with pytest.raises(InvalidInputError) as exc:
        _detector().detect(SOURCES, include_tokens=True)
    assert exc.value.code == "tokens_not_supported"
    assert exc.value.param == "include_tokens"


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
