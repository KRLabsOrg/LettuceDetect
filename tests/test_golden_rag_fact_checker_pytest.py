"""Golden (characterization) tests pinning today's ``RAGFactCheckerDetector`` output.

The detector is built without its constructor (which needs an OpenAI key) and given a fake fact
checker that returns fixed triplets and results, so the tests are network-free. Recordings live in
``tests/fixtures/golden/rag_<case>_<format>.json``.

``rag_merge_overlap_text_shorter_than_offsets_*`` records a defect: ``_merge_overlapping_spans``
extends ``end`` but keeps the first span's ``text``, so ``text`` is shorter than ``end - start``.
The pull request that fixes it updates exactly those fixtures as an intended change.
"""

from __future__ import annotations

import pytest

from lettucedetect.detectors.rag_fact_checker import RAGFactCheckerDetector
from tests.golden import assert_golden

ANSWER = "The population of France is 69 million and Paris is its capital."
ANSWER_2 = "Lisbon is the capital of Spain."
ANSWER_3 = "Madrid is mentioned here."
ANSWER_4 = "France is 69 million."


def _result(hallucinated: list[list[str]]) -> dict:
    return {
        "answer_triplets": [["France", "population", "69 million"], ["Paris", "is", "capital"]],
        "context_triplets": [["France", "population", "67 million"]],
        "hallucinated_triplets": hallucinated,
        "fact_check_results": {"0": "contradiction", "1": "entailment"},
    }


# case -> [(answer, fact checker result), ...]
CASES = {
    "basic": [
        (ANSWER, _result([["France", "population", "69 million"]])),
        (ANSWER_2, _result([["Lisbon", "capital", "Spain"]])),
    ],
    # The fourth element is only found by the single-word fallback, so confidence is 0.7.
    "single_word_match": [(ANSWER_3, _result([["Barcelona", "north", "coast", "Madrid"]]))],
    # Both triplets match at the same start; merging keeps the first, shorter, text.
    "merge_overlap_text_shorter_than_offsets": [
        (ANSWER_4, _result([["France", "is", "unknown"], ["France", "is", "69 million"]]))
    ],
}


class FakeFactChecker:
    """Returns the fixed result recorded for each answer; never touches the network."""

    def __init__(self, results: dict[str, dict]) -> None:
        """Store results keyed by answer."""
        self.results = results

    def detect_hallucinations(self, context, answer, question=None) -> dict:
        """Return the fixed result for ``answer``."""
        return self.results[answer]

    def detect_hallucinations_batch(self, contexts, answers) -> list[dict]:
        """Return the fixed results for ``answers``, in order."""
        return [self.results[answer] for answer in answers]


def _detector(case: str) -> RAGFactCheckerDetector:
    detector = RAGFactCheckerDetector.__new__(RAGFactCheckerDetector)  # skip the API-key __init__
    detector.rag = FakeFactChecker(dict(CASES[case]))
    return detector


def _record(case: str, output_format: str) -> dict:
    detector = _detector(case)
    answers = [answer for answer, _ in CASES[case]]
    recording = {
        "predict": [
            detector.predict(["context"], answer, "question", output_format=output_format)
            for answer in answers
        ],
        "predict_prompt": [
            detector.predict_prompt("prompt", answer, output_format=output_format)
            for answer in answers
        ],
    }
    if output_format != "detailed":  # the batch path does not support "detailed"
        recording["predict_prompt_batch"] = detector.predict_prompt_batch(
            ["prompt"] * len(answers), answers, output_format=output_format
        )
    return recording


@pytest.mark.parametrize("output_format", ["tokens", "spans", "detailed"])
@pytest.mark.parametrize("case", list(CASES))
def test_rag_fact_checker_golden(case, output_format):
    """Pin every entry point for one case and one output format."""
    assert_golden(f"rag_{case}_{output_format}", _record(case, output_format))


def test_batch_detailed_is_rejected():
    """Today ``predict_prompt_batch`` raises for ``"detailed"``; the message is recorded."""
    with pytest.raises(ValueError) as exc_info:
        _detector("basic").predict_prompt_batch(["p"], [ANSWER], output_format="detailed")
    assert_golden("rag_batch_detailed_error", {"message": str(exc_info.value)})


def test_single_word_match_has_lower_confidence():
    """The single-word fallback reports 0.7 instead of 0.9."""
    spans = _detector("single_word_match").predict(["c"], ANSWER_3, output_format="spans")
    assert [span["confidence"] for span in spans] == [0.7]


def test_merge_leaves_text_shorter_than_offsets():
    """Documents the defect the recording pins: merged ``text`` is shorter than its offsets."""
    spans = _detector("merge_overlap_text_shorter_than_offsets").predict(
        ["c"], ANSWER_4, output_format="spans"
    )
    assert len(spans) == 1
    assert len(spans[0]["text"]) < spans[0]["end"] - spans[0]["start"]
