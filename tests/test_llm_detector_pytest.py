"""Pytest tests for LLMDetector token output (``output_format="tokens"``).

Regression coverage for the bug where the LLM detector validated
``output_format="tokens"`` but returned span dicts unchanged. Every prediction
entry point is exercised through an injected fake client, so the tests never hit
the network and ``cache_file`` points into ``tmp_path`` to keep the on-disk cache
untouched.
"""

from __future__ import annotations

import pytest

from lettucedetect.detectors.llm import LLMDetector
from lettucedetect.detectors.llm_client import LLMClient

TOKEN_KEYS = {"token", "pred", "prob"}


class FakeClient(LLMClient):
    """An LLMClient that returns a canned JSON response without any network call."""

    def __init__(self, response: str) -> None:
        """Store the response the fake client should hand back."""
        self.response = response

    def complete(self, system, user, model, temperature, schema) -> str:
        """Ignore the request and return the canned response."""
        return self.response


@pytest.fixture
def cache_file(tmp_path):
    """Temp cache path so the default on-disk cache is never touched."""
    return str(tmp_path / "cache.json")


def make_detector(response, cache_file, **kwargs):
    return LLMDetector(client=FakeClient(response), cache_file=cache_file, **kwargs)


def is_token_list(result):
    return bool(result) and all(set(item) == TOKEN_KEYS for item in result)


class TestTokenOutput:
    """LLMDetector must honour output_format="tokens" on every entry point."""

    def test_predict_returns_token_dicts_not_spans(self, cache_file):
        """predict() with tokens yields {token, pred, prob} dicts, never spans."""
        answer = "The population of France is 69 million."
        detector = make_detector('{"hallucination_list": ["69 million"]}', cache_file)

        result = detector.predict(
            context=["France has about 67 million people."],
            answer=answer,
            question="What is the population of France?",
            output_format="tokens",
        )

        assert is_token_list(result)
        assert [t["token"] for t in result] == answer.split()
        assert all("start" not in t and "end" not in t for t in result)
        assert any(t["pred"] == 1 and "69" in t["token"] for t in result)

    def test_predict_prompt_returns_token_dicts(self, cache_file):
        """predict_prompt() routes tokens through the same conversion."""
        answer = "Lisbon is the capital of Spain."
        detector = make_detector('{"hallucination_list": ["Spain"]}', cache_file)

        result = detector.predict_prompt("some prompt", answer, output_format="tokens")

        assert is_token_list(result)
        assert [t["token"] for t in result] == answer.split()
        assert any(t["pred"] == 1 and "Spain" in t["token"] for t in result)

    def test_predict_prompt_batch_returns_token_lists(self, cache_file):
        """Batch tokens align to each answer independently, one list per pair."""
        detector = make_detector('{"hallucination_list": ["wrong"]}', cache_file)
        prompts = ["p1", "p2"]
        answers = ["this is wrong here", "all correct"]

        results = detector.predict_prompt_batch(prompts, answers, output_format="tokens")

        assert len(results) == 2
        assert [t["token"] for t in results[0]] == answers[0].split()
        assert [t["token"] for t in results[1]] == answers[1].split()
        # "wrong" is only present in the first answer; the second flags nothing.
        assert any(t["pred"] == 1 for t in results[0])
        assert all(t["pred"] == 0 for t in results[1])

    def test_supported_tokens_use_low_constant_prob(self, cache_file):
        """Tokens outside any span get pred=0 and the supported-prob constant."""
        answer = "Everything here is fine."
        detector = make_detector('{"hallucination_list": []}', cache_file)

        result = detector.predict_prompt("p", answer, output_format="tokens")

        assert all(t["pred"] == 0 and t["prob"] == 0.1 for t in result)

    def test_flagged_token_prob_uses_span_confidence(self, cache_file):
        """When the span carries a confidence, flagged tokens report it as prob."""
        answer = "The tower is 900 meters tall."
        detector = make_detector(
            '{"hallucination_list": [{"text": "900 meters", "confidence": 0.87}]}',
            cache_file,
        )

        result = detector.predict_prompt("p", answer, output_format="tokens")

        flagged = [t for t in result if t["pred"] == 1]
        assert flagged and all(t["prob"] == 0.87 for t in flagged)

    def test_flagged_token_prob_defaults_without_confidence(self, cache_file):
        """No confidence on the span falls back to the hallucinated-prob constant."""
        answer = "The capital of France is Berlin."
        detector = make_detector('{"hallucination_list": ["Berlin"]}', cache_file)

        result = detector.predict_prompt("p", answer, output_format="tokens")

        flagged = [t for t in result if t["pred"] == 1]
        assert flagged and all(t["prob"] == 0.9 for t in flagged)

    def test_min_confidence_does_not_turn_tokens_into_spans(self, cache_file):
        """min_confidence stays inapplicable to tokens; the shape is still tokens."""
        answer = "The number is 42 exactly."
        detector = make_detector('{"hallucination_list": ["42"]}', cache_file)

        result = detector.predict_prompt("p", answer, output_format="tokens", min_confidence=0.5)

        assert is_token_list(result)


class TestSpansUnaffected:
    """The spans path must behave exactly as before."""

    def test_spans_output_unchanged(self, cache_file):
        """output_format="spans" still returns character-offset spans."""
        answer = "Lisbon is the capital of Spain."
        detector = make_detector('{"hallucination_list": ["Spain"]}', cache_file)

        spans = detector.predict_prompt("p", answer, output_format="spans")

        assert spans == [
            {
                "start": answer.index("Spain"),
                "end": answer.index("Spain") + len("Spain"),
                "text": "Spain",
            }
        ]


class TestRepeatedSpanLocalization:
    """Repeated judge spans must map to distinct answer occurrences."""

    def test_repeated_spans_keep_distinct_offsets_and_metadata(self):
        """Each repeated item keeps its own occurrence and metadata."""
        answer = "Paris is mentioned; Paris is repeated."
        items: list[str | dict] = [
            {"text": "Paris", "confidence": 0.7, "category": "first"},
            {"text": "Paris", "confidence": 0.9, "category": "second"},
        ]

        spans = LLMDetector._to_spans(items, answer)

        assert [(span["start"], span["category"]) for span in spans] == [
            (0, "first"),
            (20, "second"),
        ]
        assert [span["confidence"] for span in spans] == [0.7, 0.9]

    def test_filtered_item_still_reserves_its_occurrence(self):
        """Filtering an item must not shift a later item's location."""
        answer = "Paris is mentioned; Paris is repeated."
        items: list[str | dict] = [
            {"text": "Paris", "is_hallucination": False},
            {"text": "Paris", "is_hallucination": True},
        ]

        spans = LLMDetector._to_spans(items, answer)

        assert [(span["start"], span["end"]) for span in spans] == [(20, 25)]

    def test_low_confidence_item_still_reserves_its_occurrence(self):
        """Confidence filtering also preserves later repeated-span offsets."""
        answer = "Paris is mentioned; Paris is repeated."
        items: list[str | dict] = [
            {"text": "Paris", "confidence": 0.2},
            {"text": "Paris", "confidence": 0.9},
        ]

        spans = LLMDetector._to_spans(items, answer, min_confidence=0.5)

        assert spans == [{"start": 20, "end": 25, "text": "Paris", "confidence": 0.9}]

    def test_single_ambiguous_item_keeps_first_match(self):
        """A single repeated string still uses the documented first match."""
        answer = "Paris is mentioned; Paris is repeated."

        spans = LLMDetector._to_spans(["Paris"], answer)

        assert spans == [{"start": 0, "end": 5, "text": "Paris"}]

    def test_overlapping_and_excess_items_do_not_reuse_offsets(self):
        """Items without a free non-overlapping occurrence are dropped."""
        spans = LLMDetector._to_spans(["abc", "bc", "abc"], "abc")

        assert spans == [{"start": 0, "end": 3, "text": "abc"}]

    def test_token_output_flags_both_repeated_occurrences(self, cache_file):
        """Token projection flags every separately localized occurrence."""
        answer = "Paris appears and Paris repeats."
        detector = make_detector('{"hallucination_list": ["Paris", "Paris"]}', cache_file)

        tokens = detector.predict_prompt("p", answer, output_format="tokens")

        assert [token["pred"] for token in tokens] == [1, 0, 0, 1, 0]


class TestRepeatedSpanPromptInstructions:
    """Prompt formats should match the repeated-span localization contract."""

    def test_simple_response_format_requests_answer_order(self):
        """The simple schema should ask the model to preserve answer order."""
        detector = make_detector('{"hallucination_list": []}', cache_file="unused")

        block = detector._response_format_block()

        assert "List substrings in answer order" in block

    def test_reasoning_response_format_requests_one_item_per_occurrence(self, cache_file):
        """The object schema should describe repeated-substring occurrence handling."""
        detector = make_detector('{"hallucination_list": []}', cache_file, include_reasoning=True)

        block = detector._response_format_block()

        assert "items must be listed in answer order" in block
        assert "return at most one item per distinct occurrence" in block
