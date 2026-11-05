"""Golden (characterization) tests pinning today's ``LLMDetector`` output.

Every configuration runs through a fake ``LLMClient`` that returns fixed replies, so the tests are
network-free and need no API key. Each recording holds ``predict``, ``predict_prompt`` and
``predict_prompt_batch`` output for one configuration and one output format, in
``tests/fixtures/golden/llm_<config>_<format>.json``.

The ``llm_defect_*`` recordings are today's behaviour where the detectors epic changes it (spans
without ``confidence``, unparseable replies returning no spans, items missing from the answer); the
pull request that fixes one updates its fixture as an intended change.
"""

from __future__ import annotations

import json

import pytest

from lettucedetect.detectors.llm import LLMDetector
from lettucedetect.detectors.llm_client import LLMClient
from tests.golden import assert_golden

FORMATS = ["tokens", "spans"]

# (context passages, answer, question); the answers are not substrings of each other, so the fake
# client can tell which reply to give from the request text alone.
CASES = [
    (
        ["France has about 67 million people."],
        "The population of France is 69 million and Paris is its capital.",
        "What is the population of France?",
    ),
    (["Lisbon is the capital of Portugal."], "Lisbon is the capital of Spain.", None),
]
ANSWERS = [case[1] for case in CASES]
PROMPTS = ["Source: France has about 67 million people.", "Source: Lisbon is in Portugal."]

VERIFY_SYSTEM_MARKER = "strict fact-checker"


class FakeClient(LLMClient):
    """Returns a fixed reply per answer (judge/native) or per answer (verify), no network."""

    def __init__(self, replies: dict[str, str], verify_replies: dict[str, str] | None = None):
        """Store replies keyed by the answer they respond to."""
        self.replies = replies
        self.verify_replies = verify_replies or {}

    def complete(self, system, user, model, temperature, schema) -> str:
        """Pick the reply of the answer found in ``user``; verification has its own replies."""
        table = self.verify_replies if VERIFY_SYSTEM_MARKER in system else self.replies
        for answer, reply in table.items():
            if answer in user:
                return reply
        raise AssertionError(f"no fixed reply for request: {user[:200]!r}")


def _judge(*items_per_answer: list) -> dict[str, str]:
    return {
        answer: json.dumps({"hallucination_list": items})
        for answer, items in zip(ANSWERS, items_per_answer)
    }


def _native(*spans_per_answer: list) -> dict[str, str]:
    return {
        answer: json.dumps({"hallucinated_spans": spans})
        for answer, spans in zip(ANSWERS, spans_per_answer)
    }


def _verdicts(*rejected_per_answer: list) -> dict[str, str]:
    return {
        answer: json.dumps(
            {"verifications": [{"text": t, "reasoning": "r", "is_hallucination": False} for t in r]}
        )
        for answer, r in zip(ANSWERS, rejected_per_answer)
    }


PLAIN = _judge(["69 million"], ["Spain"])
REASONED = _judge(
    [
        {
            "text": "69 million",
            "reasoning": "source says 67 million",
            "confidence": 0.93,
            "is_hallucination": True,
        },
        {
            "text": "Paris",
            "reasoning": "not stated, but harmless",
            "confidence": 0.31,
            "is_hallucination": False,
        },
    ],
    [
        {
            "text": "Spain",
            "reasoning": "source says Portugal",
            "confidence": 0.88,
            "is_hallucination": True,
        }
    ],
)
TYPED = _judge(
    [{"text": "69 million", "category": "contradiction", "subcategory": "numerical"}],
    [{"text": "Spain", "category": "contradiction", "subcategory": "entity"}],
)
TYPED_CATEGORY_ONLY = _judge(
    [{"text": "69 million", "category": "contradiction"}],
    [{"text": "Spain", "category": "contradiction"}],
)
CUSTOM_CATEGORIES = {
    "wrong_number": "A number that differs from the source",
    "other": "Anything else",
}
NATIVE_PLAIN = _native(
    [{"text": "69 million", "category": "contradiction", "subcategory": "numerical"}],
    [{"text": "Spain", "category": "contradiction", "subcategory": "entity"}],
)
NATIVE_EXPLAINED = _native(
    [
        {
            "text": "69 million",
            "category": "contradiction",
            "subcategory": "numerical",
            "explanation": "source says 67 million",
        }
    ],
    [
        {
            "text": "Spain",
            "category": "contradiction",
            "subcategory": "entity",
            "explanation": "source says Portugal",
        }
    ],
)

# name -> (constructor kwargs, judge/native replies, verification replies)
CONFIGS = {
    "judge_defaults": ({}, PLAIN, None),
    "zero_shot": ({"zero_shot": True}, PLAIN, None),
    "include_reasoning": ({"include_reasoning": True}, REASONED, None),
    "taxonomy_true": ({"include_taxonomy": True}, TYPED, None),
    "taxonomy_list": (
        {"include_taxonomy": ["contradiction", "fabricated_reference"]},
        TYPED_CATEGORY_ONLY,
        None,
    ),
    "taxonomy_flat_dict": ({"include_taxonomy": CUSTOM_CATEGORIES}, TYPED_CATEGORY_ONLY, None),
    "taxonomy_nested_dict": (
        {
            "include_taxonomy": {
                "categories": CUSTOM_CATEGORIES,
                "subcategories": {"digits": "A wrong digit", "name": "A wrong name"},
            }
        },
        TYPED,
        None,
    ),
    "verify": (
        {"verify": True},
        _judge(["69 million", "Paris"], ["Spain"]),
        _verdicts(["Paris"], []),
    ),
    "native": ({"native": True}, NATIVE_PLAIN, None),
    "native_include_reasoning": (
        {"native": True, "include_reasoning": True},
        NATIVE_EXPLAINED,
        None,
    ),
    # Defect cases: today's output where later tickets change the behaviour.
    "defect_no_hallucination_list": (
        {},
        {answer: json.dumps({"analysis": "nothing to list"}) for answer in ANSWERS},
        None,
    ),
    "defect_unparseable_reply": ({}, {answer: "not json at all" for answer in ANSWERS}, None),
    "defect_item_not_in_answer": (
        {},
        _judge(["Mars", "69 million"], ["Spain", "Madrid"]),
        None,
    ),
    "defect_span_without_confidence": (
        {"include_reasoning": True},
        _judge(
            [{"text": "69 million", "reasoning": "no score given"}],
            [{"text": "Spain", "reasoning": "no score given"}],
        ),
        None,
    ),
}


def _run(name: str, output_format: str, tmp_path) -> dict:
    kwargs, replies, verify_replies = CONFIGS[name]
    kwargs = {"native": False, **kwargs}
    counter = iter(range(100))

    def detector() -> LLMDetector:
        # A fresh cache per entry point, so no entry point is answered from another's cache.
        return LLMDetector(
            client=FakeClient(replies, verify_replies),
            cache_file=str(tmp_path / f"cache_{next(counter)}.json"),
            **kwargs,
        )

    return {
        "predict": [
            detector().predict(ctx, ans, question, output_format=output_format)
            for ctx, ans, question in CASES
        ],
        "predict_prompt": [
            detector().predict_prompt(prompt, ans, output_format=output_format)
            for prompt, ans in zip(PROMPTS, ANSWERS)
        ],
        "predict_prompt_batch": detector().predict_prompt_batch(
            PROMPTS, ANSWERS, output_format=output_format
        ),
    }


@pytest.mark.parametrize("output_format", FORMATS)
@pytest.mark.parametrize("name", list(CONFIGS))
def test_llm_golden(name, output_format, tmp_path):
    """Pin every entry point for one configuration and one output format."""
    assert_golden(f"llm_{name}_{output_format}", _run(name, output_format, tmp_path))
