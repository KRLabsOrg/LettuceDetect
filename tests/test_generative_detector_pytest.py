"""Tests for ``GenerativeDetector`` and ``GenerativeConfig``: offline, with a fake LLM client."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from lettucedetect.detectors import GenerativeConfig, GenerativeDetector
from lettucedetect.detectors.errors import ConfigurationError, UpstreamError
from lettucedetect.detectors.types import Chunk, Detection, PromptInput, SourcesInput
from lettucedetect.prompts import generative as gen
from tests.detector_contract import CONTENT_LOGGER, DetectorContract
from tests.golden import GOLDEN_DIR
from tests.test_golden_llm_pytest import ANSWERS, CASES, CONFIGS, PROMPTS
from tests.test_golden_llm_pytest import FakeClient as GoldenClient
from tests.test_llm_judge_detector_pytest import (
    ANSWER,
    PROMPT,
    SOURCES,
    FakeClient,
    OwnClientError,
    RaisingClient,
)

MODEL = "generative-model"


def _reply(*spans: object) -> str:
    return json.dumps({"hallucinated_spans": list(spans)})


BERLIN = {"text": "Berlin", "category": "contradiction", "subcategory": "entity"}
MILLION = {"text": "3 million", "category": "contradiction", "subcategory": "numerical"}
TYPED = _reply(BERLIN, MILLION)
EXPLAINED = _reply(
    BERLIN | {"explanation": "the source says Paris"},
    MILLION | {"explanation": "the source says two million"},
)


def _detector(replies: dict[str, str] | str = TYPED, **options) -> GenerativeDetector:
    return GenerativeDetector(FakeClient(replies), MODEL, config=GenerativeConfig(**options))


def _codes(result: Detection) -> list[str]:
    return [warning.code for warning in result.warnings]


def _as_dicts(result: Detection) -> list[dict]:
    """Render spans as the ``predict*`` methods do: ``confidence`` from ``score``."""
    rendered = []
    for span in result.spans:
        item = {"start": span.start, "end": span.end, "text": span.text, "confidence": span.score}
        for key in ("category", "subcategory", "reasoning"):
            if getattr(span, key) is not None:
                item[key] = getattr(span, key)
        rendered.append(item)
    return rendered


def _numbered(count: int) -> tuple[list[PromptInput], dict[str, str]]:
    """Build ``count`` requests with different answers, and the reply of each answer."""
    words = [f"word-{chr(ord('a') + i)}" for i in range(count)]
    requests = [
        PromptInput(prompt="p", answer=f"The answer mentions {word} once.") for word in words
    ]
    replies = {
        request.answer: _reply({"text": word, "category": "c", "subcategory": "s"})
        for request, word in zip(requests, words, strict=True)
    }
    return requests, replies


class TestGenerativeDetectorContract(DetectorContract):
    """The detector contract holds with a fake client."""

    sources_request = SOURCES
    prompt_request = PROMPT

    def make_detector(self, on_call=None):  # noqa: D102
        return GenerativeDetector(FakeClient(TYPED, on_call), MODEL)

    def _distinct_scores(self, request):
        """Return the one score there is: every span of this detector scores 1.0."""
        assert {span.score for span in self.make_detector().detect(request).spans} == {1.0}
        return [1.0]

    async def test_concurrent_adetect_returns_results_in_call_order(self) -> None:
        """Concurrent calls finish out of order and return in call order.

        The requests differ instead of the thresholds, which can't tell results apart here.
        """
        requests, replies = _numbered(3)
        detectors = [
            GenerativeDetector(FakeClient(replies, on_call=lambda d=delay: time.sleep(d)), MODEL)
            for delay in (0.15, 0.08, 0.0)
        ]
        expected = [GenerativeDetector(FakeClient(replies), MODEL).detect(r) for r in requests]
        assert len({e.spans[0].text for e in expected}) == 3
        results = await asyncio.gather(
            *(d.adetect(r) for d, r in zip(detectors, requests, strict=True))
        )
        assert list(results) == expected


# -- config ----------------------------------------------------------------------------------


def test_config_defaults():
    """The defaults are the ones of the epic."""
    assert GenerativeConfig().model_dump() == {
        "temperature": 0.0,
        "lang": "en",
        "include_reasoning": False,
        "include_taxonomy": False,
        "verify": False,
        "max_concurrency": 30,
    }


def test_config_is_frozen():
    """An option can't be changed after construction."""
    config = GenerativeConfig()
    with pytest.raises(Exception, match="frozen"):
        config.verify = True


@pytest.mark.parametrize(
    ("options", "code", "param"),
    [
        ({"bogus": 1}, "invalid_config", "config.bogus"),
        ({"zero_shot": True}, "invalid_config", "config.zero_shot"),
        ({"fewshot_path": "examples.json"}, "invalid_config", "config.fewshot_path"),
        ({"prompt_path": "prompt.txt"}, "invalid_config", "config.prompt_path"),
        ({"min_score": 0.5}, "invalid_config", "config.min_score"),
        ({"lang": "xx"}, "invalid_language", "config.lang"),
        ({"max_concurrency": 0}, "invalid_config", "config.max_concurrency"),
        ({"max_concurrency": -3}, "invalid_config", "config.max_concurrency"),
        ({"max_concurrency": True}, "invalid_config", "config.max_concurrency"),
    ],
)
def test_invalid_option_raises_configuration_error(options, code, param):
    """A judge-only, unknown or invalid option raises ``ConfigurationError`` with code and param."""
    with pytest.raises(ConfigurationError) as exc:
        GenerativeConfig(**options)
    assert exc.value.code == code
    assert exc.value.param == param


# -- constructor -----------------------------------------------------------------------------


def test_constructor_reads_no_environment(monkeypatch):
    """The detector works without an API key in the environment."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    detector = GenerativeDetector(FakeClient(TYPED), MODEL, model_version="2026-01")
    assert detector.model_id == MODEL
    assert detector.model_version == "2026-01"
    assert detector.config == GenerativeConfig()
    assert detector.cache is None
    assert detector.detect(SOURCES).spans


def test_client_gets_the_frozen_prompt_the_model_the_temperature_and_the_schema():
    """Both request kinds send the frozen system prompt and the generative user message."""
    detector = _detector(temperature=0.7)
    detector.detect(SOURCES)
    detector.detect(PROMPT)
    from_sources, from_prompt = detector.client.calls
    for call in (from_sources, from_prompt):
        assert call["system"] == gen.SYSTEM_BASE
        assert call["model"] == MODEL
        assert call["temperature"] == 0.7
        assert call["schema"]["required"] == ["hallucinated_spans"]
    assert from_prompt["user"] == gen.build_user_message(PROMPT.prompt, ANSWER)
    assert from_sources["user"].endswith(f"\n\nAnswer to verify:\n{ANSWER}")
    assert all(chunk.text in from_sources["user"] for chunk in SOURCES.sources)
    assert SOURCES.question in from_sources["user"]


# -- detect ----------------------------------------------------------------------------------


@pytest.mark.parametrize("request_", [SOURCES, PROMPT])
def test_spans_are_typed_and_score_one_with_one_synthetic_score_warning(request_):
    """Every span has the reply's labels and score 1.0; the result has one ``synthetic_score``."""
    result = _detector().detect(request_)
    assert [(s.text, s.score, s.category, s.subcategory) for s in result.spans] == [
        ("Berlin", 1.0, "contradiction", "entity"),
        ("3 million", 1.0, "contradiction", "numerical"),
    ]
    for span in result.spans:
        assert ANSWER[span.start : span.end] == span.text
        assert span.reasoning is None
    assert _codes(result) == ["synthetic_score"]


def test_include_reasoning_asks_for_explanations_and_returns_them_as_reasoning():
    """``explanation`` of the reply becomes ``reasoning``; the score stays synthetic."""
    detector = _detector(EXPLAINED, include_reasoning=True)
    result = detector.detect(SOURCES)
    assert [(s.text, s.score, s.reasoning) for s in result.spans] == [
        ("Berlin", 1.0, "the source says Paris"),
        ("3 million", 1.0, "the source says two million"),
    ]
    assert _codes(result) == ["synthetic_score"]
    (call,) = detector.client.calls
    assert call["system"] == gen.SYSTEM_EXPL
    assert "explanation" in call["schema"]["properties"]["hallucinated_spans"]["items"]["required"]


def test_a_confidence_or_verdict_in_the_reply_is_not_read():
    """The generative reply has no confidence or verdict: such keys change nothing."""
    reply = _reply(
        BERLIN | {"confidence": 0.2, "is_hallucination": False, "reasoning": "not the key"}
    )
    result = _detector(reply).detect(SOURCES)
    assert [(s.text, s.score, s.reasoning) for s in result.spans] == [("Berlin", 1.0, None)]
    assert _codes(result) == ["synthetic_score"]


def test_no_spans_means_no_warning():
    """An empty list is a clean result without warnings."""
    result = _detector(_reply()).detect(SOURCES)
    assert result.spans == []
    assert result.warnings == []


@pytest.mark.parametrize("reply", [f"```json\n{TYPED}\n```", f"Here you go: {TYPED} Done."])
def test_fenced_or_embedded_json_is_read(reply):
    """Code fences and text around the JSON object are tolerated."""
    assert [s.text for s in _detector(reply).detect(SOURCES).spans] == ["Berlin", "3 million"]


CUSTOM = _reply({"text": "Berlin", "category": "first", "subcategory": "sub"})


@pytest.mark.parametrize(
    ("include_taxonomy", "categories", "subcategories"),
    [
        (["first", "second"], ["first", "second"], ["unspecified"]),
        (
            {"first": "the first kind", "second": "the second kind"},
            ["first", "second"],
            ["unspecified"],
        ),
        (
            {"categories": {"first": "the first kind"}, "subcategories": {"sub": "a detail"}},
            ["first"],
            ["sub"],
        ),
    ],
)
def test_custom_taxonomy_forms_set_the_labels(include_taxonomy, categories, subcategories):
    """Each custom form reaches the schema and the prompt, and spans carry the model's labels."""
    detector = _detector(CUSTOM, include_taxonomy=include_taxonomy)
    result = detector.detect(SOURCES)
    (call,) = detector.client.calls
    properties = call["schema"]["properties"]["hallucinated_spans"]["items"]["properties"]
    assert properties["category"]["enum"] == categories
    assert properties["subcategory"]["enum"] == subcategories
    assert f"- {categories[0]}" in call["system"]
    assert "unsupported_addition" not in call["system"]
    assert [(s.category, s.subcategory) for s in result.spans] == [("first", "sub")]


@pytest.mark.parametrize("include_taxonomy", [False, True])
def test_bool_taxonomy_keeps_the_trained_labels(include_taxonomy):
    """``True`` and ``False`` both send the frozen prompt and the trained label enums."""
    detector = _detector(include_taxonomy=include_taxonomy)
    result = detector.detect(SOURCES)
    (call,) = detector.client.calls
    properties = call["schema"]["properties"]["hallucinated_spans"]["items"]["properties"]
    assert call["system"] == gen.SYSTEM_BASE
    assert properties["category"]["enum"] == list(gen.CATEGORY_DESCRIPTIONS)
    assert properties["subcategory"]["enum"] == list(gen.SUBCATEGORY_DESCRIPTIONS)
    assert [s.category for s in result.spans] == ["contradiction", "contradiction"]


def test_repeated_items_take_the_first_unused_match_without_overlaps():
    """A repeated item takes the next occurrence; an overlapping item is not located."""
    answer = "Paris is mentioned; Paris is repeated."
    paris = {"text": "Paris", "category": "c", "subcategory": "s"}
    reply = _reply(paris, paris, paris, paris | {"text": "mentioned; Paris"})
    result = _detector(reply).detect(PromptInput(prompt="p", answer=answer))
    assert [(s.start, s.end, s.text) for s in result.spans] == [(0, 5, "Paris"), (20, 25, "Paris")]
    assert _codes(result) == ["span_not_found", "synthetic_score"]


DEGRADED = _reply(BERLIN | {"text": "Madrid"}, BERLIN, 7, {"category": "contradiction"}, "")


def test_unlocated_and_malformed_items_are_dropped_with_warnings():
    """An item that is not in the answer and a malformed item are dropped and reported."""
    result = _detector(DEGRADED).detect(SOURCES)
    assert [s.text for s in result.spans] == ["Berlin"]
    assert _codes(result) == ["span_not_found", "item_malformed", "synthetic_score"]
    for warning in result.warnings:
        assert "Madrid" not in warning.message
        assert "Berlin" not in warning.message


def test_customer_text_is_logged_only_on_the_content_logger(caplog):
    """Module loggers carry counts and codes; the raw reply is on ``lettucedetect.content``."""
    with caplog.at_level(logging.DEBUG):
        _detector(DEGRADED).detect(SOURCES)
    content = [r for r in caplog.records if r.name == CONTENT_LOGGER]
    others = [r for r in caplog.records if r.name != CONTENT_LOGGER]
    assert any(DEGRADED in r.getMessage() and r.levelno == logging.DEBUG for r in content)
    dropped = [r for r in others if r.levelno == logging.WARNING]
    assert len(dropped) == 1
    assert "span_not_found=1" in dropped[0].getMessage()
    assert "item_malformed=3" in dropped[0].getMessage()
    for record in others:
        for text in ("Madrid", "Berlin", ANSWER):
            assert text not in record.getMessage()


UNPARSEABLE = [
    "not json at all",
    "",
    "{not json}",
    json.dumps({"analysis": "nothing to list"}),
    json.dumps({"hallucination_list": ["Berlin"]}),
    json.dumps({"hallucinated_spans": BERLIN}),
    json.dumps({"hallucinated_spans": "Berlin"}),
    json.dumps([BERLIN]),
]


@pytest.mark.parametrize("reply", UNPARSEABLE)
def test_unparseable_reply_raises_upstream_error(reply):
    """A reply that is not JSON, or has no list under ``hallucinated_spans``, raises."""
    with pytest.raises(UpstreamError) as exc:
        _detector(reply).detect(SOURCES)
    assert exc.value.code == "unparseable_response"
    assert "Berlin" not in str(exc.value)


@pytest.mark.parametrize(
    "error",
    [
        OwnClientError("boom"),
        ConfigurationError("bad key", code="authentication_failed"),
        UpstreamError("down", code="upstream_unavailable"),
    ],
)
async def test_client_errors_propagate_unwrapped_from_detect_and_adetect(error):
    """The client's own error object comes out of ``detect`` and out of the ``await``."""
    detector = GenerativeDetector(RaisingClient(error), MODEL)
    with pytest.raises(type(error)) as sync_exc:
        detector.detect(SOURCES)
    assert sync_exc.value is error
    with pytest.raises(type(error)) as async_exc:
        await detector.adetect(SOURCES)
    assert async_exc.value is error


def test_semaphore_is_released_when_the_client_raises():
    """A failed call gives its slot back, so the next call is not blocked."""
    detector = GenerativeDetector(
        RaisingClient(OwnClientError("boom")), MODEL, config=GenerativeConfig(max_concurrency=1)
    )
    for _ in range(3):
        with pytest.raises(OwnClientError):
            detector.detect(SOURCES)


# -- golden fixtures -------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["native", "native_include_reasoning"])
def test_output_equals_the_golden_fixtures_with_score_one(name):
    """``detect`` output equals the spans recorded for the native ``predict*``, scored 1.0."""
    kwargs, replies, _ = CONFIGS[name]
    options = {key: value for key, value in kwargs.items() if key != "native"}
    detector = GenerativeDetector(
        GoldenClient(replies), "gpt-4.1-mini", config=GenerativeConfig(**options)
    )
    recorded = json.loads((GOLDEN_DIR / f"llm_{name}_spans.json").read_text(encoding="utf-8"))
    from_sources = [
        detector.detect(
            SourcesInput(
                sources=[Chunk(id=str(i), text=text) for i, text in enumerate(context)],
                answer=answer,
                question=question,
            )
        )
        for context, answer, question in CASES
    ]
    from_prompts = [
        detector.detect(PromptInput(prompt=prompt, answer=answer))
        for prompt, answer in zip(PROMPTS, ANSWERS, strict=True)
    ]

    def scored(recording: list[list[dict]]) -> list[list[dict]]:
        return [[span | {"confidence": 1.0} for span in spans] for spans in recording]

    assert [_as_dicts(result) for result in from_sources] == scored(recorded["predict"])
    assert [_as_dicts(result) for result in from_prompts] == scored(recorded["predict_prompt"])


# -- adetect ---------------------------------------------------------------------------------


@pytest.mark.parametrize("request_", [SOURCES, PROMPT])
@pytest.mark.parametrize(
    ("reply", "options"), [(TYPED, {}), (EXPLAINED, {"include_reasoning": True})]
)
async def test_adetect_equals_detect_and_calls_the_client_off_the_loop(request_, reply, options):
    """``adetect`` returns what ``detect`` returns, and the slow client call leaves the loop free."""
    client = FakeClient(reply, on_call=lambda: time.sleep(0.2))
    detector = GenerativeDetector(client, MODEL, config=GenerativeConfig(**options))
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        result = await detector.adetect(request_)
    finally:
        task.cancel()
    assert client.thread_ids == [client.thread_ids[0]]
    assert client.thread_ids[0] != threading.get_ident()
    assert ticks >= 3
    assert result == detector.detect(request_)
    assert result.spans


@pytest.mark.parametrize("reply", UNPARSEABLE[:4])
async def test_adetect_raises_unparseable_response(reply):
    """The unparseable reply raises the same error out of the ``await``."""
    with pytest.raises(UpstreamError) as exc:
        await _detector(reply).adetect(SOURCES)
    assert exc.value.code == "unparseable_response"


async def test_adetect_reports_dropped_items():
    """``span_not_found`` and ``item_malformed`` are in the awaited result."""
    result = await _detector(DEGRADED).adetect(SOURCES)
    assert _codes(result) == ["span_not_found", "item_malformed", "synthetic_score"]
    assert result == _detector(DEGRADED).detect(SOURCES)


async def test_max_concurrency_bounds_client_calls_across_detect_and_adetect():
    """With ``max_concurrency=2``, threads and gathered calls together keep 2 calls in flight."""
    requests, replies = _numbered(12)
    client = FakeClient(replies, on_call=lambda: time.sleep(0.05))
    detector = GenerativeDetector(client, MODEL, config=GenerativeConfig(max_concurrency=2))
    from_threads, from_loop = requests[:6], requests[6:]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(detector.detect, request) for request in from_threads]
        awaited = await asyncio.gather(*(detector.adetect(request) for request in from_loop))
        results = [future.result() for future in futures] + list(awaited)
    assert client.max_in_flight == 2
    assert len(client.calls) == 12
    assert [[s.text for s in result.spans] for result in results] == [
        [f"word-{chr(ord('a') + i)}"] for i in range(12)
    ]
