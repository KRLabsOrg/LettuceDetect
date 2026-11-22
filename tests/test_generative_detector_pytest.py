"""Tests for ``GenerativeDetector`` and ``GenerativeConfig``: offline, with a fake LLM client."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from lettucedetect.detectors import (
    FileCache,
    GenerativeConfig,
    GenerativeDetector,
    InMemoryCache,
    make_cache_key,
)
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
    SyncOnlyCache,
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


# -- cache and usage -------------------------------------------------------------------------


def _cached(replies: dict[str, str] | str = TYPED, cache=None, **kwargs) -> GenerativeDetector:
    cache = SyncOnlyCache() if cache is None else cache
    return GenerativeDetector(FakeClient(replies), MODEL, cache=cache, **kwargs)


def _key(detector: GenerativeDetector, call: int = 0, kind: str = "native") -> str:
    """Return the cache key of a call the detector's client received."""
    sent = detector.client.calls[call]
    prompt = sent["user"] if kind == "verify" else sent["system"] + sent["user"]
    return make_cache_key(prompt, MODEL, detector.model_version, detector.config.temperature, kind)


def _units(result: Detection) -> dict[str, int]:
    (usage,) = result.usage
    return usage.units


@pytest.mark.parametrize(
    ("request_", "source_chars"),
    [(SOURCES, sum(len(chunk.text) for chunk in SOURCES.sources)), (PROMPT, len(PROMPT.prompt))],
)
def test_usage_without_a_cache_counts_the_call(request_, source_chars):
    """``usage`` has one entry with the model id, the version and the four units."""
    detector = GenerativeDetector(FakeClient(TYPED), MODEL, model_version="v7")
    (usage,) = detector.detect(request_).usage
    assert usage.model_id == MODEL
    assert usage.model_version == "v7"
    assert usage.units == {
        "llm_calls": 1,
        "cache_hits": 0,
        "source_chars": source_chars,
        "answer_chars": len(ANSWER),
    }


def test_cache_hit_makes_no_client_call_and_is_counted_as_a_hit():
    """The second call is served from the cache under the ``make_cache_key`` native key."""
    detector = _cached()
    miss = detector.detect(SOURCES)
    hit = detector.detect(SOURCES)
    assert len(detector.client.calls) == 1
    assert detector.cache.get(_key(detector)) == TYPED
    assert (_units(miss)["llm_calls"], _units(miss)["cache_hits"]) == (1, 0)
    assert (_units(hit)["llm_calls"], _units(hit)["cache_hits"]) == (0, 1)
    assert hit.spans == miss.spans
    assert hit.warnings == miss.warnings


def test_another_model_version_misses():
    """The key holds the model version, so a new version does not read the old reply."""
    cache = SyncOnlyCache()
    old = _cached(cache=cache, model_version="v1")
    new = _cached(cache=cache, model_version="v2")
    old.detect(SOURCES)
    new.detect(SOURCES)
    assert len(new.client.calls) == 1
    assert _key(old) != _key(new)
    old.detect(SOURCES)
    assert len(old.client.calls) == 1


def test_unparseable_reply_is_never_stored():
    """A reply that does not parse raises and leaves the cache empty."""
    detector = _cached("not json at all")
    with pytest.raises(UpstreamError):
        detector.detect(SOURCES)
    assert detector.cache.get(_key(detector)) is None


def test_unparseable_cached_entry_is_deleted_and_raises():
    """A cached entry that does not parse is removed, the call raises and no client call is made."""
    detector = _cached()
    detector.detect(SOURCES)
    key = _key(detector)
    detector.cache.set(key, "garbage")
    with pytest.raises(UpstreamError) as exc:
        detector.detect(SOURCES)
    assert exc.value.code == "unparseable_response"
    assert detector.cache.get(key) is None
    assert len(detector.client.calls) == 1
    assert _units(detector.detect(SOURCES))["llm_calls"] == 1


async def test_adetect_has_the_cache_behaviour_of_detect():
    """Through ``adetect``: a miss writes after parsing, then a hit, with ``detect``'s usage."""
    detector = _cached()
    miss = await detector.adetect(SOURCES)
    assert detector.cache.get(_key(detector)) == TYPED
    hit = await detector.adetect(SOURCES)
    assert len(detector.client.calls) == 1
    assert miss == _cached().detect(SOURCES)
    assert hit == detector.detect(SOURCES)
    assert _units(hit)["cache_hits"] == 1


async def test_adetect_stores_no_unparseable_reply_and_deletes_an_unparseable_entry():
    """Through ``adetect``: a bad reply is not stored, a bad entry is deleted and raises."""
    bad = _cached("not json at all")
    with pytest.raises(UpstreamError):
        await bad.adetect(SOURCES)
    assert bad.cache.get(_key(bad)) is None

    detector = _cached()
    await detector.adetect(SOURCES)
    key = _key(detector)
    detector.cache.set(key, "garbage")
    with pytest.raises(UpstreamError) as exc:
        await detector.adetect(SOURCES)
    assert exc.value.code == "unparseable_response"
    assert detector.cache.get(key) is None
    assert len(detector.client.calls) == 1


@pytest.mark.parametrize("reply", [TYPED, "not json at all"])
async def test_cancelled_adetect_never_leaves_an_unparsed_reply_in_the_cache(reply):
    """After a cancel the thread finishes: the key is absent or holds the parsed reply."""
    entered, gate, returned = threading.Event(), threading.Event(), threading.Event()

    def blocked() -> None:
        entered.set()
        gate.wait(10)
        returned.set()

    cache = SyncOnlyCache()
    detector = GenerativeDetector(FakeClient(reply, on_call=blocked), MODEL, cache=cache)
    task = asyncio.create_task(detector.adetect(SOURCES))
    while not entered.is_set():
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    key = _key(detector)
    assert cache.get(key) is None
    gate.set()
    while not returned.is_set():
        await asyncio.sleep(0.005)
    await asyncio.sleep(0.05)  # let the abandoned thread parse and write

    if reply == TYPED:
        assert cache.get(key) == TYPED
        expected_counts = (0, 1)
    else:
        assert cache.get(key) is None
        detector.client.replies = TYPED
        expected_counts = (1, 0)
    result = await detector.adetect(SOURCES)
    assert [s.text for s in result.spans] == ["Berlin", "3 million"]
    assert (_units(result)["llm_calls"], _units(result)["cache_hits"]) == expected_counts


@pytest.mark.parametrize("backend", ["memory", "file"])
async def test_concurrent_adetect_calls_lose_no_cache_key(backend, tmp_path):
    """Concurrent calls with different requests each store their key and report their own usage."""
    cache = InMemoryCache() if backend == "memory" else FileCache(tmp_path / "cache.json")
    requests, replies = _numbered(6)
    client = FakeClient(replies, on_call=lambda: time.sleep(0.03))
    detector = GenerativeDetector(client, MODEL, cache=cache)
    first = await asyncio.gather(*(detector.adetect(request) for request in requests))
    assert all(
        (_units(r)["llm_calls"], _units(r)["cache_hits"], _units(r)["answer_chars"])
        == (1, 0, len(request.answer))
        for r, request in zip(first, requests, strict=True)
    )
    for call in range(6):
        assert cache.get(_key(detector, call)) in replies.values()
    assert len({_key(detector, call) for call in range(6)}) == 6

    second = await asyncio.gather(*(detector.adetect(request) for request in requests))
    assert len(client.calls) == 6
    assert all((_units(r)["llm_calls"], _units(r)["cache_hits"]) == (0, 1) for r in second)
    assert [r.spans for r in second] == [r.spans for r in first]


# -- verify pass -----------------------------------------------------------------------------

REJECT_BERLIN = json.dumps(
    {"verifications": [{"text": "Berlin", "reasoning": "r", "is_hallucination": False}]}
)
REJECT_NOTHING = json.dumps({"verifications": []})


def _verifying(verify: str | Exception, replies: str = TYPED, cache=None, **options):
    return GenerativeDetector(
        FakeClient(replies, verify=verify),
        MODEL,
        config=GenerativeConfig(verify=True, **options),
        cache=cache,
    )


async def test_verify_drops_rejected_spans_and_counts_its_call():
    """A rejected span is dropped, the others are kept, and ``llm_calls`` includes the verify call."""
    detector = _verifying(REJECT_BERLIN)
    result = detector.detect(SOURCES)
    assert [(s.text, s.category) for s in result.spans] == [("3 million", "contradiction")]
    assert _codes(result) == ["synthetic_score"]
    assert _units(result)["llm_calls"] == 2
    verify_call = detector.client.calls[1]
    assert "strict fact-checker" in verify_call["system"]
    assert "- Berlin\n- 3 million" in verify_call["user"]
    assert await _verifying(REJECT_BERLIN).adetect(SOURCES) == result


def test_verify_prompt_names_the_configured_language():
    """``lang`` reaches the verify prompt."""
    detector = _verifying(REJECT_NOTHING, lang="de")
    detector.detect(SOURCES)
    assert "German" in detector.client.calls[1]["user"]


def test_verify_is_not_called_without_spans():
    """With no first-pass span there is nothing to verify and one call is made."""
    detector = _verifying(REJECT_BERLIN, _reply())
    result = detector.detect(SOURCES)
    assert result.spans == []
    assert _units(result)["llm_calls"] == 1


@pytest.mark.parametrize(
    "verify",
    [
        "not valid json",
        json.dumps({"verdicts": []}),
        json.dumps({"verifications": "none"}),
        UpstreamError("down", code="upstream_unavailable"),
    ],
)
async def test_failed_verify_keeps_first_pass_spans_with_a_warning(verify, caplog):
    """A verify reply that can't be parsed keeps the spans, warns and logs ``fallback_used`` once."""
    with caplog.at_level(logging.DEBUG):
        result = _verifying(verify).detect(SOURCES)
    assert [s.text for s in result.spans] == ["Berlin", "3 million"]
    assert _codes(result) == ["verification_skipped", "synthetic_score"]
    assert _units(result)["llm_calls"] == 2
    fallbacks = [r for r in caplog.records if getattr(r, "event", None) == "fallback_used"]
    assert len(fallbacks) == 1
    assert fallbacks[0].levelno == logging.WARNING
    assert "verification_skipped" in fallbacks[0].getMessage()
    assert "Berlin" not in fallbacks[0].getMessage()
    assert await _verifying(verify).adetect(SOURCES) == result


@pytest.mark.parametrize(
    "error",
    [ConfigurationError("bad key", code="authentication_failed"), OwnClientError("boom")],
)
async def test_verify_fallback_does_not_swallow_other_errors(error):
    """Only ``UpstreamError`` triggers the fallback; other errors of the client come out."""
    with pytest.raises(type(error)) as sync_exc:
        _verifying(error).detect(SOURCES)
    assert sync_exc.value is error
    with pytest.raises(type(error)) as async_exc:
        await _verifying(error).adetect(SOURCES)
    assert async_exc.value is error


def test_verify_reply_is_cached_under_the_verify_key():
    """The verify reply has its own key; a second call makes no client call at all."""
    detector = _verifying(REJECT_BERLIN, cache=SyncOnlyCache())
    first = detector.detect(SOURCES)
    assert detector.cache.get(_key(detector, 1, kind="verify")) == REJECT_BERLIN
    second = detector.detect(SOURCES)
    assert len(detector.client.calls) == 2
    assert (_units(second)["llm_calls"], _units(second)["cache_hits"]) == (0, 2)
    assert second.spans == first.spans


def test_unparseable_verify_reply_is_not_cached():
    """A verify reply that can't be parsed is not stored, so the next call asks again."""
    detector = _verifying("not valid json", cache=SyncOnlyCache())
    detector.detect(SOURCES)
    assert detector.cache.get(_key(detector, 1, kind="verify")) is None
    result = detector.detect(SOURCES)
    assert (_units(result)["llm_calls"], _units(result)["cache_hits"]) == (1, 1)


def test_max_concurrency_counts_the_verify_call():
    """First-pass and verify calls of concurrent requests share the one bound."""
    requests, replies = _numbered(6)
    client = FakeClient(replies, on_call=lambda: time.sleep(0.03), verify=REJECT_NOTHING)
    detector = GenerativeDetector(
        client, MODEL, config=GenerativeConfig(verify=True, max_concurrency=1)
    )
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(detector.detect, requests))
    assert client.max_in_flight == 1
    assert len(client.calls) == 12
    assert all(len(result.spans) == 1 for result in results)
