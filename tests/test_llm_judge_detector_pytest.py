"""Tests for ``LLMJudgeDetector`` and ``LLMJudgeConfig``: offline, with a fake LLM client."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest

from lettucedetect.datasets.taxonomy import CATEGORY_DEFINITIONS, SUBCATEGORY_DEFINITIONS
from lettucedetect.detectors import (
    FileCache,
    InMemoryCache,
    LLMJudgeConfig,
    LLMJudgeDetector,
    make_cache_key,
)
from lettucedetect.detectors.errors import ConfigurationError, UpstreamError
from lettucedetect.detectors.llm_client import LLMClient
from lettucedetect.detectors.types import Chunk, Detection, PromptInput, SourcesInput
from tests.detector_contract import CONTENT_LOGGER, DetectorContract
from tests.golden import GOLDEN_DIR
from tests.test_golden_llm_pytest import ANSWERS, CASES, CONFIGS, PROMPTS
from tests.test_golden_llm_pytest import FakeClient as GoldenClient

MODEL = "judge-model"
ANSWER = "The capital of France is Berlin, and about 3 million people live there."
SOURCES = SourcesInput(
    sources=[
        Chunk(id="s1", text="Paris is the capital of France."),
        Chunk(id="s2", text="Paris has two million people."),
    ],
    answer=ANSWER,
    question="What is the capital of France?",
)
PROMPT = PromptInput(
    prompt="Answer from the context: Paris is the capital of France.", answer=ANSWER
)

PLAIN = json.dumps({"hallucination_list": ["Berlin", "3 million"]})
REASONED = json.dumps(
    {
        "analysis": "two claims differ from the source",
        "hallucination_list": [
            {
                "text": "Berlin",
                "reasoning": "the source says Paris",
                "confidence": 0.93,
                "is_hallucination": True,
            },
            {
                "text": "3 million",
                "reasoning": "the source says two million",
                "confidence": 0.6,
                "is_hallucination": True,
            },
            {
                "text": "France",
                "reasoning": "supported after all",
                "confidence": 0.2,
                "is_hallucination": False,
            },
        ],
    }
)


class FakeClient(LLMClient):
    """Replies per answer found in the prompt, and records calls, threads and calls in flight."""

    def __init__(
        self,
        replies: dict[str, str] | str,
        on_call: Callable[[], None] | None = None,
        verify: str | Exception | None = None,
    ) -> None:
        """Keep the replies (one for every request, or keyed by answer) and the hook.

        ``verify`` is the reply to every verify call, or the error it raises.
        """
        self.replies = replies
        self.verify = verify
        self.on_call = on_call
        self.calls: list[dict] = []
        self.thread_ids: list[int] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._guard = threading.Lock()

    def complete(self, system, user, model, temperature, schema) -> str:
        """Return the reply of the answer found in ``user``."""
        with self._guard:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self.calls.append(
                {"system": system, "user": user, "model": model, "temperature": temperature}
                | {"schema": schema}
            )
            self.thread_ids.append(threading.get_ident())
        try:
            if self.on_call:
                self.on_call()
            if "verifications" in schema["properties"]:
                if isinstance(self.verify, Exception):
                    raise self.verify
                return self.verify
            if isinstance(self.replies, str):
                return self.replies
            for answer, reply in self.replies.items():
                if answer in user:
                    return reply
            raise AssertionError("no fixed reply for the request")
        finally:
            with self._guard:
                self._in_flight -= 1


def _detector(replies: dict[str, str] | str = PLAIN, **options) -> LLMJudgeDetector:
    return LLMJudgeDetector(FakeClient(replies), MODEL, config=LLMJudgeConfig(**options))


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


class TestLLMJudgeDetectorContract(DetectorContract):
    """The detector contract holds with a fake client."""

    sources_request = SOURCES
    prompt_request = PROMPT

    def make_detector(self, on_call=None):  # noqa: D102
        return LLMJudgeDetector(
            FakeClient(REASONED, on_call), MODEL, config=LLMJudgeConfig(include_reasoning=True)
        )


# -- config ----------------------------------------------------------------------------------


def test_config_defaults():
    """The defaults are the ones of the epic."""
    assert LLMJudgeConfig().model_dump() == {
        "temperature": 0.0,
        "lang": "en",
        "zero_shot": False,
        "fewshot_path": None,
        "prompt_path": None,
        "include_reasoning": False,
        "include_taxonomy": False,
        "min_score": 0.0,
        "verify": False,
        "max_concurrency": 30,
    }


def test_config_is_frozen():
    """An option can't be changed after construction."""
    config = LLMJudgeConfig()
    with pytest.raises(Exception, match="frozen"):
        config.verify = True


@pytest.mark.parametrize(
    ("options", "code", "param"),
    [
        ({"bogus": 1}, "invalid_config", "config.bogus"),
        ({"min_score": 1.5}, "invalid_config", "config.min_score"),
        ({"lang": "xx"}, "invalid_language", "config.lang"),
        ({"fewshot_path": "no/such/examples.json"}, "fewshot_not_found", "config.fewshot_path"),
        ({"prompt_path": "no/such/prompt.txt"}, "invalid_config", "config.prompt_path"),
        ({"max_concurrency": 0}, "invalid_config", "config.max_concurrency"),
        ({"max_concurrency": -3}, "invalid_config", "config.max_concurrency"),
        ({"max_concurrency": True}, "invalid_config", "config.max_concurrency"),
    ],
)
def test_invalid_option_raises_configuration_error(options, code, param):
    """An unknown or invalid option raises ``ConfigurationError`` with its code and param."""
    with pytest.raises(ConfigurationError) as exc:
        LLMJudgeConfig(**options)
    assert exc.value.code == code
    assert exc.value.param == param


# -- constructor -----------------------------------------------------------------------------


def test_constructor_reads_no_environment_and_model_id_is_model(monkeypatch):
    """The detector works without an API key in the environment; ``model_id`` is ``model``."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    detector = LLMJudgeDetector(FakeClient(PLAIN), MODEL, model_version="2026-01")
    assert detector.model_id == MODEL
    assert detector.model_version == "2026-01"
    assert detector.config == LLMJudgeConfig()
    assert detector.cache is None
    assert detector.detect(SOURCES).spans


def test_client_gets_model_temperature_and_schema():
    """The client is called with the model, the configured temperature and the judge schema."""
    detector = _detector(temperature=0.7)
    detector.detect(SOURCES)
    (call,) = detector.client.calls
    assert call["model"] == MODEL
    assert call["temperature"] == 0.7
    assert call["schema"]["required"] == ["hallucination_list"]
    assert ANSWER in call["user"]


def test_zero_shot_leaves_the_examples_out_and_fewshot_path_replaces_them(tmp_path):
    """``zero_shot`` sends no example; ``fewshot_path`` sends the examples of that file."""
    examples = tmp_path / "examples.json"
    examples.write_text(
        json.dumps([{"source": "custom source", "answer": "a", "hallucination_list": ["a"]}]),
        encoding="utf-8",
    )
    default, zero_shot, custom = (
        _detector(),
        _detector(zero_shot=True),
        _detector(fewshot_path=str(examples)),
    )
    for detector in (default, zero_shot, custom):
        detector.detect(PROMPT)
    assert "<example1>" in default.client.calls[0]["user"]
    assert "<example1>" not in zero_shot.client.calls[0]["user"]
    assert "custom source" in custom.client.calls[0]["user"]
    assert "custom source" not in default.client.calls[0]["user"]


def test_prompt_path_replaces_the_template(tmp_path):
    """``prompt_path`` is the template the prompt is built from."""
    template = tmp_path / "prompt.txt"
    template.write_text("CUSTOM ${lang} ${context} ${answer} ${fewshot_block}", encoding="utf-8")
    detector = _detector(prompt_path=str(template), zero_shot=True)
    detector.detect(PROMPT)
    assert detector.client.calls[0]["user"].startswith("CUSTOM English Answer from the context")


# -- detect ----------------------------------------------------------------------------------


@pytest.mark.parametrize("request_", [SOURCES, PROMPT])
def test_spans_without_confidence_score_one_with_one_synthetic_score_warning(request_):
    """Without reasoning every span scores 1.0 and the result has one ``synthetic_score``."""
    result = _detector().detect(request_)
    assert [(s.text, s.score) for s in result.spans] == [("Berlin", 1.0), ("3 million", 1.0)]
    for span in result.spans:
        assert ANSWER[span.start : span.end] == span.text
        assert span.category is None and span.subcategory is None and span.reasoning is None
    assert _codes(result) == ["synthetic_score"]


def test_include_reasoning_scores_with_the_model_confidence():
    """With reasoning, ``score`` is the model's confidence and nothing is synthetic."""
    detector = _detector(REASONED, include_reasoning=True)
    result = detector.detect(SOURCES)
    assert [(s.text, s.score, s.reasoning) for s in result.spans] == [
        ("Berlin", 0.93, "the source says Paris"),
        ("3 million", 0.6, "the source says two million"),
    ]
    assert result.warnings == []
    assert detector.client.calls[0]["schema"]["required"] == ["analysis", "hallucination_list"]


def test_reasoned_span_without_confidence_is_synthetic():
    """A span the model gave no confidence for scores 1.0 and is reported once."""
    reply = json.dumps(
        {
            "hallucination_list": [
                {"text": "Berlin", "reasoning": "r", "confidence": 0.8},
                {"text": "3 million", "reasoning": "r"},
                {"text": "France", "reasoning": "r", "confidence": "high"},
            ]
        }
    )
    result = _detector(reply, include_reasoning=True).detect(SOURCES)
    assert [(s.text, s.score) for s in result.spans] == [
        ("Berlin", 0.8),
        ("3 million", 1.0),
        ("France", 1.0),
    ]
    assert _codes(result) == ["synthetic_score"]


def test_no_spans_means_no_synthetic_score_warning():
    """An empty list is a clean result without warnings."""
    result = _detector(json.dumps({"hallucination_list": []})).detect(SOURCES)
    assert result.spans == []
    assert result.warnings == []


def test_hallucination_list_as_a_json_string_is_decoded():
    """A list the model returned as a JSON-encoded string is read as the list."""
    reply = json.dumps({"hallucination_list": json.dumps(["Berlin"])})
    assert [s.text for s in _detector(reply).detect(SOURCES).spans] == ["Berlin"]


TYPED = json.dumps(
    {"hallucination_list": [{"text": "Berlin", "category": "first", "subcategory": "sub"}]}
)


@pytest.mark.parametrize(
    ("include_taxonomy", "categories", "subcategories"),
    [
        (True, list(CATEGORY_DEFINITIONS), list(SUBCATEGORY_DEFINITIONS)),
        (["first", "second"], ["first", "second"], None),
        ({"first": "the first kind", "second": "the second kind"}, ["first", "second"], None),
        (
            {"categories": {"first": "the first kind"}, "subcategories": {"sub": "a detail"}},
            ["first"],
            ["sub"],
        ),
    ],
)
def test_every_include_taxonomy_form_sets_the_labels(include_taxonomy, categories, subcategories):
    """Each form reaches the schema and the prompt, and spans carry the model's labels."""
    detector = _detector(TYPED, include_taxonomy=include_taxonomy)
    result = detector.detect(SOURCES)
    call = detector.client.calls[0]
    properties = call["schema"]["properties"]["hallucination_list"]["items"]["properties"]
    assert properties["category"]["enum"] == categories
    assert f'"{categories[0]}"' in call["user"]
    if subcategories is None:
        assert "subcategory" not in properties
    else:
        assert properties["subcategory"]["enum"] == subcategories
    assert [(s.category, s.subcategory) for s in result.spans] == [("first", "sub")]


def test_repeated_items_take_the_first_unused_match_without_overlaps():
    """A repeated item takes the next occurrence; an overlapping item is not located."""
    answer = "Paris is mentioned; Paris is repeated."
    reply = json.dumps({"hallucination_list": ["Paris", "Paris", "Paris", "mentioned; Paris"]})
    result = _detector(reply).detect(PromptInput(prompt="p", answer=answer))
    assert [(s.start, s.end, s.text) for s in result.spans] == [(0, 5, "Paris"), (20, 25, "Paris")]
    assert _codes(result) == ["span_not_found", "synthetic_score"]


def test_config_min_score_filters_before_the_call_min_score():
    """``LLMJudgeConfig.min_score`` drops spans first; the call's ``min_score`` filters the rest."""
    detector = _detector(REASONED, include_reasoning=True, min_score=0.7)
    assert [s.text for s in detector.detect(SOURCES).spans] == ["Berlin"]
    assert detector.detect(SOURCES, min_score=0.95).spans == []
    loose = _detector(REASONED, include_reasoning=True)
    assert [s.text for s in loose.detect(SOURCES, min_score=0.7).spans] == ["Berlin"]


def test_item_the_model_rejected_is_dropped_without_a_warning():
    """An item with ``is_hallucination`` false is left out and nothing is reported."""
    result = _detector(REASONED, include_reasoning=True).detect(SOURCES)
    assert "France" not in [s.text for s in result.spans]
    assert result.warnings == []


DEGRADED = json.dumps({"hallucination_list": ["Madrid", "Berlin", 7, {"reasoning": "no text"}, ""]})


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
    json.dumps({"analysis": "nothing to list"}),
    json.dumps({"hallucination_list": {"text": "Berlin"}}),
    json.dumps({"hallucination_list": "Berlin"}),
    json.dumps(["Berlin"]),
]


@pytest.mark.parametrize("reply", UNPARSEABLE)
def test_unparseable_reply_raises_upstream_error(reply):
    """A reply that is not JSON, or has no list under ``hallucination_list``, raises."""
    with pytest.raises(UpstreamError) as exc:
        _detector(reply).detect(SOURCES)
    assert exc.value.code == "unparseable_response"
    assert "Berlin" not in str(exc.value)


class OwnClientError(RuntimeError):
    """An error of the caller's own client."""


class RaisingClient(LLMClient):
    """A client whose call raises a fixed error."""

    def __init__(self, error: Exception) -> None:
        """Keep the error to raise."""
        self.error = error

    def complete(self, system, user, model, temperature, schema) -> str:
        """Raise the error."""
        raise self.error


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
    detector = LLMJudgeDetector(RaisingClient(error), MODEL)
    with pytest.raises(type(error)) as sync_exc:
        detector.detect(SOURCES)
    assert sync_exc.value is error
    with pytest.raises(type(error)) as async_exc:
        await detector.adetect(SOURCES)
    assert async_exc.value is error


def test_semaphore_is_released_when_the_client_raises():
    """A failed call gives its slot back, so the next call is not blocked."""
    detector = LLMJudgeDetector(
        RaisingClient(OwnClientError("boom")), MODEL, config=LLMJudgeConfig(max_concurrency=1)
    )
    for _ in range(3):
        with pytest.raises(OwnClientError):
            detector.detect(SOURCES)


# -- golden fixtures -------------------------------------------------------------------------

GOLDEN_NAMES = [
    name
    for name, (kwargs, _, _) in CONFIGS.items()
    if not kwargs.get("native")
    and name not in ("defect_no_hallucination_list", "defect_unparseable_reply")
]


def _golden_spans(name: str) -> dict:
    """Load the span fixture of one configuration, with 1.0 where a span had no confidence."""
    recorded = json.loads((GOLDEN_DIR / f"llm_{name}_spans.json").read_text(encoding="utf-8"))
    for entry in ("predict", "predict_prompt"):
        for spans in recorded[entry]:
            for span in spans:
                span.setdefault("confidence", 1.0)
    return recorded


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_output_equals_the_golden_fixtures(name):
    """``detect`` output equals what ``LLMDetector`` recorded, with 1.0 for unscored spans."""
    kwargs, replies, verify_replies = CONFIGS[name]
    detector = LLMJudgeDetector(
        GoldenClient(replies, verify_replies), "gpt-4.1-mini", config=LLMJudgeConfig(**kwargs)
    )
    recorded = _golden_spans(name)
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
    assert [_as_dicts(result) for result in from_sources] == recorded["predict"]
    assert [_as_dicts(result) for result in from_prompts] == recorded["predict_prompt"]


# -- adetect ---------------------------------------------------------------------------------


@pytest.mark.parametrize("request_", [SOURCES, PROMPT])
@pytest.mark.parametrize(
    ("reply", "options"), [(PLAIN, {}), (REASONED, {"include_reasoning": True})]
)
async def test_adetect_equals_detect_and_calls_the_client_off_the_loop(request_, reply, options):
    """``adetect`` returns what ``detect`` returns, and the slow client call leaves the loop free."""
    client = FakeClient(reply, on_call=lambda: time.sleep(0.2))
    detector = LLMJudgeDetector(client, MODEL, config=LLMJudgeConfig(**options))
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


@pytest.mark.parametrize("reply", UNPARSEABLE[:3])
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


def _numbered(count: int) -> tuple[list[PromptInput], dict[str, str]]:
    """Build ``count`` requests with different answers, and the reply of each answer."""
    words = [f"word-{chr(ord('a') + i)}" for i in range(count)]
    requests = [
        PromptInput(prompt="p", answer=f"The answer mentions {word} once.") for word in words
    ]
    replies = {
        request.answer: json.dumps({"hallucination_list": [word]})
        for request, word in zip(requests, words, strict=True)
    }
    return requests, replies


async def test_gathered_adetect_calls_return_their_own_results_in_call_order():
    """Each of several concurrent calls on one detector gets the reply of its own request."""
    requests, replies = _numbered(8)
    delays = iter([0.12, 0.0, 0.08, 0.02, 0.1, 0.0, 0.06, 0.04])
    guard = threading.Lock()

    def slow() -> None:
        with guard:
            delay = next(delays)
        time.sleep(delay)

    detector = LLMJudgeDetector(FakeClient(replies, on_call=slow), MODEL)
    results = await asyncio.gather(*(detector.adetect(request) for request in requests))
    assert [[s.text for s in result.spans] for result in results] == [
        [f"word-{chr(ord('a') + i)}"] for i in range(8)
    ]


async def test_max_concurrency_bounds_client_calls_across_detect_and_adetect():
    """With ``max_concurrency=2``, threads and gathered calls together keep 2 calls in flight."""
    requests, replies = _numbered(12)
    client = FakeClient(replies, on_call=lambda: time.sleep(0.05))
    detector = LLMJudgeDetector(client, MODEL, config=LLMJudgeConfig(max_concurrency=2))
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


class SyncOnlyCache(InMemoryCache):
    """A cache whose async methods fail, to show the detector uses the sync ones."""

    async def aget(self, key):  # noqa: D102
        raise AssertionError("aget was called")

    async def aset(self, key, value, *, ttl_seconds=None):  # noqa: D102
        raise AssertionError("aset was called")

    async def adelete(self, key):  # noqa: D102
        raise AssertionError("adelete was called")


def _cached(replies: dict[str, str] | str = PLAIN, cache=None, **kwargs) -> LLMJudgeDetector:
    cache = SyncOnlyCache() if cache is None else cache
    return LLMJudgeDetector(FakeClient(replies), MODEL, cache=cache, **kwargs)


def _key(detector: LLMJudgeDetector, call: int = 0, kind: str = "judge") -> str:
    """Return the cache key of a call the detector's client received."""
    return make_cache_key(
        detector.client.calls[call]["user"],
        MODEL,
        detector.model_version,
        detector.config.temperature,
        kind,
    )


def _units(result: Detection) -> dict[str, int]:
    (usage,) = result.usage
    return usage.units


@pytest.mark.parametrize(
    ("request_", "source_chars"),
    [(SOURCES, sum(len(chunk.text) for chunk in SOURCES.sources)), (PROMPT, len(PROMPT.prompt))],
)
def test_usage_without_a_cache_counts_the_call(request_, source_chars):
    """``usage`` has one entry with the model id, the version and the four units."""
    detector = LLMJudgeDetector(FakeClient(PLAIN), MODEL, model_version="v7")
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
    """The second call is served from the cache under the ``make_cache_key`` judge key."""
    detector = _cached()
    miss = detector.detect(SOURCES)
    hit = detector.detect(SOURCES)
    assert len(detector.client.calls) == 1
    assert detector.cache.get(_key(detector)) == PLAIN
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
    assert detector.cache.get(_key(detector)) == PLAIN
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


@pytest.mark.parametrize("reply", [PLAIN, "not json at all"])
async def test_cancelled_adetect_never_leaves_an_unparsed_reply_in_the_cache(reply):
    """After a cancel the thread finishes: the key is absent or holds the parsed reply."""
    entered, gate, returned = threading.Event(), threading.Event(), threading.Event()

    def blocked() -> None:
        entered.set()
        gate.wait(10)
        returned.set()

    cache = SyncOnlyCache()
    detector = LLMJudgeDetector(FakeClient(reply, on_call=blocked), MODEL, cache=cache)
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

    if reply == PLAIN:
        assert cache.get(key) == PLAIN
        result = await detector.adetect(SOURCES)
        assert [s.text for s in result.spans] == ["Berlin", "3 million"]
        assert (_units(result)["llm_calls"], _units(result)["cache_hits"]) == (0, 1)
    else:
        assert cache.get(key) is None
        detector.client.replies = PLAIN
        result = await detector.adetect(SOURCES)
        assert [s.text for s in result.spans] == ["Berlin", "3 million"]
        assert (_units(result)["llm_calls"], _units(result)["cache_hits"]) == (1, 0)


@pytest.mark.parametrize("backend", ["memory", "file"])
async def test_concurrent_adetect_calls_lose_no_cache_key(backend, tmp_path):
    """Concurrent calls with different requests each store their key and report their own usage."""
    cache = InMemoryCache() if backend == "memory" else FileCache(tmp_path / "cache.json")
    requests, replies = _numbered(6)
    client = FakeClient(replies, on_call=lambda: time.sleep(0.03))
    detector = LLMJudgeDetector(client, MODEL, cache=cache)
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


def _verdicts(**rejected: bool) -> str:
    return json.dumps(
        {
            "verifications": [
                {"text": text.replace("_", " "), "reasoning": "r", "is_hallucination": verdict}
                for text, verdict in rejected.items()
            ]
        }
    )


REJECT_BERLIN = _verdicts(Berlin=False)


def _verifying(verify: str | Exception, replies: str = PLAIN, cache=None, **options):
    return LLMJudgeDetector(
        FakeClient(replies, verify=verify),
        MODEL,
        config=LLMJudgeConfig(verify=True, **options),
        cache=cache,
    )


async def test_verify_drops_rejected_spans_and_counts_its_call():
    """A rejected span is dropped, the others are kept, and ``llm_calls`` includes the verify call."""
    detector = _verifying(REJECT_BERLIN)
    result = detector.detect(SOURCES)
    assert [s.text for s in result.spans] == ["3 million"]
    assert _codes(result) == ["synthetic_score"]
    assert _units(result)["llm_calls"] == 2
    verify_call = detector.client.calls[1]
    assert "strict fact-checker" in verify_call["system"]
    assert "- Berlin\n- 3 million" in verify_call["user"]
    assert await _verifying(REJECT_BERLIN).adetect(SOURCES) == result


def test_verify_is_not_called_without_spans():
    """With no first-pass span there is nothing to verify and one call is made."""
    detector = _verifying(REJECT_BERLIN, json.dumps({"hallucination_list": []}))
    result = detector.detect(SOURCES)
    assert result.spans == []
    assert _units(result)["llm_calls"] == 1


def test_config_min_score_is_applied_before_the_verify_pass():
    """A span below ``LLMJudgeConfig.min_score`` is not sent to the verifier."""
    detector = _verifying(_verdicts(), REASONED, include_reasoning=True, min_score=0.7)
    result = detector.detect(SOURCES)
    assert [s.text for s in result.spans] == ["Berlin"]
    assert "- Berlin" in detector.client.calls[1]["user"]
    assert "- 3 million" not in detector.client.calls[1]["user"]


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
    client = FakeClient(replies, on_call=lambda: time.sleep(0.03), verify=_verdicts())
    detector = LLMJudgeDetector(
        client, MODEL, config=LLMJudgeConfig(verify=True, max_concurrency=1)
    )
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(detector.detect, requests))
    assert client.max_in_flight == 1
    assert len(client.calls) == 12
    assert all(len(result.spans) == 1 for result in results)
