"""Tests for ``SystemOneClient`` and ``VelaDetector.from_server``: offline, on a mock transport."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable

import httpx
import pytest

from lettucedetect.detectors import VelaDetector, vela_client
from lettucedetect.detectors.errors import ConfigurationError, InvalidInputError, UpstreamError
from lettucedetect.detectors.types import Chunk, PromptInput, SourcesInput
from lettucedetect.detectors.vela_client import SystemOneClient
from tests.detector_contract import DetectorContract
from tests.test_vela_detector_pytest import (
    PROMPT,
    SOURCES,
    FakeVela,
    _RecordingEnviron,
)

BASE_URL = "http://vela.test"
MODEL = "vllm-sr/Vela-2.0-0.3B"
API_KEY = "ZEBRA-KEY"
SECRET = "ZEBRA-ANSWER-TEXT"  # noqa: S105
STATE = {"source": "Ships Friday.", "answer": "Ships Monday."}
QUESTIONS = {"halu": {"type": "span"}}

Handler = Callable[[httpx.Request], httpx.Response]


def _served(model: FakeVela) -> Handler:
    """Return a handler that answers ``/v1/systemone`` like ``vela2_serve.py`` with ``model``."""

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json=model.system_one(body["state"], body["questions"]))

    return handle


def _client(handler: Handler, api_key: str | None = API_KEY) -> SystemOneClient:
    """Build a client whose requests go to ``handler`` instead of the network."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    http = httpx.Client(base_url=BASE_URL, headers=headers, transport=httpx.MockTransport(handler))
    return SystemOneClient(BASE_URL, model=MODEL, http_client=http)


@pytest.fixture
def mock_transport(monkeypatch):
    """Route every ``httpx.Client`` that ``SystemOneClient`` builds to a handler set by the test.

    Yields a dict; the test puts its handler under ``"handler"`` and reads the keyword arguments
    the client was built with from ``"kwargs"``.
    """
    route: dict[str, object] = {"handler": _served(FakeVela())}
    real_client = httpx.Client

    def build(**kwargs: object) -> httpx.Client:
        route["kwargs"] = kwargs
        return real_client(**kwargs, transport=httpx.MockTransport(lambda r: route["handler"](r)))

    monkeypatch.setattr(vela_client.httpx, "Client", build)
    return route


def _served_detector(model: FakeVela | None = None) -> VelaDetector:
    return VelaDetector(
        _client(_served(model or FakeVela())), model_id=MODEL, model_version="abc123"
    )


class TestServedVelaDetectorContract(DetectorContract):
    """The shared detector contract holds for a ``VelaDetector`` that calls a server."""

    sources_request = SOURCES
    prompt_request = PROMPT

    def make_detector(self, on_call=None):  # noqa: D102
        return _served_detector(FakeVela(on_call=on_call))


# -- the request ----------------------------------------------------------------------------


def test_request_is_a_systemone_post_with_state_model_and_questions():
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"spans": {"halu": []}})

    reply = _client(handle).system_one(state=STATE, questions=QUESTIONS)
    assert reply == {"spans": {"halu": []}}
    (request,) = seen
    assert (request.method, request.url.path) == ("POST", "/v1/systemone")
    assert json.loads(request.content) == {"state": STATE, "model": MODEL, "questions": QUESTIONS}


def test_from_server_sends_the_key_and_trusts_no_environment(mock_transport):
    seen: list[httpx.Request] = []
    mock_transport["handler"] = lambda r: seen.append(r) or _served(FakeVela())(r)
    detector = VelaDetector.from_server(BASE_URL, model=MODEL, api_key=API_KEY, timeout=5.0)
    detector.detect(SOURCES)
    assert seen[0].headers["authorization"] == f"Bearer {API_KEY}"
    assert str(seen[0].url) == f"{BASE_URL}/v1/systemone"
    assert mock_transport["kwargs"]["trust_env"] is False
    assert mock_transport["kwargs"]["timeout"] == 5.0


def test_no_key_sends_no_authorization_header(mock_transport):
    seen: list[httpx.Request] = []
    mock_transport["handler"] = lambda r: seen.append(r) or _served(FakeVela())(r)
    VelaDetector.from_server(BASE_URL, model=MODEL).detect(SOURCES)
    assert "authorization" not in seen[0].headers


@pytest.mark.parametrize(
    ("kwargs", "param"),
    [({"base_url": " "}, "base_url"), ({"model": ""}, "model"), ({"timeout": 0}, "timeout")],
)
def test_invalid_options_raise_configuration_error(kwargs, param):
    options = {"base_url": BASE_URL, "model": MODEL, **kwargs}
    with pytest.raises(ConfigurationError) as exc:
        VelaDetector.from_server(options.pop("base_url"), **options)
    assert (exc.value.code, exc.value.param) == ("invalid_config", param)


# -- errors ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "error", "code", "retryable", "retry_after"),
    [
        (
            httpx.Response(401, json={"detail": "Missing or invalid API key"}),
            ConfigurationError,
            "invalid_api_key",
            None,
            None,
        ),
        (httpx.Response(403), ConfigurationError, "invalid_api_key", None, None),
        (
            httpx.Response(
                422, json={"detail": [{"loc": ["body", "model"], "msg": "no", "type": "x"}]}
            ),
            ConfigurationError,
            "model_not_served",
            None,
            None,
        ),
        (
            httpx.Response(413, json={"detail": [{"loc": [], "msg": SECRET, "type": "too_long"}]}),
            InvalidInputError,
            "input_too_long",
            None,
            None,
        ),
        (
            httpx.Response(429, headers={"Retry-After": "7"}),
            UpstreamError,
            "rate_limited",
            True,
            7.0,
        ),
        (httpx.Response(429), UpstreamError, "rate_limited", True, None),
        (httpx.Response(503, text=SECRET), UpstreamError, "server_error", True, None),
        (
            httpx.Response(
                422, json={"detail": [{"loc": ["body", "state"], "msg": SECRET, "type": "x"}]}
            ),
            UpstreamError,
            "request_rejected",
            False,
            None,
        ),
        (httpx.Response(404), UpstreamError, "request_rejected", False, None),
        (httpx.Response(200, text="not json"), UpstreamError, "invalid_model_output", False, None),
        (
            httpx.Response(200, json=["a", "list"]),
            UpstreamError,
            "invalid_model_output",
            False,
            None,
        ),
    ],
)
def test_failed_replies_raise_typed_errors(response, error, code, retryable, retry_after):
    detector = VelaDetector(_client(lambda r: response), model_id=MODEL)
    with pytest.raises(error) as exc:
        detector.detect(SourcesInput(sources=[Chunk(id="s", text=SECRET)], answer=SECRET))
    assert exc.value.code == code
    if error is UpstreamError:
        assert exc.value.service == "vela"
        assert (exc.value.retryable, exc.value.retry_after) == (retryable, retry_after)
    if error is ConfigurationError:
        assert exc.value.param == ("model" if code == "model_not_served" else "api_key")
    assert SECRET not in str(exc.value)
    assert API_KEY not in str(exc.value)


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        (httpx.ReadTimeout("timed out"), "timeout"),
        (httpx.ConnectTimeout("timed out"), "timeout"),
        (httpx.ConnectError("refused"), "connection_failed"),
        (httpx.RemoteProtocolError("closed"), "connection_failed"),
    ],
)
def test_transport_failures_raise_retryable_upstream_errors(raised, code):
    def handle(request: httpx.Request) -> httpx.Response:
        raise raised

    with pytest.raises(UpstreamError) as exc:
        _client(handle).system_one(state=STATE, questions=QUESTIONS)
    assert (exc.value.code, exc.value.service, exc.value.retryable) == (code, "vela", True)
    assert exc.value.__cause__ is raised


# -- the detector over a server -------------------------------------------------------------


def test_served_spans_and_usage_match_the_local_model():
    local = VelaDetector(FakeVela(), model_id=MODEL, model_version="abc123")
    assert _served_detector().detect(SOURCES) == local.detect(SOURCES)
    assert _served_detector().detect(PROMPT) == local.detect(PROMPT)


def test_from_server_reports_the_model_name_and_the_callers_version(mock_transport):
    detector = VelaDetector.from_server(BASE_URL, model=MODEL, model_version="2026-11")
    usage = detector.detect(SOURCES).usage[0]
    assert (usage.model_id, usage.model_version) == (MODEL, "2026-11")


def test_served_input_truncated_is_reported():
    result = _served_detector(FakeVela(input_tokens=8192)).detect(SOURCES)
    assert [w.code for w in result.warnings] == ["input_truncated"]


def test_a_served_call_reads_no_environment_variable(monkeypatch, mock_transport):
    environ = _RecordingEnviron(os.environ)
    monkeypatch.setattr(os, "environ", environ)
    detector = VelaDetector.from_server(BASE_URL, model=MODEL, api_key=API_KEY)
    detector.detect(SOURCES)
    detector.detect(PROMPT)
    assert environ.read == []


async def test_concurrent_served_calls_run_in_parallel_and_match_sequential_results():
    model = FakeVela(on_call=lambda: time.sleep(0.02))
    detector = _served_detector(model)
    requests = [
        PromptInput(prompt=f"Context number {i}.", answer=["Berlin.", "3 million.", "No."][i % 3])
        for i in range(8)
    ]
    expected = [detector.detect(r) for r in requests]
    loop = asyncio.get_running_loop()
    sync_results = [loop.run_in_executor(None, detector.detect, r) for r in requests[::2]]
    async_results = [detector.adetect(r) for r in requests[1::2]]
    results = await asyncio.gather(*sync_results, *async_results)
    assert list(results) == expected[::2] + expected[1::2]
    # The client is thread-safe, so the detector takes no scoring lock around it.
    assert model.overlapped


def test_local_models_still_run_one_call_at_a_time():
    assert VelaDetector(FakeVela())._lock_calls is True
    assert _served_detector()._lock_calls is False


def test_close_closes_the_http_client():
    client = _client(_served(FakeVela()))
    with client:
        pass
    assert client._http.is_closed


def test_detector_close_closes_the_client_from_server(mock_transport):
    detector = VelaDetector.from_server(BASE_URL, model=MODEL)
    detector.detect(SOURCES)
    detector.close()
    assert detector.model._http.is_closed
    detector.close()  # a second close is harmless


def test_detector_in_a_with_block_closes_the_client(mock_transport):
    with VelaDetector.from_server(BASE_URL, model=MODEL) as detector:
        detector.detect(SOURCES)
    assert detector.model._http.is_closed


def test_close_on_a_local_model_does_nothing():
    VelaDetector(FakeVela()).close()


def test_api_key_with_a_given_http_client_raises_configuration_error():
    http = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(_served(FakeVela())))
    with pytest.raises(ConfigurationError) as exc:
        SystemOneClient(BASE_URL, model=MODEL, api_key=API_KEY, http_client=http)
    assert (exc.value.code, exc.value.param) == ("invalid_config", "api_key")
    assert API_KEY not in exc.value.message
