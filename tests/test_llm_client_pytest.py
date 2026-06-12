"""Pytest tests for the LLM client abstraction and LLMDetector backend wiring.

These tests never touch the network or require API keys. The OpenAI and Bedrock
backends are exercised by injecting fakes (or by monkeypatching the underlying
SDK client), and ``LLMDetector`` is exercised by injecting a fake
:class:`LLMClient`. ``boto3`` is an optional dependency that may be absent, so
Bedrock construction paths are guarded / faked rather than importing it.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from lettucedetect.detectors.llm import LLMDetector
from lettucedetect.detectors.llm_client import (
    HALLUCINATION_JSON_SCHEMA,
    HALLUCINATION_REASONING_JSON_SCHEMA,
    BedrockClient,
    LLMClient,
    OpenAIClient,
    make_llm_client,
)


class FakeClient(LLMClient):
    """An :class:`LLMClient` that returns a canned response and records calls."""

    def __init__(self, response: str = '{"hallucination_list": []}') -> None:
        """Store the canned response and prepare a call log."""
        self.response = response
        self.calls: list[dict] = []

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float,
        schema: dict,
    ) -> str:
        """Record the call arguments and return the canned response."""
        self.calls.append(
            {
                "system": system,
                "user": user,
                "model": model,
                "temperature": temperature,
                "schema": schema,
            }
        )
        return self.response


@pytest.fixture
def cache_file(tmp_path):
    """Return a temporary cache file path so tests never pollute the repo cache dir."""
    return str(tmp_path / "cache.json")


def make_detector(client: LLMClient, cache_file: str, **kwargs) -> LLMDetector:
    """Build an LLMDetector with an injected client and a temp cache."""
    return LLMDetector(client=client, cache_file=cache_file, **kwargs)


class TestMakeLLMClient:
    """Tests for the ``make_llm_client`` factory."""

    def test_returns_openai_client_for_openai(self):
        """Provider 'openai' yields an OpenAIClient."""
        client = make_llm_client("openai")
        assert isinstance(client, OpenAIClient)

    def test_returns_bedrock_client_for_bedrock(self, monkeypatch):
        """``make_llm_client('bedrock')`` returns a BedrockClient.

        boto3 may be absent, so inject a fake ``boto3`` module exposing a
        ``client`` factory rather than relying on the real SDK.
        """
        fake_boto3 = types.ModuleType("boto3")
        recorded = {}

        def fake_client(service, **kwargs):
            recorded["service"] = service
            recorded["kwargs"] = kwargs
            return object()

        fake_boto3.client = fake_client
        monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

        client = make_llm_client("bedrock", region_name="us-east-1")
        assert isinstance(client, BedrockClient)
        assert recorded["service"] == "bedrock-runtime"
        assert recorded["kwargs"]["region_name"] == "us-east-1"

    def test_unknown_provider_raises_value_error(self):
        """An unknown provider raises ValueError."""
        with pytest.raises(ValueError):
            make_llm_client("anthropic")


class TestLLMDetectorWithInjectedClient:
    """Tests for LLMDetector routing through an injected LLMClient."""

    def test_predict_spans_returns_character_offsets(self, cache_file):
        """predict() maps the hallucination list to character-offset spans."""
        answer = "The capital of France is Paris and it is sunny."
        client = FakeClient('{"hallucination_list": ["Paris"]}')
        detector = make_detector(client, cache_file)

        spans = detector.predict(
            context=["France is a country in Europe."],
            answer=answer,
            question="What is the capital of France?",
            output_format="spans",
        )

        assert spans == [
            {
                "start": answer.index("Paris"),
                "end": answer.index("Paris") + len("Paris"),
                "text": "Paris",
            }
        ]

    def test_client_receives_expected_model_temperature_and_schema(self, cache_file):
        """The client receives the configured model, temperature, and shared schema."""
        client = FakeClient('{"hallucination_list": []}')
        detector = make_detector(client, cache_file, model="my-model", temperature=0.7)

        detector.predict(["ctx"], "an answer", "a question", output_format="spans")

        assert len(client.calls) == 1
        call = client.calls[0]
        assert call["model"] == "my-model"
        assert call["temperature"] == 0.7
        assert call["schema"] is HALLUCINATION_JSON_SCHEMA
        assert "hallucination_list" in call["schema"]["properties"]
        assert call["schema"]["required"] == ["hallucination_list"]

    def test_predict_prompt_routes_through_client(self, cache_file):
        """predict_prompt() routes through the client and returns spans."""
        answer = "Lisbon is the capital of Spain."
        client = FakeClient('{"hallucination_list": ["Spain"]}')
        detector = make_detector(client, cache_file)

        spans = detector.predict_prompt("some prompt", answer, output_format="spans")

        assert len(client.calls) == 1
        assert spans == [
            {
                "start": answer.index("Spain"),
                "end": answer.index("Spain") + len("Spain"),
                "text": "Spain",
            }
        ]

    def test_predict_prompt_batch_returns_one_result_per_input(self, cache_file):
        """predict_prompt_batch() returns one span list per input pair."""
        client = FakeClient('{"hallucination_list": ["X"]}')
        detector = make_detector(client, cache_file)

        prompts = ["prompt one", "prompt two", "prompt three"]
        answers = ["answer X here", "no match", "trailing X"]

        results = detector.predict_prompt_batch(prompts, answers, output_format="spans")

        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)
        # First answer contains "X", second does not.
        assert results[0] == [{"start": 7, "end": 8, "text": "X"}]
        assert results[1] == []
        assert client.calls and len(client.calls) == 3

    def test_caching_invokes_client_only_once(self, cache_file):
        """Repeated identical predictions hit the cache and call the client once."""
        client = FakeClient('{"hallucination_list": ["Paris"]}')
        detector = make_detector(client, cache_file)

        first = detector.predict(["ctx"], "Paris is here", "q?", output_format="spans")
        second = detector.predict(["ctx"], "Paris is here", "q?", output_format="spans")

        assert first == second
        assert len(client.calls) == 1

    def test_invalid_output_format_raises_value_error(self, cache_file):
        """predict() rejects an unsupported output format."""
        client = FakeClient()
        detector = make_detector(client, cache_file)

        with pytest.raises(ValueError):
            detector.predict(["ctx"], "answer", "q?", output_format="bogus")

    def test_invalid_output_format_predict_prompt_raises(self, cache_file):
        """predict_prompt() rejects an unsupported output format."""
        client = FakeClient()
        detector = make_detector(client, cache_file)

        with pytest.raises(ValueError):
            detector.predict_prompt("prompt", "answer", output_format="bogus")

    def test_invalid_output_format_predict_prompt_batch_raises(self, cache_file):
        """predict_prompt_batch() rejects an unsupported output format."""
        client = FakeClient()
        detector = make_detector(client, cache_file)

        with pytest.raises(ValueError):
            detector.predict_prompt_batch(["p"], ["a"], output_format="bogus")

    def test_empty_client_output_yields_empty_spans(self, cache_file):
        """An empty client response yields no spans instead of raising."""
        client = FakeClient("")
        detector = make_detector(client, cache_file)

        spans = detector.predict(["ctx"], "answer", "q?", output_format="spans")
        assert spans == []

    def test_non_json_client_output_yields_empty_spans(self, cache_file):
        """A non-JSON client response yields no spans instead of raising."""
        client = FakeClient("not valid json at all")
        detector = make_detector(client, cache_file)

        spans = detector.predict(["ctx"], "answer", "q?", output_format="spans")
        assert spans == []

    def test_include_reasoning_returns_confidence_and_reasoning_in_spans(self, cache_file):
        """With include_reasoning=True, spans carry the LLM's confidence and reasoning."""
        answer = "The capital of France is Paris and it is sunny."
        response = json.dumps(
            {
                "hallucination_list": [
                    {
                        "text": "it is sunny",
                        "confidence": 0.85,
                        "reasoning": "The source says nothing about the weather.",
                    }
                ]
            }
        )
        client = FakeClient(response)
        detector = make_detector(client, cache_file, include_reasoning=True)

        spans = detector.predict(
            context=["France is a country in Europe."],
            answer=answer,
            question="What is the capital of France?",
            output_format="spans",
        )

        assert spans == [
            {
                "start": answer.index("it is sunny"),
                "end": answer.index("it is sunny") + len("it is sunny"),
                "text": "it is sunny",
                "confidence": 0.85,
                "reasoning": "The source says nothing about the weather.",
            }
        ]

    def test_include_reasoning_sends_reasoning_schema_and_prompt(self, cache_file):
        """include_reasoning=True switches the schema and the prompt's format instructions."""
        client = FakeClient('{"hallucination_list": []}')
        detector = make_detector(client, cache_file, include_reasoning=True)

        detector.predict(["ctx"], "an answer", "a question", output_format="spans")

        call = client.calls[0]
        assert call["schema"] is HALLUCINATION_REASONING_JSON_SCHEMA
        assert '"confidence"' in call["user"]
        assert '"reasoning"' in call["user"]

    def test_include_reasoning_off_keeps_plain_schema_and_spans(self, cache_file):
        """The default keeps the plain string-list schema and spans without extra keys."""
        answer = "Paris is sunny."
        client = FakeClient('{"hallucination_list": ["sunny"]}')
        detector = make_detector(client, cache_file)

        spans = detector.predict(["ctx"], answer, "q?", output_format="spans")

        assert client.calls[0]["schema"] is HALLUCINATION_JSON_SCHEMA
        assert spans == [{"start": 9, "end": 14, "text": "sunny"}]

    def test_missing_key_in_client_output_yields_empty_spans(self, cache_file):
        """A response missing the expected key yields no spans instead of raising."""
        client = FakeClient('{"something_else": []}')
        detector = make_detector(client, cache_file)

        spans = detector.predict(["ctx"], "answer", "q?", output_format="spans")
        assert spans == []


class _StubMessage:
    """Stub for an OpenAI chat message."""

    def __init__(self, content: str) -> None:
        """Store the message content."""
        self.content = content


class _StubChoice:
    """Stub for an OpenAI choice wrapping a message."""

    def __init__(self, content: str) -> None:
        """Wrap the content in a stub message."""
        self.message = _StubMessage(content)


class _StubResponse:
    """Stub for an OpenAI chat completion response."""

    def __init__(self, content: str) -> None:
        """Wrap the content in a single stub choice."""
        self.choices = [_StubChoice(content)]


class TestOpenAIClientComplete:
    """Tests for OpenAIClient.complete without real network access."""

    def test_complete_returns_message_content_with_strict_schema(self):
        """complete() returns the message content and sends the strict JSON schema."""
        client = OpenAIClient(api_key="test-key")

        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return _StubResponse('{"hallucination_list": ["Paris"]}')

        client._client.chat.completions.create = fake_create

        result = client.complete(
            system="sys prompt",
            user="user prompt",
            model="gpt-test",
            temperature=0.3,
            schema=HALLUCINATION_JSON_SCHEMA,
        )

        assert result == '{"hallucination_list": ["Paris"]}'
        assert captured["model"] == "gpt-test"
        assert captured["temperature"] == 0.3
        assert captured["messages"] == [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "user prompt"},
        ]
        response_format = captured["response_format"]
        assert response_format["type"] == "json_schema"
        json_schema = response_format["json_schema"]
        assert json_schema["name"] == "hallucination_detection"
        assert json_schema["strict"] is True
        assert json_schema["schema"] is HALLUCINATION_JSON_SCHEMA


def _make_bedrock_client_with_fake(fake_runtime) -> BedrockClient:
    """Build a BedrockClient bypassing __init__ (which imports boto3)."""
    client = BedrockClient.__new__(BedrockClient)
    client._client = fake_runtime
    return client


class _FakeBedrockRuntime:
    """Fake bedrock-runtime client recording the converse() call."""

    def __init__(self, response: dict) -> None:
        """Store the canned converse() response."""
        self.response = response
        self.converse_kwargs: dict | None = None

    def converse(self, **kwargs):
        """Record the call kwargs and return the canned response."""
        self.converse_kwargs = kwargs
        return self.response


class TestBedrockClientComplete:
    """Tests for BedrockClient.complete without real network access."""

    def test_complete_returns_tool_input_as_json(self):
        """complete() serialises the forced tool input to a JSON string."""
        response = {
            "output": {
                "message": {"content": [{"toolUse": {"input": {"hallucination_list": ["X"]}}}]}
            }
        }
        fake = _FakeBedrockRuntime(response)
        client = _make_bedrock_client_with_fake(fake)

        result = client.complete(
            system="sys",
            user="user",
            model="anthropic.claude-3",
            temperature=0.0,
            schema=HALLUCINATION_JSON_SCHEMA,
        )

        assert json.loads(result) == {"hallucination_list": ["X"]}
        assert fake.converse_kwargs["modelId"] == "anthropic.claude-3"
        tool_config = fake.converse_kwargs["toolConfig"]
        assert tool_config["toolChoice"] == {"tool": {"name": "record_hallucinations"}}
        assert (
            tool_config["tools"][0]["toolSpec"]["inputSchema"]["json"] is HALLUCINATION_JSON_SCHEMA
        )

    def test_complete_with_text_block_before_tool_use(self):
        """complete() finds the toolUse block even after a leading text block."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "Here you go"},
                        {"toolUse": {"input": {"hallucination_list": ["Y"]}}},
                    ]
                }
            }
        }
        client = _make_bedrock_client_with_fake(_FakeBedrockRuntime(response))

        result = client.complete("s", "u", "m", 0.0, HALLUCINATION_JSON_SCHEMA)
        assert json.loads(result) == {"hallucination_list": ["Y"]}

    def test_complete_without_tool_use_returns_empty_string(self):
        """complete() returns an empty string when no toolUse block is present."""
        response = {"output": {"message": {"content": [{"text": "no tool call here"}]}}}
        client = _make_bedrock_client_with_fake(_FakeBedrockRuntime(response))

        result = client.complete("s", "u", "m", 0.0, HALLUCINATION_JSON_SCHEMA)
        assert result == ""
