"""LLM client abstraction for the LLM-based hallucination detector.

The detector only needs one capability from a backend: given a system prompt,
a user prompt, and a JSON schema, return a JSON string that conforms to the
schema. :class:`LLMClient` defines that contract; :class:`OpenAIClient` and
:class:`BedrockClient` implement it for the OpenAI API and AWS Bedrock.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

HALLUCINATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "hallucination_list": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of exact text spans from the answer that are hallucinated",
        }
    },
    "required": ["hallucination_list"],
    "additionalProperties": False,
}

HALLUCINATION_REASONING_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "hallucination_list": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Exact text span from the answer that is hallucinated",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence between 0 and 1 that the span is hallucinated",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of why the span is hallucinated",
                    },
                },
                "required": ["text", "confidence", "reasoning"],
                "additionalProperties": False,
            },
            "description": "List of hallucinated spans with confidence and reasoning",
        }
    },
    "required": ["hallucination_list"],
    "additionalProperties": False,
}


class LLMClient(ABC):
    """Backend that turns a prompt into a schema-conforming JSON string."""

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float,
        schema: dict,
    ) -> str:
        """Run a single completion and return the raw JSON string response.

        :param system: System prompt.
        :param user: User prompt (context + answer already formatted).
        :param model: Provider-specific model identifier.
        :param temperature: Sampling temperature.
        :param schema: JSON schema the response object must conform to.
        :returns: JSON string conforming to ``schema``.
        """
        ...


class OpenAIClient(LLMClient):
    """LLM client backed by the OpenAI chat completions API."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """Initialize the OpenAI client.

        :param api_key: API key; falls back to ``OPENAI_API_KEY``.
        :param base_url: API base URL; falls back to ``OPENAI_API_BASE`` then the OpenAI default.
        """
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "EMPTY",
            base_url=base_url or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1",
        )

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float,
        schema: dict,
    ) -> str:
        """Run a completion via ``chat.completions`` with a strict JSON schema."""
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "hallucination_detection",
                    "schema": schema,
                    "strict": True,
                },
            },
            temperature=temperature,
        )
        return resp.choices[0].message.content


class BedrockClient(LLMClient):
    """LLM client backed by the AWS Bedrock ``converse`` API.

    Structured output is obtained by exposing the schema as a single tool and
    forcing the model to call it; the tool input is the schema-conforming object.
    """

    _TOOL_NAME = "record_hallucinations"
    _MAX_TOKENS = 4096

    def __init__(self, region_name: str | None = None, **client_kwargs) -> None:
        """Initialize the Bedrock runtime client.

        :param region_name: AWS region; falls back to ``AWS_REGION`` then boto3 defaults.
        :param client_kwargs: Extra keyword arguments forwarded to ``boto3.client``.
        """
        import boto3

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region_name or os.getenv("AWS_REGION"),
            **client_kwargs,
        )

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float,
        schema: dict,
    ) -> str:
        """Run a completion via ``converse`` with a forced tool call for structured output."""
        resp = self._client.converse(
            modelId=model,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"temperature": temperature, "maxTokens": self._MAX_TOKENS},
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": self._TOOL_NAME,
                            "description": "Record the hallucinated spans found in the answer.",
                            "inputSchema": {"json": schema},
                        }
                    }
                ],
                "toolChoice": {"tool": {"name": self._TOOL_NAME}},
            },
        )
        if resp.get("stopReason") == "max_tokens":
            logger.warning("Bedrock response truncated at maxTokens=%s", self._MAX_TOKENS)
        for block in resp["output"]["message"]["content"]:
            if "toolUse" in block:
                return json.dumps(block["toolUse"]["input"])
        logger.warning("Bedrock response contained no tool-use block; returning empty result")
        return ""


def make_llm_client(provider: str, **kwargs) -> LLMClient:
    """Create an :class:`LLMClient` for the requested provider.

    :param provider: ``"openai"`` or ``"bedrock"``.
    :param kwargs: Passed to the concrete client constructor.
    :returns: A concrete :class:`LLMClient` instance.
    :raises ValueError: If the provider is not supported.
    """
    if provider == "openai":
        return OpenAIClient(**kwargs)
    if provider == "bedrock":
        return BedrockClient(**kwargs)
    raise ValueError(f"Unknown LLM provider: {provider}. Use one of: openai, bedrock")
