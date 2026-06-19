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


def build_hallucination_schema(
    include_reasoning: bool = False,
    categories: list[str] | None = None,
    subcategories: list[str] | None = None,
) -> dict:
    """Build the JSON schema for the hallucination-detection response.

    With no options the response is a plain list of hallucinated substrings;
    enabling either option switches to a list of objects carrying the extra fields.

    Field order matters: the model generates the object keys left-to-right, so
    ``reasoning`` precedes ``confidence`` and the final ``is_hallucination``
    verdict, letting each judgement be conditioned on the reasoning that precedes
    it. With ``include_reasoning`` the response also opens with a top-level
    ``analysis`` scratchpad, generated before any span is listed.

    :param include_reasoning: Add a top-level ``analysis`` field plus per-span
        ``reasoning``, ``confidence``, and ``is_hallucination`` fields.
    :param categories: Allowed values for a per-span ``category`` field; omitted when None.
    :returns: JSON schema the response object must conform to.
    """
    if not include_reasoning and not categories and not subcategories:
        items: dict = {
            "type": "string",
            "description": "Exact text span from the answer that is hallucinated",
        }
    else:
        properties: dict = {
            "text": {
                "type": "string",
                "description": "Exact text span from the answer that is hallucinated",
            }
        }
        if include_reasoning:
            properties["reasoning"] = {
                "type": "string",
                "description": "Comparison of the span against the source, written before the verdict",
            }
        if categories:
            properties["category"] = {
                "type": "string",
                "enum": list(categories),
                "description": "Hallucination category of the span",
            }
        if subcategories:
            properties["subcategory"] = {
                "type": "string",
                "enum": list(subcategories),
                "description": "Hallucination subcategory of the span",
            }
        if include_reasoning:
            properties["confidence"] = {
                "type": "number",
                "description": "Confidence between 0 and 1 that the span is hallucinated",
            }
            properties["is_hallucination"] = {
                "type": "boolean",
                "description": (
                    "Final verdict: true only if the span is genuinely unsupported by or "
                    "contradicts the source; false if the reasoning concludes it is supported"
                ),
            }
        items = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
    top_properties: dict = {}
    if include_reasoning:
        top_properties["analysis"] = {
            "type": "string",
            "description": "Claim-by-claim comparison of the answer against the source, written first",
        }
    top_properties["hallucination_list"] = {
        "type": "array",
        "items": items,
        "description": "List of hallucinated spans from the answer",
    }
    return {
        "type": "object",
        "properties": top_properties,
        "required": list(top_properties),
        "additionalProperties": False,
    }


def build_generative_schema(explain: bool = False) -> dict:
    """Build the JSON schema for the fine-tuned span detectors' ``hallucinated_spans`` output.

    Distinct from :func:`build_hallucination_schema` (the LLM-judge contract): this
    matches what ``lettucedect-v2-*`` generative models were trained to emit ---
    a typed span object with ``category`` and ``subcategory`` drawn from the unified
    taxonomy, and an optional ``explanation``. Enums come from the frozen
    :mod:`lettucedetect.prompts.generative` label set.

    :param explain: Require a per-span ``explanation`` field.
    :returns: JSON schema for ``{"hallucinated_spans": [...]}``.
    """
    from lettucedetect.prompts.generative import (
        CATEGORY_DESCRIPTIONS,
        SUBCATEGORY_DESCRIPTIONS,
    )

    properties: dict = {
        "text": {"type": "string", "description": "Exact hallucinated substring of the answer"},
        "category": {"type": "string", "enum": list(CATEGORY_DESCRIPTIONS)},
        "subcategory": {"type": "string", "enum": list(SUBCATEGORY_DESCRIPTIONS)},
    }
    if explain:
        properties["explanation"] = {
            "type": "string",
            "description": "Short explanation of why the span is unsupported by the context",
        }
    item = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "hallucinated_spans": {
                "type": "array",
                "items": item,
                "description": "List of hallucinated spans from the answer",
            }
        },
        "required": ["hallucinated_spans"],
        "additionalProperties": False,
    }


def build_verification_schema() -> dict:
    """Build the JSON schema for the verification (second-opinion) response.

    Each candidate span flagged by the first pass is re-judged with a binary
    verdict; ``reasoning`` precedes ``is_hallucination`` so the verdict is
    conditioned on it.

    :returns: JSON schema the verification response object must conform to.
    """
    return {
        "type": "object",
        "properties": {
            "verifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The exact candidate span being re-judged",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why the span is or is not supported by the source",
                        },
                        "is_hallucination": {
                            "type": "boolean",
                            "description": "True only if the span genuinely contradicts or is unsupported by the source",
                        },
                    },
                    "required": ["text", "reasoning", "is_hallucination"],
                    "additionalProperties": False,
                },
                "description": "Verdict for each candidate span",
            }
        },
        "required": ["verifications"],
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
    _MAX_TOKENS = 8192

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
