"""Shared chat-completion helper with retries (sync + async).

All generation primitives (answers, injection, questions) make the same call:
send a system+user message, retry on transient errors, and post-process the
result. This centralizes that loop. The caller supplies a ``transform`` that
turns the raw response content into the desired value, or returns ``None`` to
reject it and retry (e.g. malformed JSON, too-short text).
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

T = TypeVar("T")

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_message(message: object) -> tuple[str, str | None]:
    """Return ``(content, reasoning)``, separating any thinking trace from the answer.

    Provider-agnostic. Handles the shapes seen across backends:
    - a structured ``reasoning`` field (vLLM/Qwen3, OpenRouter),
    - ``reasoning_content`` (DeepSeek-style, vLLM ``--reasoning-parser``),
    - inline ``<think>...</think>`` in the content (Qwen with no parser) — stripped
      out of the content so the transform never sees it.
    ``reasoning`` is ``None`` when the provider exposes none (e.g. OpenAI o-series
    hides it); content is returned unchanged in that case.
    """
    content = message.content or ""
    extra = getattr(message, "model_extra", None) or {}
    reasoning = extra.get("reasoning") or getattr(message, "reasoning_content", None)
    if reasoning is None and "<think>" in content:
        m = _THINK_RE.search(content)
        if m:
            reasoning = m.group(1).strip()
            content = (content[: m.start()] + content[m.end() :]).strip()
        else:  # opening tag but no close — truncated mid-thought
            reasoning = content.split("<think>", 1)[1]
            content = ""
    return content, reasoning


def complete(
    client: OpenAI,
    model: str,
    messages: list[dict],
    *,
    transform: Callable[[str | None], T | None],
    temperature: float = 0.7,
    completion_kwargs: dict | None = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    capture_reasoning: bool = False,
) -> T | tuple[T | None, str | None] | None:
    """Run a chat completion and post-process it, retrying on error or rejection.

    Retries with backoff on exceptions; retries immediately when ``transform``
    rejects the content (returns ``None``). Returns the transformed value, or
    ``None`` if all attempts are exhausted. When ``capture_reasoning`` is set,
    returns ``(value, reasoning)`` instead (reasoning is the model's thinking
    trace, or ``None``).
    """
    completion_kwargs = completion_kwargs or {}
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, **completion_kwargs
            )
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            return (None, None) if capture_reasoning else None
        content, reasoning = _split_message(response.choices[0].message)
        out = transform(content)
        if out is not None:
            return (out, reasoning) if capture_reasoning else out
    return (None, None) if capture_reasoning else None


async def complete_async(
    aclient: AsyncOpenAI,
    model: str,
    messages: list[dict],
    *,
    transform: Callable[[str | None], T | None],
    temperature: float = 0.7,
    completion_kwargs: dict | None = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    capture_reasoning: bool = False,
) -> T | tuple[T | None, str | None] | None:
    """Async twin of :func:`complete` for batched throughput."""
    completion_kwargs = completion_kwargs or {}
    for attempt in range(max_retries):
        try:
            response = await aclient.chat.completions.create(
                model=model, messages=messages, temperature=temperature, **completion_kwargs
            )
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
                continue
            return (None, None) if capture_reasoning else None
        content, reasoning = _split_message(response.choices[0].message)
        out = transform(content)
        if out is not None:
            return (out, reasoning) if capture_reasoning else out
    return (None, None) if capture_reasoning else None
