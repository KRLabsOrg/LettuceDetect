"""Shared chat-completion helper with retries (sync + async).

All generation primitives (answers, injection, questions) make the same call:
send a system+user message, retry on transient errors, and post-process the
result. This centralizes that loop. The caller supplies a ``transform`` that
turns the raw response content into the desired value, or returns ``None`` to
reject it and retry (e.g. malformed JSON, too-short text).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

T = TypeVar("T")


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
) -> T | None:
    """Run a chat completion and post-process it, retrying on error or rejection.

    Retries with backoff on exceptions; retries immediately when ``transform``
    rejects the content (returns ``None``). Returns the transformed value, or
    ``None`` if all attempts are exhausted.
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
            return None
        out = transform(response.choices[0].message.content)
        if out is not None:
            return out
    return None


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
) -> T | None:
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
            return None
        out = transform(response.choices[0].message.content)
        if out is not None:
            return out
    return None
