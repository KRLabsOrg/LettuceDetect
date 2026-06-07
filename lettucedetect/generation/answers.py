"""Generate a correct, grounded answer from a question and supporting evidence.

A shared building block for every data source that does not already ship a gold
answer (squeez, markdown, ACL, ...). The produced answer is correct *by
construction* — grounded strictly in the supplied evidence — so the injector can
then corrupt it into a hallucinated variant with known-good spans.

This is deliberately separate from
:class:`lettucedetect.models.generation.HallucinationGenerator`, which produces
*hallucinated* content. Here we produce the clean answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lettucedetect.generation._completion import complete, complete_async

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

# Default per-modality answer style. Adapters may pass their own system prompt.
_DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant answering a user's question using ONLY the provided \
evidence. Write a correct, natural answer grounded strictly in that evidence.

Your answer MUST:
- Be accurate and fully supported by the evidence — invent nothing.
- Reference concrete details from the evidence (names, values, locations) where \
relevant.
- Be concise: a few sentences, or a short code block when the question asks for code.

Your answer must NOT:
- Add claims, identifiers, or values not present in the evidence.
- Include filler like "Here's the answer" or "I'll help you with that".
"""


def generate_grounded_answer(
    client: OpenAI,
    model: str,
    *,
    question: str,
    evidence: str,
    system_prompt: str | None = None,
    extra_context: str = "",
    temperature: float = 0.7,
    completion_kwargs: dict | None = None,
    evidence_chars: int = 6000,
    min_chars: int = 40,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str | None:
    """Generate a correct answer to ``question`` grounded in ``evidence``.

    :param system_prompt: override the default grounding prompt (e.g. to tune
        answer style per source/modality).
    :param extra_context: optional extra framing (e.g. a broader task description)
        appended to the user message.
    :param min_chars: reject answers shorter than this (treated as failures).
    :return: the answer string, or None on failure.
    """
    messages = _build_messages(
        system_prompt or _DEFAULT_SYSTEM_PROMPT, question, evidence, extra_context, evidence_chars
    )
    return complete(
        client,
        model,
        messages,
        transform=lambda c: _accept(c, min_chars),
        temperature=temperature,
        completion_kwargs=completion_kwargs or {"max_tokens": 400},
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


def _accept(content: str | None, min_chars: int) -> str | None:
    """Keep a stripped answer if it meets the minimum length, else reject."""
    text = (content or "").strip()
    return text if len(text) >= min_chars else None


def _build_messages(
    system_prompt: str, question: str, evidence: str, extra_context: str, evidence_chars: int
) -> list[dict]:
    extra = f"\nAdditional context: {extra_context.strip()}\n" if extra_context.strip() else ""
    user_msg = (
        f"Question: {question.strip()}\n{extra}\n"
        f"Evidence:\n{evidence[:evidence_chars].strip()}\n\n"
        "Write the correct, grounded answer."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]


async def generate_grounded_answer_async(
    aclient: AsyncOpenAI,
    model: str,
    *,
    question: str,
    evidence: str,
    system_prompt: str | None = None,
    extra_context: str = "",
    temperature: float = 0.7,
    completion_kwargs: dict | None = None,
    evidence_chars: int = 6000,
    min_chars: int = 40,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str | None:
    """Async twin of :func:`generate_grounded_answer` for batched throughput."""
    messages = _build_messages(
        system_prompt or _DEFAULT_SYSTEM_PROMPT, question, evidence, extra_context, evidence_chars
    )
    return await complete_async(
        aclient,
        model,
        messages,
        transform=lambda c: _accept(c, min_chars),
        temperature=temperature,
        completion_kwargs=completion_kwargs or {"max_tokens": 400},
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
