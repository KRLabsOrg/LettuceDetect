"""Generate a coherent coding-assistant solution to a developer request.

The model responds the way a coding agent would: a short explanation followed by
the changed code, one fenced block per file. The gold fix is passed only as a
correctness guide — the model presents it naturally rather than copying it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lettucedetect.generation._completion import complete, complete_async

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

_GUIDE_RULES = (
    "You are given the intended fix as a guide. Present it as your own natural "
    "solution: do NOT copy it verbatim, do NOT mention that you were given a fix or "
    "a reference, and do NOT add disclaimers. Only include code you actually change."
)

_SYSTEM_PROMPTS = {
    "files": (
        "You are an expert software engineer answering a developer's request about a "
        "codebase. Respond exactly as a coding assistant would: a brief explanation of "
        "what you change and why, then the code — one fenced ```python block per file "
        "you modify, each preceded by the file path. Be concise and correct.\n\n" + _GUIDE_RULES
    ),
    "edit": (
        "You are an expert software engineer answering a developer's request about a "
        "codebase. Respond exactly as a coding assistant proposing a targeted patch: a "
        "brief explanation of what you change and why, then the concrete edits. For each "
        "edit write `In file <path>, replace:` then a fenced ```python block with the "
        "exact existing code, then `with:` then a fenced ```python block with the new "
        "code. Only include code you actually change. Be concise and correct.\n\n" + _GUIDE_RULES
    ),
}

_CLOSERS = {
    "files": "Write your solution now: a short explanation, then the modified code per file.",
    "edit": "Write your solution now: a short explanation, then each edit as "
    "`In file <path>, replace: ... with: ...`.",
}


def _build_user_msg(
    request: str, context: str, reference: str, context_chars: int, style: str
) -> str:
    return (
        f"Repository context (existing code):\n{context[:context_chars]}\n\n"
        f"Developer request:\n{request}\n\n"
        f"The intended fix (your private guide — present it naturally):\n{reference}\n\n"
        + _CLOSERS[style]
    )


def _accept(content: str | None) -> str | None:
    """Accept a solution that has prose and at least one code block."""
    if not content:
        return None
    text = content.strip()
    if "```" not in text or len(text) < 80:
        return None
    return text


def generate_agent_solution(
    client: OpenAI,
    model: str,
    *,
    request: str,
    context: str,
    reference: str,
    style: str = "files",
    temperature: float = 0.7,
    completion_kwargs: dict | None = None,
    context_chars: int = 28000,
    max_retries: int = 3,
) -> str | None:
    """Generate a coherent assistant solution (explanation + code) for ``request``.

    ``style`` is ``"files"`` (full code block per modified file) or ``"edit"``
    (targeted ``In file X, replace: ... with: ...`` edits).
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPTS[style]},
        {
            "role": "user",
            "content": _build_user_msg(request, context, reference, context_chars, style),
        },
    ]
    return complete(
        client,
        model,
        messages,
        transform=_accept,
        temperature=temperature,
        completion_kwargs=completion_kwargs or {"max_tokens": 1500},
        max_retries=max_retries,
    )


async def generate_agent_solution_async(
    aclient: AsyncOpenAI,
    model: str,
    *,
    request: str,
    context: str,
    reference: str,
    style: str = "files",
    temperature: float = 0.7,
    completion_kwargs: dict | None = None,
    context_chars: int = 28000,
    max_retries: int = 3,
) -> str | None:
    """Async twin of :func:`generate_agent_solution`."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPTS[style]},
        {
            "role": "user",
            "content": _build_user_msg(request, context, reference, context_chars, style),
        },
    ]
    return await complete_async(
        aclient,
        model,
        messages,
        transform=_accept,
        temperature=temperature,
        completion_kwargs=completion_kwargs or {"max_tokens": 1500},
        max_retries=max_retries,
    )
