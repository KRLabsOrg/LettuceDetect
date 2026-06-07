"""Generate a typed, self-contained question answerable from a document.

For sources that ship only documents (markdown wiki pages, READMEs), this turns
a doc/chunk into a realistic information-seeking question of a chosen type. The
question becomes the user request the grounded answer must address. Sources that
already supply a question (ACL, tool output) skip this.

The typed taxonomy (adapted from the acl-verbatim QA generator) drives variety:
sampling across types yields verification, comparison, procedural, causal, etc.
questions rather than only "what is X". Multi-part types are flagged because they
are the natural candidates for omission generation.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from lettucedetect.generation._completion import complete, complete_async

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

# Question type -> one-line definition. Domain-neutral; the type guides the form
# of the question, not its content.
QUESTION_TYPES: dict[str, str] = {
    "Verification": "seeks a yes/no answer confirming a specific detail",
    "Disjunctive": "presents multiple options and asks which one applies",
    "Concept Completion": "a who/what/when/where question identifying a specific element",
    "Example": "asks for an instance that illustrates a concept",
    "Feature Specification": "asks for the properties or characteristics of something",
    "Quantification": "seeks a numerical or measurable value",
    "Definition": "asks for the meaning of a term or concept",
    "Comparison": "asks for similarities and/or differences between two or more things",
    "Interpretation": "asks to infer what an observed pattern or result indicates",
    "Causal Antecedent": "asks for the reasons or causes behind something",
    "Causal Consequence": "asks for the outcomes or effects that follow from something",
    "Goal Orientation": "asks about the objective or intention behind something",
    "Instrumental/Procedural": "asks how to achieve a goal; the steps or procedure",
    "Enablement": "asks what resources or conditions make an action possible",
    "Expectation": "asks about an anticipated outcome, or why one did not occur",
    "Judgmental": "asks for an opinion or evaluation",
    "Assertion": "a statement expressing that something is not understood",
    "Request/Directive": "asks to perform a task such as summarizing or comparing",
}

# Types that naturally ask for several things — the candidates for omission
# (answer covers one part, omit another, mark the unanswered question span).
MULTI_PART_TYPES: tuple[str, ...] = (
    "Comparison",
    "Disjunctive",
    "Feature Specification",
    "Request/Directive",
)

_SYSTEM_PROMPT = (
    "You generate a single, natural information-seeking question that can be "
    "answered from a given document."
)


def sample_question_type(rng: random.Random, allowed: list[str] | None = None) -> str:
    """Pick a question type, optionally restricted to a subset (e.g. for READMEs)."""
    pool = allowed or list(QUESTION_TYPES)
    return rng.choice(pool)


def _user_msg(doc: str, q_type: str, doc_chars: int) -> str:
    return (
        f"Document:\n{doc[:doc_chars]}\n\n"
        f"Generate one **{q_type}** question ({QUESTION_TYPES.get(q_type, '')}) "
        "that the document answers.\n\n"
        "Rules:\n"
        "1. Return ONLY the question, nothing else.\n"
        "2. Use neutral, self-contained phrasing — refer to things directly, not "
        'as "this document", "the study", or "the dataset mentioned".\n'
        "3. Keep it short and natural, like a query typed into a search engine.\n"
        "4. It must be answerable from the document above."
    )


def _clean(content: str | None) -> str | None:
    """Extract a single clean question line, or None to reject and retry."""
    if not content:
        return None
    line = content.strip().splitlines()[0].strip().strip('"').strip()
    # Drop a leading "Question:" style prefix if the model added one.
    if ":" in line[:12] and line.split(":", 1)[0].lower() in {"question", "q"}:
        line = line.split(":", 1)[1].strip()
    return line if len(line) >= 8 else None


def generate_question(
    client: OpenAI,
    model: str,
    *,
    doc: str,
    q_type: str,
    temperature: float = 0.7,
    completion_kwargs: dict | None = None,
    doc_chars: int = 6000,
    max_retries: int = 3,
) -> str | None:
    """Generate one ``q_type`` question answerable from ``doc`` (or None)."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_msg(doc, q_type, doc_chars)},
    ]
    return complete(
        client,
        model,
        messages,
        transform=_clean,
        temperature=temperature,
        completion_kwargs=completion_kwargs or {"max_tokens": 80},
        max_retries=max_retries,
    )


async def generate_question_async(
    aclient: AsyncOpenAI,
    model: str,
    *,
    doc: str,
    q_type: str,
    temperature: float = 0.7,
    completion_kwargs: dict | None = None,
    doc_chars: int = 6000,
    max_retries: int = 3,
) -> str | None:
    """Async twin of :func:`generate_question`."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_msg(doc, q_type, doc_chars)},
    ]
    return await complete_async(
        aclient,
        model,
        messages,
        transform=_clean,
        temperature=temperature,
        completion_kwargs=completion_kwargs or {"max_tokens": 80},
        max_retries=max_retries,
    )
