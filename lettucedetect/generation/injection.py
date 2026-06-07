"""Universal, taxonomy-driven hallucination injection.

Corrupts a *known-correct* answer into a hallucinated one, returning exact
character-level spans. The same engine serves every structural format — source
code, tool output, markdown, prose — because the unified taxonomy is
modality-agnostic; only a short per-modality note and the category/subtype
definitions vary.

Two layers:

- The mechanical engine (:func:`inject`): request localized replacement edits
  from the model, apply them deterministically, validate the resulting spans.
- The taxonomy layer (:func:`inject_taxonomy`): build a modality-aware prompt for
  a specific ``(category, subcategory)`` and attach those labels to each span.

Each data source supplies ``(context, clean_answer, modality)`` and a desired
category/subtype distribution; everything else is shared here.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lettucedetect.datasets.taxonomy import (
    CATEGORY_DEFINITIONS,
    SUBCATEGORIES,
    SUBCATEGORY_DEFINITIONS,
    map_label,
)
from lettucedetect.generation._completion import complete, complete_async

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

# ── Default thresholds (callers may override) ──────────────────────────────────

LEAKY_TERMS: tuple[str, ...] = (
    "bug",
    "wrong",
    "incorrect",
    "incorrectly",
    "deprecated",
    "hallucination",
    "helper method",
    "should be replaced",
)
PROMPT_RESIDUE: tuple[str, ...] = (
    "Generate a hallucinated version",
    "Return JSON only",
    "hallucinated_code",
    "target_zone",
    "left_context",
    "right_context",
)
MAX_LABEL_COVERAGE = 0.40
MAX_LABEL_SPAN_CHARS = 500
MIN_LABEL_SPAN_CHARS = 12


@dataclass
class InjectionResult:
    """Outcome of an injection attempt.

    ``ok`` is True only when a valid hallucinated answer with spans was produced.
    On failure, ``reason`` carries a short machine-readable cause.
    """

    ok: bool
    reason: str | None = None
    hallucinated_answer: str | None = None
    labels: list[dict] = field(default_factory=list)
    changes: list[dict] = field(default_factory=list)
    reasoning: str | None = None  # model thinking trace, when the server emits one


# ── Edit location and application ──────────────────────────────────────────────


def _find_all_occurrences(text: str, pattern: str) -> list[dict]:
    """Return all exact matches of pattern in text."""
    if not pattern:
        return []
    offsets = []
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx == -1:
            break
        offsets.append({"start": idx, "end": idx + len(pattern)})
        start = idx + 1
    return offsets


def _locate_original_change(original_answer: str, change: dict) -> dict | None:
    """Locate a replacement span in the original answer by exact unique match."""
    original_span = change.get("original", "")
    hallucinated_span = change.get("hallucinated", "")
    if not original_span or not hallucinated_span:
        return None

    offsets = _find_all_occurrences(original_answer, original_span)
    if len(offsets) != 1:
        return None

    return {
        "start": offsets[0]["start"],
        "end": offsets[0]["end"],
        "original": original_span,
        "hallucinated": hallucinated_span,
    }


def _sort_changes_by_original_position(
    original_answer: str, changes: list[dict]
) -> list[dict] | None:
    """Return changes ordered by their matched position in the original answer."""
    located = []
    for change in changes:
        loc = _locate_original_change(original_answer, change)
        if loc is None:
            return None
        located.append((loc["start"], loc["end"], change))
    located.sort(key=lambda item: (item[0], item[1]))
    return [change for _, _, change in located]


def apply_changes_to_answer(
    original_answer: str, changes: list[dict], hall_type: str
) -> tuple[str, list[dict]] | tuple[None, None]:
    """Apply structured replacement edits and build character-level labels.

    The model returns edits only; this function deterministically constructs the
    hallucinated answer and the corresponding label offsets. Overlapping edits
    are dropped, keeping the earlier one.

    Each label's ``label`` is the edit's own ``hallucination_type`` when present
    (menu-style prompts return a type per edit), otherwise the passed ``hall_type``.
    """
    located = []
    for change in changes:
        if len(change.get("hallucinated", "")) < MIN_LABEL_SPAN_CHARS:
            continue
        located_change = _locate_original_change(original_answer, change)
        if located_change is None:
            continue
        located_change["label"] = change.get("hallucination_type") or hall_type
        located.append(located_change)

    if not located:
        return None, None

    located.sort(key=lambda item: (item["start"], item["end"]))
    previous_end = -1
    filtered = []
    for item in located:
        if item["start"] >= previous_end:
            filtered.append(item)
            previous_end = item["end"]
    located = filtered

    if not located:
        return None, None

    hallucinated_parts: list[str] = []
    labels: list[dict] = []
    cursor = 0
    for item in located:
        start = item["start"]
        end = item["end"]
        hallucinated_span = item["hallucinated"]

        hallucinated_parts.append(original_answer[cursor:start])
        label_start = sum(len(part) for part in hallucinated_parts)
        hallucinated_parts.append(hallucinated_span)
        label_end = label_start + len(hallucinated_span)
        labels.append({"start": label_start, "end": label_end, "label": item["label"]})
        cursor = end

    hallucinated_parts.append(original_answer[cursor:])
    return "".join(hallucinated_parts), labels


# ── Validation helpers ─────────────────────────────────────────────────────────


def _extract_code_regions(answer: str) -> list[tuple[int, int]]:
    """Return ranges corresponding to markdown fenced code blocks.

    If no fenced blocks are present, treat the whole answer as one region.
    """
    regions = []
    idx = 0
    while True:
        start = answer.find("```", idx)
        if start == -1:
            break
        code_start = answer.find("\n", start + 3)
        if code_start == -1:
            break
        code_start += 1
        end = answer.find("```", code_start)
        if end == -1:
            break
        regions.append((code_start, end))
        idx = end + 3
    if not regions:
        return [(0, len(answer))]
    return regions


def _span_is_in_code(answer: str, start: int, end: int) -> bool:
    """Check whether a span lies fully inside a fenced code region."""
    for code_start, code_end in _extract_code_regions(answer):
        if start >= code_start and end <= code_end:
            return True
    return False


def _contains_leakage(text: str, leaky_terms: tuple[str, ...]) -> bool:
    """Detect obvious synthetic giveaway text inside a label span."""
    lowered = text.lower()
    return any(term in lowered for term in leaky_terms)


def _max_allowed_coverage(answer_len: int, base_cap: float) -> float:
    """Coverage cap by answer length.

    Short answers/fragments (<=400 chars) get a 0.40 cap, medium answers
    (<=800) a tighter 0.35, and longer answers fall back to ``base_cap``.
    """
    if answer_len <= 400:
        return 0.40
    if answer_len <= 800:
        return 0.35
    return base_cap


def validate_labels(
    hallucinated_answer: str,
    labels: list[dict],
    *,
    require_spans_in_code: bool = False,
    require_balanced_fences: bool = False,
    base_coverage_cap: float = MAX_LABEL_COVERAGE,
    leaky_terms: tuple[str, ...] = LEAKY_TERMS,
    prompt_residue: tuple[str, ...] = PROMPT_RESIDUE,
) -> tuple[bool, str]:
    """Validate that hallucination labels meet quality thresholds.

    :param require_spans_in_code: enforce that every span lies inside a fenced
        code block (for mixed prose+code answers where only code is corrupted).
    :param require_balanced_fences: enforce an even, non-zero count of ``` fences.
    :return: (is_valid, reason).
    """
    if not labels:
        return False, "no_labels"

    for residue in prompt_residue:
        if residue in hallucinated_answer:
            return False, f"prompt_residue ({residue[:30]})"

    if require_balanced_fences:
        fence_count = hallucinated_answer.count("```")
        if fence_count % 2 != 0:
            return False, f"unbalanced_fences ({fence_count})"
        if fence_count == 0:
            return False, "no_code_fences"

    total_span = sum(lab["end"] - lab["start"] for lab in labels)
    answer_len = len(hallucinated_answer) if hallucinated_answer else 1
    coverage = total_span / answer_len

    max_coverage = _max_allowed_coverage(answer_len, base_coverage_cap)
    if coverage > max_coverage:
        return False, f"coverage_too_high ({coverage:.0%} > {max_coverage:.0%})"

    previous_end = -1
    for lab in labels:
        span_len = lab["end"] - lab["start"]
        if span_len < MIN_LABEL_SPAN_CHARS:
            return False, f"span_too_short ({span_len} chars)"
        if span_len > MAX_LABEL_SPAN_CHARS:
            return False, f"span_too_long ({span_len} chars)"
        if lab["start"] < previous_end:
            return False, "overlapping_or_unsorted_labels"
        previous_end = lab["end"]

        span_text = hallucinated_answer[lab["start"] : lab["end"]]
        if _contains_leakage(span_text, leaky_terms):
            return False, "leaky_label_text"

        if require_spans_in_code and not _span_is_in_code(
            hallucinated_answer, lab["start"], lab["end"]
        ):
            return False, "label_outside_code_block"

    return True, ""


# ── Engine: request edits, apply, validate ─────────────────────────────────────


def _parse_changes(raw: str | None) -> dict | None:
    """Parse a ``{"changes": [...]}`` object out of a raw model response."""
    if not raw:
        return None
    json_match = re.search(r"\{[\s\S]*\}", raw.strip())
    if not json_match:
        return None
    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(result.get("changes"), list) or not result["changes"]:
        return None
    return result


def _changes_messages(system_prompt: str, user_msg: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]


def _finalize_injection(
    clean_answer: str,
    result: dict | None,
    hall_type: str,
    *,
    require_spans_in_code: bool,
    require_balanced_fences: bool,
    base_coverage_cap: float,
) -> InjectionResult:
    """Apply edits and validate. Pure (no I/O); shared by sync and async paths."""
    if result is None:
        return InjectionResult(ok=False, reason="llm_no_response")

    changes = result.get("changes", [])
    hallucinated_answer, labels = apply_changes_to_answer(clean_answer, changes, hall_type)
    if hallucinated_answer is None or labels is None:
        return InjectionResult(ok=False, reason="substring_not_unique_or_too_short")

    ordered_changes = _sort_changes_by_original_position(clean_answer, changes)
    if ordered_changes is None:
        return InjectionResult(ok=False, reason="substring_not_unique_or_too_short")

    if require_balanced_fences and hallucinated_answer.count("```") % 2 != 0:
        hallucinated_answer = hallucinated_answer.rstrip() + "\n```"

    valid, reason = validate_labels(
        hallucinated_answer,
        labels,
        require_spans_in_code=require_spans_in_code,
        require_balanced_fences=require_balanced_fences,
        base_coverage_cap=base_coverage_cap,
    )
    if not valid:
        return InjectionResult(ok=False, reason=f"validation:{reason}")

    return InjectionResult(
        ok=True,
        hallucinated_answer=hallucinated_answer,
        labels=labels,
        changes=ordered_changes,
    )


def inject(
    client: OpenAI,
    model: str,
    *,
    clean_answer: str,
    hall_type: str,
    system_prompt: str,
    user_msg: str,
    temperature: float = 0.8,
    completion_kwargs: dict | None = None,
    require_spans_in_code: bool = False,
    require_balanced_fences: bool = False,
    base_coverage_cap: float = MAX_LABEL_COVERAGE,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> InjectionResult:
    """Inject localized hallucinations into ``clean_answer``.

    Requests replacement edits from the model, applies them deterministically,
    and validates the resulting spans. Low-level entry point; most callers use
    :func:`inject_taxonomy`.
    """
    result, reasoning = complete(
        client,
        model,
        _changes_messages(system_prompt, user_msg),
        transform=_parse_changes,
        temperature=temperature,
        completion_kwargs=completion_kwargs or {},
        max_retries=max_retries,
        retry_delay=retry_delay,
        capture_reasoning=True,
    )
    res = _finalize_injection(
        clean_answer,
        result,
        hall_type,
        require_spans_in_code=require_spans_in_code,
        require_balanced_fences=require_balanced_fences,
        base_coverage_cap=base_coverage_cap,
    )
    res.reasoning = reasoning
    return res


async def inject_async(
    aclient: AsyncOpenAI,
    model: str,
    *,
    clean_answer: str,
    hall_type: str,
    system_prompt: str,
    user_msg: str,
    temperature: float = 0.8,
    completion_kwargs: dict | None = None,
    require_spans_in_code: bool = False,
    require_balanced_fences: bool = False,
    base_coverage_cap: float = MAX_LABEL_COVERAGE,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> InjectionResult:
    """Async twin of :func:`inject` for batched throughput against local vLLM."""
    result, reasoning = await complete_async(
        aclient,
        model,
        _changes_messages(system_prompt, user_msg),
        transform=_parse_changes,
        temperature=temperature,
        completion_kwargs=completion_kwargs or {},
        max_retries=max_retries,
        retry_delay=retry_delay,
        capture_reasoning=True,
    )
    res = _finalize_injection(
        clean_answer,
        result,
        hall_type,
        require_spans_in_code=require_spans_in_code,
        require_balanced_fences=require_balanced_fences,
        base_coverage_cap=base_coverage_cap,
    )
    res.reasoning = reasoning
    return res


# ── Taxonomy layer: modality-aware, category/subtype-driven ────────────────────

# Per-modality framing for what "the context" is and what an edit must do.
MODALITY_NOTES: dict[str, str] = {
    "code": (
        "The context is source code. An edit must contradict, or fabricate "
        "against, what the code actually contains."
    ),
    "tool_output": (
        "The context is the output of a developer tool (logs, command output, "
        "file dumps, VCS metadata, build/test results). An edit must misreport "
        "what the output actually shows."
    ),
    "markdown": (
        "The context is a markdown document (prose, tables, code blocks, "
        "headings). An edit must contradict, or fabricate against, the document."
    ),
    "prose": ("The context is prose. An edit must contradict, or fabricate against, the text."),
}

_SYSTEM_PROMPT_TEMPLATE = """\
You are a hallucination injector for building a hallucination detection dataset.

You are given a CORRECT answer and the CONTEXT it is grounded in. Return ONLY a \
small set of localized replacement edits that turn the answer into one \
containing a specific kind of hallucination. The pipeline applies your edits; \
outside them the answer must stay identical.

{modality_note}

Target hallucination (inject exactly this kind):
- Category: {category} — {category_def}
- Subtype: {subcategory} — {subcategory_def}

CRITICAL grounding rule:
- The injected error MUST be detectable by comparing the answer against the \
provided context alone. A reader with only the context must be able to see the \
edited claim is wrong (for contradiction/fabricated_reference) or unsupported \
(for unsupported_addition). Do not rely on outside knowledge.

Rules:
- Make 1-2 DISTINCT edits targeting the {subcategory} subtype above.
- Each replacement span must be 12-120 characters and as small as possible.
- Total changed text must be LESS THAN 30% of the answer.
- Changes must be PLAUSIBLE and SUBTLE, not obviously broken.
- Do NOT add words like BUG, wrong, incorrect, deprecated, hallucination, fix.
- Each "original" must be an exact substring of the answer, appearing exactly once.
- Prefer replacing whole expressions or noun phrases over tiny fragments.

Respond in this exact JSON format (no markdown, no code fences):
{{
  "changes": [
    {{
      "original": "exact substring from the answer",
      "hallucinated": "replacement text",
      "explanation": "why this is wrong/unsupported according to the context"
    }}
  ]
}}
If you cannot find a good edit of this kind, return {{"changes": []}}.
"""


def build_injection_prompt(category: str, subcategory: str, modality: str) -> str:
    """Build the universal injection system prompt for a category/subtype/modality."""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        modality_note=MODALITY_NOTES.get(modality, MODALITY_NOTES["prose"]),
        category=category,
        category_def=CATEGORY_DEFINITIONS.get(category, ""),
        subcategory=subcategory,
        subcategory_def=SUBCATEGORY_DEFINITIONS.get(subcategory, subcategory),
    )


def _injection_user_msg(
    context: str, answer: str, category: str, subcategory: str, context_chars: int
) -> str:
    return (
        f"Context:\n{context[:context_chars]}\n\n"
        f"Correct answer to modify:\n{answer}\n\n"
        f"Return ONLY replacement edits that inject a {category}/{subcategory} hallucination."
    )


def sample_target(
    rng: random.Random,
    allowed: dict[str, list[str]] | None = None,
) -> tuple[str, str | None]:
    """Pick a (category, subcategory) pair.

    :param allowed: optional restriction ``{category: [subcategories]}``. When
        omitted, samples uniformly over the full taxonomy.
    """
    table = allowed or SUBCATEGORIES
    category = rng.choice(list(table.keys()))
    subs = table[category] or SUBCATEGORIES.get(category, [None])
    subcategory = rng.choice(subs) if subs else None
    return category, subcategory


def inject_taxonomy(
    client: OpenAI,
    model: str,
    *,
    context: str,
    clean_answer: str,
    category: str,
    subcategory: str,
    modality: str = "prose",
    temperature: float = 0.8,
    completion_kwargs: dict | None = None,
    context_chars: int = 8000,
    require_spans_in_code: bool = False,
    require_balanced_fences: bool = False,
    max_retries: int = 3,
) -> InjectionResult:
    """Inject a specific taxonomy category/subtype into ``clean_answer``.

    On success, every label carries ``label`` (= category), ``category``, and
    ``subcategory``.
    """
    system_prompt = build_injection_prompt(category, subcategory, modality)
    user_msg = _injection_user_msg(context, clean_answer, category, subcategory, context_chars)
    result = inject(
        client,
        model,
        clean_answer=clean_answer,
        hall_type=category,
        system_prompt=system_prompt,
        user_msg=user_msg,
        temperature=temperature,
        completion_kwargs=completion_kwargs,
        require_spans_in_code=require_spans_in_code,
        require_balanced_fences=require_balanced_fences,
        max_retries=max_retries,
    )
    if result.ok:
        for lab in result.labels:
            lab["category"] = category
            lab["subcategory"] = subcategory
    return result


async def inject_taxonomy_async(
    aclient: AsyncOpenAI,
    model: str,
    *,
    context: str,
    clean_answer: str,
    category: str,
    subcategory: str,
    modality: str = "prose",
    temperature: float = 0.8,
    completion_kwargs: dict | None = None,
    context_chars: int = 8000,
    require_spans_in_code: bool = False,
    require_balanced_fences: bool = False,
    max_retries: int = 3,
) -> InjectionResult:
    """Async twin of :func:`inject_taxonomy` for batched throughput."""
    system_prompt = build_injection_prompt(category, subcategory, modality)
    user_msg = _injection_user_msg(context, clean_answer, category, subcategory, context_chars)
    result = await inject_async(
        aclient,
        model,
        clean_answer=clean_answer,
        hall_type=category,
        system_prompt=system_prompt,
        user_msg=user_msg,
        temperature=temperature,
        completion_kwargs=completion_kwargs,
        require_spans_in_code=require_spans_in_code,
        require_balanced_fences=require_balanced_fences,
        max_retries=max_retries,
    )
    if result.ok:
        for lab in result.labels:
            lab["category"] = category
            lab["subcategory"] = subcategory
    return result


# ── Menu mode: model picks the fitting hallucination types per edit ────────────


def _menu_user_msg(context: str, answer: str, context_chars: int) -> str:
    return (
        f"Provided excerpts (the source of truth):\n{context[:context_chars]}\n\n"
        f"Correct answer to modify:\n{answer}\n\n"
        "Return ONLY replacement edits, each labelled with its hallucination type."
    )


def _map_menu_labels(result: InjectionResult, source: str) -> InjectionResult:
    """Attach unified category/subcategory to each span via the source's type map."""
    if result.ok:
        for lab in result.labels:
            category, subcategory = map_label(lab["label"], source)
            lab["category"] = category
            lab["subcategory"] = subcategory
    return result


def inject_menu(
    client: OpenAI,
    model: str,
    *,
    context: str,
    clean_answer: str,
    system_prompt: str,
    source: str,
    temperature: float = 0.8,
    completion_kwargs: dict | None = None,
    context_chars: int = 8000,
    max_retries: int = 3,
) -> InjectionResult:
    """Inject with a source-specific menu prompt (model picks 1-3 fitting types).

    ``system_prompt`` is the source's injection prompt; ``source`` keys the native
    type -> unified taxonomy map. Each span is labelled with its per-edit type and
    mapped to ``category``/``subcategory``.
    """
    result = inject(
        client,
        model,
        clean_answer=clean_answer,
        hall_type="",
        system_prompt=system_prompt,
        user_msg=_menu_user_msg(context, clean_answer, context_chars),
        temperature=temperature,
        completion_kwargs=completion_kwargs,
        max_retries=max_retries,
    )
    return _map_menu_labels(result, source)


async def inject_menu_async(
    aclient: AsyncOpenAI,
    model: str,
    *,
    context: str,
    clean_answer: str,
    system_prompt: str,
    source: str,
    temperature: float = 0.8,
    completion_kwargs: dict | None = None,
    context_chars: int = 8000,
    max_retries: int = 3,
) -> InjectionResult:
    """Async twin of :func:`inject_menu`."""
    result = await inject_async(
        aclient,
        model,
        clean_answer=clean_answer,
        hall_type="",
        system_prompt=system_prompt,
        user_msg=_menu_user_msg(context, clean_answer, context_chars),
        temperature=temperature,
        completion_kwargs=completion_kwargs,
        max_retries=max_retries,
    )
    return _map_menu_labels(result, source)
