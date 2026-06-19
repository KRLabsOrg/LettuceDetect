"""Prompt + output contract for the fine-tuned generative span detectors.

These models (e.g. ``lettucedect-v2-qwen``) were supervised-fine-tuned to answer a
SPECIFIC prompt and emit a ``hallucinated_spans`` JSON object. The wording here is
therefore FROZEN — it must match the training prompt verbatim, or the model
degrades. It intentionally differs from the judge prompt in
``lettucedetect/prompts/hallucination_detection.txt`` (few-shot, ``hallucination_list``)
and from the label *definitions* in :mod:`lettucedetect.datasets.taxonomy` (which are
for data construction / the LLM judge, not this model).

Used by :class:`lettucedetect.detectors.llm.LLMDetector` in its native (trained-model)
mode; see ``LLMDetector(native=True)``.
"""

from __future__ import annotations

import json
import re

# FROZEN training wording — do not edit without retraining the models.
CATEGORY_DESCRIPTIONS = {
    "contradiction": "conflicts with the context (a wrong value, number, date, name, or relationship)",
    "fabricated_reference": "an entity, name, identifier, or section that is absent from the context",
    "unsupported_addition": "a claim, detail, or behavior the context never states",
}

SUBCATEGORY_DESCRIPTIONS = {
    "entity": "a wrong or invented name, entity, or object",
    "temporal": "an incorrect date, time, duration, or ordering",
    "numerical": "an incorrect number, quantity, or amount",
    "value": "a wrong value, setting, or attribute value",
    "relational": "an incorrect relationship or association between things",
    "identifier": "an invented identifier or name not found in the context",
    "section": "a reference to a section, part, or location that does not exist",
    "attribute": "an invented or incorrect attribute or property",
    "claim": "an added factual claim the context does not support",
    "behavior": "an added or changed action or behavior the context never states",
    "elaboration": "extra detail or elaboration beyond what the context supports",
    "subjective": "an unsupported subjective or evaluative statement",
    "unspecified": "unsupported, with no more specific subtype",
}


def build_system_prompt(explain: bool = False) -> str:
    """Build the detection system prompt; if ``explain``, also request a per-span explanation."""
    cats = "\n".join(f"- {k}: {v}" for k, v in CATEGORY_DESCRIPTIONS.items())
    subs = "\n".join(f"- {k}: {v}" for k, v in SUBCATEGORY_DESCRIPTIONS.items())
    expl_clause = ", and give a short explanation of why it is unsupported" if explain else ""
    expl_field = ', "explanation": "..."' if explain else ""
    return (
        "You are an expert annotator who identifies hallucinated spans in a generated answer "
        "with respect to a given context (the only trusted evidence). A hallucinated span is a "
        "substring of the answer that is not supported by the context. Spans consistent with the "
        "context are not hallucinations.\n\n"
        "Quote each hallucinated span verbatim from the answer and classify it into exactly one "
        f"category and one subcategory{expl_clause}.\n\n"
        f"Categories (the kinds of unsupported span):\n{cats}\n\n"
        f"Subcategories:\n{subs}\n\n"
        "Reply with ONLY a JSON object (no markdown, no code fences): "
        f'{{"hallucinated_spans": [{{"text": "...", "category": "...", "subcategory": "..."{expl_field}}}]}}. '
        'If nothing is unsupported, reply {"hallucinated_spans": []}.'
    )


SYSTEM_BASE = build_system_prompt(explain=False)
SYSTEM_EXPL = build_system_prompt(explain=True)


def build_user_message(context: str, answer: str) -> str:
    """User turn: the full generation prompt (request + context) followed by the answer."""
    return f"{context}\n\nAnswer to verify:\n{answer}"


def parse_spans(raw: str) -> list[dict]:
    """Parse a model reply into the list under ``hallucinated_spans`` (tolerant of fences)."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return []
        payload = json.loads(m.group(0))
    items = payload.get("hallucinated_spans", [])
    return items if isinstance(items, list) else []


def spans_to_offsets(answer: str, spans: list[dict]) -> list[dict]:
    """Locate each span's ``text`` in the answer (first non-overlapping verbatim match)."""
    out: list[dict] = []
    used: list[tuple[int, int]] = []
    for sp in spans:
        if not isinstance(sp, dict):
            continue
        sub = sp.get("text")
        if not sub:
            continue
        start = -1
        search_from = 0
        while True:
            idx = answer.find(sub, search_from)
            if idx == -1:
                break
            end = idx + len(sub)
            if not any(idx < ue and us < end for us, ue in used):
                start = idx
                break
            search_from = idx + 1
        if start == -1:
            continue
        end = start + len(sub)
        used.append((start, end))
        span = {"start": start, "end": end, "text": sub}
        for key in ("category", "subcategory", "explanation"):
            if sp.get(key) is not None:
                span[key] = sp[key]
        out.append(span)
    return out
