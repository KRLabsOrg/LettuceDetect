"""Single source of truth for the hallucination taxonomy + detection prompt.

Shared by the generative SFT prompt (build_generative_sft), the zero-shot LLM
baseline + our-model eval (evaluate_generative_model imports those), and the
taxonomy head (train_taxonomy_head), so the label set, descriptions, and task
definition never drift across systems.

Descriptions are DATA-AGNOSTIC — they apply to prose, code, tool output, and
structured documents alike (no domain-specific wording). The taxonomy itself is
the definition of what counts as a hallucination: a span the context does not
support falls into exactly one category (and subcategory).
"""

from __future__ import annotations

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
    """Build the detection system prompt; if explain, also request a per-span explanation."""
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
