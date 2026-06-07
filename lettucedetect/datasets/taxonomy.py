"""Unified four-category hallucination taxonomy for LettuceDetect v2.

Top-level categories (mutually exclusive per span):
  contradiction         — span conflicts with context
  unsupported_addition  — span adds a claim not in context
  fabricated_reference  — span references a non-existent named element
  omission              — document-level; material incompleteness

Subcategories are optional attributes of an already-classified span.
"""

from __future__ import annotations

# ── Top-level categories ───────────────────────────────────────────────────────

TOP_LEVEL_CATEGORIES: list[str] = [
    "supported",
    "contradiction",
    "unsupported_addition",
    "fabricated_reference",
]

# omission is treated as a document-level binary flag, not a span class
DOCUMENT_LEVEL_CATEGORIES: list[str] = ["omission"]

SUBCATEGORIES: dict[str, list[str]] = {
    "contradiction": ["numerical", "temporal", "entity", "relational", "value"],
    "unsupported_addition": ["claim", "elaboration", "subjective", "behavior"],
    "fabricated_reference": ["entity", "section", "identifier", "attribute"],
}

# Natural-language definitions for prompting and documentation. Single source of
# truth shared by the data generators and docs/taxonomy.md.
CATEGORY_DEFINITIONS: dict[str, str] = {
    "contradiction": (
        "The span directly conflicts with the context: the context states one "
        "thing and the answer asserts a different, incompatible thing."
    ),
    "unsupported_addition": (
        "The span asserts something the context neither states nor implies — "
        "plausible, but not derivable from the context."
    ),
    "fabricated_reference": (
        "The span references a named element (identifier, section, file, entity, "
        "attribute) that does not appear anywhere in the context."
    ),
}

SUBCATEGORY_DEFINITIONS: dict[str, str] = {
    "numerical": "a number, count, or quantity",
    "temporal": "a date, time, or ordering",
    "entity": "a named entity (person, place, object, library, commit, ...)",
    "relational": "a relationship or mapping between entities",
    "value": "a value, argument, setting, or condition",
    "claim": "a standalone factual claim",
    "elaboration": "an added explanatory detail",
    "subjective": "an opinion or evaluation",
    "behavior": "described behavior or effect",
    "section": "a section, line number, or location reference",
    "identifier": "a function, method, class, or variable name",
    "attribute": "a field, property, or attribute name",
}

# ── Source-label mapping tables ────────────────────────────────────────────────
# Each entry: native_label -> (top_level_category, subcategory | None)

# RAGTruth — heuristic mapping; proper-noun detection handles entity split
RAGTRUTH_MAP: dict[str, tuple[str, str | None]] = {
    "Evident Conflict": ("contradiction", None),
    "Subtle Conflict": ("contradiction", None),
    "Evident Baseless Info": ("unsupported_addition", "claim"),
    "Subtle Baseless Info": ("unsupported_addition", "elaboration"),
}

# LD prose generator (existing ErrorType values)
PROSE_GENERATOR_MAP: dict[str, tuple[str, str | None]] = {
    "FACTUAL": ("contradiction", "entity"),
    "TEMPORAL": ("contradiction", "temporal"),
    "NUMERICAL": ("contradiction", "numerical"),
    "RELATIONAL": ("contradiction", "relational"),
    "CONTEXTUAL": ("unsupported_addition", "claim"),
    "OMISSION": ("omission", None),
    # Extended types (v2)
    "FABRICATED_ENTITY": ("fabricated_reference", "entity"),
    "SUBJECTIVE": ("unsupported_addition", "subjective"),
    "UNVERIFIABLE": ("unsupported_addition", "claim"),
}

# LD code hallucination (SWE-bench-derived)
CODE_MAP: dict[str, tuple[str, str | None]] = {
    "structural": ("fabricated_reference", "identifier"),
    "behavioral": ("contradiction", "value"),
    "semantic": ("unsupported_addition", "behavior"),
}

# LD code-agent hallucination; the injector returns one of these per edit
CODE_AGENT_MAP: dict[str, tuple[str, str | None]] = {
    "wrong_implementation": ("contradiction", "value"),
    "unrequested_change": ("unsupported_addition", "behavior"),
    "omission": ("omission", None),
    "fabricated_api": ("fabricated_reference", "identifier"),
}

# LD generic markdown hallucination (READMEs, wiki); one type per edit
MARKDOWN_MAP: dict[str, tuple[str, str | None]] = {
    "NUMERICAL": ("contradiction", "numerical"),
    "TEMPORAL": ("contradiction", "temporal"),
    "ENTITY": ("contradiction", "entity"),
    "RELATIONAL": ("contradiction", "relational"),
    "FABRICATED": ("fabricated_reference", "entity"),
    "CLAIM": ("unsupported_addition", "claim"),
}

# FAVA labels
FAVA_MAP: dict[str, tuple[str, str | None]] = {
    "Entity": ("contradiction", "entity"),
    "Relation": ("contradiction", "relational"),
    "Contradictory": ("contradiction", None),
    "Invented": ("fabricated_reference", "entity"),
    "Subjective": ("unsupported_addition", "subjective"),
    "Unverifiable": ("unsupported_addition", "claim"),
}

# LD academic-paper hallucination (ACL); the injector returns one of these per edit
PAPER_MAP: dict[str, tuple[str, str | None]] = {
    "NUMERICAL": ("contradiction", "numerical"),
    "ENTITY": ("contradiction", "entity"),
    "RELATIONAL": ("contradiction", "relational"),
    "METHODOLOGICAL": ("contradiction", "value"),
    "CITATIONAL": ("contradiction", "entity"),
}

# ── Lookup helpers ─────────────────────────────────────────────────────────────

_ALL_MAPS: dict[str, dict[str, tuple[str, str | None]]] = {
    "ragtruth": RAGTRUTH_MAP,
    "prose_generator": PROSE_GENERATOR_MAP,
    "code": CODE_MAP,
    "code_agent": CODE_AGENT_MAP,
    "markdown": MARKDOWN_MAP,
    "fava": FAVA_MAP,
    "paper": PAPER_MAP,
}


def map_label(native_label: str, source: str) -> tuple[str, str | None]:
    """Map a native source label to (category, subcategory).

    Falls back to (native_label, None) when the label is not in the map,
    so callers don't need to handle KeyError.
    """
    mapping = _ALL_MAPS.get(source, {})
    return mapping.get(native_label, (native_label, None))


def is_ragtruth_fabricated(span_text: str, context: str) -> bool:
    """Classify Baseless Info as fabricated_reference when the span has a proper noun absent from context."""
    words = span_text.split()
    context_lower = context.lower()
    for word in words:
        if len(word) > 1 and word[0].isupper() and word.lower() not in context_lower:
            return True
    return False


def ragtruth_map_with_context(
    source_label: str, span_text: str, context: str
) -> tuple[str, str | None]:
    """Map a RAGTruth label, splitting Baseless Info into fabricated_reference vs unsupported_addition.

    Uses the span text and context: a proper noun absent from the context becomes
    ``fabricated_reference``; otherwise ``unsupported_addition``.
    """
    if "Baseless" in source_label:
        if is_ragtruth_fabricated(span_text, context):
            return "fabricated_reference", "entity"
        sub = "claim" if "Evident" in source_label else "elaboration"
        return "unsupported_addition", sub
    return RAGTRUTH_MAP.get(source_label, (source_label, None))
