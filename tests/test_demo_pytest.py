"""Tests for the Streamlit demo's pure HTML renderer."""

from demo.streamlit_demo import CATEGORY_COLORS, DEFAULT_SPAN_COLOR, create_interactive_text


def test_create_interactive_text_renders_typed_span() -> None:
    answer = "The population is 69 million."
    start = answer.index("69 million")
    spans = [
        {
            "start": start,
            "end": start + len("69 million"),
            "text": "69 million",
            "confidence": 0.9876,
            "category": "contradiction",
            "subcategory": "numerical",
        }
    ]

    rendered = create_interactive_text(answer, spans)

    assert f"background-color: {CATEGORY_COLORS['contradiction']}" in rendered
    assert "Category: contradiction" in rendered
    assert "Subcategory: numerical" in rendered
    assert "Confidence: 0.988" in rendered
    assert ">69 million</span>" in rendered


def test_create_interactive_text_keeps_binary_span_style() -> None:
    answer = "Paris has 12 million residents."
    start = answer.index("12 million")
    spans = [
        {
            "start": start,
            "end": start + len("12 million"),
            "text": "12 million",
            "confidence": 0.75,
        }
    ]

    rendered = create_interactive_text(answer, spans)

    assert f"background-color: {DEFAULT_SPAN_COLOR}" in rendered
    assert "Confidence: 0.750" in rendered
    assert "Category:" not in rendered
    assert ">12 million</span>" in rendered


def test_create_interactive_text_preserves_multiple_span_order() -> None:
    answer = "Alpha beta gamma delta."
    spans = [
        {
            "start": answer.index("Alpha"),
            "end": answer.index("Alpha") + len("Alpha"),
            "confidence": 0.8,
            "category": "unsupported_addition",
            "subcategory": "claim",
        },
        {
            "start": answer.index("gamma"),
            "end": answer.index("gamma") + len("gamma"),
            "confidence": 0.9,
            "category": "fabricated_reference",
            "subcategory": "entity",
        },
    ]

    rendered = create_interactive_text(answer, spans)

    assert rendered.index(">Alpha</span>") < rendered.index(">gamma</span>")
    assert rendered.count('class="hallucination"') == 2
    assert f"background-color: {CATEGORY_COLORS['unsupported_addition']}" in rendered
    assert f"background-color: {CATEGORY_COLORS['fabricated_reference']}" in rendered
