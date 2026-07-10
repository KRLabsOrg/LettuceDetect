"""Streamlit demo for binary and taxonomy-typed hallucination spans."""

from __future__ import annotations

from html import escape
from typing import Any

CATEGORY_COLORS = {
    "contradiction": "rgba(255, 99, 71, 0.35)",
    "unsupported_addition": "rgba(255, 193, 7, 0.35)",
    "fabricated_reference": "rgba(156, 39, 176, 0.30)",
}
DEFAULT_SPAN_COLOR = "rgba(255, 99, 71, 0.30)"

MODEL_OPTIONS = {
    "Typed spans (v2 encoder cascade)": {
        "model_path": "KRLabsOrg/lettucedect-v2-mmbert-base",
        "taxonomy_head": "KRLabsOrg/lettucedect-v2-taxonomy-head",
    },
    "Binary spans (ModernBERT)": {
        "model_path": "KRLabsOrg/lettucedect-base-modernbert-en-v1",
    },
}


def create_interactive_text(text: str, spans: list[dict[str, Any]]) -> str:
    """Create highlighted HTML for typed and untyped hallucination spans.

    :param text: The answer text to highlight.
    :param spans: Span dictionaries containing offsets and optional taxonomy labels.
    :return: HTML containing the highlighted answer.
    """
    html_text = text

    for span in sorted(spans, key=lambda item: item["start"], reverse=True):
        start = int(span["start"])
        end = int(span["end"])
        span_text = escape(text[start:end])
        category = span.get("category")
        subcategory = span.get("subcategory")
        confidence = span.get("confidence")
        color = CATEGORY_COLORS.get(str(category), DEFAULT_SPAN_COLOR)

        tooltip_parts = []
        if category:
            tooltip_parts.append(f"Category: {category}")
        if subcategory:
            tooltip_parts.append(f"Subcategory: {subcategory}")
        if isinstance(confidence, int | float):
            tooltip_parts.append(f"Confidence: {confidence:.3f}")
        tooltip = escape(" | ".join(tooltip_parts) or "Hallucination", quote=True)

        highlighted_span = (
            f'<span class="hallucination" style="background-color: {color}" '
            f'title="{tooltip}">{span_text}</span>'
        )
        html_text = html_text[:start] + highlighted_span + html_text[end:]

    return f"""
    <style>
        .container {{
            font-family: Arial, sans-serif;
            font-size: 16px;
            line-height: 1.6;
            padding: 20px;
        }}
        .hallucination {{
            padding: 2px;
            border-radius: 3px;
            cursor: help;
        }}
        .hallucination:hover {{
            filter: saturate(1.5);
        }}
    </style>
    <div class="container">{html_text}</div>
    """


def create_legend_html() -> str:
    """Return a compact legend for typed span categories."""
    items = "".join(
        (
            '<span style="display: inline-flex; align-items: center; margin-right: 1rem">'
            f'<span style="background: {color}; width: 0.8rem; height: 0.8rem; '
            f'display: inline-block; margin-right: 0.3rem"></span>{escape(category)}</span>'
        )
        for category, color in CATEGORY_COLORS.items()
    )
    return f'<div style="margin: 0.5rem 0 1rem"><strong>Categories:</strong> {items}</div>'


def main() -> None:
    """Run the Streamlit demo."""
    import streamlit as st
    import streamlit.components.v1 as components

    from lettucedetect.models.inference import HallucinationDetector

    st.set_page_config(page_title="Lettuce Detective")

    st.image(
        "https://github.com/KRLabsOrg/LettuceDetect/blob/main/assets/lettuce_detective.png?raw=true",
        width=600,
    )

    st.title("Let Us Detect Your Hallucinations")

    model_label = st.sidebar.selectbox("Detector", list(MODEL_OPTIONS))

    @st.cache_resource
    def load_detector(selected_model: str) -> HallucinationDetector:
        return HallucinationDetector(method="transformer", **MODEL_OPTIONS[selected_model])

    detector = load_detector(model_label)

    context = st.text_area(
        "Context",
        "France is a country in Europe. The capital of France is Paris. The population of France is 67 million.",
        height=100,
    )

    question = st.text_area(
        "Question",
        "What is the capital of France? What is the population of France?",
        height=100,
    )

    answer = st.text_area(
        "Answer",
        "The capital of France is Paris. The population of France is 69 million.",
        height=100,
    )

    if st.button("Detect Hallucinations"):
        predictions = detector.predict(
            context=[context], question=question, answer=answer, output_format="spans"
        )

        if any(prediction.get("category") for prediction in predictions):
            st.markdown(create_legend_html(), unsafe_allow_html=True)

        html_content = create_interactive_text(answer, predictions)
        components.html(html_content, height=200)

        if predictions:
            rows = [
                {
                    "start": prediction.get("start"),
                    "end": prediction.get("end"),
                    "text": prediction.get("text", answer[prediction["start"] : prediction["end"]]),
                    "category": prediction.get("category", "untyped"),
                    "subcategory": prediction.get("subcategory", ""),
                    "confidence": prediction.get("confidence"),
                }
                for prediction in predictions
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
