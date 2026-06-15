import streamlit as st
import streamlit.components.v1 as components

from lettucedetect.models.inference import HallucinationDetector


def create_interactive_text(text: str, spans: list[dict[str, int | float]]) -> str:
    """Create interactive HTML with highlighting and hover effects.

    :param text: The text to create the interactive text for.
    :param spans: The spans to highlight.
    :return: The interactive text.
    """
    html_text = text

    for span in sorted(spans, key=lambda x: x["start"], reverse=True):
        span_text = text[span["start"] : span["end"]]
        highlighted_span = f'<span class="hallucination" title="Confidence: {span["confidence"]:.3f}">{span_text}</span>'
        html_text = html_text[: span["start"]] + highlighted_span + html_text[span["end"] :]

    return f"""
    <style>
        .container {{
            font-family: Arial, sans-serif;
            font-size: 16px;
            line-height: 1.6;
            padding: 20px;
        }}
        .hallucination {{
            background-color: rgba(255, 99, 71, 0.3);
            padding: 2px;
            border-radius: 3px;
            cursor: help;
        }}
        .hallucination:hover {{
            background-color: rgba(255, 99, 71, 0.5);
        }}
    </style>
    <div class="container">{html_text}</div>
    """


def main():
    st.set_page_config(page_title="Lettuce Detective")

    st.image(
        "https://github.com/KRLabsOrg/LettuceDetect/blob/main/assets/lettuce_detective.png?raw=true",
        width=600,
    )

    st.title("Let Us Detect Your Hallucinations")

    @st.cache_resource
    def load_detector():
        return HallucinationDetector(
            method="transformer",
            model_path="output/hallucination_detection_ettin_17m",
        )

    detector = load_detector()

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

        html_content = create_interactive_text(answer, predictions)
        components.html(html_content, height=200)


if __name__ == "__main__":
    main()
