# Quick Start

## Detect Hallucinations

```python
from lettucedetect.models.inference import HallucinationDetector

# Load a pre-trained model
detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedetect-base-modernbert-en-v1"
)

# Provide context, question, and answer
contexts = [
    "France is a country in Europe. The capital of France is Paris. "
    "The population of France is 67 million."
]
question = "What is the capital of France? What is the population?"
answer = "The capital of France is Paris. The population of France is 69 million."

# Get span-level predictions
predictions = detector.predict(
    context=contexts,
    question=question,
    answer=answer,
    output_format="spans"
)
print(predictions)
# [{'start': 31, 'end': 71, 'confidence': 0.99,
#   'text': ' The population of France is 69 million.'}]
```

## Available Models

| Model | Language | Context | Size |
|-------|----------|---------|------|
| `KRLabsOrg/lettucedetect-base-modernbert-en-v1` | English | 4K | 149M |
| `KRLabsOrg/lettucedetect-large-modernbert-en-v1` | English | 4K | 395M |
| `KRLabsOrg/lettucedetect-base-eurobert-multilingual-v1` | 7 languages | 8K | 210M |
| `KRLabsOrg/lettucedect-v2-qwen-2b` | code / tool / prose | — | 2B |
| `KRLabsOrg/lettucedect-v2-mmbert-base` | code / tool / prose | 8K | 307M |

See [Models](models.md) for the full list.

## Typed spans (v2)

The v2 models assign each span a **category** and **subcategory** from the hallucination taxonomy. With the encoder cascade, add a typing head on top of the fast binary detector:

```python
from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedect-v2-mmbert-base",       # finds spans
    taxonomy_head="KRLabsOrg/lettucedect-v2-taxonomy-head",  # types them
)
predictions = detector.predict(
    context=contexts, question=question, answer=answer, output_format="spans"
)
# [{'start': 31, 'end': 71, 'text': ' The population of France is 69 million.',
#   'category': 'contradiction', 'subcategory': 'numerical'}]
```

The generative `lettucedect-v2-qwen-2b` produces the same typed spans in a single pass (no separate head).

## Detection Methods

```python
# Transformer-based (recommended for production)
detector = HallucinationDetector(method="transformer", model_path="...")

# LLM-based (uses OpenAI API)
detector = HallucinationDetector(method="llm", model_path="gpt-4o-mini")

# RAG Fact Checker (triplet-based)
detector = HallucinationDetector(method="rag_fact_checker", model_path="gpt-4o-mini")
```

## Output Formats

```python
# Span-level: exact character ranges of hallucinated text
predictions = detector.predict(..., output_format="spans")

# Sentence-level: which sentences contain hallucinations
predictions = detector.predict(..., output_format="sentences")
```
