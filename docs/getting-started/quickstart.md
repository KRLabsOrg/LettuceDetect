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

## Use with plain Transformers

The encoder models are standard Hugging Face token classifiers, so you can use them without
installing `lettucedetect`. Tokenize the context and answer as a pair, then map label `1`
(unsupported) over the answer segment back to character spans:

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_id = "KRLabsOrg/lettucedect-v2-mmbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id).eval()

context = "France is in Western Europe. Its capital Paris had about 2.1 million people in 2019."
answer = "The capital of France is Paris, with a population of about 4.5 million people."

encoding = tokenizer(
    context,
    answer,
    truncation="only_first",
    max_length=4096,
    return_offsets_mapping=True,
    return_tensors="pt",
)
with torch.no_grad():
    predictions = model(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
    ).logits.argmax(-1)[0]

sequence_ids = encoding.sequence_ids(0)
offsets = encoding["offset_mapping"][0].tolist()
spans = []
current_span = None
for index, (sequence_id, (start, end)) in enumerate(zip(sequence_ids, offsets)):
    if sequence_id != 1 or start == end:
        continue
    if predictions[index].item() == 1:
        current_span = [start, end] if current_span is None else [current_span[0], end]
    elif current_span:
        spans.append(current_span)
        current_span = None
if current_span:
    spans.append(current_span)

print([{"text": answer[start:end], "start": start, "end": end} for start, end in spans])
# [{'text': '4.5 million', 'start': 59, 'end': 70}]
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
