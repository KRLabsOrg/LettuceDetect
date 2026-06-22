# Models

## Code / Tool / Agentic Models (v2)

A unified family trained on the [code + tool-output + prose benchmark](https://huggingface.co/datasets/KRLabsOrg/lettucedetect-code-hallucination). One model covers coding-agent answers, tool output, and prose, and emits **typed** spans (category + subcategory).

| Model | Base | Output | Notes |
|-------|------|--------|-------|
| [lettucedect-v2-qwen-2b](https://huggingface.co/KRLabsOrg/lettucedect-v2-qwen-2b) | Qwen3.5-2B | typed spans (+ reasoning) | generative; detection and typing in one pass |
| [lettucedect-v2-mmbert-base](https://huggingface.co/KRLabsOrg/lettucedect-v2-mmbert-base) | mmBERT-base | binary spans | fast encoder detector |
| [lettucedect-v2-taxonomy-head](https://huggingface.co/KRLabsOrg/lettucedect-v2-taxonomy-head) | mmBERT-base | span typing | types encoder spans (cascade) — see [Quick Start](quickstart.md#typed-spans-v2) |

## English Models

| Model | Base | Max Tokens | Example F1 | Span F1 |
|-------|------|-----------|-----------|---------|
| [lettucedetect-base-modernbert-en-v1](https://huggingface.co/KRLabsOrg/lettucedetect-base-modernbert-en-v1) | ModernBERT-base | 4K | 76.8% | SOTA |
| [lettucedetect-large-modernbert-en-v1](https://huggingface.co/KRLabsOrg/lettucedetect-large-modernbert-en-v1) | ModernBERT-large | 4K | 79.2% | SOTA |

## Multilingual Models

| Model | Base | Languages | Max Tokens |
|-------|------|-----------|-----------|
| [lettucedetect-base-eurobert-multilingual-v1](https://huggingface.co/KRLabsOrg/lettucedetect-base-eurobert-multilingual-v1) | EuroBERT-210M | en, de, fr, es, it, pl, cn | 8K |

## TinyLettuce (Distilled)

Smaller models for resource-constrained environments. See [TinyLettuce docs](../TINYLETTUCE.md).

## Using a Model

```python
from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedetect-large-modernbert-en-v1"
)
```

Models are downloaded automatically from HuggingFace Hub on first use.
