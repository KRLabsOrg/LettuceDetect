# LettuceDetect

<p align="center">
  <img src="https://github.com/KRLabsOrg/LettuceDetect/blob/main/assets/lettuce_detective.png?raw=true" alt="LettuceDetect Logo" width="300"/>
</p>

**A lightweight hallucination detection framework for RAG applications.**

LettuceDetect is an encoder-based model built on [ModernBERT](https://github.com/AnswerDotAI/ModernBERT) that detects unsupported spans in LLM-generated answers by comparing them against provided context. It provides token-level precision for identifying exactly which parts of an answer are hallucinated.

## Highlights

- **Token-level precision** — identifies exact hallucinated spans, not just "this answer has a problem"
- **Typed spans** — the v2 models label each span with a hallucination category and subcategory
- **Code, tool output & agentic workflows** — v2 models extend detection beyond prose to coding-agent answers, tool output, and structured documents
- **Fast inference** — 30-60 samples/sec on A100, suitable for production
- **Long context** — supports up to 4K tokens (ModernBERT) or 8K tokens (EuroBERT)
- **Multilingual** — English, German, French, Spanish, Italian, Polish, Chinese, Hungarian
- **Open source** — MIT license, models on HuggingFace

## Quick Example

```python
from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedetect-base-modernbert-en-v1"
)

contexts = ["The capital of France is Paris. The population is 67 million."]
question = "What is the capital and population of France?"
answer = "The capital of France is Paris. The population is 69 million."

predictions = detector.predict(
    context=contexts, question=question, answer=answer, output_format="spans"
)
# [{'start': 31, 'end': 71, 'confidence': 0.99, 'text': ' The population of France is 69 million.'}]
```

## Performance

| Model | Example F1 | vs GPT-4 | vs Luna | Parameters |
|-------|-----------|----------|---------|-----------|
| lettucedetect-base-v1 | 76.8% | +13.4% | +11.4% | 149M |
| lettucedetect-large-v1 | **79.2%** | +15.8% | +13.8% | 395M |

Evaluated on [RAGTruth](https://aclanthology.org/2024.acl-long.585/) test set. Surpasses GPT-4, Luna, and fine-tuned Llama-2-13B.

## What's New

- **v2 models — code, tool & agentic detection with typed spans** — a new family trained on a unified code + tool-output + prose benchmark that emits **typed** spans (category + subcategory) in one pass: generative [`lettucedect-v2-qwen-2b`](https://huggingface.co/KRLabsOrg/lettucedect-v2-qwen-2b), fast encoder [`lettucedect-v2-mmbert-base`](https://huggingface.co/KRLabsOrg/lettucedect-v2-mmbert-base), and the [`lettucedect-v2-taxonomy-head`](https://huggingface.co/KRLabsOrg/lettucedect-v2-taxonomy-head) typing head. See [Quick Start](getting-started/quickstart.md#typed-spans-v2) and [Models](getting-started/models.md).
- **[Code Hallucination Dataset](code-hallucination/index.md)** — A pipeline for generating span-level code hallucination data from SWE-bench
- **Multilingual models** — EuroBERT-based models for 8 languages
- **Web API** — FastAPI server with async client support

## Links

- [GitHub](https://github.com/KRLabsOrg/LettuceDetect)
- [PyPI](https://pypi.org/project/lettucedetect/)
- [arXiv Paper](https://arxiv.org/abs/2502.17125)
- [HuggingFace Models](https://huggingface.co/KRLabsOrg)
- [Streamlit Demo](https://huggingface.co/spaces/KRLabsOrg/LettuceDetect)
