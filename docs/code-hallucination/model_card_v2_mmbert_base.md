---
license: apache-2.0
language:
- en
- de
- fr
- es
- it
- pl
- zh
tags:
- token-classification
- hallucination-detection
- rag
- span-detection
- code
base_model: jhu-clsp/mmBERT-base
pipeline_tag: token-classification
datasets:
- KRLabsOrg/lettucedetect-code-hallucination
- KRLabsOrg/lettucedetect-prose-hallucination
---

# lettucedect-v2-mmbert-base: Encoder Hallucination Span Detection

## TL;DR — results

A **fast encoder** (binary token classifier) for span-level hallucination detection across **code, tool output, and prose**, single forward pass:

- **Unified test set (10,698):** span-F1 **0.642**, example-F1 **0.869**, IoU 0.671.
- **Code-agent answers:** span-F1 **0.508** — close to the generative 2B (0.602) at a fraction of the size, and far above the off-the-shelf detectors and LLM judges we tested (HHEM / Lynx / Granite / MiniCheck / Nemotron-550B / gpt-oss-120b), which sit near chance.
- **Prose:** RAGTruth example-F1 74.3 (above GPT-4 63.4 and Luna 65.4); multilingual PsiloQA across 14 languages.
- Emits **binary** spans; for typed spans (category + subcategory) use `lettucedect-v2-qwen-2b` or the taxonomy-head cascade.

The small, fast option when throughput/cost matter and binary spans suffice.

## Overview

`lettucedect-v2-mmbert-base` is a lightweight **encoder** hallucination detector for
Retrieval-Augmented Generation (RAG) and coding-agent settings. It is a token-level
classifier built on **mmBERT-base**: given the context, question, and answer, it labels
each answer token as supported (0) or unsupported (1), yielding the spans of the answer
not grounded in the context — across prose **and code**, in many languages.

It is the encoder counterpart to the generative `lettucedect-v2-qwen-2b`: much smaller and
faster, single forward pass, no decoding. It returns **binary** spans (no category/subtype);
for typed spans (category + subcategory) use the generative model, or pair this detector
with the taxonomy head as a typing cascade.

## Model Details

- **Base model:** jhu-clsp/mmBERT-base (ModernBERT-family multilingual encoder)
- **Task:** token classification → unsupported-answer spans
- **Input:** `[CLS] context [SEP] question [SEP] answer [SEP]`; context/question tokens masked in the loss, answer tokens labeled 0/1
- **Training:** binary token classification on the unified code + prose benchmark
- **Context:** long-context (handles full request + retrieved context + answer)
- **Languages:** English plus more (multilingual via the PsiloQA portion)

## Usage

```python
from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(method="transformer", model_path="KRLabsOrg/lettucedect-v2-mmbert-base")
spans = detector.predict(context=[context], question=question, answer=answer, output_format="spans")
# [{"start": ..., "end": ..., "text": "...", "confidence": ...}]
```

### Plain `transformers` (no lettucedetect)

It's a standard token classifier — tokenize `(context, answer)` as a pair and read the labels
over the answer segment (1 = unsupported):

```python
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

model_id = "KRLabsOrg/lettucedect-v2-mmbert-base"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForTokenClassification.from_pretrained(model_id).eval()

context = "France is in Western Europe. Its capital Paris had about 2.1 million people in 2019."
answer = "The capital of France is Paris, with a population of about 4.5 million people."

enc = tok(context, answer, truncation="only_first", max_length=4096,
          return_offsets_mapping=True, return_tensors="pt")
with torch.no_grad():
    preds = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits.argmax(-1)[0]

seq_ids, offsets, spans, cur = enc.sequence_ids(0), enc["offset_mapping"][0].tolist(), [], None
for i, (sid, (a, b)) in enumerate(zip(seq_ids, offsets)):
    if sid != 1 or a == b:                  # keep only answer-segment, non-special tokens
        continue
    if preds[i].item() == 1:                # unsupported
        cur = [a, b] if cur is None else [cur[0], b]
    elif cur:
        spans.append(cur); cur = None
if cur:
    spans.append(cur)
print([{"text": answer[s:e], "start": s, "end": e} for s, e in spans])
# -> [{'text': '4.5 million', 'start': 59, 'end': 70}]
```

For category/subcategory typing, use the `lettucedetect` cascade (`taxonomy_head=...`) or the
generative `lettucedect-v2-qwen-2b`.

## Performance

Char-level span metrics on the unified test set (10,698 samples), per source. These are
the detector's span numbers (typing is not produced by this model).

| dataset | n | span-F1 | span-P | span-R | example-F1 | IoU |
|---|--:|--:|--:|--:|--:|--:|
| **ALL** | 10698 | **0.642** | 0.684 | 0.605 | 0.869 | 0.671 |
| acl | 440 | 0.579 | 0.705 | 0.492 | 0.873 | 0.637 |
| code-agent | 2015 | 0.508 | 0.619 | 0.430 | 0.770 | 0.581 |
| readme | 641 | 0.751 | 0.789 | 0.716 | 0.900 | 0.768 |
| tool-output | 617 | 0.588 | 0.736 | 0.490 | 0.763 | 0.645 |
| wikipedia | 1388 | 0.708 | 0.741 | 0.678 | 0.917 | 0.768 |
| psiloqa (multi-lang) | 2897 | 0.714 | 0.696 | 0.733 | 0.943 | 0.627 |
| ragtruth | 2700 | 0.528 | 0.668 | 0.437 | 0.743 | 0.724 |

The generative `lettucedect-v2-qwen-2b` scores higher on span-F1 (ALL 0.689 vs 0.642) and
adds typing, but this encoder is far smaller and faster — a strong choice when throughput
or cost matters and binary spans suffice.

### Code-agent: vs other detectors

| detector | span-F1 | example-F1 |
|---|--:|--:|
| lettucedect-v2-qwen-2b (generative 2B) | 0.602 | 0.835 |
| lettucedect-v2-lfm-8b (generative 8B) | 0.507 | 0.811 |
| **lettucedect-v2-mmbert-base (this)** | **0.508** | **0.770** |
| Nemotron-3-Ultra-550B (LLM judge, task-aware) | 0.216 | 0.700 |
| gpt-oss-120b (LLM judge, task-aware) | 0.212 | 0.691 |
| HHEM-2.1 / Lynx-8B / Granite-Guardian / MiniCheck | — | ≈ chance (BAcc ~0.50) |

This base encoder matches the 8B generative model on code-agent span-F1 and crushes the
off-the-shelf detectors and large LLM judges, which over-flag generated code.

### Prose benchmarks

**RAGTruth** (official test, example-level F1): this multilingual+code base encoder scores
**74.3** — below the RAGTruth-specialized LettuceDetect-large v1 (79.2) and the generative
`lettucedect-v2-qwen-2b` (81.8), but above prompt-based GPT-4 (63.4) and Luna (65.4). It
trades a little RAGTruth-specific accuracy for unified code+tool+multilingual coverage at
base size; `lettucedect-v2-mmbert-large` is stronger.

**PsiloQA** (14 languages): span-F1 0.714, IoU 0.627 — competitive multilingual span
detection from a base encoder (the PsiloQA paper's strongest LLM judge reaches IoU ~0.40 on
English; their fine-tuned encoder, trained on PsiloQA only, ~0.71).

## Typed spans (optional cascade)

For category + subcategory typing with an encoder pipeline, run this detector to find
spans, then type each span with the label-conditioned taxonomy head (see
`scripts/evaluate_taxonomy_cascade.py`). The cascade trails the generative model on typing
(typed-F1 0.461 vs 0.585), so prefer `lettucedect-v2-qwen-2b` when typed output is the goal.

## Citing

```bibtex
@article{Kovacs2025LettuceDetect,
  title={LettuceDetect: A Hallucination Detection Framework for RAG Applications},
  author={Kovács, Ádám and Recski, Gábor},
  journal={arXiv preprint arXiv:2502.17125},
  year={2025}
}
```

*A dedicated paper for the unified code+prose benchmark and the v2 models is in
preparation; this card will be updated with its citation on release.*
