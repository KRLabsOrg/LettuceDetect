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
or cost matters and binary spans suffice. Both vastly outperform off-the-shelf detectors
(HHEM, Lynx-8B, Granite-Guardian-8B) and a frontier LLM-as-judge on code, which sit near
chance there.

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
