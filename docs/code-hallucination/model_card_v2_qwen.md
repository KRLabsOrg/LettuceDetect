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
- text-generation
- hallucination-detection
- rag
- span-detection
- code
base_model: Qwen/Qwen3.5-2B
pipeline_tag: text-generation
datasets:
- KRLabsOrg/lettucedetect-code-hallucination
- KRLabsOrg/lettucedetect-prose-hallucination
---

![LettuceCode mascot](https://github.com/KRLabsOrg/LettuceDetect/blob/main/assets/lettuce_code.png?raw=true)

# lettucedect-v2-qwen: Generative Hallucination Span Detection

## Overview

`lettucedect-v2-qwen` is a generative hallucination detector for Retrieval-Augmented
Generation (RAG) and coding-agent settings. Given a user request, the supporting
context, and an answer, it emits the **exact spans of the answer that are not
supported by the context**, each tagged with a hallucination **category** and
**subcategory**. Unlike the encoder models in the LettuceDetect family (token
classifiers), this is an instruction-tuned generative model that returns structured
JSON, so it localizes *and* types hallucinations in a single pass — across prose
**and code**.

It is the first LettuceDetect model trained on a unified benchmark spanning code
(SWE-bench-derived coding-agent traces) and prose (RAGTruth, PsiloQA, and synthetic
ACL / README / tool-output / Wikipedia sources), in 14 languages.

## Model Details

- **Base model:** Qwen3.5-2B (hybrid Gated DeltaNet + attention)
- **Training:** LoRA supervised fine-tuning (full bf16), merged to a standalone model
- **Task:** generative span detection — outputs `{"hallucinated_spans": [{"text", "category", "subcategory"}]}`
- **Taxonomy:** 3 categories (contradiction, fabricated_reference, unsupported_addition) × 13 subcategories
- **Context:** long-context (handles the full request + retrieved context + answer)
- **Languages:** English plus 13 more (via the multilingual PsiloQA portion)

## How It Works

The model is prompted with a fixed, data-agnostic system prompt that defines a
hallucination ("a substring of the answer not supported by the context") and
enumerates the taxonomy, followed by the user request, the context, and the answer
to verify. It replies with a JSON object listing each hallucinated span verbatim
with its category and subcategory; a fully supported answer returns an empty list.
Span offsets are recovered by matching each returned substring back into the answer.

## Usage

The model is a generative span detector: serve it with vLLM (OpenAI-compatible) and
send the LettuceDetect detection prompt, which defines a hallucination and enumerates
the taxonomy. It replies with JSON; match each returned `text` back into the answer to
get character offsets.

```bash
vllm serve KRLabsOrg/lettucedect-v2-qwen --served-model-name lettucedect-v2-qwen
```

The model expects the exact detection prompt it was trained on (below). Send the
user request + context as the user turn, ending with the answer to verify.

```python
import json
from openai import OpenAI

SYSTEM = """You are an expert annotator who identifies hallucinated spans in a generated answer with respect to a given context (the only trusted evidence). A hallucinated span is a substring of the answer that is not supported by the context. Spans consistent with the context are not hallucinations.

Quote each hallucinated span verbatim from the answer and classify it into exactly one category and one subcategory.

Categories (the kinds of unsupported span):
- contradiction: conflicts with the context (a wrong value, number, date, name, or relationship)
- fabricated_reference: an entity, name, identifier, or section that is absent from the context
- unsupported_addition: a claim, detail, or behavior the context never states

Subcategories:
- entity: a wrong or invented name, entity, or object
- temporal: an incorrect date, time, duration, or ordering
- numerical: an incorrect number, quantity, or amount
- value: a wrong value, setting, or attribute value
- relational: an incorrect relationship or association between things
- identifier: an invented identifier or name not found in the context
- section: a reference to a section, part, or location that does not exist
- attribute: an invented or incorrect attribute or property
- claim: an added factual claim the context does not support
- behavior: an added or changed action or behavior the context never states
- elaboration: extra detail or elaboration beyond what the context supports
- subjective: an unsupported subjective or evaluative statement
- unspecified: unsupported, with no more specific subtype

Reply with ONLY a JSON object (no markdown, no code fences): {"hallucinated_spans": [{"text": "...", "category": "...", "subcategory": "..."}]}. If nothing is unsupported, reply {"hallucinated_spans": []}."""

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
context = "France is a country in Europe. Its capital is Paris."
question = "What is the capital of France? What is its population?"
answer = "The capital of France is Paris. Its population is 2 million."
user = f"User request: {question}\n\n{context}\n\nAnswer to verify:\n{answer}"

resp = client.chat.completions.create(
    model="lettucedect-v2-qwen", temperature=0.0,
    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
)
spans = json.loads(resp.choices[0].message.content)["hallucinated_spans"]
for s in spans:  # recover character offsets in the answer
    s["start"] = answer.find(s["text"])
    s["end"] = s["start"] + len(s["text"])
# [{"text": "Its population is 2 million.", "category": "unsupported_addition",
#   "subcategory": "numerical", "start": 32, "end": 60}]
```

A first-class `HallucinationDetector(method="generative")` API (vLLM-backed, taxonomy +
explanation built in) is coming to the LettuceDetect package.

## Performance

Char-level span-F1 / example-F1 / IoU on the unified test set (10,698 samples). The
model **beats the mmBERT-base encoder on every source**, span- and example-level.

| dataset | n | span-F1 | example-F1 | IoU |
|---|--:|--:|--:|--:|
| **ALL** | 10698 | **0.689** | **0.921** | 0.758 |
| acl | 440 | 0.749 | 0.942 | 0.811 |
| code-agent | 2015 | 0.602 | 0.835 | 0.707 |
| readme | 641 | 0.866 | 0.984 | 0.894 |
| tool-output | 617 | 0.719 | 0.907 | 0.793 |
| wikipedia | 1388 | 0.817 | 0.974 | 0.871 |
| psiloqa (14 langs) | 2897 | 0.732 | 0.966 | 0.687 |
| ragtruth | 2700 | 0.574 | 0.818 | 0.765 |

On RAGTruth the example-F1 (0.818) is above LettuceDetect-large (0.792); on PsiloQA
the IoU (0.687) is competitive with PsiloQA-specialist encoders. Its strength is
**unified cross-domain coverage** — one model handling code, prose, and 14 languages
— rather than a single-benchmark record.

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
