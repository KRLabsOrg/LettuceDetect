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

# lettucedect-v2-qwen-2b: Generative Hallucination Span Detection

## TL;DR — results

A **2B** model that localizes *and* types hallucinated spans across **code, tool output, and prose** in one pass:

- **Unified test set (10,698):** span-F1 **0.689**, example-F1 **0.921**, IoU 0.758.
- **Code-agent answers:** span-F1 **0.602** / example-F1 **0.835** — where it **beats every alternative we tested**: our own 8B sibling (LFM-8B, 0.507), a self-hosted **120B** judge (0.21) and a **550B** judge (0.22), and off-the-shelf detectors (HHEM / Lynx-8B / Granite-Guardian / MiniCheck, all ≈ chance). Large general judges over-flag generated code; this model doesn't.
- **Established prose benchmarks:** RAGTruth example-F1 **0.818** (> LettuceDetect-large 0.792); PsiloQA strong across **14 languages**.

One small, fast model — no separate judge, no giant LLM — for span-level hallucination detection from RAG to agentic coding.

## Overview

`lettucedect-v2-qwen-2b` is a generative hallucination detector for Retrieval-Augmented
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
vllm serve KRLabsOrg/lettucedect-v2-qwen-2b --served-model-name lettucedect-v2-qwen-2b
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
    model="lettucedect-v2-qwen-2b", temperature=0.0,
    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
)
spans = json.loads(resp.choices[0].message.content)["hallucinated_spans"]
for s in spans:  # recover character offsets in the answer
    s["start"] = answer.find(s["text"])
    s["end"] = s["start"] + len(s["text"])
# [{"text": "Its population is 2 million.", "category": "unsupported_addition",
#   "subcategory": "numerical", "start": 32, "end": 60}]
```

Or via the LettuceDetect package: serve the model with vLLM, point `OPENAI_API_BASE` at
it, and call

```python
from lettucedetect.models.inference import HallucinationDetector
det = HallucinationDetector(method="llm", model="KRLabsOrg/lettucedect-v2-qwen-2b")
spans = det.predict(context=[context], question=question, answer=answer, output_format="spans")
```

It auto-routes this model to its native training prompt and `hallucinated_spans` output
(typed `category`/`subcategory`). Pass `native=True` if served under a different name, and
`include_reasoning=True` to request per-span explanations.

### Plain `transformers` (no vLLM, no lettucedetect)

```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "KRLabsOrg/lettucedect-v2-qwen-2b"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

SYSTEM = "..."  # the detection prompt shown above (verbatim)
context = "France is a country in Europe. Its capital is Paris."
question = "What is the capital of France? What is its population?"
answer = "The capital of France is Paris. Its population is 2 million."
user = f"User request: {question}\n\n{context}\n\nAnswer to verify:\n{answer}"

inputs = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
    add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True,
).to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
reply = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

# raw_decode reads the first JSON object and ignores any trailing text
spans = json.JSONDecoder().raw_decode(reply[reply.index("{"):])[0]["hallucinated_spans"]
for s in spans:  # recover character offsets in the answer
    s["start"] = answer.find(s["text"]); s["end"] = s["start"] + len(s["text"])
print(spans)
# -> [{'text': '2 million', 'category': 'unsupported_addition', 'subcategory': 'claim', 'start': 50, 'end': 59}]
```

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

Its strength is **unified cross-domain coverage** — one model handling code, prose, and
14 languages — rather than a single-benchmark record.

### Prose benchmarks: RAGTruth & PsiloQA

On the established prose benchmarks the same model is competitive with or better than
specialized methods, so the code/tool sources extend rather than trade off against prose.

**RAGTruth** (official test set, n=2700) — example-level F1, on the standard leaderboard:

| Method | Example-level F1 |
|---|--:|
| RAG-HAT (fine-tuned Llama-3-8B) | 83.9 |
| **lettucedect-v2-qwen-2b (ours)** | **81.8** |
| LettuceDetect-large (v1) | 79.2 |
| Fine-tuned Llama-2-13B (RAGTruth) | 78.7 |
| Luna | 65.4 |
| GPT-4 | 63.4 |

Second only to a fine-tuned 8B, and above the v1 detector, the fine-tuned 13B, Luna, and GPT-4.
(Our span-level on RAGTruth: P 0.601 / R 0.548 / F1 0.574 / IoU 0.765.)

**PsiloQA** (multilingual) — span IoU. Our generative 2B matches PsiloQA's best fine-tuned
*encoder* and is far above the strongest LLM judge:

| Method (English) | IoU |
|---|--:|
| **lettucedect-v2-qwen-2b (ours)** | **0.724** |
| mmBERT-base (PsiloQA paper, fine-tuned encoder) | 0.707 |
| Qwen2.5-32B-it, 3-shot (PsiloQA paper, LLM judge) | 0.400 |

Across all 14 languages our IoU is 0.689. *Numbers for the other methods are from the PsiloQA
paper (arXiv 2510.04849); our IoU uses the unified char-overlap scorer, so cross-paper rows
are indicative rather than an identical protocol.* Full per-language results:

**PsiloQA — per language** (n=2897, 14 languages):

| lang | n | span-F1 | example-F1 | IoU |
|---|--:|--:|--:|--:|
| **ALL** | 2897 | **0.733** | 0.966 | 0.689 |
| en | 1098 | 0.785 | 0.987 | 0.724 |
| es | 100 | 0.817 | 0.973 | 0.685 |
| ca | 100 | 0.721 | 0.959 | 0.704 |
| fi | 200 | 0.683 | 0.963 | 0.727 |
| it | 99 | 0.680 | 0.984 | 0.724 |
| cs | 100 | 0.672 | 0.974 | 0.604 |
| fa | 100 | 0.650 | 0.977 | 0.789 |
| hi | 300 | 0.649 | 0.933 | 0.637 |
| fr | 100 | 0.634 | 0.944 | 0.661 |
| zh | 300 | 0.597 | 0.952 | 0.650 |
| ar | 100 | 0.594 | 0.966 | 0.606 |
| de | 100 | 0.587 | 0.901 | 0.650 |
| sv | 100 | 0.512 | 0.959 | 0.707 |
| eu | 100 | 0.511 | 0.890 | 0.569 |

Example-F1 stays high across all 14 languages (answer-level detection is reliable);
span-F1 is strongest for higher-resource languages, as expected.

### Baselines on code-agent

Code-agent (2,015 samples) is where off-the-shelf detectors and even frontier LLM judges
collapse, and where this model's value is clearest:

| detector | span-F1 | example-F1 | notes |
|---|--:|--:|---|
| **lettucedect-v2-qwen-2b (this model)** | **0.602** | **0.835** | balanced (span P 0.60 / R 0.61) |
| lettucedect-v2-lfm-8b (our 8B sibling) | 0.507 | 0.811 | scaling our own recipe to 8B does not help |
| lettucedect-v2-mmbert-base | 0.508 | 0.770 | encoder counterpart |
| lettucedect-large (v1, EN/RAGTruth span model) | 0.172 | 0.684 | trained on prose RAG |
| gpt-oss-120b (zero-shot judge, task-aware) | 0.212 | 0.691 | self-hosted, BAcc ≈ 0.50 |
| Nemotron-3-Ultra-550B (zero-shot judge, naive prompt) | 0.186 | 0.655 | BAcc ≈ 0.50 |
| Nemotron-3-Ultra-550B (task-aware prompt) | 0.216 | 0.700 | BAcc ≈ 0.50 |
| HHEM-2.1 (sliding-window) | — | 0.632 | BAcc 0.497 |
| Lynx-8B | — | 0.609 | BAcc 0.526 |
| Granite-Guardian-4.1-8B | — | 0.663 | BAcc 0.538 |
| MiniCheck-7B (claim-level) | — | 0.670 | flags every answer, BAcc 0.500 |

The off-the-shelf detectors and the 550B judge over-flag on code-agent (balanced accuracy
near chance) because a generated code patch is not literally present in the context.

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
