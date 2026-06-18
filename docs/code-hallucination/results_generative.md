# Generative span detector — Qwen3.5-2B SFT results

Headline generative model for the unified hallucination-detection benchmark.
A small (2B) instruction model, LoRA-SFT'd to emit hallucinated spans as JSON
(`{hallucinated_spans:[{text, category, subcategory[, explanation]}]}`), scored
with the **same char-overlap span-F1 / example-F1 / IoU** the encoder uses, so
all numbers are directly comparable across detectors.

## Setup

- **Base:** Qwen3.5-2B (hybrid Gated DeltaNet + attention).
- **Training:** Unsloth LoRA SFT (r32/α64), full bf16, 2 epochs, 4×A40 DDP,
  effective batch 32, max sequence 32768 (0% of code rows truncated).
- **Data:** `data/generative_sft_fixed` — the merged code + prose datasets
  (145,250 train / 6,171 val / 10,698 test). Prompt = single data-agnostic
  system prompt (taxonomy enumerated) + `User request … + context` + answer.
- **Prompt fix that mattered:** the input uses the `prompt` field (which carries
  the **user request** plus context), not bare context — required so the model
  can judge `unsupported_addition` ("not requested"), and identical to the
  encoder's input. The explanation prompt variant is chosen **by source**
  (code-agent), used for both clean and hallucinated code rows, so it carries
  no label signal.
- **Eval:** vLLM serving (`--data-parallel-size 4`, ~72 it/s), greedy decode;
  `scripts/evaluate_generative_model.py` with `--explain-datasets lettucedetect-code-agent`.

## Results — per source (test split, 10,698 samples)

| dataset | n | span_F1 | span_P | span_R | ex_F1 | IoU | typed_F1 (cat) | typed_F1 (sub) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **ALL** | 10698 | **0.6888** | 0.6768 | 0.7013 | **0.9210** | 0.7577 | 0.5850 | 0.4684 |
| acl | 440 | 0.7486 | 0.7523 | 0.7448 | 0.9423 | 0.8106 | 0.7486 | 0.6063 |
| code-agent | 2015 | 0.6023 | 0.5957 | 0.6091 | 0.8347 | 0.7067 | 0.5628 | 0.5628 |
| readme | 641 | 0.8661 | 0.8894 | 0.8440 | 0.9842 | 0.8939 | 0.7897 | 0.6985 |
| tool-output | 617 | 0.7191 | 0.7371 | 0.7019 | 0.9070 | 0.7925 | 0.6452 | 0.5508 |
| wikipedia | 1388 | 0.8174 | 0.8357 | 0.7999 | 0.9738 | 0.8707 | 0.7872 | 0.7130 |
| psiloqa | 2897 | 0.7323 | 0.7022 | 0.7652 | 0.9663 | 0.6871 | 0.5652 | 0.3740 |
| ragtruth | 2700 | 0.5736 | 0.6013 | 0.5482 | 0.8181 | 0.7645 | 0.5532 | 0.5050 |

- Unparseable replies: **11 / 10,698** (0.1%).
- Tokens: prompt avg 1,918 · completion avg 52 · total avg 1,970 per call.

`typed_F1` = char-overlap **gated on category match**; `typed_F1 (sub)` additionally
requires subcategory. `span_F1 − typed_F1` is the pure typing cost: category is
right on ~85% of detected span-mass, subcategory on ~68%. (On code-agent the
typed and span numbers nearly coincide — one dominant span per sample.)

## vs the encoder (mmBERT-base, binary, same test)

Qwen **beats mmBERT on every source**, span- and example-level — including code,
which the encoder previously won:

| | ALL span_F1 | ALL ex_F1 | code-agent span_F1 | psiloqa IoU |
|---|--:|--:|--:|--:|
| mmBERT-base (binary) | 0.572 | 0.862 | 0.508 | 0.627 |
| **Qwen3.5-2B (SFT)** | **0.689** | **0.921** | **0.602** | **0.687** |

The decisive factor was the prompt fix (including the user request): the earlier
generative model, trained on a context-only prompt, scored only 0.383 span-F1 on
code-agent. Once the request is in the prompt, the generative model surpasses the
encoder on code as well.

## Honest SOTA framing

Four published numbers sit on **incomparable axes** — RAG-HAT (RAGTruth example-F1
83.9), MiniCheck (LLM-AggreFact average BAcc 77.4), Lynx (HaluBench accuracy 87.4),
LettuceDetect-large (RAGTruth **span-F1 58.93**). Against these:

- **RAGTruth example-F1: 0.818** — above LettuceDetect-large (0.792), below
  RAG-HAT (0.839). *Not* blanket SOTA; do not claim "new SOTA detector".
- **RAGTruth span-F1: 0.574** — ~1.5 pt below the span-localization SOTA (LD 0.589).
  This gap is **recall**, and is the target for the recall-weighted GRPO pass.
- **PsiloQA IoU 0.687** vs the PsiloQA-specialist encoder (~0.62) — likely SOTA on
  PsiloQA (verify vs their best).
- **Strongest unified cross-domain detector**: one model covering code + 7 prose
  sources + 14 languages, beating LD-large on RAGTruth example-F1 while also
  handling code and multilingual — that is the paper's contribution, not a
  single-benchmark record.

## GRPO refinement — neutral result

A GRPO pass on the SFT model (recall-weighted char-overlap F-beta reward, β=2;
KL anchor β=0.04; lr 1e-6; ~2,700 source-weighted prompts, 1 epoch; resumed
clean past an OOM) produced **no meaningful change** over SFT — every per-source
cell moved within ±0.005 (noise):

| dataset | SFT span-F1 | GRPO span-F1 | SFT recall | GRPO recall |
|---|--:|--:|--:|--:|
| ALL | .6888 | .6886 | .7013 | .7021 |
| ragtruth | .5736 | .5692 | .5482 | .5438 |
| code-agent | .6023 | .6047 | .6091 | .6132 |

RAGTruth span-F1 stayed ~.57 (did **not** cross the .589 span-localization SOTA).
Diagnosis: weak advantage signal (`frac_reward_zero_std ≈ 0.37` — a third of
generation groups had identical rewards → zero gradient) plus a conservative
setup (low lr, KL anchor, small prompt set, 1 epoch) kept the policy glued to the
already-strong SFT model. **The SFT model is the headline generative detector;**
GRPO is reported as a neutral ablation. (A stronger retry — higher lr, weaker KL,
curated recall-miss prompts, more epochs — could be revisited, but was deprioritized
in favor of the LFM2.5 encoder, taxonomy head, and baselines.)
