# Code-Agent Hallucination Dataset

A pipeline for generating span-level code hallucination detection data from
[SWE-bench](https://www.swebench.com/). Each sample pairs repository context and a
developer request with a coding-assistant answer that is either correct or carries
character-level annotations marking the hallucinated spans.

Published as the `lettucedetect-code-agent` source of
[`KRLabsOrg/lettucedetect-code-hallucination`](https://huggingface.co/datasets/KRLabsOrg/lettucedetect-code-hallucination).

## Why this dataset

Existing hallucination-detection datasets (RAGTruth, RAGBench) focus on **text** —
question answering, summarization, data-to-text. There is no established
**span-level code hallucination dataset**; CodeMirage classifies whole snippets
but doesn't localize where the hallucination is. Here a coding assistant answers a
developer's request about a real codebase and we know exactly which character
spans of the answer are hallucinated — enabling training of both token-level
classifiers (ModernBERT) and generative span detectors (decoder LLMs).

## The task it models

A coding agent is given a developer request plus repository context and produces a
solution. The answer is the project's **real fix** (the gold patch, rendered as an
edit), into which realistic, **request-grounded** mistakes are injected — the
failure modes coding agents actually exhibit:

| Injected type | Unified category | What it is |
|---|---|---|
| `wrong_implementation` | `contradiction` | addresses the request but with the wrong logic, condition, field, or value |
| `unrequested_change` | `unsupported_addition` | also does something the request never asked for (an extra block or side effect) |
| `fabricated_api` | `fabricated_reference` | references a method/attribute/keyword that does not exist on a real object |

Clean samples are the gold fix with no labels. Symbols the answer references that
are missing from the truncated context are **grounded** — their real definitions
are pulled into the context — so a clean reference is never confused with a
fabrication. Grounding has four tiers (see [Phases](phases.md)):

1. the patch's modified functions and the full changed files,
2. modules the answer **imports** (resolved anywhere in the repo),
3. modules the **changed file imports** — base-class mixins and sibling modules,
   which grounds cross-module `self.method` calls,
4. for third-party APIs, **Context7** signatures (e.g. the real `torch.cuda.set_device`),
   added as a `Library signatures` block for structural samples.

Tiers 1–3 append a `Referenced definitions` block; tier 4 a `Library signatures`
block. `GITHUB_TOKEN` (repo fetches) and `CONTEXT7_API_KEY` (third-party docs)
raise the respective rate limits.

## Dataset overview

| Property | Value |
|---|---|
| Source | SWE-bench (all splits, repository-disjoint) |
| Samples | ~16.8k (≈73% clean / ≈27% hallucinated) |
| Repos | 44+ unique repos, zero overlap between splits |
| Answer form | the gold fix as `function` (a patched function), `fragment` (the hunk), or `edit` (`In file X, replace Y with Z`) |
| Annotation | character-level spans, unified taxonomy |

## How it runs

Two steps. Preparation is cached and only runs once; generation is the part you
re-run.

### 1. Prepare (cached inputs)

```bash
python -m scripts.code_hallucination.pipeline --all
```

Three prep phases write to `data/code_hallucination/`:

1. **load** — SWE-bench instances → `swebench_instances.json`
2. **fetch source** — repository files at the base commit → `source_cache/{instance_id}.json`
3. **rewrite requests** — issue text → a natural developer request → `queries.jsonl`

### 2. Generate

```bash
GITHUB_TOKEN=... \
API_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY MODEL=google/gemma-4-31B-it \
  python scripts/generate_code_agent_hallucinations.py \
    --answer-source gold --ratio 0.4 --struct-ratio 0.15 \
    --batch-size 32 --out data/v2/code_agent
```

The generator reads the cached prep, takes the gold fix as the answer, injects the
mistakes above, grounds referenced definitions from GitHub, and writes
`{train,dev,test}.jsonl`. It keeps `batch-size` requests in flight continuously
and is resumable — re-run the same command into the same `--out` to continue.

| Flag / env | Description |
|---|---|
| `--answer-source` | `gold` (the PR fix verbatim, no generation) or `generated` (an LLM-written solution) |
| `--ratio` | share of instances that receive an injected hallucination |
| `--struct-ratio` | share of hallucinated samples that get a fabricated-API edit |
| `--repos` / `--exclude-repos` | comma-separated instance-id prefixes to include / exclude (e.g. `Lightning-AI`) |
| `--batch-size` / `BATCH_SIZE` | concurrent in-flight requests |
| `MAX_ANSWER_CHARS` | skip an instance whose chosen answer exceeds this (default 10000) — keeps answers trainable |
| `GITHUB_TOKEN` | raises the GitHub raw rate limit for repo grounding |
| `CONTEXT7_API_KEY` | enables Context7 third-party signature grounding |
| `API_BASE_URL` / `OPENAI_API_KEY` / `MODEL` | OpenAI-compatible endpoint |

## Output format

Each line is one `HallucinationSample`:

```json
{
  "prompt": "User request: ...\n\nFile: ...\n```python\n...\n```",
  "context": "...repository source + Referenced definitions...",
  "question": "the developer request",
  "answer": "In file ...py, replace:\n```python\n...\n```\nwith:\n```python\n...\n```",
  "labels": [
    {"start": 41, "end": 64, "label": "fabricated_api",
     "category": "fabricated_reference", "subcategory": "identifier"}
  ],
  "split": "train",
  "task_type": "code_generation",
  "dataset": "lettucedetect-code-agent",
  "language": "en",
  "context_modality": "code",
  "category": "fabricated_reference",
  "subcategory": "identifier",
  "metadata": "{\"instance_id\": \"...\", \"is_hallucinated\": true, \"hallucination_mode\": \"structural\", \"answer_style\": \"gold\"}"
}
```

Splits come straight from SWE-bench and are repository-disjoint, so test
performance measures generalization to unseen codebases. The HuggingFace build
renames `dev` to `validation`.

## Quality controls

Injection runs through the shared
[`lettucedetect.generation.injection`](../generation.md) engine, which rejects
edits that leak hint words, mislocate, over-cover the answer, or are no-ops, and
the generator skips trivial gold answers, over-long answers, and fabrications that
are not truly absent from the context. See [Phases](phases.md) for the per-step detail,
and [Provenance](provenance.md) for how the published data was audited and repaired.

## Inspect and audit

```bash
# grounding coverage + label-quality report for a generated dir
python scripts/check_context_quality.py --data data/v2/code_agent

# browse samples with hallucinated spans highlighted by category
streamlit run demo/code_hallucination_viewer.py
```

`check_context_quality.py` reports the share of samples with an ungrounded
reference (the missed-grounding signal) alongside category/format distributions,
span coverage, and answer length.

## Design decisions

- **One sample per instance.** Each SWE-bench instance produces one sample —
  clean (the gold fix) or hallucinated (gold fix + injected edits). No instance
  appears in both classes, so models can't learn to recognize the instance
  instead of the hallucination.
- **JSON-based span annotations.** Spans come from the model's structured
  `{"changes": [{"original": ..., "hallucinated": ...}]}` output applied
  deterministically, not from a character diff — clean, meaningful spans with no
  alignment noise.
- **Zero repo overlap between splits.** SWE-bench's train/dev/test are
  repository-disjoint, so test performance measures generalization to unseen
  codebases.

## Providers

The pipeline works with any OpenAI-compatible endpoint. Bulk generation is
fastest against a local vLLM server; set `BATCH_SIZE`/`--batch-size > 1` to keep
the GPU saturated with concurrent requests. The generation reported here used
`google/gemma-4-31B-it` via vLLM; hosted endpoints (e.g. Mistral) work the same
way through `API_BASE_URL` / `MODEL`.

## Layout

```
scripts/code_hallucination/      # preparation (cached, run once)
├── pipeline.py                  # prep CLI: load, fetch source, rewrite requests
├── config.py                    # paths and settings
├── swebench_loader.py           # load SWE-bench instances
├── source_fetcher.py            # fetch source, derive the gold edit
├── query_rewriter.py            # rewrite issues into developer requests
├── answer_grounding.py          # 4-tier repo + Context7 grounding of answer references
└── context7_docs.py             # Context7 client (third-party API docs)
scripts/generate_code_agent_hallucinations.py   # generation
scripts/build_hf_dataset.py                      # merge sources + publish
scripts/check_context_quality.py                 # audit: grounding coverage + label QC
demo/code_hallucination_viewer.py                # Streamlit viewer (highlighted spans)
```

```
data/code_hallucination/         # prep outputs (cached)
├── swebench_instances.json
├── source_cache/{instance_id}.json
└── queries.jsonl
data/v2/code_agent/              # generation outputs
└── {train,dev,test}.jsonl  (+ matching .failures.jsonl)
```

## End-to-end

```bash
# 1. Prepare (once)
python -m scripts.code_hallucination.pipeline --all

# 2. Generate (local vLLM, resumable)
GITHUB_TOKEN=... \
API_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY MODEL=google/gemma-4-31B-it \
  python scripts/generate_code_agent_hallucinations.py \
    --answer-source gold --ratio 0.4 --struct-ratio 0.15 --batch-size 32 \
    --out data/v2/code_agent

# 3. Publish (or --dry-run to preview counts)
python scripts/build_hf_dataset.py \
  --source data/v2/code_agent \
  --repo-id KRLabsOrg/lettucedetect-code-hallucination
```

## Training

The samples use the standard `HallucinationSample` schema, so they train with the
LettuceDetect pipeline either alone or mixed with the other sources. Two
approaches are supported.

> **Note:** the training and evaluation scripts below read a single assembled JSON
> array, whereas generation writes `{train,dev,test}.jsonl` splits. Concatenate the
> splits into one JSON list (or load the published dataset) before training.

### Token classification (ModernBERT)

A lightweight encoder that labels each answer token as supported or hallucinated.

```bash
# Code data only
python scripts/train_code_hallucination.py \
    --code-data-path data/v2/code_agent/code_agent_data.json \
    --model-name answerdotai/ModernBERT-base \
    --output-dir output/code_hallucination_detector \
    --batch-size 4 --epochs 6 --learning-rate 1e-5

# Code + RAGTruth combined (better text+code generalization)
python scripts/train_code_hallucination.py \
    --code-data-path data/v2/code_agent/code_agent_data.json \
    --ragtruth-path data/ragtruth/ragtruth_data.json \
    --model-name answerdotai/ModernBERT-base \
    --output-dir output/code_hallucination_detector \
    --batch-size 4 --epochs 6
```

The training script uses the SWE-bench splits directly — train for training, dev
for validation, test held out, with zero repository overlap between splits.

### Generative span detection (Qwen SFT)

Fine-tune a decoder LLM to read context + answer and emit a JSON list of
hallucinated spans with explanations — the inverse of the injection step.

```bash
# Requires: pip install peft
python scripts/train_generative_detector.py \
    --code-data-path data/v2/code_agent/code_agent_data.json \
    --model-name Qwen/Qwen3.5-2B \
    --output-dir output/generative_detector \
    --batch-size 2 --epochs 3 --lora-r 16
```

The model learns to output `{"hallucinated_spans": [{"text": "...", "explanation": "..."}]}`,
or `{"hallucinated_spans": []}` for clean samples. Training uses
[LoRA](https://arxiv.org/abs/2106.09685); only response tokens contribute to the loss.

### Evaluate

```bash
python scripts/evaluate_code_hallucination.py \
    --model_path output/code_hallucination_detector \
    --data_path data/v2/code_agent/code_agent_data.json \
    --evaluation_type example_level
```
