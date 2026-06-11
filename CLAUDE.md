# CALUDE.md — LettuceDetect

This file provides guidance for AI coding agents working in this repository.

## What the Project Does

LettuceDetect is a hallucination detection library for RAG (Retrieval-Augmented Generation) pipelines. It identifies unsupported text spans in an LLM answer by comparing the answer against retrieved context. Outputs are either token-level predictions or character-offset spans.

## Repository Layout

```
lettucedetect/           # Core library
  detectors/             # Detection implementations + factory
  models/                # Inference, training, evaluation facades
  datasets/              # HallucinationSample, HallucinationDataset
  preprocess/            # Dataset preprocessing (RAGTruth, RAGBench, etc.)
  generation/            # Synthetic hallucination generation pipeline
  integrations/          # LangChain and Elysia integrations
  prompts/               # Language-specific prompt templates + few-shot examples
  ragfactchecker.py      # RAGFactChecker wrapper
lettucedetect_api/       # FastAPI server + sync/async client
scripts/                 # Training, evaluation, data generation, upload utilities
tests/                   # Pytest test suite
notebooks/               # Jupyter notebooks
docs/                    # MkDocs documentation
demo/                    # Streamlit demo app
```

## Installation

```bash
pip install -e ".[dev]"       # development (adds pytest, ruff)
pip install -e ".[api]"       # with FastAPI server + client
pip install -e ".[docs]"      # with MkDocs documentation tools
```

Requires Python 3.11+.

## Running Tests

```bash
pytest tests/test_inference_pytest.py -v
```

Test files must follow the naming convention `test_*_pytest.py`. API tests live in `lettucedetect_api/test_server.py` and `lettucedetect_api/test_client.py`.

Some tests download model weights from HuggingFace. Skip those locally with:

```bash
pytest tests/test_inference_pytest.py -v -k "not TestAnswerStartToken"
```

## Code Style

The project uses **Ruff** for linting and formatting with a line length of 100.

```bash
ruff format lettucedetect/
ruff check lettucedetect/
```

Key conventions:
- Use `from __future__ import annotations` at the top of every module.
- Modern type hints: `list[str]`, `dict[str, Any]`, `str | None` (not `Optional`, `List`, `Dict`).
- Docstrings in Sphinx `:param:`/`:return:` style.
- Use `logging` not `print()`.
- Use `pathlib.Path` not `os.path`.

## Architecture Overview

### Detector Hierarchy

`BaseDetector` (`lettucedetect/detectors/base.py`) defines the interface:

```python
predict(context: list[str], answer: str, question: str | None, output_format: str) -> list
predict_prompt(prompt: str, answer: str, output_format: str) -> list
predict_prompt_batch(prompts: list[str], answers: list[str], output_format: str) -> list
```

Three concrete implementations:
- **TransformerDetector** — fine-tuned encoder (ModernBERT / EuroBERT). Handles long inputs via context chunking, aggregates per-token scores with `max()` across chunks.
- **LLMDetector** — OpenAI API-based (default `gpt-4.1-mini`). Supports zero-shot and few-shot prompting using language-specific templates in `lettucedetect/prompts/`. Uses `ThreadPoolExecutor` for batch concurrency.
- **RAGFactCheckerDetector** — Triplet-based detection via the `rag-fact-checker` library. Adapts triplet results into span/token format.

Use the factory to instantiate:

```python
from lettucedetect.detectors.factory import make_detector
detector = make_detector("transformer", model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1")
detector = make_detector("llm", openai_api_key="...", lang="en")
detector = make_detector("rag_fact_checker", openai_api_key="...")
```

The public `HallucinationDetector` façade (`lettucedetect/models/inference.py`) wraps the factory.

### Output Formats

`output_format="tokens"`:
```python
[{"token": "Paris", "pred": 1, "prob": 0.93}, ...]
```

`output_format="spans"`:
```python
[{"start": 12, "end": 17, "text": "Paris", "confidence": 0.93}, ...]
```

Span indices are character offsets into `answer`.

### Data Model

`HallucinationSample` (in `lettucedetect/datasets/hallucination_dataset.py`):
- `context: list[str]` — retrieved passages
- `question: str | None`
- `answer: str` — LLM output to evaluate
- `labels: list[dict]` — hallucination spans: `{"start": int, "end": int, "label": str, "category": str, "subcategory": str}`
- `split`, `task_type`, `dataset`, `language`, `category`, `subcategory` — metadata

### Generation Pipeline

`lettucedetect/generation/` orchestrates synthetic dataset creation:
1. `questions.py` — derive questions from documents
2. `answers.py` — generate grounded correct answers
3. `injection.py` — corrupt correct answers with taxonomy-aware hallucinations
4. `classify.py` — classify hallucination spans against the unified taxonomy
5. `runner.py` — batch orchestration with resumability
6. `assembly.py` — compose pipeline stages

`HallucinationGenerator` (`lettucedetect/models/generation.py`) is the public façade around `RAGFactChecker` for generation tasks.

### Web API

The FastAPI server (`lettucedetect_api/server.py`) exposes two endpoints:
- `POST /v1/lettucedetect/token` — token-level detection
- `POST /v1/lettucedetect/spans` — span-level detection

Start the server:
```bash
python scripts/start_api.py dev    # development (auto-reload)
python scripts/start_api.py prod   # production (gunicorn)
```

Clients: `lettucedetect_api.client.LettuceClient` (sync) and `LettuceClientAsync` (async).

## Supported Models (HuggingFace Hub)

| Model family | Variants | Languages |
|---|---|---|
| ModernBERT (English) | `lettucedect-base-modernbert-en-v1`, `lettucedect-large-modernbert-en-v1` | en |
| EuroBERT (multilingual) | `lettucedect-210m-eurobert-{lang}-v1`, `lettucedect-610m-eurobert-{lang}-v1` | de, fr, es, it, pl, cn, hu |
| TinyLettuce Ettin | `tinylettuce-ettin-{17m,32m,68m}-*` | en |

All models are under the `KRLabsOrg` HuggingFace organisation.

## Prompts and Multilingual Support

Language-specific assets live in `lettucedetect/prompts/`:
- `qa_prompt_{lang}.txt` / `summary_prompt_{lang}.txt` — system prompts for LLMDetector
- `examples_{lang}.json` — few-shot examples

`PromptUtils` (`lettucedetect/detectors/prompt_utils.py`) loads these and formats context + question into the prompt string. Supported language codes: `en`, `de`, `fr`, `es`, `it`, `pl`, `cn`, `hu`.

## Training & Evaluation

```bash
python scripts/train.py            # fine-tune on RAGTruth
python scripts/evaluate.py         # evaluate at example/token/span level
python scripts/evaluate_llm.py     # LLM detector baseline
python scripts/ragas_baseline.py   # RAGAS baseline comparison
```

Dataset preprocessing scripts are in `scripts/preprocess_*.py`.

## Multi-Agent Team Setup

This section defines specialized agent roles for working on LettuceDetect efficiently. Each agent has a focused scope, clear inputs/outputs, and explicit handoff points.

---

### Architect Agent

**Responsibility**: Design, decompose, and plan — never writes production code directly.

**Scope**:
- Reads the codebase and existing docs to understand current state.
- Produces a written plan: which files change, which new files are needed, which interfaces are affected, and in what order work should proceed.
- Flags coordination points (see below) that require serialization between Coder agents.
- Decides which parts of a feature are safe to parallelize.

**Inputs**: A feature request or bug description in natural language.

**Output**: A structured plan document (or in-chat plan) containing:
- List of files to create / modify / delete
- New or changed public interfaces with their signatures
- Dependency order (what must be done before what)
- Test surface (what the Test Writer should cover)
- Risks and open questions

**Coordination points in this repo** (always call out in the plan):
- `lettucedetect/__init__.py` — public API surface
- `lettucedetect/datasets/hallucination_dataset.py` — shared data contract
- `lettucedetect/detectors/factory.py` — detector registration
- `pyproject.toml` — dependency changes
- Taxonomy fields (`category`/`subcategory`) in `generation/injection.py` and `generation/classify.py`

---

### Coder Agent

**Responsibility**: Implement exactly what the Architect planned — no scope creep.

**Scope**:
- Implements new or modified detector, generation, API, or integration code.
- Works in one module boundary at a time (e.g., only `lettucedetect/detectors/` or only `lettucedetect_api/`).
- Does not write tests (that is the Test Writer's job).
- Does not modify `lettucedetect/__init__.py` unless the plan explicitly says to.

**Inputs**: Architect's plan, relevant file paths.

**Output**: Changed source files that pass `ruff format` and `ruff check` cleanly.

**Checklist before handing off**:
```bash
ruff format lettucedetect/ lettucedetect_api/
ruff check  lettucedetect/ lettucedetect_api/
python -c "import lettucedetect"      # smoke-test imports
```

**Parallelization**: Multiple Coder agents can run simultaneously if they own disjoint module subtrees (e.g., one on `detectors/`, one on `generation/`). They must not both touch the coordination points above.

---

### Test Writer Agent

**Responsibility**: Write tests for code that was just implemented or modified.

**Scope**:
- Works exclusively in `tests/` and `lettucedetect_api/test_*.py`.
- Does not modify library code.
- Covers: happy path, boundary conditions, and any invariant called out by the Architect.
- Tests that require model downloads must be skippable (mark with a condition or naming convention so `-k "not TestAnswerStartToken"` remains functional).

**Inputs**: Architect's plan (test surface section) + the Coder's diff.

**Output**: New or updated test files that pass:
```bash
pytest tests/test_inference_pytest.py -v -k "not TestAnswerStartToken"
```

**Naming rule**: Test files must match `test_*_pytest.py`. Test classes and functions must be descriptive (`TestTransformerDetectorChunking`, `test_span_offsets_are_character_level`).

---

### Reviewer Agent

**Responsibility**: Adversarial review of the Coder's output before merge.

**Scope**:
- Reads the diff produced by the Coder.
- Checks for: correctness bugs, unintended side-effects on shared interfaces, missing error handling at external boundaries (OpenAI API calls, HuggingFace downloads), and regressions to the public API contract.
- Does NOT check style (that is Ruff's job).
- Produces a short findings list: `[BLOCK]` for must-fix issues, `[SUGGEST]` for optional improvements.

**Inputs**: Coder's diff + Architect's plan.

**Output**: Findings list. If no `[BLOCK]` items, the change is approved for merge.

**Areas requiring extra scrutiny in this repo**:
- `TransformerDetector._predict_chunked()` — chunk aggregation uses `max()` per token; verify this is preserved.
- `LLMDetector` — structured-output schema must stay in sync with the JSON parsing logic.
- Any change to `HallucinationSample` fields — serialization in `preprocess/` and the API `DetectionRequest` must stay consistent.
- Generation taxonomy (`category`/`subcategory`) — changes must be atomic across `injection.py`, `classify.py`, and downstream consumers.

---

### Prompt / Multilingual Agent

**Responsibility**: Owns all content in `lettucedetect/prompts/` and multilingual correctness.

**Scope**:
- Adds or updates prompt templates (`qa_prompt_{lang}.txt`, `summary_prompt_{lang}.txt`).
- Adds or updates few-shot examples (`examples_{lang}.json`).
- Verifies that `PromptUtils` in `lettucedetect/detectors/prompt_utils.py` handles new language codes.
- Does not modify detector logic.

**Inputs**: Language code, task type, target behaviour description.

**Output**: Updated prompt files + a brief note on any edge case in the new language.

---

### Agent Handoff Protocol

```
Architect  →  writes plan
                ↓
Coder(s)   →  implement (parallel if disjoint scope)
                ↓
Test Writer →  write tests against Coder's output
                ↓
Reviewer   →  review diff; return to Coder if [BLOCK] items exist
                ↓
Merge
```

When multiple Coder agents run in parallel, the Reviewer waits for all of them to finish before reviewing the combined diff.

---

## Key Conventions for Agents

- **Do not add features beyond the task scope.** No speculative abstractions.
- **Do not add comments** unless the `why` is truly non-obvious.
- **No backwards-compatibility shims** — rename or delete unused symbols outright.
- **Error handling only at system boundaries** (user input, external APIs). Do not add defensive checks for internal invariants already guaranteed by the type system or framework.
- **Prefer editing existing files** over creating new ones.
- **Chunk-aggregation logic** in `TransformerDetector._predict_chunked()` uses `max()` across chunks — preserve this semantics when modifying chunking behaviour.
- **Taxonomy categories** (`category`/`subcategory` fields) are defined in the generation pipeline; keep them consistent across `injection.py`, `classify.py`, and data model usage.
- **Public API surface** is declared in `lettucedetect/__init__.py`. Any symbol added there should also be covered by a test.
- When touching the LLM detector, run `ruff check` — the async/thread usage and structured-output schema have subtle constraints.
