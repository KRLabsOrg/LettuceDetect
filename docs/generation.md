# Data Generation Pipeline

LettuceDetect builds hallucination-detection datasets from many grounded sources
(code, tool output, markdown documents, paper chunks) using one shared set of
composable primitives. Every source maps into the same unified
[taxonomy](taxonomy.md) and the same `HallucinationSample` schema, so a single
detector can be trained across modalities.

The primitives live in `lettucedetect/generation/`.

## Generation vs. injection

*Generating* a hallucinated answer and *injecting* a hallucination into a correct
answer are different operations:

| | Generator | Injector |
|---|---|---|
| Module | `lettucedetect.models.generation.HallucinationGenerator` (RAGFactChecker) | `lettucedetect.generation.injection` |
| Operation | **synthesizes** a hallucinated answer (can work from context alone) | **corrupts a known-correct answer** into a hallucinated one |
| Spans | recovered by diff (approximate) | exact, by construction |
| Use | the TinyLettuce synthetic-data recipe | the multi-source dataset collection |

The dataset collection uses the **injector**, because exact character-level spans
are the basis of token-level detection.

## The primitives

### `questions.py` — derive a question from raw data
For sources that are just documents (markdown READMEs, wiki pages), generate a
realistic user/developer question the document can answer.

### `answers.py` — generate a correct, grounded answer
`generate_grounded_answer(question, evidence)` produces an answer that is correct
*by construction* — grounded strictly in the supplied evidence. This is the step
that lets the injector then corrupt a known-good answer. Sync and async variants
are provided (`generate_grounded_answer_async` for batched throughput).

### `injection.py` — inject a taxonomy hallucination with exact spans
`inject_taxonomy(context, clean_answer, category, subcategory, modality)` requests
small localized replacement edits from the model, applies them deterministically,
and validates the result. It is:

- **Universal** — modality-aware (`code`, `tool_output`, `markdown`, `prose`); the
  taxonomy is the same across all of them.
- **Subtype-driven** — injects a specific `(category, subcategory)` from the
  unified taxonomy (e.g. `contradiction/numerical`, `fabricated_reference/identifier`).
- **Span-exact** — labels are derived from the applied edits, not from diffing.
- **Validated** — coverage caps, minimum span length, leakage detection, and
  (for mixed prose+code answers) in-fence enforcement.

Sync and async variants exist (`inject`, `inject_async`, `inject_taxonomy`,
`inject_taxonomy_async`).

### `runner.py` — batched, resumable orchestration
`run_batched` runs a per-item async processor over the work set with:

- **Async batching** (`asyncio.gather` over a batch size) for local vLLM throughput.
- **Resumability** — already-completed keys are skipped on restart; output is
  appended and flushed per batch, so a crash never loses finished work.
- **Failure logging** — each rejected item is written to a failures file with its
  reason; re-running retries anything not yet in the output.

## Composing a source adapter

Each source is a thin adapter that wires together only the primitives it needs:

| Source | questions | answers | injection |
|---|---|---|---|
| Code (SWE-bench) | — (gold patch) | specialized format-builder | ✓ |
| Tool output (squeez) | — (query given) | ✓ | ✓ |
| Markdown (READMEs) | ✓ | ✓ | ✓ |
| Paper chunks (ACL) | — (question given) | ✓ (or gold span) | ✓ |

The adapter supplies `(context, clean_answer, modality)` and a category/subtype
distribution; the shared modules handle the rest, so orchestration features
(batching, resumability, failure logging) are written once and reused.
