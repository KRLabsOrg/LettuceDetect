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

It has two modes. **Targeted** (`inject_taxonomy`) forces one chosen
`(category, subcategory)` — used for code and tool output. **Menu**
(`inject_menu`) hands the model a source-specific prompt that lists several
hallucination types and lets it pick the 1–3 that fit the passage, labelling
each edit with its own type (mapped to the taxonomy per source) — used for
academic papers and other markdown, where a forced subtype often does not fit.

Sync and async variants exist for both (`inject_taxonomy`, `inject_menu`, and
their `_async` twins).

### `classify.py` — label an existing, untyped span
`classify_span(context, answer, span_text)` does the inverse of injection: given a
span an annotator already marked as unsupported, it assigns a unified
`(category, subcategory)` with an LLM. It is for sources whose spans ship without
a native type, so the mechanical [`map_label`](taxonomy.md) cannot be used — most
notably **PsiloQA**, whose hallucinations are *natural* (produced by real LLMs,
not injected). It never edits text or invents spans; it only classifies, so
`supported` is not a valid output. Sync and async variants are provided.

### `runner.py` — batched, resumable orchestration
`run_batched` runs a per-item async processor over the work set with:

- **Async batching** (`asyncio.gather` over a batch size) for local vLLM throughput.
- **Resumability** — already-completed keys are skipped on restart; output is
  appended and flushed per batch, so a crash never loses finished work.
- **Failure logging** — each rejected item is written to a failures file with its
  reason; re-running retries anything not yet in the output.

## Composing a source adapter

Each source is a thin adapter that wires together only the primitives it needs.
The five built sources:

| Source (`dataset`) | modality | question | answer | injection prompt |
|---|---|---|---|---|
| `lettucedetect-code` (SWE-bench) | code | — (issue) | format-builder over the patch | code (targeted) |
| `lettucedetect-tool-output` (squeez) | tool_output | — (given) | grounded | tool-output (targeted) |
| `lettucedetect-acl` (acl-verbatim) | markdown | — (given) | grounded | paper (menu) |
| `lettucedetect-readme` (GitHub) | markdown | generated | grounded | generic factual (menu) |
| `lettucedetect-wikipedia` (open-wikipedia) | markdown | generated | grounded | generic factual (menu) |

Document sources (README, Wikipedia) share `doc_source.py` — chunk by heading,
generate a question per chunk, answer, inject — and differ only in corpus and
question-type subset. ACL groups retrieved chunks per question and uses the
paper-specific prompt. Each adapter supplies `(context, clean_answer, modality)`
and a category/subtype distribution; the shared modules handle batching,
resumability, and failure logging.

## Public prose sources (separate collection)

Two existing public RAG datasets are folded in without any generation, as a
separate prose collection, to complement the synthetic structured-context data:

| Source (`dataset`) | modality | spans | taxonomy assignment |
|---|---|---|---|
| `ragtruth` (RAGTruth) | prose | native typed | mechanical [`map_label`](taxonomy.md) via `apply_taxonomy.py --source ragtruth` |
| `psiloqa` (PsiloQA) | prose | untyped, **natural** | LLM `classify_span` via `scripts/classify_psiloqa_spans.py` |

RAGTruth's native labels map deterministically. PsiloQA's hallucinations are
produced by real LLMs (not injected) and annotated only as binary char spans, so
each span is labelled by the `classify.py` primitive. All 14 PsiloQA languages
and its original train/validation/test splits are preserved. These provide a
naturally-occurring counterpart to the injected spans — useful for checking that
detectors generalize beyond the corruption process.
