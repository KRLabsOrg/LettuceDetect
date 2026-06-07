# Unified Hallucination Taxonomy

A single taxonomy that every data source maps into, so that prose (RAGTruth,
FAVA), code (SWE-bench), and markdown hallucinations all share one label space.
This is what lets a single detector be trained across modalities and lets users
redefine the label set at inference time.

Canonical implementation: [`lettucedetect/datasets/taxonomy.py`](../lettucedetect/datasets/taxonomy.py).
Applied to data via [`lettucedetect/preprocess/apply_taxonomy.py`](../lettucedetect/preprocess/apply_taxonomy.py).

## Why a unified taxonomy

Every prior taxonomy cuts the same conceptual space slightly differently — FAVA
(6 types), RAGTruth (4 types), our prose generator (6 types), our code pipeline
(3 types), HalluVerse25 (3 levels). None of them unify in a usable way. Training
a cross-modality detector requires one label space that all of these map into
**without regenerating any data**.

The taxonomy is built on two orthogonal axes:

- **Axis 1 — relationship to context.** Does the span *conflict with*, *add
  beyond*, or *fabricate a reference into* the context? This becomes the
  top-level category.
- **Axis 2 — surface element affected.** What kind of thing is wrong — a number,
  a date, a name, an identifier? This becomes the (open-set, user-extensible)
  subcategory.

## Top-level categories

Mutually exclusive per span.

| Category | Definition |
|---|---|
| `supported` | Span is entailed by the context. The non-hallucinated default. |
| `contradiction` | Span asserts X; context asserts Y; Y ≠ X. A direct, locally checkable conflict. |
| `unsupported_addition` | Span asserts X; context neither states X nor anything contradicting it. Plausible but not derivable. |
| `fabricated_reference` | Span references a named structural element (entity, section, function, identifier, table, equation) that does not appear in the context. |

`omission` (a span that is technically correct but materially incomplete) is
treated as a **document-level** binary flag, not a span class — it cannot be
localized to a span of text that is present.

## Subcategories

Optional attributes of an already-classified span. Open-set: callers may extend
them for a vertical (legal, medical, finance) without retraining.

| Category | Subcategories |
|---|---|
| `contradiction` | `numerical`, `temporal`, `entity`, `relational`, `value` |
| `unsupported_addition` | `claim`, `elaboration`, `subjective`, `behavior` |
| `fabricated_reference` | `entity`, `section`, `identifier`, `attribute` |

## Source-label mapping

Every native label from every source maps mechanically into `(category,
subcategory)`. Nothing has to be regenerated; the synthetic and code data map
deterministically, and RAGTruth uses a light context-aware heuristic.

### Code (SWE-bench-derived)

| Native label | → Category | → Subcategory |
|---|---|---|
| `structural` (fabricated function/identifier name) | `fabricated_reference` | `identifier` |
| `behavioral` (wrong arg/value/logic) | `contradiction` | `value` |
| `semantic` (solves the wrong problem) | `unsupported_addition` | `behavior` |

The original native label is preserved in each span's `label` field for
backwards compatibility; `category`/`subcategory` are added alongside it.

### RAGTruth (prose)

| Native label | → Category | → Subcategory |
|---|---|---|
| Evident Conflict | `contradiction` | — |
| Subtle Conflict | `contradiction` | — |
| Evident Baseless Info | `unsupported_addition` | `claim` |
| Subtle Baseless Info | `unsupported_addition` | `elaboration` |

RAGTruth uses a context-aware refinement: Baseless Info whose span contains a
proper noun absent from the context is reclassified as `fabricated_reference` /
`entity` (see `ragtruth_map_with_context`).

### FAVA (prose)

| Native label | → Category | → Subcategory |
|---|---|---|
| Entity | `contradiction` | `entity` |
| Relation | `contradiction` | `relational` |
| Contradictory | `contradiction` | — |
| Invented | `fabricated_reference` | `entity` |
| Subjective | `unsupported_addition` | `subjective` |
| Unverifiable | `unsupported_addition` | `claim` |

### LD prose generator (`rag-fact-checker`)

| Native label | → Category | → Subcategory |
|---|---|---|
| FACTUAL | `contradiction` | `entity` |
| TEMPORAL | `contradiction` | `temporal` |
| NUMERICAL | `contradiction` | `numerical` |
| RELATIONAL | `contradiction` | `relational` |
| CONTEXTUAL | `unsupported_addition` | `claim` |
| OMISSION | `omission` | (document-level) |
| FABRICATED_ENTITY | `fabricated_reference` | `entity` |
| SUBJECTIVE | `unsupported_addition` | `subjective` |
| UNVERIFIABLE | `unsupported_addition` | `claim` |

### Markdown (planned)

| Native label | → Category | → Subcategory |
|---|---|---|
| contradicted_number | `contradiction` | `numerical` |
| contradicted_date | `contradiction` | `temporal` |
| contradicted_entity | `contradiction` | `entity` |
| contradicted_table_cell | `contradiction` | `value` |
| extra_claim | `unsupported_addition` | `claim` |
| fabricated_section_ref | `fabricated_reference` | `section` |
| fabricated_citation | `fabricated_reference` | `entity` |
| fabricated_equation_ref | `fabricated_reference` | `section` |

## Applying the taxonomy

`apply_taxonomy.py` enriches an already-preprocessed dataset with `category`,
`subcategory`, `context_modality`, and provenance `metadata`, writing one JSONL
file per split. The sample-level category is a majority vote over its spans'
categories.

```bash
python -m lettucedetect.preprocess.apply_taxonomy \
    --source code \
    --data_path data/code_hallucination/code_hallucination_data.json \
    --metadata_path data/code_hallucination/code_hallucination_metadata.json \
    --output_dir data/v2/code_hallucination
```

Each output sample carries both the native label (on the span) and the unified
category/subcategory (on both the span and the sample):

```json
{
  "labels": [
    {"start": 18, "end": 25, "label": "structural",
     "category": "fabricated_reference", "subcategory": "identifier"}
  ],
  "context_modality": "code",
  "category": "fabricated_reference",
  "subcategory": "identifier",
  "metadata": {"instance_id": "...", "repo": "...", "format_type": "...",
               "is_hallucinated": true, "injector_model": "..."}
}
```

## Why this matters

A single, source-agnostic label space is what lets one detector be trained
across modalities (prose, code, markdown) instead of one model per source. It
also keeps the door open to typed, span-level output — telling a user not just
*that* a span is unsupported but *how* (contradiction vs. unsupported addition
vs. fabricated reference) — which is the differentiator over scalar
faithfulness scores. Every data source mapping in cleanly is the precondition
for both.
