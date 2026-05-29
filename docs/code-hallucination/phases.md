# Pipeline Phases

Detailed documentation for each of the 9 pipeline phases.

## Phase 1: Load SWE-bench

**Module:** `swebench_loader.py`

Loads all SWE-bench splits from HuggingFace and tags each instance with split and Lite membership.

| Split | Instances | Repos | Source |
|-------|-----------|-------|--------|
| Train | 19,008 | 35 | `princeton-nlp/SWE-bench` train |
| Dev | 225 | 6 | `princeton-nlp/SWE-bench` dev |
| Test | 2,294 | 12 | `princeton-nlp/SWE-bench` test |
| Lite | 300 | 12 | `princeton-nlp/SWE-bench_Lite` (subset of test) |

**Key function:** `load_all_splits() -> list[dict]`

Each instance includes: `instance_id`, `repo`, `base_commit`, `patch`, `problem_statement`, `split`, `is_lite`.

**Output:** `data/code_hallucination/swebench_instances.json`

---

## Phase 2: Fetch Sources

**Module:** `source_fetcher.py`

Clones repositories and extracts source code at the base commit for each instance. Builds three answer format variants.

### Strategy

- **Default:** Clone repos as bare git repos to `data/code_hallucination/repos/`. Use `git show {commit}:{path}` for instant file access.
- **Test mode:** Use GitHub raw API (`raw.githubusercontent.com`) — slower but no cloning needed.
- **Fallback:** If cloning fails, automatically falls back to GitHub API.

### What it extracts per instance

| Field | Description |
|-------|-------------|
| `changed_files` | File paths modified by the gold patch |
| `source_files` | Original source code at base commit |
| `patch_code` | Added/changed lines from the diff (fragment format) |
| `edit_style` | "In file X, replace Y with Z" format |
| `modified_functions` | AST-extracted functions that changed (complete function format) |

### Key functions

- `extract_changed_files(patch)` — Parse unified diff for file paths (anchored regex, not `lstrip("b/")`)
- `clone_repo(repo)` — `git clone --bare` with 30min timeout
- `fetch_file_at_commit(repo_dir, commit, filepath)` — `git show` for file contents
- `apply_patch_and_get_file(repo_dir, commit, patch, filepath)` — Apply patch in temp worktree
- `extract_modified_functions(original, patched)` — AST-based function diff

**Output:** `data/code_hallucination/source_cache/{instance_id}.json`

---

## Phase 3: Rewrite Queries

**Module:** `query_rewriter.py`

Transforms raw GitHub issue `problem_statement` fields into natural developer queries using an LLM.

### Example

**Before (raw issue):**
> BUG: DataFrame.groupby with as_index=False gives wrong result when grouping by single column with duplicate name. Steps to reproduce: ...

**After (rewritten):**
> I'm getting wrong results when using DataFrame.groupby with as_index=False on a column that has a duplicate name. How do I fix this?

### Prompt strategy

The LLM is instructed to:

- Write conversational, natural language
- Extract the core technical ask
- Remove GitHub formatting, reproduction steps, tracebacks
- Keep to 1-3 sentences

### Resumability

Writes results to JSONL incrementally. On restart, skips already-processed `instance_id`s.

**Output:** `data/code_hallucination/queries.jsonl`

---

## Phase 4: Fetch Documentation

**Module:** `context7_docs.py`

Fetches library documentation via the [Context7](https://context7.com) API for **20% of instances** (configurable via `DOCS_RATIO`).

### Library detection

Maps the instance's GitHub repo to its primary library (e.g., `django/django` → `django`, `scikit-learn/scikit-learn` → `scikit-learn`). Only fetches docs for the matching library — not for random imports like `sys` or `re`.

### Why 20%?

A minority of samples include documentation context, while most don't. This creates training variety — models learn to detect hallucinations both with and without documentation support. Documentation is also passed to the hallucination injector (Phase 6), enabling SEMANTIC hallucinations that contradict documented API behavior.

Instances not selected for docs still get an entry written with empty docs (by design, not failure).

**Output:** `data/code_hallucination/documentation.jsonl`

---

## Phase 5: Assign Answer Formats

**Module:** `format_builder.py`

Each SWE-bench instance produces **one or more format entries** (sub-instances), depending on the number of modified functions. Uses LLM calls for `code_with_explanation` format.

### Sub-instance expansion

When a patch modifies multiple functions, `complete_function` and `code_with_explanation` formats are split into one sub-instance per function. This ensures that:
- Each training instance has exactly one function as its answer.
- Sibling functions (added to context in Phase 7) are compact signatures, not full bodies.
- The model learns to verify individual function-level answers rather than entire patches.

**Sub-instance ID format:** `{original_instance_id}::{function_name}`
**Example:** `DataDog__integrations-core-1013::check`

`fragment` and `edit_style` formats are **not** split — they already include all patch changes and are self-contained.

### Eligibility and sorting

Functions eligible for sub-instances must have a patched body of at least `MIN_FUNCTION_CHARS = 50` characters. They are sorted by priority:

1. Modified functions (have an original body before the patch) come first — these are more interesting for hallucination detection since the model must know what changed.
2. New functions (no original) come second.
3. Within each group, sorted by descending patched length.

At most `MAX_FUNCTIONS_PER_INSTANCE = 5` sub-instances are created per patch to avoid overweighting large refactors.

Each entry is stamped with the `split` field from the SWE-bench instance (needed by Phase 8 for per-split ratio selection).

### Format types

**Code with explanation** (weight: 0.40)
```
The issue is that `process_data` uses `dict.items()` instead of iterating
over the sorted keys, which causes non-deterministic output.

```python
def process_data(data):
    for key in sorted(data.keys()):
        yield key, data[key]
```

This ensures consistent ordering regardless of insertion order.
```
Natural AI assistant response with prose explanation + code block. Generated by wrapping one of the base code formats with an LLM-generated explanation. This is the most realistic format — it matches how Claude, Cursor, and other AI coding assistants actually respond.

**Complete function** (weight: 0.25)
```python
def validate_response(self, response):
    if response.status_code != 200:
        raise ValidationError(f"Unexpected status: {response.status_code}")
    return response.json()
```
Extracted via Python AST from the patched source. Only available when changes are inside a function (~60% of patches).

**Fragment** (weight: 0.20)
```python
if max_age is not None:
    self.cookies[key]["max-age"] = max_age
    self.cookies[key]["expires"] = http_date(time.time() + max_age)
```
Added/changed lines from the diff with surrounding context.

**Edit-style** (weight: 0.15)
```
In file django/http/response.py, replace:
    def set_cookie(self, key, value=""):
        self.cookies[key] = value
with:
    def set_cookie(self, key, value="", max_age=None):
        self.cookies[key] = value
        if max_age is not None:
            self.cookies[key]["max-age"] = max_age
```
Available for all patches where changed regions can be extracted.

### Entry schema (`formats.jsonl`)

```json
{
  "instance_id":   "DataDog__integrations-core-1013::check",
  "original_id":   "DataDog__integrations-core-1013",
  "format_type":   "complete_function",
  "answer":        "def check(self, instance):\n    ...",
  "function_name": "check",
  "split":         "train"
}
```

For `fragment` / `edit_style` entries, `instance_id == original_id` and `function_name` is absent.

**Output:** `data/code_hallucination/formats.jsonl`

---

## Phase 6: Inject Hallucinations

**Module:** `hallucination_injector.py`

Uses an LLM to inject realistic hallucinations into selected instances (determined by Phase 8). Returns structured JSON with span annotations.

!!! note "Shared injection engine"
    Edit application, span location, and label validation are handled by the shared [`lettucedetect.generation.injection`](../generation.md) engine. This phase supplies its code-specific injection prompt and native labels (`structural`/`behavioral`/`semantic`); the engine produces the exact span annotations.

### Hallucination types (round-robin)

| Type | Description | Example |
|------|-------------|---------|
| **Structural** | Non-existent APIs, wrong methods, invented parameters | `response.json_decode()` instead of `response.json()` |
| **Behavioral** | Wrong values, logic errors, off-by-one, swapped conditions | `if status >= 200` instead of `if status == 200` |
| **Semantic** | Code that looks right but does something subtly different | Sorting ascending instead of descending |

### JSON-based span extraction

The LLM returns structured output:

```json
{
  "hallucinated_code": "def fix(self):\n    self.data = response.json_decode()\n    ...",
  "changes": [
    {
      "original": "response.json()",
      "hallucinated": "response.json_decode()",
      "explanation": "json_decode() is not a valid method on Response objects"
    }
  ]
}
```

Spans are found by string-matching each `change["hallucinated"]` in `hallucinated_code`. This produces clean, meaningful spans (minimum 15 chars) with zero noise.

For answers containing both code and prose (code_with_explanation format), the injector places errors in both parts — e.g., wrong API in code + misleading description in text.

### Quality controls

- Each span must be 20-150 characters (enforced by prompt)
- Total hallucinated coverage must be < 40% of the answer (enforced by prompt)
- `_validate_labels()` rejects samples with coverage > 60% or spans < 15 chars
- Failed validation triggers up to 3 retries before skipping
- No comment data leaks (prompt explicitly forbids `# wrong`, `# error`, etc.)

### Quality metrics (from 100-sample test runs)

| Metric | Value |
|--------|-------|
| Noise-only samples | 0% |
| Min span length | 15 chars |
| Avg span length | 71 chars |
| Avg spans per sample | 2.8 |
| Coverage range | 2.8-43% |
| Mean coverage | 19.5% |

**Output:** `data/code_hallucination/hallucinated_samples.jsonl`

---

## Phase 7: Assemble Samples

**Module:** `sample_assembler.py`

Iterates over **format entries** (which may include sub-instance IDs) and builds the final `HallucinationSample` format. Metadata such as repo, split, and `is_lite` is resolved via `original_id` from a lookup over the SWE-bench instances.

### Sibling function context

For `complete_function` and `code_with_explanation` sub-instances, the answer contains one function body. Other functions from the same patch that are **called by the answer function** are added to the `Referenced definitions` section of the prompt — as **signatures only** (not full bodies):

```python
def _get_instance_params(self, instance):
    ...

def _perform_service_check(self, url, ssl_validation, auth):
    ...
```

This ensures every function call in a clean answer is evidenced in the context (either imported, defined in the source files, or present as a signature), making clean samples self-consistent and distinguishable from hallucinated ones.

"Called" is determined by `\b{name}\s*\(` regex match against the answer body — catches both bare calls and method calls (`self.method_name()`).

### Prompt construction

```
File: path/to/file.py
```python
<source code at base commit>
```

Referenced definitions:
```python
def _get_instance_params(self, instance):
    ...

def _perform_service_check(self, url, ssl_validation, auth):
    ...
```

Documentation for django:
<Context7 docs if available>

User request: <rewritten query>
```

### Sample types

**Clean samples** (~60%): Gold patch answer, empty labels, from format entries NOT selected for injection.

**Hallucinated samples** (~40%): LLM-modified answer with character-level span annotations.

### Outputs

- `data/code_hallucination/code_hallucination_data.json` — List of samples
- `data/code_hallucination/code_hallucination_metadata.json` — Metadata (instance_id, original_id, repo, format_type, hallucination_type, injector_model, is_hallucinated)

---

## Phase 8: Select Hallucination Targets

**Module:** `splitter.py`

Selects which **format entries** (sub-instances) receive hallucination injection. Operates on the output of Phase 5, not on raw SWE-bench instances.

Uses `select_format_targets(format_entries, ratio=0.40)` which:
- Groups entries by their `split` field.
- Applies the hallucination ratio uniformly within each split to maintain consistent class distribution.
- Returns a set of format entry `instance_id`s (may include sub-instance IDs like `orig::func`).

```
Train: ~40% of train format entries get hallucinated
Dev:   ~40% of dev format entries get hallucinated
Test:  ~40% of test format entries get hallucinated
```

!!! note
    Phase 8 runs **before** Phase 6 in the pipeline (target selection must happen before injection).

**Output:** Set of format entry `instance_id`s (used in-memory by Phase 6 and Phase 7)

---

## Phase 9: Validate

**Module:** `validator.py`

Runs automated quality checks and generates a report.

### Checks performed

| Check | Description |
|-------|-------------|
| **Span validity** | No negative offsets, empty spans, or out-of-bounds |
| **Span coverage** | Distribution of hallucinated text ratio; flags <2% or >80% |
| **Distributions** | Format type, hallucination type, injector model, repo, split |
| **Near-duplicates** | Jaccard similarity >0.95 on sampled answer pairs |
| **AST parseability** | For complete_function format, checks if answer parses as valid Python |
| **Length statistics** | Prompt and answer character length ranges |

**Output:** `data/code_hallucination/validation_report.txt`
