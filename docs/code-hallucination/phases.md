# Pipeline Steps

The pipeline has two parts: cached **preparation** (`scripts.code_hallucination.pipeline`)
and **generation** (`scripts/generate_code_agent_hallucinations.py`).

## Preparation

Run once; outputs are reused by every generation run.

```bash
python -m scripts.code_hallucination.pipeline --all      # or --phase 1 2 3
```

### Phase 1 — Load SWE-bench

**Module:** `swebench_loader.py`

Loads all SWE-bench splits and tags each instance with its split. Splits are
repository-disjoint, so train/dev/test share no repositories.

| Split | Instances | Source |
|---|---|---|
| train | 19,008 | `princeton-nlp/SWE-bench` train |
| dev | 225 | `princeton-nlp/SWE-bench` dev |
| test | 2,294 | `princeton-nlp/SWE-bench` test |

Each instance carries `instance_id`, `repo`, `base_commit`, `patch`,
`problem_statement`, and `split`.

**Output:** `data/code_hallucination/swebench_instances.json`

### Phase 2 — Fetch source

**Module:** `source_fetcher.py`

Fetches the patch-touched source files at the base commit (bare clone with a
GitHub-raw fallback) and derives the gold answer.

| Field | Description |
|---|---|
| `changed_files` | files modified by the gold patch |
| `source_files` | source at the base commit, truncated around the patch |
| `patch_code` | added/changed lines from the diff |
| `edit_style` | the gold fix as `In file X, replace Y with Z` — the answer used in `gold` mode |
| `modified_functions` | AST-extracted functions that changed |

**Output:** `data/code_hallucination/source_cache/{instance_id}.json`

### Phase 3 — Rewrite requests

**Module:** `query_rewriter.py`

Turns each raw GitHub `problem_statement` into a short, natural developer request
(conversational, core ask only, no tracebacks or reproduction steps). Resumable —
re-running skips already-processed `instance_id`s.

**Output:** `data/code_hallucination/queries.jsonl`

## Generation

**Script:** `scripts/generate_code_agent_hallucinations.py`

For each instance it builds the context, takes the answer, optionally injects a
hallucination, grounds referenced definitions, and writes a `HallucinationSample`.

### Answer

With `--answer-source gold` (default) the answer is the gold fix verbatim — no
model call, so clean samples cost nothing — rendered per-instance in one of three
styles for variety: `function` (the largest patched function that fits the length
cap), `fragment` (the patch hunk), or `edit` (`In file X, replace Y with Z`). With
`--answer-source generated` a model writes a coherent solution instead. Answers
that are trivial (e.g. a bare version bump) or longer than `MAX_ANSWER_CHARS`
(default 10000, to stay trainable) are skipped.

### Injection

A share (`--ratio`) of instances get an injected hallucination. Most are
**intent** edits — `wrong_implementation` or `unrequested_change`, judged against
the developer request; a `--struct-ratio` share get a **structural** edit —
`fabricated_api`, a reference to a member that does not exist on a pre-existing
object. The model returns replacement edits as JSON:

```json
{"changes": [
  {"original": "response.json()",
   "hallucinated": "response.json_decode()",
   "hallucination_type": "fabricated_api",
   "explanation": "Response objects have no json_decode method; it is json()"}
]}
```

The shared [`injection`](../generation.md) engine applies each edit, locates the
exact character span, attaches the native label, and maps it to the unified
taxonomy via the `code_agent` table. Native labels: `wrong_implementation`,
`unrequested_change`, `fabricated_api`.

### Grounding

After the answer is fixed, symbols it references that are absent from the context
are resolved to their real definitions at the base commit, in four tiers
(`answer_grounding.py`):

1. the patch's `modified_functions`;
2. the full **changed files** (fetched from GitHub raw);
3. modules the **answer imports** — resolved to a repo path anywhere in the tree
   (e.g. `from pvlib._deprecation import deprecated` → `pvlib/_deprecation.py`);
4. modules the **changed files import** — base-class mixins and sibling modules —
   searched for the referenced name, which grounds cross-module `self.method` calls.

These append a `Referenced definitions` block. Separately, for **structural**
samples, the real third-party API the fabrication replaced (taken from the edit's
`original`, e.g. `torch.cuda.set_device`) is looked up on **Context7** and added as
a compact `Library signatures` block. `GITHUB_TOKEN` raises the repo-fetch rate
limit; `CONTEXT7_API_KEY` enables the third-party lookups. File fetches are cached
across samples, so a same-repo batch reuses modules.

### Quality gates

A sample is rejected (logged to `{split}.failures.jsonl`) when an edit:

| Reason | Meaning |
|---|---|
| `trivial_gold_answer` | the gold answer is too small to inject into |
| `answer_too_long` | the chosen answer exceeds `MAX_ANSWER_CHARS` (would crowd out context) |
| `substring_not_unique_or_too_short` | the edit's `original` cannot be uniquely located |
| `leaky_label_text` | the edit text leaks a hint word |
| `coverage_too_high` | labels cover too much of the answer (length-aware cap) |
| `span_too_long` | a single span is too long |
| `fabrication_not_absent` | a structural edit's "fabricated" name actually exists in context |

Coverage caps are length-aware: a single genuine edit may be a large fraction of a
short answer, so short answers get a lenient cap and long answers a stricter one.

### Output

`{train,dev,test}.jsonl` under `--out`, plus matching `.failures.jsonl`. The runner
keeps `--batch-size` requests in flight continuously and skips already-written
keys, so a crashed run resumes by re-invoking the same command.

## Build the HuggingFace dataset

`scripts/build_hf_dataset.py` merges one or more `data/v2` sources into a
`DatasetDict`, renames `dev` to `validation`, serializes `metadata` to a JSON
string, and pushes:

```bash
python scripts/build_hf_dataset.py \
  --source data/v2/code_agent \
  --repo-id KRLabsOrg/lettucedetect-code-hallucination
```

Add `--dry-run` to print the merged per-split, per-source counts without pushing.
