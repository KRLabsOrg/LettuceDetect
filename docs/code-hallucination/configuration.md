# Configuration

All pipeline configuration is centralized in `scripts/code_hallucination/config.py`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (none) | API key for the LLM provider |
| `API_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible API endpoint |
| `MODEL` | `moonshotai/kimi-k2-instruct-0905` | Model name |
| `BATCH_SIZE` | `1` | Concurrent requests. Set >1 for local vLLM to saturate GPU |
| `GITHUB_TOKEN` | (none) | Raises the GitHub-raw rate limit for repo grounding |
| `CONTEXT7_API_KEY` | (none) | Enables Context7 third-party API signature grounding |
| `MAX_ANSWER_CHARS` | `10000` | Skip instances whose chosen answer exceeds this (keeps answers trainable) |

These can also be overridden via CLI flags (`--api-key`, `--base-url`, `--model`).

## Dataset Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HALLUCINATION_RATIO` | `0.4` | Fraction of instances that get hallucination injection |
| `MAX_FILE_CHARS` | `12000` | Maximum characters per source file |
| `MAX_CONTEXT7_CHARS` | `4000` | Maximum characters fetched per Context7 lookup |
| `LLM_TEMPERATURE` | `0.7` | Temperature for query rewriting |
| `HALLUCINATION_TEMPERATURE` | `0.8` | Temperature for hallucination injection (higher for variety) |
| `MAX_RETRIES` | `3` | API retry attempts |
| `RETRY_DELAY` | `2.0` | Base delay between retries (seconds) |

## Answer Source

The answer is set by the generator's `--answer-source` flag:

| Value | Description |
|-------|-------------|
| `gold` (default) | the project's real fix, used verbatim — no model call. Rendered per-instance as `function` (largest patched function fitting the cap), `fragment` (the hunk), or `edit` (`In file X, replace Y with Z`) |
| `generated` | an LLM writes a coherent solution to the request |

## Grounding

Answer references missing from the context are grounded in four tiers
(`answer_grounding.py`): the patch's modified functions, the changed files,
modules the answer imports, and modules the changed files import (base mixins /
sibling modules → cross-module `self.method` calls). Structural fabrications on
third-party APIs additionally get a Context7 `Library signatures` block. Set
`GITHUB_TOKEN` and `CONTEXT7_API_KEY` to raise the respective limits.

## Hallucination Types

Injected per instance and mapped to the unified taxonomy:

- **wrong_implementation** → `contradiction` — wrong logic, condition, field, or value
- **unrequested_change** → `unsupported_addition` — an extra block or side effect the request never asked for
- **fabricated_api** → `fabricated_reference` — a method/attribute/keyword that does not exist on a real object

The `--struct-ratio` flag sets the share of hallucinated samples that receive a
`fabricated_api` (structural) edit; the rest are intent edits.

## File Paths

Cached preparation inputs live under `data/code_hallucination/`:

| Path | Description |
|------|-------------|
| `swebench_instances.json` | loaded SWE-bench instances |
| `repos/` | bare git clones |
| `source_cache/` | per-instance source + gold edit |
| `queries.jsonl` | rewritten developer requests |
| `documentation.jsonl` | cached Context7 library docs (folded into context when present) |

Generated samples are written under the generator's `--out` (e.g.
`data/v2/code_agent/`) as `{train,dev,test}.jsonl` plus matching
`.failures.jsonl`.

## Data Sources

| Source | Dataset ID |
|--------|-----------|
| SWE-bench (full) | `princeton-nlp/SWE-bench` |
| SWE-bench Lite | `princeton-nlp/SWE-bench_Lite` |
