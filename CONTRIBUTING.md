# Contributing to LettuceDetect

Thanks for your interest in contributing! This guide will get you up and running.

## Setup

```bash
# Clone the repo
git clone https://github.com/KRLabsOrg/LettuceDetect.git
cd LettuceDetect

# Create a virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

There are also optional dependency groups:

```bash
pip install -e ".[api]"   # FastAPI server and client
pip install -e ".[docs]"  # MkDocs documentation
```

## Code style

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting, with a line length of 100.

```bash
# Check formatting
ruff format --check lettucedetect/ lettucedetect_api/ tests/

# Auto-format
ruff format lettucedetect/ lettucedetect_api/ tests/

# Lint
ruff check lettucedetect/ lettucedetect_api/ tests/

# Lint with auto-fix
ruff check --fix lettucedetect/ lettucedetect_api/ tests/
```

Key conventions:
- Use modern type hints (`list[str]`, `dict[str, Any]`, `str | None`) with `from __future__ import annotations`
- Add docstrings to public classes and methods (Sphinx `:param:` / `:return:` style)
- Use `logging` instead of `print()` for runtime messages in library code (see [Errors and logging](#errors-and-logging) for what counts as library code)
- Use `pathlib.Path` instead of `os.path`

## Errors and logging

This convention is adapted from the one in
[VerbatimRAG's contributor guide](https://github.com/KRLabsOrg/verbatim-rag/tree/stateless_core?tab=contributing-ov-file),
so both projects behave the same way.

### Where it applies

Library code in `lettucedetect/` follows every rule below. The command-line entry point
(`lettucedetect/cli.py`), `lettucedetect_api/`, `scripts/`, `demo/`, and training and evaluation code
are applications: they configure logging themselves and may print.

### Errors

- **Correct result or raise.** A call either returns a correct result or raises. If a result is
  correct but degraded (for example, a fallback was used), it says so in its `warnings`. A result that
  looks complete while part of the work failed is a bug.
- **Error class depends on where the bad value came from:**
  - `InvalidInputError` for an argument of a call. Set `param` to the argument's name.
  - `ConfigurationError` for a constructor argument, a setting, an environment problem or a missing
    package.
  - `UpstreamError` for a failed or missing reply from a service (an LLM API, a model download).
  - Bugs in our own code (`KeyError`, `TypeError`, ...) propagate unwrapped.
- **Stable codes.** Every error and every result warning has a snake_case `code`
  (for example `empty_context`). Codes are public API: don't rename or reuse them.
- **Messages** name the problem and the fix. They never contain secrets or customer text (context,
  question, answer, model replies). Refer to items by position, such as `spans[3]`.
- **Log or raise, never both.** Whoever handles the error decides whether to log it.
- **Translate foreign errors** into one of the classes above and chain them with `raise ... from exc`.
- **No bare `except Exception`** outside application boundaries or code we don't control. Where it is
  needed, add a comment saying why.
- **Fallbacks** catch only `UpstreamError` and add a documented warning code to the result. They never
  mask `ConfigurationError`, `InvalidInputError` or bugs.
- **`warnings.warn` is only for deprecations**, and uses `LettuceDetectDeprecationWarning`. The test
  suite treats that class as an error, so add each new deprecation to the catalogue in
  `tests/test_deprecation.py`.

### Logging

- **One logger per module:** `logger = logging.getLogger(__name__)`.
- **Library code never prints, shows progress bars, adds handlers or calls `logging.basicConfig`.**
  Applications do that.
- **Use %-style arguments**, not f-strings: `logger.info("loaded %s in %.1fs", name, secs)`. Keep each
  message on one line.
- **Events.** Key events carry `extra={"event": "<name>"}`. The names are `model_loaded`,
  `fallback_used` and `item_failed`.
- **Levels:**

  | Level | Use for |
  |---|---|
  | DEBUG | the steps of a call |
  | INFO | events of healthy operation (`model_loaded`) |
  | WARNING | a degraded result (`fallback_used`, `item_failed`) |
  | ERROR | only where the exception can't reach a caller |

- **Customer text** (context, question, answer, model replies) is logged only on the
  `lettucedetect.content` logger, and only at DEBUG. Module loggers record counts and positions instead.

## Running tests

```bash
# Run the complete network-free unit suite
python -m pytest # or: uv run pytest
```

Test files must follow the naming pattern `test_*_pytest.py`.

## Project structure

```
lettucedetect/
  detectors/        # Detection methods (transformer, LLM, RAGFactChecker)
    transformer.py   # Fine-tuned encoder model detector
    llm.py           # OpenAI API-based detector
    factory.py       # make_detector() factory function
    base.py          # Abstract base class
    prompt_utils.py  # Prompt formatting utilities
  datasets/          # Dataset classes and data structures
  models/            # Inference facade, training, evaluation
  preprocess/        # Data preprocessing (RAGTruth, RAGBench)
  prompts/           # Prompt templates per language
  integrations/      # Framework integrations (LangChain, etc.)
lettucedetect_api/   # FastAPI web server and client
tests/               # Pytest test suite
docs/                # MkDocs documentation source
scripts/             # Training, evaluation, and utility scripts
```

## Making changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b my-feature
   ```

2. **Make your changes.** Keep PRs focused — one feature or fix per PR.

3. **Follow [Errors and logging](#errors-and-logging)** for every error you raise and every message you log.

4. **Run lint and tests** before committing:
   ```bash
   ruff format lettucedetect/ lettucedetect_api/ tests/
   ruff check lettucedetect/ lettucedetect_api/ tests/
   python -m pytest
   ```

5. **Open a pull request** against `main`. CI will run lint and tests automatically.

## What to work on

Good areas for contribution:

- **New language support** — add prompt templates in `lettucedetect/prompts/` (see existing `qa_prompt_en.txt` and `summary_prompt_en.txt` as examples, and add the language to `LANG_TO_PASSAGE` in `prompt_utils.py`)
- **New detection methods** — implement `BaseDetector` in `lettucedetect/detectors/` and register it in `factory.py`
- **Documentation** — improve or expand the docs in `docs/`
- **Tests** — increase coverage, especially for edge cases
- **Bug fixes** — check [open issues](https://github.com/KRLabsOrg/LettuceDetect/issues)

### Issue labels

The maintainers use labels to make contribution scope clearer:

- `good first issue` marks small, well-scoped tasks that are suitable for new contributors.
- `help wanted` marks issues where outside implementation or investigation would be useful.
- `bug` marks reproducible behavior that differs from the expected result.
- `documentation` marks docs, examples, templates, and onboarding improvements.

Before starting work, read the issue thread to confirm nobody has already claimed it. If the issue is unclear, ask a focused question before opening a PR.

## Building docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

## Contribution rights

By submitting a pull request you confirm that you wrote the contribution (or
have the right to submit it) and that it may be distributed under this
repository's MIT license (this is the checkbox in the PR template, enforced
by CI). Optionally you can also sign off commits with `git commit -s`
([DCO](https://developercertificate.org/)); appreciated, not required.
