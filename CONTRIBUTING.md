# Contributing to LettuceDetect

Thanks for your interest in contributing! This guide will get you up and running.

## Setup

```bash
# Clone the repo
git clone https://github.com/KRLabsOrg/LettuceDetect.git
cd LettuceDetect

# Create a virtual environment (Python 3.10+)
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
ruff format --check lettucedetect/ tests/

# Auto-format
ruff format lettucedetect/ tests/

# Lint
ruff check lettucedetect/ tests/

# Lint with auto-fix
ruff check --fix lettucedetect/ tests/
```

Key conventions:
- Use modern type hints (`list[str]`, `dict[str, Any]`, `str | None`) with `from __future__ import annotations`
- Add docstrings to public classes and methods (Sphinx `:param:` / `:return:` style)
- Use `logging` instead of `print()` for runtime messages
- Use `pathlib.Path` instead of `os.path`

## Running tests

```bash
# Run the test suite
pytest tests/test_inference_pytest.py -v

# Skip tests that download models (faster)
pytest tests/test_inference_pytest.py -v -k "not TestAnswerStartToken"
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

3. **Run lint and tests** before committing:
   ```bash
   ruff format lettucedetect/ tests/
   ruff check lettucedetect/ tests/
   pytest tests/test_inference_pytest.py -v -k "not TestAnswerStartToken"
   ```

4. **Open a pull request** against `main`. CI will run lint and tests automatically.

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

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
