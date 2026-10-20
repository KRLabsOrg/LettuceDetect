# AGENTS.md

Entry point for coding agents and new contributors. Details live in
[CONTRIBUTING.md](CONTRIBUTING.md); keep this file short.

## Setup and checks

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -e ".[dev]"                             # add ".[api]" for the FastAPI server

ruff format --check lettucedetect/ lettucedetect_api/ tests/
ruff check lettucedetect/ lettucedetect_api/ tests/ --extend-exclude lettucedetect/integrations/
python -m pytest
```

The network-free unit suite (`python -m pytest`) is the merge gate; CI runs the
same lint and test commands. Test files must be named `tests/test_*_pytest.py`.

## Repository map

- `lettucedetect/detectors/` - detectors (`transformer.py`, `llm.py`, `rag_fact_checker.py`), `BaseDetector` in `base.py`, `make_detector()` in `factory.py`
- `lettucedetect/models/` - `HallucinationDetector` inference facade, training, evaluation
- `lettucedetect/datasets/` - dataset classes and the hallucination taxonomy
- `lettucedetect/preprocess/` - RAGTruth / RAGBench preprocessing
- `lettucedetect/prompts/` - per-language prompt templates and examples
- `lettucedetect/integrations/` - LangChain, Elysia, Claude Code hook
- `lettucedetect/cli.py` - `lettucedetect` command line entry point
- `lettucedetect_api/` - FastAPI server and client
- `scripts/` - training, evaluation and data scripts; `docs/` - MkDocs site

## Durable constraints

- Unit tests must not touch the network: no Hugging Face downloads, no API
  calls. Mock clients and build tiny tokenizers/models locally (see `tests/conftest.py`).
- Published model ids use the `lettucedect-` prefix (for example
  `KRLabsOrg/lettucedect-base-modernbert-en-v1`), not `lettucedetect-`.
- `transformers>=4.48.3`. The EuroBERT models load remote code that needs
  `transformers<5`; ModernBERT models work on 5.x.
- Ruff, line length 100, modern type hints, Sphinx-style docstrings, `logging`
  instead of `print()`, `pathlib` instead of `os.path`.
- Keep PRs focused, fill in the PR template, and note user-facing changes in
  [CHANGELOG.md](CHANGELOG.md).
