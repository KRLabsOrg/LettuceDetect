# Installation

## From PyPI

```bash
pip install lettucedetect
```

## From Source (development)

```bash
git clone https://github.com/KRLabsOrg/LettuceDetect.git
cd LettuceDetect
pip install -e .
```

## Optional Dependencies

```bash
# Web API server
pip install -e ".[api]"

# Development tools (testing, linting)
pip install -e ".[dev]"

# Documentation site
pip install -e ".[docs]"
```

## Requirements

- Python >= 3.11
- PyTorch >= 2.6.0
- Transformers >= 4.48.3

## Environment Variables

For LLM-based detection or data generation:

```bash
export OPENAI_API_KEY=your_api_key
```
