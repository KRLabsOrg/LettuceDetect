"""Configuration for the code hallucination dataset pipeline."""

import os
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "code_hallucination"
DATA_DIR = Path(os.environ.get("CODE_HALLUCINATION_OUTPUT_DIR", str(DEFAULT_DATA_DIR)))
REPOS_DIR = DATA_DIR / "repos"
SOURCE_CACHE_DIR = DATA_DIR / "source_cache"

# Intermediate outputs
INSTANCES_PATH = DATA_DIR / "swebench_instances.json"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
DOCS_PATH = DATA_DIR / "documentation.jsonl"
FORMATS_PATH = DATA_DIR / "formats.jsonl"
HALLUCINATED_PATH = DATA_DIR / "hallucinated_samples.jsonl"
INJECTION_FAILURES_PATH = DATA_DIR / "injection_failures.jsonl"

# Final outputs
DATASET_PATH = DATA_DIR / "code_hallucination_data.json"
METADATA_PATH = DATA_DIR / "code_hallucination_metadata.json"
VALIDATION_REPORT_PATH = DATA_DIR / "validation_report.txt"


def set_output_dir(path: str | os.PathLike[str]) -> Path:
    """Redirect all pipeline outputs to a specific directory."""
    global DATA_DIR
    global REPOS_DIR
    global SOURCE_CACHE_DIR
    global INSTANCES_PATH
    global QUERIES_PATH
    global DOCS_PATH
    global FORMATS_PATH
    global HALLUCINATED_PATH
    global INJECTION_FAILURES_PATH
    global DATASET_PATH
    global METADATA_PATH
    global VALIDATION_REPORT_PATH

    DATA_DIR = Path(path)
    REPOS_DIR = DATA_DIR / "repos"
    SOURCE_CACHE_DIR = DATA_DIR / "source_cache"
    INSTANCES_PATH = DATA_DIR / "swebench_instances.json"
    QUERIES_PATH = DATA_DIR / "queries.jsonl"
    DOCS_PATH = DATA_DIR / "documentation.jsonl"
    FORMATS_PATH = DATA_DIR / "formats.jsonl"
    HALLUCINATED_PATH = DATA_DIR / "hallucinated_samples.jsonl"
    INJECTION_FAILURES_PATH = DATA_DIR / "injection_failures.jsonl"
    DATASET_PATH = DATA_DIR / "code_hallucination_data.json"
    METADATA_PATH = DATA_DIR / "code_hallucination_metadata.json"
    VALIDATION_REPORT_PATH = DATA_DIR / "validation_report.txt"

    os.environ["CODE_HALLUCINATION_OUTPUT_DIR"] = str(DATA_DIR)
    return DATA_DIR


# === LLM API Config ===
# Override via env vars or CLI args
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("MODEL", "moonshotai/kimi-k2-instruct-0905")

# Context7
CONTEXT7_BASE = "https://context7.com/api/v2"
CONTEXT7_API_KEY = os.environ.get("CONTEXT7_API_KEY", "")
DOCS_RATIO = 0.2  # Only fetch docs for 20% of instances

# === Dataset Config ===
HALLUCINATION_RATIO = 0.4  # 40% hallucinated, 60% clean
MAX_FILE_CHARS = 12000  # Cap individual source file size
MAX_CONTEXT7_CHARS = 4000  # Documentation fetch limit
MAX_PROMPT_CHARS = 32000  # ~8K tokens, fits EuroBERT's full context (answer avg ~500 tokens)
MAX_ANSWER_CHARS = 50000  # drop answers longer than this (entire-file dump artifacts)

# === LLM Config ===
RETRY_DELAY = 2
MAX_RETRIES = 3
LLM_TEMPERATURE = 0.7
HALLUCINATION_TEMPERATURE = 0.8
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))  # >1 for local vLLM

# Hallucination types (round-robin assignment)
HALLUCINATION_TYPES = ["structural", "behavioral", "semantic"]

# Answer format types
FORMAT_TYPES = ["complete_function", "edit_style", "fragment", "code_with_explanation"]
FORMAT_WEIGHTS = [0.25, 0.15, 0.20, 0.40]  # Target distribution

# SWE-bench datasets
SWEBENCH_FULL = "princeton-nlp/SWE-bench"
SWEBENCH_LITE = "princeton-nlp/SWE-bench_Lite"

# Models that require max_completion_tokens instead of max_tokens
_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def token_limit_kwargs(model: str, max_tokens: int = 4000) -> dict:
    """Return the right token-limit kwarg for the given model."""
    if any(model.startswith(p) for p in _REASONING_MODEL_PREFIXES):
        return {"max_completion_tokens": max_tokens, "reasoning_effort": "none"}
    return {"max_tokens": max_tokens}
