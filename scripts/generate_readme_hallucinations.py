#!/usr/bin/env python3
r"""Generate markdown hallucination samples from GitHub READMEs.

Reads the README corpus produced by ``collect_github_readmes.py``, assigns a
repo-level train/dev/test split, and runs the shared document-source pipeline:
chunk each README by heading, generate a developer-style question, generate a
grounded answer, and inject a dev-doc hallucination detectable against the
README. Output is the v2 schema, ``dataset="lettucedetect-readme"``.

Usage::

    API_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \\
        MODEL=Qwen/Qwen3.6-35B-A3B \\
        python scripts/generate_readme_hallucinations.py \\
            --corpus data/readmes/github_readmes.jsonl --limit 20 --out data/v2/readme
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI  # noqa: E402

from lettucedetect.generation.doc_source import (  # noqa: E402
    DocSourceConfig,
    generate_doc_source,
    hash_split,
)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.6-35B-A3B")
HALLUCINATION_RATIO = float(os.environ.get("HALLUCINATION_RATIO", "0.4"))

# Developer-style questions a reader asks of a README. Injection uses the generic
# factual markdown prompt (README content is heterogeneous — not all commands/flags).
README_QUESTION_TYPES = [
    "Instrumental/Procedural",
    "Feature Specification",
    "Definition",
    "Verification",
    "Quantification",
]


def load_corpus(path: Path, limit: int) -> list[dict]:
    """Load the README corpus into doc dicts ({id, text, split})."""
    docs = []
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            repo = row.get("repo", "")
            readme = row.get("readme", "")
            if not repo or not readme:
                continue
            docs.append({"id": repo, "text": readme, "split": hash_split(repo)})
    if limit:
        docs = docs[:limit]
    return docs


def main() -> None:
    """Generate README hallucination samples per split via the shared pipeline."""
    ap = argparse.ArgumentParser(description="Generate README hallucination samples.")
    ap.add_argument("--corpus", default="data/readmes/github_readmes.jsonl")
    ap.add_argument("--limit", type=int, default=20, help="Max repos (0=all).")
    ap.add_argument("--out", type=str, default="data/v2/readme")
    ap.add_argument("--ratio", type=float, default=HALLUCINATION_RATIO)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    args = ap.parse_args()

    docs = load_corpus(Path(args.corpus), args.limit)
    print(f"LLM: {MODEL} @ {API_BASE_URL}  | repos: {len(docs)}")
    aclient = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DocSourceConfig(
        dataset_name="lettucedetect-readme",
        question_types=README_QUESTION_TYPES,
    )  # uses the generic factual markdown injection (default)

    for split in ("train", "dev", "test"):
        split_docs = [d for d in docs if d["split"] == split]
        if not split_docs:
            continue
        out_path = out_dir / f"{split}.jsonl"
        stats = generate_doc_source(
            split_docs,
            cfg,
            aclient=aclient,
            model=MODEL,
            out_path=out_path,
            failures_path=out_dir / f"{split}.failures.jsonl",
            ratio=args.ratio,
            seed=args.seed,
            batch_size=args.batch_size,
            on_progress=lambda s: print(f"  {split}: {s}"),
        )
        print(f"{split}: {stats} -> {out_path}")


if __name__ == "__main__":
    main()
