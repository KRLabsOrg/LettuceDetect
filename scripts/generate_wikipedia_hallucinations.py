#!/usr/bin/env python3
r"""Generate markdown hallucination samples from Wikipedia articles.

Streams the English shards of ``open-index/open-wikipedia-markdown`` (the dataset
script is broken, so the parquet files are loaded directly), samples substantial
articles, and runs the shared document-source pipeline: chunk by heading ->
factual question -> grounded answer -> generic factual injection. Output is the
v2 schema, ``dataset="lettucedetect-wikipedia"``.

Usage::

    API_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \\
        MODEL=Qwen/Qwen3.6-35B-A3B \\
        python scripts/generate_wikipedia_hallucinations.py --limit 200 --out data/v2/wikipedia
"""

from __future__ import annotations

import argparse
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
PARQUET_GLOB = "hf://datasets/open-index/open-wikipedia-markdown/data/en/*.parquet"
MIN_ARTICLE_LENGTH = 1500

# Factual question types that suit an encyclopedic article.
WIKI_QUESTION_TYPES = [
    "Concept Completion",
    "Quantification",
    "Definition",
    "Feature Specification",
    "Comparison",
    "Causal Antecedent",
    "Causal Consequence",
    "Verification",
    "Interpretation",
]


def load_articles(limit: int) -> list[dict]:
    """Stream English Wikipedia parquet shards into doc dicts ({id, text, split})."""
    from datasets import load_dataset

    stream = load_dataset("parquet", data_files=PARQUET_GLOB, split="train", streaming=True)
    docs = []
    for row in stream:
        if (row.get("length") or 0) < MIN_ARTICLE_LENGTH or not row.get("markdown"):
            continue
        doc_id = str(row["id"])
        docs.append({"id": doc_id, "text": row["markdown"], "split": hash_split(doc_id)})
        if limit and len(docs) >= limit:
            break
    return docs


def main() -> None:
    """Generate Wikipedia hallucination samples per split via the shared pipeline."""
    ap = argparse.ArgumentParser(description="Generate Wikipedia hallucination samples.")
    ap.add_argument("--limit", type=int, default=50, help="Max articles (0=all, streamed).")
    ap.add_argument("--out", type=str, default="data/v2/wikipedia")
    ap.add_argument("--ratio", type=float, default=HALLUCINATION_RATIO)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    args = ap.parse_args()

    docs = load_articles(args.limit)
    print(f"LLM: {MODEL} @ {API_BASE_URL}  | articles: {len(docs)}")
    aclient = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DocSourceConfig(
        dataset_name="lettucedetect-wikipedia",
        question_types=WIKI_QUESTION_TYPES,
        # Pack several article sections per chunk for a longer, more realistic
        # grounding context (single sections are often too short).
        min_chunk_chars=1500,
        max_chunk_chars=6000,
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
