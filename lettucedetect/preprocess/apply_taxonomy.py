r"""Apply the unified v2 taxonomy to any LettuceDetect data source.

Takes an already-preprocessed dataset and enriches each span and sample
with ``category`` and ``subcategory`` fields from the unified taxonomy,
plus ``context_modality`` and source provenance metadata.

Supported sources:
  code        -- SWE-bench-derived code hallucination dataset
  ragtruth    -- RAGTruth QA/summarization dataset
  ragbench    -- RAGBench dataset
  markdown    -- planned markdown hallucination dataset

Usage:
  python -m lettucedetect.preprocess.apply_taxonomy \\
      --source code \\
      --data_path data/code_hallucination/code_hallucination_data.json \\
      --metadata_path data/code_hallucination/code_hallucination_metadata.json \\
      --output_dir data/v2/code_hallucination

  python -m lettucedetect.preprocess.apply_taxonomy \\
      --source ragtruth \\
      --data_path data/ragtruth/ragtruth_data.json \\
      --output_dir data/v2/ragtruth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lettucedetect.datasets.hallucination_dataset import HallucinationSample
from lettucedetect.datasets.taxonomy import map_label, ragtruth_map_with_context

# ── Per-source converters ──────────────────────────────────────────────────────


def _convert_span(span: dict, source: str, context: str = "") -> dict:
    """Add category/subcategory to a span dict using the unified taxonomy."""
    raw = span.get("label") or span.get("label_type", "")
    if source == "ragtruth":
        category, subcategory = ragtruth_map_with_context(raw, span.get("text", ""), context)
    else:
        category, subcategory = map_label(raw, source)
    return {**span, "category": category, "subcategory": subcategory}


def _sample_category(spans: list[dict]) -> tuple[str | None, str | None]:
    """Derive sample-level category/subcategory from its spans (majority vote)."""
    if not spans:
        return None, None
    from collections import Counter

    cat_counts: Counter = Counter(s.get("category") for s in spans if s.get("category"))
    if not cat_counts:
        return None, None
    top_cat = cat_counts.most_common(1)[0][0]
    sub_counts: Counter = Counter(
        s.get("subcategory") for s in spans if s.get("category") == top_cat and s.get("subcategory")
    )
    top_sub = sub_counts.most_common(1)[0][0] if sub_counts else None
    return top_cat, top_sub


# ── code ──────────────────────────────────────────────────────────────────────


def preprocess_code(
    data_path: Path,
    metadata_path: Path,
) -> list[HallucinationSample]:
    """Load the code dataset + metadata and convert to taxonomy-tagged samples."""
    with open(data_path) as f:
        raw_samples = json.load(f)
    with open(metadata_path) as f:
        raw_meta = json.load(f)

    if len(raw_samples) != len(raw_meta):
        raise ValueError("data/metadata length mismatch")

    samples = []
    for s, m in zip(raw_samples, raw_meta):
        converted_spans = [_convert_span(span, "code") for span in s["labels"]]
        category, subcategory = _sample_category(converted_spans)

        samples.append(
            HallucinationSample(
                prompt=s["prompt"],
                answer=s["answer"],
                labels=converted_spans,
                split=s["split"],
                task_type=s["task_type"],
                dataset="lettucedetect-code",
                language=s.get("language", "en"),
                context_modality="code",
                category=category,
                subcategory=subcategory,
                context=s.get("context"),
                question=s.get("question"),
                metadata={
                    "instance_id": m.get("instance_id", ""),
                    "repo": m.get("repo", ""),
                    "format_type": m.get("format_type"),
                    "is_hallucinated": m.get("is_hallucinated", bool(s["labels"])),
                    "injector_model": m.get("injector_model"),
                    "reasoning": m.get("reasoning"),
                },
            )
        )
    return samples


# ── ragtruth ──────────────────────────────────────────────────────────────────


def preprocess_ragtruth(data_path: Path) -> list[HallucinationSample]:
    """Convert RAGTruth data to taxonomy-tagged samples (context-aware mapping)."""
    with open(data_path) as f:
        raw = json.load(f)

    samples = []
    for s in raw:
        context = s["prompt"]
        converted_spans = [_convert_span(span, "ragtruth", context) for span in s["labels"]]
        category, subcategory = _sample_category(converted_spans)

        samples.append(
            HallucinationSample(
                prompt=s["prompt"],
                answer=s["answer"],
                labels=converted_spans,
                split=s["split"],
                task_type=s["task_type"],
                dataset="ragtruth",
                language=s.get("language", "en"),
                context_modality="prose",
                category=category,
                subcategory=subcategory,
            )
        )
    return samples


# ── ragbench ──────────────────────────────────────────────────────────────────


def preprocess_ragbench(data_path: Path) -> list[HallucinationSample]:
    """RAGBench uses binary labels; map to contradiction as a placeholder."""
    with open(data_path) as f:
        raw = json.load(f)

    samples = []
    for s in raw:
        converted_spans = [_convert_span(span, "ragtruth") for span in s["labels"]]
        category, subcategory = _sample_category(converted_spans)

        samples.append(
            HallucinationSample(
                prompt=s["prompt"],
                answer=s["answer"],
                labels=converted_spans,
                split=s["split"],
                task_type=s["task_type"],
                dataset="ragbench",
                language=s.get("language", "en"),
                context_modality="prose",
                category=category,
                subcategory=subcategory,
            )
        )
    return samples


# ── markdown (stub for future use) ────────────────────────────────────────────


def preprocess_markdown(
    data_path: Path, metadata_path: Path | None = None
) -> list[HallucinationSample]:
    """Convert a generic markdown dataset to taxonomy-tagged samples."""
    with open(data_path) as f:
        raw_samples = json.load(f)

    samples = []
    for s in raw_samples:
        converted_spans = [_convert_span(span, "markdown") for span in s["labels"]]
        category, subcategory = _sample_category(converted_spans)

        meta = {}
        if "source_url" in s:
            meta["source_url"] = s["source_url"]
        if "doc_type" in s:
            meta["doc_type"] = s["doc_type"]

        samples.append(
            HallucinationSample(
                prompt=s["prompt"],
                answer=s["answer"],
                labels=converted_spans,
                split=s["split"],
                task_type=s.get("task_type", "qa"),
                dataset="lettucedetect-markdown",
                language=s.get("language", "en"),
                context_modality="markdown",
                category=category,
                subcategory=subcategory,
                metadata=meta,
            )
        )
    return samples


# ── Output ────────────────────────────────────────────────────────────────────


def write_output(samples: list[HallucinationSample], output_dir: Path) -> None:
    """Write samples to one JSONL file per split under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[HallucinationSample]] = {}
    for s in samples:
        by_split.setdefault(s.split, []).append(s)

    for split, split_samples in sorted(by_split.items()):
        out_path = output_dir / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for s in split_samples:
                f.write(json.dumps(s.to_json()) + "\n")
        n_hall = sum(1 for s in split_samples if s.labels)
        print(f"  {split}: {len(split_samples)} samples ({n_hall} hallucinated) -> {out_path}")

    print(f"\nTotal: {len(samples)} samples")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI: preprocess one data source to the unified-taxonomy schema."""
    parser = argparse.ArgumentParser(
        description="Preprocess a data source to the v2 unified schema."
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["code", "ragtruth", "ragbench", "markdown"],
        help="Data source to preprocess.",
    )
    parser.add_argument("--data_path", required=True, help="Path to the main data file.")
    parser.add_argument(
        "--metadata_path",
        default=None,
        help="Path to the metadata file (required for code source).",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to write JSONL output.")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    metadata_path = Path(args.metadata_path) if args.metadata_path else None

    print(f"Preprocessing source: {args.source}")

    if args.source == "code":
        if metadata_path is None:
            parser.error("--metadata_path is required for source=code")
        samples = preprocess_code(data_path, metadata_path)
    elif args.source == "ragtruth":
        samples = preprocess_ragtruth(data_path)
    elif args.source == "ragbench":
        samples = preprocess_ragbench(data_path)
    elif args.source == "markdown":
        samples = preprocess_markdown(data_path, metadata_path)

    write_output(samples, output_dir)


if __name__ == "__main__":
    main()
