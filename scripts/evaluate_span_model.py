#!/usr/bin/env python3
"""Evaluate a trained span detector on an HF dataset, broken down by source and language.

Reports char-span and example-level P/R/F1 overall and per `dataset` / `language`,
so one model can be scored across code + prose + every language in one table.

Usage:
    python scripts/evaluate_span_model.py \
        --model-path output/mmbert_binary_multiling \
        --dataset KRLabsOrg/lettucedetect-code-hallucination \
        --dataset KRLabsOrg/lettucedetect-prose-hallucination \
        --split test [--by language] [--limit N]
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset  # noqa: E402

from lettucedetect.datasets.hallucination_dataset import HallucinationSample  # noqa: E402
from lettucedetect.models.evaluator import (  # noqa: E402
    evaluate_detector_char_level,
    evaluate_detector_example_level_batch,
)
from lettucedetect.models.inference import HallucinationDetector  # noqa: E402


def load_samples(datasets_: list[str], split: str, limit: int) -> list[HallucinationSample]:
    """Load an HF split into HallucinationSamples (carrying dataset + language)."""
    out: list[HallucinationSample] = []
    for name in datasets_:
        ds = load_dataset(name, split=split)
        for r in ds:
            out.append(
                HallucinationSample(
                    prompt=r["prompt"],
                    answer=r["answer"],
                    labels=r.get("labels") or [],
                    split=split,
                    task_type=r.get("task_type", ""),
                    dataset=r.get("dataset", name),
                    language=r.get("language", "en"),
                )
            )
            if limit and len(out) >= limit:
                return out
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Per-source span-detector evaluation.")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", action="append", default=[], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--by", choices=["dataset", "language"], default="dataset")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--device", default=None, help="cpu / cuda / cuda:N (default auto).")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    kw = {"max_length": args.max_length, "trust_remote_code": args.trust_remote_code}
    if args.device:
        kw["device"] = args.device
    detector = HallucinationDetector(method="transformer", model_path=args.model_path, **kw)
    samples = load_samples(args.dataset, args.split, args.limit)

    groups: dict[str, list] = collections.defaultdict(list)
    groups["ALL"] = samples
    for s in samples:
        groups[getattr(s, args.by)].append(s)

    print(f"{'group':<28} {'n':>6} {'span_f1':>8} {'span_p':>7} {'span_r':>7} {'ex_f1':>7}")
    for name in ["ALL", *sorted(k for k in groups if k != "ALL")]:
        grp = groups[name]
        span = evaluate_detector_char_level(detector, grp)
        ex = evaluate_detector_example_level_batch(detector, grp, verbose=False)
        exf1 = ex.get("hallucinated", {}).get("f1", 0.0)
        print(
            f"{name:<28} {len(grp):>6} {span['f1']:>8.4f} {span['precision']:>7.4f} "
            f"{span['recall']:>7.4f} {exf1:>7.4f}"
        )


if __name__ == "__main__":
    main()
