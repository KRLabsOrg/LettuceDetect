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
from lettucedetect.models.inference import HallucinationDetector  # noqa: E402


def load_samples(
    datasets_: list[str], split: str, limit: int, only: str = ""
) -> list[HallucinationSample]:
    """Load an HF split into HallucinationSamples (carrying dataset + language)."""
    out: list[HallucinationSample] = []
    for name in datasets_:
        ds = load_dataset(name, split=split)
        for r in ds:
            if only and r.get("dataset") != only:
                continue
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
    ap.add_argument("--only", default="", help="Keep only rows whose `dataset` field == this.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--device", default=None, help="cpu / cuda / cuda:N (default auto).")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    kw = {"max_length": args.max_length, "trust_remote_code": args.trust_remote_code}
    if args.device:
        kw["device"] = args.device
    detector = HallucinationDetector(method="transformer", model_path=args.model_path, **kw)
    samples = load_samples(args.dataset, args.split, args.limit, args.only)

    # Predict each sample ONCE; metrics are computed from the cached spans.
    from tqdm import tqdm

    rows = []
    for s in tqdm(samples, desc="predict"):
        pred = detector.predict_prompt(s.prompt, s.answer, output_format="spans")
        rows.append((getattr(s, args.by), s.labels, pred))

    def char_set(spans: list) -> set:
        out: set[int] = set()
        for s in spans:
            out.update(range(s["start"], s["end"]))
        return out

    def metrics(items: list) -> tuple[float, float, float, float, float]:
        ov = pp = gg = 0  # char overlap / predicted / gold
        etp = efp = efn = 0  # example-level on "has any span"
        iou_sum = 0.0  # per-sample char IoU, PsiloQA's unit
        for _key, gold, pred in items:
            pp += sum(p["end"] - p["start"] for p in pred)
            gg += sum(g["end"] - g["start"] for g in gold)
            for p in pred:
                for g in gold:
                    ov += max(0, min(p["end"], g["end"]) - max(p["start"], g["start"]))
            ps, gs = char_set(pred), char_set(gold)
            union = ps | gs
            iou_sum += 1.0 if not union else len(ps & gs) / len(union)
            pe, ge = bool(pred), bool(gold)
            etp += pe and ge
            efp += pe and not ge
            efn += ge and not pe
        cp = ov / pp if pp else 0.0
        cr = ov / gg if gg else 0.0
        cf = 2 * cp * cr / (cp + cr) if cp + cr else 0.0
        ep = etp / (etp + efp) if etp + efp else 0.0
        er = etp / (etp + efn) if etp + efn else 0.0
        ef = 2 * ep * er / (ep + er) if ep + er else 0.0
        iou = iou_sum / len(items) if items else 0.0
        return cf, cp, cr, ef, iou

    groups: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        groups[r[0]].append(r)

    hdr = f"{'group':<28} {'n':>6} {'span_f1':>8} {'span_p':>7} {'span_r':>7} {'ex_f1':>7} {'iou':>7}"
    print(hdr)
    for name in ["ALL", *sorted(groups)]:
        grp = rows if name == "ALL" else groups[name]
        cf, cp, cr, ef, iou = metrics(grp)
        print(
            f"{name:<28} {len(grp):>6} {cf:>8.4f} {cp:>7.4f} {cr:>7.4f} {ef:>7.4f} {iou:>7.4f}"
        )


if __name__ == "__main__":
    main()
