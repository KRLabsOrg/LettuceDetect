#!/usr/bin/env python3
"""Example-level baseline panel: do off-the-shelf detectors catch hallucinated answers?

Most general/specialist detectors (HHEM, Lynx, Granite-Guardian) only emit an
answer-level verdict, not spans. This harness scores them at the example level
(gold = the answer contains >=1 annotated span) so they sit in one table with our
models (collapsed to "predicted >=1 span -> hallucinated"). The thesis: they fail
on code.

    python scripts/evaluate_example_baselines.py --baseline hhem \
        --dataset KRLabsOrg/lettucedetect-code-hallucination --only lettucedetect-code-agent
"""

from __future__ import annotations

import argparse

from datasets import load_dataset


def load_rows(dataset: str, split: str, only: str | None, limit: int) -> list[dict]:
    """Return [{premise, answer, gold}] where gold = answer has >=1 annotated span."""
    rows = []
    for r in load_dataset(dataset, split=split):
        if only and r.get("dataset") != only:
            continue
        rows.append(
            {
                "premise": r["prompt"] or r["context"],
                "answer": r["answer"],
                "gold": bool(r.get("labels")),
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def example_metrics(gold: list[bool], pred: list[bool]) -> dict[str, float]:
    """Precision/recall/F1 for the hallucinated class + balanced accuracy + accuracy."""
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    tn = sum((not g) and (not p) for g, p in zip(gold, pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    bacc = (tpr + tnr) / 2
    acc = (tp + tn) / len(gold) if gold else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "bacc": bacc, "accuracy": acc, "n": len(gold)}


def predict_hhem(rows: list[dict], threshold: float, device: str) -> list[bool]:
    """HHEM-2.1-Open: consistency prob in [0,1] per (premise, answer); <threshold => hallucinated."""
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        "vectara/hallucination_evaluation_model", trust_remote_code=True
    ).to(device)
    pairs = [(r["premise"], r["answer"]) for r in rows]
    scores = []
    bs = 32
    for i in range(0, len(pairs), bs):
        with torch.no_grad():
            scores.extend(model.predict(pairs[i : i + bs]).tolist())
    return [s < threshold for s in scores]


BASELINES = {"hhem": predict_hhem}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, choices=sorted(BASELINES))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--only", help="Keep only rows whose `dataset` field == this.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.dataset, args.split, args.only, args.limit)
    print(f"{args.baseline}: {len(rows)} rows ({sum(r['gold'] for r in rows)} hallucinated)")
    pred = BASELINES[args.baseline](rows, args.threshold, args.device)
    m = example_metrics([r["gold"] for r in rows], pred)
    print(
        f"  P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  "
        f"BAcc {m['bacc']:.3f}  Acc {m['accuracy']:.3f}  (n={m['n']})"
    )


def _selfcheck() -> None:
    """Metric sanity check (no model)."""
    g = [True, True, False, False]
    assert example_metrics(g, [True, True, False, False])["f1"] == 1.0
    assert example_metrics(g, [False, False, True, True])["f1"] == 0.0
    assert example_metrics(g, [True, True, True, True])["bacc"] == 0.5
    print("selfcheck ok")


if __name__ == "__main__":
    import sys

    _selfcheck() if len(sys.argv) == 1 else main()
