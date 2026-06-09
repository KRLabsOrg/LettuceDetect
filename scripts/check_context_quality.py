#!/usr/bin/env python3
"""Audit a generated code-agent dataset for grounding coverage and label quality.

Reports, per split/class, how often an answer references a symbol that is not
evidenced anywhere in the context (the "missing/ungrounded reference" signal),
plus category/mode/format distributions, span coverage, and answer length — the
dataset-level checks needed before training.

Usage::

    python scripts/check_context_quality.py --data data/v2/code_agent
    python scripts/check_context_quality.py --data data/v2/code_agent_review --show 15
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from collections.abc import Iterator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.code_hallucination.answer_grounding import remaining_ungrounded  # noqa: E402


def load_samples(path: str) -> Iterator[dict]:
    """Yield samples from a JSONL file or a directory of split files."""
    p = Path(path)
    files = (
        [f for f in sorted(p.glob("*.jsonl")) if not f.name.endswith(".failures.jsonl")]
        if p.is_dir()
        else [p]
    )
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                s = json.loads(line)
                meta = s.get("metadata")
                s["metadata"] = json.loads(meta) if isinstance(meta, str) else (meta or {})
                yield s


def main() -> None:
    """Print the grounding + label-quality audit for a dataset."""
    ap = argparse.ArgumentParser(description="Audit code-agent dataset grounding and labels.")
    ap.add_argument("--data", default="data/v2/code_agent")
    ap.add_argument("--show", type=int, default=10, help="Example ungrounded samples to print.")
    args = ap.parse_args()

    samples = list(load_samples(args.data))
    if not samples:
        print(f"No samples at {args.data}")
        return
    hall = [s for s in samples if s["labels"]]

    print(f"=== {len(samples)} samples ({len(hall)} hallucinated, {len(samples) - len(hall)} clean) ===\n")

    # Grounding: answer references not evidenced in the context.
    examples: list[tuple] = []
    counts = collections.Counter()
    for s in samples:
        # Blank out labeled spans: an injected fabrication is *meant* to be ungrounded,
        # so only unlabeled ungrounded references count as a missed-grounding problem.
        answer = s["answer"]
        for label in sorted(s["labels"], key=lambda x: x["start"], reverse=True):
            answer = answer[: label["start"]] + " " + answer[label["end"] :]
        ung = remaining_ungrounded(answer, s["context"])
        cls = "hall" if s["labels"] else "clean"
        counts[f"{cls}_n"] += 1
        if ung:
            counts[f"{cls}_ung"] += 1
            if len(examples) < args.show:
                examples.append((cls, s["metadata"].get("instance_id", "?"), sorted(ung)[:5]))
    print("GROUNDING — samples with >=1 ungrounded reference (lower is better):")
    for cls in ("clean", "hall"):
        n, u = counts[f"{cls}_n"], counts[f"{cls}_ung"]
        if n:
            print(f"  {cls}: {u}/{n} ({100 * u / n:.1f}%)")

    # Label quality + distributions.
    cats = collections.Counter(label["category"] for s in hall for label in s["labels"])
    modes = collections.Counter(s["metadata"].get("hallucination_mode") for s in hall)
    fmts = collections.Counter(s["metadata"].get("answer_style") for s in samples)
    empty_expl = sum(1 for s in hall for label in s["labels"] if not label.get("explanation", "").strip())
    nedits = [len(s["labels"]) for s in hall]
    covs = [sum(label["end"] - label["start"] for label in s["labels"]) / max(len(s["answer"]), 1) for s in hall]
    alens = sorted(len(s["answer"]) for s in samples)

    print("\nLABELS:")
    print(f"  categories: {dict(cats)}")
    print(f"  modes: {dict(modes)} | formats: {dict(fmts)}")
    print(f"  edits/sample: {statistics.mean(nedits):.2f}" if nedits else "  no hall labels")
    print(f"  empty explanations: {empty_expl}")
    if covs:
        print(f"  span coverage: median {statistics.median(covs):.0%}, max {max(covs):.0%}")
    print(f"  answer length chars: median {statistics.median(alens):.0f}, "
          f"p90 {alens[int(len(alens) * 0.9)]}, max {max(alens)}")

    if examples:
        print("\nUNGROUNDED EXAMPLES:")
        for cls, iid, refs in examples:
            print(f"  [{cls}] {iid}: {refs}")


if __name__ == "__main__":
    main()
