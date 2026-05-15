"""Assign train/dev/test splits to hallucinations.jsonl and write a single JSON file.

Each record's ``split`` field is set to ``train``, ``dev``, or ``test``.
The output is a JSON array readable by HallucinationData.from_json / evaluate.py.

Usage examples:
  python split_hallucinations.py
  python split_hallucinations.py --input data/acl_md_hal/raw/hallucinations.jsonl --output data/acl_md_hal/hallucinations.json
  python split_hallucinations.py --train 0.8 --dev 0.1 --test 0.1
  python split_hallucinations.py --seed 123 --no-shuffle
"""

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign train/dev/test splits and write a single JSON file.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/acl_md_hal/raw/hallucinations.jsonl"),
        help="Path to the source JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/acl_md_hal/hallucinations.json"),
        help="Output JSON file path.",
    )
    parser.add_argument("--train", type=float, default=0.8, help="Fraction for train split (default: 0.8).")
    parser.add_argument("--dev", type=float, default=0.1, help="Fraction for dev split (default: 0.1).")
    parser.add_argument("--test", type=float, default=0.1, help="Fraction for test split (default: 0.1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42).")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling; preserve original order.",
    )
    return parser.parse_args()


def main() -> None:
    """Read the source JSONL, assign split labels, and write a single combined JSON file."""
    args = parse_args()

    total = args.train + args.dev + args.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--train + --dev + --test must sum to 1.0, got {total:.4f}")

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not args.no_shuffle:
        random.seed(args.seed)
        random.shuffle(records)

    n = len(records)
    n_train = int(n * args.train)
    n_dev = int(n * args.dev)

    boundaries = [
        ("train", 0, n_train),
        ("dev", n_train, n_train + n_dev),
        ("test", n_train + n_dev, n),
    ]

    counts = {}
    for split_name, start, end in boundaries:
        for record in records[start:end]:
            record["split"] = split_name
        counts[split_name] = end - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {n} records → {args.output}")
    print(f"  train: {counts['train']}  dev: {counts['dev']}  test: {counts['test']}")
    print(f"  ({args.train:.0%}/{args.dev:.0%}/{args.test:.0%}, seed={args.seed}, shuffle={not args.no_shuffle})")


if __name__ == "__main__":
    main()
