import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from lettucedetect.datasets.hallucination_dataset import HallucinationData


def analyze_token_distribution(samples, tokenizer):
    token_counts = []

    for sample in samples:
        # Combine prompt and answer
        full_text = f"{sample.prompt}\n{sample.answer}"

        # Tokenize
        tokens = tokenizer.encode(full_text)
        token_counts.append(len(tokens))

    # Calculate statistics
    stats = {
        "mean": np.mean(token_counts),
        "median": np.median(token_counts),
        "std": np.std(token_counts),
        "min": np.min(token_counts),
        "max": np.max(token_counts),
        "percentile_90": np.percentile(token_counts, 90),
        "percentile_95": np.percentile(token_counts, 95),
        "total_samples": len(token_counts),
    }

    return token_counts, stats


def main():
    parser = argparse.ArgumentParser(description="Analyze token distribution in the dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data (JSON format)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Name or path of the tokenizer to use",
    )
    args = parser.parse_args()

    # Load data
    data_path = Path(args.data_path)
    hallucination_data = HallucinationData.from_json(json.loads(data_path.read_text()))

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Analyze all samples
    print("\nAnalyzing token distribution for all samples...")
    token_counts, stats = analyze_token_distribution(hallucination_data.samples, tokenizer)

    # Print results
    print("\nToken Distribution Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Mean tokens: {stats['mean']:.1f}")
    print(f"Median tokens: {stats['median']:.1f}")
    print(f"Standard deviation: {stats['std']:.1f}")
    print(f"Min tokens: {stats['min']}")
    print(f"Max tokens: {stats['max']}")
    print(f"90th percentile: {stats['percentile_90']:.1f}")
    print(f"95th percentile: {stats['percentile_95']:.1f}")

    # Print distribution by split
    for split in ["train", "validation", "test"]:
        split_samples = [s for s in hallucination_data.samples if s.split == split]
        if split_samples:
            print(f"\n{split.capitalize()} split:")
            _, split_stats = analyze_token_distribution(split_samples, tokenizer)
            print(f"Samples: {split_stats['total_samples']}")
            print(f"Mean tokens: {split_stats['mean']:.1f}")
            print(f"Median tokens: {split_stats['median']:.1f}")
            print(f"90th percentile: {split_stats['percentile_90']:.1f}")


if __name__ == "__main__":
    main()
