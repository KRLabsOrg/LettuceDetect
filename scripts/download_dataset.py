"""Script to download LettuceDetect datasets from Hugging Face Hub to JSON format."""

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dataset_downloader")


def download_dataset(
    repository_id: str,
    output_path: Path,
    token: str = None,
) -> None:
    """Download a dataset from the Hugging Face Hub and save it as JSON.

    :param repository_id: Repository ID on Hugging Face (e.g., "username/dataset-name")
    :param output_path: Path where to save the JSON file
    :param token: Authentication token for Hugging Face
    """
    logger.info(f"Downloading dataset from Hugging Face Hub: {repository_id}")

    # Load the dataset from Hugging Face Hub
    dataset_dict = load_dataset(repository_id, token=token)
    logger.info("Dataset successfully downloaded from Hugging Face Hub")

    # Convert back to the LettuceDetect format
    samples = []

    for split_name, split_dataset in dataset_dict.items():
        for sample in split_dataset:
            samples.append(
                {
                    "prompt": sample["prompt"],
                    "answer": sample["answer"],
                    "labels": sample["labels"],
                    "task_type": sample["task_type"],
                    "dataset": sample["dataset"],
                    "language": sample["language"],
                    "split": split_name if split_name != "dev" else "dev",
                }
            )

    # Save to JSON (as a list of samples, not wrapped in a "samples" field)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved dataset to {output_path} with {len(samples)} samples")


def main():
    parser = argparse.ArgumentParser(description="Download a dataset from the Hugging Face Hub")
    parser.add_argument(
        "--repository-id",
        type=str,
        required=True,
        help="Repository ID on Hugging Face (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path where to save the JSON file",
    )
    parser.add_argument("--token", type=str, help="Authentication token for Hugging Face")

    args = parser.parse_args()

    download_dataset(
        repository_id=args.repository_id,
        output_path=Path(args.output_path),
        token=args.token,
    )


if __name__ == "__main__":
    main()
