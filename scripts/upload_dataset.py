#!/usr/bin/env python3
"""Script to upload preprocessed LettuceDetect datasets to Hugging Face Hub."""

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict

from lettucedetect.datasets.hallucination_dataset import HallucinationData

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dataset_uploader")


def convert_to_hf_dataset(hallucination_data: HallucinationData) -> DatasetDict:
    """Convert LettuceDetect's HallucinationData to a Hugging Face DatasetDict.

    :param hallucination_data: The HallucinationData to convert
    :return: A DatasetDict with train, validation, and test splits
    """
    train_samples = []
    dev_samples = []
    test_samples = []

    for sample in hallucination_data.samples:
        sample_dict = {
            "prompt": sample.prompt,
            "answer": sample.answer,
            "labels": sample.labels,
            "task_type": sample.task_type,
            "dataset": sample.dataset,
            "language": sample.language,
        }

        if sample.split == "train":
            train_samples.append(sample_dict)
        elif sample.split == "dev":
            dev_samples.append(sample_dict)
        elif sample.split == "test":
            test_samples.append(sample_dict)

    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(train_samples),
            "dev": Dataset.from_list(dev_samples),
            "test": Dataset.from_list(test_samples),
        }
    )

    return dataset_dict


def upload_dataset(
    input_path: Path,
    dataset_name: str,
    push_to_hub: bool = False,
    repository_id: str = None,
    private: bool = False,
    token: str = None,
) -> DatasetDict:
    """Upload a dataset to the Hugging Face Hub.

    :param input_path: Path to the JSON file containing the dataset
    :param dataset_name: Name of the dataset (used in logging and for repository name if not specified)
    :param push_to_hub: Whether to push the dataset to the Hugging Face Hub
    :param repository_id: Repository ID on Hugging Face (e.g., "username/dataset-name")
    :param private: Whether the repository should be private
    :param token: Authentication token for Hugging Face
    :return: The DatasetDict that was created
    """
    logger.info(f"Loading dataset from {input_path}")

    # Load the dataset
    try:
        data = json.loads(input_path.read_text())
        hallucination_data = HallucinationData.from_json(data)
        logger.info(f"Loaded {len(hallucination_data.samples)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    logger.info("Converting to HF Dataset format")
    dataset_dict = convert_to_hf_dataset(hallucination_data)

    for split, dataset in dataset_dict.items():
        logger.info(f"Split '{split}': {len(dataset)} samples")

        # Skip empty datasets
        if len(dataset) == 0:
            logger.info(f"  Split '{split}' is empty, skipping statistics")
            continue

        languages = dataset.unique("language")
        for lang in languages:
            count = len([s for s in dataset if s["language"] == lang])
            logger.info(f"  Language '{lang}': {count} samples")

        datasets = dataset.unique("dataset")
        for ds in datasets:
            count = len([s for s in dataset if s["dataset"] == ds])
            logger.info(f"  Dataset '{ds}': {count} samples")

    # Filter out empty splits before pushing to Hub
    non_empty_splits = {k: v for k, v in dataset_dict.items() if len(v) > 0}
    if len(non_empty_splits) != len(dataset_dict):
        logger.info(f"Filtered out empty splits. Keeping {len(non_empty_splits)} non-empty splits.")
        dataset_dict = DatasetDict(non_empty_splits)

    if push_to_hub:
        if not repository_id:
            repository_id = f"lettucedetect/{dataset_name}"

        logger.info(f"Pushing dataset to Hugging Face Hub: {repository_id}")
        dataset_dict.push_to_hub(
            repository_id,
            private=private,
            token=token,
        )
        logger.info("Dataset successfully uploaded to Hugging Face Hub")

    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="Upload a dataset to the Hugging Face Hub")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the JSON file containing the preprocessed dataset",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name for the dataset (used for repository name if not specified)",
    )
    parser.add_argument(
        "--repository-id",
        type=str,
        help="Repository ID on Hugging Face (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--push", action="store_true", help="Push the dataset to the Hugging Face Hub"
    )
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    parser.add_argument("--token", type=str, help="Authentication token for Hugging Face")

    args = parser.parse_args()

    upload_dataset(
        input_path=Path(args.input_path),
        dataset_name=args.dataset_name,
        push_to_hub=args.push,
        repository_id=args.repository_id,
        private=args.private,
        token=args.token,
    )


if __name__ == "__main__":
    main()
