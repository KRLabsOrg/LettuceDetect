import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
    HallucinationDataset,
    HallucinationSample,
)
from lettucedetect.models.trainer import Trainer


def set_seed(seed: int = 42):
    """Set all seeds for reproducibility.

    Args:
        seed: The seed to use

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For complete determinism at cost of performance:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train hallucination detector model")
    parser.add_argument(
        "--ragtruth-path",
        type=str,
        default="data/ragtruth/ragtruth_data.json",
        help="Path to the training data JSON file",
    )
    parser.add_argument(
        "--ragbench-path",
        type=str,
        help="Optional path to the RAGBench training data JSON file. If not provided, only RAGTruth data will be used.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Name or path of the pretrained model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/hallucination_detector",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training and testing"
    )
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Learning rate for training"
    )
    return parser.parse_args()


def split_train_dev(
    samples: list[HallucinationSample], dev_ratio: float = 0.1, seed: int = 42
) -> tuple[list[HallucinationSample], list[HallucinationSample]]:
    """Split the samples into train and dev sets.

    :param samples: List of HallucinationSample objects.
    :param dev_ratio: Ratio of the dev set.
    :param seed: Seed for the random number generator.
    :return: Tuple of train and dev sets.
    """
    random.seed(seed)
    random.shuffle(samples)
    dev_size = int(len(samples) * dev_ratio)
    train_samples = samples[:-dev_size]
    dev_samples = samples[-dev_size:]
    return train_samples, dev_samples


def main():
    # Set seeds for reproducibility
    set_seed(123)

    args = parse_args()
    ragtruth_path = Path(args.ragtruth_path)
    ragtruth_data = HallucinationData.from_json(json.loads(ragtruth_path.read_text()))
    ragtruth_train_samples = [sample for sample in ragtruth_data.samples if sample.split == "train"]
    ragtruth_train_samples, ragtruth_dev_samples = split_train_dev(ragtruth_train_samples)

    train_samples = ragtruth_train_samples
    dev_samples = ragtruth_dev_samples

    if args.ragbench_path:
        print(f"Loading RAGBench data from {args.ragbench_path}")
        ragbench_path = Path(args.ragbench_path)
        ragbench_data = HallucinationData.from_json(json.loads(ragbench_path.read_text()))
        ragbench_train_samples = [
            sample for sample in ragbench_data.samples if sample.split == "train"
        ]
        ragbench_dev_samples = [sample for sample in ragbench_data.samples if sample.split == "dev"]

        print(f"Adding {len(ragbench_train_samples)} RAGBench train samples")
        train_samples.extend(ragbench_train_samples)
        dev_samples.extend(ragbench_dev_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)

    train_dataset = HallucinationDataset(train_samples, tokenizer)
    dev_dataset = HallucinationDataset(dev_samples, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=2, trust_remote_code=True
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        test_loader=dev_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=args.output_dir,
    )

    trainer.train()


if __name__ == "__main__":
    main()
