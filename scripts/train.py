import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
)

from lettucedetect.datasets.hallucination_dataset import HallucinationData, HallucinationDataset
from lettucedetect.models.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train hallucination detector model")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ragtruth/ragtruth_data.json",
        help="Path to the training data JSON file",
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


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    hallucination_data = HallucinationData.from_json(json.loads(data_path.read_text()))

    train_samples = [sample for sample in hallucination_data.samples if sample.split == "train"]
    test_samples = [sample for sample in hallucination_data.samples if sample.split == "test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, label_pad_token_id=-100)

    train_dataset = HallucinationDataset(train_samples, tokenizer)
    test_dataset = HallucinationDataset(test_samples, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=2)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_path=args.output_dir,
    )

    trainer.train()


if __name__ == "__main__":
    main()
