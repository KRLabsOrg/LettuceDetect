import argparse
import json
from pathlib import Path

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
)
from lettucedetect.models.evaluator import (
    evaluate_detector_char_level,
    evaluate_model,
    evaluate_model_example_level,
    print_metrics,
)
from lettucedetect.models.inference import HallucinationDetector


def evaluate_task_samples(
    samples,
    evaluation_type,
    model=None,
    tokenizer=None,
    detector=None,
    device=None,
    batch_size=8,
):
    print(f"\nEvaluating model on {len(samples)} samples")

    if evaluation_type in {"token_level", "example_level"}:
        # Prepare the dataset and dataloader
        test_dataset = HallucinationDataset(samples, tokenizer)
        data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer, label_pad_token_id=-100
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )

        eval_map = {
            "token_level": (evaluate_model, "Token-Level Evaluation"),
            "example_level": (evaluate_model_example_level, "Example-Level Evaluation"),
        }
        eval_fn, eval_title = eval_map[evaluation_type]
        print(f"\n---- {eval_title} ----")
        metrics = eval_fn(model, test_loader, device)
        print_metrics(metrics)
        return metrics

    else:  # char_level
        print("\n---- Character-Level Span Evaluation ----")
        metrics = evaluate_detector_char_level(detector, samples)
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        return metrics


def load_data(data_path):
    data_path = Path(data_path)
    hallucination_data = HallucinationData.from_json(json.loads(data_path.read_text()))

    # Filter test samples from the data
    test_samples = [sample for sample in hallucination_data.samples if sample.split == "test"]

    # group samples by task type
    task_type_map = {}
    for sample in test_samples:
        if sample.task_type not in task_type_map:
            task_type_map[sample.task_type] = []
        task_type_map[sample.task_type].append(sample)
    return test_samples, task_type_map


def main():
    parser = argparse.ArgumentParser(description="Evaluate a hallucination detection model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the evaluation data (JSON format)",
    )
    parser.add_argument(
        "--evaluation_type",
        type=str,
        default="example_level",
        help="Evaluation type (token_level, example_level or char_level)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )

    args = parser.parse_args()

    test_samples, task_type_map = load_data(args.data_path)

    print(f"\nEvaluating model on test samples: {len(test_samples)}")

    # Setup model/detector based on evaluation type
    if args.evaluation_type in {"token_level", "example_level"}:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_path, trust_remote_code=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        detector = None
    else:  # char_level
        model, tokenizer, device = None, None, None
        detector = HallucinationDetector(method="transformer", model_path=args.model_path)

    # Evaluate each task type separately
    for task_type, samples in task_type_map.items():
        print(f"\nTask type: {task_type}")
        evaluate_task_samples(
            samples,
            args.evaluation_type,
            model=model,
            tokenizer=tokenizer,
            detector=detector,
            device=device,
            batch_size=args.batch_size,
        )

    # Evaluate the whole dataset
    print("\nTask type: whole dataset")
    evaluate_task_samples(
        test_samples,
        args.evaluation_type,
        model=model,
        tokenizer=tokenizer,
        detector=detector,
        device=device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
