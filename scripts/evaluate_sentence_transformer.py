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

import lettucedetect.detectors.factory
from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
    HallucinationDataset,
)
from lettucedetect.models.evaluator import (
    evaluate_detector_example_level_batch,
    print_metrics,
)
from lettucedetect.models.sentece_model import SentenceModel


def evaluate_task_samples_sentence(
    samples,
    detector=None,
):
    print(f"\nEvaluating model on {len(samples)} samples")
    print("\n---- Example-Level Span Evaluation ----")
    metrics = evaluate_detector_example_level_batch(detector, samples)
    print_metrics(metrics)
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

    args = parser.parse_args()

    test_samples, task_type_map = load_data(args.data_path)

    print(f"\nEvaluating model on test samples: {len(test_samples)}")

    model = SentenceModel.from_pretrained(args.model_path)
    base_model = getattr(model.config, "model_name", "answerdotai/ModernBERT-base")

    detector = make_detector(
        method="sentencetransformer",
        model_path=base_model,
    )

    # Evaluate the whole dataset
    print("\nTask type: whole dataset")
    evaluate_task_samples_sentence(
        test_samples,
        detector=detector,
    )

    for task_type, samples in task_type_map.items():
        print(f"\nTask type: {task_type}")
        evaluate_task_samples_sentence(
            samples,
            detector=detector,
        )


if __name__ == "__main__":
    main()
