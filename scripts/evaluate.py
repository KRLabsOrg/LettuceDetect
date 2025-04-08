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
        metrics = evaluate_detector_char_level(detector, samples)[0]
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        return metrics


def evaluate_task_samples_llm(
    samples, evaluation_type, detector, samples_llm, baseline_file_exists
):
    print(f"\nEvaluating model on {len(samples)} samples")

    if evaluation_type == "example_level":
        print("\n---- Example-Level Span Evaluation ----")
        metrics, hallucination_data_llm = evaluate_detector_example_level(
            detector, samples, samples_llm, baseline_file_exists
        )
        print_metrics(metrics)
        return metrics, hallucination_data_llm
    elif evaluation_type == "char_level":
        print("\n---- Character-Level Span Evaluation ----")
        metrics, hallucination_data_llm = evaluate_detector_char_level(
            detector, samples, samples_llm, baseline_file_exists
        )
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        return metrics, hallucination_data_llm
    else:
        raise ValueError(
            "This evaluation type is not available for this method. Use either 'example_level' or 'char_level'."
        )


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


def save_baseline_data(data_path_llm, hallucination_data_llm):
    """This function saves the LLM baseline data into a file."""
    data_path_llm = Path(data_path_llm)
    (data_path_llm).write_text(json.dumps(hallucination_data_llm.to_json(), indent=4))


def exists_baseline_data(data_path, data_path_llm):
    """This function checks whether there is already an existing file containing LLM labels."""
    data_path = Path(data_path)
    data_path_llm = Path(data_path_llm)

    if data_path_llm.exists() and data_path_llm.is_file():
        hallucination_data = HallucinationData.from_json(json.loads(data_path.read_text()))
        hallucination_data_llm = HallucinationData.from_json(json.loads(data_path_llm.read_text()))
        if len(hallucination_data.samples) == len(hallucination_data_llm.samples):
            return True
        else:
            return False
    else:
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate a hallucination detection model")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Detector method. Choose either 'transformer' or 'llm'.",
    )
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

    parser.add_argument(
        "--data_path_llm",
        type=int,
        default=None,
        help="Path to LLM baseline data (JSON Format)",
    )

    args = parser.parse_args()

    test_samples, task_type_map = load_data(args.data_path)

    baseline_file_exists = exists_baseline_data(args.data_path, args.data_path_llm)

    print(f"\nEvaluating model on test samples: {len(test_samples)}")

    # Setup model/detector based on evaluation type
    if args.method == "transformer":
        if args.evaluation_type in {"token_level", "example_level"}:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AutoModelForTokenClassification.from_pretrained(
                args.model_path, trust_remote_code=True
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            detector = None
        else:  # char_level
            model, tokenizer, device = None, None, None
            detector = HallucinationDetector(method=args.method, model_path=args.model_path)

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

    elif args.method == "llm":
        if baseline_file_exists:
            test_samples_llm, task_type_map_llm = load_data(args.data_path_llm)
        else:
            test_samples_llm, task_type_map_llm = None, None
        model, tokenizer, device = None, None, None
        detector = HallucinationDetector(method=args.method)
        samples, samples_llm = (test_samples, test_samples_llm)

        # Evaluate the whole dataset
        print("\nTask type: whole dataset")
        metrics, hallucination_data_llm = evaluate_task_samples_llm(
            samples,
            args.evaluation_type,
            detector=detector,
            samples_llm=samples_llm,
            baseline_file_exists=baseline_file_exists,
        )

        if not baseline_file_exists:
            save_baseline_data(args.data_path_llm, hallucination_data_llm)

        test_samples_llm, task_type_map_llm = load_data(args.data_path_llm)

        for task_type, samples in task_type_map.items():
            for task_type_llm, samples_llm in task_type_map_llm.items():
                print(f"\nTask type: {task_type}")
                evaluate_task_samples_llm(
                    samples,
                    args.evaluation_type,
                    detector=detector,
                    samples_llm=samples_llm,
                    baseline_file_exists=True,
                )

    else:
        raise ValueError("Unsupported method. Choose 'transformer' or 'llm'.")


if __name__ == "__main__":
    main()
