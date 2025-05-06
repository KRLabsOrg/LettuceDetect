import argparse
import json
from pathlib import Path

from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
    HallucinationSample,
)
from lettucedetect.models.evaluator import (
    evaluate_detector_char_level,
    evaluate_detector_example_level,
    print_metrics,
)
from lettucedetect.models.inference import HallucinationDetector


def evaluate_task_samples_llm(
    samples: list[HallucinationSample], evaluation_type: str, detector: HallucinationDetector
):
    """Evaluate the model on the samples.

    :param samples: list of samples to evaluate
    :param evaluation_type: evaluation type (example_level or char_level)
    :param detector: detector to use
    :return: metrics and hallucination data
    """
    print(f"\nEvaluating model on {len(samples)} samples")

    if evaluation_type == "example_level":
        print("\n---- Example-Level Span Evaluation ----")
        metrics = evaluate_detector_example_level(detector, samples)
        print_metrics(metrics)
        return metrics
    elif evaluation_type == "char_level":
        print("\n---- Character-Level Span Evaluation ----")
        metrics = evaluate_detector_char_level(detector, samples)
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        return metrics
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a hallucination detection model based on LLM"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name to evaluate, e.g. 'gpt-4o'"
    )
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
        help="Evaluation type (example_level or char_level)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language of the evaluation data",
    )
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="Whether to use zero-shot prompting",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="Path to the cache file",
    )

    args = parser.parse_args()

    test_samples, task_type_map = load_data(args.data_path)

    print(f"\nEvaluating model on test samples: {len(test_samples)}")

    detector = HallucinationDetector(
        method="llm",
        lang=args.lang,
        cache_file=args.cache_path,
        model=args.model,
        zero_shot=args.zero_shot,
    )

    # Evaluate the whole dataset
    print("\nTask type: whole dataset")
    evaluate_task_samples_llm(
        test_samples,
        args.evaluation_type,
        detector=detector,
    )

    for task_type, samples in task_type_map.items():
        print(f"\nTask type: {task_type}")
        evaluate_task_samples_llm(
            samples,
            args.evaluation_type,
            detector=detector,
        )


if __name__ == "__main__":
    main()
