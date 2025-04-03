import argparse
import json
from pathlib import Path

from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
)
from tqdm.auto import tqdm

from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
    HallucinationSample,
)


def evaluate_ragas(
    samples: list[HallucinationSample],
    samples_ragas: list[HallucinationSample],
    threshold: bool = None,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Evaluate the Ragas Baseline at the example level.

    This function assumes that each sample is a dictionary containing:
      - "prompt": the prompt text.
      - "answer": the answer text.
      - "gold_spans": a list of dictionaries where each dictionary has "start" and "end" keys
                      indicating the character indices of the gold (human-labeled) span.

    If any span is present, the example is considered hallucinated. Otherwise it is supported.

    The predictions come from the RAGAS Baseline file where each sample contains a label of the following format:
        - "faithfulness": Faithfulness score by RAGAS
        - "threshold_0.4": if faithfulness < 0.4 sample is hallucinated, otherwise it is supported
        - "threshold_0.5": if faithfulness < 0.5 sample is hallucinated, otherwise it is supported
        - "threshold_0.6":if faithfulness < 0.6 sample is hallucinated, otherwise it is supported
        - "threshold_0.7": if faithfulness < 0.7 sample is hallucinated, otherwise it is supported

    :param samples: A list of samples to evaluate containing the ground truth labels.
    :param samples_ragas : A list of samples containing RAGAS generated labels
    :param threshold: Threshold user is interested in for RAGAS evaluation
    :return: A dict containing example-level metrics:
        {
            "supported": {"precision": float, "recall": float, "f1": float},
            "hallucinated": {"precision": float, "recall": float, "f1": float}
        }
    """
    example_preds: list[int] = []
    example_labels: list[int] = []

    for i, sample in enumerate(tqdm(samples, desc="Evaluating", leave=False)):
        prompt = sample.prompt
        answer = sample.answer
        gold_spans = sample.labels
        predicted_spans = samples_ragas[i].labels
        true_example_label = 1 if gold_spans else 0
        pred_example_label = 1 if predicted_spans[0][f"threshold_{threshold}"] else 0

        example_labels.append(true_example_label)
        example_preds.append(pred_example_label)

    precision, recall, f1, _ = precision_recall_fscore_support(
        example_labels, example_preds, labels=[0, 1], average=None, zero_division=0
    )

    results: dict[str, dict[str, float]] = {
        "supported": {  # Class 0
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
        },
        "hallucinated": {  # Class 1
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
        },
    }

    # Calculating AUROC
    fpr, tpr, _ = roc_curve(example_labels, example_preds)
    auroc = auc(fpr, tpr)

    results: dict[str, dict[str, float]] = {
        "supported": {  # Class 0
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
        },
        "hallucinated": {  # Class 1
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
        },
    }
    results["auroc"] = auroc

    if verbose:
        report = classification_report(
            example_labels,
            example_preds,
            target_names=["Supported", "Hallucinated"],
            digits=4,
            zero_division=0,
        )
        print("\nDetailed Example-Level Classification Report:")
        print(report)
        results["classification_report"] = report

    return results


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


def main(ground_truth_file: Path, ragas_baseline: Path, threshold):
    """Evaluates RAGAS baseline.

    :param ground_truth_file: Path to the input file.
    :param ragas_baseline: Path to the output file.
    :param threshold: Threshold user is interested in for the evaluation.
    """
    test_samples, task_type_map = load_data(ground_truth_file)
    test_samples_ragas, task_type_map_ragas = load_data(ragas_baseline)

    # Evaluate the whole dataset
    print("\nTask type: whole dataset")
    evaluate_ragas(
        test_samples,
        test_samples_ragas,
        threshold=threshold,
    )

    for task_type, samples in task_type_map.items():
        for task_type_llm, samples_llm in task_type_map_ragas.items():
            print(task_type_llm)
            print(f"\nTask type: {task_type_llm}")
            evaluate_ragas(
                test_samples,
                test_samples_ragas,
                threshold=threshold,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_file", type=str, required=True)
    parser.add_argument("--ragas_baseline", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    main(args.ground_truth_file, args.ragas_baseline, args.threshold)
