import torch
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
)
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lettucedetect.datasets.hallucination_dataset import HallucinationSample
from lettucedetect.models.inference import HallucinationDetector


def evaluate_model(
    model: Module,
    dataloader: DataLoader,
    device: torch.device,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Evaluate a model for hallucination detection.

    :param model: The model to evaluate.
    :param dataloader: The data loader to use for evaluation.
    :param device: The device to use for evaluation.
    :param verbose: If True, print the evaluation metrics.
    :return: A dictionary containing the evaluation metrics.
        {
            "supported": {"precision": float, "recall": float, "f1": float},
            "hallucinated": {"precision": float, "recall": float, "f1": float}
        }
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            logits: torch.Tensor = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # Only evaluate on tokens that have labels (not -100)
            mask = batch["labels"] != -100
            predictions = predictions[mask].cpu().numpy()
            labels = batch["labels"][mask].cpu().numpy()

            all_preds.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], average=None
    )

    # Calculating AUROC
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
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
            all_labels, all_preds, target_names=["Supported", "Hallucinated"], digits=4
        )
        print("\nDetailed Classification Report:")
        print(report)
        results["classification_report"] = report

    return results


def print_metrics(metrics: dict[str, dict[str, float]]) -> None:
    """Print evaluation metrics in a readable format.

    :param metrics: A dictionary containing the evaluation metrics.
    :return: None
    """
    print("\nEvaluation Results:")
    print("\nHallucination Detection (Class 1):")
    print(f"  Precision: {metrics['hallucinated']['precision']:.4f}")
    print(f"  Recall: {metrics['hallucinated']['recall']:.4f}")
    print(f"  F1: {metrics['hallucinated']['f1']:.4f}")

    print("\nSupported Content (Class 0):")
    print(f"  Precision: {metrics['supported']['precision']:.4f}")
    print(f"  Recall: {metrics['supported']['recall']:.4f}")
    print(f"  F1: {metrics['supported']['f1']:.4f}")

    print(f"\nAUROC: {metrics['auroc']:.4f}")


def evaluate_model_example_level(
    model: Module,
    dataloader: DataLoader,
    device: torch.device,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Evaluate a model for hallucination detection at the example level.

    For each example, if any token is marked as hallucinated (label=1),
    then the whole example is considered hallucinated. Otherwise, it is supported.

    :param model: The model to evaluate.
    :param dataloader: DataLoader providing the evaluation batches.
    :param device: Device on which to perform evaluation.
    :param verbose: If True, prints a detailed classification report.

    :return: A dict containing example-level metrics:
        {
            "supported": {"precision": float, "recall": float, "f1": float},
            "hallucinated": {"precision": float, "recall": float, "f1": float}
        }
    """
    model.eval()
    example_preds: list[int] = []
    example_labels: list[int] = []
    example_probs: list[float] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating (Example Level)", leave=False):
            # Move inputs to device. Note that `batch["labels"]`
            # can stay on CPU if you wish to avoid unnecessary transfers.
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits: torch.Tensor = outputs.logits  # Shape: [batch_size, seq_len, num_labels]
            predictions: torch.Tensor = torch.argmax(logits, dim=-1)  # Shape: [batch_size, seq_len]
            probs = torch.softmax(logits, dim=-1)

            # Process each example in the batch separately.
            for i in range(batch["labels"].size(0)):
                sample_labels = batch["labels"][i]  # [seq_len]
                sample_preds = predictions[i].cpu()  # [seq_len]
                valid_mask = sample_labels != -100

                if valid_mask.sum().item() == 0:
                    true_example_label = 0
                    pred_example_label = 0
                    # Add a default probability score
                    max_prob = 0.0
                else:
                    # Apply the valid mask and bring labels to CPU if needed.
                    sample_labels = sample_labels[valid_mask].cpu()
                    sample_preds = sample_preds[valid_mask]
                    sample_probs = probs[i][valid_mask]

                    # If any token in the sample is hallucinated (1), consider the whole sample hallucinated.
                    true_example_label = 1 if (sample_labels == 1).any().item() else 0
                    pred_example_label = 1 if (sample_preds == 1).any().item() else 0
                    # Get the max probability for class 1 (hallucinated)
                    max_prob = sample_probs[:, 1].max().item()

                example_labels.append(true_example_label)
                example_preds.append(pred_example_label)
                example_probs.append(max_prob)

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
    fpr, tpr, _ = roc_curve(example_labels, example_probs)
    auroc = auc(fpr, tpr)
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


def evaluate_detector_char_level(
    detector: HallucinationDetector, samples: list[HallucinationSample]
) -> dict[str, float]:
    """Evaluate the HallucinationDetector at the character level.

    This function assumes that each sample is a dictionary containing:
      - "prompt": the prompt text.
      - "answer": the answer text.
      - "gold_spans": a list of dictionaries where each dictionary has "start" and "end" keys
                      indicating the character indices of the gold (human-labeled) span.

    It uses the detector (which should have been initialized with the appropriate model)
    to obtain predicted spans, compares those spans with the gold spans, and computes global
    precision, recall, and F1 based on character overlap.

    :param detector: The detector to evaluate.
    :param samples: A list of samples to evaluate.
    :return: A dictionary with global metrics: {"char_precision": ..., "char_recall": ..., "char_f1": ...}
    """
    total_overlap = 0
    total_predicted = 0
    total_gold = 0

    for sample in tqdm(samples, desc="Evaluating", leave=False):
        prompt = sample.prompt
        answer = sample.answer
        gold_spans = sample.labels
        predicted_spans = detector.predict_prompt(prompt, answer, output_format="spans")

        # Compute total predicted span length for this sample.
        sample_predicted_length = sum(pred["end"] - pred["start"] for pred in predicted_spans)
        total_predicted += sample_predicted_length

        # Compute total gold span length once for this sample.
        sample_gold_length = sum(gold["end"] - gold["start"] for gold in gold_spans)
        total_gold += sample_gold_length

        # Now, compute the overlap between each predicted span and each gold span.
        sample_overlap = 0
        for pred in predicted_spans:
            for gold in gold_spans:
                overlap_start = max(pred["start"], gold["start"])
                overlap_end = min(pred["end"], gold["end"])
                if overlap_end > overlap_start:
                    sample_overlap += overlap_end - overlap_start
        total_overlap += sample_overlap

    precision = total_overlap / total_predicted if total_predicted > 0 else 0
    recall = total_overlap / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}
