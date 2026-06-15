#!/usr/bin/env python3
"""Train a token-level hallucination span detector on the v2 dataset.

Fast path: samples are tokenized once into an on-disk arrow dataset, then
trained with the HF Trainer (bf16, dynamic padding, step-based eval). The
label semantics match ``HallucinationDataset``: prompt tokens are ignored
(-100), answer tokens are 0 (supported) or 1 (overlapping an annotated span).

Usage:
    # From the published dataset
    python scripts/train_span_detector.py \
        --dataset KRLabsOrg/lettucedetect-code-hallucination \
        --model-name jhu-clsp/mmBERT-base \
        --output-dir output/mmbert_span_detector

    # From local data/v2 source dirs
    python scripts/train_span_detector.py \
        --data data/v2/code_agent --data data/v2/wikipedia \
        --model-name jhu-clsp/mmBERT-base \
        --output-dir output/mmbert_span_detector
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

SPLIT_ALIASES = {"dev": "validation"}


def load_rows(
    datasets_: list[str], data_dirs: list[str], sources: list[str] | None
) -> dict[str, list[dict]]:
    """Return ``{split: [{prompt, answer, labels}, ...]}`` merged from hub datasets and/or local dirs."""
    rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for dataset in datasets_:
        from datasets import load_dataset

        dd = load_dataset(dataset)
        for split in rows:
            if split not in dd:
                continue
            for r in dd[split]:
                if sources and r.get("dataset") not in sources:
                    continue
                labels = r.get("labels") or []
                rows[split].append({"prompt": r["prompt"], "answer": r["answer"], "labels": labels})
    for d in data_dirs:
        for path in sorted(Path(d).glob("*.jsonl")):
            if path.name.endswith(".failures.jsonl"):
                continue
            split = SPLIT_ALIASES.get(path.stem, path.stem)
            if split not in rows:
                continue
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if sources and r.get("dataset") not in sources:
                    continue
                rows[split].append(
                    {"prompt": r["prompt"], "answer": r["answer"], "labels": r.get("labels") or []}
                )
    return rows


def token_labels(
    seq_ids: list[int | None], offsets: list[tuple[int, int]], spans: list[dict]
) -> list[int]:
    """Per-token labels: -100 off the answer, else 1 iff the token overlaps a span."""
    labels = []
    for seq_id, (start, end) in zip(seq_ids, offsets):
        if seq_id != 1 or start == end:
            labels.append(-100)
            continue
        label = 0
        for span in spans:
            if end > span["start"] and start < span["end"]:
                label = 1
                break
        labels.append(label)
    return labels


def tokenize_split(
    rows: list[dict], tokenizer: AutoTokenizer, max_length: int, doc_stride: int, num_proc: int
) -> Dataset:
    """Tokenize via a cached, parallel ``.map`` (disk-cached, so reruns skip it).

    ``doc_stride`` is not supported here (kept truncation-only); add windowing back
    if a 4k encoder ever needs it.
    """
    if doc_stride > 0:
        raise SystemExit("--doc-stride not supported with cached tokenization; use 0")

    def tok(batch: dict) -> dict:
        enc = tokenizer(
            batch["prompt"], batch["answer"],
            truncation="only_first", max_length=max_length, return_offsets_mapping=True,
        )
        labels = [
            token_labels(enc.sequence_ids(i), enc["offset_mapping"][i], batch["labels"][i])
            for i in range(len(batch["prompt"]))
        ]
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    ds = Dataset.from_list(rows)
    return ds.map(tok, batched=True, num_proc=num_proc, remove_columns=ds.column_names)


def compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """Token-level precision/recall/F1 for the hallucinated class."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    pred_pos = (preds == 1) & mask
    gold_pos = (labels == 1) & mask
    tp = int((pred_pos & gold_pos).sum())
    fp = int((pred_pos & ~gold_pos).sum())
    fn = int((gold_pos & ~pred_pos).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Train a hallucination span detector (fast path).")
    ap.add_argument(
        "--dataset", action="append", default=[],
        help="HF hub dataset id (repeatable; merged across all given).",
    )
    ap.add_argument(
        "--data", action="append", default=[], help="Local data/v2 source dir (repeatable)."
    )
    ap.add_argument(
        "--sources", help="Comma-separated `dataset` field values to keep (default: all)."
    )
    ap.add_argument("--model-name", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument(
        "--doc-stride", type=int, default=0, help="Window stride over the prompt; 0 truncates."
    )
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-remote-code", action="store_true", help="Needed for e.g. EuroBERT.")
    ap.add_argument("--limit", type=int, default=0, help="Cap rows per split (0 = all). For smoke tests.")
    ap.add_argument("--num-proc", type=int, default=8, help="Tokenization workers.")
    ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint in --output-dir.")
    args = ap.parse_args()
    if not args.dataset and not args.data:  # both empty lists
        ap.error("provide --dataset and/or --data")

    sources = args.sources.split(",") if args.sources else None
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    rows = load_rows(args.dataset, args.data, sources)
    if args.limit:
        rows = {split: r[: args.limit] for split, r in rows.items()}
    print({split: len(r) for split, r in rows.items()})

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "supported", 1: "hallucinated"},
        label2id={"supported": 0, "hallucinated": 1},
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(config, "reference_compile"):
        # ModernBERT-style auto torch.compile conflicts with Trainer eval/checkpoint paths.
        config.reference_compile = False
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        config=config,
        trust_remote_code=args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    import torch

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        seed=args.seed,
        report_to="none",
    )
    # Under DDP, only rank 0 tokenizes + writes the cache; other ranks load it.
    with training_args.main_process_first():
        train_ds = tokenize_split(rows["train"], tokenizer, args.max_length, args.doc_stride, args.num_proc)
        eval_ds = tokenize_split(rows["validation"], tokenizer, args.max_length, args.doc_stride, args.num_proc)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer, padding=True),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if rows["test"]:
        with training_args.main_process_first():
            test_ds = tokenize_split(rows["test"], tokenizer, args.max_length, args.doc_stride, args.num_proc)
        print("test:", trainer.evaluate(test_ds, metric_key_prefix="test"))


if __name__ == "__main__":
    main()
