#!/usr/bin/env python3
"""Train a token-level span detector on the LFM2.5 bidirectional encoder backbone.

The LFM2.5 retriever ships a bidirectional ``Lfm2BidirectionalModel`` (via
``trust_remote_code``) but no token-classification head, so we wrap the backbone
(1024-dim per-token ``last_hidden_state``) with a linear tagger. Everything else
— row loading, label alignment, tokenization, metrics — is reused verbatim from
``train_span_detector.py`` so this stays a thin backbone swap.

    torchrun --nproc_per_node=2 scripts/train_lfm_span_detector.py \
        --dataset KRLabsOrg/lettucedetect-code-hallucination \
        --dataset KRLabsOrg/lettucedetect-prose-hallucination \
        --model-name LiquidAI/LFM2.5-ColBERT-350M \
        --output-dir /mnt/workspace/users/adamko/lfm_span_detector
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import TokenClassifierOutput

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from train_span_detector import compute_metrics, load_rows, tokenize_split  # noqa: E402


class LfmForSpanTagging(nn.Module):
    """LFM2.5 bidirectional backbone + linear token-classification head."""

    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.1) -> None:
        """Load the remote bidirectional backbone and attach a tagger head."""
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float32
        )
        self.config = self.backbone.config
        hidden = self.config.hidden_size
        # LayerNorm the backbone features before the head: the retriever's raw
        # hidden states are not scaled for classification (large magnitude ->
        # saturated logits, exploding grads), so normalize them first.
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

    def gradient_checkpointing_enable(self, **kwargs: object) -> None:
        """Forward gradient-checkpointing to the backbone if it supports it."""
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **_: object,
    ) -> TokenClassifierOutput:
        """Per-token logits over {supported, hallucinated}; CE loss masks -100."""
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.norm(hidden.to(self.classifier.weight.dtype))
        logits = self.classifier(self.dropout(hidden))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.num_labels).float(), labels.view(-1), ignore_index=-100
            )
        return TokenClassifierOutput(loss=loss, logits=logits)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Train an LFM2.5-encoder span detector.")
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--data", action="append", default=[])
    ap.add_argument("--sources")
    ap.add_argument("--model-name", default="LiquidAI/LFM2.5-ColBERT-350M")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-proc", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not args.dataset and not args.data:
        ap.error("provide --dataset and/or --data")

    sources = args.sources.split(",") if args.sources else None
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=True
    )
    rows = load_rows(args.dataset, args.data, sources)
    if args.limit:
        rows = {split: r[: args.limit] for split, r in rows.items()}
    print({split: len(r) for split, r in rows.items()})

    model = LfmForSpanTagging(args.model_name)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.06,
        max_grad_norm=1.0,
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
    with training_args.main_process_first():
        train_ds = tokenize_split(rows["train"], tokenizer, args.max_length, 0, args.num_proc)
        eval_ds = tokenize_split(rows["validation"], tokenizer, args.max_length, 0, args.num_proc)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer, padding=True),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    has_ckpt = bool(list(Path(args.output_dir).glob("checkpoint-*")))
    trainer.train(resume_from_checkpoint=args.resume and has_ckpt)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if rows["test"]:
        with training_args.main_process_first():
            test_ds = tokenize_split(rows["test"], tokenizer, args.max_length, 0, args.num_proc)
        print("test:", trainer.evaluate(test_ds, metric_key_prefix="test"))


def _selfcheck() -> None:
    """Shape/loss sanity check for the head (no model download)."""
    head = nn.Linear(8, 2)
    logits = head(torch.randn(2, 5, 8))
    labels = torch.tensor([[0, 1, -100, 1, 0], [-100, 0, 1, 1, -100]])
    loss = nn.functional.cross_entropy(
        logits.view(-1, 2).float(), labels.view(-1), ignore_index=-100
    )
    assert loss.requires_grad and loss.item() > 0
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck() if len(sys.argv) == 1 else main()
