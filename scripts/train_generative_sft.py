#!/usr/bin/env python3
"""LoRA SFT of LFM2.5-8B-A1B as a generative hallucination-span detector (Unsloth).

Trains on the chat data from build_generative_sft.py (system + user + assistant,
assistant = the {"hallucinated_spans": [...]} JSON). Loss is masked to the
assistant turn only (train_on_responses_only), mirroring the mmBERT prompt mask.

Single-GPU or multi-GPU: launch with torchrun and DDP auto-enables, e.g.
    torchrun --nproc_per_node=4 scripts/train_generative_sft.py \
        --data-dir data/generative_sft --output-dir output/lfm2_sft
Smoke test on one GPU:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_generative_sft.py \
        --data-dir data/generative_sft --output-dir /tmp/lfm_smoke --limit 64 --epochs 1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# LFM2 module names: in_proj/out_proj (attention), w1/w2/w3 (gated MLP / experts).
# LoRA targets differ by architecture. LFM2: in_proj/out_proj + w1/w2/w3.
# Qwen3.5 is hybrid (Gated DeltaNet + attention) -> DeltaNet in_proj_* + attn + MLP.
LORA_TARGETS = {
    "lfm": ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"],
    "qwen": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "default": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def lora_targets_for(model_name: str) -> list[str]:
    """Pick LoRA target modules by model family."""
    n = model_name.lower()
    if "qwen" in n:
        return LORA_TARGETS["qwen"]
    if "lfm" in n:
        return LORA_TARGETS["lfm"]
    return LORA_TARGETS["default"]


def parse_args() -> argparse.Namespace:
    """CLI args."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", default="unsloth/LFM2.5-8B-A1B")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--per-device-train-batch-size", type=int, default=2)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--num-proc", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="Cap rows per split (smoke test).")
    ap.add_argument("--lora-targets", default="", help="Comma-sep LoRA modules (default: by family).")
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def main() -> None:
    """Train the LoRA span detector."""
    # Unsloth must be imported before transformers/trl so its patches apply.
    from unsloth import FastLanguageModel  # noqa: I001
    from unsloth.chat_templates import train_on_responses_only

    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    args = parse_args()

    # Bind this rank to its own GPU before loading, so Unsloth doesn't pile every
    # rank onto cuda:0 (unsloth#3942). All GPUs stay visible so DDP's device_ids
    # = [local_rank] indexing remains valid.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    extra = {}
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        extra["device_map"] = {"": local_rank}

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        **extra,
    )
    targets = args.lora_targets.split(",") if args.lora_targets else lora_targets_for(args.model_name)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=targets,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    splits = {"train": "train.jsonl", "validation": "validation.jsonl"}
    files = {k: str(args.data_dir / v) for k, v in splits.items()}
    ds = load_dataset("json", data_files=files)
    if args.limit:
        ds = {k: ds[k].select(range(min(args.limit, len(ds[k])))) for k in ds}

    def to_text(batch: dict) -> dict:
        texts = tokenizer.apply_chat_template(
            batch["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": [t.removeprefix(tokenizer.bos_token) for t in texts]}

    ds = {k: ds[k].map(to_text, batched=True, num_proc=args.num_proc) for k in ds}

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            packing=False,
            per_device_train_batch_size=args.per_device_train_batch_size,
            # Long-context eval OOMs when the Trainer materializes full
            # [batch, seq, vocab] logits (Accelerate upcasts them to fp32).
            # prediction_loss_only + tiny eval batch + CPU offload fixes it.
            per_device_eval_batch_size=1,
            prediction_loss_only=True,
            eval_accumulation_steps=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=0.03,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            bf16=True,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            save_total_limit=3,
            seed=3407,
            output_dir=str(args.output_dir),
            report_to="none",
        ),
    )
    # Mask loss to the assistant turn only (LFM chat markers).
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    has_ckpt = bool(list(args.output_dir.glob("checkpoint-*")))
    trainer.train(resume_from_checkpoint=args.resume and has_ckpt)

    model.save_pretrained(str(args.output_dir / "lora"))
    tokenizer.save_pretrained(str(args.output_dir / "lora"))


if __name__ == "__main__":
    main()
