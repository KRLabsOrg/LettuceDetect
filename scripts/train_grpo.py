#!/usr/bin/env python3
"""GRPO refinement of the SFT'd span detector — reward = recall-weighted span F-beta.

Continues from the SFT model (loaded as base + a fresh LoRA) and optimizes the
ACTUAL span metric we report (char-overlap F-beta vs gold), which cross-entropy
SFT does not. beta>1 leans into recall (where we trail the span SOTA). Clean rows
reward emitting [] (so recall can't be gamed by over-flagging); unparseable -> 0.

    torchrun --nproc_per_node=4 scripts/train_grpo.py \
        --base /mnt/workspace/users/adamko/qwen_sft_merged \
        --data-dir data/generative_sft_fixed --output-dir /mnt/workspace/users/adamko/qwen_grpo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))


def char_fbeta(pred: list, gold: list, beta: float) -> float:
    """Char-overlap F-beta of predicted vs gold spans; clean rows reward emptiness."""
    pp = sum(p["end"] - p["start"] for p in pred)
    gg = sum(g["end"] - g["start"] for g in gold)
    if gg == 0:  # clean example: correct iff nothing predicted
        return 1.0 if pp == 0 else 0.0
    if pp == 0:
        return 0.0
    ov = 0
    for p in pred:
        for g in gold:
            ov += max(0, min(p["end"], g["end"]) - max(p["start"], g["start"]))
    prec, rec = ov / pp, ov / gg
    if prec + rec == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def main() -> None:
    """Run GRPO."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="SFT'd model (merged) to refine.")
    ap.add_argument("--dataset", action="append", default=[], required=True, help="HF dataset(s).")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-ragtruth", type=int, default=1500, help="GRPO prompts from ragtruth.")
    ap.add_argument("--n-code", type=int, default=1500, help="GRPO prompts from code-agent.")
    ap.add_argument("--n-other", type=int, default=300, help="GRPO prompts per OTHER source.")
    ap.add_argument("--beta-recall", type=float, default=2.0, help="F-beta for the reward (>1 = recall).")
    ap.add_argument("--kl-beta", type=float, default=0.04, help="GRPO KL coefficient (anchor to SFT).")
    ap.add_argument("--num-generations", type=int, default=6)
    ap.add_argument("--max-prompt-length", type=int, default=8192)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--resume", action="store_true", help="Resume from last checkpoint in --output-dir.")
    args = ap.parse_args()

    from unsloth import FastLanguageModel  # noqa: I001  must precede trl/transformers

    import random

    import torch
    from datasets import Dataset, load_dataset
    from trl import GRPOConfig, GRPOTrainer

    from evaluate_generative_model import parse_spans, spans_to_offsets
    from taxonomy import SYSTEM_BASE, SYSTEM_EXPL, build_user_message
    from train_generative_sft import lora_targets_for

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    extra = {}
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        extra["device_map"] = {"": local_rank}

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_prompt_length + args.max_completion_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        **extra,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=lora_targets_for(args.base),
        lora_alpha=args.lora_r * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Build GRPO rows from HF (source-weighted toward ragtruth + code-agent): each row =
    # prompt (system+user) + answer + gold span offsets (labels are already char offsets).
    explain_src = {"lettucedetect-code-agent"}
    by_src: dict[str, list] = {}
    for name in args.dataset:
        for r in load_dataset(name, split="train"):
            by_src.setdefault(r.get("dataset"), []).append(r)
    rng = random.Random(3407)
    picked = []
    for src, group in by_src.items():
        n = args.n_ragtruth if src == "ragtruth" else args.n_code if src in explain_src else args.n_other
        rng.shuffle(group)
        picked += group[: min(n, len(group))]
    rng.shuffle(picked)
    if args.limit:
        picked = picked[: args.limit]
    print(f"GRPO prompts: {len(picked)} (per-source caps rt={args.n_ragtruth} code={args.n_code} other={args.n_other})")

    def to_grpo(r: dict) -> dict:
        src = r.get("dataset")
        system = SYSTEM_EXPL if src in explain_src else SYSTEM_BASE
        prompt = [{"role": "system", "content": system}, {"role": "user", "content": build_user_message(r)}]
        gold = [{"start": lab["start"], "end": lab["end"]} for lab in (r.get("labels") or [])]
        return {"prompt": prompt, "answer": r["answer"], "gold": json.dumps(gold)}

    ds = Dataset.from_list([to_grpo(r) for r in picked])

    def reward_spans(completions: list, answer: list, gold: list, **_: object) -> list[float]:
        out = []
        for comp, ans, g in zip(completions, answer, gold):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            spans = parse_spans(text)
            pred = spans_to_offsets(ans, spans) if spans else []
            out.append(char_fbeta(pred, json.loads(g), args.beta_recall))
        return out

    cfg = GRPOConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.lr,
        beta=args.kl_beta,
        num_train_epochs=args.epochs,
        logging_steps=5,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        optim="adamw_8bit",
        temperature=1.0,
        report_to="none",
        seed=3407,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_spans],
        args=cfg,
        train_dataset=ds,
    )
    has_ckpt = bool(list(args.output_dir.glob("checkpoint-*")))
    trainer.train(resume_from_checkpoint=args.resume and has_ckpt)
    model.save_pretrained(str(args.output_dir / "lora"))
    tokenizer.save_pretrained(str(args.output_dir / "lora"))
    print(f"saved GRPO LoRA -> {args.output_dir}/lora")


def _selfcheck() -> None:
    """Offline reward check."""
    g = [{"start": 4, "end": 7}]
    assert char_fbeta([{"start": 4, "end": 7}], g, 2.0) == 1.0  # exact
    assert char_fbeta([], g, 2.0) == 0.0  # missed
    assert char_fbeta([], [], 2.0) == 1.0  # clean correct
    assert char_fbeta([{"start": 0, "end": 3}], [], 2.0) == 0.0  # clean over-flag
    # recall-weighted: an over-predicting span (full recall, half precision) scores
    # HIGHER under beta=2 (recall-weighted) than beta=0.5 (precision-weighted).
    g2 = [{"start": 0, "end": 5}]
    p2 = [{"start": 0, "end": 10}]  # full recall, half precision
    assert char_fbeta(p2, g2, 2.0) > char_fbeta(p2, g2, 0.5)
    print("selfcheck ok")


if __name__ == "__main__":
    _selfcheck() if len(sys.argv) == 1 else main()
