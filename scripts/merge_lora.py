#!/usr/bin/env python3
"""Merge a trained LoRA adapter into the base model -> standalone 16-bit model.

Produces a plain HF checkpoint (safetensors) that vLLM can serve and that can be
pushed to the Hub. Run on one GPU (no torchrun).

    CUDA_VISIBLE_DEVICES=0 python scripts/merge_lora.py \
        --adapter /mnt/workspace/users/adamko/lfm2_sft/lora \
        --out /mnt/workspace/users/adamko/lfm2_sft_merged
    # optional: --push-to KRLabsOrg/lettucedetect-lfm2-8b   (needs HF_TOKEN)
"""

from __future__ import annotations

import argparse

from unsloth import FastLanguageModel


def main() -> None:
    """Merge and save / push."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, help="Path to the saved LoRA adapter dir.")
    ap.add_argument("--out", help="Local dir for the merged 16-bit model.")
    ap.add_argument("--push-to", help="HF repo id to push the merged model to.")
    ap.add_argument("--max-seq-length", type=int, default=32768)
    args = ap.parse_args()
    if not (args.out or args.push_to):
        ap.error("give --out and/or --push-to")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
    )
    if args.out:
        model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
        print(f"merged 16-bit model -> {args.out}")
    if args.push_to:
        model.push_to_hub_merged(args.push_to, tokenizer, save_method="merged_16bit")
        print(f"pushed -> {args.push_to}")


if __name__ == "__main__":
    main()
