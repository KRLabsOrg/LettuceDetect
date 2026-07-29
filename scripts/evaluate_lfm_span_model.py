#!/usr/bin/env python3
"""Per-source evaluation for the LFM2.5 span tagger.

``LfmForSpanTagging`` is a custom module (bidirectional LFM backbone + linear
head), so it cannot be loaded through ``HallucinationDetector``'s
``AutoModelForTokenClassification`` path. This script loads it directly and
reuses ``TransformerDetector``'s prediction/span-decoding plus the shared
char-overlap metrics, so numbers are directly comparable to
``evaluate_span_model.py`` outputs.

Usage:
    python scripts/evaluate_lfm_span_model.py \
        --model-path /mnt/workspace/users/adamko/lfm25_encoder_binary \
        --dataset KRLabsOrg/lettucedetect-code-hallucination \
        --dataset KRLabsOrg/lettucedetect-prose-hallucination \
        --split test [--by dataset]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

from evaluate_span_model import load_samples  # noqa: E402
from span_eval_metrics import print_metrics_table  # noqa: E402
from train_lfm_span_detector import LfmForSpanTagging  # noqa: E402

from lettucedetect.detectors.transformer import TransformerDetector  # noqa: E402


def build_detector(
    model_path: str, backbone: str, max_length: int, device: str
) -> TransformerDetector:
    """Duck-type a TransformerDetector around the custom LFM tagger."""
    model = LfmForSpanTagging(backbone)
    state = load_file(Path(model_path) / "model.safetensors")
    model.load_state_dict(state)
    model.to(device).eval()

    det = TransformerDetector.__new__(TransformerDetector)
    det.model = model
    det.tokenizer = AutoTokenizer.from_pretrained(model_path)
    det.device = torch.device(device)
    det.max_length = max_length
    det.lang = "en"
    det.typer = None
    return det


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Per-source LFM span-tagger evaluation.")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--backbone", default="LiquidAI/LFM2.5-Encoder-350M")
    ap.add_argument("--dataset", action="append", default=[], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--by", choices=["dataset", "language"], default="dataset")
    ap.add_argument("--only", default="", help="Keep only rows whose `dataset` field == this.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    detector = build_detector(args.model_path, args.backbone, args.max_length, args.device)
    samples = load_samples(args.dataset, args.split, args.limit, args.only)

    from tqdm import tqdm

    rows = []
    with torch.no_grad():
        for s in tqdm(samples, desc="predict"):
            pred = detector.predict_prompt(s.prompt, s.answer, output_format="spans")
            rows.append((getattr(s, args.by), s.labels, pred))

    print_metrics_table(rows)


if __name__ == "__main__":
    main()
