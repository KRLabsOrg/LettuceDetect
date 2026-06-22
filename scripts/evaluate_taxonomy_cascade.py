#!/usr/bin/env python3
"""Cascade typed-span eval: encoder detector + taxonomy head.

The mmBERT binary detector finds hallucinated spans; the label-conditioned taxonomy
head then types each predicted span (category + subcategory). Char-overlap typed-F1
is reported per source via the shared scorer, so the encoder+head cascade is directly
comparable to the generative model's typed numbers.

    python scripts/evaluate_taxonomy_cascade.py \
        --detector <mmbert_binary_detector_dir> \
        --head <supervised_taxonomy_head_dir> \
        --dataset KRLabsOrg/lettucedetect-code-hallucination \
        --dataset KRLabsOrg/lettucedetect-prose-hallucination \
        --split test --by dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

SEP = " [CTX] "  # must match scripts/train_taxonomy_head.py


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Encoder detector + taxonomy-head cascade eval.")
    ap.add_argument("--detector", required=True, help="mmBERT binary span detector dir.")
    ap.add_argument("--head", required=True, help="Supervised taxonomy-head encoder dir.")
    ap.add_argument("--dataset", action="append", default=[], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--by", choices=["dataset", "language"], default="dataset")
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=8192, help="Detector max length.")
    ap.add_argument("--head-max-length", type=int, default=1024, help="Head encoder max length.")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from span_eval_metrics import print_metrics_table
    from taxonomy import CATEGORY_DESCRIPTIONS as CAT_DESC
    from taxonomy import SUBCATEGORY_DESCRIPTIONS as SUB_DESC
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    from lettucedetect.models.inference import HallucinationDetector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = HallucinationDetector(method="transformer", model_path=args.detector, max_length=args.max_length)

    tok = AutoTokenizer.from_pretrained(args.head)
    enc = AutoModel.from_pretrained(args.head, dtype=torch.bfloat16).to(device).eval()
    cat_names, sub_names = list(CAT_DESC), list(SUB_DESC)

    @torch.no_grad()
    def label_vecs(names: list[str], descs: dict) -> "torch.Tensor":
        e = tok([f"{n}: {descs[n]}" for n in names], padding=True, return_tensors="pt").to(device)
        h = enc(input_ids=e.input_ids, attention_mask=e.attention_mask).last_hidden_state.float()
        m = e.attention_mask.unsqueeze(-1).float()
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)

    cv = label_vecs(cat_names, CAT_DESC)
    sv = label_vecs(sub_names, SUB_DESC)

    @torch.no_grad()
    def type_spans(answer: str, context: str, spans: list[dict]) -> list[dict]:
        """Type each predicted span via the head (cosine to label descriptions)."""
        if not spans:
            return []
        e = tok(
            answer + SEP + context, truncation=True, max_length=args.head_max_length,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offs = e.pop("offset_mapping")[0].tolist()
        h = enc(input_ids=e.input_ids.to(device), attention_mask=e.attention_mask.to(device)).last_hidden_state.float()[0]
        out = []
        for sp in spans:
            s, en = sp["start"], sp["end"]
            mask = torch.tensor(
                [1.0 if (a < en and b > s and b > a) else 0.0 for a, b in offs], device=device
            )
            if mask.sum() == 0:  # span truncated away -> CLS fallback (matches training)
                mask[0] = 1.0
            v = torch.nn.functional.normalize((h * mask.unsqueeze(-1)).sum(0) / mask.sum().clamp(min=1), dim=-1)
            out.append({
                "start": s, "end": en,
                "category": cat_names[int((v @ cv.T).argmax())],
                "subcategory": sub_names[int((v @ sv.T).argmax())],
            })
        return out

    rows_in = []
    for name in args.dataset:
        for r in load_dataset(name, split=args.split):
            if args.only and r.get("dataset") != args.only:
                continue
            rows_in.append(r)
            if args.limit and len(rows_in) >= args.limit:
                break

    items = []
    for r in tqdm(rows_in, desc="cascade"):
        pred = det.predict_prompt(r["prompt"], r["answer"], output_format="spans")
        typed = type_spans(r["answer"], r.get("context") or "", pred)
        items.append((r.get(args.by, "?"), r.get("labels") or [], typed))

    print_metrics_table(items, by_label=args.by)


if __name__ == "__main__":
    main()
