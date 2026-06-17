#!/usr/bin/env python3
"""Zero-shot label-conditioned taxonomy head (cascade Stage-B).

A shared mmBERT bi-encoder types a span the binary detector already found:
  span vector  = span-token mean-pool of encode(answer + [CTX] + context)
  label vector = mean-pool of encode("name: description") for each taxonomy label
  category    = argmax cosine(span, category labels)
  subcategory = argmax cosine(span, subcategory labels)
Labels enter only as text, so a held-out label works from its description alone
(zero-shot). Detection is a separate frozen pass; this only TYPES a given span.

    python scripts/train_taxonomy_head.py --data-dir data/taxonomy \
        --output-dir output/taxonomy_head --model-name jhu-clsp/mmBERT-base
    # zero-shot eval: hold a subcategory out of training, score it at test
    python scripts/train_taxonomy_head.py ... --holdout-subcategory temporal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import (
    CATEGORY_DESCRIPTIONS as CATEGORY_LABELS,
)
from taxonomy import (
    SUBCATEGORY_DESCRIPTIONS as SUBCATEGORY_LABELS,
)

SEP = " [CTX] "


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model-name", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temp", type=float, default=0.05, help="cosine softmax temperature")
    ap.add_argument("--holdout-subcategory", default="", help="exclude from TRAIN, score at test")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name, dtype=torch.bfloat16).to(device)

    cat_names = list(CATEGORY_LABELS)
    sub_names = list(SUBCATEGORY_LABELS)

    def encode(texts: list[str], max_len: int) -> dict:
        return tok(
            texts, truncation=True, max_length=max_len, padding=True,
            return_offsets_mapping=True, return_tensors="pt",
        )

    def label_vecs(names: list[str], descs: dict) -> "torch.Tensor":
        enc = tok([f"{n}: {descs[n]}" for n in names], padding=True, return_tensors="pt").to(device)
        h = encoder(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state.float()
        m = enc.attention_mask.unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)  # mean-pool
        return torch.nn.functional.normalize(v, dim=-1)

    def load(split: str) -> list[dict]:
        rows = []
        for ln in (args.data_dir / f"{split}.jsonl").open():
            r = json.loads(ln)
            if split == "train" and args.holdout_subcategory == r["subcategory"]:
                continue
            rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break
        return rows

    def collate(rows: list[dict]) -> dict:
        texts = [r["answer"] + SEP + r["context"] for r in rows]
        enc = encode(texts, args.max_length)
        offsets = enc.pop("offset_mapping")
        span_mask = torch.zeros_like(enc.input_ids, dtype=torch.float)
        for i, r in enumerate(rows):
            s, e = r["start"], r["end"]  # span lives at the start (answer is first)
            for j, (a, b) in enumerate(offsets[i].tolist()):
                if a < e and b > s and b > a:
                    span_mask[i, j] = 1.0
            if span_mask[i].sum() == 0:  # span truncated away -> fall back to CLS
                span_mask[i, 0] = 1.0
        return {
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "span_mask": span_mask,
            "cat": torch.tensor([cat_names.index(r["category"]) for r in rows]),
            "sub": torch.tensor([sub_names.index(r["subcategory"]) for r in rows]),
        }

    def span_vec(batch: dict) -> "torch.Tensor":
        h = encoder(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).last_hidden_state.float()
        m = batch["span_mask"].to(device).unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return torch.nn.functional.normalize(v, dim=-1)

    import collections

    from tqdm import tqdm

    def evaluate(rows: list) -> dict:
        encoder.eval()
        correct: dict = collections.Counter()
        n: dict = collections.Counter()
        with torch.no_grad():
            cv = label_vecs(cat_names, CATEGORY_LABELS)
            svv = label_vecs(sub_names, SUBCATEGORY_LABELS)
            for i in range(0, len(rows), args.batch_size):
                batch = collate(rows[i : i + args.batch_size])
                sv = span_vec(batch)
                pc = (sv @ cv.T).argmax(-1).cpu()
                ps = (sv @ svv.T).argmax(-1).cpu()
                for k in range(len(pc)):
                    grp = "holdout" if sub_names[batch["sub"][k]] == args.holdout_subcategory else "seen"
                    n[("cat", grp)] += 1
                    correct[("cat", grp)] += int(pc[k] == batch["cat"][k])
                    correct[("sub", grp)] += int(ps[k] == batch["sub"][k])
        encoder.train()
        return {
            g: (correct[("cat", g)] / n[("cat", g)], correct[("sub", g)] / n[("cat", g)], n[("cat", g)])
            for g in ("seen", "holdout")
            if n[("cat", g)]
        }

    def save() -> None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        encoder.save_pretrained(args.output_dir)
        tok.save_pretrained(args.output_dir)
        (args.output_dir / "labels.json").write_text(
            json.dumps({"category": CATEGORY_LABELS, "subcategory": SUBCATEGORY_LABELS}, indent=2)
        )

    train_rows, val_rows = load("train"), load("validation")
    print(f"train {len(train_rows)} / val {len(val_rows)} (holdout={args.holdout_subcategory!r})", flush=True)
    dl = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(encoder.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss()

    steps_total = int(len(dl) * args.epochs)
    pbar = tqdm(total=steps_total, desc="train")
    step = 0
    encoder.train()
    while step < steps_total:
        for batch in dl:
            sv = span_vec(batch)
            cv = label_vecs(cat_names, CATEGORY_LABELS)
            svv = label_vecs(sub_names, SUBCATEGORY_LABELS)
            loss = ce((sv @ cv.T) / args.temp, batch["cat"].to(device)) + ce(
                (sv @ svv.T) / args.temp, batch["sub"].to(device)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3f}")
            if step % args.eval_steps == 0:
                ev = evaluate(val_rows)
                msg = " ".join(f"{g}:cat={a:.3f}/sub={b:.3f}(n{c})" for g, (a, b, c) in ev.items())
                tqdm.write(f"step {step}/{steps_total} {msg}")
                save()  # checkpoint (overwrite latest) — resumable + observable
            if step >= steps_total:
                break
    pbar.close()
    for g, (a, b, c) in evaluate(val_rows).items():
        print(f"[{g}] n={c} cat_acc={a:.4f} sub_acc={b:.4f}", flush=True)
    save()
    print(f"saved -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
