#!/usr/bin/env python3
"""Evaluate the generative span detector (LFM2.5 LoRA) with the SAME metrics as the encoder.

Queries a vLLM OpenAI-compatible endpoint for the hallucinated-spans JSON, maps
each emitted span's verbatim text back to char offsets in the answer, then scores
span char-F1 / example-F1 / IoU per dataset/language via the shared metric. This
makes the generative model directly comparable to mmBERT.

    # 1. serve the merged model (separate process / tmux):
    TRITON_CACHE_DIR=$HOME/.triton_cache vllm serve <merged_dir> \
        --served-model-name lfm2-sft --max-model-len 32768 --port 8000
    # 2. run the eval client (needs only openai + datasets):
    python scripts/evaluate_generative_model.py --model lfm2-sft \
        --base-url http://localhost:8000/v1 \
        --dataset KRLabsOrg/lettucedetect-code-hallucination \
        --dataset KRLabsOrg/lettucedetect-prose-hallucination \
        --split test --by dataset
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))


def parse_spans(text: str) -> list[str] | None:
    """Pull hallucinated-span texts out of a model reply. None = unparseable."""
    obj = None
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except (ValueError, TypeError):
                obj = None
    if not isinstance(obj, dict):
        return None
    spans = obj.get("hallucinated_spans")
    if spans is None:
        return None
    return [s.get("text", "") for s in spans if isinstance(s, dict)]


def spans_to_offsets(answer: str, span_texts: list[str]) -> list[dict]:
    """Map each verbatim span text to its first non-overlapping {start,end} in answer."""
    out: list[dict] = []
    used: list[tuple[int, int]] = []
    for t in span_texts:
        if not t or not t.strip():
            continue
        start = 0
        while (i := answer.find(t, start)) >= 0:
            j = i + len(t)
            if not any(i < ue and us < j for us, ue in used):
                out.append({"start": i, "end": j})
                used.append((i, j))
                break
            start = i + 1
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Generative span-detector eval (OpenAI endpoint).")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM OpenAI URL.")
    ap.add_argument("--model", required=True, help="Served model name (--served-model-name).")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--dataset", action="append", default=[], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--by", choices=["dataset", "language"], default="dataset")
    ap.add_argument("--only", default="", help="Keep only rows whose `dataset` == this.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument(
        "--explain",
        action="store_true",
        help="Query ALL rows with the explanation prompt.",
    )
    ap.add_argument(
        "--explain-datasets",
        action="append",
        default=[],
        help="Dataset name(s) whose rows use the explanation prompt (e.g. their training prompt).",
    )
    args = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor

    from build_generative_sft import SYSTEM_BASE, SYSTEM_EXPL
    from datasets import load_dataset
    from openai import OpenAI
    from span_eval_metrics import print_metrics_table
    from tqdm import tqdm

    explain_ds = set(args.explain_datasets)

    def system_for(r: dict) -> str:
        return SYSTEM_EXPL if args.explain or r.get("dataset") in explain_ds else SYSTEM_BASE

    rows_in = []
    for name in args.dataset:
        for r in load_dataset(name, split=args.split):
            if args.only and r.get("dataset") != args.only:
                continue
            rows_in.append(r)
            if args.limit and len(rows_in) >= args.limit:
                break

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    def infer(r: dict) -> str | None:
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_for(r)},
                    {
                        "role": "user",
                        "content": f"Context:\n{r['context'] or r['prompt']}\n\n"
                        f"Answer to verify:\n{r['answer']}",
                    },
                ],
                temperature=0.0,
                max_tokens=args.max_new_tokens,
            )
            return resp.choices[0].message.content
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        replies = list(tqdm(ex.map(infer, rows_in), total=len(rows_in), desc="generate"))

    rows = []
    bad = 0
    for r, reply in zip(rows_in, replies):
        texts = parse_spans(reply) if reply is not None else None
        if texts is None:
            bad += 1
            texts = []
        pred = spans_to_offsets(r["answer"], texts)
        rows.append((r.get(args.by, "?"), r.get("labels") or [], pred))

    print_metrics_table(rows, by_label=args.by)
    print(f"\nunparseable/failed replies: {bad}/{len(rows)}")


def _selfcheck() -> None:
    """Offline check of parse + offset mapping (no model needed)."""
    ans = "foo bar baz bar"
    reply = '{"hallucinated_spans": [{"text": "bar", "category": "x"}, {"text": "baz"}]}'
    texts = parse_spans(reply)
    assert texts == ["bar", "baz"], texts
    offs = spans_to_offsets(ans, texts)
    assert offs == [{"start": 4, "end": 7}, {"start": 8, "end": 11}], offs
    # second "bar" picked when first is taken
    assert spans_to_offsets(ans, ["bar", "bar"]) == [
        {"start": 4, "end": 7},
        {"start": 12, "end": 15},
    ]
    assert parse_spans("garbage no json") is None
    assert parse_spans('{"hallucinated_spans": []}') == []
    print("selfcheck ok")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selfcheck()
    else:
        main()
