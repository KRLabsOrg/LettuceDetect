#!/usr/bin/env python3
"""Evaluate the generative span detector (LFM2.5 LoRA) with the SAME metrics as the encoder.

Generates the hallucinated-spans JSON with vLLM, maps each emitted span's verbatim
text back to char offsets in the answer, then scores span char-F1 / example-F1 /
IoU per dataset/language via the shared metric in evaluate_span_model. This makes
the generative model directly comparable to mmBERT.

    # merge the adapter first (scripts/merge_lora.py), then:
    python scripts/evaluate_generative_model.py \
        --model /mnt/workspace/users/adamko/lfm2_sft_merged \
        --dataset KRLabsOrg/lettucedetect-code-hallucination \
        --dataset KRLabsOrg/lettucedetect-prose-hallucination \
        --split test --by dataset
    # or evaluate the adapter directly without merging:
    python scripts/evaluate_generative_model.py --model unsloth/LFM2.5-8B-A1B \
        --lora /mnt/workspace/users/adamko/lfm2_sft/lora --dataset ... --split test
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
    ap = argparse.ArgumentParser(description="Generative span-detector evaluation (vLLM).")
    ap.add_argument("--model", required=True, help="Merged model dir or base model id.")
    ap.add_argument("--lora", help="LoRA adapter dir (if --model is the base model).")
    ap.add_argument("--dataset", action="append", default=[], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--by", choices=["dataset", "language"], default="dataset")
    ap.add_argument("--only", default="", help="Keep only rows whose `dataset` == this.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-seq-length", type=int, default=32768)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size.")
    args = ap.parse_args()

    from build_generative_sft import SYSTEM_BASE
    from datasets import load_dataset
    from span_eval_metrics import print_metrics_table
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows_in = []
    for name in args.dataset:
        for r in load_dataset(name, split=args.split):
            if args.only and r.get("dataset") != args.only:
                continue
            rows_in.append(r)
            if args.limit and len(rows_in) >= args.limit:
                break

    convos = [
        [
            {"role": "system", "content": SYSTEM_BASE},
            {
                "role": "user",
                "content": f"Context:\n{r['context'] or r['prompt']}\n\n"
                f"Answer to verify:\n{r['answer']}",
            },
        ]
        for r in rows_in
    ]

    llm = LLM(
        model=args.model,
        max_model_len=args.max_seq_length,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        enable_lora=bool(args.lora),
        max_lora_rank=64,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    lora_req = LoRARequest("sft", 1, args.lora) if args.lora else None
    outputs = llm.chat(convos, sp, lora_request=lora_req)

    rows = []
    bad = 0
    for r, out in zip(rows_in, outputs):
        texts = parse_spans(out.outputs[0].text)
        if texts is None:
            bad += 1
            texts = []
        pred = spans_to_offsets(r["answer"], texts)
        rows.append((r.get(args.by, "?"), r.get("labels") or [], pred))

    print_metrics_table(rows, by_label=args.by)
    print(f"\nunparseable replies: {bad}/{len(rows)}")


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
