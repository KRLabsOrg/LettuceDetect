#!/usr/bin/env python3
"""Example-level baseline panel: do off-the-shelf detectors catch hallucinated answers?

Most general/specialist detectors (HHEM, Lynx, Granite-Guardian) only emit an
answer-level verdict, not spans. This harness scores them at the example level
(gold = the answer contains >=1 annotated span) so they sit in one table with our
models (collapsed to "predicted >=1 span -> hallucinated"). The thesis: they fail
on code.

    python scripts/evaluate_example_baselines.py --baseline hhem \
        --dataset KRLabsOrg/lettucedetect-code-hallucination --only lettucedetect-code-agent
"""

from __future__ import annotations

import argparse

from datasets import load_dataset


def load_rows(dataset: str, split: str, only: str | None, limit: int) -> list[dict]:
    """Return [{premise, answer, gold}] where gold = answer has >=1 annotated span."""
    rows = []
    for r in load_dataset(dataset, split=split):
        if only and r.get("dataset") != only:
            continue
        rows.append(
            {
                "premise": r["prompt"] or r["context"],
                "answer": r["answer"],
                "gold": bool(r.get("labels")),
                "source": r.get("dataset"),
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def example_metrics(gold: list[bool], pred: list[bool]) -> dict[str, float]:
    """Precision/recall/F1 for the hallucinated class + balanced accuracy + accuracy."""
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    tn = sum((not g) and (not p) for g, p in zip(gold, pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    bacc = (tpr + tnr) / 2
    acc = (tp + tn) / len(gold) if gold else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "bacc": bacc,
        "accuracy": acc,
        "n": len(gold),
    }


def predict_hhem(rows: list[dict], threshold: float, device: str) -> list[bool]:
    """HHEM-2.1-Open consistency, SLIDING-WINDOW over the context.

    HHEM's 512-token window can't fit code-agent contexts (median ~5k tokens), so a
    single truncated pass makes every answer look unsupported (recall->1). Instead we
    chunk the context into window-sized pieces, score (chunk, answer) for each, and
    take the MAX consistency: the answer is grounded if its best-supporting chunk
    supports it. Hallucinated iff no chunk supports it (max consistency < threshold).
    """
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        "vectara/hallucination_evaluation_model", trust_remote_code=True
    ).to(device)
    chunk, stride, max_chunks, ans_cap = 1000, 800, 30, 400  # chars (~<=512 tok with answer)
    pairs: list[tuple[str, str]] = []
    owner: list[int] = []
    for i, r in enumerate(rows):
        ctx, ans = r["premise"], r["answer"][:ans_cap]
        chunks = [ctx[j : j + chunk] for j in range(0, max(1, len(ctx)), stride)][:max_chunks] or [
            ""
        ]
        for c in chunks:
            pairs.append((c, ans))
            owner.append(i)
    scores = []
    bs = 64
    for i in range(0, len(pairs), bs):
        with torch.no_grad():
            scores.extend(model.predict(pairs[i : i + bs]).tolist())
    best = [0.0] * len(rows)
    for o, s in zip(owner, scores):
        best[o] = max(best[o], s)
    return [b < threshold for b in best]  # grounded if any chunk supports; else hallucinated


LYNX_PROMPT = (
    "Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer "
    "and determine whether it is faithful to the contents of the DOCUMENT. The ANSWER must not "
    "offer new information beyond the context provided in the DOCUMENT. The ANSWER also must not "
    "contradict information provided in the DOCUMENT. Output your final verdict by strictly "
    'following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the '
    "answer is not faithful to the DOCUMENT. Show your reasoning.\n\n--\nQUESTION:\n{q}\n--\n"
    "DOCUMENT:\n{doc}\n--\nANSWER:\n{ans}\n--\n\n"
    'Your output should be in JSON format with the keys "REASONING" and "SCORE":\n'
    '{{"REASONING": <bullet points>, "SCORE": <"PASS" or "FAIL">}}'
)


def predict_lynx(rows: list[dict], threshold: float, device: str) -> list[bool]:
    """Lynx-8B faithfulness judge (vLLM offline); SCORE=FAIL => hallucinated. 8K window."""
    import re

    from vllm import LLM, SamplingParams

    llm = LLM(
        model="PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct",
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
    )
    tok = llm.get_tokenizer()

    def _clip(text: str, n_tok: int) -> str:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return tok.decode(ids[:n_tok]) if len(ids) > n_tok else text

    prompts = []
    for r in rows:
        prem = r["premise"]
        if prem.startswith("User request:"):
            q, _, doc = prem.partition("\n\n")
            q = q[len("User request:") :].strip()
        else:
            q, doc = "", prem
        # Budget for the 8K window: doc <=6000 tok, answer <=900 tok, +template/gen.
        user = LYNX_PROMPT.format(
            q=q or "(not provided)", doc=_clip(doc, 6000), ans=_clip(r["answer"], 900)
        )
        prompts.append(
            tok.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
            )
        )
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=600))
    preds = []
    for o in outs:
        t = o.outputs[0].text.upper()
        m = re.search(r'"?SCORE"?\s*:?\s*"?\s*(PASS|FAIL)', t)
        preds.append((m.group(1) if m else ("FAIL" if "FAIL" in t else "PASS")) == "FAIL")
    return preds


GRANITE_NOTHINK = (
    "<no-think>As a judge agent, assess whether the provided text meets the given "
    "judging criteria using all available information, including conversations, "
    "documents, and tools. Provide your score immediately without explanation. "
    "Output empty <think>\\n</think> tags followed by your score in <score></score> tags."
)
GRANITE_GROUNDEDNESS = (
    "A text is considered ungrounded or unfaithful if it includes information "
    "lacking support from, or directly contradicting, the provided document(s). "
    "This risk arises when the text fabricates details, misinterprets the content, "
    "or makes unsupported extrapolations beyond what is explicitly stated in the document(s)."
)


def predict_granite(rows: list[dict], threshold: float, device: str) -> list[bool]:
    """Granite-Guardian-4.1-8b groundedness (vLLM offline, no-think); <score>yes => hallucinated."""
    import re

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    mid = "ibm-granite/granite-guardian-4.1-8b"
    tok = AutoTokenizer.from_pretrained(mid)
    llm = LLM(model=mid, max_model_len=8192, gpu_memory_utilization=0.9, dtype="bfloat16")
    block = (
        f"<guardian>{GRANITE_NOTHINK}\n\n### Criteria: {GRANITE_GROUNDEDNESS}\n\n"
        "### Scoring Schema: If the last assistant's text meets the criteria, "
        "return 'yes'; otherwise, return 'no'."
    )

    def _clip(text: str, n: int) -> str:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return tok.decode(ids[:n]) if len(ids) > n else text

    prompts = []
    for r in rows:
        docs = [{"doc_id": "0", "text": _clip(r["premise"], 5000)}]
        msgs = [
            {"role": "assistant", "content": _clip(r["answer"], 600)},
            {"role": "user", "content": block},
        ]
        prompts.append(
            tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, documents=docs
            )
        )
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512))
    preds = []
    for o in outs:
        t = re.sub(r"<think>.*?</think>", "", o.outputs[0].text, flags=re.DOTALL)
        m = re.findall(r"<score>\s*(.*?)\s*</score>", t, re.DOTALL)
        v = m[0].strip().lower() if m else ("yes" if "yes" in t.lower() else "no")
        preds.append(v == "yes")  # yes = ungrounded = hallucinated
    return preds


def predict_minicheck(rows: list[dict], threshold: float, device: str) -> list[bool]:
    """Bespoke-MiniCheck-7B, CLAIM-level (atomic line per claim).

    MiniCheck is a sentence/claim-level checker, so scoring a whole multi-line answer
    as ONE claim degenerates (it returns unsupported almost always). The faithful use
    is to decompose the answer into atomic units and score each against the context.
    For code-agent answers the atomic unit is a LINE. The answer is hallucinated if ANY
    line is unsupported (support prob < threshold). MiniCheck chunks long docs internally.
    """
    from minicheck.minicheck import MiniCheck

    scorer = MiniCheck(model_name="Bespoke-MiniCheck-7B", enable_prefix_caching=False)
    docs: list[str] = []
    claims: list[str] = []
    owner: list[int] = []
    for i, r in enumerate(rows):
        for ln in r["answer"].splitlines():
            ln = ln.strip()
            if len(ln) >= 3 and any(c.isalnum() for c in ln):  # skip bare braces/punctuation
                docs.append(r["premise"])
                claims.append(ln)
                owner.append(i)
    _, raw_prob, _, _ = scorer.score(docs=docs, claims=claims)
    min_sup = [1.0] * len(rows)  # answers with no scorable line stay "supported"
    for o, p in zip(owner, raw_prob):
        min_sup[o] = min(min_sup[o], p)
    return [s < threshold for s in min_sup]  # any line unsupported => hallucinated


BASELINES = {
    "hhem": predict_hhem,
    "lynx": predict_lynx,
    "granite": predict_granite,
    "minicheck": predict_minicheck,
}


def _report(label: str, gold: list[bool], pred: list[bool]) -> None:
    m = example_metrics(gold, pred)
    print(
        f"  {label:24s} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  "
        f"BAcc {m['bacc']:.3f}  Acc {m['accuracy']:.3f}  (n={m['n']})"
    )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, choices=sorted(BASELINES))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--only", help="Keep only rows whose `dataset` field == this.")
    ap.add_argument(
        "--by-source", action="store_true", help="Report per-source + ALL (one model load)."
    )
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(args.dataset, args.split, args.only, args.limit)
    print(f"{args.baseline}: {len(rows)} rows ({sum(r['gold'] for r in rows)} hallucinated)")
    pred = BASELINES[args.baseline](rows, args.threshold, args.device)
    gold = [r["gold"] for r in rows]
    if args.by_source:
        _report("ALL", gold, pred)
        for s in sorted({r["source"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["source"] == s]
            _report(s, [gold[i] for i in idx], [pred[i] for i in idx])
    else:
        _report(args.baseline, gold, pred)


def _selfcheck() -> None:
    """Metric sanity check (no model)."""
    g = [True, True, False, False]
    assert example_metrics(g, [True, True, False, False])["f1"] == 1.0
    assert example_metrics(g, [False, False, True, True])["f1"] == 0.0
    assert example_metrics(g, [True, True, True, True])["bacc"] == 0.5
    print("selfcheck ok")


if __name__ == "__main__":
    import sys

    _selfcheck() if len(sys.argv) == 1 else main()
