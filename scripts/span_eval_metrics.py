"""Dependency-free span metrics shared by the encoder and generative evaluators.

Pure stdlib so it imports in any env (e.g. the vLLM env without torch/lettucedetect).
Each "row" is (group_key, gold_spans, pred_spans) where spans are {start,end} dicts.
"""

from __future__ import annotations

import collections


def char_set(spans: list) -> set:
    """Set of character indices covered by the spans."""
    out: set[int] = set()
    for s in spans:
        out.update(range(s["start"], s["end"]))
    return out


def _f1(num: int, pp: int, gg: int) -> float:
    """F1 from an overlap numerator and predicted/gold totals."""
    p = num / pp if pp else 0.0
    r = num / gg if gg else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def span_metrics(items: list) -> dict:
    """Compute detection + (optional) typed metrics for (key, gold, pred) rows.

    Spans are {start,end[,category,subcategory]}. `typed_f1` is the same char
    overlap but counted only when pred & gold categories match; `typed_f1_sub`
    additionally requires subcategory. They appear only if any span carries a
    category, so binary/encoder-only runs are unaffected.
    """
    ov = pp = gg = t_ov = ts_ov = 0  # overlap / pred / gold / typed / typed+sub
    etp = efp = efn = 0  # example-level on "has any span"
    iou_sum = 0.0  # per-sample char IoU, PsiloQA's unit
    has_cat = False
    for _key, gold, pred in items:
        pp += sum(p["end"] - p["start"] for p in pred)
        gg += sum(g["end"] - g["start"] for g in gold)
        for p in pred:
            has_cat = has_cat or "category" in p
            for g in gold:
                o = max(0, min(p["end"], g["end"]) - max(p["start"], g["start"]))
                ov += o
                if o and p.get("category") == g.get("category"):
                    t_ov += o
                    if p.get("subcategory") == g.get("subcategory"):
                        ts_ov += o
        ps, gs = char_set(pred), char_set(gold)
        union = ps | gs
        iou_sum += 1.0 if not union else len(ps & gs) / len(union)
        pe, ge = bool(pred), bool(gold)
        etp += pe and ge
        efp += pe and not ge
        efn += ge and not pe
    ep = etp / (etp + efp) if etp + efp else 0.0
    er = etp / (etp + efn) if etp + efn else 0.0
    return {
        "span_f1": _f1(ov, pp, gg),
        "span_p": ov / pp if pp else 0.0,
        "span_r": ov / gg if gg else 0.0,
        "ex_f1": 2 * ep * er / (ep + er) if ep + er else 0.0,
        "iou": iou_sum / len(items) if items else 0.0,
        "typed_f1": _f1(t_ov, pp, gg),
        "typed_f1_sub": _f1(ts_ov, pp, gg),
        "has_cat": has_cat,
    }


def print_metrics_table(rows: list, by_label: str = "group") -> None:
    """Print ALL + per-group detection (and, when present, typed) metrics."""
    groups: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        groups[r[0]].append(r)
    typed = span_metrics(rows)["has_cat"]
    hdr = f"{by_label:<28} {'n':>6} {'span_f1':>8} {'span_p':>7} {'span_r':>7} {'ex_f1':>7} {'iou':>7}"
    if typed:
        hdr += f" {'typ_f1':>7} {'typ_sub':>7}"
    print(hdr)
    for name in ["ALL", *sorted(groups)]:
        grp = rows if name == "ALL" else groups[name]
        m = span_metrics(grp)
        line = (
            f"{name:<28} {len(grp):>6} {m['span_f1']:>8.4f} {m['span_p']:>7.4f} "
            f"{m['span_r']:>7.4f} {m['ex_f1']:>7.4f} {m['iou']:>7.4f}"
        )
        if typed:
            line += f" {m['typed_f1']:>7.4f} {m['typed_f1_sub']:>7.4f}"
        print(line)
