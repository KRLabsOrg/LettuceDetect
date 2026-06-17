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


def span_metrics(items: list) -> tuple[float, float, float, float, float]:
    """Char-overlap span (F1/P/R), example-level F1, and per-sample char IoU."""
    ov = pp = gg = 0  # char overlap / predicted / gold
    etp = efp = efn = 0  # example-level on "has any span"
    iou_sum = 0.0  # per-sample char IoU, PsiloQA's unit
    for _key, gold, pred in items:
        pp += sum(p["end"] - p["start"] for p in pred)
        gg += sum(g["end"] - g["start"] for g in gold)
        for p in pred:
            for g in gold:
                ov += max(0, min(p["end"], g["end"]) - max(p["start"], g["start"]))
        ps, gs = char_set(pred), char_set(gold)
        union = ps | gs
        iou_sum += 1.0 if not union else len(ps & gs) / len(union)
        pe, ge = bool(pred), bool(gold)
        etp += pe and ge
        efp += pe and not ge
        efn += ge and not pe
    cp = ov / pp if pp else 0.0
    cr = ov / gg if gg else 0.0
    cf = 2 * cp * cr / (cp + cr) if cp + cr else 0.0
    ep = etp / (etp + efp) if etp + efp else 0.0
    er = etp / (etp + efn) if etp + efn else 0.0
    ef = 2 * ep * er / (ep + er) if ep + er else 0.0
    iou = iou_sum / len(items) if items else 0.0
    return cf, cp, cr, ef, iou


def print_metrics_table(rows: list, by_label: str = "group") -> None:
    """Print ALL + per-group span/example/IoU metrics for (key, gold, pred) rows."""
    groups: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        groups[r[0]].append(r)
    print(f"{by_label:<28} {'n':>6} {'span_f1':>8} {'span_p':>7} {'span_r':>7} {'ex_f1':>7} {'iou':>7}")
    for name in ["ALL", *sorted(groups)]:
        grp = rows if name == "ALL" else groups[name]
        cf, cp, cr, ef, iou = span_metrics(grp)
        print(f"{name:<28} {len(grp):>6} {cf:>8.4f} {cp:>7.4f} {cr:>7.4f} {ef:>7.4f} {iou:>7.4f}")
