#!/usr/bin/env python3
r"""Generate code-agent hallucination samples: realistic solution + grounded mistake.

Per instance:
  1. build context (repo files) + request;
  2. generate a correct assistant solution (``agent_solution``);
  3. inject 1-3 request-grounded mistakes (wrong_implementation / unrequested_change)
     with exact spans, mapped to the unified taxonomy via ``code_agent``;
  4. emit a v2 HallucinationSample (``dataset="lettucedetect-code-agent"``).

A configurable share is left clean (correct solution, no labels) as negatives.

Usage::

    API_BASE_URL=https://api.mistral.ai/v1 OPENAI_API_KEY=... MODEL=mistral-small-2603 \
        python scripts/generate_code_agent_hallucinations.py --limit 30 --out data/v2/code_agent
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI  # noqa: E402

from lettucedetect.datasets.taxonomy import CODE_AGENT_MAP  # noqa: E402
from lettucedetect.generation.agent_solution import generate_agent_solution_async  # noqa: E402
from lettucedetect.generation.assembly import format_prompt  # noqa: E402
from lettucedetect.generation.doc_source import hash_split  # noqa: E402
from lettucedetect.generation.injection import _map_menu_labels, inject_async  # noqa: E402
from lettucedetect.generation.runner import Outcome, run_batched_sync  # noqa: E402
from scripts.code_hallucination.answer_grounding import (  # noqa: E402
    ground_fabricated_apis,
    render_definitions,
    resolve_definitions,
)
from scripts.code_hallucination.source_fetcher import build_source_context  # noqa: E402

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.6-35B-A3B")
HALLUCINATION_RATIO = float(os.environ.get("HALLUCINATION_RATIO", "0.5"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "28000"))
MAX_ANSWER_CHARS = int(os.environ.get("MAX_ANSWER_CHARS", "10000"))
DATA = PROJECT_ROOT / "data" / "code_hallucination"

INJECTION_PROMPT = """\
You introduce realistic mistakes into a coding assistant's CORRECT solution — the kind a competent
coding agent actually makes. The code stays plausible, well-formed, and uses real names; it just
fails to do what the developer asked. Return replacement edits as JSON; the pipeline applies them.

Each mistake MUST be judgeable by comparing the solution to the DEVELOPER REQUEST (and the
repository context): a reviewer reading the request and the answer should see the solution does
NOT do what was asked. Do NOT rely on outside library knowledge or math.

Pick the type that fits each edit and label it:
- wrong_implementation: the code addresses the request but with WRONG logic, condition, field, or
  value, so it does not actually achieve what was asked — e.g. it checks/handles the wrong case,
  uses the wrong field or default, sets/returns the wrong thing, or writes a whole block whose
  logic differs from what was requested.
- unrequested_change: the code also does something the request did NOT ask for — insert a NEW
  statement or block (an extra side effect, an unrelated operation, broadened scope) that the
  request never called for.

An edit may be a small value/condition swap OR an added/rewritten BLOCK of several lines with
different or unrequested logic — prefer a mix, not only one-token swaps. To insert a new block,
set "original" to an exact existing line and "hallucinated" to that line plus the added code.

Make each mistake PLAUSIBLE: the kind of slip a good agent makes, not obvious vandalism. The
edited code stays syntactically valid and looks reasonable on its own. Do NOT make a no-op (each
edit must change behavior), do NOT introduce a name that exists nowhere, no hint words.

Keep each edit TIGHT: "original" and "hallucinated" should be the smallest snippet that captures
the change — the changed line(s) only, not the surrounding correct code. For an added block, the
"hallucinated" is the anchor line plus the new lines, nothing more.

Rules: make 1-3 DISTINCT mistakes in different places — prefer 2-3 when the solution is long
enough to support genuinely distinct ones; use 1 only when nothing else fits (never force a
no-op). Each "original" is an exact substring appearing once in the answer.
Respond ONLY with JSON (no prose, no code fence):
{"changes":[{"original":"...","hallucinated":"...","hallucination_type":"wrong_implementation|unrequested_change","explanation":"how this fails to satisfy the developer request"}]}
"""


STRUCTURAL_PROMPT = """\
You introduce realistic FABRICATIONS into a coding assistant's CORRECT solution — the kind of slip
an agent makes when it misremembers an existing API. Change a call, attribute, or keyword-argument
on an object or library that ALREADY exists so it references a member that does NOT exist —
e.g. `config.get_value(x)` where the real method is `config.get(x)`, `resp.json_body` where it is
`resp.json()`, `timeout_seconds=5` where the real keyword is `timeout=5`, or
`psutil.iter_processes()` where it is `psutil.process_iter()`.

The fabricated member must be a pure REFERENCE to something the code assumes already exists — NOT a
name the answer itself introduces. Therefore:
- It MUST be an attribute/method access (`obj.fabricated(...)`, `obj.fabricated`) or a keyword
  argument (`fabricated=...`) on a pre-existing object/function.
- It MUST appear NOWHERE else — not in the context, not elsewhere in the answer.
- Do NOT rename a parameter, local variable, or dict key that the answer defines or assigns
  (e.g. do NOT change `def f(query)` + its uses to `query_param`, and do NOT invent
  `config['new_key'] = ...`). Those are self-consistent, not fabrications.
- Keep the code otherwise valid; no hint words.

Rules: make 1-3 edits, each a DISTINCT fabricated member in a different place — prefer 2-3 when the
solution is long enough. Each "original" is an exact substring appearing once in the answer;
"hallucinated" is the smallest expression containing the fabricated member.
Respond ONLY with JSON (no prose, no code fence):
{"changes":[{"original":"...","hallucinated":"...","hallucination_type":"fabricated_api","explanation":"which member is fabricated and why it does not exist on that object/library"}]}
"""

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SELF_CALL = re.compile(r"\bself\.([A-Za-z_]\w*)\s*\(")
_DEF = re.compile(r"\bdef\s+([A-Za-z_]\w*)")


def _ungrounded_self_calls(answer: str, context: str) -> set[str]:
    """Return self-method calls in the answer not defined in it or present in context."""
    defined = set(_DEF.findall(answer))
    ctx_names = set(_IDENT.findall(context))
    return {
        name
        for name in _SELF_CALL.findall(answer)
        if name not in defined and name not in ctx_names and not name.startswith("__")
    }


def _introduced_names(changes: list[dict], present: set[str]) -> set[str]:
    """Return identifiers an edit adds (in hallucinated, not original) absent elsewhere."""
    names: set[str] = set()
    for change in changes:
        original_names = set(_IDENT.findall(change.get("original", "")))
        for name in set(_IDENT.findall(change.get("hallucinated", ""))) - original_names:
            if len(name) > 2 and name not in present:
                names.add(name)
    return names


def _has_absent_fabrication(changes: list[dict], haystack: str) -> bool:
    """Check whether some edit introduces an identifier absent from context + clean answer."""
    return bool(_introduced_names(changes, set(_IDENT.findall(haystack))))


def _is_trivial_answer(answer: str) -> bool:
    """Check whether a gold answer is too trivial to inject into (e.g. just a version bump)."""
    code = " ".join(re.findall(r"```python(.*?)```", answer, re.S)) or answer
    stripped = re.sub(r"__version\w*__\s*=\s*[\"'][^\"']*[\"']", "", code)
    return len(stripped.strip()) < 40


def _build_context(sc: dict, docs: dict | None) -> str:
    """Source files, plus dependency definitions and docs when present."""
    parts = [build_source_context(sc)]
    deps = sc.get("dependency_files") or {}
    if deps:
        parts.append(
            "Referenced definitions:\n"
            + "\n\n".join(f"# {path}\n{code}" for path, code in deps.items())
        )
    if docs:
        parts.append(
            "Documentation:\n" + "\n\n".join(f"## {name}\n{text}" for name, text in docs.items())
        )
    context = "\n\n".join(p for p in parts if p)
    return context[:MAX_CONTEXT_CHARS]


def _reference(sc: dict) -> str:
    """Return the gold change used to guide the (correct) solution."""
    return sc.get("edit_style") or sc.get("patch_code") or ""


def _gold_answer(sc: dict, style: str) -> str:
    """Return the gold fix rendered in the requested answer style.

    ``function`` is the patched body of the modified functions, ``fragment`` the
    raw patch hunk, and ``edit`` the ``In file X, replace Y with Z`` form. Falls
    back to ``edit_style``/``patch_code`` when the requested style is unavailable.
    """
    funcs = [f["patched"] for f in sc.get("modified_functions", []) if f.get("patched")]
    if style == "function" and funcs:
        # One function keeps the answer short/trainable; grounding pulls in the
        # siblings it calls, so it stays coherent. Prefer the largest function that
        # fits the length cap rather than the largest overall (which often blows it).
        fitting = [f for f in funcs if len(f) <= MAX_ANSWER_CHARS]
        return max(fitting, key=len) if fitting else min(funcs, key=len)
    if style == "fragment" and (sc.get("patch_code") or "").strip():
        return sc["patch_code"]
    return sc.get("edit_style") or sc.get("patch_code") or ""


def _make_sample(
    oid: str,
    request: str,
    context: str,
    answer: str,
    labels: list[dict],
    reasoning: str | None,
    split: str,
    style: str,
    mode: str | None,
) -> dict:
    is_hall = bool(labels)
    return {
        "prompt": format_prompt(context, request),
        "context": context,
        "question": request,
        "answer": answer,
        "labels": labels,
        "split": split,
        "task_type": "code_generation",
        "dataset": "lettucedetect-code-agent",
        "language": "en",
        "context_modality": "code",
        "category": labels[0]["category"] if labels else None,
        "subcategory": labels[0].get("subcategory") if labels else None,
        "metadata": {
            "instance_id": oid,
            "is_hallucinated": is_hall,
            "model": MODEL,
            "reasoning": reasoning,
            "answer_style": style,
            "hallucination_mode": mode,
        },
    }


async def _ground_context(solution: str, context: str, sc: dict, repo_meta: tuple) -> str:
    """Prepend real repo definitions for answer references missing from the context."""
    repo, commit = repo_meta
    if not repo or not commit:
        return context
    defs = await asyncio.to_thread(
        resolve_definitions,
        solution,
        context,
        repo=repo,
        commit=commit,
        changed_files=sc.get("changed_files", []),
        modified_functions=sc.get("modified_functions", []),
    )
    block = render_definitions(defs)
    if not block:
        return context
    budget = max(MAX_CONTEXT_CHARS - len(block) - 4, 0)
    return block + "\n\n" + context[:budget]


def _make_process(
    aclient: AsyncOpenAI,
    hall_ids: set[str],
    struct_ids: set[str],
    docs: dict[str, dict],
    repo_meta: dict[str, tuple],
    answer_source: str,
) -> Callable[[dict], Awaitable[Outcome]]:
    async def process(item: dict) -> Outcome:
        oid, split, style = item["id"], item["split"], item.get("style", "files")
        sc = json.loads((DATA / "source_cache" / f"{oid}.json").read_text())
        request = item["request"]
        context = _build_context(sc, docs.get(oid))
        if answer_source == "gold":
            # Use the gold PR fix verbatim in the assigned style — no generation.
            solution = _gold_answer(sc, style).strip()
            if _is_trivial_answer(solution):
                return Outcome(key=oid, ok=False, reason="trivial_gold_answer")
        else:
            solution = await generate_agent_solution_async(
                aclient,
                MODEL,
                request=request,
                context=context,
                reference=_reference(sc),
                style=style,
                completion_kwargs={"max_tokens": 1600},
            )
        if not solution:
            return Outcome(key=oid, ok=False, reason="solution_failed")
        if len(solution) > MAX_ANSWER_CHARS:
            return Outcome(key=oid, ok=False, reason="answer_too_long")

        context = await _ground_context(solution, context, sc, repo_meta.get(oid, (None, None)))

        if oid not in hall_ids:
            # Gold answers reference real repo symbols by construction; only the
            # generated path can hallucinate a reference worth dropping over.
            if answer_source == "generated" and _ungrounded_self_calls(solution, context):
                return Outcome(key=oid, ok=False, reason="clean_solution_ungrounded")
            return Outcome(
                key=oid,
                ok=True,
                record=_make_sample(oid, request, context, solution, [], None, split, style, None),
            )

        structural = oid in struct_ids
        prompt = STRUCTURAL_PROMPT if structural else INJECTION_PROMPT
        ask = (
            "Introduce 1-3 distinct plausible fabricated names (prefer 2-3 if the solution allows)."
            if structural
            else "Introduce 1-3 distinct request-grounded mistakes (prefer 2-3; mix value swaps "
            "and new/unrequested blocks)."
        )
        user = (
            f"Repository context:\n{context[:MAX_CONTEXT_CHARS]}\n\nDeveloper request:\n{request}"
            f"\n\nThe assistant's correct answer:\n{solution}\n\n{ask}"
        )
        res = await inject_async(
            aclient,
            MODEL,
            clean_answer=solution,
            hall_type="",
            system_prompt=prompt,
            user_msg=user,
            temperature=0.4,
            completion_kwargs={"max_tokens": 2000},
        )
        if not res.ok:
            return Outcome(key=oid, ok=False, reason=res.reason or "inject_failed")
        if structural and not _has_absent_fabrication(res.changes, context + "\n" + solution):
            return Outcome(key=oid, ok=False, reason="fabrication_not_absent")
        if structural:
            # Ground the real third-party API the fabrication replaced (exact-method Context7).
            sigs = await asyncio.to_thread(ground_fabricated_apis, res.changes)
            if sigs:
                context = (sigs + "\n\n" + context)[:MAX_CONTEXT_CHARS]
        default_type = "fabricated_api" if structural else "wrong_implementation"
        for lab in res.labels:
            if lab.get("label") not in CODE_AGENT_MAP:
                lab["label"] = default_type
        res = _map_menu_labels(res, "code_agent")
        by_hall = {c.get("hallucinated"): c for c in res.changes}
        for lab in res.labels:
            lab["explanation"] = by_hall.get(
                res.hallucinated_answer[lab["start"] : lab["end"]], {}
            ).get("explanation", "")
        rec = _make_sample(
            oid,
            request,
            context,
            res.hallucinated_answer,
            res.labels,
            res.reasoning,
            split,
            style,
            "structural" if structural else "intent",
        )
        return Outcome(key=oid, ok=True, record=rec)

    return process


def _print_progress(split: str, stats: dict, total: int) -> None:
    """Render a tail-able progress bar for one split (resume-aware)."""
    todo = stats.get("total", total)
    done = (total - todo) + stats.get("processed", 0)
    pct = done / total if total else 0.0
    filled = int(pct * 24)
    bar = "█" * filled + "░" * (24 - filled)
    print(
        f"  {split} [{bar}] {done}/{total} ({pct * 100:.0f}%) "
        f"ok={stats.get('ok', 0)} fail={stats.get('fail', 0)}",
        flush=True,
    )


def _record_key(record: dict) -> str:
    return f"{record['split']}::{record['metadata']['instance_id']}"


def main() -> None:
    """Generate code-agent hallucination samples via the batched runner."""
    import random

    ap = argparse.ArgumentParser(description="Generate code-agent hallucination samples.")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", default="data/v2/code_agent")
    ap.add_argument("--ratio", type=float, default=HALLUCINATION_RATIO)
    ap.add_argument("--repos", help="comma-separated instance-id prefixes to include (e.g. Lightning-AI)")
    ap.add_argument("--exclude-repos", help="comma-separated instance-id prefixes to exclude")
    ap.add_argument(
        "--edit-ratio",
        type=float,
        default=0.3,
        help="Share of answers rendered as targeted edits rather than full files.",
    )
    ap.add_argument(
        "--struct-ratio",
        type=float,
        default=0.25,
        help="Share of hallucinated samples that get a structural fabrication.",
    )
    ap.add_argument(
        "--answer-source",
        choices=["gold", "generated"],
        default="gold",
        help="gold: use the PR fix verbatim (fast, no generation); "
        "generated: LLM writes a coherent solution (slow, realistic prose).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    args = ap.parse_args()

    queries = {
        json.loads(line)["instance_id"]: json.loads(line).get("query")
        for line in (DATA / "queries.jsonl").read_text().splitlines()
        if line.strip()
    }
    cached = {p.stem for p in (DATA / "source_cache").glob("*.json")}
    instances = json.loads((DATA / "swebench_instances.json").read_text())
    instances = instances if isinstance(instances, list) else list(instances.values())
    splits = {inst["instance_id"]: inst.get("split", "train") for inst in instances}
    repo_meta = {
        inst["instance_id"]: (inst.get("repo"), inst.get("base_commit")) for inst in instances
    }
    docs = {}
    doc_path = DATA / "documentation.jsonl"
    if doc_path.exists():
        for line in doc_path.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                docs[obj["instance_id"]] = obj.get("docs") or {}
    items = [
        {"id": oid, "request": queries[oid], "split": splits.get(oid) or hash_split(oid)}
        for oid in sorted(queries)
        if oid in cached and queries.get(oid)
    ]
    if args.repos:
        keep = tuple(args.repos.split(","))
        items = [it for it in items if it["id"].startswith(keep)]
    if args.exclude_repos:
        drop = tuple(args.exclude_repos.split(","))
        items = [it for it in items if not it["id"].startswith(drop)]
    if args.limit:
        items = items[: args.limit]

    rng = random.Random(args.seed)  # noqa: S311 - reproducibility, not crypto
    for it in items:
        if args.answer_source == "gold":
            r = rng.random()
            it["style"] = "function" if r < 0.45 else ("fragment" if r < 0.72 else "edit")
        else:
            it["style"] = "edit" if rng.random() < args.edit_ratio else "files"
    n_hall = int(len(items) * args.ratio)
    hall_items = rng.sample(items, min(n_hall, len(items)))
    hall_ids = {it["id"] for it in hall_items}
    n_struct = int(len(hall_items) * args.struct_ratio)
    struct_ids = {it["id"] for it in rng.sample(hall_items, min(n_struct, len(hall_items)))}

    print(
        f"LLM: {MODEL} @ {API_BASE_URL} | {len(items)} instances, {len(hall_ids)} hallucination "
        f"targets ({len(struct_ids)} structural)"
    )
    aclient = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        split_items = [it for it in items if it["split"] == split]
        if not split_items:
            continue
        total = len(split_items)
        stats = run_batched_sync(
            split_items,
            _make_process(aclient, hall_ids, struct_ids, docs, repo_meta, args.answer_source),
            out_path=out_dir / f"{split}.jsonl",
            failures_path=out_dir / f"{split}.failures.jsonl",
            key_of=lambda it: f"{it['split']}::{it['id']}",
            record_key=_record_key,
            batch_size=args.batch_size,
            progress_every=args.batch_size,
            on_progress=lambda s, sp=split, t=total: _print_progress(sp, s, t),
        )
        print(f"{split}: {stats}")


if __name__ == "__main__":
    main()
