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
from scripts.code_hallucination.source_fetcher import build_source_context  # noqa: E402

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.6-35B-A3B")
HALLUCINATION_RATIO = float(os.environ.get("HALLUCINATION_RATIO", "0.5"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "28000"))
DATA = PROJECT_ROOT / "data" / "code_hallucination"

INJECTION_PROMPT = """\
You introduce ONE realistic mistake into a coding assistant's CORRECT solution — the kind a
competent coding agent actually makes. The code stays plausible, well-formed, and uses real
names; it just fails to do what the developer asked. Return ONLY a small replacement edit as
JSON; the pipeline applies it.

The mistake MUST be judgeable by comparing the solution to the DEVELOPER REQUEST (and the
repository context): a reviewer reading the request and the answer should see the solution does
NOT do what was asked. Do NOT rely on outside library knowledge or math.

Pick the type that fits and label it:
- wrong_implementation: the code addresses the request but with WRONG logic, condition, field,
  or value, so it does not actually achieve what was asked — e.g. it checks/handles the wrong
  case, uses the wrong field or default, or sets/returns the wrong thing. (Most common — prefer this.)
- unrequested_change: the code also does something the request did NOT ask for — an extra side
  effect, an unrelated edit, or broadened scope beyond the request.

Make each mistake PLAUSIBLE and subtle: the kind of slip a good agent makes, not obvious
vandalism. The edited code must stay syntactically valid and look reasonable on its own. Do NOT
make a no-op (each edit must change behavior), do NOT introduce a name that exists nowhere, no
hint words.

Rules: 1-3 edits, each a DISTINCT realistic mistake in a different place (use fewer if only one
genuinely fits — do not force extras). Each "original" is an exact substring appearing once in
the answer; the replacement is the smallest expression/statement that captures the change.
Respond ONLY with JSON (no prose, no code fence):
{"changes":[{"original":"...","hallucinated":"...","hallucination_type":"wrong_implementation|unrequested_change","explanation":"how this fails to satisfy the developer request"}]}
"""


STRUCTURAL_PROMPT = """\
You introduce ONE realistic FABRICATION into a coding assistant's CORRECT solution — the kind of
slip an agent makes when it misremembers an API. Replace a real call, attribute, or
keyword-argument name with a plausible-looking one that does NOT exist: it must appear NOWHERE in
the repository context or in the answer. It must look idiomatic for the object or library (not a
typo, not gibberish) — e.g. `config.get_value(x)` where the real method is `config.get(x)`,
`timeout_seconds=5` where the parameter is `timeout=5`, or `resp.json_body` where it is `resp.json()`.

The surrounding code stays plausible and syntactically valid. Do NOT rename to a name that already
exists anywhere in the code or context. Do NOT add hint words.

Rules: 1-2 edits, each a DISTINCT fabricated name in a different place. Each "original" is an exact
substring appearing once in the answer; "hallucinated" is the smallest expression containing the
fabricated name. Respond ONLY with JSON (no prose, no code fence):
{"changes":[{"original":"...","hallucinated":"...","hallucination_type":"fabricated_api","explanation":"which name is fabricated and why it does not exist"}]}
"""

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _has_absent_fabrication(changes: list[dict], haystack: str) -> bool:
    """Check whether some edit introduces an identifier absent from context + clean answer."""
    present = set(_IDENT.findall(haystack))
    for change in changes:
        original_names = set(_IDENT.findall(change.get("original", "")))
        for name in set(_IDENT.findall(change.get("hallucinated", ""))) - original_names:
            if len(name) > 2 and name not in present:
                return True
    return False


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


def _make_process(
    aclient: AsyncOpenAI, hall_ids: set[str], struct_ids: set[str], docs: dict[str, dict]
) -> Callable[[dict], Awaitable[Outcome]]:
    async def process(item: dict) -> Outcome:
        oid, split, style = item["id"], item["split"], item.get("style", "files")
        sc = json.loads((DATA / "source_cache" / f"{oid}.json").read_text())
        request = item["request"]
        context = _build_context(sc, docs.get(oid))
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

        if oid not in hall_ids:
            return Outcome(
                key=oid,
                ok=True,
                record=_make_sample(oid, request, context, solution, [], None, split, style, None),
            )

        structural = oid in struct_ids
        prompt = STRUCTURAL_PROMPT if structural else INJECTION_PROMPT
        ask = (
            "Introduce 1-2 plausible fabricated names."
            if structural
            else "Introduce 1-3 realistic, request-grounded mistakes."
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


def _record_key(record: dict) -> str:
    return f"{record['split']}::{record['metadata']['instance_id']}"


def main() -> None:
    """Generate code-agent hallucination samples via the batched runner."""
    import random

    ap = argparse.ArgumentParser(description="Generate code-agent hallucination samples.")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", default="data/v2/code_agent")
    ap.add_argument("--ratio", type=float, default=HALLUCINATION_RATIO)
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
    if args.limit:
        items = items[: args.limit]

    rng = random.Random(args.seed)  # noqa: S311 - reproducibility, not crypto
    for it in items:
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
        stats = run_batched_sync(
            split_items,
            _make_process(aclient, hall_ids, struct_ids, docs),
            out_path=out_dir / f"{split}.jsonl",
            failures_path=out_dir / f"{split}.failures.jsonl",
            key_of=lambda it: f"{it['split']}::{it['id']}",
            record_key=_record_key,
            batch_size=args.batch_size,
            on_progress=lambda s: print(f"  {split}: {s}"),
        )
        print(f"{split}: {stats}")


if __name__ == "__main__":
    main()
