#!/usr/bin/env python3
r"""Generate code-hallucination samples from the tool-output-extraction dataset.

Source: ``KRLabsOrg/tool-output-extraction-swebench``. Each row gives a focused
extraction query, the original issue (background task), a verbose tool output
(numbered source lines), and the gold ``<relevant_lines>`` evidence block.

We turn each row into a hallucination-detection sample in two steps:

1. **Answer generation** — feed the query + gold relevant lines to the model and
   ask for a short, natural coding-agent explanation that references the code
   (identifiers, line numbers, behavior). This clean answer is grounded by
   construction in the gold evidence.
2. **Injection** — for the hallucinated subset, corrupt that explanation with
   localized errors that contradict the tool output: fabricated identifiers
   (structural), wrong line refs / values (behavioral), or claims the code does
   not support (semantic).

The sample context is the full tool output, so a detector must verify the answer
against the actual code. Output is written directly in the LettuceDetect v2
``HallucinationSample`` schema with unified-taxonomy fields.

Usage::

    API_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \\
        MODEL=Qwen/Qwen3.6-35B-A3B \\
        python scripts/generate_squeez_hallucinations.py --limit 20 --out data/v2/squeez
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI  # noqa: E402

from lettucedetect.generation.answers import generate_grounded_answer_async  # noqa: E402
from lettucedetect.generation.assembly import format_prompt  # noqa: E402
from lettucedetect.generation.injection import (  # noqa: E402
    InjectionResult,
    inject_taxonomy_async,
    sample_target,
)
from lettucedetect.generation.runner import Outcome, run_batched_sync  # noqa: E402

# ── Config (env-overridable, mirrors the code pipeline) ─────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.6-35B-A3B")
HALLUCINATION_RATIO = float(os.environ.get("HALLUCINATION_RATIO", "0.4"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "32000"))

# Tool outputs are code-like, so we inject the code-relevant taxonomy subtypes.
# The universal injector handles the prompting; we just choose the distribution.
ALLOWED_TARGETS = {
    "contradiction": ["value", "numerical"],
    "unsupported_addition": ["claim", "behavior"],
    "fabricated_reference": ["identifier", "section"],
}
CONTEXT_MODALITY = "tool_output"

# ── Prompts ─────────────────────────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT = """\
You are a coding assistant helping a developer understand a codebase. You are \
given a focused question and the exact relevant lines of source code (with line \
numbers) that answer it. Write a short, natural explanation of what that code \
does and where it is, as if replying to the developer.

Your response MUST:
- Be 2-4 sentences of plain prose (no code block, no bullet lists).
- Reference concrete identifiers (function/method/class/variable names) and line \
numbers that appear in the provided relevant lines.
- Accurately describe what the code does, grounded ONLY in the provided lines.

Your response must NOT:
- Invent identifiers or behavior not present in the provided lines.
- Include phrases like "Here's the answer" or "I'll help you with that".
- Quote large code blocks verbatim; explain in prose instead.
"""

# ── Squeez row parsing ──────────────────────────────────────────────────────────


def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_row(row: dict) -> dict | None:
    """Pull (query, background, tool_output, relevant_lines) out of a squeez row."""
    prompt = row.get("prompt", "")
    response = row.get("response", "")
    query = _extract_tag(prompt, "query")
    background = _extract_tag(prompt, "background_task")
    background = re.sub(r"<!--.*?-->", "", background, flags=re.DOTALL).strip()
    tool_output = _extract_tag(prompt, "tool_output")
    relevant_lines = _extract_tag(response, "relevant_lines") or response.strip()
    if not query or not tool_output or not relevant_lines:
        return None
    return {
        "query": query,
        "background": background,
        "tool_output": tool_output,
        "relevant_lines": relevant_lines,
        "instance_id": row.get("metadata", {}).get("instance_id", ""),
        "tool_type": row.get("metadata", {}).get("tool_type", ""),
    }


def build_context(tool_output: str) -> str:
    """Return the grounding context: the tool's output (truncated to the budget)."""
    return f"Tool output:\n```\n{tool_output}\n```"[:MAX_CONTEXT_CHARS]


def make_sample(parsed: dict, answer: str, hall_result: InjectionResult | None, split: str) -> dict:
    """Build one v2 HallucinationSample dict (clean or hallucinated)."""
    is_hall = hall_result is not None and hall_result.ok
    if is_hall:
        final_answer = hall_result.hallucinated_answer
        labels = hall_result.labels  # already carry category + subcategory
        sample_cat = labels[0]["category"] if labels else None
        sample_sub = labels[0].get("subcategory") if labels else None
    else:
        final_answer = answer
        labels = []
        sample_cat, sample_sub = None, None

    context = build_context(parsed["tool_output"])
    question = parsed["query"] or None
    return {
        "prompt": format_prompt(context, question),
        "context": context,
        "question": question,
        "answer": final_answer,
        "labels": labels,
        "split": split,
        "task_type": "code_generation",
        "dataset": "lettucedetect-tool-output",
        "language": "en",
        "context_modality": CONTEXT_MODALITY,
        "category": sample_cat,
        "subcategory": sample_sub,
        "metadata": {
            "instance_id": parsed["instance_id"],
            "tool_type": parsed["tool_type"],
            "is_hallucinated": is_hall,
            "injector_model": MODEL if is_hall else None,
        },
    }


def _build_items(rows: list[dict], split: str, ratio: float, rng: random.Random) -> list[dict]:
    """Parse rows and assign hallucination targets. Each item is one work unit."""
    parsed_rows = []
    for row in rows:
        p = parse_row(row)
        if p:
            parsed_rows.append(p)
    n_targets = int(len(parsed_rows) * ratio)
    hall_idx = set(rng.sample(range(len(parsed_rows)), min(n_targets, len(parsed_rows))))
    items = []
    for i, p in enumerate(parsed_rows):
        target = sample_target(rng, ALLOWED_TARGETS) if i in hall_idx else (None, None)
        items.append({"parsed": p, "split": split, "target": target})
    return items


def _item_key(item: dict) -> str:
    return f"{item['split']}::{item['parsed']['instance_id']}::{item['parsed']['tool_type']}"


def _record_key(record: dict) -> str:
    """Derive the resumability key from a written sample (no synthetic field needed)."""
    meta = record["metadata"]
    return f"{record['split']}::{meta['instance_id']}::{meta['tool_type']}"


def _make_process(aclient: AsyncOpenAI) -> Callable[[dict], Awaitable[Outcome]]:
    """Build the async per-item processor for the runner."""

    async def process(item: dict) -> Outcome:
        parsed = item["parsed"]
        key = _item_key(item)

        answer = await generate_grounded_answer_async(
            aclient,
            MODEL,
            question=parsed["query"],
            evidence=parsed["relevant_lines"],
            extra_context=parsed["background"],
            system_prompt=ANSWER_SYSTEM_PROMPT,
            temperature=0.7,
            completion_kwargs={"max_tokens": 400},
        )
        if not answer:
            return Outcome(key=key, ok=False, reason="answer_generation_failed")

        hall_result = None
        category, subcategory = item["target"]
        if category is not None:
            hall_result = await inject_taxonomy_async(
                aclient,
                MODEL,
                context=parsed["tool_output"],
                clean_answer=answer,
                category=category,
                subcategory=subcategory,
                modality=CONTEXT_MODALITY,
                temperature=0.8,
                completion_kwargs={"max_tokens": 600},
            )
            # Injection failure is not fatal: keep the valid clean answer instead,
            # but record the reason so we can inspect injection yield.
            if not hall_result.ok:
                sample = make_sample(parsed, answer, None, item["split"])
                return Outcome(
                    key=key, ok=True, record=sample, extra={"inject_failed": hall_result.reason}
                )

        sample = make_sample(parsed, answer, hall_result, item["split"])
        return Outcome(key=key, ok=True, record=sample)

    return process


def main() -> None:
    """Generate tool-output hallucination samples per split via the batched runner."""
    ap = argparse.ArgumentParser(description="Generate squeez tool-output hallucination samples.")
    ap.add_argument("--limit", type=int, default=20, help="Max rows per split (None=all).")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--out", type=str, default="data/v2/squeez")
    ap.add_argument("--ratio", type=float, default=HALLUCINATION_RATIO)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "16")))
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"LLM: {MODEL} @ {API_BASE_URL}  (batch_size={args.batch_size})")
    aclient = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    ds = load_dataset("KRLabsOrg/tool-output-extraction-swebench")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)  # noqa: S311 - seeded reproducibility, not crypto
    process = _make_process(aclient)

    for split in args.splits:
        rows = list(ds[split])
        if args.limit:
            rows = rows[: args.limit]
        items = _build_items(rows, split, args.ratio, rng)
        out_path = out_dir / f"{split}.jsonl"
        failures_path = out_dir / f"{split}.failures.jsonl"

        stats = run_batched_sync(
            items,
            process,
            out_path=out_path,
            failures_path=failures_path,
            key_of=_item_key,
            record_key=_record_key,
            batch_size=args.batch_size,
            on_progress=lambda s: print(f"  {split}: {s}"),
        )
        print(f"{split}: {stats} -> {out_path}")


if __name__ == "__main__":
    main()
