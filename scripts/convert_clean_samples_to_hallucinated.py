#!/usr/bin/env python3
"""Convert clean samples to hallucinated ones in place, raising the class balance.

For each split, a seeded selection of clean samples is run through the same
injection engine (and prompts) its source used at generation time. On success
the sample's answer, labels, and category fields are replaced; on QC failure
the clean sample is kept unchanged, so nothing is ever lost. Each sample stays
single-class (no clean/hallucinated twins).

Usage:
    API_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY MODEL=... \
        python scripts/convert_clean_samples_to_hallucinated.py \
        --data data/v2/readme --source markdown [--target-rate 0.5] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import shutil
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

from openai import AsyncOpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os  # noqa: E402

from lettucedetect.generation.injection import (  # noqa: E402
    InjectionResult,
    inject_menu_async,
    inject_taxonomy_async,
)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
MODEL = os.environ.get("MODEL", "google/gemma-4-31B-it")

# squeez rotates through code-relevant taxonomy targets (matches its generator).
Injector = Callable[[AsyncOpenAI, dict, int], Awaitable[InjectionResult]]

SQUEEZ_TARGETS = [
    ("contradiction", "value"),
    ("contradiction", "numerical"),
    ("unsupported_addition", "claim"),
    ("unsupported_addition", "behavior"),
    ("fabricated_reference", "identifier"),
    ("fabricated_reference", "section"),
]


def _menu_injector(source_key: str, prompt: str) -> 'Injector':
    async def run(client: AsyncOpenAI, sample: dict, _i: int) -> InjectionResult:
        return await inject_menu_async(
            client,
            MODEL,
            context=sample["context"],
            clean_answer=sample["answer"],
            system_prompt=prompt,
            source=source_key,
            temperature=0.8,
        )

    return run


async def _squeez_injector(client: AsyncOpenAI, sample: dict, i: int) -> InjectionResult:
    category, subcategory = SQUEEZ_TARGETS[i % len(SQUEEZ_TARGETS)]
    return await inject_taxonomy_async(
        client,
        MODEL,
        context=sample["context"],
        clean_answer=sample["answer"],
        category=category,
        subcategory=subcategory,
        modality="tool_output",
        temperature=0.8,
        completion_kwargs={"max_tokens": 600},
    )


def _build_injector(source: str) -> 'Injector':
    if source == "squeez":
        return _squeez_injector
    if source == "paper":
        from scripts.generate_acl_hallucinations import PAPER_INJECTION_PROMPT

        return _menu_injector("paper", PAPER_INJECTION_PROMPT)
    if source == "markdown":
        from lettucedetect.generation.doc_source import GENERIC_MARKDOWN_INJECTION_PROMPT

        return _menu_injector("markdown", GENERIC_MARKDOWN_INJECTION_PROMPT)
    raise SystemExit(f"unknown --source {source!r}")


def _apply(sample: dict, res: InjectionResult) -> None:
    sample["answer"] = res.hallucinated_answer
    sample["labels"] = [
        {k: lab.get(k) for k in ("start", "end", "label", "category", "subcategory")}
        for lab in res.labels
    ]
    cats = collections.Counter(lab["category"] for lab in sample["labels"])
    top = cats.most_common(1)[0][0]
    sample["category"] = top
    sample["subcategory"] = next(
        lab["subcategory"] for lab in sample["labels"] if lab["category"] == top
    )
    if isinstance(sample.get("metadata"), dict):
        sample["metadata"]["is_hallucinated"] = True
        sample["metadata"]["converted_from_clean"] = True


async def convert_split(
    path: Path, injector: Injector, target_rate: float, batch: int, seed: int
) -> None:
    """Convert clean samples in one split file until the rate is reached."""
    lines = path.read_text().splitlines()
    samples = [json.loads(line) for line in lines if line.strip()]
    hall = sum(1 for s in samples if s["labels"])
    needed = max(0, int(target_rate * len(samples)) - hall) if samples else 0
    cleans = [i for i, s in enumerate(samples) if not s["labels"] and len(s["answer"]) >= 80]
    rng = random.Random(seed)  # noqa: S311 - reproducibility, not crypto
    rng.shuffle(cleans)
    print(f"{path.name}: {hall}/{len(samples)} hall, need {needed} conversions, "
          f"{len(cleans)} candidates")
    if not needed:
        return

    client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    sem = asyncio.Semaphore(batch)
    converted = 0
    attempted = 0
    queue = collections.deque(cleans)

    async def attempt(idx: int, n: int) -> tuple[int, InjectionResult]:
        async with sem:
            try:
                return idx, await injector(client, samples[idx], n)
            except Exception as exc:
                return idx, InjectionResult(ok=False, reason=f"error: {exc}")

    while converted < needed and queue:
        wave = [queue.popleft() for _ in range(min(len(queue), max(needed - converted, batch)))]
        results = await asyncio.gather(*(attempt(i, attempted + n) for n, i in enumerate(wave)))
        attempted += len(wave)
        for idx, res in results:
            if res.ok and converted < needed:
                _apply(samples[idx], res)
                converted += 1
        path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n")
        print(f"  {path.name}: {converted}/{needed} converted ({attempted} attempted)")
    print(f"{path.name}: done — {converted}/{needed} converted, {attempted} attempts")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Convert clean samples to hallucinated in place.")
    ap.add_argument("--data", type=Path, required=True, help="Source dir with split JSONLs.")
    ap.add_argument("--source", required=True, choices=["markdown", "paper", "squeez"])
    ap.add_argument("--target-rate", type=float, default=0.5)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    injector = _build_injector(args.source)
    backup = args.data / "quarantine" / "pre_50pct_backup"
    backup.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        path = args.data / f"{split}.jsonl"
        if not path.exists():
            continue
        bak = backup / f"{split}.jsonl.bak"
        if not bak.exists():
            shutil.copy(path, bak)
        asyncio.run(convert_split(path, injector, args.target_rate, args.batch_size, args.seed))


if __name__ == "__main__":
    main()
