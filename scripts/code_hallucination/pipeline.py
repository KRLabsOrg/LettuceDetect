#!/usr/bin/env python3
"""Prep for the code-agent hallucination dataset: build repo context + requests.

This produces the inputs the generator reads — per-instance source files
(``source_cache/``) and rewritten developer requests (``queries.jsonl``):

    1. load SWE-bench instances
    2. fetch the patch-touched source files at the base commit
    3. rewrite each issue into a developer request

Generation (the coherent agent solution + the request-grounded intent mistakes)
lives in ``scripts/generate_code_agent_hallucinations.py``, which consumes these.

Usage::

    python -m scripts.code_hallucination.pipeline --all        # phases 1-3
    python -m scripts.code_hallucination.pipeline --phase 2
    OPENAI_API_KEY=... API_BASE_URL=... MODEL=... \
        python -m scripts.code_hallucination.pipeline --phase 3
"""

import argparse

from . import config
from .config import API_BASE_URL, API_KEY, MODEL


def filter_instances_by_splits(instances: list[dict], splits: list[str] | None) -> list[dict]:
    """Optionally filter instances to a subset of SWE-bench splits."""
    if not splits:
        return instances
    split_set = set(splits)
    filtered = [inst for inst in instances if inst.get("split") in split_set]
    print(f"Using splits {sorted(split_set)}: {len(filtered)}/{len(instances)} instances")
    return filtered


def main() -> None:
    """Run the prep phases (1: load, 2: fetch source, 3: rewrite requests)."""
    parser = argparse.ArgumentParser(description="Code-agent dataset prep (context + requests)")
    parser.add_argument("--phase", nargs="+", type=int, choices=range(1, 4), help="Run phase(s)")
    parser.add_argument("--all", action="store_true", help="Run all prep phases (1-3)")
    parser.add_argument("--api-key", type=str, default=API_KEY, help="LLM API key")
    parser.add_argument("--base-url", type=str, default=API_BASE_URL, help="LLM API base URL")
    parser.add_argument("--model", type=str, default=MODEL, help="LLM model name")
    parser.add_argument("--output-dir", type=str, help="Optional output directory for prep files")
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "dev", "test"], help="Optional SWE-bench splits"
    )
    args = parser.parse_args()

    if args.output_dir:
        print(f"Using output directory: {config.set_output_dir(args.output_dir)}")

    if not args.phase and not args.all:
        parser.print_help()
        return

    for phase in sorted([1, 2, 3] if args.all else args.phase):
        print(f"\n{'#' * 60}\n# Phase {phase}\n{'#' * 60}\n")
        if phase == 1:
            from .swebench_loader import run

            run()
        elif phase == 2:
            from .source_fetcher import run
            from .swebench_loader import load_instances

            run(filter_instances_by_splits(load_instances(), args.splits))
        elif phase == 3:
            from .query_rewriter import run
            from .swebench_loader import load_instances

            run(
                filter_instances_by_splits(load_instances(), args.splits),
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
            )

    print("\nPrep complete! Now run scripts/generate_code_agent_hallucinations.py")


if __name__ == "__main__":
    main()
