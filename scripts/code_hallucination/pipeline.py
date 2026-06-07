#!/usr/bin/env python3
"""Orchestrator for the code hallucination dataset pipeline.

Usage:
    # Run all phases
    python -m scripts.code_hallucination.pipeline --all

    # Run specific phase
    python -m scripts.code_hallucination.pipeline --phase 1

    # Test with a few examples
    python -m scripts.code_hallucination.pipeline --test 5

    # Override LLM settings via env vars:
    OPENAI_API_KEY=xxx API_BASE_URL=https://api.groq.com/openai/v1 MODEL=moonshotai/kimi-k2-instruct-0905 \
        python -m scripts.code_hallucination.pipeline --test 10
"""

import argparse
import json
import random

from . import config
from .config import API_BASE_URL, API_KEY, MODEL


def load_source_cache(instance_ids: list[str]) -> dict[str, dict]:
    """Load source cache keyed by original_id (strips sub-instance suffix if present)."""
    cache = {}
    for iid in instance_ids:
        original = iid.split("::")[0]
        if original in cache:
            continue
        cache_path = config.SOURCE_CACHE_DIR / f"{original}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                cache[original] = json.load(f)
    return cache


def load_jsonl_all(path) -> list:
    """Load a JSONL file as a list of dicts."""
    result = []
    if not path.exists():
        return result
    with open(path) as f:
        for line in f:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return result


def load_jsonl_dict(path, key="instance_id", value_key=None) -> dict:
    """Load a JSONL file into a dict keyed by instance_id."""
    result = {}
    if not path.exists():
        return result
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
                if value_key:
                    result[entry[key]] = entry[value_key]
                else:
                    result[entry[key]] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    return result


def filter_instances_by_splits(instances: list[dict], splits: list[str] | None) -> list[dict]:
    """Optionally filter instances to a subset of SWE-bench splits."""
    if not splits:
        return instances
    split_set = set(splits)
    filtered = [inst for inst in instances if inst.get("split") in split_set]
    print(f"Using splits {sorted(split_set)}: {len(filtered)}/{len(instances)} instances")
    return filtered


def run_test(n: int = 5, api_key: str = API_KEY, base_url: str = API_BASE_URL, model: str = MODEL):
    """Run a quick test with n instances from the test split."""
    print("=" * 60)
    print(f"TEST MODE: Running pipeline with {n} instances")
    print(f"LLM: {model} @ {base_url}")
    print("=" * 60)

    from .swebench_loader import load_all_splits

    # Load and filter to test split
    all_instances = load_all_splits()
    test_instances = [i for i in all_instances if i["split"] == "test"]

    # Take n random instances
    rng = random.Random(42)
    selected = rng.sample(test_instances, min(n, len(test_instances)))
    print(f"Selected {len(selected)} test instances")

    # Save temporary instances
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    test_path = config.DATA_DIR / "test_instances.json"
    with open(test_path, "w") as f:
        json.dump(selected, f, indent=2)

    # Phase 2: Fetch sources (use GitHub API for test mode — no cloning needed)
    from .source_fetcher import run as run_fetch

    sources = run_fetch(selected, use_github_api=True)

    if not sources:
        print("No sources fetched, aborting test")
        return

    # Phase 3: Rewrite queries
    from .query_rewriter import run as run_queries

    run_queries(selected, api_key=api_key, base_url=base_url, model=model)

    # Phase 4: Context7 docs
    from .context7_docs import run as run_docs

    run_docs(selected)

    # Phase 5: Assign formats (needs LLM for code_with_explanation)
    from .format_builder import run as run_formats

    queries_dict = load_jsonl_dict(config.QUERIES_PATH, value_key="query")
    run_formats(selected, api_key=api_key, base_url=base_url, model=model, queries=queries_dict)

    # Phase 8: Select targets (before phase 6)
    from .splitter import select_format_targets

    format_entries_list = load_jsonl_all(config.FORMATS_PATH)
    formats = {e["instance_id"]: e for e in format_entries_list}
    docs = load_jsonl_dict(config.DOCS_PATH, value_key="docs")
    targets = select_format_targets(format_entries_list)

    # Phase 6: Inject hallucinations
    from .hallucination_injector import run as run_inject

    to_inject = [e for e in format_entries_list if e["instance_id"] in targets]
    original_ids = {e.get("original_id", e["instance_id"]) for e in to_inject}
    sc = load_source_cache(list(original_ids))
    run_inject(
        to_inject,
        formats,
        queries_dict,
        docs=docs,
        source_cache=sc,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    # Phase 7: Assemble
    from .sample_assembler import run as run_assemble

    hallucinations = load_jsonl_dict(config.HALLUCINATED_PATH)
    targets = set(hallucinations.keys())
    samples, metadata = run_assemble(selected, queries_dict, docs, formats, hallucinations, targets)

    # Phase 9: Validate
    from .validator import run as run_validate

    run_validate(samples, metadata)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print(f"Generated {len(samples)} samples from {n} test instances")

    # Show a sample
    if samples:
        print("\n--- Sample Example ---")
        s = samples[0]
        print(f"Prompt length: {len(s['prompt'])} chars")
        print(f"Answer length: {len(s['answer'])} chars")
        print(f"Labels: {len(s['labels'])}")
        print(f"Split: {s['split']}")
        print(f"Answer preview: {s['answer'][:200]}...")


def main():
    parser = argparse.ArgumentParser(description="Code hallucination dataset pipeline")
    parser.add_argument(
        "--phase", nargs="+", type=int, choices=range(1, 10), help="Run specific phase(s)"
    )
    parser.add_argument("--all", action="store_true", help="Run all phases")
    parser.add_argument("--test", type=int, metavar="N", help="Test with N instances")
    parser.add_argument("--api-key", type=str, default=API_KEY, help="LLM API key")
    parser.add_argument("--base-url", type=str, default=API_BASE_URL, help="LLM API base URL")
    parser.add_argument("--model", type=str, default=MODEL, help="LLM model name")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Optional output directory for all intermediate and final pipeline files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "dev", "test"],
        help="Optional SWE-bench splits to operate on",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = config.set_output_dir(args.output_dir)
        print(f"Using output directory: {output_dir}")

    if args.test:
        run_test(args.test, api_key=args.api_key, base_url=args.base_url, model=args.model)
        return

    if not args.phase and not args.all:
        parser.print_help()
        return

    phases = list(range(1, 10)) if args.all else args.phase

    for phase in sorted(phases):
        print(f"\n{'#' * 60}")
        print(f"# Running Phase {phase}")
        print(f"{'#' * 60}\n")

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
        elif phase == 4:
            from .context7_docs import run
            from .swebench_loader import load_instances

            run(filter_instances_by_splits(load_instances(), args.splits))
        elif phase == 5:
            from .format_builder import run
            from .swebench_loader import load_instances

            queries = load_jsonl_dict(config.QUERIES_PATH, value_key="query")
            run(
                filter_instances_by_splits(load_instances(), args.splits),
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                queries=queries,
            )
        elif phase == 6:
            from .hallucination_injector import run
            from .splitter import select_format_targets

            format_entries_list = load_jsonl_all(config.FORMATS_PATH)
            formats = {e["instance_id"]: e for e in format_entries_list}
            queries = load_jsonl_dict(config.QUERIES_PATH, value_key="query")
            docs = load_jsonl_dict(config.DOCS_PATH, value_key="docs")
            targets = select_format_targets(format_entries_list)
            to_inject = [e for e in format_entries_list if e["instance_id"] in targets]
            original_ids = {e.get("original_id", e["instance_id"]) for e in to_inject}
            sc = load_source_cache(list(original_ids))
            run(
                to_inject,
                formats,
                queries,
                docs=docs,
                source_cache=sc,
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
            )
        elif phase == 7:
            from .sample_assembler import run
            from .swebench_loader import load_instances

            instances = filter_instances_by_splits(load_instances(), args.splits)
            queries = load_jsonl_dict(config.QUERIES_PATH, value_key="query")
            docs = load_jsonl_dict(config.DOCS_PATH, value_key="docs")
            formats = load_jsonl_dict(config.FORMATS_PATH)
            hallucinations = load_jsonl_dict(config.HALLUCINATED_PATH)
            # Targets are all sub-instance IDs that have a hallucination entry
            targets = set(hallucinations.keys())
            run(instances, queries, docs, formats, hallucinations, targets)
        elif phase == 8:
            from .splitter import run
            from .swebench_loader import load_instances

            run(filter_instances_by_splits(load_instances(), args.splits))
        elif phase == 9:
            from .validator import run

            run()

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
