"""Phase 7: Assemble final HallucinationSample format."""

import json

from .config import (
    DATASET_PATH,
    MAX_ANSWER_CHARS,
    MAX_PROMPT_CHARS,
    METADATA_PATH,
    SOURCE_CACHE_DIR,
)


def build_prompt(
    source_files: dict[str, str],
    documentation: dict[str, str],
    user_query: str,
) -> str:
    """Build the prompt (context) for a sample.

    Format: source files + documentation + user query.
    """
    parts = []

    for filepath, content in source_files.items():
        parts.append(f"File: {filepath}\n```python\n{content}\n```")

    for lib, doc in documentation.items():
        parts.append(f"Documentation for {lib}:\n{doc}")

    parts.append(f"User request: {user_query}")

    prompt = "\n\n".join(parts)
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS]
    return prompt


def assemble_samples(
    instances: list[dict],
    source_cache: dict[str, dict],
    queries: dict[str, str],
    docs: dict[str, dict],
    formats: dict[str, dict],
    hallucinations: dict[str, dict],
    hallucination_instance_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    """Assemble all samples into HallucinationSample format.

    Returns (samples, metadata) where each instance maps to exactly 1 sample.
    """
    samples = []
    metadata = []

    for inst in instances:
        instance_id = inst["instance_id"]
        split = inst["split"]
        repo = inst["repo"]

        # Skip if no source data or format
        if instance_id not in source_cache or instance_id not in formats:
            continue

        source_data = source_cache[instance_id]
        fmt_data = formats[instance_id]
        query = queries.get(instance_id, inst.get("problem_statement", "")[:500])
        doc = docs.get(instance_id, {})

        # Build prompt from source files + docs + query
        source_files = source_data.get("source_files", {})
        prompt = build_prompt(source_files, doc, query)

        if instance_id in hallucination_instance_ids and instance_id in hallucinations:
            # Hallucinated sample
            hall_data = hallucinations[instance_id]
            if len(hall_data.get("hallucinated_answer", "")) > MAX_ANSWER_CHARS:
                continue
            sample = {
                "prompt": prompt,
                "answer": hall_data["hallucinated_answer"],
                "labels": hall_data["labels"],
                "split": split,
                "task_type": "code_generation",
                "dataset": "swebench_code",
                "language": "en",
            }
            meta = {
                "instance_id": instance_id,
                "repo": repo,
                "split": split,
                "is_lite": inst.get("is_lite", False),
                "format_type": hall_data.get("format_type", fmt_data.get("format_type")),
                "hallucination_type": hall_data.get("hallucination_type"),
                "injector_model": hall_data.get("injector_model"),
                "is_hallucinated": True,
            }
        else:
            # Clean sample
            answer = fmt_data.get("answer", "")
            if not answer.strip():
                continue
            if len(answer) > MAX_ANSWER_CHARS:
                continue

            # Reject code_with_explanation with unbalanced fences
            if fmt_data.get("format_type") == "code_with_explanation":
                fence_count = answer.count("```")
                if fence_count % 2 != 0 or fence_count == 0:
                    continue

            sample = {
                "prompt": prompt,
                "answer": answer,
                "labels": [],
                "split": split,
                "task_type": "code_generation",
                "dataset": "swebench_code",
                "language": "en",
            }
            meta = {
                "instance_id": instance_id,
                "repo": repo,
                "split": split,
                "is_lite": inst.get("is_lite", False),
                "format_type": fmt_data.get("format_type"),
                "hallucination_type": None,
                "injector_model": None,
                "is_hallucinated": False,
            }

        samples.append(sample)
        metadata.append(meta)

    return samples, metadata


def run(
    instances: list[dict],
    queries: dict[str, str],
    docs: dict[str, dict],
    formats: dict[str, dict],
    hallucinations: dict[str, dict],
    hallucination_instance_ids: set[str],
):
    """Run Phase 7: Assemble all samples."""
    print("=" * 60)
    print("Phase 7: Sample Assembly")
    print("=" * 60)

    # Load source cache
    source_cache = {}
    for inst in instances:
        cache_path = SOURCE_CACHE_DIR / f"{inst['instance_id']}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                source_cache[inst["instance_id"]] = json.load(f)

    print(f"Source cache: {len(source_cache)} instances")
    print(f"Queries: {len(queries)}")
    print(f"Docs: {len(docs)}")
    print(f"Formats: {len(formats)}")
    print(f"Hallucinations: {len(hallucinations)}")
    print(f"Hallucination targets: {len(hallucination_instance_ids)}")

    samples, metadata = assemble_samples(
        instances,
        source_cache,
        queries,
        docs,
        formats,
        hallucinations,
        hallucination_instance_ids,
    )

    # Save
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(DATASET_PATH, "w") as f:
        json.dump(samples, f, indent=2)

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # Stats
    n_clean = sum(1 for s in samples if not s["labels"])
    n_hall = sum(1 for s in samples if s["labels"])
    print(f"\nTotal samples: {len(samples)}")
    print(f"  Clean: {n_clean} ({n_clean * 100 // max(len(samples), 1)}%)")
    print(f"  Hallucinated: {n_hall} ({n_hall * 100 // max(len(samples), 1)}%)")

    split_counts = {}
    for s in samples:
        split_counts[s["split"]] = split_counts.get(s["split"], 0) + 1
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")

    return samples, metadata


if __name__ == "__main__":
    print("Run via pipeline.py")
