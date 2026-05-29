"""Phase 7: Assemble final HallucinationSample format."""

import ast
import json
import re
import warnings

from .config import (
    DATASET_PATH,
    MAX_ANSWER_CHARS,
    MAX_PROMPT_CHARS,
    METADATA_PATH,
    SOURCE_CACHE_DIR,
)


def _extract_signature(body: str) -> str:
    """Return the function signature lines + '...' (no implementation body)."""
    lines = body.splitlines()
    sig_lines = []
    for line in lines:
        sig_lines.append(line)
        if line.rstrip().endswith(":"):
            break
    return "\n".join(sig_lines) + "\n    ..."


def _is_called(name: str, body: str) -> bool:
    """Return True if *name* appears as a callable in *body*."""
    return bool(re.search(rf"\b{re.escape(name)}\s*\(", body))


def _condense_source_for_function(source: str, func_name: str, original_body: str | None) -> str:
    """Return a condensed source block for a complete_function sub-instance.

    Keeps only: module preamble (imports + top-level constants) + class header
    (if the function is a method) + the original function body (pre-patch).
    This avoids bloating the prompt with unrelated methods from the same file.
    """
    lines = source.splitlines()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except SyntaxError:
        return "\n".join(lines[:80])

    # Find the end of the module preamble: everything before the first top-level
    # class or function definition (imports, constants, __all__, etc.)
    preamble_end = len(lines)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            preamble_end = node.lineno - 1
            break
    preamble = "\n".join(lines[:preamble_end]).strip()

    # Find the class that contains func_name (if any)
    class_header = None
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == func_name:
                class_header = lines[node.lineno - 1]
                break
        if class_header:
            break

    parts = [p for p in [preamble, class_header, original_body] if p and p.strip()]
    return "\n\n".join(parts)


def build_prompt(
    source_files: dict[str, str],
    documentation: dict[str, str],
    user_query: str,
    dependency_files: dict[str, dict] | None = None,
) -> str:
    """Build the prompt (context) for a sample.

    Format: source files + dependency definitions + documentation + user query.

    Args:
        source_files: Mapping of filepath -> source content for patch-touched files.
        documentation: Mapping of library name -> documentation string.
        user_query: The user's request / problem statement.
        dependency_files: Optional mapping of identifier name -> {"file": path,
            "source": definition_source} for answer-based dependencies, OR
            filepath -> definition_source for import-based dependencies.
            Both formats are handled.
    """
    # Build non-source sections first so they are never truncated.
    tail_parts = []

    if dependency_files:
        dep_parts = []
        for name, info in dependency_files.items():
            if isinstance(info, dict):
                # answer_dependency_files format: {name: {"file": ..., "source": ...}}
                filepath = info.get("file", "")
                source = info.get("source", "")
                dep_parts.append(f"# {name} (from {filepath})\n{source}")
            else:
                # import dependency_files format: {filepath: definition_source_str}
                dep_parts.append(f"# {name}\n{info}")
        if dep_parts:
            tail_parts.append(
                "Referenced definitions:\n```python\n" + "\n\n".join(dep_parts) + "\n```"
            )

    for lib, doc in documentation.items():
        tail_parts.append(f"Documentation for {lib}:\n{doc}")

    tail_parts.append(f"User request: {user_query}")

    tail = "\n\n".join(tail_parts)
    source_budget = MAX_PROMPT_CHARS - len(tail) - 2  # -2 for the joining "\n\n"

    # Build source file sections, truncating collectively to stay within budget.
    source_parts = []
    used = 0
    for filepath, content in source_files.items():
        block = f"File: {filepath}\n```python\n{content}\n```"
        if used + len(block) > source_budget:
            remaining = source_budget - used
            if remaining > 0:
                source_parts.append(block[:remaining])
            break
        source_parts.append(block)
        used += len(block) + 2  # +2 for "\n\n" separator

    parts = source_parts + [tail] if source_parts else [tail]
    return "\n\n".join(parts)


def assemble_samples(
    format_entries: list[dict],
    swebench_lookup: dict[str, dict],
    source_cache: dict[str, dict],
    queries: dict[str, str],
    docs: dict[str, dict],
    hallucinations: dict[str, dict],
    hallucination_instance_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    """Assemble all samples into HallucinationSample format.

    Iterates over format entries (which may be sub-instances).  For each entry:
    - ``original_id`` (or ``instance_id`` for non-sub-instances) is used to
      look up source_cache, swebench metadata, queries, and docs.
    - ``function_name`` (if present) triggers sibling-function context: all
      other functions from the same patch are added to Referenced definitions.

    Returns (samples, metadata).
    """
    samples = []
    metadata = []

    for fmt_entry in format_entries:
        instance_id = fmt_entry["instance_id"]
        original_id = fmt_entry.get("original_id", instance_id)
        function_name = fmt_entry.get("function_name")

        swe_inst = swebench_lookup.get(original_id)
        if not swe_inst:
            continue
        split = swe_inst["split"]
        repo = swe_inst["repo"]

        if original_id not in source_cache:
            continue

        source_data = source_cache[original_id]
        fmt_data = fmt_entry
        query = queries.get(original_id, swe_inst.get("problem_statement", "")[:500])
        doc = docs.get(original_id, {})

        # Build dependency definitions: import deps + sibling functions from same patch
        source_files = dict(source_data.get("source_files", {}))
        dependency_files: dict = {}
        import_deps = source_data.get("dependency_files", {})
        answer_deps = source_data.get("answer_dependency_files", {})
        if import_deps:
            dependency_files.update(import_deps)
        if answer_deps:
            dependency_files.update(answer_deps)

        # For complete_function / code_with_explanation: replace the function's
        # source file with a condensed version (preamble + class header + original body).
        # This keeps context tight and avoids drowning out relevant content.
        format_type = fmt_data.get("format_type", "")
        if function_name and format_type in ("complete_function", "code_with_explanation"):
            func_record = next(
                (f for f in source_data.get("modified_functions", []) if f["name"] == function_name),
                None,
            )
            if func_record:
                func_file = func_record.get("file")
                if func_file and func_file in source_files:
                    source_files[func_file] = _condense_source_for_function(
                        source_files[func_file],
                        function_name,
                        func_record.get("original"),
                    )

        # Add signatures of sibling functions that are called by the answer function
        if function_name:
            answer_body = fmt_data.get("answer", "")
            for f in source_data.get("modified_functions", []):
                if f["name"] != function_name and f.get("patched"):
                    if _is_called(f["name"], answer_body):
                        dependency_files[f["name"]] = _extract_signature(f["patched"])

        prompt = build_prompt(source_files, doc, query, dependency_files=dependency_files or None)

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
                "original_id": original_id,
                "repo": repo,
                "split": split,
                "is_lite": swe_inst.get("is_lite", False),
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
                "original_id": original_id,
                "repo": repo,
                "split": split,
                "is_lite": swe_inst.get("is_lite", False),
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

    # Build lookup from original SWE-bench instance_id -> instance metadata
    swebench_lookup = {inst["instance_id"]: inst for inst in instances}

    # Format entries (may include sub-instance IDs like "orig::func")
    format_entries = list(formats.values())

    # Load source cache keyed by original_id (not sub-instance ID)
    seen_original_ids = {
        entry.get("original_id", entry["instance_id"]) for entry in format_entries
    }
    source_cache = {}
    for original_id in seen_original_ids:
        cache_path = SOURCE_CACHE_DIR / f"{original_id}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                source_cache[original_id] = json.load(f)

    print(f"Source cache: {len(source_cache)} instances")
    print(f"Queries: {len(queries)}")
    print(f"Docs: {len(docs)}")
    print(f"Format entries: {len(format_entries)}")
    print(f"Hallucinations: {len(hallucinations)}")
    print(f"Hallucination targets: {len(hallucination_instance_ids)}")

    samples, metadata = assemble_samples(
        format_entries,
        swebench_lookup,
        source_cache,
        queries,
        docs,
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
