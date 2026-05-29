"""Phase 5: Assign answer format to each instance.

Supports both sequential (remote API) and async batch (local vLLM) modes.
Set BATCH_SIZE>1 env var for parallel requests to local vLLM.
"""

import asyncio
import json
import random
import textwrap
import time

from openai import AsyncOpenAI, OpenAI

from .config import (
    API_BASE_URL,
    API_KEY,
    BATCH_SIZE,
    FORMAT_TYPES,
    FORMAT_WEIGHTS,
    FORMATS_PATH,
    LLM_TEMPERATURE,
    MAX_RETRIES,
    MODEL,
    RETRY_DELAY,
    SOURCE_CACHE_DIR,
    token_limit_kwargs,
)

EXPLANATION_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a helpful AI coding assistant (like Claude or Cursor).
    Given a user's coding question and the correct code fix, write a natural response
    that a developer would receive from an AI assistant.

    Your response MUST:
    - Start with 1-2 sentences explaining what was wrong and how to fix it
    - Include the code in a properly formatted code block (```python)
    - Do NOT add anything after the code block

    Your response must NOT:
    - Include phrases like "Here's the fix" or "I'll help you with that"
    - Be longer than 2 sentences of explanation + the code block
    - Change the code in any way — use it exactly as provided
    - Add any imports or code not in the original

    Example:
    The `process_data` function uses `dict.items()` instead of iterating over sorted keys, causing non-deterministic output.

    ```python
    def process_data(data):
        for key in sorted(data.keys()):
            yield key, data[key]
    ```
""")


def _generate_explanation(
    client: OpenAI, model: str, code: str, query: str, context: str
) -> str | None:
    """Use LLM to wrap code in a natural explanation."""
    user_msg = f"""User's question: {query}

Context (relevant source code):
{context[:3000]}

Correct code fix:
```python
{code}
```

Write a natural AI assistant response that includes this exact code."""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=LLM_TEMPERATURE,
                **token_limit_kwargs(model, 200),
            )
            result = response.choices[0].message.content.strip()
            # Verify the code is actually in the response
            if code[:50] in result or "```" in result:
                return result
            if attempt < MAX_RETRIES - 1:
                continue
            return None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  Explanation error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return None
    return None


async def _generate_explanation_async(
    aclient: AsyncOpenAI, model: str, code: str, query: str, context: str
) -> str | None:
    """Async version of _generate_explanation for batch processing."""
    user_msg = f"""User's question: {query}

Context (relevant source code):
{context[:3000]}

Correct code fix:
```python
{code}
```

Write a natural AI assistant response that includes this exact code."""

    for attempt in range(MAX_RETRIES):
        try:
            response = await aclient.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=LLM_TEMPERATURE,
                **token_limit_kwargs(model, 200),
            )
            result = response.choices[0].message.content.strip()
            if code[:50] in result or "```" in result:
                return result
            if attempt < MAX_RETRIES - 1:
                continue
            return None
        except Exception:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return None
    return None


SUB_INSTANCE_SEP = "::"
MAX_FUNCTIONS_PER_INSTANCE = 5
MIN_FUNCTION_CHARS = 50


def _get_eligible_functions(source_data: dict) -> list[dict]:
    """Return functions eligible for sub-instances, sorted by priority.

    Priority: modified functions (have an original body) before new ones,
    then by descending patched length.  Capped at MAX_FUNCTIONS_PER_INSTANCE.
    """
    funcs = source_data.get("modified_functions", [])
    eligible = [f for f in funcs if len(f.get("patched", "")) >= MIN_FUNCTION_CHARS]
    eligible.sort(key=lambda f: (f["original"] is None, -len(f.get("patched", ""))))
    return eligible[:MAX_FUNCTIONS_PER_INSTANCE]


def assign_format_entries(source_data: dict, instance_id: str) -> list[dict]:
    """Return a list of format entries for an instance.

    For complete_function / code_with_explanation: one entry per eligible
    function (up to MAX_FUNCTIONS_PER_INSTANCE), using sub-instance IDs
    ``{instance_id}::{function_name}``.

    For fragment / edit_style: exactly one entry, keeping the original
    instance_id.

    code_with_explanation entries contain the raw function body as ``answer``
    — the caller is responsible for wrapping it with an LLM-generated
    explanation before writing to disk.
    """
    has_functions = bool(source_data.get("modified_functions"))
    has_edit = bool(source_data.get("edit_style"))
    has_fragment = bool(source_data.get("patch_code", "").strip())

    available = []
    if has_functions:
        available.append("complete_function")
    if has_edit:
        available.append("edit_style")
    if has_fragment:
        available.append("fragment")

    if not available:
        return []

    all_available = available + ["code_with_explanation"]

    weights = [FORMAT_WEIGHTS[FORMAT_TYPES.index(fmt)] for fmt in all_available]
    total = sum(weights)
    weights = [w / total for w in weights]
    chosen = random.choices(all_available, weights=weights, k=1)[0]

    if chosen in ("complete_function", "code_with_explanation"):
        eligible = _get_eligible_functions(source_data)
        if not eligible:
            # Fallback: no eligible functions — use fragment or edit_style
            if has_fragment:
                return [
                    {
                        "instance_id": instance_id,
                        "original_id": instance_id,
                        "format_type": "fragment",
                        "answer": source_data["patch_code"],
                    }
                ]
            elif has_edit:
                return [
                    {
                        "instance_id": instance_id,
                        "original_id": instance_id,
                        "format_type": "edit_style",
                        "answer": source_data["edit_style"],
                    }
                ]
            return []
        return [
            {
                "instance_id": f"{instance_id}{SUB_INSTANCE_SEP}{func['name']}",
                "original_id": instance_id,
                "format_type": chosen,
                "answer": func["patched"],
                "function_name": func["name"],
            }
            for func in eligible
        ]

    elif chosen == "edit_style":
        return [
            {
                "instance_id": instance_id,
                "original_id": instance_id,
                "format_type": "edit_style",
                "answer": source_data["edit_style"],
            }
        ]
    else:  # fragment
        return [
            {
                "instance_id": instance_id,
                "original_id": instance_id,
                "format_type": "fragment",
                "answer": source_data["patch_code"],
            }
        ]


def assign_format(source_data: dict) -> tuple[str, str]:
    """Legacy single-entry helper (used by run_test). Picks the longest function."""
    has_functions = bool(source_data.get("modified_functions"))
    has_edit = bool(source_data.get("edit_style"))
    has_fragment = bool(source_data.get("patch_code", "").strip())

    available = []
    if has_functions:
        available.append("complete_function")
    if has_edit:
        available.append("edit_style")
    if has_fragment:
        available.append("fragment")

    if not available:
        return None, None

    all_available = available + ["code_with_explanation"]
    weights = [FORMAT_WEIGHTS[FORMAT_TYPES.index(fmt)] for fmt in all_available]
    total = sum(weights)
    weights = [w / total for w in weights]
    chosen = random.choices(all_available, weights=weights, k=1)[0]

    if chosen == "code_with_explanation":
        if has_functions:
            funcs = source_data["modified_functions"]
            func = max(funcs, key=lambda f: len(f.get("patched", "")))
            answer = func["patched"]
        elif has_fragment:
            answer = source_data["patch_code"]
        else:
            answer = source_data["edit_style"]
        return "code_with_explanation", answer
    elif chosen == "complete_function":
        funcs = source_data["modified_functions"]
        func = max(funcs, key=lambda f: len(f.get("patched", "")))
        return chosen, func["patched"]
    elif chosen == "edit_style":
        return chosen, source_data["edit_style"]
    else:
        return chosen, source_data["patch_code"]


def run(
    instances: list[dict],
    source_cache_dir=SOURCE_CACHE_DIR,
    api_key: str = API_KEY,
    base_url: str = API_BASE_URL,
    model: str = MODEL,
    queries: dict[str, str] | None = None,
):
    """Run Phase 5: Assign formats and build answers.

    Uses async batch processing when BATCH_SIZE > 1 (for local vLLM).
    Falls back to sequential processing for remote APIs (BATCH_SIZE=1).
    """
    print("=" * 60)
    print("Phase 5: Answer Format Building")
    print("=" * 60)

    FORMATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if queries is None:
        queries = {}

    # Load existing for resumability — keyed by sub-instance ID
    existing = {}
    if FORMATS_PATH.exists():
        with open(FORMATS_PATH) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    existing[entry["instance_id"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"Already processed: {len(existing)} format entries")

    # Track which original instances are fully done (any entry present = skip whole instance)
    existing_originals = {
        entry.get("original_id", entry["instance_id"]) for entry in existing.values()
    }
    to_process = [inst for inst in instances if inst["instance_id"] not in existing_originals]
    print(f"Remaining: {len(to_process)} instances to process")
    print(f"Batch size: {BATCH_SIZE}")

    # First pass: assign formats for all instances (no LLM needed)
    # needs_explanation items: (entry_dict, query, context)
    needs_explanation = []
    entries_no_llm = []

    for inst in to_process:
        instance_id = inst["instance_id"]

        cache_path = source_cache_dir / f"{instance_id}.json"
        if not cache_path.exists():
            continue

        with open(cache_path) as fp:
            source_data = json.load(fp)

        fmt_entries = assign_format_entries(source_data, instance_id)
        if not fmt_entries:
            continue

        split = inst.get("split", "train")
        for entry in fmt_entries:
            entry["split"] = split
            if entry["format_type"] == "code_with_explanation":
                query = queries.get(instance_id, inst.get("problem_statement", "")[:500])
                context = source_data.get("patch_code", "")
                needs_explanation.append((entry, query, context))
            else:
                entries_no_llm.append(entry)

    # Write non-LLM entries immediately
    results = list(existing.values())
    format_counts = {fmt: 0 for fmt in FORMAT_TYPES}
    for entry in results:
        fmt = entry.get("format_type")
        if fmt in format_counts:
            format_counts[fmt] += 1

    processed = 0
    explanation_failures = 0

    with open(FORMATS_PATH, "a") as f:
        for entry in entries_no_llm:
            f.write(json.dumps(entry) + "\n")
            results.append(entry)
            format_counts[entry["format_type"]] += 1
            processed += 1
        f.flush()

    print(f"  Assigned {len(entries_no_llm)} non-LLM format entries")
    print(f"  Need LLM explanation: {len(needs_explanation)} sub-instances")

    # Second pass: generate explanations (batched or sequential)
    if needs_explanation:
        if BATCH_SIZE > 1:
            explanation_failures = _run_explanations_batched(
                needs_explanation, format_counts, results, api_key, base_url, model
            )
        else:
            explanation_failures = _run_explanations_sequential(
                needs_explanation, format_counts, results, api_key, base_url, model
            )

    processed += len(needs_explanation)

    print(f"\nAssigned formats for {len(results)} instances")
    if explanation_failures:
        print(f"  Explanation generation failures (fell back to fragment): {explanation_failures}")
    for fmt, count in format_counts.items():
        pct = count * 100 // max(len(results), 1)
        print(f"  {fmt}: {count} ({pct}%)")

    return results


def _run_explanations_sequential(
    needs_explanation, format_counts, results, api_key, base_url, model
):
    """Generate explanations sequentially (for remote APIs).

    needs_explanation items: (entry_dict, query, context)
    entry_dict contains instance_id, original_id, answer (base code), function_name.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    explanation_failures = 0
    processed = 0

    with open(FORMATS_PATH, "a") as f:
        for base_entry, query, context in needs_explanation:
            code = base_entry["answer"]
            explained = _generate_explanation(client, model, code, query, context)

            if explained is None:
                fmt = "fragment"
                answer = code
                explanation_failures += 1
            else:
                fmt = "code_with_explanation"
                answer = explained

            entry = {**base_entry, "format_type": fmt, "answer": answer}
            f.write(json.dumps(entry) + "\n")
            f.flush()
            results.append(entry)
            format_counts[fmt] += 1
            processed += 1

            if processed % 100 == 0:
                print(
                    f"  Phase 5 (explanations): {processed}/{len(needs_explanation)} "
                    f"(failures: {explanation_failures})"
                )

    return explanation_failures


def _run_explanations_batched(needs_explanation, format_counts, results, api_key, base_url, model):
    """Generate explanations with async batching (for local vLLM).

    needs_explanation items: (entry_dict, query, context)
    """
    aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)
    explanation_failures = 0
    processed = 0

    async def process_batches():
        nonlocal explanation_failures, processed

        with open(FORMATS_PATH, "a") as f:
            for batch_start in range(0, len(needs_explanation), BATCH_SIZE):
                batch = needs_explanation[batch_start : batch_start + BATCH_SIZE]

                tasks = [
                    _generate_explanation_async(
                        aclient, model, base_entry["answer"], query, context
                    )
                    for base_entry, query, context in batch
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for (base_entry, _, _), explained in zip(batch, batch_results):
                    if isinstance(explained, Exception) or explained is None:
                        fmt = "fragment"
                        answer = base_entry["answer"]
                        explanation_failures += 1
                    else:
                        fmt = "code_with_explanation"
                        answer = explained

                    entry = {**base_entry, "format_type": fmt, "answer": answer}
                    f.write(json.dumps(entry) + "\n")
                    results.append(entry)
                    format_counts[fmt] += 1
                    processed += 1

                f.flush()

                if processed % 100 == 0 or batch_start + BATCH_SIZE >= len(needs_explanation):
                    print(
                        f"  Phase 5 (explanations): {processed}/{len(needs_explanation)} "
                        f"(failures: {explanation_failures})"
                    )

    asyncio.run(process_batches())
    return explanation_failures


if __name__ == "__main__":
    from .swebench_loader import load_instances

    instances = load_instances()
    run(instances)
