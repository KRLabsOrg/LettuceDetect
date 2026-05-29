"""Phase 6: Inject hallucinations using LLM with JSON span annotations.

Supports both sequential (remote API) and async batch (local vLLM) modes.
Set BATCH_SIZE>1 env var for parallel requests to local vLLM.
"""

import asyncio
import json
import re
import textwrap
import time

from openai import AsyncOpenAI, OpenAI

from .config import (
    API_BASE_URL,
    API_KEY,
    BATCH_SIZE,
    HALLUCINATED_PATH,
    HALLUCINATION_TEMPERATURE,
    HALLUCINATION_TYPES,
    INJECTION_FAILURES_PATH,
    MAX_PROMPT_CHARS,
    MAX_RETRIES,
    MODEL,
    RETRY_DELAY,
    token_limit_kwargs,
)


def _original_id(iid: str) -> str:
    """Strip sub-instance suffix to get the original SWE-bench instance ID."""
    return iid.split("::")[0]


INJECTION_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a code hallucination injector for building a hallucination detection dataset.

    Given a correct answer (which may be pure code OR code with natural language explanation)
    and SOURCE CODE CONTEXT, return ONLY a small set of localized replacement edits that will
    turn the answer into a hallucinated answer.

    IMPORTANT: You are NOT allowed to rewrite the full answer.
    - Return replacement edits only.
    - The pipeline will apply those edits to the original answer.
    - Outside the returned edits, the answer must remain unchanged.

    IMPORTANT: Only inject hallucinations into CODE portions of the answer.
    - If the answer contains markdown code fences, edits must be inside the fenced code block(s).
    - Do NOT modify natural language explanations before or after the code block.
    - Do NOT add explanatory comments inside code.
    - The explanation text must remain correct and neutral; only the code should be wrong.

    CRITICAL RULES FOR GROUNDING:
    - Every error you inject MUST BE DETECTABLE by comparing the answer against
      the provided source code context AND/OR the user's request.
    - ONLY reference functions, methods, classes, variables, and parameters that
      appear in the PROVIDED source context. Do NOT use your own knowledge of the
      library — pretend you only know what's in the context.
    - A human reading ONLY the source files and user query must be able to spot
      that the hallucinated part is wrong. If they can't, the hallucination is useless.
    - Do NOT inject errors that require running code, external docs, or knowledge
      beyond what's in the provided context to detect.

    Hallucination types:
    - STRUCTURAL: Change a function/method/class name to something that does NOT
      appear anywhere in the provided source context.
    - BEHAVIORAL: Use correct names from the source but with wrong values or logic
      that visibly contradicts the source.
    - SEMANTIC: Make the CODE solve a different problem than the user asked for, or
      make the code behave differently than what the source context shows.

    Rules:
    - Make 1-3 DISTINCT replacement edits spread across different parts of the answer
    - Each edit MUST contradict something VISIBLE in the provided source code or user request
    - Do NOT reference functions/classes/methods not present in the provided context
    - Do NOT make any unlabeled edits outside the returned replacement edits
    - Each replacement span must be 12-120 characters long and as small as possible
    - Total hallucinated text must be LESS THAN 30% of the original answer length
    - Keep most of the answer CORRECT — do NOT rewrite the entire thing
    - Changes should be in different functions/blocks, not adjacent lines
    - Make changes PLAUSIBLE — something an LLM would realistically generate
    - Changes must be SUBTLE, not obviously broken
    - The edited code must still be syntactically valid
    - Do NOT add comments explaining or hinting at the hallucination
    - Do NOT add words like BUG, wrong, incorrect, deprecated, hallucination, fix, helper
    - Do NOT include editorial text that describes the mistake inside the answer itself
    - Preserve the overall structure: keep markdown formatting, code blocks, indentation, imports, and surrounding text unchanged
    - Do NOT add or remove markdown fences
    - Do NOT add explanation text, tutorial text, wrapper text, or placeholder text
    - Do NOT add imports, helper functions, or surrounding code
    - Prefer changing existing lines over insertions or deletions
    - Each edit must replace an existing substring of the original answer; no insert-only edits
    - Choose exact substrings that appear exactly once in the original answer whenever possible
    - Prefer whole expressions or full lines over tiny fragments

    Respond in this exact JSON format (no markdown, no code blocks):
    {
        "changes": [
            {
                "original": "exact original substring from the correct answer",
                "hallucinated": "replacement text for that substring",
                "target_zone": "code",
                "explanation": "why this replacement is wrong according to the source code or user request"
            }
        ]
    }

    Example 1:
    Original answer contains:
        return self.steps[-1][-1].transform(X)
    Good JSON change:
    {
      "changes": [
        {
          "original": "return self.steps[-1][-1].transform(X)",
          "hallucinated": "return self.steps[-1][-1].predict(X)",
          "target_zone": "code",
          "explanation": "The source context shows this method should transform the data, not run prediction."
        }
      ]
    }

    Example 2:
    Original answer contains:
        if handle_unknown == 'error':
    Good JSON change:
    {
      "changes": [
        {
          "original": "if handle_unknown == 'error':",
          "hallucinated": "if handle_unknown != 'error':",
          "target_zone": "code",
          "explanation": "This flips the branch condition and contradicts the intended error handling in the source."
        }
      ]
    }

    IMPORTANT:
    - You MUST include 1-3 changes in the "changes" array
    - The returned changes must be sufficient to construct the full hallucinated answer
    - "original" must be a non-empty exact substring of the correct answer
    - Before returning, verify that each "original" substring appears verbatim in the provided correct answer
    - Prefer substrings that appear exactly once in the correct answer
    - If a substring appears multiple times, pick a different, longer substring that uniquely identifies the target location
    - "hallucinated" is the exact replacement text for that substring
    - "target_zone" must always be "code"
    - Each "explanation" must reference what the source code or user request actually says
    - If you cannot find 1-3 exact editable substrings in the provided answer, return {"changes": []}
    - Return ONLY valid JSON, nothing else
""")

LEAKY_TERMS = (
    "bug",
    "wrong",
    "incorrect",
    "incorrectly",
    "deprecated",
    "hallucination",
    "helper method",
    "should be replaced",
)
PROMPT_RESIDUE = (
    "Generate a hallucinated version",
    "Return JSON only",
    "hallucinated_code",
    "target_zone",
    "left_context",
    "right_context",
)
MAX_LABEL_COVERAGE = 0.40
MAX_LABEL_SPAN_CHARS = 500
MIN_LABEL_SPAN_CHARS = 12


def build_source_context(source_data: dict) -> str:
    """Build source code context string from cached source data.

    Truncates to MAX_PROMPT_CHARS so the final sample fits in 8K model context.
    """
    parts = []
    for filepath, content in source_data.get("source_files", {}).items():
        parts.append(f"File: {filepath}\n```python\n{content}\n```")
    context = "\n\n".join(parts)
    if len(context) > MAX_PROMPT_CHARS:
        context = context[:MAX_PROMPT_CHARS]
    return context


def inject_hallucination(
    client: OpenAI,
    model: str,
    clean_answer: str,
    hall_type: str,
    user_query: str = "",
    context: str = "",
    documentation: dict[str, str] | None = None,
) -> dict | None:
    """Request structured replacement edits for hallucination injection."""
    docs_section = ""
    if documentation:
        docs_parts = [f"Documentation for {lib}:\n{doc}" for lib, doc in documentation.items()]
        docs_section = (
            "\n\nLibrary documentation (the hallucination could contradict this):\n"
            + "\n\n".join(docs_parts)
        )

    user_msg = f"""Hallucination type to inject: {hall_type.upper()}

User's original request: {user_query}

Context (source code):
{context}{docs_section}

Correct answer to modify:
{clean_answer}

Return ONLY replacement edits for {hall_type} error(s). Do not return the full rewritten answer."""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": INJECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=HALLUCINATION_TEMPERATURE,
                **token_limit_kwargs(model),
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON from response
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if not json_match:
                if attempt < MAX_RETRIES - 1:
                    continue
                return None

            result = json.loads(json_match.group())

            if "changes" not in result or not isinstance(result["changes"], list):
                if attempt < MAX_RETRIES - 1:
                    continue
                return None
            if not result["changes"]:
                if attempt < MAX_RETRIES - 1:
                    continue
                return None

            return result

        except (json.JSONDecodeError, Exception) as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  Injection error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return None


def _find_all_occurrences(text: str, pattern: str) -> list[dict]:
    """Return all exact matches of pattern in text."""
    if not pattern:
        return []
    offsets = []
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx == -1:
            break
        offsets.append({"start": idx, "end": idx + len(pattern)})
        start = idx + 1
    return offsets


def _extract_code_regions(answer: str) -> list[tuple[int, int]]:
    """Return ranges that correspond to markdown fenced code blocks.

    If no fenced blocks are present, treat the whole answer as code.
    """
    regions = []
    idx = 0
    while True:
        start = answer.find("```", idx)
        if start == -1:
            break
        code_start = answer.find("\n", start + 3)
        if code_start == -1:
            break
        code_start += 1
        end = answer.find("```", code_start)
        if end == -1:
            break
        regions.append((code_start, end))
        idx = end + 3
    if not regions:
        return [(0, len(answer))]
    return regions


def _span_is_in_code(answer: str, start: int, end: int) -> bool:
    """Check whether a span lies fully inside a code region."""
    for code_start, code_end in _extract_code_regions(answer):
        if start >= code_start and end <= code_end:
            return True
    return False


def _contains_leakage(text: str) -> bool:
    """Detect obvious synthetic giveaway text inside a label span."""
    lowered = text.lower()
    return any(term in lowered for term in LEAKY_TERMS)


def _max_allowed_coverage(answer_len: int) -> float:
    """Use a looser coverage cap for short answers and fragments."""
    if answer_len <= 400:
        return 0.40
    if answer_len <= 800:
        return 0.35
    return MAX_LABEL_COVERAGE


def _locate_original_change(original_answer: str, change: dict) -> dict | None:
    """Locate a replacement span in the original answer by exact unique match."""
    original_span = change.get("original", "")
    hallucinated_span = change.get("hallucinated", "")
    if not original_span or not hallucinated_span:
        return None
    if change.get("target_zone") not in (None, "code"):
        return None

    offsets = _find_all_occurrences(original_answer, original_span)
    if len(offsets) != 1:
        return None

    return {
        "start": offsets[0]["start"],
        "end": offsets[0]["end"],
        "original": original_span,
        "hallucinated": hallucinated_span,
    }


def _sort_changes_by_original_position(
    original_answer: str, changes: list[dict]
) -> list[dict] | None:
    """Return changes ordered by their matched position in the original answer."""
    located = []
    for change in changes:
        loc = _locate_original_change(original_answer, change)
        if loc is None:
            return None
        located.append((loc["start"], loc["end"], change))
    located.sort(key=lambda item: (item[0], item[1]))
    return [change for _, _, change in located]


def apply_changes_to_answer(
    original_answer: str, changes: list[dict], hall_type: str
) -> tuple[str, list[dict]] | tuple[None, None]:
    """Apply structured replacement edits to the original answer and build labels.

    The model returns edits only. This function deterministically constructs the
    hallucinated answer and the corresponding label offsets.
    """
    located = []
    for change in changes:
        if len(change.get("hallucinated", "")) < MIN_LABEL_SPAN_CHARS:
            continue
        located_change = _locate_original_change(original_answer, change)
        if located_change is None:
            continue
        located.append(located_change)

    if not located:
        return None, None

    # Skip overlapping edits, keeping the earlier one.
    located.sort(key=lambda item: (item["start"], item["end"]))
    previous_end = -1
    filtered = []
    for item in located:
        if item["start"] >= previous_end:
            filtered.append(item)
            previous_end = item["end"]
    located = filtered

    if not located:
        return None, None

    hallucinated_parts = []
    labels = []
    cursor = 0
    for item in located:
        start = item["start"]
        end = item["end"]
        hallucinated_span = item["hallucinated"]

        hallucinated_parts.append(original_answer[cursor:start])
        label_start = sum(len(part) for part in hallucinated_parts)
        hallucinated_parts.append(hallucinated_span)
        label_end = label_start + len(hallucinated_span)
        labels.append({"start": label_start, "end": label_end, "label": hall_type})
        cursor = end

    hallucinated_parts.append(original_answer[cursor:])
    hallucinated_answer = "".join(hallucinated_parts)
    return hallucinated_answer, labels


def replay_failures(
    formats: dict[str, dict],
    failures_path=INJECTION_FAILURES_PATH,
    model: str = MODEL,
) -> int:
    """Re-run post-processing on saved failures without calling the LLM again.

    Useful after fixing _process_result logic. Appends new successes to
    HALLUCINATED_PATH and removes them from INJECTION_FAILURES_PATH.
    Returns number of newly recovered samples.
    """
    if not failures_path.exists():
        return 0

    failures = []
    with open(failures_path) as f:
        for line in f:
            try:
                failures.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    existing = load_existing_hallucinations()
    recovered = 0
    remaining = []

    with open(HALLUCINATED_PATH, "a") as out:
        for failure in failures:
            instance_id = failure["instance_id"]
            if instance_id in existing:
                continue
            fmt_data = formats.get(instance_id, {})
            if not fmt_data:
                remaining.append(failure)
                continue
            entry, reason = _process_result(
                failure["llm_result"], instance_id, failure["hall_type"], fmt_data, model
            )
            if entry is not None:
                out.write(json.dumps(entry) + "\n")
                existing[instance_id] = entry
                recovered += 1
            else:
                failure["failure_reason"] = reason
                remaining.append(failure)

    with open(failures_path, "w") as f:
        for r in remaining:
            f.write(json.dumps(r) + "\n")

    print(f"Replay: {recovered} recovered, {len(remaining)} still failing")
    return recovered


def load_existing_hallucinations(path=HALLUCINATED_PATH) -> dict[str, dict]:
    """Load already-processed hallucinations for resumability."""
    existing = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    existing[entry["instance_id"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
    return existing


async def _inject_one_async(
    aclient: AsyncOpenAI,
    model: str,
    clean_answer: str,
    hall_type: str,
    user_query: str,
    context: str,
    documentation: dict[str, str] | None = None,
) -> dict | None:
    """Async version of inject_hallucination for batch processing."""
    docs_section = ""
    if documentation:
        docs_parts = [f"Documentation for {lib}:\n{doc}" for lib, doc in documentation.items()]
        docs_section = (
            "\n\nLibrary documentation (the hallucination could contradict this):\n"
            + "\n\n".join(docs_parts)
        )

    user_msg = f"""Hallucination type to inject: {hall_type.upper()}

User's original request: {user_query}

Context (source code):
{context}{docs_section}

Correct answer to modify:
{clean_answer}

Return ONLY replacement edits for {hall_type} error(s). Do not return the full rewritten answer."""

    for attempt in range(MAX_RETRIES):
        try:
            response = await aclient.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": INJECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=HALLUCINATION_TEMPERATURE,
                **token_limit_kwargs(model),
            )
            raw = response.choices[0].message.content.strip()
            json_match = re.search(r"\{[\s\S]*\}", raw)
            if not json_match:
                continue
            result = json.loads(json_match.group())
            if "changes" not in result or not isinstance(result["changes"], list):
                continue
            if not result["changes"]:
                continue
            return result
        except Exception:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return None
    return None


def _validate_labels(
    original_answer: str, hallucinated_code: str, labels: list[dict], format_type: str
) -> tuple[bool, str]:
    """Validate that hallucination labels meet quality thresholds.

    :return: (is_valid, reason) tuple.
    """
    if not labels:
        return False, "no_labels"

    # Reject prompt contamination (LLM leaked its instructions into the answer)
    for residue in PROMPT_RESIDUE:
        if residue in hallucinated_code:
            return False, f"prompt_residue ({residue[:30]})"

    # Reject unbalanced code fences for code_with_explanation
    if format_type == "code_with_explanation":
        fence_count = hallucinated_code.count("```")
        if fence_count % 2 != 0:
            return False, f"unbalanced_fences ({fence_count})"
        if fence_count == 0:
            return False, "no_code_fences"

    total_span = sum(lab["end"] - lab["start"] for lab in labels)
    code_len = len(hallucinated_code) if hallucinated_code else 1
    coverage = total_span / code_len

    max_coverage = _max_allowed_coverage(code_len)
    if coverage > max_coverage:
        return False, f"coverage_too_high ({coverage:.0%} > {max_coverage:.0%})"

    previous_end = -1
    for lab in labels:
        span_len = lab["end"] - lab["start"]
        if span_len < MIN_LABEL_SPAN_CHARS:
            return False, f"span_too_short ({span_len} chars)"
        if span_len > MAX_LABEL_SPAN_CHARS:
            return False, f"span_too_long ({span_len} chars)"
        if lab["start"] < previous_end:
            return False, "overlapping_or_unsorted_labels"
        previous_end = lab["end"]

        span_text = hallucinated_code[lab["start"] : lab["end"]]
        if _contains_leakage(span_text):
            return False, "leaky_label_text"

        if format_type == "code_with_explanation" and not _span_is_in_code(
            hallucinated_code, lab["start"], lab["end"]
        ):
            return False, "label_outside_code_block"

    return True, ""


def _process_result(result, instance_id, hall_type, fmt_data, model):
    """Process a single injection result into a JSONL entry.

    Returns (entry, failure_reason) where entry is None on failure.
    """
    if result is None:
        return None, "llm_no_response"
    original_answer = fmt_data.get("answer", "")
    changes = result.get("changes", [])
    if not changes:
        return None, "llm_empty_changes"
    hallucinated_code, labels = apply_changes_to_answer(original_answer, changes, hall_type)
    if hallucinated_code is None or labels is None:
        return None, "substring_not_unique_or_too_short"
    ordered_changes = _sort_changes_by_original_position(original_answer, changes)
    if ordered_changes is None:
        return None, "substring_not_unique_or_too_short"
    format_type = fmt_data.get("format_type", "fragment")

    # Repair unbalanced fences before validation (appending doesn't shift label offsets).
    if format_type == "code_with_explanation" and hallucinated_code.count("```") % 2 != 0:
        hallucinated_code = hallucinated_code.rstrip() + "\n```"

    valid, reason = _validate_labels(original_answer, hallucinated_code, labels, format_type)
    if not valid:
        return None, f"validation:{reason}"

    return {
        "instance_id": instance_id,
        "hallucinated_answer": hallucinated_code,
        "labels": labels,
        "hallucination_type": hall_type,
        "injector_model": model,
        "format_type": format_type,
        "changes": ordered_changes,
    }, None


def run(
    instances_to_inject: list[dict],
    formats: dict[str, dict],
    queries: dict[str, str],
    docs: dict[str, dict] | None = None,
    source_cache: dict[str, dict] | None = None,
    api_key: str = API_KEY,
    base_url: str = API_BASE_URL,
    model: str = MODEL,
):
    """Run Phase 6: Inject hallucinations into selected instances.

    Uses async batch processing when BATCH_SIZE > 1 (for local vLLM).
    Falls back to sequential processing for remote APIs (BATCH_SIZE=1).
    """
    print("=" * 60)
    print("Phase 6: Hallucination Injection")
    print("=" * 60)

    if docs is None:
        docs = {}
    if source_cache is None:
        source_cache = {}

    HALLUCINATED_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using {base_url} with model {model}")
    print(f"Batch size: {BATCH_SIZE}")

    existing = load_existing_hallucinations()
    print(f"Already processed: {len(existing)}")

    to_process = [
        inst
        for inst in instances_to_inject
        if inst["instance_id"] not in existing and inst["instance_id"] in formats
    ]
    print(f"Remaining: {len(to_process)} instances to inject")

    if BATCH_SIZE > 1:
        results = _run_batched(
            to_process, formats, queries, docs, source_cache, api_key, base_url, model
        )
    else:
        results = _run_sequential(
            to_process, formats, queries, docs, source_cache, api_key, base_url, model
        )

    # Stats
    type_counts = {}
    for r in results:
        t = r["hallucination_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print("By type:", type_counts)

    if results:
        avg_spans = sum(len(r["labels"]) for r in results) / len(results)
        span_sizes = [lab["end"] - lab["start"] for r in results for lab in r["labels"]]
        print(f"Avg spans per sample: {avg_spans:.1f}")
        print(
            f"Span sizes: min={min(span_sizes)}, max={max(span_sizes)}, avg={sum(span_sizes) // len(span_sizes)}"
        )

    return results


def _run_sequential(to_process, formats, queries, docs, source_cache, api_key, base_url, model):
    """Sequential processing for remote APIs (rate-limited)."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    processed = 0
    failed = 0
    no_spans = 0
    results = []

    failure_reasons: dict[str, int] = {}

    with open(HALLUCINATED_PATH, "a") as f, open(INJECTION_FAILURES_PATH, "a") as ff:
        for i, inst in enumerate(to_process):
            instance_id = inst["instance_id"]
            fmt_data = formats.get(instance_id, {})
            clean_answer = fmt_data.get("answer", "")
            if not clean_answer:
                failure_reasons["no_answer"] = failure_reasons.get("no_answer", 0) + 1
                failed += 1
                continue

            hall_type = HALLUCINATION_TYPES[i % len(HALLUCINATION_TYPES)]
            orig = _original_id(instance_id)
            query = queries.get(orig, "")
            source_data = source_cache.get(orig, {})
            context = (
                build_source_context(source_data)
                if source_data
                else inst.get("problem_statement", "")
            )
            instance_docs = docs.get(orig, {})

            entry = None
            last_reason = None
            last_result = None
            for attempt in range(3):
                result = inject_hallucination(
                    client,
                    model,
                    clean_answer,
                    hall_type,
                    query,
                    context,
                    documentation=instance_docs,
                )
                entry, last_reason = _process_result(
                    result, instance_id, hall_type, fmt_data, model
                )
                if entry is not None:
                    break
                if result is not None:
                    last_result = result

            if entry is None:
                failure_reasons[last_reason] = failure_reasons.get(last_reason, 0) + 1
                failed += 1
                if last_result is not None:
                    ff.write(
                        json.dumps(
                            {
                                "instance_id": instance_id,
                                "hall_type": hall_type,
                                "format_type": fmt_data.get("format_type"),
                                "failure_reason": last_reason,
                                "llm_result": last_result,
                            }
                        )
                        + "\n"
                    )
                continue

            f.write(json.dumps(entry) + "\n")
            f.flush()
            results.append(entry)
            processed += 1

            if processed % 100 == 0:
                print(f"  Phase 6: {processed}/{len(to_process)} (failed: {failed})")

    print(f"\nDone: {processed} injected, {failed} failed")
    if failure_reasons:
        print("Failure breakdown:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    return results


def _run_batched(to_process, formats, queries, docs, source_cache, api_key, base_url, model):
    """Async batch processing for local vLLM (no rate limiting needed)."""
    aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)
    processed = 0
    failed = 0
    results = []
    failure_reasons: dict[str, int] = {}

    async def process_batches():
        nonlocal processed, failed

        with open(HALLUCINATED_PATH, "a") as f, open(INJECTION_FAILURES_PATH, "a") as ff:
            for batch_start in range(0, len(to_process), BATCH_SIZE):
                batch = to_process[batch_start : batch_start + BATCH_SIZE]

                # Build async tasks for the batch
                tasks = []
                batch_meta = []
                for i, inst in enumerate(batch):
                    global_idx = batch_start + i
                    instance_id = inst["instance_id"]
                    fmt_data = formats.get(instance_id, {})
                    clean_answer = fmt_data.get("answer", "")
                    if not clean_answer:
                        failed += 1
                        continue

                    hall_type = HALLUCINATION_TYPES[global_idx % len(HALLUCINATION_TYPES)]
                    orig = _original_id(instance_id)
                    query = queries.get(orig, "")
                    source_data = source_cache.get(orig, {})
                    context = (
                        build_source_context(source_data)
                        if source_data
                        else inst.get("problem_statement", "")
                    )
                    instance_docs = docs.get(orig, {})

                    tasks.append(
                        _inject_one_async(
                            aclient,
                            model,
                            clean_answer,
                            hall_type,
                            query,
                            context,
                            documentation=instance_docs,
                        )
                    )
                    batch_meta.append((instance_id, hall_type, fmt_data))

                if not tasks:
                    continue

                # Run batch concurrently
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and write immediately
                for result, (instance_id, hall_type, fmt_data) in zip(batch_results, batch_meta):
                    if isinstance(result, Exception):
                        failure_reasons["llm_exception"] = (
                            failure_reasons.get("llm_exception", 0) + 1
                        )
                        failed += 1
                        continue

                    entry, reason = _process_result(result, instance_id, hall_type, fmt_data, model)
                    if entry is None:
                        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                        failed += 1
                        if result is not None:
                            ff.write(
                                json.dumps(
                                    {
                                        "instance_id": instance_id,
                                        "hall_type": hall_type,
                                        "format_type": fmt_data.get("format_type"),
                                        "failure_reason": reason,
                                        "llm_result": result,
                                    }
                                )
                                + "\n"
                            )
                        continue

                    f.write(json.dumps(entry) + "\n")
                    f.flush()
                    results.append(entry)
                    processed += 1

                if processed % 100 == 0 or batch_start + BATCH_SIZE >= len(to_process):
                    total = processed + failed
                    print(f"  Phase 6: {total}/{len(to_process)} ({processed} ok, {failed} failed)")

    asyncio.run(process_batches())
    print(f"\nDone: {processed} injected, {failed} failed")
    if failure_reasons:
        print("Failure breakdown:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    return results


if __name__ == "__main__":
    print("Run via pipeline.py")
