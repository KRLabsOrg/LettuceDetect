from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import random
import re
import textwrap
import time
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from lettucedetect.models.generation import HallucinationGenerator

from datasets import IterableDataset, load_dataset
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

PAPER_INJECTION_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a factual hallucination injector for building a hallucination detection
    dataset on academic papers (ACL Anthology) in Markdown format.

    Given a correct passage from an academic paper in Markdown, return ONLY a small
    set of localized replacement edits that will turn the passage into a hallucinated
    version. The PAPER ITSELF is the source of truth — the rest of the paper provides
    the grounding context against which a reader can detect the hallucination.

    IMPORTANT: You are NOT allowed to rewrite the full passage.
    - Return replacement edits only.
    - The pipeline will apply those edits to the original passage.
    - Outside the returned edits, the passage must remain unchanged.

    IMPORTANT: Preserve Markdown structure and syntax EXACTLY.
    - Do NOT add, remove, or modify Markdown syntax characters:
      headers (#, ##, ###), bold/italic markers (**, *, __, _), inline code (`),
      code fences (```), block quotes (>), list markers (-, *, 1.),
      horizontal rules (---), or escape characters.
    - Do NOT change link or image syntax: [text](url), ![alt](url) — the brackets,
      parentheses, exclamation marks, and URLs must remain byte-identical.
    - Do NOT change citation markers or reference brackets such as [1], [12, 34],
      (Author, 2020), (Author et al., 2020). Their textual form must be preserved
      verbatim, including punctuation and whitespace inside the brackets.
    - Do NOT alter table structure: the number of rows, columns, pipe characters,
      or alignment separators (|---|, |:---:|) must be unchanged. You may change
      the text inside a single cell, but the cell count must remain identical.
    - Do NOT modify math delimiters ($...$, $$...$$, \\( \\), \\[ \\]) or break the
      LaTeX inside them. You may change a numeric literal or variable name inside
      math as long as the expression still parses.
    - Do NOT modify footnote markers, HTML tags, or anchor IDs if present.
    - The output of applying your edits must be valid, well-formed Markdown that
      renders with identical structure to the input.

    IMPORTANT: Edits must occur inside PROSE, TABLE CELL TEXT, MATH CONTENT, LIST
    ITEM TEXT, or HEADING TEXT — never inside Markdown syntax tokens or structural
    delimiters.

    CRITICAL RULES FOR GROUNDING:
    - Every error you inject MUST BE DETECTABLE by reading the rest of the paper
      (and the passage itself). A reader with access to the full paper must be able
      to spot the contradiction.
    - ONLY reference entities, numbers, methods, datasets, authors, and terms that
      appear in the PROVIDED paper context. Do NOT invent names the paper never
      mentions — use real entities from the paper placed in the wrong role.
    - Hallucinations should contradict facts ESTABLISHED ELSEWHERE in the paper
      (the abstract, a results table, a figure caption, a previous section), or
      be objectively wrong (e.g., broken arithmetic, percentages outside 0-100,
      totals that no longer sum, mismatched row/column counts cited in prose).
    - Do NOT inject errors that require external knowledge, running experiments,
      or sources beyond the provided paper to detect.

    Hallucination types:
    - NUMERICAL: Change a reported number (accuracy, F1, BLEU, perplexity, dataset
      size, model parameter count, number of layers, year, percentage, ablation
      delta) to a value that contradicts what the paper states elsewhere or makes
      arithmetic inconsistent.
    - ENTITY: Swap a named entity (model, dataset, benchmark, method, language,
      institution, author) for a different one that the paper does NOT use in this
      role, OR for one mentioned elsewhere in the paper but in the wrong context.
    - RELATIONAL: Flip a comparison or relationship (outperforms <-> underperforms,
      larger <-> smaller, increases <-> decreases, significant <-> not significant,
      higher <-> lower, positive <-> negative correlation) so the claim contradicts
      results reported elsewhere.
    - METHODOLOGICAL: Change a procedural detail (e.g., "fine-tuned" -> "pre-trained",
      "supervised" -> "unsupervised", "encoder" -> "decoder", "frozen" -> "trainable",
      "greedy decoding" -> "beam search") where the change contradicts the method
      described in the paper.
    - CITATIONAL: Misattribute a claim to a different work mentioned in the paper,
      or change a cited year/author within the citation text (e.g., "Devlin et al.
      (2019)" -> "Devlin et al. (2017)") when the paper makes the original
      attribution clear. Do not invent citations the paper does not contain.

    Rules:
    - Make 1-3 DISTINCT replacement edits spread across different parts of the passage.
    - Each edit MUST contradict something VISIBLE in the provided paper context.
    - Do NOT reference entities, datasets, or methods not present in the paper.
    - Do NOT make any unlabeled edits outside the returned replacement edits.
    - Each replacement span must be 8-120 characters long and as small as possible.
    - Total hallucinated text must be LESS THAN 30% of the original passage length.
    - Keep most of the passage CORRECT — do NOT rewrite the entire thing.
    - Changes should be in different sentences/paragraphs, not adjacent words.
    - Make changes PLAUSIBLE — something an LLM would realistically generate.
    - Changes must be SUBTLE, not obviously broken.
    - The edited Markdown must still render correctly with identical structure.
    - Do NOT add comments, footnotes, or notes hinting at the hallucination.
    - Do NOT add words like ERROR, wrong, incorrect, hallucination, fix, note, [sic].
    - Do NOT include editorial text describing the mistake inside the passage.
    - Preserve all surrounding text, formatting, whitespace, and casing outside
      the changed substring.
    - Prefer changing existing words/phrases over insertions or deletions.
    - Each edit must replace an existing substring of the original passage; no
      insert-only edits.
    - Choose exact substrings that appear exactly once in the passage whenever
      possible; otherwise pick a longer span that uniquely identifies the location.
    - Prefer whole phrases or full claims over tiny fragments.
    - The hallucinated text must NOT introduce or remove Markdown syntax tokens.
      If a substring contains *, _, `, |, $, [, ], (, ), or # characters, the
      replacement must contain the same syntax characters in the same positions.

    Respond in this exact JSON format (no markdown, no code blocks):
    {
        "changes": [
            {
                "original": "exact original substring from the correct passage",
                "hallucinated": "replacement text for that substring",
                "target_zone": "one of: prose, table_cell, math, list_item, heading_text",
                "hallucination_type": "one of: NUMERICAL, ENTITY, RELATIONAL, METHODOLOGICAL, CITATIONAL",
                "explanation": "why this replacement contradicts the paper, citing what the paper actually states elsewhere"
            }
        ]
    }

    Example 1 (NUMERICAL, prose):
    Original passage contains:
        Our model achieves an F1 of 87.3 on the SQuAD dev set, outperforming the
        baseline by 2.1 points.
    Elsewhere the paper's results table reports F1 = 87.3 for this model.
    Good JSON change:
    {
      "changes": [
        {
          "original": "F1 of 87.3 on the SQuAD dev set",
          "hallucinated": "F1 of 83.7 on the SQuAD dev set",
          "target_zone": "prose",
          "hallucination_type": "NUMERICAL",
          "explanation": "Table 2 in Section 4 reports F1 = 87.3 for this model, so 83.7 contradicts the paper's own results."
        }
      ]
    }

    Example 2 (RELATIONAL, prose):
    Original passage contains:
        BERT-large outperforms BERT-base by a wide margin on all GLUE tasks.
    Good JSON change:
    {
      "changes": [
        {
          "original": "BERT-large outperforms BERT-base",
          "hallucinated": "BERT-large underperforms BERT-base",
          "target_zone": "prose",
          "hallucination_type": "RELATIONAL",
          "explanation": "Table 2 in the paper shows BERT-large scoring higher than BERT-base on every GLUE task, so 'underperforms' contradicts those results."
        }
      ]
    }

    Example 3 (ENTITY, prose):
    Original passage contains:
        We evaluate on the CoNLL-2003 NER benchmark.
    The paper only ever uses CoNLL-2003; OntoNotes is mentioned in related work
    but is not used for evaluation.
    Good JSON change:
    {
      "changes": [
        {
          "original": "the CoNLL-2003 NER benchmark",
          "hallucinated": "the OntoNotes 5.0 NER benchmark",
          "target_zone": "prose",
          "hallucination_type": "ENTITY",
          "explanation": "The Experiments section uses CoNLL-2003 throughout; OntoNotes 5.0 appears only as related work and is never the evaluation benchmark."
        }
      ]
    }

    Example 4 (NUMERICAL, table_cell — preserving pipe structure):
    Original table row:
        | BERT-base | 84.6 | 88.5 | 90.5 |
    Good JSON change:
    {
      "changes": [
        {
          "original": "| BERT-base | 84.6 | 88.5 | 90.5 |",
          "hallucinated": "| BERT-base | 84.6 | 78.5 | 90.5 |",
          "target_zone": "table_cell",
          "hallucination_type": "NUMERICAL",
          "explanation": "The text on the next page reports BERT-base scoring 88.5 on this metric, so 78.5 contradicts the paper's prose summary. Pipe count is preserved."
        }
      ]
    }

    Example 5 (CITATIONAL, prose):
    Original passage contains:
        following the approach of Vaswani et al. (2017)
    Good JSON change:
    {
      "changes": [
        {
          "original": "Vaswani et al. (2017)",
          "hallucinated": "Vaswani et al. (2015)",
          "target_zone": "prose",
          "hallucination_type": "CITATIONAL",
          "explanation": "The references section lists Vaswani et al. as a 2017 paper; 2015 contradicts the paper's own bibliography."
        }
      ]
    }

    IMPORTANT:
    - You MUST include 1-3 changes in the "changes" array.
    - "original" must be a non-empty exact substring of the correct passage.
    - Before returning, verify each "original" substring appears verbatim in the
      passage (including whitespace, punctuation, and case).
    - Prefer substrings that appear exactly once in the passage.
    - If a substring appears multiple times, pick a longer span that uniquely
      identifies the target location.
    - "hallucinated" is the exact replacement text for that substring.
    - The hallucinated text must contain the same Markdown syntax characters in
      the same positions as the original — only the textual content changes.
    - Each "explanation" must reference what the paper actually says elsewhere
      (which section, table, figure, or earlier statement is contradicted).
    - If you cannot find 1-3 valid editable substrings that satisfy all rules,
      return {"changes": []}.
    - Return ONLY valid JSON, nothing else.
""")

HALLUCINATION_TYPES = ["NUMERICAL", "ENTITY", "RELATIONAL", "METHODOLOGICAL", "CITATIONAL"]
MAX_LABEL_COVERAGE = 0.30
MAX_LABEL_SPAN_CHARS = 500
MIN_LABEL_SPAN_CHARS = 12


def rephrase_verbatim_answer(
    client: OpenAI,
    model: str,
    answer: str,
) -> str:
    """Rephrase a verbatim answer to a different wording with the same meaning."""
    prompt = textwrap.dedent(f"""Rephrase the following answer in different words while preserving its
    exact meaning. Do not add, modify or remove any (factually accurate) information, just change the wording and sentence
    structure. The answer's style should be quite similar like a typical LLM-generated answer.
    \n\n Context:{answer}
    Return only the rephrased answer, without any additional commentary or formatting.""")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )
    return response.choices[0].message.content.strip()


async def _rephrase_one_async(
    aclient: AsyncOpenAI,
    model: str,
    text: str,
    temperature: float,
) -> str | None:
    prompt = textwrap.dedent(f"""Rephrase the following answer in different words while preserving its
    exact meaning. Do not add, modify or remove any (factually accurate) information, just change the wording and sentence
    structure. The answer's style should be quite similar like a typical LLM-generated answer.
    \n\n Context:{text}
    Return only the rephrased answer, without any additional commentary or formatting.""")
    try:
        response = await aclient.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


async def _inject_one_async(
    aclient: AsyncOpenAI,
    model: str,
    clean_answer: str,
    hall_type: str,
    user_query: str,
    context: str,
    max_retries: int,
    retry_delay: int,
    temperature: float,
) -> dict | None:
    user_msg = f"""Hallucination type to inject: {hall_type.upper()}

User's original request: {user_query}

Context (paper contents):
{context}

Correct answer to modify:
{clean_answer}

Return ONLY replacement edits for {hall_type} error(s). Do not return the full rewritten answer."""

    for attempt in range(max_retries):
        try:
            response = await aclient.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PAPER_INJECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
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
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                return None
    return None


async def _process_sample_async(
    aclient: AsyncOpenAI,
    args: argparse.Namespace,
    idx: int,
    row: dict,
    hall_type: str,
    should_hallucinate: bool,
) -> tuple[str | None, dict | None]:
    """Rephrase and optionally inject a hallucination for one sample."""
    llm_answer = await _rephrase_one_async(
        aclient, args.model, "\n\n".join(row["predicted_texts"]), args.temperature
    )
    if not llm_answer:
        return None, None

    if not should_hallucinate:
        return llm_answer, None

    result = await _inject_one_async(
        aclient,
        args.model,
        llm_answer,
        hall_type,
        row["question"],
        row["chunk"],
        args.max_retries,
        args.retry_delay,
        args.temperature,
    )
    processed = _process_result(result, llm_answer, hall_type, args.model)
    return llm_answer, processed


def _run_batched(
    args: argparse.Namespace,
    ds: IterableDataset,
    done_indices: set[int],
    out_f: IO[str],
    meta_f: IO[str],
) -> int:
    """Async batch processing for high-throughput endpoints (local vLLM, etc.)."""
    aclient = AsyncOpenAI(api_key=args.api_key, base_url=args.api_base_url)
    written = 0

    async def process_batches() -> None:
        nonlocal written
        batch_rows: list[dict] = []
        batch_meta: list[tuple[int, str, bool]] = []

        async def flush_batch() -> None:
            nonlocal written
            tasks = [
                _process_sample_async(aclient, args, idx, row, hall_type, should_hallucinate)
                for (idx, hall_type, should_hallucinate), row in zip(batch_meta, batch_rows)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for (idx, hall_type, should_hallucinate), row, result in zip(
                batch_meta, batch_rows, results
            ):
                if isinstance(result, Exception) or result is None:
                    continue
                llm_answer, processed = result
                if llm_answer is None:
                    continue

                if processed is not None:
                    answer = processed["hallucinated_answer"]
                    labels = processed["labels"]
                elif not should_hallucinate:
                    answer = llm_answer
                    labels = []
                else:
                    continue

                sample = {
                    "prompt": build_prompt(row["chunk"], row["question"], args.max_prompt_chars),
                    "answer": answer,
                    "labels": labels,
                    "split": "train",
                    "task_type": "markdown_generation",
                    "dataset": "acl_anthology_md",
                    "language": "en",
                }
                out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                out_f.flush()

                meta_entry = {
                    "idx": idx,
                    "llm_answer": llm_answer,
                    "row": dict(row),
                    "processed": processed,
                }
                meta_f.write(json.dumps(meta_entry, ensure_ascii=False) + "\n")
                meta_f.flush()

                written += 1

        for idx, row in enumerate(tqdm(ds, desc="verb-spans")):
            if args.limit is not None and idx >= args.limit:
                break
            if done_indices and idx < max(done_indices):
                continue
            if idx in done_indices:
                continue

            hall_type = HALLUCINATION_TYPES[idx % len(HALLUCINATION_TYPES)]
            should_hallucinate = random.random() < args.hallucination_ratio  # nosec B311
            batch_rows.append(row)
            batch_meta.append((idx, hall_type, should_hallucinate))

            if len(batch_rows) >= args.batch_size:
                await flush_batch()
                batch_rows = []
                batch_meta = []

        if batch_rows:
            await flush_batch()

    asyncio.run(process_batches())
    return written


def inject_hallucination(
    client: OpenAI,
    model: str,
    clean_answer: str,
    hall_type: str,
    user_query: str = "",
    context: str = "",
    max_retries: int = 3,
    retry_delay: int = 2,
    temperature: float = 0.8,
) -> dict | None:
    """Request structured replacement edits for hallucination injection."""
    user_msg = f"""Hallucination type to inject: {hall_type.upper()}

User's original request: {user_query}

Context (paper contents):
{context}

Correct answer to modify:
{clean_answer}

Return ONLY replacement edits for {hall_type} error(s). Do not return the full rewritten answer."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PAPER_INJECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
            )
            raw = response.choices[0].message.content.strip()

            json_match = re.search(r"\{[\s\S]*\}", raw)
            if not json_match:
                if attempt < max_retries - 1:
                    continue
                return None

            result = json.loads(json_match.group())

            if "changes" not in result or not isinstance(result["changes"], list):
                if attempt < max_retries - 1:
                    continue
                return None
            if not result["changes"]:
                if attempt < max_retries - 1:
                    continue
                return None

            return result

        except (json.JSONDecodeError, Exception) as e:
            if attempt < max_retries - 1:
                wait = retry_delay * (attempt + 1)
                print(f"  Injection error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return None


def build_prompt(
    chunk: str,
    user_query: str,
    max_chars: int = 24000,
) -> str:
    """Build the prompt (context) for a sample.

    Format: source files + documentation + user query.
    """
    parts = []

    parts.append(f"Context: {chunk}")

    parts.append(f"User request: {user_query}")

    prompt = "\n\n".join(parts)
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars]
    return prompt


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


def _locate_original_change(original_answer: str, change: dict) -> dict | None:
    """Locate a replacement span in the original answer by exact unique match."""
    original_span = change.get("original", "")
    hallucinated_span = change.get("hallucinated", "")
    if not original_span or not hallucinated_span:
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
            return None, None
        located_change = _locate_original_change(original_answer, change)
        if located_change is None:
            return None, None
        located.append(located_change)

    # Reject overlapping edits in the original answer.
    located.sort(key=lambda item: (item["start"], item["end"]))
    previous_end = -1
    for item in located:
        if item["start"] < previous_end:
            return None, None
        previous_end = item["end"]

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


def _max_allowed_coverage(answer_len: int) -> float:
    """Use a looser coverage cap for short answers and fragments."""
    if answer_len <= 400:
        return 0.40
    if answer_len <= 800:
        return 0.35
    return MAX_LABEL_COVERAGE


def _validate_labels(
    original_answer: str, hallucinated_code: str, labels: list[dict]
) -> tuple[bool, str]:
    """Validate that hallucination labels meet quality thresholds.

    :return: (is_valid, reason) tuple.
    """
    if not labels:
        return False, "no_labels"

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

    return True, ""


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


def _process_result(result, original_answer, hall_type, model):
    """Process a single injection result into a JSONL entry."""
    if result is None:
        return None
    changes = result.get("changes", [])
    hallucinated_answer, labels = apply_changes_to_answer(original_answer, changes, hall_type)
    if hallucinated_answer is None or labels is None:
        return None
    ordered_changes = _sort_changes_by_original_position(original_answer, changes)
    if ordered_changes is None:
        return None

    valid, reason = _validate_labels(original_answer, hallucinated_answer, labels)
    if not valid:
        return None
    return {
        "hallucinated_answer": hallucinated_answer,
        "labels": labels,
        "hallucination_type": hall_type,
        "injector_model": model,
        "changes": ordered_changes,
    }


def inject_hallucination_rag(
    generator: HallucinationGenerator,
    clean_answer: str,
    hall_type: str,
    user_query: str = "",
    context: str = "",
    max_retries: int = 3,
    retry_delay: int = 2,
) -> dict | None:
    """Use HallucinationGenerator to inject hallucinations."""
    for attempt in range(max_retries):
        try:
            result = generator.generate(
                context=[context],
                question=user_query,
                answer=clean_answer,
                error_types=[hall_type],
            )
            if result and result.get("hallucinated_answer"):
                return result
            if attempt < max_retries - 1:
                continue
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                wait = retry_delay * (attempt + 1)
                print(f"  RAG injection error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return None
    return None


def _process_rag_result(
    result: dict | None, original_answer: str, hall_type: str, model: str
) -> dict | None:
    """Process a HallucinationGenerator result into a JSONL entry using difflib span labels."""
    if result is None:
        return None
    hallucinated_answer = result.get("hallucinated_answer", "")
    if not hallucinated_answer or hallucinated_answer == original_answer:
        return None

    labels = []
    matcher = difflib.SequenceMatcher(None, original_answer, hallucinated_answer, autojunk=False)
    for tag, _, __, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            labels.append({"start": j1, "end": j2, "label": hall_type})

    valid, _ = _validate_labels(original_answer, hallucinated_answer, labels)
    if not valid:
        return None

    return {
        "hallucinated_answer": hallucinated_answer,
        "labels": labels,
        "hallucination_type": hall_type,
        "injector_model": model,
        "changes": [],
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate hallucinated samples from ACL verbatim spans dataset."
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key (defaults to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--api-base-url",
        default="https://api.mistral.ai/v1",
        help="API base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="mistral-small-latest",
        help="Model to use for hallucination injection (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="hallucinations.jsonl",
        help="Output JSONL file path (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to generate (default: unlimited)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per injection call (default: %(default)s)",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=2,
        help="Base retry delay in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=24000,
        help="Max characters in the prompt context (default: %(default)s)",
    )

    def hallucination_ratio_type(value: str) -> float:
        v = float(value)
        if not (0.0 <= v <= 1.0):
            raise argparse.ArgumentTypeError(
                f"hallucination-ratio must be between 0 and 1, got {v}"
            )
        return v

    parser.add_argument(
        "--hallucination-ratio",
        type=hallucination_ratio_type,
        default=1.0,
        help="Fraction of samples that are hallucinated, between 0 and 1 (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of samples to process concurrently with asyncio. "
            "Set >1 for local vLLM or other high-throughput endpoints. "
            "Only supported with --injector custom. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--injector",
        choices=["custom", "rag_fact_checker"],
        default="custom",
        help=(
            "Injection backend: 'custom' uses the built-in PAPER_INJECTION_SYSTEM_PROMPT pipeline; "
            "'rag_fact_checker' uses HallucinationGenerator from lettucedetect. (default: %(default)s)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: load dataset, inject hallucinations, write output JSONL."""
    args = parse_args()
    random.seed(args.seed)

    if not args.api_key:
        raise SystemExit("Error: API key required. Set OPENAI_API_KEY or use --api-key.")

    client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)

    generator = None
    if args.injector == "rag_fact_checker":
        from lettucedetect.models.generation import HallucinationGenerator

        generator = HallucinationGenerator(
            openai_api_key=args.api_key,
            model=args.model,
            base_url=args.api_base_url,
            temperature=args.temperature,
        )

    ds = load_dataset("KRLabsOrg/acl-verbatim-spans", "canonical", split="train", streaming=True)
    ds = ds.filter(lambda x: x["answerable"])

    meta_path = args.output + ".meta"
    done_indices: set[int] = set()
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as meta_f:
            for line in meta_f:
                line = line.strip()
                if not line:
                    continue
                if line.isdigit():
                    done_indices.add(int(line))
                    continue
                try:
                    entry = json.loads(line)
                    if "idx" in entry:
                        done_indices.add(entry["idx"])
                except json.JSONDecodeError:
                    pass
        if done_indices:
            print(f"Resuming: skipping {len(done_indices)} already-computed instance(s).")

    written = 0
    open_mode = "a" if done_indices else "w"
    with (
        open(args.output, open_mode, encoding="utf-8") as out_f,
        open(meta_path, "a", encoding="utf-8") as meta_f,
    ):
        if args.batch_size > 1 and args.injector == "custom":
            written = _run_batched(args, ds, done_indices, out_f, meta_f)
            print(f"Wrote {written} samples to {args.output}")
            return

        hall_debt = False  # unpaid hallucinations from previous failed injections
        for idx, row in enumerate(tqdm(ds, desc="verb-spans")):
            if args.limit is not None and idx >= args.limit:
                break

            if done_indices and idx < max(done_indices):
                continue

            hall_type = HALLUCINATION_TYPES[idx % len(HALLUCINATION_TYPES)]
            llm_answer = rephrase_verbatim_answer(
                client, args.model, "\n\n".join(row["predicted_texts"])
            )
            if llm_answer is None:
                continue

            should_hallucinate = random.random() < args.hallucination_ratio  # nosec B311
            processed = None
            if should_hallucinate or hall_debt:
                if args.injector == "rag_fact_checker":
                    injection_result = inject_hallucination_rag(
                        generator=generator,
                        clean_answer=llm_answer,
                        hall_type=hall_type,
                        user_query=row["question"],
                        context=row["chunk"],
                        max_retries=args.max_retries,
                        retry_delay=args.retry_delay,
                    )
                    processed = _process_rag_result(injection_result, llm_answer, hall_type, args.model)
                else:
                    injection_result = inject_hallucination(
                        client=client,
                        model=args.model,
                        clean_answer=llm_answer,
                        hall_type=hall_type,
                        user_query=row["question"],
                        context=row["chunk"],
                        max_retries=args.max_retries,
                        retry_delay=args.retry_delay,
                        temperature=args.temperature,
                    )
                    processed = _process_result(injection_result, llm_answer, hall_type, args.model)
                if processed is not None:
                    answer = processed["hallucinated_answer"]
                    labels = processed["labels"]
                    hall_debt = False
                else:
                    hall_debt = True
                    continue
            else:
                answer = llm_answer
                labels = []

            sample = {
                "prompt": build_prompt(row["chunk"], row["question"], args.max_prompt_chars),
                "answer": answer,
                "labels": labels,
                "split": "train",
                "task_type": "markdown_generation",
                "dataset": "acl_anthology_md",
                "language": "en",
            }
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            out_f.flush()

            meta_entry = {
                "idx": idx,
                "llm_answer": llm_answer,
                "row": dict(row),
                "processed": processed,
            }
            meta_f.write(json.dumps(meta_entry, ensure_ascii=False) + "\n")
            meta_f.flush()

            written += 1

    print(f"Wrote {written} samples to {args.output}")


if __name__ == "__main__":
    main()
