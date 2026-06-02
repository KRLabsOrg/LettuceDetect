"""Phase 6: Inject hallucinations into code answers via the shared injector.

Uses the unified menu-style injector (the model self-labels each edit's type and
the server's thinking trace is captured) over the shared resumable
``run_batched`` runner — the same path every other source uses. Output records
feed Phase 7 (``sample_assembler``): one JSONL line per instance with the
hallucinated answer, character-span ``labels`` (native type + unified
category/subcategory), and the model ``reasoning``.
"""

import json

from openai import AsyncOpenAI

from lettucedetect.generation.injection import _map_menu_labels, inject_async
from lettucedetect.generation.runner import Outcome, run_batched_sync

from .config import (
    API_BASE_URL,
    API_KEY,
    BATCH_SIZE,
    HALLUCINATED_PATH,
    HALLUCINATION_TEMPERATURE,
    INJECTION_FAILURES_PATH,
    MAX_PROMPT_CHARS,
    MODEL,
    token_limit_kwargs,
)

# Thinking traces run ~4-5K tokens on complex code; leave room for the JSON
# answer afterwards or it truncates to an empty result.
INJECTION_MAX_TOKENS = 12000

# Menu-style code injection prompt: the model picks AND labels the error type per
# edit (no forced type), must keep edits grounded and genuinely wrong, and always
# injects (a fabricated reference / unsupported addition is always available).
CODE_INJECTION_PROMPT = """\
You inject a subtle, GROUNDED hallucination into a correct code answer, to build a
hallucination-detection dataset. Return ONLY small replacement edits as JSON; the
pipeline applies them and everything outside the edits stays identical.

GROUNDING (critical):
- Every error MUST be detectable by comparing the answer against the PROVIDED
  source context OR the user's request. A reader with only those must be able to
  tell it is wrong.
- Use only names that appear in the provided context. Do NOT rely on outside
  library knowledge. (Exception: a structural edit deliberately introduces a name
  that is NOT in the context.)

THE EDIT MUST BE A REAL ERROR:
- It must change behavior or meaning. NEVER make a semantically-equivalent or
  cosmetic edit (e.g. `1.` <-> `1.0`, whitespace, reordering, renaming an unused
  local variable). Assigning an existing value to a new variable is NOT an error.
- ALWAYS inject 1-3 hallucinations. A grounded error is always constructible:
  - First choice: a grounded CONTRADICTION (wrong value/operator/name that
    conflicts with the context or request).
  - If the answer has no value/logic you can contradict, fall back to a
    FABRICATED reference (call/reference a name that does NOT appear in the
    context) or an UNSUPPORTED addition (a statement or behavior the request
    never asked for and the context does not support).
  One of these is always available, so never return an empty list.

ERROR TYPES (choose the one that fits each edit and label it):
- structural: replace a real name (function/method/attribute/parameter) with one
  that does NOT appear in the context (a fabricated reference).
- behavioral: keep real names but use a wrong value/argument/operator/condition
  that contradicts the context.
- semantic: make the code contradict the USER REQUEST or behavior explicitly
  shown in the context. Never use library knowledge; if neither the request nor
  the context specifies the behavior, do not inject a semantic error.

EDIT RULES:
- 1-3 edits, in different places. Each "original" is an exact substring that
  appears exactly once in the answer.
- Each replacement is the SMALLEST exact substring that captures the change (an
  expression or token, not a whole line unless necessary).
- Only edit code (inside fences if present); keep any explanation/markdown text
  unchanged. No comments hinting at the error; no words like bug/wrong/incorrect.
- The edited code must stay syntactically valid.

Respond ONLY with JSON (no prose, no code fence):
{"changes": [{"original": "...", "hallucinated": "...",
  "hallucination_type": "structural|behavioral|semantic",
  "explanation": "what in the context or request this contradicts"}]}
"""


def _original_id(iid: str) -> str:
    """Strip sub-instance suffix to get the original SWE-bench instance ID."""
    return iid.split("::")[0]


def build_source_context(source_data: dict) -> str:
    """Build the source-code context string from cached source data (truncated)."""
    parts = [
        f"File: {filepath}\n```python\n{content}\n```"
        for filepath, content in source_data.get("source_files", {}).items()
    ]
    context = "\n\n".join(parts)
    return context[:MAX_PROMPT_CHARS] if len(context) > MAX_PROMPT_CHARS else context


def _user_msg(query: str, context: str, documentation: dict | None, clean_answer: str) -> str:
    """Build the injection request (grounding = source context + request + docs)."""
    docs_section = ""
    if documentation:
        rendered = "\n\n".join(f"Documentation for {lib}:\n{d}" for lib, d in documentation.items())
        docs_section = f"\n\nLibrary documentation:\n{rendered}"
    return (
        f"User's request: {query}\n\n"
        f"Source context:\n{context}{docs_section}\n\n"
        f"Correct answer to modify:\n{clean_answer}\n\n"
        f"Inject 1-3 grounded hallucinations."
    )


async def _inject_record(
    aclient: AsyncOpenAI,
    model: str,
    instance_id: str,
    fmt_data: dict,
    query: str,
    context: str,
    documentation: dict | None,
) -> tuple[dict | None, str | None]:
    """Inject via the shared menu injector; return (record, failure_reason)."""
    clean_answer = fmt_data.get("answer", "")
    format_type = fmt_data.get("format_type", "fragment")
    is_mixed = format_type == "code_with_explanation"  # prose+code: enforce fences
    res = await inject_async(
        aclient,
        model,
        clean_answer=clean_answer,
        hall_type="",  # menu: per-edit type comes from the model, not a forced one
        system_prompt=CODE_INJECTION_PROMPT,
        user_msg=_user_msg(query, context, documentation, clean_answer),
        temperature=HALLUCINATION_TEMPERATURE,
        completion_kwargs=token_limit_kwargs(model, INJECTION_MAX_TOKENS),
        require_spans_in_code=is_mixed,
        require_balanced_fences=is_mixed,
    )
    if not res.ok:
        return None, res.reason or "injection_failed"
    res = _map_menu_labels(res, "code")
    return {
        "instance_id": instance_id,
        "hallucinated_answer": res.hallucinated_answer,
        "labels": res.labels,
        "hallucination_type": res.labels[0]["label"] if res.labels else "menu",
        "injector_model": model,
        "format_type": format_type,
        "changes": res.changes,
        "reasoning": res.reasoning,
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
) -> list[dict]:
    """Run Phase 6: inject hallucinations into the selected instances.

    Resumable and failure-logging via the shared runner; already-injected
    instances are skipped on restart.
    """
    print("=" * 60)
    print(f"Phase 6: Hallucination Injection ({base_url}, {model}, batch={BATCH_SIZE})")
    print("=" * 60)
    docs = docs or {}
    source_cache = source_cache or {}
    HALLUCINATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

    items = [
        inst
        for inst in instances_to_inject
        if inst["instance_id"] in formats and formats[inst["instance_id"]].get("answer")
    ]

    async def process(inst: dict) -> Outcome:
        iid = inst["instance_id"]
        orig = _original_id(iid)
        source_data = source_cache.get(orig, {})
        context = (
            build_source_context(source_data)
            if source_data
            else inst.get("problem_statement", "")
        )
        record, reason = await _inject_record(
            aclient, model, iid, formats[iid], queries.get(orig, ""), context, docs.get(orig, {})
        )
        if record is None:
            return Outcome(key=iid, ok=False, reason=reason)
        return Outcome(key=iid, ok=True, record=record)

    stats = run_batched_sync(
        items,
        process,
        out_path=HALLUCINATED_PATH,
        failures_path=INJECTION_FAILURES_PATH,
        key_of=lambda inst: inst["instance_id"],
        record_key=lambda rec: rec["instance_id"],
        batch_size=BATCH_SIZE,
        on_progress=lambda s: print(f"  Phase 6: {s}"),
    )
    print(f"Done: {stats}")
    return load_existing_hallucinations(HALLUCINATED_PATH)


def load_existing_hallucinations(path: object = HALLUCINATED_PATH) -> list[dict]:
    """Load all injected records (also used by Phase 7)."""
    records = []
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


if __name__ == "__main__":
    print("Run via pipeline.py")
