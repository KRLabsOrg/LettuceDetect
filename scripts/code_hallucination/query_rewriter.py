"""Phase 3: Rewrite problem statements into natural user queries via LLM.

Supports both sequential (remote API) and async batch (local vLLM) modes.
Set BATCH_SIZE>1 env var for parallel requests to local vLLM.
"""

import asyncio
import json
import textwrap
import time

from openai import AsyncOpenAI, OpenAI

from .config import (
    API_BASE_URL,
    API_KEY,
    BATCH_SIZE,
    LLM_TEMPERATURE,
    MAX_RETRIES,
    MODEL,
    QUERIES_PATH,
    RETRY_DELAY,
    token_limit_kwargs,
)

REWRITE_SYSTEM_PROMPT = textwrap.dedent("""\
    You transform GitHub issue descriptions into realistic user queries
    that a developer would type into an AI coding assistant (like Claude Code or Cursor).

    Rules:
    - Make it conversational and natural
    - Keep the core technical ask but remove GitHub formatting
    - Remove reproduction steps, stack traces, verbose details
    - Keep it to 1-3 sentences
    - Don't mention "issue" or "bug report"
    - Sound like someone asking for help, not filing a report
""")


def get_client(api_key: str = API_KEY, base_url: str = API_BASE_URL) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def llm_call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = 300,
) -> str:
    """Make an LLM call with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                **token_limit_kwargs(model, max_tokens),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  LLM error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def rewrite_query(client: OpenAI, model: str, problem_statement: str, repo: str) -> str:
    """Rewrite a problem statement into a natural user query."""
    user_msg = f"Repository: {repo}\n\nGitHub Issue:\n{problem_statement[:3000]}"
    return llm_call(client, model, REWRITE_SYSTEM_PROMPT, user_msg)


async def _rewrite_query_async(
    aclient: AsyncOpenAI, model: str, problem_statement: str, repo: str
) -> str:
    """Async version of rewrite_query with retries."""
    user_msg = f"Repository: {repo}\n\nGitHub Issue:\n{problem_statement[:3000]}"
    for attempt in range(MAX_RETRIES):
        try:
            response = await aclient.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=LLM_TEMPERATURE,
                **token_limit_kwargs(model, 300),
            )
            return response.choices[0].message.content.strip()
        except Exception:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


def load_existing_queries(path=QUERIES_PATH) -> dict[str, str]:
    """Load already-processed queries for resumability."""
    existing = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    existing[entry["instance_id"]] = entry["query"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return existing


def run(
    instances: list[dict],
    api_key: str = API_KEY,
    base_url: str = API_BASE_URL,
    model: str = MODEL,
):
    """Run Phase 3: Rewrite all queries.

    Uses async batch processing when BATCH_SIZE > 1 (for local vLLM).
    Falls back to sequential processing for remote APIs (BATCH_SIZE=1).
    """
    print("=" * 60)
    print("Phase 3: Query Rewriting")
    print("=" * 60)

    QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using {base_url} with model {model}")
    print(f"Batch size: {BATCH_SIZE}")

    existing = load_existing_queries()
    print(f"Already processed: {len(existing)} queries")

    to_process = [inst for inst in instances if inst["instance_id"] not in existing]
    print(f"Remaining: {len(to_process)} queries to process")

    if not to_process:
        print("Nothing to do.")
        return

    if BATCH_SIZE > 1:
        _run_async(to_process, existing, api_key, base_url, model)
    else:
        _run_sequential(to_process, existing, api_key, base_url, model)


def _run_sequential(to_process, existing, api_key, base_url, model):
    client = get_client(api_key, base_url)
    processed = 0
    failed = 0

    with open(QUERIES_PATH, "a") as f:
        for inst in to_process:
            instance_id = inst["instance_id"]
            try:
                query = rewrite_query(client, model, inst["problem_statement"], inst["repo"])
                f.write(json.dumps({"instance_id": instance_id, "query": query}) + "\n")
                f.flush()
                processed += 1
                if processed % 100 == 0:
                    print(f"  Phase 3: {processed}/{len(to_process)} (failed: {failed})")
            except Exception as e:
                print(f"  ERROR {instance_id}: {e}")
                failed += 1

    print(f"\nDone: {processed} new queries, {failed} failed")
    print(f"Total queries: {len(existing) + processed}")


def _run_async(to_process, existing, api_key, base_url, model):
    aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)
    results: list[tuple[str, str]] = []
    failed = 0

    async def process_batches():
        nonlocal failed
        with open(QUERIES_PATH, "a") as f:
            for batch_start in range(0, len(to_process), BATCH_SIZE):
                batch = to_process[batch_start : batch_start + BATCH_SIZE]
                tasks = [
                    _rewrite_query_async(aclient, model, inst["problem_statement"], inst["repo"])
                    for inst in batch
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                for inst, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        print(f"  ERROR {inst['instance_id']}: {result}")
                        failed += 1
                    else:
                        entry = {"instance_id": inst["instance_id"], "query": result}
                        f.write(json.dumps(entry) + "\n")
                        results.append((inst["instance_id"], result))

                done = batch_start + len(batch)
                if done % 500 == 0 or done >= len(to_process):
                    f.flush()
                    print(f"  Phase 3: {done}/{len(to_process)} (failed: {failed})")

    asyncio.run(process_batches())
    print(f"\nDone: {len(results)} new queries, {failed} failed")
    print(f"Total queries: {len(existing) + len(results)}")


if __name__ == "__main__":
    from .swebench_loader import load_instances

    instances = load_instances()
    run(instances)
