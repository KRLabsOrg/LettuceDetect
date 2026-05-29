"""Reusable orchestration for dataset generation.

Centralizes the batched, resumable, failure-logging generation loop that the
code-hallucination pipeline previously implemented inline. Every source adapter
uses this so the proven behavior — async batching against local vLLM, skipping
already-done items on restart, incremental flush, and failure logging for later
replay — is written once and never re-dropped.

The per-item work is supplied as an async callable ``process(item) -> Outcome``;
the runner handles batching, resumability, and I/O.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Outcome:
    """Result of processing one work item.

    On success, ``record`` is the dict to append to the output JSONL. On failure,
    ``reason`` explains why (logged to the failures file for later inspection or
    replay). ``key`` uniquely identifies the item for resumability.
    """

    key: str
    ok: bool
    record: dict | None = None
    reason: str | None = None
    extra: dict = field(default_factory=dict)


def load_done_keys(path: str | Path, key_field: str = "key") -> set[str]:
    """Return the set of already-completed keys from an output JSONL file."""
    path = Path(path)
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = entry.get(key_field)
            if key is not None:
                done.add(key)
    return done


async def run_batched(
    items: Iterable,
    process: Callable[[object], Awaitable[Outcome]],
    *,
    out_path: str | Path,
    failures_path: str | Path | None = None,
    key_of: Callable[[object], str],
    done_keys: set[str] | None = None,
    batch_size: int = 16,
    key_field: str = "key",
    progress_every: int = 50,
    on_progress: Callable[[dict], None] | None = None,
) -> dict:
    """Run ``process`` over ``items`` in async batches, resumably.

    - Skips items whose ``key_of(item)`` is already present in ``done_keys``
      (defaults to keys found in ``out_path``).
    - Appends each success record (augmented with ``key_field``) to ``out_path``
      and flushes per batch, so a crash never loses completed work.
    - Logs each failure to ``failures_path`` (if given) as ``{key, reason, ...extra}``.

    Returns a stats dict.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if done_keys is None:
        done_keys = load_done_keys(out_path, key_field)

    todo = [it for it in items if key_of(it) not in done_keys]
    stats = {"total": len(todo), "ok": 0, "fail": 0, "skipped_done": 0}

    ferr = open(failures_path, "a") if failures_path else None
    try:
        with open(out_path, "a") as fout:
            for start in range(0, len(todo), batch_size):
                batch = todo[start : start + batch_size]
                results = await asyncio.gather(
                    *(process(it) for it in batch), return_exceptions=True
                )
                for it, res in zip(batch, results):
                    if isinstance(res, Exception):
                        stats["fail"] += 1
                        if ferr:
                            ferr.write(
                                json.dumps({"key": key_of(it), "reason": f"exception:{res}"}) + "\n"
                            )
                        continue
                    if res.ok and res.record is not None:
                        record = {key_field: res.key, **res.record}
                        fout.write(json.dumps(record) + "\n")
                        stats["ok"] += 1
                    else:
                        stats["fail"] += 1
                        if ferr:
                            ferr.write(
                                json.dumps(
                                    {"key": res.key, "reason": res.reason or "unknown", **res.extra}
                                )
                                + "\n"
                            )
                fout.flush()
                if ferr:
                    ferr.flush()
                done = start + len(batch)
                if on_progress and (done % progress_every == 0 or done >= len(todo)):
                    on_progress({**stats, "processed": done})
    finally:
        if ferr:
            ferr.close()

    return stats


def run_batched_sync(
    items: Iterable,
    process: Callable[[object], Awaitable[Outcome]],
    **kwargs,
) -> dict:
    """Drive :func:`run_batched` from synchronous code."""
    return asyncio.run(run_batched(items, process, **kwargs))
