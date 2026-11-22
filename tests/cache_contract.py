"""Reusable contract suite that every ``ResponseCache`` backend must pass.

The suite reads the public behaviour of a cache and nothing else, so it does not care how a
backend stores its data. ``InMemoryCache`` runs it, and so does every later backend (``FileCache``).

How a backend task plugs in
===========================

1. Subclass :class:`CacheContract` inside a ``tests/test_*_pytest.py`` file. The base class is not
   collected by pytest itself, only your ``Test...`` subclass is::

       from tests.cache_contract import CacheContract

       class TestMyCacheContract(CacheContract):
           def make_cache(self, on_call=None):
               return MyCache(on_call=on_call)

2. Implement ``make_cache(on_call)``. It returns a fresh, empty cache that works offline. When
   ``on_call`` is not ``None``, the cache must call ``on_call()`` once per storage operation
   (``get``, ``set`` or ``delete``), on the thread that runs the operation, before it touches its
   storage. The suite uses it to make a backend slow, to see which thread runs it, and to block it
   on an event. A backend with native async storage calls it from its async methods too (from a
   worker thread or any other way that keeps the loop free).
3. Every check is a plain method without pytest fixtures, so ``tests/test_cache_contract_pytest.py``
   can also run a check against a deliberately broken backend and assert that it fails.

About cancelling ``aset``
=========================

Cancelling the awaiting task of a ``SyncCacheBase`` backend stops the wait only. The storage call
keeps running until it returns, because Python cannot stop a thread. So the cancel check does not
say which of the two values wins. It says that the key reads as the old value or the new one, at
every moment, and that the cache works afterwards.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import threading
import time
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from lettucedetect.detectors.response_cache import ResponseCache

_PAIRS = (("get", "aget"), ("set", "aset"), ("delete", "adelete"))
_WAIT_SECONDS = 10.0
_TTL = 0.2
_AFTER_TTL = 0.4
_SLOW_SECONDS = 0.3


class CacheProbe:
    """The ``on_call`` hook. It records the thread of each call, and can delay or block a call."""

    def __init__(self) -> None:
        """Create a probe that records only."""
        self.delay = 0.0
        self.threads: list[int] = []
        self.entered = threading.Event()
        self._gate: threading.Event | None = None

    def block_next(self, gate: threading.Event) -> None:
        """Make the next call wait until ``gate`` is set."""
        self.entered.clear()
        self._gate = gate

    def __call__(self) -> None:
        """Record the thread, then wait as armed."""
        self.threads.append(threading.get_ident())
        gate, self._gate = self._gate, None
        self.entered.set()
        if gate is not None:
            assert gate.wait(_WAIT_SECONDS), "the probe was never released"
        if self.delay:
            time.sleep(self.delay)


async def _count_ticks(awaitable: Awaitable[object]) -> int:
    """Await ``awaitable`` while a ticker runs on the loop; return how often it ticked."""
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        await awaitable
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return ticks


class CacheContract:
    """The checks every cache backend must pass. Subclass it and implement ``make_cache``."""

    def make_cache(self, on_call: Callable[[], None] | None = None) -> ResponseCache:
        """Return a fresh cache that calls ``on_call()`` once per storage operation."""
        raise NotImplementedError

    # -- the protocol -----------------------------------------------------------------------

    def test_protocol_shape(self) -> None:
        """Each ``a...`` method is a coroutine function with the arguments of its sync pair."""
        cache = self.make_cache()
        for sync_name, async_name in _PAIRS:
            assert hasattr(cache, sync_name), f"the cache has no {sync_name}"
            assert hasattr(cache, async_name), f"the cache has no {async_name}"
            sync_method = getattr(cache, sync_name)
            async_method = getattr(cache, async_name)
            assert inspect.iscoroutinefunction(async_method), f"{async_name} is not async def"
            sync_params = [
                (p.name, p.kind, p.default)
                for p in inspect.signature(sync_method).parameters.values()
            ]
            async_params = [
                (p.name, p.kind, p.default)
                for p in inspect.signature(async_method).parameters.values()
            ]
            assert async_params == sync_params, (
                f"{async_name}{inspect.signature(async_method)} differs from "
                f"{sync_name}{inspect.signature(sync_method)}"
            )
        ttl = inspect.signature(cache.set).parameters["ttl_seconds"]
        assert ttl.kind is inspect.Parameter.KEYWORD_ONLY, "ttl_seconds must be keyword-only"
        assert ttl.default is None, "ttl_seconds must default to None"

    # -- sync behaviour ---------------------------------------------------------------------

    def test_value_reads_back(self) -> None:
        """A stored value reads back unchanged, including non-ASCII text."""
        cache = self.make_cache()
        cache.set("k", 'héllo ✓\n"quoted"')
        assert cache.get("k") == 'héllo ✓\n"quoted"'

    def test_empty_string_is_a_value(self) -> None:
        """An empty string is stored and read back; only a missing key is ``None``."""
        cache = self.make_cache()
        cache.set("k", "")
        assert cache.get("k") == ""

    def test_missing_key_is_none(self) -> None:
        """A key that was never set reads as ``None``."""
        assert self.make_cache().get("missing") is None

    def test_overwrite(self) -> None:
        """A second ``set`` replaces the first value."""
        cache = self.make_cache()
        cache.set("k", "one")
        cache.set("k", "two")
        assert cache.get("k") == "two"

    def test_delete(self) -> None:
        """``delete`` removes a key and leaves the others."""
        cache = self.make_cache()
        cache.set("a", "1")
        cache.set("b", "2")
        cache.delete("a")
        assert cache.get("a") is None
        assert cache.get("b") == "2"

    def test_delete_missing_key_does_nothing(self) -> None:
        """Deleting a missing key is not an error."""
        cache = self.make_cache()
        cache.delete("missing")
        assert cache.get("missing") is None

    def test_ttl_expires(self) -> None:
        """A value with a TTL reads back at first and as ``None`` after the TTL."""
        cache = self.make_cache()
        cache.set("k", "v", ttl_seconds=_TTL)
        assert cache.get("k") == "v"
        time.sleep(_AFTER_TTL)
        assert cache.get("k") is None

    def test_no_ttl_never_expires(self) -> None:
        """A value without a TTL is still there after the time a TTL value would have expired."""
        cache = self.make_cache()
        cache.set("k", "v")
        cache.set("other", "w", ttl_seconds=_TTL)
        time.sleep(_AFTER_TTL)
        assert cache.get("other") is None
        assert cache.get("k") == "v"

    def test_set_without_ttl_clears_the_old_deadline(self) -> None:
        """Overwriting a value that has a TTL with one that has none removes the deadline."""
        cache = self.make_cache()
        cache.set("k", "old", ttl_seconds=_TTL)
        cache.set("k", "new")
        time.sleep(_AFTER_TTL)
        assert cache.get("k") == "new"

    def test_a_new_ttl_replaces_the_old_deadline(self) -> None:
        """Overwriting a value that has a short TTL with a longer one keeps it past the first."""
        cache = self.make_cache()
        cache.set("k", "old", ttl_seconds=_TTL)
        cache.set("k", "new", ttl_seconds=60)
        time.sleep(_AFTER_TTL)
        assert cache.get("k") == "new"

    # -- async behaviour --------------------------------------------------------------------

    async def test_async_methods_work(self) -> None:
        """``aset``, ``aget`` and ``adelete`` store, read and remove a value."""
        cache = self.make_cache()
        assert await cache.aget("k") is None
        await cache.aset("k", "v")
        assert await cache.aget("k") == "v"
        await cache.adelete("k")
        assert await cache.aget("k") is None
        await cache.adelete("k")

    async def test_each_sync_method_equals_its_async_pair(self) -> None:
        """For the same state, a sync method and its async pair return the same value."""
        cache = self.make_cache()
        cache.set("k", "v")
        assert cache.get("k") == await cache.aget("k") == "v"
        assert cache.get("missing") is None
        assert await cache.aget("missing") is None
        assert cache.set("a", "1") is None
        assert await cache.aset("b", "2") is None
        assert cache.delete("a") is None
        assert await cache.adelete("b") is None
        assert cache.get("a") is None
        assert cache.get("b") is None
        cache.set("c", "")
        assert cache.get("c") == await cache.aget("c") == ""

    async def test_aset_with_ttl_expires_like_set(self) -> None:
        """``aset(..., ttl_seconds=...)`` expires like ``set``, seen through both read methods."""
        cache = self.make_cache()
        await cache.aset("k", "v", ttl_seconds=_TTL)
        assert await cache.aget("k") == "v"
        assert cache.get("k") == "v"
        await asyncio.sleep(_AFTER_TTL)
        assert await cache.aget("k") is None
        assert cache.get("k") is None

    async def test_value_written_by_set_reads_via_aget_and_the_reverse(self) -> None:
        """The sync and async methods share one storage."""
        cache = self.make_cache()
        cache.set("sync", "from set")
        assert await cache.aget("sync") == "from set"
        await cache.aset("async", "from aset")
        assert cache.get("async") == "from aset"
        await cache.aset("sync", "replaced")
        assert cache.get("sync") == "replaced"
        cache.delete("async")
        assert await cache.aget("async") is None

    async def test_async_methods_run_off_the_loop_thread(self) -> None:
        """The storage work of every async method runs on a thread other than the loop's."""
        probe = CacheProbe()
        cache = self.make_cache(probe)
        cache.set("k", "v")
        probe.threads.clear()
        await cache.aget("k")
        await cache.aset("k2", "v")
        await cache.adelete("k")
        assert len(probe.threads) == 3, "the cache must call on_call once per storage operation"
        assert threading.get_ident() not in probe.threads, "an async method ran on the loop thread"

    async def test_loop_keeps_running_while_an_async_method_waits(self) -> None:
        """A slow storage call does not stop other tasks on the loop (the ticker keeps ticking)."""
        probe = CacheProbe()
        cache = self.make_cache(probe)
        cache.set("k", "v")
        probe.delay = _SLOW_SECONDS
        calls = {
            "aget": lambda: cache.aget("k"),
            "aset": lambda: cache.aset("k2", "v"),
            "adelete": lambda: cache.adelete("k"),
        }
        for name, call in calls.items():
            ticks = await _count_ticks(call())
            assert ticks >= 5, f"the loop was blocked while {name} waited ({ticks} ticks)"

    async def test_cancelled_aset_leaves_the_cache_valid(self) -> None:
        """After cancelling an ``aset``, the key is the old value or the new one, never else."""
        probe = CacheProbe()
        cache = self.make_cache(probe)
        cache.set("k", "old value")
        gate = threading.Event()
        probe.block_next(gate)
        task = asyncio.create_task(cache.aset("k", "new value"))
        assert await asyncio.to_thread(probe.entered.wait, _WAIT_SECONDS), "aset never started"
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        try:
            # The storage call is still blocked: the cache is in the middle of the write.
            assert cache.get("k") in ("old value", "new value")
        finally:
            gate.set()
        await asyncio.sleep(0.2)  # the storage call finishes: nothing stops a thread
        assert cache.get("k") in ("old value", "new value")
        cache.set("k", "later")
        assert cache.get("k") == "later"
        await cache.aset("other", "value")
        assert await cache.aget("other") == "value"

    # -- concurrency ------------------------------------------------------------------------

    def test_concurrent_threads_lose_no_key(self) -> None:
        """Writers on several threads, each with its own keys, lose none of them."""
        cache = self.make_cache()
        writers, per_writer = 8, 25

        def write(writer: int) -> None:
            for i in range(per_writer):
                cache.set(f"k{writer}-{i}", f"v{writer}-{i}")

        with ThreadPoolExecutor(max_workers=writers) as pool:
            list(pool.map(write, range(writers)))
        for writer in range(writers):
            for i in range(per_writer):
                assert cache.get(f"k{writer}-{i}") == f"v{writer}-{i}", f"lost k{writer}-{i}"

    async def test_concurrent_aset_over_gather_loses_no_key(self) -> None:
        """Many ``aset`` calls awaited together with ``asyncio.gather`` lose no key."""
        cache = self.make_cache()
        count = 200
        await asyncio.gather(*(cache.aset(f"k{i}", f"v{i}") for i in range(count)))
        values = await asyncio.gather(*(cache.aget(f"k{i}") for i in range(count)))
        assert values == [f"v{i}" for i in range(count)]


def run_check(contract: CacheContract, name: str) -> None:
    """Run one check of a contract instance, awaiting it when it is a coroutine function."""
    method = getattr(contract, name)
    if inspect.iscoroutinefunction(method):
        asyncio.run(method())
    else:
        method()
