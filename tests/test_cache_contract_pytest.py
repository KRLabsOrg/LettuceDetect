"""Run the cache contract suite against good backends and deliberately broken ones."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable

import pytest

from lettucedetect.detectors import InMemoryCache, ResponseCache, SyncCacheBase
from tests.cache_contract import CacheContract, run_check

Hook = Callable[[], None] | None


class HookedInMemoryCache(InMemoryCache):
    """``InMemoryCache`` that calls the contract's ``on_call`` before each storage operation."""

    def __init__(self, on_call: Hook = None) -> None:
        """Create the cache."""
        super().__init__()
        self._on_call = on_call

    def get(self, key):
        """Call the hook, then read."""
        if self._on_call:
            self._on_call()
        return super().get(key)

    def set(self, key, value, *, ttl_seconds=None):
        """Call the hook, then write."""
        if self._on_call:
            self._on_call()
        super().set(key, value, ttl_seconds=ttl_seconds)

    def delete(self, key):
        """Call the hook, then delete."""
        if self._on_call:
            self._on_call()
        super().delete(key)


class DictBackend(SyncCacheBase):
    """A backend that implements only ``get``, ``set`` and ``delete`` on ``SyncCacheBase``."""

    def __init__(self, on_call: Hook = None) -> None:
        """Create the backend."""
        self._data: dict[str, tuple[str, float | None]] = {}
        self._lock = threading.Lock()
        self._on_call = on_call

    def _hook(self) -> None:
        if self._on_call:
            self._on_call()

    def get(self, key):
        """Return the value, or ``None`` when it is missing or expired."""
        self._hook()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value, deadline = entry
            if deadline is not None and time.monotonic() >= deadline:
                return None
            return value

    def set(self, key, value, *, ttl_seconds=None):
        """Store the value."""
        self._hook()
        deadline = None if ttl_seconds is None else time.monotonic() + ttl_seconds
        with self._lock:
            self._data[key] = (value, deadline)

    def delete(self, key):
        """Remove the key."""
        self._hook()
        with self._lock:
            self._data.pop(key, None)


class NativeAsyncCache:
    """Implements the protocol directly, with no ``SyncCacheBase``: async storage of its own."""

    def __init__(self, on_call: Hook = None) -> None:
        """Create the cache."""
        self._inner = DictBackend(on_call)
        self._on_call = on_call

    def get(self, key):
        """Read synchronously."""
        return self._inner.get(key)

    def set(self, key, value, *, ttl_seconds=None):
        """Write synchronously."""
        self._inner.set(key, value, ttl_seconds=ttl_seconds)

    def delete(self, key):
        """Delete synchronously."""
        self._inner.delete(key)

    async def aget(self, key):
        """Read; the blocking part runs off the loop."""
        return await asyncio.to_thread(self._inner.get, key)

    async def aset(self, key, value, *, ttl_seconds=None):
        """Write; the blocking part runs off the loop."""
        await asyncio.to_thread(self._inner.set, key, value, ttl_seconds=ttl_seconds)

    async def adelete(self, key):
        """Delete; the blocking part runs off the loop."""
        await asyncio.to_thread(self._inner.delete, key)


class TestInMemoryCacheContract(CacheContract):
    """``InMemoryCache`` passes the whole suite."""

    def make_cache(self, on_call=None):
        """Make cache."""
        return HookedInMemoryCache(on_call)


class TestSyncCacheBaseContract(CacheContract):
    """A backend with only ``get``, ``set`` and ``delete`` passes, async methods included."""

    def make_cache(self, on_call=None):
        """Make cache."""
        return DictBackend(on_call)

    def test_backend_defines_no_async_method_of_its_own(self):
        """Backend defines no async method of its own."""
        for name in ("aget", "aset", "adelete"):
            assert name not in vars(DictBackend)
            assert name in vars(SyncCacheBase)


class TestDirectProtocolContract(CacheContract):
    """A class that implements the protocol directly is not forced to use ``SyncCacheBase``."""

    def make_cache(self, on_call=None):
        """Make cache."""
        return NativeAsyncCache(on_call)

    def test_is_a_response_cache_without_the_base(self):
        """Is a response cache without the base."""
        assert isinstance(NativeAsyncCache(), ResponseCache)
        assert not issubclass(NativeAsyncCache, SyncCacheBase)


# -- deliberately broken backends: each must fail the matching check -------------------------


class MissingAget(DictBackend):
    """Has no ``aget`` at all."""

    @property
    def aget(self):
        """Aget."""
        raise AttributeError("aget")


class SyncAset(DictBackend):
    """``aset`` is a plain ``def`` that blocks the loop."""

    def aset(self, key, value, *, ttl_seconds=None):
        """Aset."""
        self.set(key, value, ttl_seconds=ttl_seconds)


class AsetWithoutTtl(DictBackend):
    """``aset`` has no ``ttl_seconds`` argument."""

    async def aset(self, key, value):
        """Aset."""
        await asyncio.to_thread(self.set, key, value)


class AsetIgnoresTtl(DictBackend):
    """``aset`` accepts ``ttl_seconds`` and drops it."""

    async def aset(self, key, value, *, ttl_seconds=None):
        """Aset."""
        await asyncio.to_thread(self.set, key, value)


class ExpiredStillReads(DictBackend):
    """``get`` ignores the deadline."""

    def get(self, key):
        """Get."""
        self._hook()
        entry = self._data.get(key)
        return None if entry is None else entry[0]


class MissingAsEmpty(DictBackend):
    """``aget`` returns ``""`` for a missing key, ``get`` returns ``None``."""

    async def aget(self, key):
        """Aget."""
        return await asyncio.to_thread(self.get, key) or ""


class OnLoop(DictBackend):
    """The async methods run the storage call on the event loop's thread."""

    async def aget(self, key):
        """Aget."""
        return self.get(key)

    async def aset(self, key, value, *, ttl_seconds=None):
        """Aset."""
        self.set(key, value, ttl_seconds=ttl_seconds)

    async def adelete(self, key):
        """Adelete."""
        self.delete(key)


class LostUpdates(DictBackend):
    """``set`` copies the dict, yields, and writes the copy back: writers overwrite each other."""

    def set(self, key, value, *, ttl_seconds=None):
        """Set."""
        self._hook()
        snapshot = dict(self._data)
        time.sleep(0.001)
        snapshot[key] = (value, None if ttl_seconds is None else time.monotonic() + ttl_seconds)
        self._data = snapshot


class PartialWrite(DictBackend):
    """``set`` stores a half-written value, then does the slow part, then the real value."""

    def set(self, key, value, *, ttl_seconds=None):
        """Set."""
        with self._lock:
            self._data[key] = (value[: len(value) // 2], None)
        self._hook()
        super().set(key, value, ttl_seconds=ttl_seconds)


def _contract_of(factory: Callable[[Hook], object]) -> CacheContract:
    class Broken(CacheContract):
        def make_cache(self, on_call=None):
            return factory(on_call)

    return Broken()


BROKEN = [
    (MissingAget, "test_protocol_shape"),
    (SyncAset, "test_protocol_shape"),
    (AsetWithoutTtl, "test_protocol_shape"),
    (AsetIgnoresTtl, "test_aset_with_ttl_expires_like_set"),
    (ExpiredStillReads, "test_ttl_expires"),
    (MissingAsEmpty, "test_each_sync_method_equals_its_async_pair"),
    (OnLoop, "test_async_methods_run_off_the_loop_thread"),
    (OnLoop, "test_loop_keeps_running_while_an_async_method_waits"),
    (LostUpdates, "test_concurrent_threads_lose_no_key"),
    (LostUpdates, "test_concurrent_aset_over_gather_loses_no_key"),
    (PartialWrite, "test_cancelled_aset_leaves_the_cache_valid"),
]


@pytest.mark.parametrize(("backend", "check"), BROKEN, ids=[f"{b.__name__}-{c}" for b, c in BROKEN])
def test_broken_backend_fails_the_matching_check(backend, check):
    """Each stand-in breaks one rule, and the suite must notice."""
    with pytest.raises(AssertionError):
        run_check(_contract_of(backend), check)


@pytest.mark.parametrize("backend", [HookedInMemoryCache, DictBackend, NativeAsyncCache])
def test_good_backend_passes_every_check_through_run_check(backend):
    """``run_check`` runs sync and async checks alike."""
    contract = _contract_of(backend)
    run_check(contract, "test_protocol_shape")
    run_check(contract, "test_async_methods_work")
