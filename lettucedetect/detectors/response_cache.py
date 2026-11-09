"""The ``ResponseCache`` protocol, the cache key function and an in-memory backend.

A service that wants to cache LLM replies implements :class:`ResponseCache`. Backends that only
have synchronous storage subclass :class:`SyncCacheBase` and get the async methods for free.
:class:`InMemoryCache` is the simplest backend.

New error codes (public API, do not rename):

- ``invalid_cache_kind``: ``kind`` of :func:`make_cache_key` is not ``"judge"``, ``"native"``
  or ``"verify"``.
- ``invalid_ttl_seconds``: ``ttl_seconds`` is not a positive, finite number.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import math
import threading
import time
from typing import Literal, Protocol, runtime_checkable

from lettucedetect.detectors.cache import CacheManager
from lettucedetect.detectors.errors import InvalidInputError

logger = logging.getLogger(__name__)

CacheKind = Literal["judge", "native", "verify"]
_KINDS = ("judge", "native", "verify")


@runtime_checkable
class ResponseCache(Protocol):
    """A key-value store for model replies.

    Keys are opaque strings and values are strings. A missing or expired key reads as ``None``.
    Deleting a missing key does nothing. Each ``a...`` method takes the same arguments as its
    synchronous pair and returns the same value.

    ``isinstance(obj, ResponseCache)`` only checks that the six names exist. The reusable suite in
    ``tests/cache_contract.py`` checks the rest, such as that the ``a...`` methods are coroutine
    functions.
    """

    def get(self, key: str) -> str | None:
        """Return the value stored under ``key``, or ``None`` if it is missing or expired."""
        ...

    def set(self, key: str, value: str, *, ttl_seconds: float | None = None) -> None:
        """Store ``value`` under ``key``; it expires after ``ttl_seconds`` (``None``: never)."""
        ...

    def delete(self, key: str) -> None:
        """Remove ``key``. Does nothing if the key is missing."""
        ...

    async def aget(self, key: str) -> str | None:
        """Async pair of :meth:`get`."""
        ...

    async def aset(self, key: str, value: str, *, ttl_seconds: float | None = None) -> None:
        """Async pair of :meth:`set`."""
        ...

    async def adelete(self, key: str) -> None:
        """Async pair of :meth:`delete`."""
        ...


def make_cache_key(
    prompt: str,
    model: str,
    model_version: str | None,
    temperature: float,
    kind: CacheKind,
) -> str:
    """Build the cache key of one model call.

    With ``model_version=None`` the key equals the key ``LLMDetector`` builds today, so existing
    cache files stay valid. With a version, a ``model_version=<version>`` part is added, so a
    redeployed model does not read the replies of the old one.

    :param prompt: The full text sent to the model.
    :param model: The model name.
    :param model_version: A version or deployment id of the model, or ``None``.
    :param temperature: The sampling temperature.
    :param kind: ``"judge"``, ``"native"`` or ``"verify"``.
    :returns: A SHA-256 hex digest.
    :raises InvalidInputError: ``kind`` is not one of the three values (``invalid_cache_kind``).
    """
    if kind not in _KINDS:
        raise InvalidInputError(
            f"kind must be one of {', '.join(_KINDS)}.",
            code="invalid_cache_kind",
            param="kind",
        )
    parts = [prompt, model, str(temperature)]
    if kind != "judge":  # today's judge key has no kind part
        parts.append(kind)
    if model_version is not None:
        # Prefixed, so a version can never read as a kind ("native") and collide with its key.
        parts.append(f"model_version={model_version}")
    return CacheManager._hash(*parts)


def _validate_ttl(ttl_seconds: float | None) -> None:
    """Raise ``invalid_ttl_seconds`` unless ``ttl_seconds`` is ``None`` or a positive finite number."""
    if ttl_seconds is None:
        return
    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, (int, float))
        or not math.isfinite(ttl_seconds)
        or ttl_seconds <= 0
    ):
        raise InvalidInputError(
            "ttl_seconds must be a positive, finite number, or None for no expiry.",
            code="invalid_ttl_seconds",
            param="ttl_seconds",
        )


class SyncCacheBase(abc.ABC):
    """Base class for backends that only have synchronous storage.

    Implement :meth:`get`, :meth:`set` and :meth:`delete`. The base supplies :meth:`aget`,
    :meth:`aset` and :meth:`adelete` by running them in a worker thread, so the event loop stays
    free while storage blocks.

    Cancelling an ``a...`` call stops the wait, not the thread: the storage call still finishes.
    So ``set`` and ``delete`` must be atomic, and a cancelled ``aset`` may still land.

    A backend does not have to use this class. Any class that implements :class:`ResponseCache`
    works.
    """

    @abc.abstractmethod
    def get(self, key: str) -> str | None:
        """Return the value stored under ``key``, or ``None`` if it is missing or expired."""

    @abc.abstractmethod
    def set(self, key: str, value: str, *, ttl_seconds: float | None = None) -> None:
        """Store ``value`` under ``key``; it expires after ``ttl_seconds`` (``None``: never)."""

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """Remove ``key``. Does nothing if the key is missing."""

    async def aget(self, key: str) -> str | None:
        """Run :meth:`get` in a worker thread."""
        return await asyncio.to_thread(self.get, key)

    async def aset(self, key: str, value: str, *, ttl_seconds: float | None = None) -> None:
        """Run :meth:`set` in a worker thread."""
        await asyncio.to_thread(self.set, key, value, ttl_seconds=ttl_seconds)

    async def adelete(self, key: str) -> None:
        """Run :meth:`delete` in a worker thread."""
        await asyncio.to_thread(self.delete, key)


class InMemoryCache(SyncCacheBase):
    """A thread-safe cache in process memory, with a time to live per entry.

    Entries are lost when the process ends. There is no size limit: expired entries are removed
    when they are read, and the cache grows with the number of distinct keys.
    """

    def __init__(self) -> None:
        """Create an empty cache."""
        self._lock = threading.Lock()
        self._data: dict[str, tuple[str, float | None]] = {}

    def get(self, key: str) -> str | None:
        """Return the value stored under ``key``, or ``None`` if it is missing or expired."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value, deadline = entry
            if deadline is not None and time.monotonic() >= deadline:
                del self._data[key]
                return None
            return value

    def set(self, key: str, value: str, *, ttl_seconds: float | None = None) -> None:
        """Store ``value`` under ``key``, replacing the value and the deadline of an older entry.

        :raises InvalidInputError: ``ttl_seconds`` is not a positive, finite number
            (``invalid_ttl_seconds``).
        """
        _validate_ttl(ttl_seconds)
        deadline = None if ttl_seconds is None else time.monotonic() + ttl_seconds
        with self._lock:
            self._data[key] = (value, deadline)

    def delete(self, key: str) -> None:
        """Remove ``key``. Does nothing if the key is missing."""
        with self._lock:
            self._data.pop(key, None)
