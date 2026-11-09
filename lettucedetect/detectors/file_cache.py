"""A file-backed ``ResponseCache`` with atomic writes and a time to live.

New error code (public API, do not rename):

- ``cache_file_invalid`` (:class:`~lettucedetect.detectors.errors.ConfigurationError`): the cache file
  exists but is not a JSON object. Every operation raises it and the file is never overwritten.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from lettucedetect.detectors.errors import ConfigurationError
from lettucedetect.detectors.response_cache import SyncCacheBase, _validate_ttl

logger = logging.getLogger(__name__)

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    """Return the one lock of a file path, shared by every ``FileCache`` on that path."""
    key = os.path.abspath(path)  # noqa: PTH100 - no disk access, unlike Path.resolve()
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def _is_ttl_entry(entry: object) -> bool:
    """Whether ``entry`` is the ``{"value": str, "expires_at": number}`` form of a TTL entry."""
    return (
        isinstance(entry, dict)
        and set(entry) == {"value", "expires_at"}
        and isinstance(entry["value"], str)
        and isinstance(entry["expires_at"], (int, float))
        and not isinstance(entry["expires_at"], bool)
    )


def _is_expired(entry: object, now: float) -> bool:
    return _is_ttl_entry(entry) and now >= entry["expires_at"]  # type: ignore[index]


class FileCache(SyncCacheBase):
    """A ``ResponseCache`` stored in one JSON file, for **a single process**.

    The file is a JSON object. A value without a TTL is stored as ``key: "value"``, the format of
    ``CacheManager``, so a file written by it reads back here. A value with a TTL is stored as
    ``key: {"value": "...", "expires_at": <unix time>}`` and reads as ``None`` once expired.
    Expired entries are removed whenever the file is rewritten.

    - **Single process.** The lock covers the threads of one process, including several
      ``FileCache`` objects on the same path. Two processes writing the same file can lose each
      other's updates. The atomic rename keeps the file valid, it does not merge writers.
    - **Atomic writes.** Each change goes to a temporary file in the same directory, which is then
      moved over the file with ``os.replace``. A crash or a failed write leaves the previous file
      in place and no temporary file behind. There is no ``fsync``, so a power loss can still lose
      the latest write.
    - **No disk access until needed.** The constructor reads and creates nothing. The directory is
      created by the first write. The file is read on every call, so nothing goes stale.
    - **Errors.** An invalid file raises ``ConfigurationError`` (``cache_file_invalid``). An error
      from the file system (a failed write) is raised as it is, also out of ``aset``.
    - **Async.** ``aget``, ``aset`` and ``adelete`` run the file work, the lock and the rename
      included, in a worker thread. Cancelling the awaiting task stops the wait, not the thread:
      the write still finishes, and the file holds the old or the new content, never a part.
    """

    def __init__(self, path: str | Path) -> None:
        """Remember the path. Touches no disk.

        :param path: The JSON file. Its directory is created on the first write.
        """
        self.path = Path(path)
        self._lock = _lock_for(self.path)

    def get(self, key: str) -> str | None:
        """Return the value stored under ``key``, or ``None`` if it is missing or expired."""
        with self._lock:
            value = self._decode(self._load().get(key), time.time())
        logger.debug("FileCache get on %s: %s", self.path, "hit" if value is not None else "miss")
        return value

    def _decode(self, entry: object, now: float) -> Any:  # noqa: ANN401
        """Return the string an entry holds, or ``None`` for a miss.

        Entries of any other shape (such as the dict replies an old ``CacheManager`` file may
        hold) are a miss: a ``ResponseCache`` holds strings only.
        """
        if isinstance(entry, str):
            return entry
        if _is_ttl_entry(entry):
            if not _is_expired(entry, now):
                return entry["value"]  # type: ignore[index]  # ty: ignore[not-subscriptable]
            logger.debug("FileCache entry on %s has expired", self.path)
        elif entry is not None:
            logger.debug("FileCache entry on %s is not a string, read as a miss", self.path)
        return None

    def set(self, key: str, value: str, *, ttl_seconds: float | None = None) -> None:
        """Store ``value`` under ``key`` and write the file atomically.

        :raises InvalidInputError: ``ttl_seconds`` is not a positive, finite number
            (``invalid_ttl_seconds``).
        :raises ConfigurationError: The file is invalid (``cache_file_invalid``).
        """
        _validate_ttl(ttl_seconds)
        with self._lock:
            now = time.time()
            loaded = self._load()
            data = self._purge(loaded, now)
            if len(data) != len(loaded):
                logger.debug(
                    "FileCache set on %s: dropped %d expired entries",
                    self.path,
                    len(loaded) - len(data),
                )
            if ttl_seconds is None:
                data[key] = value
            else:
                data[key] = {"value": value, "expires_at": now + ttl_seconds}
            self._write(data)
        logger.debug("FileCache set on %s: stored an entry, ttl_seconds=%s", self.path, ttl_seconds)

    def delete(self, key: str) -> None:
        """Remove ``key``. Does nothing, and writes nothing, if the key is missing."""
        with self._lock:
            data = self._load()
            kept = self._purge(data, time.time())
            existed = key in kept
            kept.pop(key, None)
            if kept != data:
                self._write(kept)
        logger.debug(
            "FileCache delete on %s: %s", self.path, "removed" if existed else "key not found"
        )

    @staticmethod
    def _purge(data: dict[str, Any], now: float) -> dict[str, Any]:
        return {k: v for k, v in data.items() if not _is_expired(v, now)}

    def _load(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise self._invalid() from exc
        if not isinstance(data, dict):
            raise self._invalid()
        return data

    def _invalid(self) -> ConfigurationError:
        return ConfigurationError(
            f"The cache file {self.path} is not a JSON object. Repair or delete it; "
            "FileCache never overwrites it.",
            code="cache_file_invalid",
        )

    def _write(self, data: dict[str, Any]) -> None:
        """Write ``data`` to a temporary file next to the cache file, then rename it over it."""
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(data, ensure_ascii=False))
            os.replace(tmp, self.path)  # noqa: PTH105 - the tests patch os.replace
            logger.debug("FileCache wrote %d entries to %s", len(data), self.path)
        except BaseException:
            # Clean up after any failure, even a KeyboardInterrupt, then let it propagate unchanged.
            Path(tmp).unlink(missing_ok=True)
            raise
