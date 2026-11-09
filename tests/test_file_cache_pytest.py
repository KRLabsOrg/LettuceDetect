"""Tests for ``FileCache``: the cache contract, atomic writes, CacheManager files and errors."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from unittest.mock import patch

import pytest

import lettucedetect.detectors as detectors
from lettucedetect.detectors import (
    ConfigurationError,
    FileCache,
    InvalidInputError,
    ResponseCache,
    SyncCacheBase,
)
from lettucedetect.detectors.cache import CacheManager
from tests.cache_contract import CacheContract

REPLACE = "lettucedetect.detectors.file_cache.os.replace"
_WAIT = 10.0


class HookedFileCache(FileCache):
    """``FileCache`` that calls the contract's ``on_call`` before each storage operation."""

    def __init__(self, path, on_call: Callable[[], None] | None = None) -> None:
        """Create the cache."""
        super().__init__(path)
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


class TestFileCacheContract(CacheContract):
    """``FileCache`` passes the whole cache contract suite, async methods included."""

    @pytest.fixture(autouse=True)
    def _directory(self, tmp_path):
        self.directory = tmp_path
        self._counter = itertools.count()

    def make_cache(self, on_call=None):
        """Return a ``FileCache`` on a fresh file."""
        return HookedFileCache(self.directory / f"cache{next(self._counter)}.json", on_call)


def _gated_replace(gate: threading.Event, entered: threading.Event):
    """Return an ``os.replace`` that waits on ``gate`` before it renames."""
    real = os.replace

    def replace(src, dst):
        entered.set()
        assert gate.wait(_WAIT), "the write was never released"
        real(src, dst)

    return replace


async def _until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + _WAIT
    while not condition():
        assert time.monotonic() < deadline, "the worker thread never finished"
        await asyncio.sleep(0.01)


class TestConstruction:
    """The constructor and the read methods touch no disk."""

    def test_constructor_touches_no_disk(self, tmp_path):
        """Building a cache on a path in a missing directory creates nothing."""
        FileCache(tmp_path / "a" / "b" / "cache.json")
        assert list(tmp_path.iterdir()) == []

    def test_get_and_delete_create_nothing(self, tmp_path):
        """Reading or deleting on a missing file leaves the disk as it was."""
        cache = FileCache(tmp_path / "a" / "cache.json")
        assert cache.get("k") is None
        cache.delete("k")
        assert list(tmp_path.iterdir()) == []

    def test_first_write_creates_the_directory_and_the_file(self, tmp_path):
        """The first ``set`` makes the directory, and only the cache file is left in it."""
        path = tmp_path / "a" / "b" / "cache.json"
        FileCache(path).set("k", "v")
        assert json.loads(path.read_text(encoding="utf-8")) == {"k": "v"}
        assert [p.name for p in path.parent.iterdir()] == ["cache.json"]

    def test_deleting_a_missing_key_writes_nothing(self, tmp_path):
        """``delete`` of a key that is not there leaves the file byte for byte as it was."""
        path = tmp_path / "cache.json"
        cache = FileCache(path)
        cache.set("k", "v")
        before = path.read_bytes()
        cache.delete("missing")
        assert path.read_bytes() == before


@pytest.mark.filterwarnings(
    "ignore::lettucedetect.detectors._deprecation.LettuceDetectDeprecationWarning"
)
class TestFileFormat:
    """The file is compatible with ``CacheManager`` and stores TTL entries as a pair."""

    def test_file_written_by_cache_manager_reads_back(self, tmp_path):
        """A ``CacheManager`` file with string values reads through ``get`` and ``aget``."""
        path = tmp_path / "cache.json"
        legacy = CacheManager(path)
        legacy.set("k", "héllo ✓")
        cache = FileCache(path)
        assert cache.get("k") == "héllo ✓"
        assert asyncio.run(cache.aget("k")) == "héllo ✓"

    def test_legacy_non_string_entries_are_a_miss_and_survive_writes(self, tmp_path):
        """Dict and null values (the LLM detector's replies) read as ``None`` and are kept."""
        path = tmp_path / "cache.json"
        legacy = CacheManager(path)
        legacy.set("reply", {"hallucination_list": []})
        legacy.set("nothing", None)
        cache = FileCache(path)
        assert cache.get("reply") is None
        assert cache.get("nothing") is None
        cache.set("k", "v")
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "reply": {"hallucination_list": []},
            "nothing": None,
            "k": "v",
        }

    def test_file_without_ttls_is_readable_by_cache_manager(self, tmp_path):
        """A value without a TTL is stored as ``key: "value"``."""
        path = tmp_path / "cache.json"
        FileCache(path).set("k", "v")
        assert CacheManager(path).get("k") == "v"

    def test_ttl_entry_is_value_and_expires_at(self, tmp_path):
        """A value with a TTL is stored as ``{"value": ..., "expires_at": <unix time>}``."""
        path = tmp_path / "cache.json"
        before = time.time()
        FileCache(path).set("k", "v", ttl_seconds=60)
        entry = json.loads(path.read_text(encoding="utf-8"))["k"]
        assert set(entry) == {"value", "expires_at"}
        assert entry["value"] == "v"
        assert before + 60 <= entry["expires_at"] <= time.time() + 60

    def test_expired_entry_reads_as_none(self, tmp_path):
        """An entry whose ``expires_at`` has passed is a miss; a later one is a hit."""
        path = tmp_path / "cache.json"
        path.write_text(
            json.dumps(
                {
                    "old": {"value": "x", "expires_at": time.time() - 1},
                    "new": {"value": "y", "expires_at": time.time() + 60},
                }
            ),
            encoding="utf-8",
        )
        cache = FileCache(path)
        assert cache.get("old") is None
        assert cache.get("new") == "y"

    def test_get_never_rewrites_the_file(self, tmp_path):
        """Reading an expired entry leaves the file as it was."""
        path = tmp_path / "cache.json"
        path.write_text(json.dumps({"old": {"value": "x", "expires_at": 1.0}}), encoding="utf-8")
        before = path.read_bytes()
        assert FileCache(path).get("old") is None
        assert path.read_bytes() == before

    def test_expired_entries_are_purged_on_write(self, tmp_path):
        """A write drops the entries that have expired."""
        path = tmp_path / "cache.json"
        path.write_text(json.dumps({"old": {"value": "x", "expires_at": 1.0}}), encoding="utf-8")
        FileCache(path).set("new", "y")
        assert json.loads(path.read_text(encoding="utf-8")) == {"new": "y"}

    def test_two_caches_on_one_path_share_the_data_and_the_lock(self, tmp_path):
        """A second object on the same path sees the first one's writes at once."""
        path = tmp_path / "cache.json"
        first, second = FileCache(path), FileCache(path)
        first.set("a", "1")
        second.set("b", "2")
        assert first.get("b") == "2"
        assert second.get("a") == "1"
        assert first._lock is second._lock


class TestInvalidTtl:
    """``ttl_seconds`` must be a positive, finite number."""

    @pytest.mark.parametrize("ttl", [0, -1, math.nan, math.inf, True, "5"])
    def test_bad_ttl_raises_and_writes_nothing(self, tmp_path, ttl):
        """The same error and code as ``InMemoryCache``, and no file is created."""
        cache = FileCache(tmp_path / "cache.json")
        with pytest.raises(InvalidInputError) as info:
            cache.set("k", "v", ttl_seconds=ttl)
        assert info.value.code == "invalid_ttl_seconds"
        assert info.value.param == "ttl_seconds"
        assert list(tmp_path.iterdir()) == []


class TestConcurrentWrites:
    """The file stays valid JSON and holds every key."""

    async def test_gather_of_aset_leaves_valid_json_with_every_key(self, tmp_path):
        """200 ``aset`` calls awaited together: the file parses and has all 200 keys."""
        path = tmp_path / "cache.json"
        cache = FileCache(path)
        await asyncio.gather(*(cache.aset(f"k{i}", f"v{i}") for i in range(200)))
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {f"k{i}": f"v{i}" for i in range(200)}
        assert [p.name for p in tmp_path.iterdir()] == ["cache.json"]

    def test_threads_writing_different_keys_lose_none(self, tmp_path):
        """Writers on separate ``FileCache`` objects of one path lose no key."""
        path = tmp_path / "cache.json"
        errors: list[BaseException] = []

        def write(worker: int) -> None:
            cache = FileCache(path)
            try:
                for i in range(20):
                    cache.set(f"k{worker}-{i}", f"v{worker}-{i}")
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {f"k{w}-{i}": f"v{w}-{i}" for w in range(8) for i in range(20)}


class TestFailedWrite:
    """A failed write leaves the previous file readable and raises the original error."""

    def test_set_raises_and_keeps_the_old_file(self, tmp_path):
        """``os.replace`` fails: ``set`` raises it, the file is byte for byte as it was."""
        path = tmp_path / "cache.json"
        cache = FileCache(path)
        cache.set("k", "old")
        before = path.read_bytes()
        with patch(REPLACE, side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                cache.set("k", "new")
        assert path.read_bytes() == before
        assert cache.get("k") == "old"
        assert [p.name for p in tmp_path.iterdir()] == ["cache.json"]

    async def test_aset_raises_the_same_error_from_the_await(self, tmp_path):
        """The error of the worker thread comes out of ``await aset`` unchanged."""
        path = tmp_path / "cache.json"
        cache = FileCache(path)
        cache.set("k", "old")
        before = path.read_bytes()
        error = OSError("disk full")
        with patch(REPLACE, side_effect=error):
            with pytest.raises(OSError) as info:
                await cache.aset("k", "new")
        assert info.value is error
        assert path.read_bytes() == before
        assert await cache.aget("k") == "old"
        assert [p.name for p in tmp_path.iterdir()] == ["cache.json"]
        await cache.aset("k", "later")
        assert await cache.aget("k") == "later"


class TestInterruptedWrite:
    """Cancelling ``aset``, or timing it out, while the write is in progress."""

    @pytest.mark.parametrize("how", ["cancel", "timeout"])
    async def test_file_stays_valid_and_no_temporary_file_is_left(self, tmp_path, how):
        """Before the thread ends the file is the old content; after, old or new, and clean."""
        path = tmp_path / "cache.json"
        cache = FileCache(path)
        cache.set("k", "old")
        gate, entered = threading.Event(), threading.Event()
        with patch(REPLACE, side_effect=_gated_replace(gate, entered)):
            write = cache.aset("k", "new")
            if how == "cancel":
                task = asyncio.create_task(write)
                assert await asyncio.to_thread(entered.wait, _WAIT), "the write never started"
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(write, timeout=0.1)
                assert entered.is_set()
            # The thread is still inside the write: the temporary file exists, the file is old.
            assert json.loads(path.read_text(encoding="utf-8")) == {"k": "old"}
            assert any(p.name.endswith(".tmp") for p in tmp_path.iterdir())
            gate.set()
            await _until(lambda: [p.name for p in tmp_path.iterdir()] == ["cache.json"])
            await _until(lambda: json.loads(path.read_text(encoding="utf-8")) == {"k": "new"})
        assert json.loads(path.read_text(encoding="utf-8")) in ({"k": "old"}, {"k": "new"})
        assert [p.name for p in tmp_path.iterdir()] == ["cache.json"]
        assert cache.get("k") == "new"
        cache.set("k", "later")
        assert await cache.aget("k") == "later"
        await cache.aset("other", "value")
        assert cache.get("other") == "value"


class TestInvalidFile:
    """A file that is not a JSON object is an error on every operation, and is never touched."""

    @pytest.mark.parametrize("content", ["{not json", "", "[1, 2]", '"text"', "null"])
    def test_every_operation_raises(self, tmp_path, content):
        """``get``, ``set``, ``delete`` and ``aget`` raise ``cache_file_invalid``."""
        path = tmp_path / "cache.json"
        path.write_text(content, encoding="utf-8")
        cache = FileCache(path)
        operations = [
            lambda: cache.get("k"),
            lambda: cache.set("k", "v"),
            lambda: cache.delete("k"),
            lambda: asyncio.run(cache.aget("k")),
        ]
        for operation in operations:
            with pytest.raises(ConfigurationError) as info:
                operation()
            assert info.value.code == "cache_file_invalid"
            assert str(path) in str(info.value)
        assert path.read_text(encoding="utf-8") == content
        assert [p.name for p in tmp_path.iterdir()] == ["cache.json"]

    def test_json_error_is_chained(self, tmp_path):
        """The error keeps the parser error as its cause."""
        path = tmp_path / "cache.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigurationError) as info:
            FileCache(path).get("k")
        assert isinstance(info.value.__cause__, ValueError)


class TestPublicApi:
    """The class is public and says it is for one process."""

    def test_exported_and_a_response_cache(self, tmp_path):
        """Importable from ``lettucedetect.detectors`` and follows the protocol."""
        assert "FileCache" in detectors.__all__
        cache = FileCache(tmp_path / "cache.json")
        assert isinstance(cache, ResponseCache)
        assert isinstance(cache, SyncCacheBase)

    def test_docstring_says_single_process(self):
        """The limit is documented."""
        assert "single process" in (FileCache.__doc__ or "").lower()

    def test_async_methods_come_from_the_base(self):
        """The async pairs are written once, in ``SyncCacheBase``."""
        for name in ("aget", "aset", "adelete"):
            assert name not in vars(FileCache)


CUSTOMER_KEY = "key-with-CUSTOMER-TEXT-7f3a"
CUSTOMER_VALUE = "reply-with-CUSTOMER-TEXT-9c1e"


class TestLogging:
    """Module logger only: DEBUG steps, no customer text, no printing, never log and raise."""

    def test_steps_are_logged_at_debug_without_keys_or_values(self, tmp_path, caplog, capsys):
        """Every operation leaves DEBUG records on the module logger; none carries a key or value."""
        cache = FileCache(tmp_path / "cache.json")
        with caplog.at_level(logging.DEBUG, logger="lettucedetect"):
            cache.set(CUSTOMER_KEY, CUSTOMER_VALUE)
            cache.set("short", CUSTOMER_VALUE, ttl_seconds=0.05)
            assert cache.get(CUSTOMER_KEY) == CUSTOMER_VALUE
            assert cache.get("missing") is None
            time.sleep(0.1)
            assert cache.get("short") is None
            cache.set("other", "v")  # drops the expired entry
            cache.delete(CUSTOMER_KEY)
            cache.delete(CUSTOMER_KEY)
        records = [r for r in caplog.records if r.name == "lettucedetect.detectors.file_cache"]
        text = "\n".join(r.getMessage() for r in records)
        assert records
        assert {r.levelno for r in records} == {logging.DEBUG}
        for secret in (CUSTOMER_KEY, CUSTOMER_VALUE, "CUSTOMER"):
            assert secret not in text
        for expected in ("get on", "miss", "hit", "expired", "removed", "key not found", "wrote"):
            assert expected in text, expected
        captured = capsys.readouterr()
        assert captured.out == captured.err == ""

    def test_a_non_string_entry_is_logged_as_a_miss(self, tmp_path, caplog):
        """A legacy dict value reads as a miss and says so at DEBUG, without its content."""
        path = tmp_path / "cache.json"
        path.write_text(
            json.dumps({"k": {"hallucination_list": [CUSTOMER_VALUE]}}), encoding="utf-8"
        )
        with caplog.at_level(logging.DEBUG, logger="lettucedetect"):
            assert FileCache(path).get("k") is None
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "not a string" in text
        assert CUSTOMER_VALUE not in text

    def test_errors_are_raised_not_logged(self, tmp_path, caplog):
        """A failed write and an invalid file raise; the cache adds no WARNING or ERROR record."""
        path = tmp_path / "cache.json"
        cache = FileCache(path)
        cache.set("k", "v")
        with caplog.at_level(logging.DEBUG, logger="lettucedetect"):
            with patch(REPLACE, side_effect=OSError("disk full")):
                with pytest.raises(OSError):
                    cache.set("k", "new")
            path.write_text("{not json", encoding="utf-8")
            with pytest.raises(ConfigurationError):
                cache.get("k")
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
