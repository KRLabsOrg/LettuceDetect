"""Tests for ``CacheManager``, now a deprecated subclass of ``FileCache``."""

from __future__ import annotations

import json
import subprocess
import sys
import warnings

import pytest

from lettucedetect.detectors import FileCache, LettuceDetectDeprecationWarning, make_cache_key
from lettucedetect.detectors._deprecation import DEPRECATED_SINCE, REMOVED_IN
from lettucedetect.detectors.cache import CacheManager
from lettucedetect.detectors.llm import LLMDetector
from lettucedetect.detectors.response_cache import SyncCacheBase
from tests.golden import GOLDEN_DIR
from tests.test_golden_llm_pytest import PLAIN, FakeClient

IGNORE = pytest.mark.filterwarnings(
    "ignore::lettucedetect.detectors._deprecation.LettuceDetectDeprecationWarning"
)


def _recorded() -> dict[str, str]:
    return json.loads((GOLDEN_DIR / "cache_hash_strings.json").read_text(encoding="utf-8"))


class TestDeprecation:
    """Constructing it from outside the package warns, and names ``FileCache``."""

    def test_is_a_file_cache_and_warns_naming_file_cache(self, tmp_path):
        """The warning has the helper's message and points at the line that built the cache."""
        with pytest.warns(LettuceDetectDeprecationWarning, match="FileCache") as rec:
            cache = CacheManager(tmp_path / "cache.json")  # ty: ignore[deprecated]
        assert isinstance(cache, FileCache)
        assert isinstance(cache, SyncCacheBase)
        assert len(rec) == 1
        assert str(rec[0].message) == (
            f"CacheManager is deprecated since {DEPRECATED_SINCE} and will be removed in "
            f"{REMOVED_IN}. Use FileCache."
        )
        assert rec[0].filename == __file__

    def test_docstring_has_the_deprecated_note(self):
        """The class docstring carries ``.. deprecated::`` and names the replacement."""
        assert ".. deprecated::" in (CacheManager.__doc__ or "")
        assert "FileCache" in (CacheManager.__doc__ or "")
        assert "FileCache" in getattr(CacheManager, "__deprecated__")

    def test_internal_caller_is_silent_but_external_warns(self, tmp_path):
        """A ``lettucedetect.*`` module builds it silently; the same code elsewhere warns."""
        namespace = {"CacheManager": CacheManager, "path": tmp_path / "c.json"}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exec("CacheManager(path)", {**namespace, "__name__": "lettucedetect.fake"})  # noqa: S102
        with pytest.warns(LettuceDetectDeprecationWarning):
            exec("CacheManager(path)", {**namespace, "__name__": "somewhere_else"})  # noqa: S102

    def test_llm_detector_builds_it_without_a_warning(self, tmp_path):
        """``LLMDetector`` keeps using ``CacheManager`` internally, silently."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            detector = LLMDetector(client=FakeClient(PLAIN), cache_file=str(tmp_path / "c.json"))
        assert isinstance(detector.cache, CacheManager)

    def test_importing_the_module_has_no_import_cycle_or_warning(self):
        """A fresh interpreter imports ``cache`` first, then ``response_cache``, with no warning."""
        code = (
            "import warnings; warnings.simplefilter('error')\n"
            "import lettucedetect.detectors.cache\n"
            "import lettucedetect.detectors.response_cache\n"
            "import lettucedetect.detectors\n"
        )
        result = subprocess.run(  # noqa: S603 - our own interpreter and a fixed script
            [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


@IGNORE
class TestAsyncMethods:
    """``aget``, ``aset`` and ``adelete`` are inherited and equal the sync methods."""

    def test_no_override(self):
        """The async methods are not redefined on ``CacheManager`` or ``FileCache``."""
        for name in ("aget", "aset", "adelete"):
            assert name not in vars(CacheManager)
            assert name not in vars(FileCache)
            assert name in vars(SyncCacheBase)

    async def test_async_equals_sync_and_adds_no_warning(self, tmp_path):
        """A value crosses the sync/async boundary both ways, with no warning after construction."""
        cache = CacheManager(tmp_path / "cache.json")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cache.set("sync", "from set")
            assert await cache.aget("sync") == cache.get("sync") == "from set"
            await cache.aset("async", "from aset")
            assert cache.get("async") == await cache.aget("async") == "from aset"
            await cache.aset("ttl", "v", ttl_seconds=60)
            assert cache.get("ttl") == "v"
            assert cache.delete("sync") is None
            assert await cache.adelete("async") is None
            assert cache.get("sync") is None
            assert await cache.aget("async") is None


@IGNORE
class TestBehavesAsBefore:
    """``_hash``, ``path``, ``get``, ``set`` and ``delete`` keep their behaviour."""

    @pytest.mark.parametrize("kind", ["judge", "native", "verify"])
    def test_hash_returns_the_recorded_strings(self, kind):
        """``_hash`` is the key function behind the recorded keys."""
        parts = ("prompt", "gpt-4.1-mini", str(0.0)) + (() if kind == "judge" else (kind,))
        assert CacheManager._hash(*parts) == _recorded()[kind]
        assert make_cache_key("prompt", "gpt-4.1-mini", None, 0.0, kind) == _recorded()[kind]

    def test_path_is_a_path(self, tmp_path):
        """``path`` is the ``Path`` of the file, for a string or a ``Path`` argument."""
        assert CacheManager(str(tmp_path / "c.json")).path == tmp_path / "c.json"
        assert CacheManager(tmp_path / "c.json").path == tmp_path / "c.json"

    @pytest.mark.parametrize(
        "value", ["text", "héllo ✓", {"x": 1, "y": [1, 2]}, [1, "a"], None, 3, ""]
    )
    def test_any_json_value_round_trips(self, tmp_path, value):
        """As before, ``set`` stores any JSON value and ``get`` returns it."""
        cache = CacheManager(tmp_path / "cache.json")
        cache.set("k", value)
        assert cache.get("k") == value
        assert CacheManager(tmp_path / "cache.json").get("k") == value

    def test_missing_key_is_none_and_delete_is_quiet(self, tmp_path):
        """A missing key reads ``None``; deleting it does nothing."""
        cache = CacheManager(tmp_path / "cache.json")
        assert cache.get("missing") is None
        cache.delete("missing")
        cache.set("k", "v")
        cache.set("k", "w")
        assert cache.get("k") == "w"
        cache.delete("k")
        assert cache.get("k") is None

    def test_file_bytes_are_unchanged(self, tmp_path):
        """The golden sequence still ends in the same file, byte for byte."""
        path = tmp_path / "nested" / "cache.json"
        cache = CacheManager(path)
        cache.set("a", {"x": 1, "y": [1, 2]})
        cache.set("b", "héllo ✓")
        cache.set("c", None)
        cache.delete("a")
        cache.delete("missing")
        cache.set("b", "overwritten")
        assert path.read_text(encoding="utf-8") == '{"b": "overwritten", "c": null}'

    def test_the_directory_appears_on_the_first_write_only(self, tmp_path):
        """The constructor no longer creates the directory (an intended change)."""
        path = tmp_path / "nested" / "cache.json"
        cache = CacheManager(path)
        assert not path.parent.exists()
        cache.set("k", "v")
        assert path.exists()

    def test_ttl_entries_still_expire_through_get(self, tmp_path):
        """A TTL entry written by ``FileCache`` reads as a value, and as ``None`` once expired."""
        path = tmp_path / "cache.json"
        path.write_text(
            json.dumps(
                {
                    "live": {"value": "x", "expires_at": 4102444800},
                    "dead": {"value": "y", "expires_at": 1},
                    "plain": {"value": 1, "other": 2},
                }
            ),
            encoding="utf-8",
        )
        cache = CacheManager(path)
        assert cache.get("live") == "x"
        assert cache.get("dead") is None
        assert cache.get("plain") == {"value": 1, "other": 2}
