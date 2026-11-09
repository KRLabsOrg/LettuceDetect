"""Tests for ``make_cache_key``, ``InMemoryCache`` specifics and the public exports."""

from __future__ import annotations

import math

import pytest

import lettucedetect.detectors as detectors
from lettucedetect.detectors import (
    InMemoryCache,
    InvalidInputError,
    ResponseCache,
    SyncCacheBase,
    make_cache_key,
)
from lettucedetect.detectors.cache import CacheManager
from tests.golden import GOLDEN_DIR

KINDS = ("judge", "native", "verify")


def _recorded() -> dict[str, str]:
    import json

    return json.loads((GOLDEN_DIR / "cache_hash_strings.json").read_text(encoding="utf-8"))


class TestMakeCacheKey:
    """The key function reproduces today's keys and adds the model version."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_without_version_equals_the_recorded_key(self, kind):
        """``model_version=None`` gives the key recorded by the golden tests (#131)."""
        assert make_cache_key("prompt", "gpt-4.1-mini", None, 0.0, kind) == _recorded()[kind]

    def test_without_version_equals_the_inline_hash_of_the_detector(self):
        """Same parts, same hash as the inline ``CacheManager._hash`` calls in ``llm.py``."""
        assert make_cache_key("p", "m", None, 0.5, "judge") == CacheManager._hash("p", "m", "0.5")
        assert make_cache_key("p", "m", None, 0.5, "native") == CacheManager._hash(
            "p", "m", "0.5", "native"
        )
        assert make_cache_key("p", "m", None, 0.5, "verify") == CacheManager._hash(
            "p", "m", "0.5", "verify"
        )

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_version_changes_the_key(self, kind):
        """A version differs from no version, and two versions differ from each other."""
        none = make_cache_key("p", "m", None, 0.0, kind)
        v1 = make_cache_key("p", "m", "v1", 0.0, kind)
        v2 = make_cache_key("p", "m", "v2", 0.0, kind)
        assert len({none, v1, v2}) == 3

    def test_the_same_version_gives_the_same_key(self):
        """The same version gives the same key."""
        assert make_cache_key("p", "m", "v1", 0.0, "judge") == make_cache_key(
            "p", "m", "v1", 0.0, "judge"
        )

    def test_a_version_cannot_forge_a_kind(self):
        """A judge key with version ``"native"`` is not the native key without a version."""
        assert make_cache_key("p", "m", "native", 0.0, "judge") != make_cache_key(
            "p", "m", None, 0.0, "native"
        )

    def test_unknown_kind_raises_a_typed_error(self):
        """Unknown kind raises a typed error."""
        with pytest.raises(InvalidInputError) as info:
            make_cache_key("p", "m", None, 0.0, "other")  # type: ignore[arg-type]
        assert info.value.code == "invalid_cache_kind"
        assert info.value.param == "kind"


class TestInMemoryCacheTtl:
    """``ttl_seconds`` must be a positive, finite number."""

    @pytest.mark.parametrize("ttl", [0, -1, 0.0, math.nan, math.inf, -math.inf])
    def test_bad_ttl_raises_and_stores_nothing(self, ttl):
        """Bad ttl raises and stores nothing."""
        cache = InMemoryCache()
        with pytest.raises(InvalidInputError) as info:
            cache.set("k", "v", ttl_seconds=ttl)
        assert info.value.code == "invalid_ttl_seconds"
        assert info.value.param == "ttl_seconds"
        assert cache.get("k") is None

    @pytest.mark.parametrize("ttl", [True, "5"])
    def test_non_numbers_raise(self, ttl):
        """Non numbers raise."""
        with pytest.raises(InvalidInputError):
            InMemoryCache().set("k", "v", ttl_seconds=ttl)  # type: ignore[arg-type]

    async def test_aset_validates_like_set(self):
        """Aset validates like set."""
        cache = InMemoryCache()
        with pytest.raises(InvalidInputError) as info:
            await cache.aset("k", "v", ttl_seconds=0)
        assert info.value.code == "invalid_ttl_seconds"


class TestExports:
    """The protocol, key function and both cache classes are public."""

    def test_names_are_exported(self):
        """Names are exported."""
        for name in ("ResponseCache", "make_cache_key", "SyncCacheBase", "InMemoryCache"):
            assert name in detectors.__all__
            assert hasattr(detectors, name)

    def test_in_memory_cache_follows_the_protocol(self):
        """In memory cache follows the protocol."""
        assert isinstance(InMemoryCache(), ResponseCache)
        assert issubclass(InMemoryCache, SyncCacheBase)

    def test_the_protocol_does_not_force_the_base_class(self):
        """A class that has the six methods is a ``ResponseCache`` without ``SyncCacheBase``."""

        class Direct:
            def get(self, key): ...
            def set(self, key, value, *, ttl_seconds=None): ...
            def delete(self, key): ...
            async def aget(self, key): ...
            async def aset(self, key, value, *, ttl_seconds=None): ...
            async def adelete(self, key): ...

        assert isinstance(Direct(), ResponseCache)
        assert not issubclass(Direct, SyncCacheBase)
