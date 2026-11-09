"""The old file cache, kept for compatibility. Use :class:`~lettucedetect.detectors.FileCache`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lettucedetect.detectors._deprecation import deprecated
from lettucedetect.detectors.file_cache import FileCache, _is_ttl_entry
from lettucedetect.detectors.response_cache import _hash_parts


@deprecated("FileCache")
class CacheManager(FileCache):
    """Disk-backed cache for expensive LLM calls.

    .. deprecated:: 0.3.0
        Use :class:`~lettucedetect.detectors.FileCache`. ``CacheManager`` will be removed in 0.5.0.

    It is a ``FileCache`` that keeps its old behaviour: ``get`` returns any JSON value that was
    stored, not only strings. ``aget``, ``aset`` and ``adelete`` are the ones of ``FileCache``. The
    directory of the file is created by the first write, not by the constructor.
    """

    def __init__(self, file_path: str | Path) -> None:
        """Remember the path of the cache file.

        :param file_path: Path to the JSON cache file.
        """
        super().__init__(file_path)

    @staticmethod
    def _hash(*parts: str) -> str:
        return _hash_parts(*parts)

    def _decode(self, entry: object, now: float) -> Any:  # noqa: ANN401
        """Return any stored JSON value as it is; only TTL entries are unwrapped."""
        if _is_ttl_entry(entry):
            return super()._decode(entry, now)
        return entry
