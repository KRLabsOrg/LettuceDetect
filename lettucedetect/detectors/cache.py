"""Thread-safe JSON cache with SHA-256 keys."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


class CacheManager:
    """Disk-backed cache for expensive LLM calls."""

    def __init__(self, file_path: str | Path) -> None:
        """Initialize cache from disk.

        :param file_path: Path to the JSON cache file.
        """
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = (
            json.loads(self.path.read_text("utf-8")) if self.path.exists() else {}
        )

    @staticmethod
    def _hash(*parts: str) -> str:
        return hashlib.sha256("||".join(parts).encode()).hexdigest()

    def get(self, key: str) -> Any | None:  # noqa: ANN401
        """Retrieve a cached value by key."""
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Store a value in the cache and persist to disk."""
        with self._lock:
            self._data[key] = value
            self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")

    def delete(self, key: str) -> None:
        """Remove a key from the cache and persist to disk."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
