"""Thread‑safe JSON cache with SHA‑256 keys."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


class CacheManager:
    """Disk‑backed cache for expensive LLM calls."""

    def __init__(self, file_path: str | Path):
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = (
            json.loads(self.path.read_text("utf-8")) if self.path.exists() else {}
        )

    @staticmethod
    def _hash(*parts: str) -> str:
        return hashlib.sha256("||".join(parts).encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
