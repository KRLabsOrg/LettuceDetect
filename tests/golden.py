"""Golden-file helper for characterization tests.

``assert_golden(name, data)`` compares ``data`` with ``tests/fixtures/golden/<name>.json`` and
fails on any difference, printing a unified diff. Run with ``UPDATE_GOLDEN=1`` to rewrite the
file instead (review the resulting git diff: a changed fixture is an intended behaviour change).

Floats are rounded to 6 places on both sides before comparing, so cross-platform softmax noise
does not flake the suite while any real change still fails.
"""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
from typing import Any

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"
_FLOAT_PLACES = 6


def _normalize(value: Any) -> Any:  # noqa: ANN401
    """Round floats and turn tuples into lists, recursively."""
    if isinstance(value, float):
        return round(value, _FLOAT_PLACES)
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def _dump(data: Any) -> str:  # noqa: ANN401
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def assert_golden(name: str, data: Any) -> None:  # noqa: ANN401
    """Assert that ``data`` equals the checked-in golden file ``<name>.json``.

    :param name: Fixture name, without the ``.json`` suffix.
    :param data: JSON-serializable value to compare.
    """
    path = GOLDEN_DIR / f"{name}.json"
    actual = _dump(_normalize(json.loads(json.dumps(data, ensure_ascii=False))))

    if os.environ.get("UPDATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        return

    if not path.exists():
        raise AssertionError(
            f"Golden file {path} does not exist. Record it with UPDATE_GOLDEN=1 and review it."
        )
    expected = _dump(_normalize(json.loads(path.read_text(encoding="utf-8"))))
    if actual != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=f"golden/{name}.json (checked in)",
                tofile=f"golden/{name}.json (actual)",
            )
        )
        raise AssertionError(
            f"Output differs from golden file {name}.json "
            f"(UPDATE_GOLDEN=1 rewrites it if the change is intended):\n{diff}"
        )
