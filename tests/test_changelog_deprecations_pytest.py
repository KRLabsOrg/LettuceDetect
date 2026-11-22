"""Every name in the changelog's "Deprecated" section has the note in its docstring."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import lettucedetect.detectors as detectors
from lettucedetect.detectors import BaseDetector, LLMDetector
from lettucedetect.detectors import _deprecation as dep
from lettucedetect.detectors import cache as cache_module
from lettucedetect.detectors import factory as factory_module

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"

# "- <names>: use <replacements>. Removed in <version>." and optional further sentences.
ENTRY = re.compile(
    r"^- (?P<names>.+?): use (?P<replacements>.+?)\. Removed in (?P<removed>\d+(?:\.\d+)+)\."
)
NAME = re.compile(r"`([^`]+)`")

PREDICT_METHODS = ("predict", "predict_prompt", "predict_prompt_batch")
PREDICT_ARGUMENTS = ("min_confidence", "output_format")
LLM_DETECTOR_ARGUMENTS = ("provider", "client_kwargs", "cache_file", "client")


def _deprecated_section() -> list[str]:
    """Return the lines of the "Deprecated" section of the release that started the deprecations."""
    text = CHANGELOG.read_text(encoding="utf-8")
    release = text.split(f"\n## [{dep.DEPRECATED_SINCE}]", 1)[1].split("\n## [", 1)[0]
    return release.split("\n### Deprecated\n", 1)[1].split("\n### ", 1)[0].splitlines()


def _entries() -> list[tuple[str, list[str], str]]:
    """Return ``(name, replacements, removal version)`` for every name the section lists."""
    entries = []
    for line in _deprecated_section():
        if not line.startswith("- "):
            continue
        match = ENTRY.match(line)
        assert match, f"Deprecated entry is not '- <names>: use <x>. Removed in <v>.': {line}"
        replacements = NAME.findall(match["replacements"])
        assert replacements, f"Deprecated entry names no replacement: {line}"
        for name in NAME.findall(match["names"]):
            entries.append((name, replacements, match["removed"]))
    return entries


def _predict_owners(methods: tuple[str, ...]) -> list[object]:
    """Return the given ``predict*`` methods of ``BaseDetector`` and of every public detector.

    A deprecated class is left out: its own note covers its methods.
    """
    classes = [BaseDetector] + [
        cls
        for cls in (getattr(detectors, name) for name in detectors.__all__)
        if inspect.isclass(cls)
        and issubclass(cls, BaseDetector)
        and cls is not BaseDetector
        and not hasattr(cls, "__deprecated__")
    ]
    return [vars(cls)[method] for cls in classes for method in methods if method in vars(cls)]


def _owners(name: str) -> list[object]:
    """Resolve a name from the changelog to the objects whose docstring must carry the note."""
    if name in PREDICT_METHODS:
        return _predict_owners((name,))
    if name in PREDICT_ARGUMENTS:
        owners = _predict_owners(PREDICT_METHODS)
        assert all(name in inspect.signature(owner).parameters for owner in owners)
        return owners
    if name in LLM_DETECTOR_ARGUMENTS:
        assert name in inspect.signature(LLMDetector.__init__).parameters
        return [LLMDetector]
    for module in (detectors, factory_module, cache_module):
        if hasattr(module, name):
            return [getattr(module, name)]
    raise AssertionError(f"The changelog lists {name!r} as deprecated, but it can't be resolved.")


def _note(obj: object) -> str:
    """Return the ``.. deprecated::`` paragraph of a docstring, or fail if there is none."""
    doc = inspect.cleandoc(obj.__doc__ or "")
    marker = f".. deprecated:: {dep.DEPRECATED_SINCE}"
    label = getattr(obj, "__qualname__", repr(obj))
    assert marker in doc, f"{label} has no '{marker}' note in its docstring"
    return " ".join(doc.split(marker, 1)[1].split("\n\n", 1)[0].split())


def _check(name: str, owner: object, replacements: list[str], removed: str) -> None:
    """Assert that ``owner`` documents the deprecation of ``name`` as the changelog states it."""
    note = _note(owner)
    label = getattr(owner, "__qualname__", repr(owner))
    for replacement in replacements:
        assert replacement in note, f"{label}: the note does not name {replacement}"
    assert removed in note, f"{label}: the note does not name the removal version {removed}"
    if name in PREDICT_ARGUMENTS + LLM_DETECTOR_ARGUMENTS:
        assert name in note, f"{label}: the note does not name the argument {name}"


def test_the_section_lists_every_deprecated_name():
    """The section covers the names the release deprecates, each removed in ``REMOVED_IN``."""
    entries = _entries()
    assert {name for name, _, _ in entries} == {
        "TransformerDetector",
        "LLMDetector",
        "make_detector",
        "CacheManager",
        *PREDICT_METHODS,
        *PREDICT_ARGUMENTS,
        *LLM_DETECTOR_ARGUMENTS,
    }
    assert {removed for _, _, removed in entries} == {dep.REMOVED_IN}


@pytest.mark.parametrize(("name", "replacements", "removed"), _entries())
def test_deprecated_name_has_the_note_in_its_docstring(name, replacements, removed):
    """The docstring has ``.. deprecated::`` with the replacement and the removal version."""
    owners = _owners(name)
    assert owners
    for owner in owners:
        _check(name, owner, replacements, removed)


def test_the_check_fails_when_a_name_loses_its_note():
    """A docstring without the note, or with another replacement or version, fails the check."""

    def no_note() -> None:
        """Do a thing."""

    def other_replacement() -> None:
        """Do a thing.

        .. deprecated:: 0.3.0
            Use ``something_else``. ``other_replacement`` will be removed in 0.5.0.
        """

    other_replacement.__doc__ = other_replacement.__doc__.replace("0.3.0", dep.DEPRECATED_SINCE)
    with pytest.raises(AssertionError, match="has no"):
        _check("no_note", no_note, ["new_name"], dep.REMOVED_IN)
    with pytest.raises(AssertionError, match="does not name new_name"):
        _check("other_replacement", other_replacement, ["new_name"], dep.REMOVED_IN)
    with pytest.raises(AssertionError, match="removal version"):
        _check("other_replacement", other_replacement, ["something_else"], "9.9.9")
