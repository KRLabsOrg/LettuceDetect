"""Tests for the deprecation helper, using stand-ins defined here."""

from __future__ import annotations

import subprocess
import sys
import warnings

import pytest

from lettucedetect.detectors import LettuceDetectDeprecationWarning
from lettucedetect.detectors import _deprecation as dep
from lettucedetect.detectors._deprecation import _UNSET, deprecated, warn_if_passed


def expected(name: str, replacement: str) -> str:
    """Build the exact warning message from the module constants."""
    return (
        f"{name} is deprecated since {dep.DEPRECATED_SINCE} and will be removed in "
        f"{dep.REMOVED_IN}. Use {replacement}."
    )


@deprecated("new_func")
def old_func(x: int = 1) -> int:
    """Do a thing."""
    return x


@deprecated("NewClass")
class OldClass:
    """A class."""

    def __init__(self, v: int = 0) -> None:  # noqa: D107
        self.v = v


class Holder:
    """Stand-in."""

    @deprecated("Holder.new")
    def old(self) -> str:
        """Old."""
        return "ok"


def with_arg(a: int, legacy: object = _UNSET) -> int:
    """Stand-in function with a deprecated ``legacy`` argument."""
    warn_if_passed(legacy, "legacy", "modern")
    return a


def test_is_deprecation_warning_subclass():
    """The warning subclasses DeprecationWarning, and filters on either do not affect the other."""
    assert issubclass(LettuceDetectDeprecationWarning, DeprecationWarning)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=LettuceDetectDeprecationWarning)
        old_func()  # ty: ignore[deprecated]
        warnings.warn("other", DeprecationWarning)
    assert [str(w.message) for w in caught] == ["other"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message="other")
        old_func()  # ty: ignore[deprecated]
        warnings.warn("other", DeprecationWarning)
    assert len(caught) == 1 and caught[0].category is LettuceDetectDeprecationWarning


def test_function_warns_with_exact_message_and_caller_line():
    """A deprecated function warns with the exact message, attributed to the caller's line."""
    with pytest.warns(LettuceDetectDeprecationWarning) as rec:
        line = sys._getframe().f_lineno + 1
        assert old_func(3) == 3  # ty: ignore[deprecated]
    assert str(rec[0].message) == expected("old_func", "new_func")
    assert rec[0].filename == __file__ and rec[0].lineno == line


def test_class_and_method_warn():
    """Instantiating a deprecated class and calling a deprecated method both warn."""
    with pytest.warns(LettuceDetectDeprecationWarning) as rec:
        assert OldClass(2).v == 2  # ty: ignore[deprecated]
        assert Holder().old() == "ok"  # ty: ignore[deprecated]
    assert str(rec[0].message) == expected("OldClass", "NewClass")
    assert str(rec[1].message) == expected("Holder.old", "Holder.new")
    assert all(r.filename == __file__ for r in rec)


def test_warns_once_per_call_site_under_default_filter():
    """Under the default filter, repeated calls from one call site warn only once."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(3):
            old_func()  # ty: ignore[deprecated]
    assert len(caught) == 1


def test_metadata():
    """Deprecated names carry the PEP 702 ``__deprecated__`` message and keep their name."""
    assert getattr(old_func, "__deprecated__") == expected("old_func", "new_func")  # ty: ignore[deprecated]
    assert getattr(OldClass, "__deprecated__") == expected("OldClass", "NewClass")  # ty: ignore[deprecated]
    assert old_func.__name__ == "old_func"  # ty: ignore[deprecated]


def test_argument_warns_only_when_passed():
    """A deprecated argument warns only when the caller passes it."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert with_arg(1) == 1
    with pytest.warns(LettuceDetectDeprecationWarning) as rec:
        line = sys._getframe().f_lineno + 1
        with_arg(1, legacy=None)
    assert str(rec[0].message) == expected("The 'legacy' argument", "modern")
    assert rec[0].filename == __file__ and rec[0].lineno == line


def test_internal_caller_is_silent_but_external_warns():
    """Calls from a ``lettucedetect.*`` module stay silent; the same calls elsewhere warn."""
    namespace = {
        "__name__": "lettucedetect.fake_internal",
        "f": old_func,  # ty: ignore[deprecated]
        "g": with_arg,
        "C": OldClass,  # ty: ignore[deprecated]
    }
    code = "f()\ng(1, legacy=1)\nC(1)"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exec(code, namespace)  # noqa: S102
    with pytest.warns(LettuceDetectDeprecationWarning):
        exec(code, {**namespace, "__name__": "somewhere_else"})  # noqa: S102


def test_importing_detectors_emits_no_warning():
    """Importing ``lettucedetect.detectors`` in a fresh interpreter emits no deprecation warning."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import warnings\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    import lettucedetect.detectors as d\n"
            "bad=[x for x in w if issubclass(x.category, d.LettuceDetectDeprecationWarning)]\n"
            "raise SystemExit(len(bad))",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
