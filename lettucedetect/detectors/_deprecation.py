"""One consistent deprecation mechanism for the public API.

Deprecated names warn with :class:`LettuceDetectDeprecationWarning`, carry a
``__deprecated__`` attribute (PEP 702).
Calls made from inside ``lettucedetect`` itself stay silent.

Type checkers (ty, pyright, editors) only mark a name as deprecated when its
decorator resolves to ``typing_extensions.deprecated``. So ``deprecated`` is that
class as far as they can tell (see the ``TYPE_CHECKING`` block below) and our own
implementation at runtime. Use it as one decorator; the argument is the
replacement, which checkers show as the deprecation message::

    @deprecated("HallucinationDetector")
    def make_detector(...): ...

Convention for whoever deprecates a name: also add a ``.. deprecated::`` note
(replacement and removal version) to its docstring.
"""

# ruff: noqa: ANN401
from __future__ import annotations

import functools
import inspect
import sys
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, LiteralString, TypeVar, cast

from typing_extensions import deprecated as _deprecated

DEPRECATED_SINCE = "0.3.0"
# The release that removes the deprecated API: the 0.3.x and 0.4.x releases warn first.
# Every warning reads both versions from here.
REMOVED_IN = "0.5.0"

_T = TypeVar("_T")


class _Unset:
    """Sentinel for a deprecated argument the caller did not pass."""

    def __repr__(self) -> str:
        return "<UNSET>"


_UNSET: Any = _Unset()


class LettuceDetectDeprecationWarning(DeprecationWarning):
    """Emitted when a deprecated LettuceDetect name or argument is used."""


def _message(name: str, replacement: str) -> str:
    return (
        f"{name} is deprecated since {DEPRECATED_SINCE} and will be removed in "
        f"{REMOVED_IN}. Use {replacement}."
    )


def _is_internal(module_name: str | None) -> bool:
    return module_name == "lettucedetect" or (module_name or "").startswith("lettucedetect.")


def _warn(message: str, frame_depth: int) -> None:
    """Warn unless the frame ``frame_depth`` levels above this function is internal.

    ``frame_depth=1`` means this function's caller; the warning is attributed to
    that same frame, so ``stacklevel`` equals ``frame_depth``.
    """
    caller_module = sys._getframe(frame_depth).f_globals.get("__name__")
    if _is_internal(caller_module):
        return
    warnings.warn(message, LettuceDetectDeprecationWarning, stacklevel=frame_depth + 1)


def _deprecated_with_convention(replacement: str) -> Callable[[_T], _T]:
    """Apply the standard ``deprecated`` decorator with our message and category.

    It adds the one thing the standard decorator lacks: calls from inside
    ``lettucedetect`` stay silent.
    """

    def decorate(target: _T) -> _T:
        obj: Any = target
        message = _message(getattr(obj, "__qualname__", repr(obj)), replacement)
        # stacklevel=2: one extra frame, our guard, sits between the caller and the warning.
        mark = _deprecated(
            cast(LiteralString, message), category=LettuceDetectDeprecationWarning, stacklevel=2
        )

        if inspect.isclass(obj):
            original_new = obj.__new__
            marked = mark(obj)
            warning_new = marked.__new__

            def new(cls: Any, *args: Any, **kwargs: Any) -> Any:
                if _is_internal(sys._getframe(1).f_globals.get("__name__")):
                    if original_new is object.__new__:
                        return original_new(cls)
                    return original_new(cls, *args, **kwargs)
                return warning_new(cls, *args, **kwargs)

            marked.__new__ = staticmethod(new)
            return cast(_T, marked)

        marked = mark(obj)

        @functools.wraps(obj)
        def guard(*args: Any, **kwargs: Any) -> Any:
            if _is_internal(sys._getframe(1).f_globals.get("__name__")):
                return obj(*args, **kwargs)
            return marked(*args, **kwargs)

        setattr(guard, "__deprecated__", message)
        return cast(_T, guard)

    return decorate


if TYPE_CHECKING:
    from typing_extensions import deprecated
else:
    deprecated = _deprecated_with_convention


def warn_if_passed(value: object, name: str, replacement: str) -> None:
    """Warn that argument ``name`` is deprecated if the caller passed it.

    Give the argument a default of ``_UNSET`` and call this from the function
    body; the warning is attributed to the line that called that function.
    """
    if value is _UNSET:
        return
    _warn(_message(f"The {name!r} argument", replacement), 3)
