"""Typed errors raised by the detectors.

Every error carries a stable snake_case ``code`` so a caller can handle it
without parsing the message. Each class has a default code; a raise site may
pass a more specific one. Messages never hold secrets or customer text and
refer to items by position (for example ``sources[2]``). Chain foreign errors
with ``raise ... from``; ``__cause__`` is kept as Python sets it.
"""

from __future__ import annotations


class LettuceDetectError(Exception):
    """Base class of every error the library raises on purpose."""

    code = "lettucedetect_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        """Create the error.

        :param message: What the problem is and how to fix it, without secrets or customer text.
        :param code: A stable snake_case code, in place of the default code of the class.
        """
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class InvalidInputError(LettuceDetectError, ValueError):
    """An argument of a call is invalid.

    Also a :class:`ValueError`, so code that catches ``ValueError`` keeps working.
    """

    code = "invalid_input"

    def __init__(self, message: str, *, code: str | None = None, param: str | None = None) -> None:
        """Create the error.

        :param message: What the problem is and how to fix it.
        :param code: A stable snake_case code, in place of the default code of the class.
        :param param: Name of the offending argument.
        """
        super().__init__(message, code=code)
        self.param = param


class ConfigurationError(LettuceDetectError):
    """A constructor argument, a setting or a package is wrong or missing.

    Only the operator of the deployment can fix it, so it is not a ``ValueError``.
    """

    code = "configuration_error"

    def __init__(self, message: str, *, code: str | None = None, param: str | None = None) -> None:
        """Create the error.

        :param message: What the problem is and how to fix it.
        :param code: A stable snake_case code, in place of the default code of the class.
        :param param: Name of the offending argument or setting, if there is one.
        """
        super().__init__(message, code=code)
        self.param = param


class UpstreamError(LettuceDetectError):
    """A service the library calls failed or did not answer in time."""

    code = "upstream_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        service: str = "unknown",
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        """Create the error.

        :param message: What the problem is and how to fix it.
        :param code: A stable snake_case code, in place of the default code of the class.
        :param service: The service that failed, for example `llm`.
        :param retryable: Whether the same call may succeed when retried.
        :param retry_after: Seconds to wait before retrying, when the service said so.
        """
        super().__init__(message, code=code)
        self.service = service
        self.retryable = retryable
        self.retry_after = retry_after
