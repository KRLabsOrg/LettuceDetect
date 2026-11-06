"""Tests for the typed errors and the result warning."""

import pytest
from pydantic import ValidationError

from lettucedetect.detectors import (
    ConfigurationError,
    InvalidInputError,
    LettuceDetectError,
    ResultWarning,
    UpstreamError,
)


def _make(cls):
    # Build an instance of any error class with the minimal valid arguments.
    if cls is InvalidInputError:
        return cls("bad value", param="answer")
    return cls("some message")


@pytest.mark.parametrize("cls", [InvalidInputError, ConfigurationError, UpstreamError])
def test_subclasses_are_caught_by_base(cls):
    # One `except LettuceDetectError` catches every subclass, and each carries a code and message.
    with pytest.raises(LettuceDetectError) as exc:
        raise _make(cls)
    assert exc.value.code
    assert exc.value.message == str(exc.value)


def test_value_error_only_for_invalid_input():
    # Only InvalidInputError is also a ValueError; deployment and upstream failures are not.
    assert issubclass(InvalidInputError, ValueError)
    assert not issubclass(ConfigurationError, ValueError)
    assert not issubclass(UpstreamError, ValueError)


def test_param_attributes():
    # `code` and `param` are readable attributes; `param` is optional (default None)
    # on both InvalidInputError and ConfigurationError.
    err = InvalidInputError("min_score out of range", param="min_score", code="invalid_min_score")
    assert (err.code, err.param) == ("invalid_min_score", "min_score")
    assert ConfigurationError("m").param is None
    assert ConfigurationError("m", param="config.lang").param == "config.lang"
    assert InvalidInputError("m").param is None


@pytest.mark.parametrize("cls", [InvalidInputError, ConfigurationError, UpstreamError])
def test_cause_is_kept(cls):
    # Chaining with `raise ... from` keeps the original error as __cause__.
    original = KeyError("x")
    with pytest.raises(cls) as exc:
        try:
            raise original
        except KeyError as e:
            raise _make(cls) from e
    assert exc.value.__cause__ is original


def test_message_built_from_request_has_no_customer_text():
    # A message that refers to an item by position never contains the request text.
    answer_text = "The capital of Atlantis is Poseidonis."
    request = {"sources": ["a", "b"], "answer": answer_text}
    position = len(request["sources"]) - 1
    err = InvalidInputError(f"sources[{position}] is empty", param="request.sources")
    assert answer_text not in err.message
    assert answer_text not in str(err)


def test_result_warning():
    # The warning exposes a stable code, is frozen, and survives a JSON round trip unchanged.
    w = ResultWarning(code="input_truncated")
    assert w.code == "input_truncated"
    assert w.message is None
    with pytest.raises(ValidationError):
        w.code = "other"  # ty: ignore[invalid-assignment]
    again = ResultWarning.model_validate_json(w.model_dump_json())
    assert again == w


def test_default_and_custom_codes():
    # Each class has a default code; a raise site may override it for one instance.
    assert LettuceDetectError("m").code == "lettucedetect_error"
    assert InvalidInputError("m", param="p").code == "invalid_input"
    assert ConfigurationError("m").code == "configuration_error"
    assert UpstreamError("m").code == "upstream_error"
    assert UpstreamError("m", code="empty_response").code == "empty_response"
    # a custom code is per instance and leaves the class default alone
    assert UpstreamError.code == "upstream_error"


def test_upstream_fields():
    # UpstreamError defaults to an unknown, non-retryable service and takes retry hints.
    err = UpstreamError("m")
    assert (err.service, err.retryable, err.retry_after) == ("unknown", False, None)
    err = UpstreamError("m", service="llm", retryable=True, retry_after=2.5)
    assert (err.service, err.retryable, err.retry_after) == ("llm", True, 2.5)
