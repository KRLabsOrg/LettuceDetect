"""Result types shared by the detectors."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ResultWarning(BaseModel):
    """A note that a result is correct but degraded.

    :param code: stable snake_case identifier, for example ``input_truncated``.
    :param message: optional human-readable text without customer text.
    """

    model_config = ConfigDict(frozen=True)

    code: str
    message: str | None = None
