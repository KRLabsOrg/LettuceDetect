"""HTTP client for a SystemOne server, so :class:`VelaDetector` can use a served Vela model.

A SystemOne server, such as the ``vela2_serve.py`` script in each Vela model repository, answers
``POST /v1/systemone`` with the reply a local model's ``system_one`` returns. :class:`SystemOneClient`
has that same ``system_one(state=..., questions=...)`` method, so the detector uses it in place of
a model. Every failure of the call is raised as a typed error from :mod:`.errors`.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from lettucedetect.detectors.errors import ConfigurationError, InvalidInputError, UpstreamError

logger = logging.getLogger(__name__)

__all__ = ["SystemOneClient"]

_PATH = "/v1/systemone"
_SERVICE = "vela"


def _retry_after(response: httpx.Response) -> float | None:
    """Return the ``Retry-After`` header in seconds, or ``None`` when it is absent or a date."""
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _rejected_field(response: httpx.Response) -> str | None:
    """Return the field a 422 reply names, the last item of ``detail[0].loc`` (``["body", "model"]``)."""
    try:
        return str(response.json()["detail"][0]["loc"][-1])
    except (ValueError, KeyError, IndexError, TypeError):
        return None


def _status_error(
    response: httpx.Response,
) -> ConfigurationError | InvalidInputError | UpstreamError:
    """Build the error for a reply whose status is not 2xx.

    The message names the status only: the server's body can echo the request, so it never
    goes into the error.
    """
    status = response.status_code
    if status in (401, 403):
        return ConfigurationError(
            f"The SystemOne server rejected the API key (HTTP {status}); check api_key.",
            code="invalid_api_key",
            param="api_key",
        )
    if status == 413:
        # The server's "too_long": the same error as a local model that cannot fit the input.
        return InvalidInputError(
            "The input does not fit the model's token window; pass shorter sources.",
            code="input_too_long",
            param="request",
        )
    if status == 422 and _rejected_field(response) == "model":
        return ConfigurationError(
            "The SystemOne server does not serve this model; check model.",
            code="model_not_served",
            param="model",
        )
    if status == 429:
        return UpstreamError(
            "The SystemOne server is rate limiting requests (HTTP 429); retry later.",
            code="rate_limited",
            service=_SERVICE,
            retryable=True,
            retry_after=_retry_after(response),
        )
    if status >= 500:
        return UpstreamError(
            f"The SystemOne server failed (HTTP {status}); retry later.",
            code="server_error",
            service=_SERVICE,
            retryable=True,
            retry_after=_retry_after(response),
        )
    return UpstreamError(
        f"The SystemOne server rejected the request (HTTP {status}).",
        code="request_rejected",
        service=_SERVICE,
    )


class SystemOneClient:
    """Call a SystemOne server with the ``system_one`` method of a local Vela model.

    The caller passes everything: the client reads no environment variable, including the proxy
    and certificate settings httpx would otherwise pick up. One client is safe to share between
    threads, so :class:`VelaDetector` does not serialize calls through it.

    :param base_url: the server's base URL, for example ``http://127.0.0.1:8001``.
    :param model: the model name the server serves, sent with every request.
    :param api_key: sent as ``Authorization: Bearer <api_key>``; ``None`` sends no header.
    :param timeout: seconds to wait for the server, per request.
    :param http_client: an ``httpx.Client`` to use instead of building one, for example with a
        custom transport. Its base URL, headers and timeout are the caller's to set, so
        ``base_url`` and ``timeout`` are not applied to it, and passing ``api_key`` with it raises
        :class:`ConfigurationError`: set the ``Authorization`` header on the client instead.
    """

    # Read by VelaDetector: calls through this client need no scoring lock.
    thread_safe = True

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Build the client; no request is sent."""
        if not base_url.strip():
            raise ConfigurationError(
                "base_url must not be blank.", code="invalid_config", param="base_url"
            )
        if not model.strip():
            raise ConfigurationError(
                "model must not be blank.", code="invalid_config", param="model"
            )
        if timeout <= 0:
            raise ConfigurationError(
                "timeout must be greater than 0.", code="invalid_config", param="timeout"
            )
        if http_client is not None and api_key is not None:
            # The key would not reach a client the caller built, and the server would answer
            # 401 with no hint why.
            raise ConfigurationError(
                "api_key is not applied to a given http_client; set the Authorization header "
                "on the http_client instead.",
                code="invalid_config",
                param="api_key",
            )
        self.model = model
        if http_client is None:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            http_client = httpx.Client(
                base_url=base_url, headers=headers, timeout=timeout, trust_env=False
            )
        self._http = http_client

    def system_one(self, *, state: dict[str, str], questions: dict[str, Any]) -> dict[str, Any]:
        """Send one SystemOne request and return the server's reply.

        :param state: the SystemOne state, for example ``request``, ``source`` and ``answer``.
        :param questions: the SystemOne questions by id.
        :returns: the reply object, in the shape a local model's ``system_one`` returns.
        :raises ConfigurationError: (``invalid_api_key``) the server rejected the key;
            (``model_not_served``) the server does not serve ``model``.
        :raises InvalidInputError: (``input_too_long``) the input does not fit the model's window.
        :raises UpstreamError: (``timeout``, ``connection_failed``, ``rate_limited``,
            ``server_error``, ``request_rejected``) the call failed; (``invalid_model_output``)
            the reply is not a JSON object.
        """
        body = {"state": state, "model": self.model, "questions": questions}
        try:
            response = self._http.post(_PATH, json=body)
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                "The SystemOne server did not answer in time; retry or raise timeout.",
                code="timeout",
                service=_SERVICE,
                retryable=True,
            ) from exc
        except httpx.TransportError as exc:
            raise UpstreamError(
                "Could not connect to the SystemOne server; check base_url and the network.",
                code="connection_failed",
                service=_SERVICE,
                retryable=True,
            ) from exc
        if not response.is_success:
            logger.debug("systemone request failed: status=%d", response.status_code)
            raise _status_error(response)
        try:
            reply = response.json()
        except ValueError as exc:
            raise UpstreamError(
                "The SystemOne server returned a reply that is not JSON.",
                code="invalid_model_output",
                service=_SERVICE,
            ) from exc
        if not isinstance(reply, dict):
            raise UpstreamError(
                "The SystemOne server returned a reply that is not a JSON object.",
                code="invalid_model_output",
                service=_SERVICE,
            )
        return reply

    def close(self) -> None:
        """Close the HTTP connections."""
        self._http.close()

    def __enter__(self) -> SystemOneClient:
        """Return the client, for use in a ``with`` block."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the HTTP connections at the end of a ``with`` block."""
        self.close()
