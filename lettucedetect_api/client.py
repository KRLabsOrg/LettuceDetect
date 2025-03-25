from typing import Type, TypeVar
from urllib.parse import urljoin

import httpx
from pydantic import BaseModel, ValidationError

from lettucedetect_api.models import DetectionRequest, SpanDetectionResponse, TokenDetectionResponse

T = TypeVar("T", bound=BaseModel)


class InvalidRequestError(Exception):
    """Raised for invalid requests by the client."""


class InvalidResponseError(Exception):
    """Raised for invalid responses by the server."""


class HTTPError(Exception):
    """Raised when errors happen during HTTP request/response."""


def _create_request_safe(contexts: list[str], question: str, answer: str) -> DetectionRequest:
    try:
        return DetectionRequest(contexts=contexts, question=question, answer=answer)
    except ValidationError as e:
        raise InvalidRequestError from e


def _httpx_request_wrapper(
    method: str,
    url: str,
    request: BaseModel,
    response_model: Type[T],
) -> T:
    try:
        response = httpx.request(method, url, json=dict(request))
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPError from e
    try:
        return response_model.model_validate_json(response.text)
    except ValidationError as e:
        raise InvalidResponseError from e


async def _httpx_request_wrapper_async(
    method: str,
    url: str,
    request: BaseModel,
    response_model: Type[T],
) -> T:
    async with httpx.AsyncClient() as client:
        response = await client.request(method, url, json=dict(request))
    response.raise_for_status()
    return response_model.model_validate_json(response.text)


class LettuceClientBase:
    """Base class class for lettucedetect clients."""

    _TOKEN_ENDPOINT = "/v1/lettucedetect/token"  # noqa: S105
    _SPANS_ENDPOINT = "/v1/lettucedetect/spans"

    def __init__(self, base_url: str):
        """Initialize lettucedetect client sub-classes.

        :param base_url: The full URL of the lettucedetect web server as a
        string. For a local server on port 8000 use "http://127.0.0.1:8000".
        """
        self.base_url = base_url


class LettuceClient(LettuceClientBase):
    """Synchronous client class for lettucedetect web API."""

    def detect_token(
        self, contexts: list[str], question: str, answer: str
    ) -> TokenDetectionResponse:
        """Token-level hallucination detection (synchronous version).

        Predicts hallucination scores for each token in `answer`. A higher score
        correlates to a higher probability that this token is hallucinated.

        :param contexts: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.

        :return: `TokenDetectionResponse` pydantic model instance which contains
        the detected tokens in the `predictions` attribute.

        :raises InvalidRequestError: Raised when the function is called with
        invalid arguments (e.g. wrong types).
        :raises HTTPError: Raised when something goes wrong during the HTTP
        request/response (e.g. server not reachable).
        :raises InvalidResponseError: Raised when the server response is
        invalid. This can happen if the client and server versions are not
        compatible.
        """
        request = _create_request_safe(contexts=contexts, question=question, answer=answer)
        url = urljoin(self.base_url, self._TOKEN_ENDPOINT)
        return _httpx_request_wrapper("post", url, request, TokenDetectionResponse)

    def detect_spans(
        self, contexts: list[str], question: str, answer: str
    ) -> SpanDetectionResponse:
        """Token-level hallucination detection (synchronous version).

        Predicts hallucination scores for each token in `answer`. A higher score
        correlates to a higher probability that this token is hallucinated.

        :param contexts: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.

        :return: `SpanDetectionResponse` pydantic model instance which contains
        the detected spans in the `predictions` attribute.

        :raises InvalidRequestError: Raised when the function is called with
        invalid arguments (e.g. wrong types).
        :raises HTTPError: Raised when something goes wrong during the HTTP
        request/response (e.g. server not reachable).
        :raises InvalidResponseError: Raised when the server response is
        invalid. This can happen if the client and server versions are not
        compatible.
        """
        request = _create_request_safe(contexts=contexts, question=question, answer=answer)
        url = urljoin(self.base_url, self._SPANS_ENDPOINT)
        return _httpx_request_wrapper("post", url, request, SpanDetectionResponse)


class LettuceClientAsync(LettuceClientBase):
    """Asynchronous client class for lettucedetect web API."""

    async def detect_token(
        self, contexts: list[str], question: str, answer: str
    ) -> TokenDetectionResponse:
        """Token-level hallucination detection (asynchronous version).

        Predicts hallucination scores for each token in `answer`. A higher score
        correlates to a higher probability that this token is hallucinated.

        :param contexts: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.

        :return: `TokenDetectionResponse` pydantic model instance which contains
        the detected tokens in the `predictions` attribute.

        :raises InvalidRequestError: Raised when the function is called with
        invalid arguments (e.g. wrong types).
        :raises HTTPError: Raised when something goes wrong during the HTTP
        request/response (e.g. server not reachable).
        :raises InvalidResponseError: Raised when the server response is
        invalid. This can happen if the client and server versions are not
        compatible.
        """
        request = _create_request_safe(contexts=contexts, question=question, answer=answer)
        url = urljoin(self.base_url, self._TOKEN_ENDPOINT)
        return await _httpx_request_wrapper_async("post", url, request, TokenDetectionResponse)

    async def detect_spans(
        self, contexts: list[str], question: str, answer: str
    ) -> SpanDetectionResponse:
        """Token-level hallucination detection (synchronous version).

        Predicts hallucination scores for each token in `answer`. A higher score
        correlates to a higher probability that this token is hallucinated.

        :param contexts: A list of context strings.
        :param answer: The answer string.
        :param question: The question string.

        :return: `SpanDetectionResponse` pydantic model instance which contains
        the detected spans in the `predictions` attribute.

        :raises InvalidRequestError: Raised when the function is called with
        invalid arguments (e.g. wrong types).
        :raises HTTPError: Raised when something goes wrong during the HTTP
        request/response (e.g. server not reachable).
        :raises InvalidResponseError: Raised when the server response is
        invalid. This can happen if the client and server versions are not
        compatible.
        """
        request = _create_request_safe(contexts=contexts, question=question, answer=answer)
        url = urljoin(self.base_url, self._SPANS_ENDPOINT)
        return await _httpx_request_wrapper_async("post", url, request, SpanDetectionResponse)
