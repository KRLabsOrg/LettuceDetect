"""Abstract base class for hallucination detectors."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from lettucedetect.detectors.errors import InvalidInputError
from lettucedetect.detectors.types import Detection, DetectRequest


class BaseDetector(ABC):
    """All hallucination detectors implement a common interface.

    :cvar supports_tokens: whether ``include_tokens=True`` is supported.
    :cvar supports_detail: whether ``include_detail=True`` is supported.
    :cvar model_id: id of the model behind the detector, ``None`` if there is none.
    """

    supports_tokens: bool = False
    supports_detail: bool = False
    model_id: str | None = None

    def detect(
        self,
        request: DetectRequest,
        *,
        min_score: float = 0.0,
        include_tokens: bool = False,
        include_detail: bool = False,
    ) -> Detection:
        """Detect hallucinated spans in the answer of a request.

        :param request: the sources (or prompt) and the answer to check.
        :param min_score: drop spans scored below this value, in ``[0, 1]``. Tokens are kept.
        :param include_tokens: also return token scores (needs ``supports_tokens``).
        :param include_detail: also return fact checker evidence (needs ``supports_detail``).
        :returns: the detection result.
        :raises InvalidInputError: if ``min_score`` is outside ``[0, 1]``, or a flag is set that
            the detector does not support.
        """
        if not 0.0 <= min_score <= 1.0:
            raise InvalidInputError(
                "min_score must be in [0, 1]; pass a value between 0.0 and 1.0.",
                code="invalid_min_score",
                param="min_score",
            )
        if include_tokens and not self.supports_tokens:
            raise InvalidInputError(
                "This detector does not return token scores; call with include_tokens=False.",
                code="tokens_not_supported",
                param="include_tokens",
            )
        if include_detail and not self.supports_detail:
            raise InvalidInputError(
                "This detector does not return detail; call with include_detail=False.",
                code="detail_not_supported",
                param="include_detail",
            )
        result = self._detect(request, include_tokens=include_tokens, include_detail=include_detail)
        if min_score > 0.0:
            result = result.model_copy(
                update={"spans": [s for s in result.spans if s.score >= min_score]}
            )
        return result

    async def adetect(
        self,
        request: DetectRequest,
        *,
        min_score: float = 0.0,
        include_tokens: bool = False,
        include_detail: bool = False,
    ) -> Detection:
        """Async version of :meth:`detect`; it runs ``detect`` in a worker thread.

        The event loop is never blocked. Validation, ``min_score`` filtering and the support
        checks are those of :meth:`detect`, and its errors are raised from the awaited call.

        Things to know:

        * It uses the event loop's default executor, which has ``min(32, cpu + 4)`` workers.
          A service with many concurrent calls should size it.
        * That executor is shared with the rest of the service (``loop.getaddrinfo`` uses it
          too), so a burst of long LLM calls can delay DNS lookups and other
          ``asyncio.to_thread`` users.
        * Cancelling ``adetect`` stops the wait, not the thread.

        There is no ``executor`` argument.

        :param request: the sources (or prompt) and the answer to check.
        :param min_score: drop spans scored below this value, in ``[0, 1]``.
        :param include_tokens: also return token scores.
        :param include_detail: also return fact checker evidence.
        :returns: the detection result, equal to what :meth:`detect` returns.
        """
        return await asyncio.to_thread(
            self.detect,
            request,
            min_score=min_score,
            include_tokens=include_tokens,
            include_detail=include_detail,
        )

    def _detect(
        self, request: DetectRequest, *, include_tokens: bool, include_detail: bool
    ) -> Detection:
        """Detect hallucinated spans; detectors implement this.

        Arguments are already validated, and ``min_score`` is applied by :meth:`detect`.

        :param request: the sources (or prompt) and the answer to check.
        :param include_tokens: whether to fill ``Detection.tokens``.
        :param include_detail: whether to fill ``Detection.detail``.
        :returns: the detection result.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement _detect")

    @abstractmethod
    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
        min_confidence: float = 0.0,
    ) -> list:
        """Predict hallucination tokens or spans given passages and an answer.

        :param context: List of passages that were supplied to the LLM / user.
        :param answer: Model-generated answer to inspect.
        :param question: Original question (``None`` for summarisation).
        :param output_format: ``"tokens"`` for token-level dicts, ``"spans"`` for character spans.
        :param min_confidence: Drop ``"spans"`` whose ``confidence`` is below this threshold
            (in ``[0, 1]``; ``0.0`` keeps every span). Ignored for ``"tokens"`` output.
        :returns: List of predictions in requested format.
        """
        pass

    @abstractmethod
    def predict_prompt(
        self, prompt: str, answer: str, output_format: str = "tokens", min_confidence: float = 0.0
    ) -> list:
        """Predict hallucinations from a pre-built prompt string.

        :param prompt: Full prompt (context + question already concatenated).
        :param answer: Model-generated answer to inspect.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :param min_confidence: Drop ``"spans"`` below this confidence threshold (``[0, 1]``).
        :returns: List of predictions in requested format.
        """
        pass

    @abstractmethod
    def predict_prompt_batch(
        self,
        prompts: list[str],
        answers: list[str],
        output_format: str = "tokens",
        min_confidence: float = 0.0,
    ) -> list:
        """Batch version of :meth:`predict_prompt`.

        :param prompts: List of full prompt strings.
        :param answers: List of answers to inspect.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :param min_confidence: Drop ``"spans"`` below this confidence threshold (``[0, 1]``).
        :returns: List of prediction lists, one per input pair.
        """
        pass

    @staticmethod
    def _validate_min_confidence(min_confidence: float) -> None:
        """Validate that ``min_confidence`` is a probability in ``[0, 1]``.

        :param min_confidence: The threshold to validate.
        :raises ValueError: If ``min_confidence`` is outside ``[0, 1]``.
        """
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")

    @staticmethod
    def _filter_spans_by_confidence(
        result: list, output_format: str, min_confidence: float
    ) -> list:
        """Drop spans scored below ``min_confidence`` from a ``"spans"`` result.

        Only ``"spans"`` output is filtered, and only when ``min_confidence > 0``;
        token-level output is returned unchanged so callers can still inspect
        low-confidence tokens. Spans without a ``confidence`` key are kept, mirroring
        the existing behaviour of the LLM detector's ``_to_spans``.

        :param result: The prediction list returned by a detector.
        :param output_format: The format the result was produced in.
        :param min_confidence: Minimum span confidence to keep (``[0, 1]``).
        :returns: The (possibly filtered) result list.
        """
        if output_format != "spans" or min_confidence <= 0.0:
            return result
        return [
            span
            for span in result
            if span.get("confidence") is None or span["confidence"] >= min_confidence
        ]
