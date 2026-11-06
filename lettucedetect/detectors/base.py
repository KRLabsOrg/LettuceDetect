"""Abstract base class for hallucination detectors."""

from __future__ import annotations

import asyncio
import threading
import warnings
from abc import ABC

from lettucedetect.detectors._deprecation import LettuceDetectDeprecationWarning, _message
from lettucedetect.detectors.errors import InvalidInputError
from lettucedetect.detectors.types import (
    Chunk,
    DetectedSpan,
    Detection,
    DetectRequest,
    PromptInput,
    ResultWarning,
    SourcesInput,
    Usage,
)

_OUTPUT_FORMATS = ("tokens", "spans")

# Name of the class attribute that marks a legacy class as already warned about. It is read from
# the class's own ``__dict__``, so a subclass of a warned class warns once more on its own.
_WARNED_FLAG = "_legacy_warning_emitted"
_warned_lock = threading.Lock()


def _reset_legacy_warnings() -> None:
    """Forget which legacy classes were warned about, so each warns again. For tests."""
    pending: list[type[BaseDetector]] = [BaseDetector]
    while pending:
        cls = pending.pop()
        if _WARNED_FLAG in cls.__dict__:
            delattr(cls, _WARNED_FLAG)
        pending.extend(cls.__subclasses__())


class BaseDetector(ABC):
    """All hallucination detectors implement a common interface.

    A new detector implements ``_detect`` and gets ``detect``, ``adetect`` and the ``predict*``
    methods for free. A detector that implements only the legacy ``predict`` and
    ``predict_prompt`` still works through ``detect``, with one deprecation warning per class.
    A subclass that overrides none of ``detect``, ``_detect`` and ``predict`` cannot be defined.

    :cvar supports_tokens: whether ``include_tokens=True`` is supported.
    :cvar supports_detail: whether ``include_detail=True`` is supported.
    :cvar model_id: id of the model behind the detector, ``None`` if there is none.
    """

    supports_tokens: bool = False
    supports_detail: bool = False
    model_id: str | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Reject a subclass that implements no detection method.

        ``detect`` and ``predict`` call each other, so a class that overrides none of
        ``detect``, ``_detect`` and ``predict`` would recurse forever when called.

        :raises TypeError: if the class overrides none of ``detect``, ``_detect``, ``predict``.
        """
        super().__init_subclass__(**kwargs)
        if not any(
            getattr(cls, name) is not getattr(BaseDetector, name)
            for name in ("detect", "_detect", "predict")
        ):
            raise TypeError(
                f"{cls.__name__} must override _detect (or detect, or the legacy predict)."
            )

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
        The default serves a subclass that overrides only the legacy ``predict`` methods.

        :param request: the sources (or prompt) and the answer to check.
        :param include_tokens: whether to fill ``Detection.tokens``.
        :param include_detail: whether to fill ``Detection.detail``.
        :returns: the detection result.
        """
        if type(self).predict is BaseDetector.predict:
            raise NotImplementedError(f"{type(self).__name__} does not implement _detect")
        return self._detect_via_predict(request, include_tokens, include_detail)

    def _detect_via_predict(
        self, request: DetectRequest, include_tokens: bool, include_detail: bool
    ) -> Detection:
        """Answer ``detect`` with the legacy ``predict`` / ``predict_prompt`` of a subclass."""
        if include_tokens:
            raise InvalidInputError(
                "This detector does not return token scores; call with include_tokens=False.",
                code="tokens_not_supported",
                param="include_tokens",
            )
        if include_detail:
            raise InvalidInputError(
                "This detector does not return detail; call with include_detail=False.",
                code="detail_not_supported",
                param="include_detail",
            )
        cls = type(self)
        first_call = False
        if not cls.__dict__.get(_WARNED_FLAG):
            with _warned_lock:
                if not cls.__dict__.get(_WARNED_FLAG):
                    setattr(cls, _WARNED_FLAG, True)
                    first_call = True
        if first_call:
            warnings.warn(
                _message(f"{cls.__qualname__}.predict (without _detect)", "_detect"),
                LettuceDetectDeprecationWarning,
                stacklevel=4,
            )
        if isinstance(request, SourcesInput):
            raw = self.predict(
                [chunk.text for chunk in request.sources],
                request.answer,
                request.question,
                "spans",
                min_confidence=0.0,
            )
        elif type(self).predict_prompt is BaseDetector.predict_prompt:
            raise InvalidInputError(
                "This detector does not accept a prompt request; pass sources instead.",
                code="prompt_input_not_supported",
                param="request",
            )
        else:
            raw = self.predict_prompt(request.prompt, request.answer, "spans", min_confidence=0.0)
        spans = []
        synthetic = False
        for item in raw:
            confidence = item.get("confidence")
            if confidence is None:
                synthetic = True
                confidence = 1.0
            spans.append(
                DetectedSpan(
                    start=item["start"],
                    end=item["end"],
                    text=item["text"],
                    score=confidence,
                    category=item.get("category"),
                    subcategory=item.get("subcategory"),
                    reasoning=item.get("reasoning"),
                )
            )
        warnings_ = (
            [
                ResultWarning(
                    code="synthetic_score",
                    message="The detector gave no confidence; spans without one score 1.0.",
                )
            ]
            if synthetic
            else []
        )
        usage = [Usage(model_id=getattr(self, "model_id", None), model_version=None, units={})]
        return Detection(spans=spans, usage=usage, warnings=warnings_)

    @staticmethod
    def _check_legacy_arguments(output_format: str, min_confidence: float) -> None:
        """Validate the arguments the ``predict*`` methods share."""
        if output_format not in _OUTPUT_FORMATS:
            raise InvalidInputError(
                'output_format must be "tokens" or "spans".',
                code="unsupported_output_format",
                param="output_format",
            )
        if not 0.0 <= min_confidence <= 1.0:
            raise InvalidInputError(
                "min_confidence must be in [0, 1]; pass a value between 0.0 and 1.0.",
                code="invalid_min_score",
                param="min_confidence",
            )

    def _predict_via_detect(
        self, request: DetectRequest, output_format: str, min_confidence: float
    ) -> list:
        """Answer a ``predict*`` call with :meth:`detect`, in the legacy dict format."""
        self._check_legacy_arguments(output_format, min_confidence)
        detection = self.detect(
            request, min_score=min_confidence, include_tokens=output_format == "tokens"
        )
        if output_format == "tokens":
            assert detection.tokens is not None  # noqa: S101 - detect fills it when asked
            return [
                {"token": t.token, "pred": 1 if t.flagged else 0, "prob": t.score}
                for t in detection.tokens
            ]
        result = []
        for span in detection.spans:
            item = {
                "start": span.start,
                "end": span.end,
                "text": span.text,
                "confidence": span.score,
            }
            for key in ("category", "subcategory", "reasoning"):
                value = getattr(span, key)
                if value is not None:
                    item[key] = value
            result.append(item)
        return result

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
        min_confidence: float = 0.0,
    ) -> list:
        """Predict hallucination tokens or spans given passages and an answer.

        The default builds a request and calls :meth:`detect`; a legacy detector overrides it.

        :param context: List of passages that were supplied to the LLM / user.
        :param answer: Model-generated answer to inspect.
        :param question: Original question (``None`` for summarisation).
        :param output_format: ``"tokens"`` for token-level dicts, ``"spans"`` for character spans.
        :param min_confidence: Drop ``"spans"`` whose ``confidence`` is below this threshold
            (in ``[0, 1]``; ``0.0`` keeps every span).
        :returns: List of predictions in requested format.
        :raises InvalidInputError: for an unsupported ``output_format`` or a ``min_confidence``
            outside ``[0, 1]``.
        """
        request = SourcesInput(
            sources=[Chunk(id=str(i), text=text) for i, text in enumerate(context)],
            answer=answer,
            question=question,
        )
        return self._predict_via_detect(request, output_format, min_confidence)

    def predict_prompt(
        self, prompt: str, answer: str, output_format: str = "tokens", min_confidence: float = 0.0
    ) -> list:
        """Predict hallucinations from a pre-built prompt string.

        The default builds a request and calls :meth:`detect`; a legacy detector overrides it.

        :param prompt: Full prompt (context + question already concatenated).
        :param answer: Model-generated answer to inspect.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :param min_confidence: Drop ``"spans"`` below this confidence threshold (``[0, 1]``).
        :returns: List of predictions in requested format.
        :raises InvalidInputError: for an unsupported ``output_format`` or a ``min_confidence``
            outside ``[0, 1]``.
        """
        request = PromptInput(prompt=prompt, answer=answer)
        return self._predict_via_detect(request, output_format, min_confidence)

    def predict_prompt_batch(
        self,
        prompts: list[str],
        answers: list[str],
        output_format: str = "tokens",
        min_confidence: float = 0.0,
    ) -> list:
        """Batch version of :meth:`predict_prompt`; for now one pair at a time.

        :param prompts: List of full prompt strings.
        :param answers: List of answers to inspect.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :param min_confidence: Drop ``"spans"`` below this confidence threshold (``[0, 1]``).
        :returns: List of prediction lists, one per input pair.
        :raises InvalidInputError: for an unsupported ``output_format``, a ``min_confidence``
            outside ``[0, 1]`` or lists of different lengths.
        """
        self._check_legacy_arguments(output_format, min_confidence)
        if len(prompts) != len(answers):
            raise InvalidInputError(
                "prompts and answers must have the same length.",
                code="length_mismatch",
                param="answers",
            )
        return [
            self.predict_prompt(prompt, answer, output_format, min_confidence)
            for prompt, answer in zip(prompts, answers, strict=True)
        ]

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
