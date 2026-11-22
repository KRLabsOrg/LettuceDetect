"""RAGFactChecker detector: hallucinated spans from triplet fact checking."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from pydantic import ValidationError

from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.errors import ConfigurationError, InvalidInputError, UpstreamError
from lettucedetect.detectors.types import (
    Chunk,
    DetailedResult,
    DetectedSpan,
    Detection,
    DetectRequest,
    PromptInput,
    ResultWarning,
    SourcesInput,
    Triplets,
    Usage,
)

_LEGACY_FORMATS = ("tokens", "spans", "detailed")

logger = logging.getLogger(__name__)


class RAGFactCheckerDetector(BaseDetector):
    """Detect hallucinations by extracting triplets and fact checking them against the sources.

    The span scores are fixed values, not model probabilities: 0.9 when a hallucinated triplet
    is found in the answer as a phrase, 0.7 when only one of its words is found. A result with
    spans carries a ``synthetic_score`` warning that says so.

    Pass ``fact_checker`` to use the detector without an API key, for example in tests::

        detector = RAGFactCheckerDetector(fact_checker=my_fact_checker)
        result = detector.detect(SourcesInput(sources=[...], answer="..."))

    :param openai_api_key: OpenAI API key; unused when ``fact_checker`` is passed.
    :param model: OpenAI model (default ``"gpt-4o"``); also the ``model_id`` in ``usage``.
    :param base_url: optional base URL of the API, for example ``"http://localhost:1234/v1"``.
    :param temperature: sampling temperature (default ``0.0``).
    :param fact_checker: an object with ``detect_hallucinations(context, answer, question)``
        (and ``detect_hallucinations_batch`` for the batch method). When passed, nothing is built
        and the environment is not read. When omitted, ``RAGFactChecker`` is built from the
        arguments above, for backward compatibility (it lives outside this package).
    :param max_concurrency: the most ``detect``, ``adetect``, ``predict`` and ``predict_prompt``
        calls that run the fact checker at once on this instance (default 30).
    :param kwargs: ignored, kept for backward compatibility.
    :raises ConfigurationError: (``invalid_config``) ``max_concurrency`` is not an integer >= 1.
    """

    supports_tokens = False
    supports_detail = True

    def __init__(
        self,
        openai_api_key: str | None = None,
        model: str = "gpt-4o",
        base_url: str | None = None,
        temperature: float = 0.0,
        *,
        fact_checker: Any = None,  # noqa: ANN401
        max_concurrency: int = 30,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Build the detector; see the class docstring for the arguments."""
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency < 1
        ):
            raise ConfigurationError(
                "max_concurrency must be an integer >= 1; pass for example max_concurrency=30.",
                code="invalid_config",
                param="max_concurrency",
            )
        self.model_id = model
        self.max_concurrency = max_concurrency
        if fact_checker is None:
            # Kept for backward compatibility; pass fact_checker= to skip building a client.
            from lettucedetect.ragfactchecker import RAGFactChecker

            fact_checker = RAGFactChecker(
                openai_api_key=openai_api_key,
                model=model,
                base_url=base_url,
                temperature=temperature,
            )
        self.rag = fact_checker
        # The LLM calls happen inside the fact checker and can't be counted from outside, so the
        # bound is on whole fact-check calls.
        self._semaphore = threading.BoundedSemaphore(max_concurrency)

    def _detect(
        self, request: DetectRequest, *, include_tokens: bool, include_detail: bool
    ) -> Detection:
        """Fact check the answer and return its hallucinated spans.

        :param request: the sources (or prompt) and the answer to check.
        :param include_tokens: unused, ``detect`` rejects it (``supports_tokens`` is ``False``).
        :param include_detail: whether to fill ``Detection.detail`` with the triplets, the
            fact-check results and the source triplet of each span.
        :returns: the detection result.
        :raises UpstreamError: (``invalid_model_output``) ``include_detail`` is set and the fact
            checker returned triplets or results of an unexpected shape.
        """
        answer = request.answer
        if isinstance(request, SourcesInput):
            context = [chunk.text for chunk in request.sources]
            question = request.question
        else:
            context = [request.prompt]
            question = None
        units = {
            "sources": len(context),
            "source_chars": sum(len(text) for text in context),
            "answer_chars": len(answer),
        }

        requested = time.perf_counter()
        with self._semaphore:
            started = time.perf_counter()
            logger.debug(
                "detect started: kind=%s sources=%d model_id=%s wait_ms=%.1f",
                request.kind,
                units["sources"],
                self.model_id,
                (started - requested) * 1000,
            )
            result = self.rag.detect_hallucinations(context, answer, question)
            check_ms = (time.perf_counter() - started) * 1000

        raw_spans = self._convert_to_spans(answer, result)
        spans = [
            DetectedSpan(
                start=item["start"], end=item["end"], text=item["text"], score=item["confidence"]
            )
            for item in raw_spans
        ]
        detail = self._detail(result, raw_spans) if include_detail else None
        warnings = (
            [
                ResultWarning(
                    code="synthetic_score",
                    message=(
                        "Scores are fixed values (0.9 for a full match, 0.7 for a partial "
                        "match), not model probabilities."
                    ),
                )
            ]
            if spans
            else []
        )
        logger.debug("detect finished: spans=%d check_ms=%.1f", len(spans), check_ms)
        return Detection(
            spans=spans,
            detail=detail,
            usage=[Usage(model_id=self.model_id, model_version=None, units=units)],
            warnings=warnings,
        )

    @staticmethod
    def _detail(result: dict[str, Any], raw_spans: list[dict[str, Any]]) -> DetailedResult:
        """Build the evidence of one fact check; ``span_triplets`` is parallel to the spans."""
        try:
            return DetailedResult(
                triplets=Triplets(
                    answer=result.get("answer_triplets", []),
                    context=result.get("context_triplets", []),
                    hallucinated=result.get("hallucinated_triplets", []),
                ),
                fact_check_results=result.get("fact_check_results", {}),
                span_triplets=[item.get("triplet") for item in raw_spans],
            )
        except ValidationError as exc:
            raise UpstreamError(
                "The fact checker returned triplets or results of an unexpected shape.",
                code="invalid_model_output",
                service="rag_fact_checker",
            ) from exc

    @staticmethod
    def _check_rag_arguments(output_format: str, min_confidence: float) -> None:
        """Validate the arguments the ``predict*`` methods share."""
        if output_format not in _LEGACY_FORMATS:
            raise InvalidInputError(
                f"Invalid output format '{output_format}'. "
                "RAGFactChecker supports 'tokens', 'spans', or 'detailed'",
                code="unsupported_output_format",
                param="output_format",
            )
        if not 0.0 <= min_confidence <= 1.0:
            raise InvalidInputError(
                "min_confidence must be in [0, 1]; pass a value between 0.0 and 1.0.",
                code="invalid_min_score",
                param="min_confidence",
            )

    def _legacy_output(
        self, request: DetectRequest, output_format: str, min_confidence: float
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Answer a ``predict*`` call with :meth:`detect`, in the legacy dict format."""
        detection = self.detect(request, include_detail=True)
        detail = detection.detail
        assert detail is not None  # noqa: S101 - detect fills it when asked
        if output_format == "tokens":
            return self._convert_to_tokens(
                request.answer, {"hallucinated_triplets": detail.triplets.hallucinated}
            )
        spans = [
            {
                "start": span.start,
                "end": span.end,
                "text": span.text,
                "confidence": span.score,
                "triplet": triplet,
            }
            for span, triplet in zip(detection.spans, detail.span_triplets, strict=True)
        ]
        if output_format == "detailed":
            # Detailed output is not filtered by min_confidence.
            return {
                "spans": spans,
                "triplets": detail.triplets.model_dump(),
                "fact_check_results": dict(detail.fact_check_results),
            }
        return self._filter_spans_by_confidence(spans, output_format, min_confidence)

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
        min_confidence: float = 0.0,
        **kwargs,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Predict hallucinations using RAGFactChecker, through :meth:`detect`.

        :param context: List of context documents
        :param answer: Answer text to check for hallucinations
        :param question: Question (optional)
        :param output_format: "tokens", "spans", or "detailed"
        :param min_confidence: Drop ``"spans"`` below this confidence threshold (``[0, 1]``).
            Does not filter ``output_format="detailed"``; detailed output returns
            the unfiltered spans, triplets, and fact-check results.
        :param kwargs: Additional arguments

        :return: List of predictions in lettuceDetect format, or dict for detailed format
        :raises InvalidInputError: (``unsupported_output_format``) for an unknown
            ``output_format``; (``invalid_min_score``) for ``min_confidence`` outside ``[0, 1]``.
        """
        self._check_rag_arguments(output_format, min_confidence)
        request = SourcesInput(
            sources=[Chunk(id=str(i), text=text) for i, text in enumerate(context)],
            answer=answer,
            question=question,
        )
        return self._legacy_output(request, output_format, min_confidence)

    def predict_prompt(
        self,
        prompt: str,
        answer: str,
        output_format: str = "tokens",
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Predict using a single prompt string as context, through :meth:`detect`.

        :raises InvalidInputError: as :meth:`predict`.
        """
        self._check_rag_arguments(output_format, min_confidence)
        request = PromptInput(prompt=prompt, answer=answer)
        return self._legacy_output(request, output_format, min_confidence)

    def predict_prompt_batch(
        self,
        prompts: list[str],
        answers: list[str],
        output_format: str = "tokens",
        min_confidence: float = 0.0,
    ) -> list[list[dict[str, Any]]]:
        """Batch prediction using RAGFactChecker's batch processing.

        :raises InvalidInputError: (``length_mismatch``) for lists of different lengths;
            (``invalid_min_score``) for ``min_confidence`` outside ``[0, 1]``;
            (``unsupported_output_format``) for anything but ``"tokens"`` or ``"spans"``.
        """
        if len(prompts) != len(answers):
            raise InvalidInputError(
                "Number of prompts must match number of answers",
                code="length_mismatch",
                param="answers",
            )
        if not 0.0 <= min_confidence <= 1.0:
            raise InvalidInputError(
                "min_confidence must be in [0, 1]; pass a value between 0.0 and 1.0.",
                code="invalid_min_score",
                param="min_confidence",
            )
        if output_format not in ("tokens", "spans"):
            raise InvalidInputError(
                f"Unknown output format: {output_format}",
                code="unsupported_output_format",
                param="output_format",
            )

        contexts = [[prompt] for prompt in prompts]  # Convert prompts to context lists
        rag_results = self.rag.detect_hallucinations_batch(contexts, answers)

        # Convert each result to lettuceDetect format
        converted_results = []
        for i, (answer, rag_result) in enumerate(zip(answers, rag_results)):
            if output_format == "tokens":
                converted = self._convert_to_tokens(answer, rag_result)
            else:  # spans
                converted = self._convert_to_spans(answer, rag_result)
                converted = self._filter_spans_by_confidence(
                    converted, output_format, min_confidence
                )
            converted_results.append(converted)

        return converted_results

    def _convert_to_tokens(self, answer: str, rag_result: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert RAGFactChecker result to token format."""
        tokens = answer.split()
        hallucinated_triplets = rag_result.get("hallucinated_triplets", [])

        token_predictions = []
        for i, token in enumerate(tokens):
            # Simple check if token appears in any hallucinated triplet
            is_hallucinated = any(
                token.lower() in " ".join(triplet).lower() for triplet in hallucinated_triplets
            )

            token_predictions.append(
                {
                    "token": token,
                    "pred": 1 if is_hallucinated else 0,
                    "prob": 0.9 if is_hallucinated else 0.1,
                }
            )

        return token_predictions

    def _convert_to_spans(self, answer: str, rag_result: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert RAGFactChecker result to span format with improved triplet matching."""
        spans = []
        hallucinated_triplets = rag_result.get("hallucinated_triplets", [])

        for triplet in hallucinated_triplets:
            if len(triplet) < 3:
                continue

            # Try different patterns to find triplet elements in text
            patterns = [
                f"{triplet[0]} {triplet[1]} {triplet[2]}",  # Full triplet phrase
                f"{triplet[0]} {triplet[2]}",  # Subject + object
                triplet[2],  # Object (often contains the hallucination)
                triplet[0],  # Subject
                triplet[1],  # Predicate
            ]

            found_span = False
            for pattern in patterns:
                if not pattern or not pattern.strip():
                    continue

                # Try exact match first, then case-insensitive
                start = answer.find(pattern)
                if start == -1:
                    start = answer.lower().find(pattern.lower())
                    if start != -1:
                        # Get the actual text from the answer with correct case
                        pattern = answer[start : start + len(pattern)]

                if start != -1:
                    spans.append(
                        {
                            "start": start,
                            "end": start + len(pattern),
                            "text": pattern,
                            "confidence": 0.9,
                            "triplet": triplet,  # Include source triplet for transparency
                        }
                    )
                    found_span = True
                    break

            # If no pattern matched, try individual words from the triplet
            if not found_span:
                for element in triplet:
                    if element and element.strip() and len(element) > 3:  # Skip short words
                        start = answer.lower().find(element.lower())
                        if start != -1:
                            actual_text = answer[start : start + len(element)]
                            spans.append(
                                {
                                    "start": start,
                                    "end": start + len(element),
                                    "text": actual_text,
                                    "confidence": 0.7,  # Lower confidence for partial matches
                                    "triplet": triplet,
                                }
                            )
                            break

        # Merge overlapping spans
        return self._merge_overlapping_spans(spans, answer)

    def _merge_overlapping_spans(
        self, spans: list[dict[str, Any]], answer: str
    ) -> list[dict[str, Any]]:
        """Merge overlapping spans to avoid duplicates; the merged text is cut from ``answer``."""
        if not spans:
            return spans

        # Sort spans by start position
        sorted_spans = sorted(spans, key=lambda x: x["start"])
        merged = [sorted_spans[0]]

        for current in sorted_spans[1:]:
            last = merged[-1]

            # Check if spans overlap
            if current["start"] <= last["end"]:
                # Merge spans - extend the end
                end = max(last["end"], current["end"])
                merged[-1] = {
                    "start": last["start"],
                    "end": end,
                    "text": answer[last["start"] : end],
                    "confidence": max(last["confidence"], current["confidence"]),
                    "triplet": last.get("triplet", current.get("triplet")),
                }
            else:
                merged.append(current)

        return merged
