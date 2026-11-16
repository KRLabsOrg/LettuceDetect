"""Detector backed by a Vela 2.0 model that answers the SystemOne hallucination span question."""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import numbers
import threading
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from packaging.version import Version
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from transformers import AutoModel

from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.encoder import (
    _finish_load,
    _log_model_loaded,
    _model_load_error,
    _resolved_commit,
)
from lettucedetect.detectors.errors import (
    ConfigurationError,
    InvalidInputError,
    LettuceDetectError,
    UpstreamError,
)
from lettucedetect.detectors.types import (
    DetectedSpan,
    Detection,
    DetectRequest,
    ResultWarning,
    SourcesInput,
    Usage,
)
from lettucedetect.detectors.vela_client import SystemOneClient

logger = logging.getLogger(__name__)

__all__ = ["VelaConfig", "VelaDetector"]

# Id of the span question in the SystemOne request and reply.
_QUESTION_ID = "halu"

# The hallucination question as Vela documents it (``halu_schema`` in the model's calibration.json).
_DEFAULT_INSTRUCTIONS = "Which spans of the answer are not supported by the context?"
_DEFAULT_CRITERIA = {"unsupported": "a claim not supported by the context"}

# How Vela's hallucination question was trained to see a question: in front of the context, as
# "User request: <question>\n\n<context>" (``detect_hallucination`` in the model's code).
_REQUEST_PREFIX = "User request: "

# Oldest transformers any Vela 2.0 model card suggests; the decoders ask for more (see the extra).
_MIN_TRANSFORMERS = "4.57"
_RUNTIME_PACKAGES = ("safetensors", "tokenizers")
_EXTRA_HINT = "pip install 'lettucedetect[vela]'"


def _missing_dependency(what: str) -> ConfigurationError:
    """Build the error for a missing or too old runtime package; the caller chains the cause."""
    return ConfigurationError(
        f"Vela needs {what}. Install the runtime with: {_EXTRA_HINT}",
        code="missing_dependency",
    )


def _transformers_version() -> str:
    """Return the installed transformers version."""
    return transformers.__version__


def _check_runtime() -> None:
    """Raise :class:`ConfigurationError` if a package the Vela remote code needs is missing.

    :raises ConfigurationError: (``missing_dependency``) transformers is too old or a package is
        not installed.
    """
    if Version(_transformers_version()) < Version(_MIN_TRANSFORMERS):
        raise _missing_dependency(f"transformers>={_MIN_TRANSFORMERS}")
    for package in _RUNTIME_PACKAGES:
        if importlib.util.find_spec(package) is None:
            raise _missing_dependency(f"the {package} package")


def _invalid_output(message: str) -> UpstreamError:
    """Build the error for a reply that does not have the SystemOne span shape."""
    return UpstreamError(
        f"Vela returned an unreadable reply: {message}. Check that the model speaks SystemOne.",
        code="invalid_model_output",
        service="vela",
    )


def _is_int(value: object) -> bool:
    """Whether ``value`` is an integer (NumPy integers included) and not a bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    """Whether ``value`` is a real number (NumPy floats included) and not a bool."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


class VelaConfig(BaseModel):
    """Options of a :class:`VelaDetector`.

    The defaults are the hallucination question Vela was trained with; changing the wording can
    change the scores. An invalid option raises :class:`ConfigurationError` with the code
    ``invalid_config`` and ``param`` set to the option.

    :param instructions: the span question asked about the answer.
    :param criteria: span labels and their descriptions, at least one. With more than one, each
        span's ``category`` is the label it matched; with one, ``category`` is ``None``.
    :param threshold: probability above which Vela returns a span, in ``[0, 1]``. ``None`` uses
        the model's calibrated threshold. ``min_score`` of ``detect`` filters on top of it.
    :param max_length: the model's input budget in tokens. It does not change the model; a reply
        that used the whole budget is reported with the ``input_truncated`` warning. The reply
        cannot tell a cut input from one that fits exactly or a long answer scored in windows,
        so those are reported too.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    instructions: str = Field(default=_DEFAULT_INSTRUCTIONS, min_length=1)
    criteria: dict[str, str] = Field(default_factory=lambda: dict(_DEFAULT_CRITERIA), min_length=1)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    max_length: int = Field(default=8192, gt=0)

    @model_validator(mode="wrap")
    @classmethod
    def _raise_configuration_error(cls, data: Any, handler: Any) -> VelaConfig:  # noqa: ANN401
        try:
            config = handler(data)
        except ValidationError as exc:
            first = exc.errors(include_input=False, include_url=False)[0]
            param = str(first["loc"][0]) if first["loc"] else None
            raise ConfigurationError(
                f"Invalid VelaConfig option {param!r}; see the VelaConfig docstring.",
                code="invalid_config",
                param=param,
            ) from exc
        if not config.instructions.strip():
            raise ConfigurationError(
                "VelaConfig.instructions must not be blank.",
                code="invalid_config",
                param="instructions",
            )
        if any(not key.strip() or not value.strip() for key, value in config.criteria.items()):
            raise ConfigurationError(
                "VelaConfig.criteria must not have blank labels or descriptions.",
                code="invalid_config",
                param="criteria",
            )
        return config


class VelaDetector(BaseDetector):
    r"""Detect hallucinations with a Vela 2.0 model the caller already holds.

    Vela answers typed SystemOne questions about a request, a context and an answer. This detector
    asks one span question, "which spans of the answer are not supported by the context?", and
    returns Vela's spans with their probability as score. Any model with a
    ``system_one(state=..., questions=...)`` method that speaks SystemOne works.

    The state sent to the model has two SystemOne keys, ``source`` and ``answer``:

    * ``SourcesInput``: ``"User request: <question>\n\n<sources>"``, the non-blank sources joined
      with blank lines. Either the question or the sources may be left out, but not both.
    * ``PromptInput``: the prompt. A LettuceDetect prompt holds the request and the context.

    The question goes into ``source`` because that is how Vela's hallucination question was
    trained. Measured on the 10,698-example test set with every row sent as ``SourcesInput``, the
    0.3B scores an example-F1 of 0.847 with this join, 0.844 with the question and the sources
    joined by a blank line, and 0.840 with the encoder's prompt (``PromptUtils.format_context``).
    With the question as a separate ``request`` part it scored 0.830, and 586 rows failed: Vela
    fits an input into its token window by cutting the first part it does not score, which was
    then the request, so long sources could not be cut. With one ``source`` a long input is cut
    and reported with the ``input_truncated`` warning. That warning also marks an input that fits
    exactly or a long answer scored in windows, since the reply cannot tell them apart.

    With more than one criterion in the config, each span's ``category`` is the label Vela matched.

    Construction downloads nothing, reads no environment variable and writes no file. Calls to a
    local model run under one private lock, so concurrent calls on one instance run one at a time.

    Vela ships its code in the model repository, so loading needs ``trust_remote_code=True``.
    :meth:`from_pretrained` sets it; pin ``revision`` to a commit you have reviewed.
    :meth:`from_server` calls a SystemOne server instead, such as the ``vela2_serve.py`` script in
    the model repository; nothing is loaded in-process then.

    Example::

        detector = VelaDetector.from_pretrained("vllm-sr/Vela-2.0-0.3B", revision="<commit>")
        result = detector.detect(
            SourcesInput(
                question="When does it ship?",
                sources=[Chunk(id="doc-1", text="Ships Friday.")],
                answer="Ships Monday.",
            ),
            min_score=0.5,
        )

        served = VelaDetector.from_server(
            "http://127.0.0.1:8001", model="vllm-sr/Vela-2.0-0.3B", api_key="<key>"
        )

    :param model: a loaded Vela model, or any object with a system_one(state=..., questions=...) method..
    :param config: options, ``VelaConfig()`` when ``None``.
    :param model_id: id reported in ``usage``.
    :param model_version: version reported in ``usage``.
    """

    supports_tokens = False
    supports_detail = False
    model_id: str | None = None
    model_version: str | None = None

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        *,
        config: VelaConfig | None = None,
        model_id: str | None = None,
        model_version: str | None = None,
    ) -> None:
        """Build the detector from a loaded model."""
        self.config = config if config is not None else VelaConfig()
        self.model = model
        self.model_id = model_id
        self.model_version = model_version
        # Scoring lock is created here, not lazily as in EncoderDetector: from_pretrained
        # calls cls(), so __init__ always runs and no first-call race exists. A plain Lock is
        # enough because only _detect takes it, once per call. A model that says it is
        # thread_safe, such as SystemOneClient, is called without it.
        self._scoring_lock = threading.Lock()
        self._lock_calls = not getattr(model, "thread_safe", False)
        logger.debug(
            "vela detector built: model_id=%s criteria=%d threshold=%s max_length=%d",
            model_id,
            len(self.config.criteria),
            self.config.threshold,
            self.config.max_length,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        device: torch.device | str | None = None,
        config: VelaConfig | None = None,
        model_version: str | None = None,
        **loader_kwargs: Any,  # noqa: ANN401
    ) -> VelaDetector:
        """Load the Vela model for ``model_id`` and build the detector.

        This is the one place where a model name becomes a model. It may download, unless the
        caller set ``HF_HUB_OFFLINE`` (this method neither reads nor sets it). The model's own
        code runs with ``trust_remote_code=True``, so pin ``revision`` to a reviewed commit.

        :param model_id: Hugging Face id or local directory.
        :param revision: branch, tag or commit of the model repository; pin a commit.
        :param device: device for the model; CUDA when available, else CPU, when ``None``.
        :param config: options, ``VelaConfig()`` when ``None``.
        :param model_version: version reported in ``usage`` for a local path. For a hub id the
            resolved commit is used and this is only the fallback.
        :param loader_kwargs: passed to ``AutoModel.from_pretrained``, for example ``cache_dir``.
        :raises ConfigurationError: (``missing_dependency``) the ``vela`` extra is not installed;
            (``model_load_failed``) a local model could not be loaded.
        :raises UpstreamError: (``model_load_failed``) a hub model could not be loaded.
        """
        _check_runtime()
        is_local = Path(model_id).is_dir()
        kwargs = {**loader_kwargs, "revision": revision, "trust_remote_code": True}
        started = time.perf_counter()
        try:
            model = AutoModel.from_pretrained(model_id, **kwargs)
        except ImportError as exc:
            raise _missing_dependency("a package its model code imports") from exc
        except OSError as exc:
            raise _model_load_error(model_id, is_local) from exc
        device, _ = _finish_load(model, device, None, None, None)
        # Vela builds its inference engine on first use; build it now so a call does not.
        getattr(model, "vela2_engine", None)
        version = model_version
        if not is_local:
            version = _resolved_commit(model_id, kwargs) or model_version
        detector = cls(model, config=config, model_id=model_id, model_version=version)
        _log_model_loaded(model_id, version, device, started, module_logger=logger)
        return detector

    @classmethod
    def from_server(
        cls,
        base_url: str,
        *,
        model: str,
        api_key: str | None = None,
        timeout: float = 30.0,
        config: VelaConfig | None = None,
        model_version: str | None = None,
    ) -> VelaDetector:
        """Build a detector that calls a SystemOne server (``POST /v1/systemone``).

        No request is sent here, and nothing is read from the environment: the URL and the key
        come from the caller. The server does not report which commit of the model it runs, so
        ``usage`` carries the caller's ``model_version``.

        :param base_url: the server's base URL, for example ``http://127.0.0.1:8001``.
        :param model: the model name the server serves; also the ``model_id`` in ``usage``.
        :param api_key: sent as ``Authorization: Bearer <api_key>``; ``None`` sends no header.
        :param timeout: seconds to wait for the server, per request.
        :param config: options, ``VelaConfig()`` when ``None``.
        :param model_version: version reported in ``usage``.
        :raises ConfigurationError: (``invalid_config``) ``base_url`` or ``model`` is blank, or
            ``timeout`` is not positive.
        """
        client = SystemOneClient(base_url, model=model, api_key=api_key, timeout=timeout)
        return cls(client, config=config, model_id=model, model_version=model_version)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect(
        self, request: DetectRequest, *, include_tokens: bool, include_detail: bool
    ) -> Detection:
        """Ask Vela the span question and return its spans, usage and warnings.

        :param request: the sources (or prompt) and the answer to check.
        :param include_tokens: unused, there are no token scores.
        :param include_detail: unused, there is no detail.
        :returns: the detection result.
        :raises InvalidInputError: (``empty_context``) the request has no context to check against;
            (``input_too_long``) Vela could not cut the input into its token window.
        :raises UpstreamError: (``invalid_model_output``) Vela rejected the request, or the reply
            does not have the span shape.
            A detector from :meth:`from_server` also raises the errors of
            :meth:`SystemOneClient.system_one`.
        """
        answer = request.answer
        units: dict[str, int] = {"answer_chars": len(answer)}
        if isinstance(request, SourcesInput):
            units["sources"] = len(request.sources)
            units["source_chars"] = sum(len(chunk.text) for chunk in request.sources)
        else:
            units["source_chars"] = len(request.prompt)
        state = self._state(request)
        if not answer.strip():
            # Vela rejects an empty answer part, and there is nothing to flag in it.
            logger.debug("detect skipped: the answer is blank")
            return Detection(usage=[self._usage(units)])

        requested = time.perf_counter()
        with self._scoring_lock if self._lock_calls else contextlib.nullcontext():
            started = time.perf_counter()
            logger.debug(
                "detect started: kind=%s model_id=%s lock_wait_ms=%.1f",
                request.kind,
                self.model_id,
                (started - requested) * 1000,
            )
            try:
                reply = self.model.system_one(state=state, questions=self._questions())
            except LettuceDetectError:
                # A model that already raises typed errors (InvalidInputError is a ValueError)
                # passes them through as they are.
                raise
            except ValueError as exc:
                # Vela raises its SystemOneError (a ValueError) for a request it rejects. Its
                # message may quote the input, so it is chained but never repeated.
                if getattr(exc, "type", None) == "too_long":
                    # With one source part Vela can cut the input, so this is left for questions
                    # that alone do not fit the window.
                    raise InvalidInputError(
                        "The input does not fit the model's token window; pass a shorter input.",
                        code="input_too_long",
                        param="request",
                    ) from exc
                raise UpstreamError(
                    "Vela rejected the request. Check that the model speaks SystemOne.",
                    code="invalid_model_output",
                    service="vela",
                ) from exc
            scoring_ms = (time.perf_counter() - started) * 1000

        if not isinstance(reply, dict):
            raise _invalid_output("the reply is not an object")
        spans = self._parse_spans(reply, answer)
        input_tokens = self._input_tokens(reply)
        warnings: list[ResultWarning] = []
        if input_tokens is not None:
            units["input_tokens"] = input_tokens
            # Vela cuts one part of the input until the sequence is exactly max_length tokens, and
            # scores a long answer in windows whose tokens add up past it. An input that fits
            # exactly is reported too: the reply has no way to tell the two apart.
            if input_tokens >= self.config.max_length:
                # The reply counts tokens after the cut, so the input's size is logged in chars.
                logger.warning(
                    "input_truncated: the input was cut to max_length=%d tokens "
                    "(source_chars=%d answer_chars=%d)",
                    self.config.max_length,
                    units["source_chars"],
                    units["answer_chars"],
                )
                warnings.append(
                    ResultWarning(
                        code="input_truncated",
                        message=(
                            "The input reached the model's token budget, so part of it may not "
                            "have been scored. An input that fits exactly, or a long answer "
                            "scored in windows, is reported the same way."
                        ),
                    )
                )
        logger.debug(
            "detect finished: spans=%d input_tokens=%s truncated=%s scoring_ms=%.1f",
            len(spans),
            input_tokens,
            bool(warnings),
            scoring_ms,
        )
        return Detection(spans=spans, usage=[self._usage(units)], warnings=warnings)

    def _usage(self, units: dict[str, int]) -> Usage:
        """Return the usage entry of the Vela model."""
        return Usage(model_id=self.model_id, model_version=self.model_version, units=units)

    @staticmethod
    def _state(request: DetectRequest) -> dict[str, str]:
        r"""Return the SystemOne state for a request: ``source`` and ``answer``.

        For ``SourcesInput`` the source is ``"User request: <question>\n\n<sources>"``. A blank
        question or blank sources are left out, since Vela rejects an empty part.

        :raises InvalidInputError: (``empty_context``) there is neither a question nor a source.
        """
        if isinstance(request, SourcesInput):
            parts = []
            if request.question and request.question.strip():
                parts.append(_REQUEST_PREFIX + request.question)
            parts.extend(chunk.text for chunk in request.sources if chunk.text.strip())
            source = "\n\n".join(parts)
        else:
            source = request.prompt if request.prompt.strip() else ""
        if not source:
            raise InvalidInputError(
                "The request has no context to check the answer against; pass non-empty "
                "sources, a question or a prompt.",
                code="empty_context",
                param="request",
            )
        return {"source": source, "answer": request.answer}

    def _questions(self) -> dict[str, dict[str, Any]]:
        """Return the SystemOne questions: the one span question, which targets the answer."""
        question: dict[str, Any] = {
            "type": "span",
            "instructions": self.config.instructions,
            "criteria": dict(self.config.criteria),
        }
        if self.config.threshold is not None:
            question["threshold"] = self.config.threshold
        return {_QUESTION_ID: question}

    def _parse_spans(self, reply: dict[str, Any], answer: str) -> list[DetectedSpan]:
        """Map Vela's ``{label, start, end, text, probability}`` spans to detected spans.

        The span text is taken from the answer, so it always matches the offsets. With more than
        one criterion the label goes into ``category``; with one it says nothing and is dropped.

        :raises UpstreamError: (``invalid_model_output``) a span is missing or malformed.
        """
        labelled = len(self.config.criteria) > 1
        spans_by_question = reply.get("spans")
        items = spans_by_question.get(_QUESTION_ID) if isinstance(spans_by_question, dict) else None
        if not isinstance(items, list):
            raise _invalid_output("it has no span list for the hallucination question")
        spans = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise _invalid_output(f"spans[{index}] is not an object")
            start, end, probability = item.get("start"), item.get("end"), item.get("probability")
            if not (_is_int(start) and _is_int(end) and 0 <= start < end <= len(answer)):
                raise _invalid_output(f"spans[{index}] has offsets outside the answer")
            if not (_is_number(probability) and 0.0 <= probability <= 1.0):
                raise _invalid_output(f"spans[{index}] has a probability outside [0, 1]")
            category = None
            if labelled:
                category = item.get("label")
                if not isinstance(category, str) or category not in self.config.criteria:
                    raise _invalid_output(f"spans[{index}] has a label that is not a criterion")
            start, end = int(start), int(end)
            spans.append(
                DetectedSpan(
                    start=start,
                    end=end,
                    text=answer[start:end],
                    score=float(probability),
                    category=category,
                )
            )
        return spans

    @staticmethod
    def _input_tokens(reply: dict[str, Any]) -> int | None:
        """Return ``usage.input_tokens`` of the reply, ``None`` when the model does not report it.

        :raises UpstreamError: (``invalid_model_output``) the count is not a non-negative integer.
        """
        usage = reply.get("usage")
        value = usage.get("input_tokens") if isinstance(usage, dict) else None
        if value is None:
            return None
        if not _is_int(value) or value < 0:
            raise _invalid_output("usage.input_tokens is not a non-negative integer")
        return int(value)
