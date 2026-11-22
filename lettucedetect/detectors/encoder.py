"""Detector backed by a fine-tuned token classifier that the caller has already loaded."""

from __future__ import annotations

import logging
import threading
import time
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import try_to_load_from_cache
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from transformers import AutoModelForTokenClassification, AutoTokenizer

from lettucedetect.datasets.hallucination_dataset import HallucinationDataset
from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.errors import ConfigurationError, LettuceDetectError, UpstreamError
from lettucedetect.detectors.prompt_utils import LANG_TO_PASSAGE, PromptUtils
from lettucedetect.detectors.types import (
    BatchResult,
    DetectedSpan,
    Detection,
    DetectRequest,
    ResultWarning,
    SourcesInput,
    TokenScore,
    Usage,
)

logger = logging.getLogger(__name__)

__all__ = ["EncoderConfig", "EncoderDetector"]


def _resolved_commit(model_id: str, loader_kwargs: dict[str, Any]) -> str | None:
    """Return the Hugging Face commit that is cached for ``model_id``, or ``None``.

    Reads the snapshot directory name from the local cache, so it needs no network and
    names the revision that was just loaded (``revision`` and ``cache_dir`` are honoured).
    """
    cached = try_to_load_from_cache(
        model_id,
        "config.json",
        cache_dir=loader_kwargs.get("cache_dir"),
        revision=loader_kwargs.get("revision"),
    )
    return Path(cached).parent.name if isinstance(cached, str) else None


# Makes two racing first calls get the same scoring lock.
_LOCK_GUARD = threading.Lock()
# One scoring lock per tokenizer, by id(tokenizer). A fast tokenizer fails with "Already borrowed"
# when two threads use it at once, also through two detectors. An entry lives as long as a
# detector holds its lock, and that detector holds the tokenizer, so an id is never reused.
_TOKENIZER_LOCKS: weakref.WeakValueDictionary[int, threading.RLock] = weakref.WeakValueDictionary()


def _resolve_taxonomy(
    taxonomy_head: str | None, include_taxonomy: bool | list | dict
) -> tuple[dict | None, dict | None]:
    """Resolve the label set of the typing head; ``(None, None)`` without a head."""
    if taxonomy_head is None:
        return None, None
    from lettucedetect.detectors.taxonomy_head import resolve_taxonomy

    return resolve_taxonomy(include_taxonomy)


def _model_load_error(model_id: str, is_local: bool) -> ConfigurationError | UpstreamError:
    """Build the error for a model that could not be loaded; the caller chains the cause.

    A local path is the caller's configuration, a hub id is an upstream failure. The message
    never carries the loader's own text, which may hold paths or tokens.
    """
    if is_local:
        return ConfigurationError(
            "The model could not be loaded from the local path. Check that model_id "
            "is a directory with a saved model and tokenizer.",
            code="model_load_failed",
            param="model_id",
        )
    return UpstreamError(
        f"The model {model_id!r} could not be loaded. Check the id, the network or "
        "HF_HUB_OFFLINE setting and your access to the repository.",
        code="model_load_failed",
        service="huggingface",
    )


# Loader options that say where and how to reach the Hugging Face cache. They describe the
# environment, not one repository, so the typing head is loaded with the same ones. ``revision``
# is left out on purpose: it names a revision of the encoder's repository, not of the head's.
_HEAD_LOADER_KWARGS = ("cache_dir", "local_files_only", "token")


def _head_loader_kwargs(loader_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pick the loader options the typing head shares with the encoder."""
    return {key: loader_kwargs[key] for key in _HEAD_LOADER_KWARGS if key in loader_kwargs}


def _finish_load(
    model: Any,  # noqa: ANN401
    device: torch.device | str | None,
    taxonomy_head: str | None,
    categories: dict | None,
    subcategories: dict | None,
    loader_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.device, Any]:
    """Move the model to the device, set eval mode and load the typer, if any.

    Shared by ``EncoderDetector.from_pretrained`` and the deprecated ``TransformerDetector``.
    ``loader_kwargs`` are the caller's loader options; only the cache location and access
    options reach the typing head.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    typer = None
    if taxonomy_head is not None:
        from lettucedetect.detectors.taxonomy_head import TaxonomyTyper

        typer = TaxonomyTyper.from_pretrained(
            taxonomy_head,
            device=device,
            categories=categories,
            subcategories=subcategories,
            **_head_loader_kwargs(loader_kwargs or {}),
        )
    return device, typer


def _log_model_loaded(
    model_id: str,
    version: str | None,
    device: torch.device,
    started: float,
    module_logger: logging.Logger | None = None,
) -> None:
    """Log the ``model_loaded`` event: ids and timing only, never customer text.

    :param module_logger: the calling module's logger, this module's logger when ``None``.
    """
    (module_logger or logger).info(
        "model loaded: model_id=%s model_version=%s device=%s load_seconds=%.1f",
        model_id,
        version,
        device,
        time.perf_counter() - started,
        extra={"event": "model_loaded"},
    )


class EncoderConfig(BaseModel):
    """Options of an :class:`EncoderDetector`.

    An invalid option raises :class:`ConfigurationError` with the code ``invalid_config``,
    ``invalid_language`` or ``invalid_batch_size`` and ``param`` set to the option.

    :param max_length: maximum number of tokens of prompt plus answer.
    :param batch_size: number of requests ``detect_batch`` scores together in one padded forward
        pass. ``detect`` scores one request at a time.
    :param lang: language of the prompt template, one of the supported languages.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    max_length: int = Field(default=4096, gt=0)
    batch_size: int = Field(default=16, ge=1)
    lang: str = "en"

    @model_validator(mode="wrap")
    @classmethod
    def _raise_configuration_error(cls, data: Any, handler: Any) -> EncoderConfig:  # noqa: ANN401
        try:
            config = handler(data)
        except ValidationError as exc:
            first = exc.errors(include_input=False, include_url=False)[0]
            param = str(first["loc"][0]) if first["loc"] else None
            code = {"lang": "invalid_language", "batch_size": "invalid_batch_size"}.get(
                param or "", "invalid_config"
            )
            if first["type"] == "extra_forbidden":
                code = "invalid_config"
            raise ConfigurationError(
                f"Invalid EncoderConfig option {param!r}; see the EncoderConfig docstring.",
                code=code,
                param=param,
            ) from exc
        if config.lang not in LANG_TO_PASSAGE:
            raise ConfigurationError(
                f"Invalid language; choose from {', '.join(LANG_TO_PASSAGE)}.",
                code="invalid_language",
                param="lang",
            )
        return config


class EncoderDetector(BaseDetector):
    """Detect hallucinations with a token classifier and tokenizer the caller already holds.

    Construction downloads nothing, reads no environment variable and writes no file. The model
    is not moved or switched to eval mode: pass it ready for inference.

    When prompt plus answer exceed ``max_length`` tokens, sources are split into passage groups,
    each group is scored on its own and the per-token probability is the ``max()`` over groups.
    Scoring runs under one private lock, so concurrent calls on one instance run one at a time.
    ``detect_batch`` scores requests in padded groups of ``batch_size``, one forward pass per group.

    Example, loading the model once at startup::

        from transformers import AutoModelForTokenClassification, AutoTokenizer

        model_id = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id).eval()

        detector = EncoderDetector(
            model,
            tokenizer,
            config=EncoderConfig(max_length=4096, batch_size=16, lang="en"),
            model_id=model_id,
        )
        result = detector.detect(
            SourcesInput(
                question="What is the capital of France?",
                sources=[Chunk(id="doc-1", text="The capital of France is Paris.")],
                answer="The capital of France is Lyon.",
            )
        )

    :param model: a token classification model.
    :param tokenizer: the model's fast tokenizer.
    :param typer: optional span typer, for example a :class:`TaxonomyTyper`. With it every span
        of ``detect`` carries ``category`` and ``subcategory``, and ``usage`` has a second
        entry for the typer (unit ``spans``).
    :param config: options, ``EncoderConfig()`` when ``None``.
    :param model_id: id reported in ``usage``.
    :param model_version: version reported in ``usage``.
    """

    supports_tokens = True
    model_id: str | None = None
    model_version: str | None = None
    typer: Any = None
    batch_size: int = EncoderConfig().batch_size

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        tokenizer: Any,  # noqa: ANN401
        *,
        typer: Any = None,  # noqa: ANN401
        config: EncoderConfig | None = None,
        model_id: str | None = None,
        model_version: str | None = None,
    ) -> None:
        """Build the detector from loaded objects."""
        self.config = config if config is not None else EncoderConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.typer = typer
        self.max_length = self.config.max_length
        self.lang = self.config.lang
        self.batch_size = self.config.batch_size
        self.model_id = model_id
        self.model_version = model_version
        device = getattr(model, "device", None)
        self.device = device if isinstance(device, torch.device) else torch.device("cpu")
        logger.debug(
            "encoder detector built: model_id=%s max_length=%d batch_size=%d lang=%s device=%s",
            model_id,
            self.max_length,
            self.batch_size,
            self.lang,
            self.device,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: torch.device | str | None = None,
        taxonomy_head: str | None = None,
        include_taxonomy: bool | list | dict = True,
        model_version: str | None = None,
        config: EncoderConfig | None = None,
        **loader_kwargs: Any,  # noqa: ANN401
    ) -> EncoderDetector:
        """Load the tokenizer and model for ``model_id`` and build the detector.

        This is the one place where a model name becomes a model. It may download, unless
        the caller set ``HF_HUB_OFFLINE`` (this method neither reads nor sets it).

        :param model_id: Hugging Face id or local directory.
        :param device: device for the model; CUDA when available, else CPU, when ``None``.
        :param taxonomy_head: optional id or path of a label-conditioned typing head. It is
            loaded through ``TaxonomyTyper.from_pretrained`` and types the spans of ``detect``.
        :param include_taxonomy: label set for the typing head.
        :param model_version: version reported in ``usage`` for a local path. For a hub id the
            resolved commit is used and this is only the fallback.
        :param config: options, ``EncoderConfig()`` when ``None``.
        :param loader_kwargs: passed to both ``from_pretrained`` loaders.
        :raises UpstreamError: (``model_load_failed``) a hub model could not be loaded.
        :raises ConfigurationError: (``model_load_failed``) a local model could not be loaded.
        """
        categories, subcategories = _resolve_taxonomy(taxonomy_head, include_taxonomy)
        is_local = Path(model_id).is_dir()
        started = time.perf_counter()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **loader_kwargs)
            model = AutoModelForTokenClassification.from_pretrained(model_id, **loader_kwargs)
        except OSError as exc:
            raise _model_load_error(model_id, is_local) from exc
        device, typer = _finish_load(
            model, device, taxonomy_head, categories, subcategories, loader_kwargs
        )
        version = model_version
        if not is_local:
            version = _resolved_commit(model_id, loader_kwargs) or model_version
        # ``__new__`` plus the base ``__init__``: a subclass with its own ``__init__`` still works.
        detector = cls.__new__(cls)
        EncoderDetector.__init__(
            detector,
            model,
            tokenizer,
            typer=typer,
            config=config,
            model_id=model_id,
            model_version=version,
        )
        detector.device = device
        _log_model_loaded(model_id, version, device, started)
        return detector

    # ------------------------------------------------------------------
    # Lock
    # ------------------------------------------------------------------

    def _get_scoring_lock(self) -> threading.RLock:
        """Return the scoring lock, taking it on first use (not in ``__init__``).

        Detectors built on the same tokenizer share one lock.
        """
        lock = vars(self).get("_scoring_lock")
        if lock is None:
            with _LOCK_GUARD:
                lock = vars(self).get("_scoring_lock")
                if lock is None:
                    lock = _TOKENIZER_LOCKS.get(id(self.tokenizer))
                    if lock is None:
                        lock = threading.RLock()
                        _TOKENIZER_LOCKS[id(self.tokenizer)] = lock
                    self._scoring_lock = lock
        return lock

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect(
        self, request: DetectRequest, *, include_tokens: bool, include_detail: bool
    ) -> Detection:
        """Score a request and return its spans, optional token scores, usage and warnings.

        One forward pass per passage group serves the spans, the tokens, the usage and the
        truncation check.

        :param request: the sources (or prompt) and the answer to check.
        :param include_tokens: whether to fill ``Detection.tokens``.
        :param include_detail: unused, there is no detail.
        :returns: the detection result.
        """
        requested = time.perf_counter()
        with self._get_scoring_lock():
            started = time.perf_counter()
            logger.debug(
                "detect started: kind=%s model_id=%s include_tokens=%s lock_wait_ms=%.1f",
                request.kind,
                self.model_id,
                include_tokens,
                (started - requested) * 1000,
            )
            prompts, units = self._prompts_and_units(request)
            scored = [self._score_tokens(prompt, request.answer) for prompt in prompts]
            scoring_ms = (time.perf_counter() - started) * 1000
        return self._detection(request, units, scored, include_tokens, scoring_ms)

    def detect_batch(
        self,
        requests: Sequence[DetectRequest],
        *,
        min_score: float = 0.0,
        include_tokens: bool = False,
        include_detail: bool = False,
    ) -> list[BatchResult]:
        """Score many requests in padded groups of ``batch_size``, one forward pass per group.

        Each result equals what :meth:`detect` returns for that request. A request whose sources
        need several passage groups is scored on its own, as ``detect`` scores it. The scoring
        lock is held for the whole batch, so a batch and single calls never score at the same
        time. A typed error raised while a padded group is scored is the result of every request
        of that group.

        :param requests: the requests, any mix of ``SourcesInput`` and ``PromptInput``.
        :param min_score: drop spans scored below this value, in ``[0, 1]``. Tokens are kept.
        :param include_tokens: also return token scores.
        :param include_detail: not supported, raises ``InvalidInputError``.
        :returns: one result per request, in input order.
        :raises InvalidInputError: before any request runs, as :meth:`BaseDetector.detect_batch`.
        """
        self._check_batch(requests, min_score, include_tokens, include_detail)
        cls = type(self)
        if (
            cls._detect is not EncoderDetector._detect
            or cls.detect is not BaseDetector.detect
            or any(self._answers_through_predict(r, include_tokens, False) for r in requests)
        ):
            # A subclass that answers detect in its own way gets every request through detect.
            return super().detect_batch(
                requests,
                min_score=min_score,
                include_tokens=include_tokens,
                include_detail=include_detail,
            )

        # Per request: the scored passage groups, or the error its scoring raised.
        scored: list[list[tuple[list[dict], int, int | None]] | LettuceDetectError]
        scored = [[] for _ in requests]
        scoring_ms = [0.0] * len(requests)
        with self._get_scoring_lock():
            prepared = [self._prompts_and_units(request) for request in requests]
            # Without right padding a group of one needs no padding at all.
            batch_size = self.batch_size if self.tokenizer.padding_side == "right" else 1
            single = [i for i, (prompts, _) in enumerate(prepared) if len(prompts) == 1]
            groups = [single[i : i + batch_size] for i in range(0, len(single), batch_size)]
            groups += [[i] for i, (prompts, _) in enumerate(prepared) if len(prompts) > 1]
            logger.debug(
                "detect_batch started: requests=%d forward_groups=%d batch_size=%d model_id=%s",
                len(requests),
                len(groups),
                batch_size,
                self.model_id,
            )
            for group in groups:
                started = time.perf_counter()
                prompts = prepared[group[0]][0]
                try:
                    if len(prompts) > 1:
                        answer = requests[group[0]].answer
                        outcome = [[self._score_tokens(prompt, answer) for prompt in prompts]]
                    else:
                        outcome = [
                            [item]
                            for item in self._score_tokens_batch(
                                [prepared[i][0][0] for i in group],
                                [requests[i].answer for i in group],
                            )
                        ]
                except LettuceDetectError as exc:
                    outcome = [exc] * len(group)
                elapsed = (time.perf_counter() - started) * 1000
                for i, item in zip(group, outcome, strict=True):
                    scored[i], scoring_ms[i] = item, elapsed

        results = []
        for i, request in enumerate(requests):
            try:
                item = scored[i]
                if isinstance(item, LettuceDetectError):
                    raise item
                detection = self._detection(
                    request, prepared[i][1], item, include_tokens, scoring_ms[i]
                )
                results.append(BatchResult(detection=self._drop_spans_below(detection, min_score)))
            except LettuceDetectError as exc:
                results.append(BatchResult(error=exc))
        return results

    def _prompts_and_units(self, request: DetectRequest) -> tuple[list[str], dict[str, int]]:
        """Build the prompts of a request and the usage counts known before scoring.

        It tokenizes to group the passages, so the caller holds the scoring lock.

        :param request: the sources (or prompt) and the answer to check.
        :returns: one prompt per passage group, and the usage units.
        """
        answer = request.answer
        units: dict[str, int] = {"answer_chars": len(answer)}
        if isinstance(request, SourcesInput):
            context = [chunk.text for chunk in request.sources]
            groups = self._group_passages_into_chunks(context, request.question, answer)
            logger.debug(
                "scoring sources request: sources=%d passage_groups=%d",
                len(context),
                len(groups),
            )
            prompts = [
                PromptUtils.format_context(group, request.question, self.lang) for group in groups
            ]
            units["sources"] = len(context)
            units["source_chars"] = sum(len(text) for text in context)
        else:
            prompts = [request.prompt]
            units["source_chars"] = len(request.prompt)
        return prompts, units

    def _detection(
        self,
        request: DetectRequest,
        units: dict[str, int],
        scored: list[tuple[list[dict], int, int | None]],
        include_tokens: bool,
        scoring_ms: float,
    ) -> Detection:
        """Build the result of a request from its scored passage groups.

        It runs without the scoring lock.

        :param request: the sources (or prompt) and the answer that were scored.
        :param units: the usage units from :meth:`_prompts_and_units`; ``input_tokens`` is added.
        :param scored: what :meth:`_score_tokens` returned for each passage group.
        :param include_tokens: whether to fill ``Detection.tokens``.
        :param scoring_ms: how long scoring took; in a batch, the time of the request's group.
        :returns: the detection result.
        """
        answer = request.answer
        truncated = False
        for index, (_, input_tokens, full_tokens) in enumerate(scored):
            if full_tokens is not None:
                truncated = True
                logger.warning(
                    "input_truncated: group %d of %d has %d tokens, max_length is %d",
                    index + 1,
                    len(scored),
                    full_tokens,
                    self.max_length,
                )
        units["input_tokens"] = sum(input_tokens for _, input_tokens, _ in scored)

        tokens = (
            scored[0][0]
            if len(scored) == 1
            else self._max_aggregate([group_tokens for group_tokens, _, _ in scored])
        )
        span_items = self._spans_from_scored_tokens(tokens, answer)
        typing_ms = 0.0
        if self.typer is not None and span_items:
            # Typing runs outside the scoring lock: the typer serializes itself, and this keeps
            # the encoder free for the next call.
            typer_started = time.perf_counter()
            span_items = self.typer.type_spans(answer, self._typer_context(request), span_items)
            typing_ms = (time.perf_counter() - typer_started) * 1000
        spans = [
            DetectedSpan(
                start=item["start"],
                end=item["end"],
                text=item["text"],
                score=item["confidence"],
                category=item.get("category"),
                subcategory=item.get("subcategory"),
            )
            for item in span_items
        ]
        logger.debug(
            "detect finished: spans=%d answer_tokens=%d input_tokens=%d groups=%d "
            "truncated=%s scoring_ms=%.1f typed=%s typing_ms=%.1f",
            len(spans),
            len(tokens),
            units["input_tokens"],
            len(scored),
            truncated,
            scoring_ms,
            self.typer is not None,
            typing_ms,
        )
        token_scores = (
            [
                TokenScore(
                    token=t["token"],
                    start=t["start"],
                    end=t["end"],
                    score=t["prob"],
                    flagged=t["pred"] == 1,
                )
                for t in tokens
                # Special tokens such as the trailing [SEP] are zero-width: not answer tokens.
                if t["end"] > t["start"]
            ]
            if include_tokens
            else None
        )
        warnings = (
            [
                ResultWarning(
                    code="input_truncated",
                    message=(
                        "The input exceeded max_length tokens and was cut; raise max_length "
                        "or pass sources so passages are split into groups."
                    ),
                )
            ]
            if truncated
            else []
        )
        usage = [Usage(model_id=self.model_id, model_version=self.model_version, units=units)]
        if self.typer is not None:
            # The typer meters the spans it was given, before ``min_score`` filters them.
            usage.append(
                Usage(
                    model_id=getattr(self.typer, "model_id", None),
                    model_version=getattr(self.typer, "model_version", None),
                    units={"spans": len(span_items)},
                )
            )
        return Detection(spans=spans, tokens=token_scores, usage=usage, warnings=warnings)

    @staticmethod
    def _typer_context(request: DetectRequest) -> str:
        """Return the text the typer reads with the answer: joined sources, or the prompt."""
        if isinstance(request, SourcesInput):
            return "\n".join(chunk.text for chunk in request.sources)
        return request.prompt

    def _score_tokens(self, prompt: str, answer: str) -> tuple[list[dict], int, int | None]:
        """Score the answer tokens of one (prompt, answer) pair.

        :param prompt: the prompt string.
        :param answer: the answer string.
        :returns: the answer tokens as dicts with ``token``, ``pred``, ``prob`` and the answer
            offsets ``start`` and ``end``; the number of tokens fed to the model; and the
            untruncated token count when the input was cut, else ``None``.
        """
        return self._scored_tokens(prompt, answer, *self._run_model(prompt, answer))

    def _score_tokens_batch(
        self, prompts: list[str], answers: list[str]
    ) -> list[tuple[list[dict], int, int | None]]:
        """Score the answer tokens of several pairs in one padded forward pass.

        :param prompts: the prompt strings.
        :param answers: the answer strings, parallel to ``prompts``.
        :returns: per pair, what :meth:`_score_tokens` returns for it.
        """
        scored = []
        outputs = self._run_model_batch(prompts, answers)
        for prompt, answer, (input_ids, probabilities, offsets, _) in zip(
            prompts, answers, outputs, strict=True
        ):
            # The first answer token is located as in the single path, so results are equal.
            answer_tokens = self.tokenizer(answer, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].shape[1]
            answer_start_token = int(input_ids.size(0)) - answer_tokens - 1
            scored.append(
                self._scored_tokens(
                    prompt, answer, input_ids, probabilities, offsets, answer_start_token
                )
            )
        return scored

    def _scored_tokens(
        self,
        prompt: str,
        answer: str,
        input_ids: torch.Tensor,
        probabilities: torch.Tensor,
        offsets: torch.Tensor,
        answer_start_token: int,
    ) -> tuple[list[dict], int, int | None]:
        """Turn the model output of one pair into what :meth:`_score_tokens` returns.

        :param prompt: the prompt string.
        :param answer: the answer string.
        :param input_ids: token ids of the pair, shape ``(seq_len,)``.
        :param probabilities: softmax probabilities, shape ``(seq_len, num_labels)``.
        :param offsets: character offsets, shape ``(seq_len, 2)``.
        :param answer_start_token: index of the first answer token.
        """
        seq_len = int(input_ids.size(0))
        logger.debug(
            "scored one group: input_tokens=%d answer_tokens=%d max_length=%d",
            seq_len,
            seq_len - answer_start_token,
            self.max_length,
        )
        full_tokens = None
        if seq_len >= self.max_length:
            # Count only at the limit: an input that fits exactly is not truncated.
            total = self.tokenizer(prompt, answer, add_special_tokens=True, return_tensors="pt")[
                "input_ids"
            ].shape[1]
            if total > self.max_length:
                full_tokens = int(total)
        token_preds = torch.argmax(probabilities, dim=-1)
        if answer_start_token < offsets.size(0):
            answer_char_offset = offsets[answer_start_token][0].item()
        else:
            answer_char_offset = 0
        tokens: list[dict] = []
        for i in range(answer_start_token, seq_len):
            start, end = offsets[i].tolist()
            if start == end:  # special token such as the trailing [SEP]: zero-width at the end
                start = end = len(answer)
            else:
                start, end = start - answer_char_offset, end - answer_char_offset
            tokens.append(
                {
                    "token": self.tokenizer.decode([input_ids[i]]),
                    "pred": token_preds[i].item(),
                    "prob": probabilities[i, 1].item(),
                    "start": start,
                    "end": end,
                }
            )
        return tokens, seq_len, full_tokens

    @staticmethod
    def _max_aggregate(all_token_results: list[list[dict]]) -> list[dict]:
        """Aggregate per-group token results: the probability of a token is the ``max()``.

        A token is flagged when the aggregated probability is at least 0.5. Other keys (such as
        offsets) come from the first group, whose answer offsets equal those of every group.

        :param all_token_results: one token list per group.
        :returns: the aggregated token list.
        """
        aggregated: list[dict] = []
        for tok_idx in range(len(all_token_results[0])):
            max_prob = max(
                chunk_result[tok_idx]["prob"]
                for chunk_result in all_token_results
                if tok_idx < len(chunk_result)
            )
            aggregated.append(
                {
                    **all_token_results[0][tok_idx],
                    "pred": 1 if max_prob >= 0.5 else 0,
                    "prob": max_prob,
                }
            )
        return aggregated

    @staticmethod
    def _spans_from_scored_tokens(tokens: list[dict], answer: str) -> list[dict]:
        """Merge consecutive flagged tokens into spans with the highest probability as score.

        :param tokens: token dicts with ``pred``, ``prob``, ``start`` and ``end``.
        :param answer: the answer string.
        :returns: span dicts with ``start``, ``end``, ``confidence`` and ``text``.
        """
        spans: list[dict] = []
        current: dict | None = None
        for token in tokens:
            if token["start"] == token["end"]:
                continue
            if token["pred"] == 1:
                if current is None:
                    current = {
                        "start": token["start"],
                        "end": token["end"],
                        "confidence": token["prob"],
                    }
                else:
                    current["end"] = token["end"]
                    current["confidence"] = max(current["confidence"], token["prob"])
            elif current is not None:
                current["text"] = answer[current["start"] : current["end"]]
                spans.append(current)
                current = None
        if current is not None:
            current["text"] = answer[current["start"] : current["end"]]
            spans.append(current)
        return spans

    # ------------------------------------------------------------------
    # Chunking helpers
    # ------------------------------------------------------------------

    def _group_passages_into_chunks(
        self,
        context: list[str],
        question: str | None,
        answer: str,
    ) -> list[list[str]]:
        """Group passages into chunks so each chunk, wrapped in the full instruction template, fits in ``max_length``.

        Preserves the instruction template (question, "Bear in mind...", etc.)
        in every chunk by working at the passage level instead of slicing raw
        tokens.

        :param context: List of passage strings.
        :param question: Original question (``None`` for summarisation).
        :param answer: The answer string (token budget is reserved).
        :returns: List of passage groups.  Each group will be formatted into a
            complete prompt via ``PromptUtils.format_context``.
        """
        answer_tokens = self.tokenizer(answer, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].shape[1]
        # Total prompt-token budget: max_length minus answer tokens minus 3 special tokens
        total_budget = self.max_length - answer_tokens - 3

        # Fast path: check whether all passages fit in one prompt.
        full_prompt = PromptUtils.format_context(context, question, self.lang)
        full_prompt_tokens = self.tokenizer(
            full_prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].shape[1]

        if full_prompt_tokens <= total_budget:
            return [context]

        # Measure instruction overhead (everything except the passage content).
        minimal_prompt = PromptUtils.format_context([""], question, self.lang)
        instruction_overhead = self.tokenizer(
            minimal_prompt, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].shape[1]

        passage_budget = total_budget - instruction_overhead
        if passage_budget <= 0:
            # Instructions + answer alone exceed max_length; single group, will truncate.
            return [context]

        # Tokenize each formatted passage line to get its token count.
        p_word = LANG_TO_PASSAGE[self.lang]
        passage_token_counts: list[int] = []
        for i, passage in enumerate(context):
            line = f"{p_word} {i + 1}: {passage}"
            tok_count = self.tokenizer(line, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].shape[1]
            passage_token_counts.append(tok_count)

        # Greedily group passages into buckets that fit within the budget.
        groups: list[list[str]] = []
        current_group: list[str] = []
        current_tokens = 0

        for passage, tok_count in zip(context, passage_token_counts):
            # +1 for the newline separator between passages
            effective = tok_count + (1 if current_group else 0)
            if current_tokens + effective > passage_budget and current_group:
                groups.append(current_group)
                current_group = []
                current_tokens = 0
                effective = tok_count
            current_group.append(passage)
            current_tokens += effective

        if current_group:
            groups.append(current_group)

        return groups if groups else [context]

    # ------------------------------------------------------------------
    # Shared decoding helper for single-example and batched prediction
    # ------------------------------------------------------------------

    def _decode_sample(
        self,
        input_ids: torch.Tensor,
        probabilities: torch.Tensor,
        offsets: torch.Tensor,
        answer_start_token: int,
        answer: str,
        output_format: str,
    ) -> list:
        """Decode one sample's model outputs into tokens or spans.

        Shared by the single-example and batched prediction paths so both produce
        identical output for the same inputs. All tensors must already be trimmed
        to this sample's true (unpadded) sequence length.

        :param input_ids: Token ids for the sample, shape ``(seq_len,)``.
        :param probabilities: Softmax probabilities, shape ``(seq_len, num_labels)``.
        :param offsets: Character offset mapping, shape ``(seq_len, 2)``.
        :param answer_start_token: Index of the first answer-side token.
        :param answer: The original answer string.
        :param output_format: ``"tokens"`` or ``"spans"``.
        """
        token_preds = torch.argmax(probabilities, dim=-1)

        if output_format == "tokens":
            token_probs: list[dict] = []
            for i in range(answer_start_token, input_ids.size(0)):
                token_probs.append(
                    {
                        "token": self.tokenizer.decode([input_ids[i]]),
                        "pred": token_preds[i].item(),
                        "prob": probabilities[i, 1].item(),
                    }
                )
            return token_probs

        # output_format == "spans"
        if answer_start_token < offsets.size(0):
            answer_char_offset = offsets[answer_start_token][0].item()
        else:
            answer_char_offset = 0

        spans: list[dict] = []
        current_span: dict | None = None

        for i in range(answer_start_token, token_preds.size(0)):
            token_start, token_end = offsets[i].tolist()
            if token_start == token_end:
                continue

            rel_start = token_start - answer_char_offset
            rel_end = token_end - answer_char_offset

            is_hallucination = token_preds[i].item() == 1
            confidence = probabilities[i, 1].item() if is_hallucination else 0.0

            if is_hallucination:
                if current_span is None:
                    current_span = {"start": rel_start, "end": rel_end, "confidence": confidence}
                else:
                    current_span["end"] = rel_end
                    current_span["confidence"] = max(current_span["confidence"], confidence)
            else:
                if current_span is not None:
                    current_span["text"] = answer[current_span["start"] : current_span["end"]]
                    spans.append(current_span)
                    current_span = None

        if current_span is not None:
            current_span["text"] = answer[current_span["start"] : current_span["end"]]
            spans.append(current_span)

        return spans

    # ------------------------------------------------------------------
    # Single-chunk prediction (the original _predict logic)
    # ------------------------------------------------------------------

    def _predict_single(self, prompt: str, answer: str, output_format: str) -> list:
        """Run prediction on a single (prompt, answer) pair that fits in ``max_length``.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: ``"tokens"`` or ``"spans"``.
        """
        input_ids, probabilities, offsets, answer_start_token = self._run_model(prompt, answer)
        return self._decode_sample(
            input_ids, probabilities, offsets, answer_start_token, answer, output_format
        )

    def _run_model(self, prompt: str, answer: str) -> tuple[torch.Tensor, torch.Tensor, Any, int]:
        """Tokenize one (prompt, answer) pair and run the model on it.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :returns: token ids ``(seq_len,)``, probabilities ``(seq_len, num_labels)``, character
            offsets and the index of the first answer token.
        """
        encoding, _, offsets, answer_start_token = HallucinationDataset.prepare_tokenized_input(
            self.tokenizer, prompt, answer, self.max_length
        )

        encoding = {
            key: value.to(self.device)
            for key, value in encoding.items()
            if key in ["input_ids", "attention_mask", "labels"]
        }

        with torch.no_grad():
            outputs = self.model(**encoding)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        return encoding["input_ids"][0], probabilities, offsets, answer_start_token

    # ------------------------------------------------------------------
    # Batched prediction with one forward pass for many pairs
    # ------------------------------------------------------------------

    def _run_model_batch(
        self, prompts: list[str], answers: list[str]
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        """Tokenize the pairs as one right-padded batch and run the model on it once.

        :param prompts: the prompt strings.
        :param answers: the answer strings, parallel to ``prompts``.
        :returns: per pair, in input order and cut to the pair's own length: token ids
            ``(seq_len,)``, probabilities ``(seq_len, num_labels)``, character offsets and the
            index of the first token of the answer sequence.
        """
        batch = self.tokenizer(
            prompts,
            answers,
            truncation="only_first",
            max_length=self.max_length,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        offsets = batch.pop("offset_mapping")
        seq_lens = batch["attention_mask"].sum(dim=1)

        model_inputs = {
            key: value.to(self.device)
            for key, value in batch.items()
            if key in ["input_ids", "attention_mask"]
        }

        with torch.no_grad():
            outputs = self.model(**model_inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)

        results = []
        for i in range(len(answers)):
            seq_len = int(seq_lens[i].item())
            answer_start_token = next(
                (idx for idx, seq_id in enumerate(batch.sequence_ids(i)) if seq_id == 1),
                seq_len - 1,
            )
            results.append(
                (
                    batch["input_ids"][i, :seq_len],
                    probabilities[i, :seq_len],
                    offsets[i, :seq_len],
                    answer_start_token,
                )
            )
        return results

    def _predict_batch(
        self, prompts: list[str], answers: list[str], output_format: str
    ) -> list[list]:
        """Tokenize ``prompts``/``answers`` as one padded batch and score them in one pass.

        :param prompts: Prompt strings for this batch (already length-validated
            against ``answers`` by the caller).
        :param answers: Answer strings for this batch.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :returns: One prediction list per (prompt, answer) pair, in input order.
        """
        with self._get_scoring_lock():
            if self.tokenizer.padding_side != "right":
                raise ValueError(
                    "TransformerDetector batched inference requires a right-padding tokenizer."
                )

            results: list[list] = []
            outputs = self._run_model_batch(prompts, answers)
            for i, (answer, output) in enumerate(zip(answers, outputs, strict=True)):
                input_ids, probabilities, offsets, answer_start_token = output
                seq_len = int(input_ids.size(0))
                if seq_len >= self.max_length:
                    logger.warning(
                        "predict_prompt_batch: item %d (%d tokens) reached "
                        "max length (%d) and may have been truncated.",
                        i,
                        seq_len,
                        self.max_length,
                    )

                results.append(
                    self._decode_sample(
                        input_ids, probabilities, offsets, answer_start_token, answer, output_format
                    )
                )
            return results

        # ------------------------------------------------------------------
        # Multi-chunk prediction with max() aggregation
        # ------------------------------------------------------------------

    def _predict_chunked(self, chunk_prompts: list[str], answer: str, output_format: str) -> list:
        """Run prediction over multiple context chunks and aggregate with ``max()``.

        For each answer token, the hallucination probability is the maximum
        across all chunks.  This is conservative: a token is only considered
        supported if *every* chunk considers it supported.

        :param chunk_prompts: Context chunk strings.
        :param answer: The answer string (same for all chunks).
        :param output_format: ``"tokens"`` or ``"spans"``.
        """
        all_token_results: list[list[dict]] = []
        for chunk in chunk_prompts:
            tokens = self._predict_single(chunk, answer, output_format="tokens")
            all_token_results.append(tokens)

        aggregated = self._max_aggregate(all_token_results)

        if output_format == "tokens":
            return aggregated

        # For "spans", build character spans from the aggregated token predictions.
        # Use offset mapping from the first chunk (answer offsets are identical
        # across chunks because BERT tokenizers process segments independently).
        return self._build_spans_from_tokens(aggregated, chunk_prompts[0], answer)

    def _build_spans_from_tokens(
        self, token_results: list[dict], prompt: str, answer: str
    ) -> list[dict]:
        """Convert aggregated token predictions into character-level spans.

        Uses the tokenizer offset mapping from encoding *(prompt, answer)* to
        get precise character positions within the answer.

        :param token_results: Per-answer-token dicts with ``token``, ``pred``, ``prob``.
        :param prompt: A context prompt string (used to get answer offset mapping).
        :param answer: The original answer string.
        :returns: List of span dicts with ``start``, ``end``, ``confidence``, ``text``.
        """
        _, _, offsets, answer_start_token = HallucinationDataset.prepare_tokenized_input(
            self.tokenizer, prompt, answer, self.max_length
        )

        if answer_start_token < offsets.size(0):
            answer_char_offset = offsets[answer_start_token][0].item()
        else:
            answer_char_offset = 0

        # Collect answer-region offsets (skip special tokens with zero length).
        answer_offsets: list[tuple[int, int]] = []
        for i in range(answer_start_token, offsets.size(0)):
            s, e = offsets[i].tolist()
            if s == e:
                continue
            answer_offsets.append((s - answer_char_offset, e - answer_char_offset))

        spans: list[dict] = []
        current_span: dict | None = None

        for tok_idx, tok in enumerate(token_results):
            if tok_idx >= len(answer_offsets):
                break
            rel_start, rel_end = answer_offsets[tok_idx]
            is_hall = tok["pred"] == 1
            confidence = tok["prob"] if is_hall else 0.0

            if is_hall:
                if current_span is None:
                    current_span = {
                        "start": rel_start,
                        "end": rel_end,
                        "confidence": confidence,
                    }
                else:
                    current_span["end"] = rel_end
                    current_span["confidence"] = max(current_span["confidence"], confidence)
            else:
                if current_span is not None:
                    current_span["text"] = answer[current_span["start"] : current_span["end"]]
                    spans.append(current_span)
                    current_span = None

        if current_span is not None:
            current_span["text"] = answer[current_span["start"] : current_span["end"]]
            spans.append(current_span)

        return spans
