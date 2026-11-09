"""Label-conditioned taxonomy typing head for the encoder cascade.

A shared mmBERT bi-encoder that assigns a ``category`` and ``subcategory`` to a span the
binary :class:`~lettucedetect.detectors.transformer.TransformerDetector` already found. Each
span is typed by mean-pooling its answer tokens and taking the nearest taxonomy-label
*description* by cosine similarity, so it turns the fast binary encoder into a typed detector
without a generative model. Labels enter only as text, matching the training setup in
``scripts/train_taxonomy_head.py``.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from lettucedetect.detectors.encoder import (
    _LOCK_GUARD,
    _log_model_loaded,
    _model_load_error,
    _resolved_commit,
)
from lettucedetect.prompts.generative import CATEGORY_DESCRIPTIONS, SUBCATEGORY_DESCRIPTIONS

logger = logging.getLogger(__name__)

_SEP = " [CTX] "  # answer + _SEP + context; must match the taxonomy-head training script


def resolve_taxonomy(
    include_taxonomy: bool | list | dict,
) -> tuple[dict | None, dict | None]:
    """Normalize an ``include_taxonomy`` argument into (categories, subcategories) dicts.

    ``True``/``False`` -> frozen defaults (None, None); flat ``{name: description}`` dict ->
    custom categories; ``{"categories": {...}, "subcategories": {...}}`` -> full control;
    list of names -> subset of the frozen categories.
    """
    if isinstance(include_taxonomy, bool):
        return None, None
    if isinstance(include_taxonomy, dict):
        if (
            include_taxonomy
            and set(include_taxonomy) <= {"categories", "subcategories"}
            and all(isinstance(v, dict) for v in include_taxonomy.values())
        ):
            return (
                dict(include_taxonomy.get("categories") or {}) or None,
                dict(include_taxonomy.get("subcategories") or {}) or None,
            )
        return dict(include_taxonomy) or None, None
    return {name: CATEGORY_DESCRIPTIONS.get(name) for name in include_taxonomy} or None, None


class TaxonomyTyper:
    """Types spans (category + subcategory) via a label-conditioned encoder.

    Built from an encoder and tokenizer the caller already holds, so construction downloads
    nothing. :meth:`from_pretrained` is where a model name becomes a model.

    Typing runs under one private lock, so concurrent calls on one typer (for example from
    ``adetect`` worker threads, or from several detectors sharing it) run one at a time.

    Example, loading the typing head once at startup and giving it to an encoder::

        from transformers import AutoModel, AutoTokenizer

        head_id = "KRLabsOrg/lettucedect-v2-taxonomy-head"
        tokenizer = AutoTokenizer.from_pretrained(head_id)
        encoder = AutoModel.from_pretrained(head_id).eval()

        typer = TaxonomyTyper(
            encoder,
            tokenizer,
            max_length=1024,
            model_id=head_id,
        )
        detector = EncoderDetector(model, model_tokenizer, typer, model_id=model_id)
        result = detector.detect(request)  # every span has category and subcategory

    The typer reads its device from ``encoder.device``, so move the encoder before building it.
    To let the typer load itself, and record the resolved commit as ``model_version`` and log
    ``model_loaded``, use ``TaxonomyTyper.from_pretrained(head_id, device="cuda")``. Custom
    labels work with both::

        TaxonomyTyper(encoder, tokenizer, categories={"pricing_claim": "a price not in the context"})

    :param encoder: the label-conditioned encoder, ready for inference (it is not moved or
        switched to eval mode).
    :param tokenizer: the encoder's fast tokenizer.
    :param categories: custom ``{name: description}`` category labels; REPLACES the frozen
        set (labels enter only as text, so held-out labels type zero-shot from their
        description). ``None`` keeps the trained taxonomy.
    :param subcategories: custom subcategory labels, same semantics.
    :param max_length: max tokens for the ``answer + context`` input.
    :param model_id: id reported in ``usage``.
    :param model_version: version reported in ``usage``.
    """

    model_id: str | None = None
    model_version: str | None = None

    def __init__(
        self,
        encoder: Any,  # noqa: ANN401
        tokenizer: Any,  # noqa: ANN401
        *,
        categories: dict[str, str] | None = None,
        subcategories: dict[str, str] | None = None,
        max_length: int = 1024,
        model_id: str | None = None,
        model_version: str | None = None,
    ) -> None:
        """Build the typer from loaded objects and pre-compute taxonomy-label embeddings."""
        device = getattr(encoder, "device", None)
        self.device = device if isinstance(device, torch.device) else torch.device("cpu")
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.model_id = model_id
        self.model_version = model_version
        cats = categories if categories is not None else CATEGORY_DESCRIPTIONS
        subs = subcategories if subcategories is not None else SUBCATEGORY_DESCRIPTIONS
        self.cat_names = list(cats)
        self.sub_names = list(subs)
        self._cat_vecs = self._label_vecs(self.cat_names, cats)
        self._sub_vecs = self._label_vecs(self.sub_names, subs)
        logger.debug(
            "taxonomy typer built: model_id=%s categories=%d subcategories=%d max_length=%d "
            "device=%s",
            model_id,
            len(self.cat_names),
            len(self.sub_names),
            max_length,
            self.device,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: torch.device | str | None = None,
        categories: dict[str, str] | None = None,
        subcategories: dict[str, str] | None = None,
        model_version: str | None = None,
        **loader_kwargs: Any,  # noqa: ANN401
    ) -> TaxonomyTyper:
        """Load the encoder and tokenizer for ``model_id`` and build the typer.

        Like :meth:`EncoderDetector.from_pretrained`, this may download, unless the caller set
        ``HF_HUB_OFFLINE`` (this method neither reads nor sets it).

        :param model_id: Hugging Face id or local directory of the label-conditioned encoder.
        :param device: device for the encoder; CUDA when available, else CPU, when ``None``.
        :param categories: custom category labels, see the class docstring.
        :param subcategories: custom subcategory labels.
        :param model_version: version reported in ``usage`` for a local path. For a hub id the
            resolved commit is used and this is only the fallback.
        :param loader_kwargs: passed to both ``from_pretrained`` loaders.
        :raises UpstreamError: (``model_load_failed``) a hub model could not be loaded.
        :raises ConfigurationError: (``model_load_failed``) a local model could not be loaded.
        """
        is_local = Path(model_id).is_dir()
        started = time.perf_counter()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, **loader_kwargs)
            encoder = AutoModel.from_pretrained(model_id, **loader_kwargs)
        except OSError as exc:
            raise _model_load_error(model_id, is_local) from exc
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        encoder = encoder.to(device).eval()
        version = model_version
        if not is_local:
            version = _resolved_commit(model_id, loader_kwargs) or model_version
        typer = cls(
            encoder,
            tokenizer,
            categories=categories,
            subcategories=subcategories,
            model_id=model_id,
            model_version=version,
        )
        typer.device = device
        _log_model_loaded(model_id, version, device, started, logger)
        return typer

    @torch.no_grad()
    def _label_vecs(self, names: list[str], descs: dict) -> torch.Tensor:
        enc = self.tokenizer(
            [f"{n}: {descs[n]}" if descs.get(n) else n for n in names],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        h = self.encoder(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask
        ).last_hidden_state.float()
        m = enc.attention_mask.unsqueeze(-1).float()
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)

    def _get_scoring_lock(self) -> threading.RLock:
        """Return the typing lock, creating it on first use (not in ``__init__``)."""
        lock = vars(self).get("_scoring_lock")
        if lock is None:
            with _LOCK_GUARD:
                lock = vars(self).get("_scoring_lock")
                if lock is None:
                    lock = threading.RLock()
                    self._scoring_lock = lock
        return lock

    @torch.no_grad()
    def type_spans(self, answer: str, context: str, spans: list[dict]) -> list[dict]:
        """Attach ``category`` and ``subcategory`` to each span (offsets relative to ``answer``).

        :param answer: The answer text the spans index into.
        :param context: The grounding context (appended after the answer for the encoder).
        :param spans: Span dicts with ``start``/``end`` (answer-relative); typed in place.
        :returns: The same span list, each with ``category``/``subcategory`` set.
        """
        if not spans:
            return spans
        # The tokenizer and encoder are shared state; everything after this block is local.
        with self._get_scoring_lock():
            enc = self.tokenizer(
                answer + _SEP + (context or ""),
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offs = enc.pop("offset_mapping")[0].tolist()
            h = self.encoder(
                input_ids=enc.input_ids.to(self.device),
                attention_mask=enc.attention_mask.to(self.device),
            ).last_hidden_state.float()[0]
        for sp in spans:
            s, e = sp["start"], sp["end"]
            mask = torch.tensor(
                [1.0 if (a < e and b > s and b > a) else 0.0 for a, b in offs], device=self.device
            )
            if mask.sum() == 0:  # span truncated away -> CLS fallback (matches training)
                mask[0] = 1.0
            v = torch.nn.functional.normalize(
                (h * mask.unsqueeze(-1)).sum(0) / mask.sum().clamp(min=1), dim=-1
            )
            sp["category"] = self.cat_names[int((v @ self._cat_vecs.T).argmax())]
            sp["subcategory"] = self.sub_names[int((v @ self._sub_vecs.T).argmax())]
        return spans
