"""Label-conditioned taxonomy typing head for the encoder cascade.

A shared mmBERT bi-encoder that assigns a ``category`` and ``subcategory`` to a span the
binary :class:`~lettucedetect.detectors.transformer.TransformerDetector` already found. Each
span is typed by mean-pooling its answer tokens and taking the nearest taxonomy-label
*description* by cosine similarity, so it turns the fast binary encoder into a typed detector
without a generative model. Labels enter only as text, matching the training setup in
``scripts/train_taxonomy_head.py``.
"""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer

from lettucedetect.prompts.generative import CATEGORY_DESCRIPTIONS, SUBCATEGORY_DESCRIPTIONS

_SEP = " [CTX] "  # answer + _SEP + context; must match the taxonomy-head training script


class TaxonomyTyper:
    """Types spans (category + subcategory) via a label-conditioned encoder."""

    def __init__(
        self,
        model_path: str,
        device: torch.device | str | None = None,
        max_length: int = 1024,
        **tok_kwargs: object,
    ) -> None:
        """Load the typing encoder and pre-compute taxonomy-label embeddings.

        :param model_path: Path or HF id of the label-conditioned encoder.
        :param device: Device for inference (defaults to CUDA when available).
        :param max_length: Max tokens for the ``answer + context`` input.
        :param tok_kwargs: Extra arguments for the tokenizer / model loaders.
        """
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
        self.encoder = AutoModel.from_pretrained(model_path, **tok_kwargs).to(self.device).eval()
        self.cat_names = list(CATEGORY_DESCRIPTIONS)
        self.sub_names = list(SUBCATEGORY_DESCRIPTIONS)
        self._cat_vecs = self._label_vecs(self.cat_names, CATEGORY_DESCRIPTIONS)
        self._sub_vecs = self._label_vecs(self.sub_names, SUBCATEGORY_DESCRIPTIONS)

    @torch.no_grad()
    def _label_vecs(self, names: list[str], descs: dict) -> torch.Tensor:
        enc = self.tokenizer(
            [f"{n}: {descs[n]}" for n in names], padding=True, return_tensors="pt"
        ).to(self.device)
        h = self.encoder(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask
        ).last_hidden_state.float()
        m = enc.attention_mask.unsqueeze(-1).float()
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1)

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
