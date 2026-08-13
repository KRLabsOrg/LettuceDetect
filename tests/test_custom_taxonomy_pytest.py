"""Tests for custom zero-shot taxonomies across the detector paths (no model, no network)."""

import json

import pytest
import torch

from lettucedetect.detectors.llm import LLMDetector
from lettucedetect.detectors.llm_client import LLMClient, build_generative_schema
from lettucedetect.detectors.taxonomy_head import resolve_taxonomy
from lettucedetect.prompts.generative import build_system_prompt

CUSTOM_CATS = {"pricing_claim": "any statement about price not supported by the context"}
CUSTOM_SUBS = {"currency": "a currency amount not present in the context"}


class TestPromptAndSchemaBuilders:
    """Custom labels reach the generative prompt and schema."""

    def test_system_prompt_defaults_are_frozen(self):
        """Test system prompt defaults are frozen."""
        assert "fabricated" in build_system_prompt() or "unsupported" in build_system_prompt()

    def test_system_prompt_custom_categories_replace(self):
        """Test system prompt custom categories replace."""
        prompt = build_system_prompt(categories=CUSTOM_CATS, subcategories=CUSTOM_SUBS)
        assert "pricing_claim: any statement about price" in prompt
        assert "currency: a currency amount" in prompt
        assert "unsupported_addition" not in prompt

    def test_system_prompt_label_without_description(self):
        """Test system prompt label without description."""
        prompt = build_system_prompt(categories={"bare_label": None})
        assert "- bare_label" in prompt
        assert "bare_label: None" not in prompt

    def test_generative_schema_custom_enums(self):
        """Test generative schema custom enums."""
        schema = build_generative_schema(categories=["pricing_claim"], subcategories=["currency"])
        item = schema["properties"]["hallucinated_spans"]["items"]["properties"]
        assert item["category"]["enum"] == ["pricing_claim"]
        assert item["subcategory"]["enum"] == ["currency"]

    def test_generative_schema_defaults_unchanged(self):
        """Test generative schema defaults unchanged."""
        item = build_generative_schema()["properties"]["hallucinated_spans"]["items"]["properties"]
        assert "unsupported_addition" in item["category"]["enum"]


class CaptureClient(LLMClient):
    """Stub client capturing the system prompt and schema it is called with."""

    def __init__(self):
        """Init capture store."""
        self.calls = []

    def complete(self, system, user, model, temperature, schema=None):
        """Record the call and return an empty span list."""
        self.calls.append({"system": system, "schema": schema})
        return json.dumps({"hallucinated_spans": []})


class TestNativeCustomTaxonomy:
    """Native generative path builds prompt and schema from custom labels."""

    def make_detector(self, tmp_path, **kwargs):
        """Build a native detector with an injected capture client."""
        client = CaptureClient()
        det = LLMDetector(
            model="KRLabsOrg/lettucedect-v2-qwen-2b",
            client=client,
            cache_file=str(tmp_path / "cache.json"),
            **kwargs,
        )
        return det, client

    def test_custom_flat_dict_reaches_prompt_and_schema(self, tmp_path):
        """Test custom flat dict reaches prompt and schema."""
        det, client = self.make_detector(tmp_path, include_taxonomy=CUSTOM_CATS)
        det.predict(context=["ctx"], answer="ans", question="q", output_format="spans")
        call = client.calls[0]
        assert "pricing_claim" in call["system"]
        item = call["schema"]["properties"]["hallucinated_spans"]["items"]["properties"]
        assert item["category"]["enum"] == ["pricing_claim"]
        # no custom subcategories: collapses to unspecified
        assert item["subcategory"]["enum"] == ["unspecified"]

    def test_nested_dict_controls_both(self, tmp_path):
        """Test nested dict controls both."""
        det, client = self.make_detector(
            tmp_path,
            include_taxonomy={"categories": CUSTOM_CATS, "subcategories": CUSTOM_SUBS},
        )
        det.predict(context=["ctx"], answer="ans", question="q", output_format="spans")
        item = client.calls[0]["schema"]["properties"]["hallucinated_spans"]["items"]["properties"]
        assert item["subcategory"]["enum"] == ["currency"]

    def test_default_native_prompt_stays_frozen(self, tmp_path):
        """Test default native prompt stays frozen."""
        det, client = self.make_detector(tmp_path)
        det.predict(context=["ctx"], answer="ans", question="q", output_format="spans")
        assert "pricing_claim" not in client.calls[0]["system"]
        assert "unsupported_addition" in client.calls[0]["system"]

    def test_native_rejects_dead_args(self, tmp_path):
        """Test native rejects dead args."""
        for kwargs in ({"zero_shot": True}, {"fewshot_path": "x.json"}, {"prompt_path": "p.txt"}):
            with pytest.raises(ValueError, match="native"):
                LLMDetector(
                    model="KRLabsOrg/lettucedect-v2-qwen-2b",
                    client=CaptureClient(),
                    cache_file=str(tmp_path / "c.json"),
                    **kwargs,
                )

    def test_non_native_still_accepts_zero_shot(self, tmp_path):
        """Test non native still accepts zero shot."""
        LLMDetector(
            model="gpt-4.1-mini",
            zero_shot=True,
            client=CaptureClient(),
            cache_file=str(tmp_path / "c.json"),
        )


class TestResolveTaxonomy:
    """resolve_taxonomy normalization."""

    def test_bool_is_default(self):
        """Test bool is default."""
        assert resolve_taxonomy(True) == (None, None)
        assert resolve_taxonomy(False) == (None, None)

    def test_flat_dict_is_categories(self):
        """Test flat dict is categories."""
        assert resolve_taxonomy(CUSTOM_CATS) == (CUSTOM_CATS, None)

    def test_nested_dict_controls_both(self):
        """Test nested dict controls both."""
        cats, subs = resolve_taxonomy({"categories": CUSTOM_CATS, "subcategories": CUSTOM_SUBS})
        assert cats == CUSTOM_CATS and subs == CUSTOM_SUBS

    def test_list_selects_frozen_subset(self):
        """Test list selects frozen subset."""
        cats, subs = resolve_taxonomy(["unsupported_addition"])
        assert list(cats) == ["unsupported_addition"] and subs is None


class TestTaxonomyTyperCustomLabels:
    """Custom labels reach the typer's label-embedding call."""

    def test_custom_labels_embedded(self, monkeypatch):
        """Test custom labels embedded."""
        import lettucedetect.detectors.taxonomy_head as th

        embedded_batches = []

        class StubEnc(dict):
            """Attribute-style batch dict."""

            def __init__(self, n):
                """Build minimal tensors for n inputs."""
                ids = torch.ones((n, 4), dtype=torch.long)
                super().__init__(input_ids=ids, attention_mask=torch.ones_like(ids))
                self.input_ids = ids
                self.attention_mask = torch.ones_like(ids)

            def to(self, device):
                """No-op device move."""
                return self

        class StubTokenizer:
            """Records texts passed for embedding."""

            def __call__(self, texts, **kwargs):
                """Tokenize stub."""
                if isinstance(texts, list):
                    embedded_batches.append(list(texts))
                    return StubEnc(len(texts))
                return StubEnc(1)

        class StubOut:
            """Encoder output with a deterministic hidden state."""

            def __init__(self, n):
                """Build hidden state."""
                self.last_hidden_state = torch.randn(n, 4, 8)

        class StubModel:
            """Minimal encoder stub."""

            def __call__(self, input_ids=None, attention_mask=None):
                """Forward stub."""
                return StubOut(input_ids.shape[0])

            def to(self, device):
                """No-op move."""
                return self

            def eval(self):
                """No-op eval."""
                return self

        monkeypatch.setattr(
            th, "AutoTokenizer", type("T", (), {"from_pretrained": lambda *a, **k: StubTokenizer()})
        )
        monkeypatch.setattr(
            th, "AutoModel", type("M", (), {"from_pretrained": lambda *a, **k: StubModel()})
        )
        typer = th.TaxonomyTyper(
            "stub", device="cpu", categories=CUSTOM_CATS, subcategories=CUSTOM_SUBS
        )
        assert typer.cat_names == ["pricing_claim"]
        assert typer.sub_names == ["currency"]
        flat = [x for batch in embedded_batches for x in batch]
        assert any("pricing_claim: any statement about price" in x for x in flat)
        assert any(x.startswith("currency:") for x in flat)
