"""Tests for ``TaxonomyTyper``: built from loaded objects, and ``from_pretrained``, offline."""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from lettucedetect.detectors.encoder import EncoderDetector
from lettucedetect.detectors.errors import ConfigurationError, UpstreamError
from lettucedetect.detectors.taxonomy_head import TaxonomyTyper
from tests.test_encoder_detector_pytest import _stub_model

CATS = {"pricing_claim": "any statement about price not supported by the context"}
SUBS = {"currency": "a currency amount not present in the context"}
SHA_HEAD = "c" * 40
SHA_MAIN = "a" * 40
SECRET_ANSWER = "ZEBRA-ANSWER-TEXT"  # noqa: S105


class _StubEncoder:
    """A label-conditioned encoder with a fixed hidden state."""

    def __init__(self, device: torch.device | None = None) -> None:
        if device is not None:
            self.device = device
        self.calls = 0

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        self.calls += 1
        hidden = torch.arange(input_ids.numel() * 8, dtype=torch.float).reshape(*input_ids.shape, 8)
        return SimpleNamespace(last_hidden_state=hidden)


@pytest.fixture
def block_network(monkeypatch):
    """Fail any test that opens a socket."""

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network used")

    monkeypatch.setattr(socket.socket, "connect", no_network)


@pytest.fixture
def offline_cache(tmp_path, monkeypatch, block_network):
    """Build a Hugging Face cache with the encoder and the head, with the network blocked."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    for repo_name, sha in (("models--org--name", SHA_MAIN), ("models--org--head", SHA_HEAD)):
        repo = tmp_path / "hub" / repo_name
        (repo / "snapshots" / sha).mkdir(parents=True)
        (repo / "snapshots" / sha / "config.json").write_text("{}")
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(sha)
    return tmp_path / "hub"


@pytest.fixture
def head_loaders(local_wordpiece_tokenizer):
    """Fake ``from_pretrained`` loaders for the head; yields the encoder and both mocks."""
    encoder = _StubEncoder()
    tok_loader = MagicMock(return_value=local_wordpiece_tokenizer)
    model_loader = MagicMock(return_value=encoder)
    with (
        patch("transformers.AutoTokenizer.from_pretrained", tok_loader),
        patch("transformers.AutoModel.from_pretrained", model_loader),
    ):
        yield encoder, tok_loader, model_loader


# ----------------------------------------------------------------------
# Construction from loaded objects
# ----------------------------------------------------------------------


def test_construction_downloads_nothing(local_wordpiece_tokenizer, block_network):
    boom = MagicMock(side_effect=AssertionError("loader called"))
    with (
        patch("transformers.AutoTokenizer.from_pretrained", boom),
        patch("transformers.AutoModel.from_pretrained", boom),
    ):
        typer = TaxonomyTyper(_StubEncoder(), local_wordpiece_tokenizer)
    boom.assert_not_called()
    assert typer.cat_names and typer.sub_names


def test_construction_does_not_move_or_switch_the_encoder(local_wordpiece_tokenizer):
    class Spy(_StubEncoder):
        moved = switched = False

        def to(self, device):
            self.moved = True
            return super().to(device)

        def eval(self):
            self.switched = True
            return super().eval()

    encoder = Spy()
    TaxonomyTyper(encoder, local_wordpiece_tokenizer, categories=CATS, subcategories=SUBS)
    assert (encoder.moved, encoder.switched) == (False, False)


def test_constructor_records_ids_and_takes_the_device_from_the_encoder(local_wordpiece_tokenizer):
    typer = TaxonomyTyper(
        _StubEncoder(torch.device("cpu")),
        local_wordpiece_tokenizer,
        categories=CATS,
        subcategories=SUBS,
        max_length=64,
        model_id="org/head",
        model_version="v3",
    )
    assert (typer.model_id, typer.model_version, typer.max_length) == ("org/head", "v3", 64)
    assert typer.device == torch.device("cpu")


def test_a_device_less_encoder_falls_back_to_cpu(local_wordpiece_tokenizer):
    typer = TaxonomyTyper(_StubEncoder(), local_wordpiece_tokenizer)
    assert typer.device == torch.device("cpu")
    assert (typer.model_id, typer.model_version) == (None, None)


def test_custom_labels_replace_the_frozen_set(local_wordpiece_tokenizer):
    typer = TaxonomyTyper(
        _StubEncoder(), local_wordpiece_tokenizer, categories=CATS, subcategories=SUBS
    )
    assert (typer.cat_names, typer.sub_names) == (["pricing_claim"], ["currency"])


def test_type_spans_attaches_a_category_and_subcategory(local_wordpiece_tokenizer):
    typer = TaxonomyTyper(
        _StubEncoder(), local_wordpiece_tokenizer, categories=CATS, subcategories=SUBS
    )
    spans = [{"start": 0, "end": 3, "confidence": 0.9, "text": "the"}]
    typed = typer.type_spans("the capital", "paris", spans)
    assert typed is spans
    assert (typed[0]["category"], typed[0]["subcategory"]) == ("pricing_claim", "currency")
    assert typer.type_spans("the capital", "paris", []) == []


# ----------------------------------------------------------------------
# from_pretrained
# ----------------------------------------------------------------------


def test_from_pretrained_builds_a_typer_and_passes_device_and_kwargs(head_loaders):
    encoder, tok_loader, model_loader = head_loaders
    typer = TaxonomyTyper.from_pretrained(
        "org/head", device="cpu", categories=CATS, subcategories=SUBS, local_files_only=True
    )
    tok_loader.assert_called_once_with("org/head", local_files_only=True)
    model_loader.assert_called_once_with("org/head", local_files_only=True)
    assert typer.encoder is encoder
    assert typer.device == torch.device("cpu")
    assert encoder.device == torch.device("cpu")
    assert (typer.cat_names, typer.sub_names) == (["pricing_claim"], ["currency"])
    assert typer.model_id == "org/head"


def test_hub_id_reports_the_resolved_commit(head_loaders, offline_cache):
    typer = TaxonomyTyper.from_pretrained("org/head", cache_dir=str(offline_cache))
    assert typer.model_version == SHA_HEAD


def test_resolved_commit_wins_over_the_callers_version(head_loaders, offline_cache):
    typer = TaxonomyTyper.from_pretrained(
        "org/head", model_version="mine", cache_dir=str(offline_cache)
    )
    assert typer.model_version == SHA_HEAD


def test_callers_version_is_the_fallback_when_nothing_is_cached(head_loaders, offline_cache):
    kwargs = {"cache_dir": str(offline_cache)}
    assert TaxonomyTyper.from_pretrained("other/head", **kwargs).model_version is None
    assert (
        TaxonomyTyper.from_pretrained("other/head", model_version="v7", **kwargs).model_version
        == "v7"
    )


def test_local_path_uses_the_callers_version_and_no_cache_lookup(head_loaders, tmp_path):
    boom = MagicMock(side_effect=AssertionError("cache consulted"))
    with patch("lettucedetect.detectors.encoder.try_to_load_from_cache", boom):
        assert TaxonomyTyper.from_pretrained(str(tmp_path)).model_version is None
        typer = TaxonomyTyper.from_pretrained(str(tmp_path), model_version="2024-05")
    assert typer.model_version == "2024-05"
    boom.assert_not_called()


def test_model_loaded_is_logged_exactly_once_at_info(head_loaders, offline_cache, caplog):
    caplog.set_level(logging.DEBUG)
    typer = TaxonomyTyper.from_pretrained("org/head", device="cpu", cache_dir=str(offline_cache))
    typer.type_spans(SECRET_ANSWER, "context", [{"start": 0, "end": 4}])
    loud = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert len(loud) == 1
    record = loud[0]
    assert (record.levelno, record.name, record.event) == (
        logging.INFO,
        "lettucedetect.detectors.taxonomy_head",
        "model_loaded",
    )
    assert record.args[:3] == ("org/head", SHA_HEAD, torch.device("cpu"))
    assert SECRET_ANSWER not in caplog.text


def test_building_from_objects_logs_nothing_at_info(local_wordpiece_tokenizer, caplog):
    caplog.set_level(logging.DEBUG)
    TaxonomyTyper(_StubEncoder(), local_wordpiece_tokenizer)
    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []


def test_loader_failure_for_a_hub_id_is_an_upstream_error(head_loaders):
    _, tok_loader, _ = head_loaders
    tok_loader.side_effect = OSError("secret detail")
    with pytest.raises(UpstreamError) as exc:
        TaxonomyTyper.from_pretrained("org/head")
    assert (exc.value.code, exc.value.service) == ("model_load_failed", "huggingface")
    assert "secret detail" not in str(exc.value)
    assert isinstance(exc.value.__cause__, OSError)


def test_loader_failure_for_a_local_path_is_a_configuration_error(head_loaders, tmp_path):
    _, _, model_loader = head_loaders
    model_loader.side_effect = OSError("secret detail")
    with pytest.raises(ConfigurationError) as exc:
        TaxonomyTyper.from_pretrained(str(tmp_path))
    assert (exc.value.code, exc.value.param) == ("model_load_failed", "model_id")
    assert "secret detail" not in str(exc.value)
    assert isinstance(exc.value.__cause__, OSError)


# ----------------------------------------------------------------------
# Offline behaviour matches the encoder's
# ----------------------------------------------------------------------


def test_uncached_hub_head_offline_raises_upstream_error(offline_cache):
    """With ``HF_HUB_OFFLINE=1`` and nothing cached the loader error is translated."""
    with pytest.raises(UpstreamError) as exc:
        TaxonomyTyper.from_pretrained("nope/not-cached", cache_dir=str(offline_cache))
    assert (exc.value.code, exc.value.service) == ("model_load_failed", "huggingface")
    assert isinstance(exc.value.__cause__, OSError)


def test_from_pretrained_neither_reads_nor_sets_the_offline_variable(head_loaders, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    TaxonomyTyper.from_pretrained("org/head")
    assert "HF_HUB_OFFLINE" not in os.environ
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    TaxonomyTyper.from_pretrained("org/head")
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_a_typed_encoder_load_follows_the_cache_and_skips_the_revision(
    local_wordpiece_tokenizer, offline_cache
):
    """Both models load from the caller's cache; ``revision`` names only the encoder's repo."""
    tok_loader = MagicMock(return_value=local_wordpiece_tokenizer)
    encoder_loader = MagicMock(return_value=_stub_model())
    head_loader = MagicMock(return_value=_StubEncoder())
    with (
        patch("transformers.AutoTokenizer.from_pretrained", tok_loader),
        patch("transformers.AutoModelForTokenClassification.from_pretrained", encoder_loader),
        patch("transformers.AutoModel.from_pretrained", head_loader),
    ):
        detector = EncoderDetector.from_pretrained(
            "org/name",
            device="cpu",
            taxonomy_head="org/head",
            include_taxonomy={"categories": CATS, "subcategories": SUBS},
            cache_dir=str(offline_cache),
            revision="main",
            token="t",  # noqa: S106
        )
    head_loader.assert_called_once_with("org/head", cache_dir=str(offline_cache), token="t")  # noqa: S106
    encoder_loader.assert_called_once_with(
        "org/name",
        cache_dir=str(offline_cache),
        revision="main",
        token="t",  # noqa: S106
    )
    assert detector.model_version == SHA_MAIN
    assert detector.typer.model_version == SHA_HEAD
    assert (detector.typer.cat_names, detector.typer.sub_names) == (["pricing_claim"], ["currency"])


@pytest.mark.parametrize(
    ("include_taxonomy", "expected"),
    [
        (True, (None, None)),
        (["unsupported_addition"], (["unsupported_addition"], None)),
        (CATS, (["pricing_claim"], None)),
        ({"categories": CATS, "subcategories": SUBS}, (["pricing_claim"], ["currency"])),
    ],
    ids=["true", "list", "flat-dict", "nested-dict"],
)
def test_all_four_taxonomy_forms_reach_the_typer(include_taxonomy, expected, head_loaders):
    typer_class = MagicMock()
    with (
        patch("lettucedetect.detectors.taxonomy_head.TaxonomyTyper", typer_class),
        patch("transformers.AutoModelForTokenClassification.from_pretrained") as model_loader,
    ):
        model_loader.return_value = _stub_model()
        EncoderDetector.from_pretrained(
            "org/name", device="cpu", taxonomy_head="org/head", include_taxonomy=include_taxonomy
        )
    kwargs = typer_class.from_pretrained.call_args.kwargs
    cats = None if kwargs["categories"] is None else list(kwargs["categories"])
    subs = None if kwargs["subcategories"] is None else list(kwargs["subcategories"])
    assert (cats, subs) == expected


# ----------------------------------------------------------------------
# Concurrency
# ----------------------------------------------------------------------


def test_concurrent_type_spans_calls_run_one_at_a_time(local_wordpiece_tokenizer):
    """Worker threads (``adetect``) or detectors sharing one typer never overlap in the encoder."""
    state = {"active": 0, "max_active": 0}
    guard = threading.Lock()

    class Slow(_StubEncoder):
        def __call__(self, input_ids, attention_mask):
            with guard:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.02)
            try:
                return super().__call__(input_ids, attention_mask)
            finally:
                with guard:
                    state["active"] -= 1

    typer = TaxonomyTyper(Slow(), local_wordpiece_tokenizer, categories=CATS, subcategories=SUBS)
    state["max_active"] = 0  # ignore the label-embedding calls made at construction

    def run() -> list[dict]:
        return typer.type_spans("the capital", "paris", [{"start": 0, "end": 3}])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [f.result() for f in [pool.submit(run) for _ in range(8)]]
    assert state["max_active"] == 1
    assert all(r[0]["category"] == "pricing_claim" for r in results)


def test_the_typing_lock_is_per_typer_and_reentrant(local_wordpiece_tokenizer):
    first = TaxonomyTyper(_StubEncoder(), local_wordpiece_tokenizer)
    second = TaxonomyTyper(_StubEncoder(), local_wordpiece_tokenizer)
    assert first._get_scoring_lock() is first._get_scoring_lock()
    assert first._get_scoring_lock() is not second._get_scoring_lock()
    with first._get_scoring_lock(), first._get_scoring_lock():
        pass
