"""Tests for ``EncoderDetector.from_pretrained``: offline, with fake loaders."""

from __future__ import annotations

import logging
import socket
from unittest.mock import MagicMock, patch

import pytest
import torch

from lettucedetect.detectors.encoder import EncoderConfig, EncoderDetector
from lettucedetect.detectors.errors import ConfigurationError, UpstreamError
from lettucedetect.detectors.types import PromptInput
from tests.test_encoder_detector_pytest import _stub_model

REQUEST = PromptInput(prompt="The capital of France is Paris.", answer="It is Lyon.")
SHA_MAIN = "a" * 40
SHA_V1 = "b" * 40
SECRET_PROMPT = "ZEBRA-PROMPT-TEXT"  # noqa: S105
SECRET_ANSWER = "ZEBRA-ANSWER-TEXT"  # noqa: S105


@pytest.fixture
def loaders(local_wordpiece_tokenizer):
    """Fake ``from_pretrained`` loaders; yields the model and both mocks."""
    model = _stub_model()
    tok_loader = MagicMock(return_value=local_wordpiece_tokenizer)
    model_loader = MagicMock(return_value=model)
    with (
        patch("transformers.AutoTokenizer.from_pretrained", tok_loader),
        patch("transformers.AutoModelForTokenClassification.from_pretrained", model_loader),
    ):
        yield model, tok_loader, model_loader


@pytest.fixture
def offline_cache(tmp_path, monkeypatch):
    """Build a Hugging Face cache with ``org/name`` at two revisions, with the network blocked."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network used")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    repo = tmp_path / "hub" / "models--org--name"
    for sha in (SHA_MAIN, SHA_V1):
        (repo / "snapshots" / sha).mkdir(parents=True)
        (repo / "snapshots" / sha / "config.json").write_text("{}")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(SHA_MAIN)
    (repo / "refs" / "v1").write_text(SHA_V1)
    return tmp_path / "hub"


def test_builds_a_working_detector_and_passes_device_through(loaders, local_wordpiece_tokenizer):
    model, tok_loader, model_loader = loaders
    detector = EncoderDetector.from_pretrained(
        "org/name",
        device="cpu",
        config=EncoderConfig(max_length=256),
        local_files_only=True,
    )
    tok_loader.assert_called_once_with("org/name", local_files_only=True)
    model_loader.assert_called_once_with("org/name", local_files_only=True)
    model.to.assert_called_once_with(torch.device("cpu"))
    model.eval.assert_called_once_with()
    assert detector.device == torch.device("cpu")
    assert detector.max_length == 256
    assert detector.model_id == "org/name"
    expected = EncoderDetector(
        _stub_model(), local_wordpiece_tokenizer, config=EncoderConfig(max_length=256)
    ).detect(REQUEST)
    assert detector.detect(REQUEST).spans == expected.spans


def test_hub_id_reports_the_resolved_commit(loaders, offline_cache):
    detector = EncoderDetector.from_pretrained("org/name", cache_dir=str(offline_cache))
    assert detector.model_version == SHA_MAIN
    assert detector.detect(REQUEST).usage[0].model_version == SHA_MAIN


def test_hub_id_follows_the_requested_revision(loaders, offline_cache):
    detector = EncoderDetector.from_pretrained(
        "org/name", revision="v1", cache_dir=str(offline_cache)
    )
    assert detector.model_version == SHA_V1


def test_resolved_commit_wins_over_the_callers_version(loaders, offline_cache):
    detector = EncoderDetector.from_pretrained(
        "org/name", model_version="mine", cache_dir=str(offline_cache)
    )
    assert detector.model_version == SHA_MAIN


def test_callers_version_is_the_fallback_when_nothing_is_cached(loaders, offline_cache):
    kwargs = {"cache_dir": str(offline_cache)}
    assert EncoderDetector.from_pretrained("other/model", **kwargs).model_version is None
    assert (
        EncoderDetector.from_pretrained("other/model", model_version="v7", **kwargs).model_version
        == "v7"
    )


def test_local_path_uses_the_callers_version_and_no_cache_lookup(loaders, tmp_path):
    boom = MagicMock(side_effect=AssertionError("cache consulted"))
    with patch("lettucedetect.detectors.encoder.try_to_load_from_cache", boom):
        assert EncoderDetector.from_pretrained(str(tmp_path)).model_version is None
        detector = EncoderDetector.from_pretrained(str(tmp_path), model_version="2024-05")
    assert detector.model_version == "2024-05"
    boom.assert_not_called()


def test_model_loaded_is_logged_exactly_once_at_info(loaders, offline_cache, caplog):
    caplog.set_level(logging.DEBUG)
    detector = EncoderDetector.from_pretrained(
        "org/name", device="cpu", cache_dir=str(offline_cache)
    )
    detector.detect(PromptInput(prompt=SECRET_PROMPT, answer=SECRET_ANSWER))
    loud = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert len(loud) == 1
    record = loud[0]
    assert (record.levelno, record.name, record.event) == (
        logging.INFO,
        "lettucedetect.detectors.encoder",
        "model_loaded",
    )
    assert record.args[:3] == ("org/name", SHA_MAIN, torch.device("cpu"))
    assert SECRET_PROMPT not in caplog.text
    assert SECRET_ANSWER not in caplog.text


def test_building_from_objects_logs_nothing_at_info(local_wordpiece_tokenizer, caplog):
    caplog.set_level(logging.DEBUG)
    EncoderDetector(_stub_model(), local_wordpiece_tokenizer)
    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []


def test_taxonomy_head_loads_the_typer_with_the_device(loaders):
    typer_class = MagicMock()
    with patch("lettucedetect.detectors.taxonomy_head.TaxonomyTyper", typer_class):
        detector = EncoderDetector.from_pretrained(
            "org/name",
            device="cpu",
            taxonomy_head="org/head",
            include_taxonomy={"Fabricated": "Made up facts"},
        )
    typer_class.from_pretrained.assert_called_once_with(
        "org/head",
        device=torch.device("cpu"),
        categories={"Fabricated": "Made up facts"},
        subcategories=None,
    )
    assert detector.typer is typer_class.from_pretrained.return_value


def test_no_taxonomy_head_means_no_typer(loaders):
    typer_class = MagicMock()
    with patch("lettucedetect.detectors.taxonomy_head.TaxonomyTyper", typer_class):
        assert EncoderDetector.from_pretrained("org/name").typer is None
    typer_class.assert_not_called()


def test_bad_taxonomy_spec_fails_before_any_download(loaders):
    _, tok_loader, model_loader = loaders
    with pytest.raises(TypeError):
        EncoderDetector.from_pretrained("org/name", taxonomy_head="org/head", include_taxonomy=5)
    tok_loader.assert_not_called()
    model_loader.assert_not_called()


def test_uncached_hub_model_offline_raises_upstream_error(offline_cache, monkeypatch):
    """With ``HF_HUB_OFFLINE=1`` and nothing cached the loader error is translated."""
    with pytest.raises(UpstreamError) as exc:
        EncoderDetector.from_pretrained("nope/not-cached", cache_dir=str(offline_cache))
    assert (exc.value.code, exc.value.service) == ("model_load_failed", "huggingface")
    assert isinstance(exc.value.__cause__, OSError)


def test_loader_failure_for_a_hub_id_is_an_upstream_error(loaders):
    _, tok_loader, _ = loaders
    tok_loader.side_effect = OSError("secret detail")
    with pytest.raises(UpstreamError) as exc:
        EncoderDetector.from_pretrained("org/name")
    assert exc.value.code == "model_load_failed"
    assert "secret detail" not in str(exc.value)
    assert isinstance(exc.value.__cause__, OSError)


def test_loader_failure_for_a_local_path_is_a_configuration_error(loaders, tmp_path):
    _, _, model_loader = loaders
    model_loader.side_effect = OSError("secret detail")
    with pytest.raises(ConfigurationError) as exc:
        EncoderDetector.from_pretrained(str(tmp_path))
    assert (exc.value.code, exc.value.param) == ("model_load_failed", "model_id")
    assert "secret detail" not in str(exc.value)
    assert isinstance(exc.value.__cause__, OSError)
