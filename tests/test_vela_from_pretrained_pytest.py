"""Tests for ``VelaDetector.from_pretrained``: offline with a fake loader, plus one real-model test."""

from __future__ import annotations

import logging
import socket
from unittest.mock import MagicMock, patch

import pytest
import torch

from lettucedetect.detectors.errors import ConfigurationError, UpstreamError
from lettucedetect.detectors.types import Chunk, PromptInput, SourcesInput
from lettucedetect.detectors.vela import VelaConfig, VelaDetector
from tests.test_vela_detector_pytest import FakeVela

REQUEST = PromptInput(prompt="The capital of France is Paris.", answer="It is Berlin.")
SHA_MAIN = "a" * 40
SHA_V1 = "b" * 40
SECRET_PROMPT = "ZEBRA-PROMPT-TEXT"  # noqa: S105
SECRET_ANSWER = "ZEBRA-ANSWER-TEXT Berlin"  # noqa: S105

# The Vela 2.0 0.3B commit the real-model test runs against.
VELA_03B = "vllm-sr/Vela-2.0-0.3B"
VELA_03B_REVISION = "d6f03aa9baca4f9017fa8f8e9c2764d3bed3f409"


@pytest.fixture
def loader():
    """Patch ``AutoModel.from_pretrained`` to return a fake model; yield the model and the mock."""
    model = MagicMock()
    model.system_one.side_effect = FakeVela().system_one
    model_loader = MagicMock(return_value=model)
    with patch("lettucedetect.detectors.vela.AutoModel.from_pretrained", model_loader):
        yield model, model_loader


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


def test_loads_with_trust_remote_code_and_revision_and_moves_the_model(loader):
    model, model_loader = loader
    detector = VelaDetector.from_pretrained(
        "org/name",
        revision="v1",
        device="cpu",
        config=VelaConfig(threshold=0.4),
        local_files_only=True,
        trust_remote_code=False,
    )
    model_loader.assert_called_once_with(
        "org/name", local_files_only=True, revision="v1", trust_remote_code=True
    )
    model.to.assert_called_once_with(torch.device("cpu"))
    model.eval.assert_called_once_with()
    assert detector.model is model
    assert detector.model_id == "org/name"
    assert detector.config.threshold == 0.4
    assert [s.text for s in detector.detect(REQUEST).spans] == ["Berlin"]


def test_hub_id_reports_the_resolved_commit(loader, offline_cache):
    detector = VelaDetector.from_pretrained("org/name", cache_dir=str(offline_cache))
    assert detector.model_version == SHA_MAIN
    assert detector.detect(REQUEST).usage[0].model_version == SHA_MAIN


def test_hub_id_follows_the_requested_revision(loader, offline_cache):
    detector = VelaDetector.from_pretrained(
        "org/name", revision="v1", model_version="mine", cache_dir=str(offline_cache)
    )
    assert detector.model_version == SHA_V1


def test_callers_version_is_the_fallback_when_nothing_is_cached(loader, offline_cache):
    kwargs = {"cache_dir": str(offline_cache)}
    assert VelaDetector.from_pretrained("other/model", **kwargs).model_version is None
    assert (
        VelaDetector.from_pretrained("other/model", model_version="v7", **kwargs).model_version
        == "v7"
    )


def test_local_path_uses_the_callers_version_and_no_cache_lookup(loader, tmp_path):
    boom = MagicMock(side_effect=AssertionError("cache consulted"))
    with patch("lettucedetect.detectors.encoder.try_to_load_from_cache", boom):
        detector = VelaDetector.from_pretrained(str(tmp_path), model_version="2026-11")
    assert detector.model_version == "2026-11"
    boom.assert_not_called()


def test_model_loaded_is_logged_exactly_once_at_info(loader, offline_cache, caplog):
    caplog.set_level(logging.DEBUG)
    detector = VelaDetector.from_pretrained("org/name", device="cpu", cache_dir=str(offline_cache))
    detector.detect(PromptInput(prompt=SECRET_PROMPT, answer=SECRET_ANSWER))
    loud = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert len(loud) == 1
    record = loud[0]
    assert (record.name, record.event) == ("lettucedetect.detectors.vela", "model_loaded")
    assert record.args[:3] == ("org/name", SHA_MAIN, torch.device("cpu"))
    assert SECRET_PROMPT not in caplog.text
    assert SECRET_ANSWER not in caplog.text


def test_old_transformers_raises_missing_dependency(loader):
    _, model_loader = loader
    with (
        patch("lettucedetect.detectors.vela._transformers_version", return_value="4.48.3"),
        pytest.raises(ConfigurationError) as exc,
    ):
        VelaDetector.from_pretrained("org/name")
    assert exc.value.code == "missing_dependency"
    assert "lettucedetect[vela]" in exc.value.message
    model_loader.assert_not_called()


def test_missing_runtime_package_raises_missing_dependency(loader):
    with (
        patch("lettucedetect.detectors.vela.importlib.util.find_spec", return_value=None),
        pytest.raises(ConfigurationError) as exc,
    ):
        VelaDetector.from_pretrained("org/name")
    assert exc.value.code == "missing_dependency"
    assert "lettucedetect[vela]" in exc.value.message


def test_import_error_from_the_model_code_raises_missing_dependency(loader):
    _, model_loader = loader
    model_loader.side_effect = ImportError("requires the following packages: fla")
    with pytest.raises(ConfigurationError) as exc:
        VelaDetector.from_pretrained("org/name")
    assert exc.value.code == "missing_dependency"
    assert "lettucedetect[vela]" in exc.value.message
    assert isinstance(exc.value.__cause__, ImportError)


def test_loader_failure_for_a_hub_id_is_an_upstream_error(loader):
    _, model_loader = loader
    model_loader.side_effect = OSError("not found")
    with pytest.raises(UpstreamError) as exc:
        VelaDetector.from_pretrained("org/name")
    assert exc.value.code == "model_load_failed"


def test_loader_failure_for_a_local_path_is_a_configuration_error(loader, tmp_path):
    _, model_loader = loader
    model_loader.side_effect = OSError("no weights")
    with pytest.raises(ConfigurationError) as exc:
        VelaDetector.from_pretrained(str(tmp_path))
    assert exc.value.code == "model_load_failed"


@pytest.mark.model_download
def test_real_vela_03b_flags_the_unsupported_day():
    detector = VelaDetector.from_pretrained(VELA_03B, revision=VELA_03B_REVISION, device="cpu")
    request = SourcesInput(
        question="When does it ship?",
        sources=[Chunk(id="doc-1", text="Ships Friday.")],
        answer="Ships Monday.",
    )
    result = detector.detect(request, min_score=0.5)
    assert any("Monday" in span.text for span in result.spans)
    for span in result.spans:
        assert request.answer[span.start : span.end] == span.text
    usage = result.usage[0]
    assert (usage.model_id, usage.model_version) == (VELA_03B, VELA_03B_REVISION)
    assert usage.units["sources"] == 1
    assert usage.units["input_tokens"] > 0
    assert result.warnings == []
