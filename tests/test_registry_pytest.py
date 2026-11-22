"""Tests for ``load_detector`` and the model registry: offline, with fake loaders and clients."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import pytest
import torch

import lettucedetect.detectors as detectors
from lettucedetect.detectors import (
    EncoderConfig,
    EncoderDetector,
    GenerativeConfig,
    GenerativeDetector,
    InMemoryCache,
    RAGFactCheckerDetector,
    load_detector,
    registry,
)
from lettucedetect.detectors.errors import ConfigurationError, InvalidInputError
from tests.test_encoder_detector_pytest import _stub_model
from tests.test_llm_judge_detector_pytest import FakeClient

ENCODER_IDS = [
    "KRLabsOrg/lettucedect-base-modernbert-en-v1",
    "KRLabsOrg/lettucedect-large-modernbert-en-v1",
    "KRLabsOrg/lettucedect-v2-mmbert-base",
    "KRLabsOrg/lettucedect-mmbert-base-hu-v1",
    "KRLabsOrg/lettucedect-210m-eurobert-de-v1",
    "KRLabsOrg/lettucedect-210m-eurobert-pl-v1",
    "KRLabsOrg/lettucedect-210m-eurobert-it-v1",
    "KRLabsOrg/lettucedect-210m-eurobert-fr-v1",
    "KRLabsOrg/lettucedect-210m-eurobert-es-v1",
    "KRLabsOrg/lettucedect-210m-eurobert-cn-v1",
    "KRLabsOrg/lettucedect-610m-eurobert-de-v1",
    "KRLabsOrg/tinylettuce-ettin-17m-en",
]
GENERATIVE_IDS = ["KRLabsOrg/lettucedect-v2-qwen-2b"]
ENCODER_ID = ENCODER_IDS[0]
GENERATIVE_ID = GENERATIVE_IDS[0]


@pytest.fixture(autouse=True)
def no_network(tmp_path, monkeypatch):
    """Block the network, clear the API key and point the Hugging Face cache at an empty one."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("network used")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"))


@pytest.fixture
def loaders(local_wordpiece_tokenizer):
    """Fake ``from_pretrained`` loaders; yields both mocks."""
    tok_loader = MagicMock(return_value=local_wordpiece_tokenizer)
    model_loader = MagicMock(return_value=_stub_model())
    with (
        patch("transformers.AutoTokenizer.from_pretrained", tok_loader),
        patch("transformers.AutoModelForTokenClassification.from_pretrained", model_loader),
    ):
        yield tok_loader, model_loader


def test_the_registry_holds_exactly_the_known_ids():
    assert sorted(registry._REGISTRY) == sorted(ENCODER_IDS + GENERATIVE_IDS)


@pytest.mark.parametrize("model_id", ENCODER_IDS)
def test_an_encoder_id_loads_an_encoder_detector(model_id, loaders):
    tok_loader, model_loader = loaders
    detector = load_detector(model_id, device="cpu", model_version="v-test")
    assert type(detector) is EncoderDetector
    assert detector.model_id == model_id
    assert detector.model_version == "v-test"
    assert detector.device == torch.device("cpu")
    tok_loader.assert_called_once_with(model_id)
    model_loader.assert_called_once_with(model_id)


@pytest.mark.parametrize("model_id", GENERATIVE_IDS)
def test_a_generative_id_builds_a_generative_detector_on_the_client(model_id):
    client = FakeClient("{}")
    detector = load_detector(model_id, client=client, model_version="v-test")
    assert type(detector) is GenerativeDetector
    assert detector.client is client
    assert detector.model_id == model_id
    assert detector.model_version == "v-test"
    assert detector.config == GenerativeConfig()


def test_encoder_options_reach_from_pretrained_and_the_config(loaders):
    tok_loader, model_loader = loaders
    typer_class = MagicMock()
    with patch("lettucedetect.detectors.taxonomy_head.TaxonomyTyper", typer_class):
        detector = load_detector(
            ENCODER_ID,
            device="cpu",
            taxonomy_head="org/head",
            include_taxonomy={"Fabricated": "Made up facts"},
            max_length=256,
            lang="de",
            revision="v1",
        )
    typer_class.from_pretrained.assert_called_once_with(
        "org/head",
        device=torch.device("cpu"),
        categories={"Fabricated": "Made up facts"},
        subcategories=None,
    )
    assert detector.typer is typer_class.from_pretrained.return_value
    assert detector.config == EncoderConfig(max_length=256, lang="de")
    tok_loader.assert_called_once_with(ENCODER_ID, revision="v1")
    model_loader.assert_called_once_with(ENCODER_ID, revision="v1")


def test_an_encoder_config_object_is_passed_through(loaders):
    config = EncoderConfig(batch_size=2)
    assert load_detector(ENCODER_ID, config=config).config is config


def test_generative_options_reach_the_config_and_the_constructor():
    cache = InMemoryCache()
    detector = load_detector(
        GENERATIVE_ID,
        client=FakeClient("{}"),
        include_reasoning=True,
        include_taxonomy=["minor", "major"],
        cache=cache,
    )
    assert detector.config == GenerativeConfig(
        include_reasoning=True, include_taxonomy=["minor", "major"]
    )
    assert detector.cache is cache
    config = GenerativeConfig(verify=True)
    assert load_detector(GENERATIVE_ID, client=FakeClient("{}"), config=config).config is config


@pytest.mark.parametrize("model_id", ["KRLabsOrg/lettucedect-v2-qwen-2", "gpt-4.1-mini", ""])
def test_unknown_id_lists_the_known_ids_and_loads_nothing(model_id, loaders):
    tok_loader, model_loader = loaders
    with pytest.raises(InvalidInputError) as exc:
        load_detector(model_id, client=FakeClient("{}"))
    assert exc.value.code == "unknown_model_id"
    assert exc.value.param == "model_id"
    for known in ENCODER_IDS + GENERATIVE_IDS:
        assert known in exc.value.message
    tok_loader.assert_not_called()
    model_loader.assert_not_called()


def test_generative_id_without_a_client_raises_client_required():
    with pytest.raises(ConfigurationError) as exc:
        load_detector(GENERATIVE_ID)
    assert exc.value.code == "client_required"
    assert exc.value.param == "client"


def test_client_for_an_encoder_id_is_rejected_before_loading(loaders):
    tok_loader, model_loader = loaders
    with pytest.raises(ConfigurationError) as exc:
        load_detector(ENCODER_ID, client=FakeClient("{}"))
    assert (exc.value.code, exc.value.param) == ("invalid_config", "client")
    tok_loader.assert_not_called()
    model_loader.assert_not_called()


@pytest.mark.parametrize(
    ("arguments", "param"),
    [
        ({"device": "cpu"}, "device"),
        ({"taxonomy_head": "org/head"}, "taxonomy_head"),
        ({"zero_shot": True}, "zero_shot"),
        ({"temperature": "hot"}, "config.temperature"),
        ({"config": GenerativeConfig(), "verify": True}, "config"),
    ],
)
def test_arguments_a_generative_id_does_not_accept_raise_invalid_config(arguments, param):
    with pytest.raises(ConfigurationError) as exc:
        load_detector(GENERATIVE_ID, client=FakeClient("{}"), **arguments)
    assert (exc.value.code, exc.value.param) == ("invalid_config", param)


def test_invalid_encoder_options_raise_before_loading(loaders):
    tok_loader, _ = loaders
    with pytest.raises(ConfigurationError) as exc:
        load_detector(ENCODER_ID, max_length=0)
    assert (exc.value.code, exc.value.param) == ("invalid_config", "max_length")
    with pytest.raises(ConfigurationError) as exc:
        load_detector(ENCODER_ID, config=EncoderConfig(), lang="de")
    assert (exc.value.code, exc.value.param) == ("invalid_config", "config")
    tok_loader.assert_not_called()


def test_there_is_no_public_way_to_register_and_no_rag_fact_checker_entry():
    assert registry.__all__ == ["load_detector"]
    assert [name for name in vars(registry) if "register" in name.lower()] == []
    assert [name for name in detectors.__all__ if "regist" in name.lower()] == []
    assert RAGFactCheckerDetector not in registry._REGISTRY.values()
    assert set(registry._REGISTRY.values()) == {EncoderDetector, GenerativeDetector}
