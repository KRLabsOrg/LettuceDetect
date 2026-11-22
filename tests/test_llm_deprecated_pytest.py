"""``LLMDetector`` is a deprecated adapter over the judge and the generative detector: offline."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

import lettucedetect
from lettucedetect.detectors import _deprecation as dep
from lettucedetect.detectors._deprecation import LettuceDetectDeprecationWarning
from lettucedetect.detectors.errors import ConfigurationError
from lettucedetect.detectors.factory import make_detector
from lettucedetect.detectors.file_cache import FileCache
from lettucedetect.detectors.generative import GenerativeDetector
from lettucedetect.detectors.llm import LLMDetector
from lettucedetect.detectors.llm_judge import LLMJudgeDetector
from lettucedetect.models.inference import HallucinationDetector
from tests.test_golden_llm_pytest import CASES, PLAIN, FakeClient

WARNING = LettuceDetectDeprecationWarning
NATIVE_MODEL = "KRLabsOrg/lettucedect-v2-qwen-2b"
MAKE_CLIENT = "lettucedetect.detectors.llm.make_llm_client"


def _message(name: str, replacement: str) -> str:
    return (
        f"{name} is deprecated since {dep.DEPRECATED_SINCE} and will be removed in "
        f"{dep.REMOVED_IN}. Use {replacement}."
    )


CLASS_MESSAGE = _message("LLMDetector", "LLMJudgeDetector or GenerativeDetector")
NO_CLIENT_MESSAGE = _message(
    "LLMDetector without the 'client' argument",
    "client=OpenAIClient(...) or client=BedrockClient(...)",
)
PROVIDER_MESSAGE = _message("The 'provider' argument", "the 'client' argument")
CLIENT_KWARGS_MESSAGE = _message("The 'client_kwargs' argument", "the 'client' argument")
CACHE_FILE_MESSAGE = _message(
    "The 'cache_file' argument", "cache=FileCache(path) on LLMJudgeDetector or GenerativeDetector"
)


def _construct(**kwargs) -> LLMDetector:
    """Build an LLMDetector from a test module: this warns."""
    kwargs.setdefault("client", FakeClient(PLAIN))
    with pytest.warns(WARNING):
        return LLMDetector(**kwargs)


# -- deprecation --------------------------------------------------------------------------


def test_construction_warns_once_and_names_both_replacements():
    with pytest.warns(WARNING) as caught:
        LLMDetector(client=FakeClient(PLAIN))
    assert [str(w.message) for w in caught] == [CLASS_MESSAGE]
    assert caught[0].filename == __file__


def test_each_deprecated_use_warns_once():
    client = FakeClient(PLAIN)
    with patch(MAKE_CLIENT, return_value=client) as make_client, pytest.warns(WARNING) as caught:
        detector = LLMDetector(provider="bedrock", cache_file="unused.json", region_name="eu")
    assert [str(w.message) for w in caught] == [
        CLASS_MESSAGE,
        PROVIDER_MESSAGE,
        CLIENT_KWARGS_MESSAGE,
        CACHE_FILE_MESSAGE,
        NO_CLIENT_MESSAGE,
    ]
    assert all(w.filename == __file__ for w in caught)
    make_client.assert_called_once_with("bedrock", region_name="eu")
    assert detector.client is client


def test_without_client_builds_the_openai_client_and_warns():
    client = FakeClient(PLAIN)
    with patch(MAKE_CLIENT, return_value=client) as make_client, pytest.warns(WARNING) as caught:
        detector = LLMDetector()
    assert [str(w.message) for w in caught] == [CLASS_MESSAGE, NO_CLIENT_MESSAGE]
    make_client.assert_called_once_with("openai")
    assert detector._judge.client is client


def test_provider_with_a_client_still_warns_as_a_deprecated_argument():
    with pytest.warns(WARNING) as caught:
        LLMDetector(client=FakeClient(PLAIN), provider="openai")
    assert [str(w.message) for w in caught] == [CLASS_MESSAGE, PROVIDER_MESSAGE]


def test_construction_inside_the_package_is_silent(tmp_path):
    kwargs = {"client": FakeClient(PLAIN), "cache_file": str(tmp_path / "c.json")}
    namespace = {"__name__": "lettucedetect.fake_internal", "D": LLMDetector, "kwargs": kwargs}
    context, answer, question = CASES[0]
    with patch(MAKE_CLIENT, return_value=FakeClient(PLAIN)), warnings.catch_warnings():
        warnings.simplefilter("error")
        exec("D(**kwargs)\nD(provider='bedrock', region_name='eu')", namespace)  # noqa: S102
        make_detector("llm", **kwargs)
        facade = HallucinationDetector(method="llm", provider="openai", **kwargs)
        assert facade.predict(context, answer, question, output_format="spans") == _construct(
            native=False
        ).predict(context, answer, question, output_format="spans")


def test_docstring_note_and_metadata():
    assert ".. deprecated::" in LLMDetector.__doc__
    assert "LLMJudgeDetector" in LLMDetector.__doc__
    assert "GenerativeDetector" in LLMDetector.__doc__
    assert LLMDetector.__deprecated__ == CLASS_MESSAGE  # ty: ignore[unresolved-attribute]


# -- choosing the delegate ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"model": "gpt-4.1-mini"}, LLMJudgeDetector),
        ({"model": NATIVE_MODEL}, GenerativeDetector),
        ({"model": "KRLabsOrg/lettucedect-base-modernbert-en-v1"}, LLMJudgeDetector),
        ({"model": "gpt-4.1-mini", "native": True}, GenerativeDetector),
        ({"model": NATIVE_MODEL, "native": False}, LLMJudgeDetector),
    ],
)
def test_name_rule_and_native_override_choose_the_delegate(kwargs, expected):
    detector = _construct(**kwargs)
    delegate = detector._generative or detector._judge
    assert type(delegate) is expected
    assert delegate.model_id == kwargs["model"]
    assert (detector._generative is None) != (detector._judge is None)


@pytest.mark.parametrize("native", [{"native": True}, {"model": NATIVE_MODEL}])
@pytest.mark.parametrize(
    ("option", "value"),
    [("zero_shot", True), ("fewshot_path", "x.json"), ("prompt_path", "p.txt")],
)
def test_native_with_a_judge_option_raises_native_options_conflict(native, option, value):
    with pytest.warns(WARNING), pytest.raises(ConfigurationError) as exc:
        LLMDetector(client=FakeClient(PLAIN), **native, **{option: value})
    assert exc.value.code == "native_options_conflict"
    assert exc.value.param == option


# -- invalid options ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "code", "param"),
    [
        ({"lang": "xx"}, "invalid_language", "config.lang"),
        ({"lang": "xx", "native": True}, "invalid_language", "config.lang"),
        ({"fewshot_path": "no/such.json"}, "fewshot_not_found", "config.fewshot_path"),
        ({"prompt_path": "no/such.txt"}, "invalid_config", "config.prompt_path"),
        ({"min_confidence": 1.5}, "invalid_config", "config.min_score"),
    ],
)
def test_invalid_option_raises_configuration_error_before_a_client_is_built(kwargs, code, param):
    with (
        patch(MAKE_CLIENT) as make_client,
        pytest.warns(WARNING),
        pytest.raises(ConfigurationError) as exc,
    ):
        LLMDetector(**kwargs)
    assert (exc.value.code, exc.value.param) == (code, param)
    make_client.assert_not_called()


# -- cache --------------------------------------------------------------------------------


def test_cache_file_builds_a_file_cache_and_warns_naming_cache(tmp_path):
    path = tmp_path / "cache.json"
    with pytest.warns(WARNING) as caught:
        detector = LLMDetector(client=FakeClient(PLAIN), cache_file=str(path))
    assert [str(w.message) for w in caught] == [CLASS_MESSAGE, CACHE_FILE_MESSAGE]
    assert "cache=" in CACHE_FILE_MESSAGE
    assert type(detector.cache) is FileCache
    assert detector.cache.path == path
    assert detector._judge.cache is detector.cache
    detector.predict(*CASES[0][:2], CASES[0][2])
    assert path.is_file()


@pytest.mark.parametrize("native", [False, True])
def test_without_cache_file_nothing_is_cached_and_no_file_is_written(native, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    package = Path(lettucedetect.__file__).parent
    before = sorted(p for p in package.rglob("*") if "__pycache__" not in p.parts)
    reply = '{"hallucinated_spans": []}' if native else '{"hallucination_list": []}'
    detector = _construct(client=FakeClient({"answer": reply}), native=native)

    assert detector.predict(["context"], "answer", "question") == []
    assert detector.predict_prompt("prompt", "answer") == []

    assert detector.cache is None
    assert (detector._generative or detector._judge).cache is None
    assert list(tmp_path.iterdir()) == []
    assert sorted(p for p in package.rglob("*") if "__pycache__" not in p.parts) == before
