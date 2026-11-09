"""Golden (characterization) tests for ``make_detector``, ``HallucinationDetector`` and the cache.

Builds on the stubs of the transformer, LLM and RAG golden tests. Recordings live in
``tests/fixtures/golden/{factory,facade,cache}_*.json``.

The cache keys must stay byte-identical when a later change adds ``model_version`` and it is
``None``. They are SHA-256 hashes of the full prompt, so they also change whenever a prompt
template or few-shot example file changes, which invalidates users' caches: re-record on purpose.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from lettucedetect.detectors.cache import CacheManager
from lettucedetect.detectors.factory import make_detector
from lettucedetect.detectors.llm import LLMDetector
from lettucedetect.models.inference import HallucinationDetector
from lettucedetect.prompts import generative as gen
from tests.golden import assert_golden
from tests.test_golden_llm_pytest import (
    ANSWERS,
    CASES,
    NATIVE_PLAIN,
    PLAIN,
    PROMPTS,
    FakeClient,
    _judge,
    _verdicts,
)
from tests.test_golden_rag_fact_checker_pytest import ANSWER as RAG_ANSWER
from tests.test_golden_rag_fact_checker_pytest import FakeFactChecker, _result
from tests.test_golden_transformer_pytest import (
    ANSWERS as TRANSFORMER_ANSWERS,
)
from tests.test_golden_transformer_pytest import (
    PREDICT_INPUTS,
    _build,
    _stub_forward,
)
from tests.test_golden_transformer_pytest import (
    PROMPTS as TRANSFORMER_PROMPTS,
)

TRANSFORMER_PATCH = "lettucedetect.detectors.transformer.{}.from_pretrained"
RAG_PATCH = "lettucedetect.ragfactchecker.RAGFactChecker"


def _type_name(obj: object) -> str:
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


class _Patched:
    """Context manager replacing model/tokenizer loading with the local stubs."""

    def __init__(self, tokenizer) -> None:
        self.patches = [
            patch(TRANSFORMER_PATCH.format("AutoTokenizer"), return_value=tokenizer),
            patch(
                TRANSFORMER_PATCH.format("AutoModelForTokenClassification"),
                return_value=MagicMock(side_effect=_stub_forward),
            ),
        ]

    def __enter__(self) -> None:
        for p in self.patches:
            p.start()

    def __exit__(self, *exc: object) -> None:
        for p in reversed(self.patches):
            p.stop()


class TestMakeDetector:
    """The factory builds the right class per method and rejects unknown ones."""

    def test_returns_the_right_class_per_method(self, local_wordpiece_tokenizer, tmp_path):
        """Each method maps to its detector class (loading replaced by fakes)."""
        with _Patched(local_wordpiece_tokenizer):
            transformer = make_detector("transformer", model_path="dummy_path", max_length=64)
        llm = make_detector(
            "llm", client=FakeClient({}), cache_file=str(tmp_path / "cache.json"), native=False
        )
        with patch(RAG_PATCH):
            rag = make_detector("rag_fact_checker")
        assert_golden(
            "factory_make_detector",
            {
                "transformer": _type_name(transformer),
                "llm": _type_name(llm),
                "rag_fact_checker": _type_name(rag),
            },
        )

    def test_unknown_method_error(self):
        """An unknown method raises ValueError with this exact message."""
        with pytest.raises(ValueError) as exc_info:
            make_detector("bogus")
        assert_golden("factory_unknown_method_error", {"message": str(exc_info.value)})

    def test_facade_unknown_method_error(self):
        """The facade surfaces the same error."""
        with pytest.raises(ValueError) as exc_info:
            HallucinationDetector(method="bogus")
        assert_golden("facade_unknown_method_error", {"message": str(exc_info.value)})


def _facade_outputs(detector) -> dict:
    """Run the three facade entry points with default format and with ``"spans"``."""
    context, answer, question = PREDICT_INPUTS[0]
    out = {}
    for label, kwargs in (("default", {}), ("spans", {"output_format": "spans"})):
        out[label] = {
            "predict": detector.predict(context, answer, question, **kwargs),
            "predict_prompt": detector.predict_prompt(TRANSFORMER_PROMPTS[0], answer, **kwargs),
            "predict_prompt_batch": detector.predict_prompt_batch(
                TRANSFORMER_PROMPTS, TRANSFORMER_ANSWERS, **kwargs
            ),
        }
    return out


@pytest.mark.filterwarnings(
    "ignore::lettucedetect.detectors._deprecation.LettuceDetectDeprecationWarning"
)
class TestFacade:
    """``HallucinationDetector`` returns exactly what the detector alone returns."""

    def test_transformer(self, local_wordpiece_tokenizer):
        """Facade over the stub transformer detector."""
        bare = _build(local_wordpiece_tokenizer, 512)
        with _Patched(local_wordpiece_tokenizer):
            facade = HallucinationDetector(
                method="transformer", model_path="dummy_path", max_length=512
            )
        outputs = _facade_outputs(facade)
        assert outputs == _facade_outputs(bare)
        assert_golden("facade_transformer", outputs)

    def test_llm(self, tmp_path):
        """Facade over the fake-client LLM detector."""

        def build(name: str) -> LLMDetector:
            return LLMDetector(
                client=FakeClient(PLAIN), cache_file=str(tmp_path / name), native=False
            )

        facade = HallucinationDetector(
            method="llm",
            client=FakeClient(PLAIN),
            cache_file=str(tmp_path / "f.json"),
            native=False,
        )
        context, answer, question = CASES[0]
        results = {}
        for label, kwargs in (("default", {}), ("spans", {"output_format": "spans"})):
            args = (context, answer, question)
            results[label] = {
                "predict": facade.predict(*args, **kwargs),
                "predict_prompt": facade.predict_prompt(PROMPTS[0], answer, **kwargs),
                "predict_prompt_batch": facade.predict_prompt_batch(PROMPTS, ANSWERS, **kwargs),
            }
            # LLMDetector alone defaults to "spans", the facade to "tokens": pass the facade's.
            bare_kwargs = {"output_format": "tokens", **kwargs}
            bare = build(f"b_{label}.json")
            assert results[label] == {
                "predict": bare.predict(*args, **bare_kwargs),
                "predict_prompt": bare.predict_prompt(PROMPTS[0], answer, **bare_kwargs),
                "predict_prompt_batch": bare.predict_prompt_batch(PROMPTS, ANSWERS, **bare_kwargs),
            }
        assert_golden("facade_llm", results)

    def test_rag_fact_checker(self):
        """Facade over the RAG detector with a fake fact checker."""
        fake = FakeFactChecker({RAG_ANSWER: _result([["France", "population", "69 million"]])})
        with patch(RAG_PATCH):
            facade = HallucinationDetector(method="rag_fact_checker")
        facade.detector.rag = fake
        results = {}
        for fmt in ("tokens", "spans"):
            results[fmt] = {
                "predict": facade.predict(["c"], RAG_ANSWER, "q", output_format=fmt),
                "predict_prompt": facade.predict_prompt("p", RAG_ANSWER, output_format=fmt),
                "predict_prompt_batch": facade.predict_prompt_batch(
                    ["p"], [RAG_ANSWER], output_format=fmt
                ),
            }
            bare = facade.detector
            assert results[fmt] == {
                "predict": bare.predict(["c"], RAG_ANSWER, "q", output_format=fmt),
                "predict_prompt": bare.predict_prompt("p", RAG_ANSWER, output_format=fmt),
                "predict_prompt_batch": bare.predict_prompt_batch(
                    ["p"], [RAG_ANSWER], output_format=fmt
                ),
            }
        assert_golden("facade_rag_fact_checker", results)


@pytest.mark.filterwarnings(
    "ignore::lettucedetect.detectors._deprecation.LettuceDetectDeprecationWarning"
)
class TestCacheManager:
    """The cache file format and the key strings the LLM detector builds."""

    def test_file_after_set_and_delete(self, tmp_path):
        """The JSON file after a few set and delete calls, byte for byte."""
        path = tmp_path / "nested" / "cache.json"
        cache = CacheManager(path)
        cache.set("a", {"x": 1, "y": [1, 2]})
        cache.set("b", "héllo ✓")
        cache.set("c", None)
        cache.delete("a")
        cache.delete("missing")
        cache.set("b", "overwritten")
        raw = path.read_text(encoding="utf-8")
        assert json.loads(raw) == {"b": "overwritten", "c": None}
        assert CacheManager(path).get("b") == "overwritten"
        assert_golden("cache_file_after_set_and_delete", {"raw": raw})

    def test_hash_strings(self):
        """``_hash`` is SHA-256 of the parts joined by ``||``."""
        assert_golden(
            "cache_hash_strings",
            {
                "judge": CacheManager._hash("prompt", "gpt-4.1-mini", str(0.0)),
                "native": CacheManager._hash("prompt", "gpt-4.1-mini", str(0.0), "native"),
                "verify": CacheManager._hash("prompt", "gpt-4.1-mini", str(0.0), "verify"),
            },
        )

    def test_keys_written_by_llm_detector(self, tmp_path):
        """The judge, native and verify keys the detector writes for fixed prompts."""
        answer, prompt = ANSWERS[0], PROMPTS[0]

        judge_file = tmp_path / "judge.json"
        judge = LLMDetector(
            client=FakeClient(_judge(["69 million", "Paris"], ["Spain"]), _verdicts(["Paris"], [])),
            cache_file=str(judge_file),
            native=False,
            verify=True,
        )
        judge.predict_prompt(prompt, answer)
        judge_key, verify_key = list(json.loads(judge_file.read_text(encoding="utf-8")))
        assert judge_key == CacheManager._hash(
            judge._build_prompt(prompt, answer), judge.model, str(judge.temperature)
        )

        native_file = tmp_path / "native.json"
        native = LLMDetector(
            client=FakeClient(NATIVE_PLAIN), cache_file=str(native_file), native=True
        )
        native.predict_prompt(prompt, answer)
        (native_key,) = list(json.loads(native_file.read_text(encoding="utf-8")))
        assert native_key == CacheManager._hash(
            gen.SYSTEM_BASE + gen.build_user_message(prompt, answer),
            native.model,
            str(native.temperature),
            "native",
        )

        assert_golden(
            "cache_llm_detector_keys",
            {"judge": judge_key, "native": native_key, "verify": verify_key},
        )
