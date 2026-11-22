"""The ``predict*`` methods warn when called from outside the package: offline, fakes only."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest

from lettucedetect.detectors import _deprecation as dep
from lettucedetect.detectors._deprecation import LettuceDetectDeprecationWarning
from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.encoder import EncoderConfig, EncoderDetector
from lettucedetect.detectors.generative import GenerativeDetector
from lettucedetect.detectors.llm import LLMDetector
from lettucedetect.detectors.llm_judge import LLMJudgeDetector
from lettucedetect.detectors.rag_fact_checker import RAGFactCheckerDetector
from lettucedetect.detectors.transformer import TransformerDetector
from lettucedetect.detectors.types import DetectedSpan, Detection, PromptInput, SourcesInput
from lettucedetect.models.inference import HallucinationDetector
from tests.test_golden_llm_pytest import CASES as LLM_CASES
from tests.test_golden_llm_pytest import NATIVE_PLAIN, PLAIN, FakeClient
from tests.test_golden_transformer_pytest import _stub_forward
from tests.test_rag_fact_checker_detector_pytest import ANSWER as RAG_ANSWER
from tests.test_rag_fact_checker_detector_pytest import _detector as _rag_detector
from tests.test_vela_detector_pytest import _detector as _vela_detector

WARNING = LettuceDetectDeprecationWarning
ENCODER_ANSWER = "paris is the capital"
LLM_ANSWER = LLM_CASES[0][1]

REPLACEMENTS = {
    "predict": "detect",
    "predict_prompt": "detect",
    "predict_prompt_batch": "detect_batch",
}
CALLS = {
    "predict": "detector.predict(['the context .'], answer, output_format='spans')",
    "predict_prompt": "detector.predict_prompt('the prompt .', answer, output_format='spans')",
    "predict_prompt_batch": (
        "detector.predict_prompt_batch(['the prompt .'], [answer], output_format='spans')"
    ),
}


class Plain(BaseDetector):
    """Implements only ``_detect``, so its ``predict*`` methods are the ``BaseDetector`` ones."""

    def _detect(self, request, *, include_tokens, include_detail):
        return Detection(spans=[DetectedSpan(start=0, end=3, text="abc", score=0.8)])


def _encoder(tokenizer) -> EncoderDetector:
    return EncoderDetector(
        MagicMock(side_effect=_stub_forward),
        tokenizer,
        config=EncoderConfig(max_length=512),
        model_id="dummy_path",
    )


def _transformer(tokenizer) -> TransformerDetector:
    encoder = _encoder(tokenizer)
    with pytest.warns(WARNING):
        detector = TransformerDetector.__new__(TransformerDetector)
    vars(detector).update(vars(encoder))
    return detector


def _llm(tokenizer) -> LLMDetector:
    with pytest.warns(WARNING):
        return LLMDetector(client=FakeClient(PLAIN), native=False)


# class name -> (builder taking the tokenizer fixture, an answer the fake behind it knows)
DETECTORS = {
    "Plain": (lambda tokenizer: Plain(), "abcdef"),
    "EncoderDetector": (_encoder, ENCODER_ANSWER),
    "TransformerDetector": (_transformer, ENCODER_ANSWER),
    "LLMDetector": (_llm, LLM_ANSWER),
    "LLMJudgeDetector": (lambda tokenizer: LLMJudgeDetector(FakeClient(PLAIN), "m"), LLM_ANSWER),
    "GenerativeDetector": (
        lambda tokenizer: GenerativeDetector(FakeClient(NATIVE_PLAIN), "m"),
        LLM_ANSWER,
    ),
    "RAGFactCheckerDetector": (lambda tokenizer: _rag_detector(), RAG_ANSWER),
    "VelaDetector": (lambda tokenizer: _vela_detector(), "the answer ."),
}
EVERY_METHOD = [(name, method) for name in DETECTORS for method in CALLS]


def _namespace(name: str, tokenizer, module: str) -> dict:
    """Return the globals for a ``CALLS`` snippet that runs as a module called ``module``."""
    build, answer = DETECTORS[name]
    return {"__name__": module, "detector": build(tokenizer), "answer": answer}


@pytest.mark.parametrize(("name", "method"), EVERY_METHOD)
def test_a_call_from_outside_the_package_warns_once(name, method, local_wordpiece_tokenizer):
    detector = _namespace(name, local_wordpiece_tokenizer, __name__)["detector"]
    answer = DETECTORS[name][1]
    with pytest.warns(WARNING) as caught:
        if method == "predict":
            detector.predict(["the context ."], answer, output_format="spans")
        elif method == "predict_prompt":
            detector.predict_prompt("the prompt .", answer, output_format="spans")
        else:
            detector.predict_prompt_batch(["the prompt ."], [answer], output_format="spans")
    assert [str(w.message) for w in caught] == [
        f"{name}.{method} is deprecated since {dep.DEPRECATED_SINCE} and will be removed in "
        f"{dep.REMOVED_IN}. Use {REPLACEMENTS[method]}."
    ]
    assert caught[0].filename == __file__


@pytest.mark.parametrize(("name", "method"), EVERY_METHOD)
def test_a_call_from_inside_the_package_is_silent(name, method, local_wordpiece_tokenizer):
    namespace = _namespace(name, local_wordpiece_tokenizer, "lettucedetect.fake_internal")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exec(CALLS[method], namespace)  # noqa: S102


@pytest.mark.parametrize("name", list(DETECTORS))
def test_the_facade_is_silent(name, local_wordpiece_tokenizer):
    build, answer = DETECTORS[name]
    facade = HallucinationDetector.__new__(HallucinationDetector)
    facade.detector = build(local_wordpiece_tokenizer)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        facade.predict(["the context ."], answer, output_format="spans")
        facade.predict_prompt("the prompt .", answer, output_format="spans")
        facade.predict_prompt_batch(["the prompt ."], [answer], output_format="spans")


@pytest.mark.parametrize("method", list(CALLS))
def test_the_methods_carry_the_deprecation_metadata(method):
    for cls in (BaseDetector, TransformerDetector, LLMDetector, RAGFactCheckerDetector):
        message = vars(cls)[method].__deprecated__
        assert message.startswith(f"{cls.__name__}.{method} is deprecated")
        assert message.endswith(f"Use {REPLACEMENTS[method]}.")


def _calls_super() -> type[Plain]:
    """Return a new subclass whose ``predict`` and ``predict_prompt`` call the inherited ones."""

    class CallsSuper(Plain):
        max_concurrency = 4

        def predict(
            self, context, answer, question=None, output_format="tokens", min_confidence=0.0
        ):
            return super().predict(context, answer, question, output_format, min_confidence)

        def predict_prompt(self, prompt, answer, output_format="tokens", min_confidence=0.0):
            return super().predict_prompt(prompt, answer, output_format, min_confidence)

    return CallsSuper


def _messages(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, WARNING)]


def test_a_batch_over_an_overridden_predict_prompt_warns_once():
    """The overrides call the inherited methods for every pair; only the user's call warns.

    The first call also gives the once-per-class warning of a class that overrides ``predict*``.
    """
    detector = _calls_super()()
    for class_warnings in (1, 0):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            detector.predict_prompt_batch(["p", "q", "r"], ["abcdef"] * 3, "spans")
        messages = _messages(caught)
        assert "CallsSuper.predict_prompt_batch is deprecated" in messages[0]
        assert caught[0].filename == __file__
        assert ["(without _detect)" in message for message in messages[1:]] == [
            True
        ] * class_warnings


def test_detect_through_an_overridden_predict_gives_only_the_class_warning():
    detector = _calls_super()()
    sources = SourcesInput(sources=[{"id": "a", "text": "s"}], answer="abcdef")
    prompt = PromptInput(prompt="p", answer="abcdef")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detector.detect(sources)
        detector.detect(prompt)
        assert all(result.ok for result in detector.detect_batch([sources, prompt] * 4))
    assert len(_messages(caught)) == 1
    assert "CallsSuper.predict (without _detect)" in _messages(caught)[0]


def test_a_concurrent_batch_warns_once():
    class Concurrent(Plain):
        max_concurrency = 4

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Concurrent().predict_prompt_batch(["p"] * 8, ["abcdef"] * 8, "spans")
    assert len(_messages(caught)) == 1
    assert caught[0].filename == __file__
