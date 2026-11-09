"""Tests for ``EncoderDetector`` and ``EncoderConfig``: offline, with a stub model."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from lettucedetect.detectors.encoder import EncoderConfig, EncoderDetector
from lettucedetect.detectors.errors import ConfigurationError
from lettucedetect.detectors.types import Chunk, PromptInput, SourcesInput
from tests.detector_contract import DetectorContract
from tests.golden import GOLDEN_DIR
from tests.test_golden_transformer_pytest import (
    ANSWERS,
    MULTI_ANSWER,
    MULTI_CONTEXT,
    MULTI_MAX_LENGTH,
    MULTI_QUESTION,
    PREDICT_INPUTS,
    PROMPTS,
    _stub_forward,
)


def _stub_model(on_call=None, delay: float = 0.0):
    def forward(input_ids=None, attention_mask=None, **kwargs):
        if on_call is not None:
            on_call()
        if delay:
            time.sleep(delay)
        return _stub_forward(input_ids=input_ids, attention_mask=attention_mask)

    return MagicMock(side_effect=forward)


def _detector(tokenizer, max_length=512, **model_kwargs) -> EncoderDetector:
    return EncoderDetector(
        _stub_model(**model_kwargs),
        tokenizer,
        config=EncoderConfig(max_length=max_length),
        model_id="stub-encoder",
    )


def _sources(context, answer, question=None) -> SourcesInput:
    return SourcesInput(
        sources=[Chunk(id=str(i), text=t) for i, t in enumerate(context)],
        answer=answer,
        question=question,
    )


def _as_confidence(detection) -> list[dict]:
    return [
        {"start": s.start, "end": s.end, "confidence": s.score, "text": s.text}
        for s in detection.spans
    ]


# -- construction and config --------------------------------------------------------------


def test_construction_is_offline_and_leaves_cwd_empty(
    tmp_path, monkeypatch, local_wordpiece_tokenizer
):
    """Building from objects downloads nothing and writes no file."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    boom = MagicMock(side_effect=AssertionError("download attempted"))
    with (
        patch("transformers.AutoTokenizer.from_pretrained", boom),
        patch("transformers.AutoModelForTokenClassification.from_pretrained", boom),
    ):
        EncoderDetector(_stub_model(), local_wordpiece_tokenizer)
    assert list(tmp_path.iterdir()) == []
    boom.assert_not_called()


def test_config_defaults():
    config = EncoderConfig()
    assert (config.max_length, config.batch_size, config.lang) == (4096, 16, "en")


@pytest.mark.parametrize(
    ("kwargs", "code", "param"),
    [
        ({"colour": 1}, "invalid_config", "colour"),
        ({"lang": "xx"}, "invalid_language", "lang"),
        ({"batch_size": 0}, "invalid_batch_size", "batch_size"),
        ({"max_length": 0}, "invalid_config", "max_length"),
    ],
)
def test_config_errors_carry_code_and_param(kwargs, code, param):
    with pytest.raises(ConfigurationError) as exc:
        EncoderConfig(**kwargs)
    assert (exc.value.code, exc.value.param) == (code, param)
    with pytest.raises(ConfigurationError):
        EncoderConfig.model_validate(kwargs)


# -- results ------------------------------------------------------------------------------


def _golden(name: str):
    return json.loads((GOLDEN_DIR / f"{name}.json").read_text())


def _round(spans: list[dict]) -> list[dict]:
    return [{**s, "confidence": round(s["confidence"], 6)} for s in spans]


def test_sources_spans_equal_golden_output(local_wordpiece_tokenizer):
    """Detect spans equal today's predict(spans) fixtures, with score = confidence."""
    det = _detector(local_wordpiece_tokenizer)
    got = [_round(_as_confidence(det.detect(_sources(*inp)))) for inp in PREDICT_INPUTS]
    assert got == [_round(x) for x in _golden("transformer_predict_spans")]


def test_prompt_spans_equal_golden_output(local_wordpiece_tokenizer):
    det = _detector(local_wordpiece_tokenizer)
    got = [
        _round(_as_confidence(det.detect(PromptInput(prompt=p, answer=a))))
        for p, a in zip(PROMPTS, ANSWERS, strict=True)
    ]
    assert got == [_round(x) for x in _golden("transformer_predict_prompt_spans")]


def test_multi_group_sources_equal_golden_output(local_wordpiece_tokenizer):
    """Several passage groups are aggregated with max(), as today."""
    det = _detector(local_wordpiece_tokenizer, max_length=MULTI_MAX_LENGTH)
    assert len(det._group_passages_into_chunks(MULTI_CONTEXT, MULTI_QUESTION, MULTI_ANSWER)) > 1
    got = _round(_as_confidence(det.detect(_sources(MULTI_CONTEXT, MULTI_ANSWER, MULTI_QUESTION))))
    assert got == _round(_golden("transformer_predict_multi_group_spans")["output"])


async def test_adetect_equals_detect(local_wordpiece_tokenizer):
    det = _detector(local_wordpiece_tokenizer)
    for request in (
        _sources(*PREDICT_INPUTS[0]),
        PromptInput(prompt=PROMPTS[0], answer=ANSWERS[0]),
    ):
        assert await det.adetect(request) == det.detect(request)


# -- concurrency --------------------------------------------------------------------------


async def test_concurrent_adetect_calls_do_not_overlap(local_wordpiece_tokenizer):
    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    def on_call():
        with guard:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.005)
        with guard:
            state["active"] -= 1

    det = _detector(local_wordpiece_tokenizer, max_length=MULTI_MAX_LENGTH, on_call=on_call)
    requests = []
    for i in range(20):
        word = ["paris", "word", "france", "capital"][i % 4]
        answer = " ".join([word, "the", "capital", "of", "france"] * (1 + i % 5))
        if i % 3 == 0:
            requests.append(_sources(MULTI_CONTEXT, answer, MULTI_QUESTION))
        elif i % 3 == 1:
            requests.append(
                PromptInput(prompt=" ".join(["the capital of france"] * (i * 3)), answer=answer)
            )
        else:
            requests.append(_sources(["short answer word ."], answer))
    expected = [det.detect(r) for r in requests]
    state["max"] = 0
    results = await asyncio.gather(*(det.adetect(r) for r in requests))
    assert list(results) == expected
    assert state["max"] == 1


async def test_padded_batches_and_adetect_do_not_overlap(local_wordpiece_tokenizer):
    """Padded batch scoring takes the same lock as detect, so mixed calls never overlap."""
    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    def on_call():
        with guard:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.005)
        with guard:
            state["active"] -= 1

    det = _detector(local_wordpiece_tokenizer, on_call=on_call)
    batches = [(PROMPTS[: 1 + i % 3], ANSWERS[: 1 + i % 3]) for i in range(8)]
    requests = [_sources(*PREDICT_INPUTS[i % len(PREDICT_INPUTS)]) for i in range(8)]
    expected_batches = [det._predict_batch(p, a, "spans") for p, a in batches]
    expected = [det.detect(r) for r in requests]
    probe = local_wordpiece_tokenizer(
        PROMPTS[:3], ANSWERS[:3], padding=True, truncation="only_first", return_tensors="pt"
    )
    assert (probe["attention_mask"] == 0).any()  # the batches really contain padding
    state["max"] = 0
    results = await asyncio.gather(
        *(asyncio.to_thread(det._predict_batch, p, a, "spans") for p, a in batches),
        *(det.adetect(r) for r in requests),
    )
    assert list(results[:8]) == expected_batches
    assert list(results[8:]) == expected
    assert state["max"] == 1


def test_racing_first_calls_create_exactly_one_lock(local_wordpiece_tokenizer):
    det = EncoderDetector.__new__(EncoderDetector)
    det.model, det.tokenizer = _stub_model(), local_wordpiece_tokenizer
    det.device, det.max_length, det.lang, det.typer = torch.device("cpu"), 512, "en", None
    created = []
    real = threading.RLock

    def slow_rlock():
        time.sleep(0.05)
        lock = real()
        created.append(lock)
        return lock

    barrier = threading.Barrier(8)
    seen = []

    def race():
        barrier.wait()
        seen.append(det._get_scoring_lock())

    with patch("lettucedetect.detectors.encoder.threading.RLock", slow_rlock):
        threads = [threading.Thread(target=race) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert len(created) == 1
    assert all(lock is created[0] for lock in seen)
    assert det.detect(_sources(*PREDICT_INPUTS[0])).spans is not None


# -- contract -----------------------------------------------------------------------------


class TestEncoderDetectorContract(DetectorContract):
    """The detector contract suite, run against an EncoderDetector with a stub model."""

    sources_request = _sources(
        ["the capital of france is paris . short answer word ."],
        "paris is the capital of france and the answer is a short word and paris again",
        "the capital of france ?",
    )
    prompt_request = PromptInput(
        prompt="the capital of france is paris . the capital of france",
        answer="the capital of france is paris and paris is a short answer word again",
    )

    @pytest.fixture(autouse=True)
    def _tokenizer(self, local_wordpiece_tokenizer):
        self.tokenizer = local_wordpiece_tokenizer

    def make_detector(self, on_call=None):
        """Build a detector whose stub model calls ``on_call`` in its forward pass."""
        return _detector(self.tokenizer, on_call=on_call)
