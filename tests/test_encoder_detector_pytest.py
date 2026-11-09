"""Tests for ``EncoderDetector`` and ``EncoderConfig``: offline, with a stub model."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from lettucedetect.detectors.encoder import EncoderConfig, EncoderDetector
from lettucedetect.detectors.errors import ConfigurationError
from lettucedetect.detectors.prompt_utils import PromptUtils
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


# -- tokens, truncation and usage ---------------------------------------------------------


def _as_old_tokens(detection) -> list[dict]:
    return [
        {"token": t.token, "pred": int(t.flagged), "prob": round(t.score, 6)}
        for t in detection.tokens
    ]


def _round_tokens(tokens: list[dict]) -> list[dict]:
    return [{**t, "prob": round(t["prob"], 6)} for t in tokens]


def test_tokens_equal_golden_output(local_wordpiece_tokenizer):
    det = _detector(local_wordpiece_tokenizer)
    got = [
        _as_old_tokens(det.detect(_sources(*inp), include_tokens=True)) for inp in PREDICT_INPUTS
    ]
    assert got == [_round_tokens(x) for x in _golden("transformer_predict_tokens")]
    got = [
        _as_old_tokens(det.detect(PromptInput(prompt=p, answer=a), include_tokens=True))
        for p, a in zip(PROMPTS, ANSWERS, strict=True)
    ]
    assert got == [_round_tokens(x) for x in _golden("transformer_predict_prompt_tokens")]


def test_multi_group_tokens_equal_golden_output(local_wordpiece_tokenizer):
    det = _detector(local_wordpiece_tokenizer, max_length=MULTI_MAX_LENGTH)
    request = _sources(MULTI_CONTEXT, MULTI_ANSWER, MULTI_QUESTION)
    got = _as_old_tokens(det.detect(request, include_tokens=True))
    assert got == _round_tokens(_golden("transformer_predict_multi_group_tokens")["output"])


def test_token_offsets_lie_in_the_answer_and_spans_cover_flagged_tokens(local_wordpiece_tokenizer):
    det = _detector(local_wordpiece_tokenizer)
    for inp in PREDICT_INPUTS:
        request = _sources(*inp)
        result = det.detect(request, include_tokens=True)
        previous_end = 0
        for t in result.tokens:
            assert 0 <= t.start <= t.end <= len(request.answer)
            assert t.start >= previous_end or t.start == t.end
            if t.start != t.end:
                assert request.answer[t.start : t.end] == t.token
                previous_end = t.end
        assert result.tokens[-1].token == "[SEP]"  # noqa: S105
        assert (result.tokens[-1].start, result.tokens[-1].end) == (
            len(request.answer),
            len(request.answer),
        )
        flagged = {(t.start, t.end) for t in result.tokens if t.flagged and t.start != t.end}
        for span in result.spans:
            assert any(s >= span.start and e <= span.end for s, e in flagged)


def test_tokens_are_not_filtered_by_min_score_and_absent_by_default(local_wordpiece_tokenizer):
    det = _detector(local_wordpiece_tokenizer)
    request = _sources(*PREDICT_INPUTS[0])
    everything = det.detect(request, include_tokens=True)
    filtered = det.detect(request, include_tokens=True, min_score=1.0)
    assert filtered.spans == []
    assert filtered.tokens == everything.tokens
    assert det.detect(request).tokens is None


def test_group_scores_are_aggregated_with_max_of_the_per_group_scores(local_wordpiece_tokenizer):
    """With group-dependent scores, each token's score is the max over the groups' own scores."""

    def forward(input_ids=None, attention_mask=None, **kwargs):
        score = ((input_ids * 7 + input_ids.sum()) % 11 - 5).float() / 2
        output = MagicMock()
        output.logits = torch.stack([-score, score], dim=-1)
        return output

    det = EncoderDetector(
        MagicMock(side_effect=forward),
        local_wordpiece_tokenizer,
        config=EncoderConfig(max_length=MULTI_MAX_LENGTH),
    )
    request = _sources(MULTI_CONTEXT, MULTI_ANSWER, MULTI_QUESTION)
    groups = det._group_passages_into_chunks(MULTI_CONTEXT, MULTI_QUESTION, MULTI_ANSWER)
    per_group = [
        det.detect(
            PromptInput(
                prompt=PromptUtils.format_context(group, MULTI_QUESTION, "en"), answer=MULTI_ANSWER
            ),
            include_tokens=True,
        ).tokens
        for group in groups
    ]
    assert len(groups) > 1
    assert any(len({g[i].score for g in per_group}) > 1 for i in range(len(per_group[0])))
    merged = det.detect(request, include_tokens=True).tokens
    assert [t.score for t in merged] == [
        max(g[i].score for g in per_group) for i in range(len(merged))
    ]


def test_tokens_cost_no_extra_forward_pass(local_wordpiece_tokenizer):
    det = _detector(local_wordpiece_tokenizer, max_length=MULTI_MAX_LENGTH)
    request = _sources(MULTI_CONTEXT, MULTI_ANSWER, MULTI_QUESTION)
    det.detect(request)
    plain_calls = det.model.call_count
    det.detect(request, include_tokens=True)
    assert plain_calls > 1
    assert det.model.call_count == 2 * plain_calls


def _warning_codes(detection) -> list[str]:
    return [w.code for w in detection.warnings]


def _truncation_records(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def test_debug_log_traces_the_call_with_counts_only(local_wordpiece_tokenizer, caplog):
    det = _detector(local_wordpiece_tokenizer, max_length=MULTI_MAX_LENGTH)
    request = _sources(MULTI_CONTEXT, MULTI_ANSWER, MULTI_QUESTION)
    with caplog.at_level(logging.DEBUG, logger="lettucedetect"):
        det.detect(request)
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        m.startswith("detect started: kind=sources") and "lock_wait_ms=" in m for m in messages
    )
    assert any(m.startswith("scoring sources request: sources=4 passage_groups=") for m in messages)
    assert sum(m.startswith("scored one group: input_tokens=") for m in messages) > 1
    assert any(m.startswith("detect finished: spans=") and "scoring_ms=" in m for m in messages)
    assert all(r.levelno == logging.DEBUG for r in caplog.records)
    assert not any(MULTI_ANSWER in m or MULTI_CONTEXT[0] in m for m in messages)


def test_overlong_prompt_warns_and_logs_counts_without_text(local_wordpiece_tokenizer, caplog):
    prompt = " ".join(["the capital of france is paris ."] * 30)
    answer = "paris is the capital of france"
    det = _detector(local_wordpiece_tokenizer, max_length=40)
    full = local_wordpiece_tokenizer(prompt, answer, return_tensors="pt")["input_ids"].shape[1]
    with caplog.at_level(logging.DEBUG, logger="lettucedetect"):
        result = det.detect(PromptInput(prompt=prompt, answer=answer))
    assert _warning_codes(result) == ["input_truncated"]
    records = _truncation_records(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "input_truncated" in message and str(full) in message and "40" in message
    for record in caplog.records:
        assert prompt not in record.getMessage() and answer not in record.getMessage()


def test_passage_group_that_still_exceeds_max_length_warns(local_wordpiece_tokenizer, caplog):
    huge = " ".join(["the capital of france is paris ."] * 30)
    det = _detector(local_wordpiece_tokenizer, max_length=100)
    request = _sources([huge, "short answer word ."], "paris is the capital", "capital ?")
    groups = det._group_passages_into_chunks(
        [huge, "short answer word ."], "capital ?", "paris is the capital"
    )
    assert len(groups) == 2  # the oversized passage is alone in its group
    with caplog.at_level(logging.WARNING, logger="lettucedetect"):
        result = det.detect(request)
    assert _warning_codes(result) == ["input_truncated"]
    assert len(_truncation_records(caplog)) >= 1


def test_multi_group_input_that_fits_each_group_does_not_warn(local_wordpiece_tokenizer, caplog):
    det = _detector(local_wordpiece_tokenizer, max_length=MULTI_MAX_LENGTH)
    with caplog.at_level(logging.WARNING, logger="lettucedetect"):
        result = det.detect(_sources(MULTI_CONTEXT, MULTI_ANSWER, MULTI_QUESTION))
    assert result.warnings == []
    assert _truncation_records(caplog) == []


def test_input_of_exactly_max_length_is_not_truncated(local_wordpiece_tokenizer):
    prompt, answer = PROMPTS[2], ANSWERS[2]
    full = local_wordpiece_tokenizer(prompt, answer, return_tensors="pt")["input_ids"].shape[1]
    request = PromptInput(prompt=prompt, answer=answer)
    assert _detector(local_wordpiece_tokenizer, max_length=full).detect(request).warnings == []
    cut = _detector(local_wordpiece_tokenizer, max_length=full - 1).detect(request)
    assert _warning_codes(cut) == ["input_truncated"]


def _recording_detector(tokenizer, max_length=512):
    lengths = []

    def on_call():
        pass

    det = _detector(tokenizer, max_length=max_length, on_call=on_call)
    real = det.model.side_effect

    def record(input_ids=None, attention_mask=None, **kwargs):
        lengths.append(int(input_ids.shape[1]))
        return real(input_ids=input_ids, attention_mask=attention_mask)

    det.model.side_effect = record
    return det, lengths


def test_usage_units_for_sources_sum_over_groups(local_wordpiece_tokenizer):
    det, lengths = _recording_detector(local_wordpiece_tokenizer, MULTI_MAX_LENGTH)
    result = det.detect(_sources(MULTI_CONTEXT, MULTI_ANSWER, MULTI_QUESTION))
    assert len(lengths) > 1
    (usage,) = result.usage
    assert (usage.model_id, usage.model_version) == ("stub-encoder", None)
    assert usage.units == {
        "sources": len(MULTI_CONTEXT),
        "source_chars": sum(len(t) for t in MULTI_CONTEXT),
        "answer_chars": len(MULTI_ANSWER),
        "input_tokens": sum(lengths),
    }


def test_usage_units_for_prompt_have_no_sources(local_wordpiece_tokenizer):
    det, lengths = _recording_detector(local_wordpiece_tokenizer)
    result = det.detect(PromptInput(prompt=PROMPTS[0], answer=ANSWERS[0]))
    assert result.usage[0].units == {
        "source_chars": len(PROMPTS[0]),
        "answer_chars": len(ANSWERS[0]),
        "input_tokens": lengths[0],
    }


def test_usage_input_tokens_count_the_truncated_input(local_wordpiece_tokenizer):
    det, lengths = _recording_detector(local_wordpiece_tokenizer, max_length=40)
    prompt = " ".join(["the capital of france is paris ."] * 30)
    result = det.detect(PromptInput(prompt=prompt, answer="paris is the capital of france"))
    assert lengths == [40]
    assert result.usage[0].units["input_tokens"] == 40


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


# -- typed spans and typer usage ------------------------------------------------------------

TYPER_ID, TYPER_VERSION = "stub-typer", "c0ffee"
ZERO_SPAN_REQUEST = PromptInput(prompt="the capital of france is paris .", answer="the")


class _Typer:
    """A stub typer: tags each span from its text and context, and records its calls."""

    model_id = TYPER_ID
    model_version = TYPER_VERSION

    def __init__(self, delay: float = 0.0, error: Exception | None = None) -> None:
        self.delay, self.error = delay, error
        self.calls: list[tuple[str, str, int]] = []
        self.threads: list[int] = []

    def type_spans(self, answer, context, spans):
        self.threads.append(threading.get_ident())
        self.calls.append((answer, context, len(spans)))
        if self.error is not None:
            raise self.error
        if self.delay:
            time.sleep(self.delay)
        for span in spans:
            span["category"] = f"CAT[{span['text']}]"
            span["subcategory"] = f"SUB[{len(context)}]"
        return spans


def _typed(tokenizer, typer=None, **kwargs) -> EncoderDetector:
    return EncoderDetector(
        _stub_model(),
        tokenizer,
        typer if typer is not None else _Typer(),
        config=EncoderConfig(max_length=512),
        model_id="stub-encoder",
        **kwargs,
    )


def test_every_span_is_typed_with_a_typer_and_untyped_without(local_wordpiece_tokenizer):
    request = _sources(*PREDICT_INPUTS[0])
    typed = _typed(local_wordpiece_tokenizer).detect(request).spans
    untyped = _detector(local_wordpiece_tokenizer).detect(request).spans
    assert typed and len(typed) == len(untyped)
    assert all(s.category == f"CAT[{s.text}]" and s.subcategory for s in typed)
    assert all(s.category is None and s.subcategory is None for s in untyped)
    # Typing adds the two labels and changes nothing else.
    assert [s.model_dump(exclude={"category", "subcategory"}) for s in typed] == [
        s.model_dump(exclude={"category", "subcategory"}) for s in untyped
    ]


def test_typer_reads_the_joined_sources_or_the_prompt(local_wordpiece_tokenizer):
    typer = _Typer()
    det = _typed(local_wordpiece_tokenizer, typer)
    det.detect(_sources(["the capital", "of france"], "paris is the capital ."))
    det.detect(PromptInput(prompt=PROMPTS[0], answer=ANSWERS[0]))
    assert [(answer, context) for answer, context, _ in typer.calls] == [
        ("paris is the capital .", "the capital\nof france"),
        (ANSWERS[0], PROMPTS[0]),
    ]


def test_usage_has_the_encoder_and_the_typer_with_its_span_count(local_wordpiece_tokenizer):
    result = _typed(local_wordpiece_tokenizer).detect(_sources(*PREDICT_INPUTS[0]))
    encoder, typer = result.usage
    assert (encoder.model_id, encoder.model_version) == ("stub-encoder", None)
    assert (typer.model_id, typer.model_version) == (TYPER_ID, TYPER_VERSION)
    assert typer.units == {"spans": len(result.spans)} and result.spans


def test_usage_has_one_entry_without_a_typer(local_wordpiece_tokenizer):
    assert len(_detector(local_wordpiece_tokenizer).detect(_sources(*PREDICT_INPUTS[0])).usage) == 1


def test_typer_usage_reports_zero_spans_when_nothing_is_flagged(local_wordpiece_tokenizer):
    typer = _Typer()
    result = _typed(local_wordpiece_tokenizer, typer).detect(ZERO_SPAN_REQUEST)
    assert result.spans == []
    assert [u.model_id for u in result.usage] == ["stub-encoder", TYPER_ID]
    assert result.usage[1].units == {"spans": 0}
    assert typer.calls == []  # nothing to type, so the typer is not called


def test_a_typer_without_ids_is_metered_with_none(local_wordpiece_tokenizer):
    class Bare:
        def type_spans(self, answer, context, spans):
            for span in spans:
                span["category"], span["subcategory"] = "c", "s"
            return spans

    typer_usage = _typed(local_wordpiece_tokenizer, Bare()).detect(_sources(*PREDICT_INPUTS[0]))
    assert (typer_usage.usage[1].model_id, typer_usage.usage[1].model_version) == (None, None)


def test_the_typer_meters_spans_before_min_score_filters_them(local_wordpiece_tokenizer):
    det = _typed(local_wordpiece_tokenizer)
    request = TestEncoderDetectorContract.sources_request
    everything = det.detect(request)
    top = max(s.score for s in everything.spans)
    kept = det.detect(request, min_score=top)
    assert 0 < len(kept.spans) < len(everything.spans)
    assert kept.usage[1].units == everything.usage[1].units == {"spans": len(everything.spans)}
    assert all(s.category for s in kept.spans)


async def test_adetect_on_a_typed_detector_equals_detect(local_wordpiece_tokenizer):
    typer = _Typer()
    det = _typed(local_wordpiece_tokenizer, typer)
    for request in (
        _sources(*PREDICT_INPUTS[0]),
        PromptInput(prompt=PROMPTS[0], answer=ANSWERS[0]),
        ZERO_SPAN_REQUEST,
    ):
        asynchronous = await det.adetect(request)
        assert asynchronous == det.detect(request)
        assert len(asynchronous.usage) == 2
    assert threading.get_ident() not in typer.threads[:1]  # the first (async) call ran off-loop


async def test_a_slow_typer_does_not_block_the_event_loop(local_wordpiece_tokenizer):
    det = _typed(local_wordpiece_tokenizer, _Typer(delay=0.3))
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    result = await det.adetect(_sources(*PREDICT_INPUTS[0]))
    task.cancel()
    assert result.spans and ticks >= 10


def test_a_failing_typer_raises_instead_of_returning_untyped_spans(local_wordpiece_tokenizer):
    det = _typed(local_wordpiece_tokenizer, _Typer(error=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        det.detect(_sources(*PREDICT_INPUTS[0]))


def test_a_typed_call_logs_no_customer_text(local_wordpiece_tokenizer, caplog):
    caplog.set_level(logging.DEBUG)
    _typed(local_wordpiece_tokenizer).detect(
        PromptInput(prompt="the capital of france is paris .", answer="paris is the capital")
    )
    assert "paris is the capital" not in caplog.text
    assert "france is paris" not in caplog.text


class TestTypedEncoderDetectorContract(TestEncoderDetectorContract):
    """The detector contract suite, run against an EncoderDetector with a stub typer."""

    def make_detector(self, on_call=None):
        """Build a typed detector whose stub model calls ``on_call`` in its forward pass."""
        return EncoderDetector(
            _stub_model(on_call=on_call),
            self.tokenizer,
            _Typer(),
            config=EncoderConfig(max_length=512),
            model_id="stub-encoder",
        )

    def test_every_span_is_typed(self):
        """Every span of both request kinds carries both labels."""
        detector = self.make_detector()
        for request in self._requests():
            spans = detector.detect(request).spans
            assert spans and all(s.category and s.subcategory for s in spans)
