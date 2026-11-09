"""``TransformerDetector`` is a deprecated subclass of ``EncoderDetector``: offline, stub model."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch

from lettucedetect.detectors import _deprecation as dep
from lettucedetect.detectors._deprecation import LettuceDetectDeprecationWarning
from lettucedetect.detectors.encoder import EncoderConfig, EncoderDetector
from lettucedetect.detectors.factory import make_detector
from lettucedetect.detectors.transformer import TransformerDetector
from lettucedetect.detectors.types import Chunk, PromptInput, SourcesInput
from lettucedetect.models.inference import HallucinationDetector
from tests.golden import assert_golden
from tests.test_golden_transformer_pytest import (
    ANSWERS,
    PREDICT_INPUTS,
    PROMPTS,
    _stub_forward,
)

WARNING = LettuceDetectDeprecationWarning


def _construct(tokenizer, model=None, **kwargs) -> TransformerDetector:
    """Build a TransformerDetector from a test module: this warns."""
    model = model or MagicMock(side_effect=_stub_forward)
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForTokenClassification.from_pretrained", return_value=model),
        pytest.warns(WARNING),
    ):
        return TransformerDetector(model_path="dummy_path", **kwargs)


def _encoder(tokenizer, max_length=512) -> EncoderDetector:
    return EncoderDetector(
        MagicMock(side_effect=_stub_forward),
        tokenizer,
        config=EncoderConfig(max_length=max_length),
        model_id="dummy_path",
    )


def _sources(context, answer, question=None) -> SourcesInput:
    return SourcesInput(
        sources=[Chunk(id=str(i), text=t) for i, t in enumerate(context)],
        answer=answer,
        question=question,
    )


def _bare(tokenizer, model=None) -> TransformerDetector:
    """Make a detector with ``__new__`` and only the six attributes set by hand."""
    with pytest.warns(WARNING):
        det = TransformerDetector.__new__(TransformerDetector)
    det.model = model or MagicMock(side_effect=_stub_forward)
    det.tokenizer = tokenizer
    det.device = torch.device("cpu")
    det.max_length, det.lang, det.typer = 512, "en", None
    return det


# -- deprecation --------------------------------------------------------------------------


def test_construction_warns_and_names_the_replacement(local_wordpiece_tokenizer):
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=local_wordpiece_tokenizer),
        patch(
            "transformers.AutoModelForTokenClassification.from_pretrained",
            return_value=MagicMock(side_effect=_stub_forward),
        ),
        pytest.warns(WARNING) as caught,
    ):
        TransformerDetector(model_path="dummy_path")
    assert [str(w.message) for w in caught] == [
        f"TransformerDetector is deprecated since {dep.DEPRECATED_SINCE} and will be removed in "
        f"{dep.REMOVED_IN}. Use EncoderDetector.from_pretrained."
    ]


def test_construction_inside_the_package_is_silent(local_wordpiece_tokenizer):
    namespace = {"__name__": "lettucedetect.fake_internal", "T": TransformerDetector}
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=local_wordpiece_tokenizer),
        patch(
            "transformers.AutoModelForTokenClassification.from_pretrained",
            return_value=MagicMock(side_effect=_stub_forward),
        ),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")
        exec("T(model_path='dummy_path')", namespace)  # noqa: S102
        make_detector("transformer", model_path="dummy_path")
        HallucinationDetector(method="transformer", model_path="dummy_path")


def test_is_an_encoder_detector_with_the_old_attributes(local_wordpiece_tokenizer):
    det = _construct(local_wordpiece_tokenizer, max_length=300, batch_size=4, device="meta")
    assert isinstance(det, EncoderDetector)
    assert det.model_id == "dummy_path"
    assert det.tokenizer is local_wordpiece_tokenizer
    assert (det.max_length, det.lang, det.batch_size, det.typer) == (300, "en", 4, None)
    assert det.device == torch.device("meta")  # the requested device, not the stub model's
    assert TransformerDetector.__deprecated__.endswith("EncoderDetector.from_pretrained.")  # ty: ignore[unresolved-attribute]
    assert ".. deprecated::" in TransformerDetector.__doc__
    for name in (
        "_predict_single",
        "_predict_batch",
        "_predict_chunked",
        "_build_spans_from_tokens",
    ):
        assert callable(getattr(det, name))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"lang": "xx"}, "Invalid language"), ({"batch_size": 0}, "batch_size must be >= 1")],
)
def test_bad_options_still_raise_value_error(kwargs, message):
    with pytest.warns(WARNING), pytest.raises(ValueError, match=message):
        TransformerDetector(model_path="dummy_path", **kwargs)


def test_loader_os_error_is_not_translated():
    """Old callers see the raw loader error."""
    boom = MagicMock(side_effect=OSError("gone"))
    with (
        patch("transformers.AutoTokenizer.from_pretrained", boom),
        pytest.warns(WARNING),
        pytest.raises(OSError, match="gone"),
    ):
        TransformerDetector(model_path="dummy_path")


def test_construction_logs_model_loaded_once(local_wordpiece_tokenizer, caplog):
    caplog.set_level(logging.DEBUG)
    det = _construct(local_wordpiece_tokenizer)
    det.predict(*PREDICT_INPUTS[0][:1], PREDICT_INPUTS[0][1])
    loud = [r for r in caplog.records if r.levelno >= logging.INFO]
    assert [r.event for r in loud] == ["model_loaded"]
    assert loud[0].args[0] == "dummy_path"


# -- detect and adetect -------------------------------------------------------------------


def test_detect_and_adetect_equal_an_encoder_detector_without_warning(local_wordpiece_tokenizer):
    old = _construct(local_wordpiece_tokenizer)
    new = _encoder(local_wordpiece_tokenizer)
    requests = [_sources(*inp) for inp in PREDICT_INPUTS] + [
        PromptInput(prompt=p, answer=a) for p, a in zip(PROMPTS, ANSWERS)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for request in requests:
            assert old.detect(request, include_tokens=True) == new.detect(
                request, include_tokens=True
            )
            assert asyncio.run(old.adetect(request)) == new.detect(request)


# -- predict output and __new__ -----------------------------------------------------------


def test_predict_and_predict_prompt_equal_the_golden_files(local_wordpiece_tokenizer):
    det = _construct(local_wordpiece_tokenizer)
    for fmt in ("tokens", "spans"):
        assert_golden(
            f"transformer_predict_{fmt}",
            [det.predict(c, a, q, output_format=fmt) for c, a, q in PREDICT_INPUTS],
        )
        assert_golden(
            f"transformer_predict_prompt_{fmt}",
            [det.predict_prompt(p, a, output_format=fmt) for p, a in zip(PROMPTS, ANSWERS)],
        )


def test_new_with_six_attributes_answers_predict_and_detect(local_wordpiece_tokenizer):
    det = _bare(local_wordpiece_tokenizer)
    assert_golden(
        "transformer_predict_spans",
        [det.predict(c, a, q, output_format="spans") for c, a, q in PREDICT_INPUTS],
    )
    assert det.detect(_sources(*PREDICT_INPUTS[0])).spans is not None


def test_new_answers_concurrent_predict_with_one_lock(local_wordpiece_tokenizer):
    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    def forward(input_ids=None, attention_mask=None, **kwargs):
        with guard:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.005)
        with guard:
            state["active"] -= 1
        return _stub_forward(input_ids=input_ids, attention_mask=attention_mask)

    det = _bare(local_wordpiece_tokenizer, MagicMock(side_effect=forward))
    inputs = [PREDICT_INPUTS[i % len(PREDICT_INPUTS)] for i in range(16)]
    expected = [det.predict(c, a, q, output_format="spans") for c, a, q in inputs]
    prompt_expected = [
        det.predict_prompt(p, a, output_format="spans") for p, a in zip(PROMPTS, ANSWERS)
    ]
    state["max"] = 0
    det.__dict__.pop("_scoring_lock", None)  # race the first creation again

    created, real = [], threading.RLock

    def slow_rlock():
        time.sleep(0.05)
        created.append(real())
        return created[-1]

    barrier = threading.Barrier(16)
    results: dict[int, list] = {}

    def work(i):
        barrier.wait()
        c, a, q = inputs[i]
        results[i] = det.predict(c, a, q, output_format="spans")

    with patch("lettucedetect.detectors.encoder.threading.RLock", slow_rlock):
        threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert len(created) == 1
    assert [results[i] for i in range(16)] == expected
    assert state["max"] == 1
    assert [
        det.predict_prompt(p, a, output_format="spans") for p, a in zip(PROMPTS, ANSWERS)
    ] == prompt_expected


# -- from_pretrained on the subclass ------------------------------------------------------


def test_inherited_from_pretrained_builds_a_transformer_detector(local_wordpiece_tokenizer):
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=local_wordpiece_tokenizer),
        patch(
            "transformers.AutoModelForTokenClassification.from_pretrained",
            return_value=MagicMock(side_effect=_stub_forward),
        ),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")  # the instance is made inside the package: silent
        det = TransformerDetector.from_pretrained("dummy_path", device="cpu")
    assert type(det) is TransformerDetector
    assert det.model_id == "dummy_path"
