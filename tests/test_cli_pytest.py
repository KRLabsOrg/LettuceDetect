"""Pytest tests for the lettucedetect CLI entry point (lettucedetect/cli.py)."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lettucedetect.cli import _build_parser, _read_source, main


class TestReadSource:
    """Tests for the _read_source helper."""

    def test_returns_literal_when_no_file(self) -> None:
        """Non-path strings are returned as-is."""
        assert _read_source("hello world") == "hello world"

    def test_reads_file_when_path_exists(self, tmp_path: Path) -> None:
        """Content is read from disk when the path points to an existing file."""
        f = tmp_path / "ctx.txt"
        f.write_text("file content", encoding="utf-8")
        assert _read_source(str(f)) == "file content"

    def test_returns_string_that_looks_like_nonexistent_path(self) -> None:
        """A path-like string for a non-existent file is returned as-is."""
        result = _read_source("/nonexistent/path/that/does/not/exist.txt")
        assert result == "/nonexistent/path/that/does/not/exist.txt"

    def test_stdin_sentinel_uses_stdin_text_arg(self) -> None:
        """When value is '-' and stdin_text is provided, stdin_text is returned."""
        assert _read_source("-", stdin_text="piped content") == "piped content"

    def test_stdin_sentinel_reads_sys_stdin_when_no_stdin_text(self) -> None:
        """When value is '-' and stdin_text is None, sys.stdin is read."""
        with patch("sys.stdin", StringIO("stdin content")):
            assert _read_source("-") == "stdin content"


class TestBuildParser:
    """Tests for the argument parser returned by _build_parser."""

    def setup_method(self) -> None:
        """Create a fresh parser for each test."""
        self.parser = _build_parser()

    def test_required_args_missing_exits(self) -> None:
        """Omitting all required args raises SystemExit."""
        with pytest.raises(SystemExit):
            self.parser.parse_args([])

    def test_minimal_valid_args_parsed(self) -> None:
        """Minimal required args are parsed correctly."""
        args = self.parser.parse_args(
            ["--model", "my-model", "--context", "ctx", "--answer", "ans"]
        )
        assert args.model == "my-model"
        assert args.context == "ctx"
        assert args.answer == "ans"

    def test_default_format_is_spans(self) -> None:
        """Output format defaults to 'spans'."""
        args = self.parser.parse_args(
            ["--model", "m", "--context", "c", "--answer", "a"]
        )
        assert args.output_format == "spans"

    def test_default_method_is_transformer(self) -> None:
        """Detection method defaults to 'transformer'."""
        args = self.parser.parse_args(
            ["--model", "m", "--context", "c", "--answer", "a"]
        )
        assert args.method == "transformer"

    def test_format_tokens_accepted(self) -> None:
        """--format tokens is a valid choice."""
        args = self.parser.parse_args(
            ["--model", "m", "--context", "c", "--answer", "a", "--format", "tokens"]
        )
        assert args.output_format == "tokens"

    def test_invalid_format_exits(self) -> None:
        """An unrecognised --format value raises SystemExit."""
        with pytest.raises(SystemExit):
            self.parser.parse_args(
                ["--model", "m", "--context", "c", "--answer", "a", "--format", "invalid"]
            )

    def test_question_captured(self) -> None:
        """--question is captured correctly when provided."""
        args = self.parser.parse_args(
            ["--model", "m", "--context", "c", "--answer", "a", "--question", "Q?"]
        )
        assert args.question == "Q?"

    def test_question_defaults_to_none(self) -> None:
        """--question defaults to None when omitted."""
        args = self.parser.parse_args(
            ["--model", "m", "--context", "c", "--answer", "a"]
        )
        assert args.question is None

    def test_method_llm_accepted(self) -> None:
        """--method llm is a valid choice."""
        args = self.parser.parse_args(
            ["--model", "m", "--context", "c", "--answer", "a", "--method", "llm"]
        )
        assert args.method == "llm"

    def test_stdin_sentinel_accepted_for_context(self) -> None:
        """'-' is accepted as the --context value."""
        args = self.parser.parse_args(
            ["--model", "m", "--context", "-", "--answer", "ans"]
        )
        assert args.context == "-"

    def test_stdin_sentinel_accepted_for_answer(self) -> None:
        """'-' is accepted as the --answer value."""
        args = self.parser.parse_args(
            ["--model", "m", "--context", "ctx", "--answer", "-"]
        )
        assert args.answer == "-"


_MOCK_DETECTOR_PATH = "lettucedetect.cli.HallucinationDetector"


def _make_mock_detector(predictions: list) -> MagicMock:
    """Return a MagicMock detector whose predict() returns *predictions*."""
    mock = MagicMock()
    mock.predict.return_value = predictions
    return mock


class TestMain:
    """Tests for the main() entry point using a mocked HallucinationDetector."""

    def _run(
        self,
        argv: list[str],
        predictions: list | None = None,
        capsys: pytest.CaptureFixture | None = None,
        stdin_text: str = "",
    ) -> str:
        """Invoke main() with *argv* and return captured stdout."""
        if predictions is None:
            predictions = []

        mock_instance = _make_mock_detector(predictions)

        with (
            patch("sys.argv", ["lettucedetect"] + argv),
            patch("sys.stdin", StringIO(stdin_text)),
            patch(_MOCK_DETECTOR_PATH, return_value=mock_instance) as MockCls,
        ):
            main()
            self._last_mock_cls = MockCls
            self._last_mock_instance = mock_instance

        if capsys is not None:
            return capsys.readouterr().out
        return ""

    # -- happy path ----------------------------------------------------------

    def test_outputs_valid_json(self, capsys: pytest.CaptureFixture) -> None:
        """Predictions are printed as valid JSON to stdout."""
        predictions = [{"start": 0, "end": 5, "text": "Berlin", "label": "hallucination"}]
        out = self._run(
            ["--model", "m", "--context", "Paris is in France.", "--answer", "Paris is in Berlin."],
            predictions=predictions,
            capsys=capsys,
        )
        assert json.loads(out) == predictions

    def test_question_forwarded_to_predict(self, capsys: pytest.CaptureFixture) -> None:
        """The --question value is forwarded to detector.predict()."""
        self._run(
            [
                "--model", "m",
                "--context", "The sky is blue.",
                "--answer", "The sky is green.",
                "--question", "What color is the sky?",
            ],
            capsys=capsys,
        )
        kwargs = self._last_mock_instance.predict.call_args.kwargs
        assert kwargs.get("question") == "What color is the sky?"

    def test_format_spans_forwarded(self, capsys: pytest.CaptureFixture) -> None:
        """--format spans is forwarded to detector.predict() as output_format."""
        self._run(
            ["--model", "m", "--context", "ctx", "--answer", "ans", "--format", "spans"],
            capsys=capsys,
        )
        kwargs = self._last_mock_instance.predict.call_args.kwargs
        assert kwargs.get("output_format") == "spans"

    def test_format_tokens_forwarded(self, capsys: pytest.CaptureFixture) -> None:
        """--format tokens is forwarded to detector.predict() as output_format."""
        self._run(
            ["--model", "m", "--context", "ctx", "--answer", "ans", "--format", "tokens"],
            capsys=capsys,
        )
        kwargs = self._last_mock_instance.predict.call_args.kwargs
        assert kwargs.get("output_format") == "tokens"

    def test_multi_passage_context_split(self, capsys: pytest.CaptureFixture) -> None:
        """Context paragraphs separated by blank lines are split into a list."""
        multi = "Passage one.\n\nPassage two."
        self._run(
            ["--model", "m", "--context", multi, "--answer", "ans"],
            capsys=capsys,
        )
        kwargs = self._last_mock_instance.predict.call_args.kwargs
        assert kwargs.get("context") == ["Passage one.", "Passage two."]

    def test_context_from_file(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Context is read from a file when --context is a valid path."""
        ctx_file = tmp_path / "ctx.txt"
        ctx_file.write_text("File context.", encoding="utf-8")
        self._run(
            ["--model", "m", "--context", str(ctx_file), "--answer", "ans"],
            capsys=capsys,
        )
        kwargs = self._last_mock_instance.predict.call_args.kwargs
        assert kwargs.get("context") == ["File context."]

    def test_answer_from_file(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """Answer is read from a file when --answer is a valid path."""
        ans_file = tmp_path / "ans.txt"
        ans_file.write_text("File answer.", encoding="utf-8")
        self._run(
            ["--model", "m", "--context", "ctx", "--answer", str(ans_file)],
            capsys=capsys,
        )
        kwargs = self._last_mock_instance.predict.call_args.kwargs
        assert kwargs.get("answer") == "File answer."

    def test_context_from_stdin(self, capsys: pytest.CaptureFixture) -> None:
        """Context is read from stdin when --context is '-'."""
        self._run(
            ["--model", "m", "--context", "-", "--answer", "ans"],
            capsys=capsys,
            stdin_text="Stdin context.",
        )
        kwargs = self._last_mock_instance.predict.call_args.kwargs
        assert kwargs.get("context") == ["Stdin context."]

    def test_answer_from_stdin(self, capsys: pytest.CaptureFixture) -> None:
        """Answer is read from stdin when --answer is '-'."""
        self._run(
            ["--model", "m", "--context", "ctx", "--answer", "-"],
            capsys=capsys,
            stdin_text="Stdin answer.",
        )
        kwargs = self._last_mock_instance.predict.call_args.kwargs
        assert kwargs.get("answer") == "Stdin answer."

    def test_model_kwargs_passed_to_constructor(self, capsys: pytest.CaptureFixture) -> None:
        """model_path kwarg and method are forwarded to HallucinationDetector."""
        self._run(
            ["--model", "KRLabsOrg/my-model", "--context", "ctx", "--answer", "ans"],
            capsys=capsys,
        )
        self._last_mock_cls.assert_called_once_with(
            method="transformer", model_path="KRLabsOrg/my-model"
        )

    def test_empty_predictions_prints_empty_array(self, capsys: pytest.CaptureFixture) -> None:
        """An empty predictions list is printed as a JSON empty array."""
        out = self._run(
            ["--model", "m", "--context", "ctx", "--answer", "ans"],
            predictions=[],
            capsys=capsys,
        )
        assert json.loads(out) == []

    # -- error paths ---------------------------------------------------------

    def test_exits_when_both_stdin(self) -> None:
        """SystemExit when both --context and --answer are '-'."""
        with (
            patch("sys.argv", ["lettucedetect", "--model", "m", "--context", "-", "--answer", "-"]),
            patch(_MOCK_DETECTOR_PATH),
            pytest.raises(SystemExit),
        ):
            main()

    def test_exits_on_empty_context(self) -> None:
        """Whitespace-only context causes SystemExit."""
        with (
            patch("sys.argv", ["lettucedetect", "--model", "m", "--context", "   ", "--answer", "ans"]),
            patch(_MOCK_DETECTOR_PATH),
            pytest.raises(SystemExit),
        ):
            main()

    def test_exits_on_empty_answer(self) -> None:
        """Whitespace-only answer causes SystemExit."""
        with (
            patch("sys.argv", ["lettucedetect", "--model", "m", "--context", "ctx", "--answer", "   "]),
            patch(_MOCK_DETECTOR_PATH),
            pytest.raises(SystemExit),
        ):
            main()

    def test_exits_when_model_load_fails(self) -> None:
        """A failing HallucinationDetector constructor causes SystemExit."""
        with (
            patch("sys.argv", ["lettucedetect", "--model", "bad-model", "--context", "ctx", "--answer", "ans"]),
            patch(_MOCK_DETECTOR_PATH, side_effect=Exception("model not found")),
            pytest.raises(SystemExit),
        ):
            main()

    def test_exits_when_predict_fails(self) -> None:
        """A RuntimeError in detector.predict() causes SystemExit."""
        mock_instance = MagicMock()
        mock_instance.predict.side_effect = RuntimeError("GPU OOM")
        with (
            patch("sys.argv", ["lettucedetect", "--model", "m", "--context", "ctx", "--answer", "ans"]),
            patch(_MOCK_DETECTOR_PATH, return_value=mock_instance),
            pytest.raises(SystemExit),
        ):
            main()

    def test_missing_model_exits(self) -> None:
        """Omitting --model raises SystemExit."""
        with (
            patch("sys.argv", ["lettucedetect", "--context", "ctx", "--answer", "ans"]),
            pytest.raises(SystemExit),
        ):
            main()

    def test_missing_context_exits(self) -> None:
        """Omitting --context raises SystemExit."""
        with (
            patch("sys.argv", ["lettucedetect", "--model", "m", "--answer", "ans"]),
            pytest.raises(SystemExit),
        ):
            main()

    def test_missing_answer_exits(self) -> None:
        """Omitting --answer raises SystemExit."""
        with (
            patch("sys.argv", ["lettucedetect", "--model", "m", "--context", "ctx"]),
            pytest.raises(SystemExit),
        ):
            main()