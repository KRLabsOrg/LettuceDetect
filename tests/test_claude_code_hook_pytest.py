"""Tests for the Claude Code Stop-hook integration (no model, no network)."""

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lettucedetect.integrations.claude_code.check_answer import (
    format_report,
    load_context,
    main,
    parse_transcript,
)

FIXTURE = Path(__file__).parent / "fixtures" / "claude_code_transcript.jsonl"


class TestParseTranscript:
    """parse_transcript pure-function behavior."""

    def test_extracts_last_user_and_assistant_messages(self):
        """Test extracts last user and assistant messages."""
        question, answer = parse_transcript(FIXTURE)
        assert question == "What is the population of France?"
        assert answer == (
            "The capital of France is Paris.\nThe population of France is 69 million."
        )

    def test_missing_file_returns_nones(self):
        """Test missing file returns nones."""
        question, answer = parse_transcript("does/not/exist.jsonl")
        assert question is None and answer is None

    def test_string_content_and_skipped_junk(self, tmp_path):
        """Test string content and skipped junk."""
        p = tmp_path / "t.jsonl"
        p.write_text(
            json.dumps({"message": {"role": "user", "content": "hi"}})
            + "\n"
            + "not json\n"
            + json.dumps({"message": {"role": "assistant", "content": "hello"}})
            + "\n"
        )
        assert parse_transcript(p) == ("hi", "hello")


class TestLoadContext:
    """load_context file handling."""

    def test_reads_existing_skips_missing_and_empty(self, tmp_path):
        """Test reads existing skips missing and empty."""
        a = tmp_path / "a.md"
        a.write_text("passage one")
        empty = tmp_path / "empty.md"
        empty.write_text("   ")
        contexts = load_context([str(a), str(empty), str(tmp_path / "missing.md")])
        assert contexts == ["passage one"]


class TestFormatReport:
    """format_report output for typed and untyped spans."""

    def test_untyped_span(self):
        """Test untyped span."""
        report = format_report([{"text": " 69 million ", "hallucination_score": 0.93}])
        assert '"69 million" (confidence 0.93)' in report
        assert report.startswith("LettuceDetect flagged 1 unsupported span(s)")

    def test_typed_unsupported_addition_gets_note(self):
        """Test typed unsupported addition gets note."""
        report = format_report(
            [
                {
                    "text": "added retry logic",
                    "confidence": 0.88,
                    "category": "unsupported_addition",
                    "subcategory": "behavior",
                }
            ]
        )
        assert "unsupported_addition/behavior" in report
        assert "the request did not ask for this" in report

    def test_multiple_spans_counted(self):
        """Test multiple spans counted."""
        report = format_report([{"text": "a", "confidence": 0.9}, {"text": "b", "confidence": 0.8}])
        assert "2 unsupported span(s)" in report


def run_main(argv, event, spans=None, tmp_path=None):
    """Run main() with stubbed stdin and a stubbed local detector."""

    class StubDetector:
        """Stub in-process detector."""

        def __init__(self, **kwargs):
            pass

        def predict(self, **kwargs):
            return spans or []

    with (
        patch("sys.stdin", io.StringIO(json.dumps(event))),
        patch("lettucedetect.models.inference.HallucinationDetector", StubDetector),
    ):
        return main(argv)


class TestMain:
    """End-to-end main() with stubbed detectors."""

    def make_context(self, tmp_path):
        """Make context."""
        ctx = tmp_path / "context.md"
        ctx.write_text("France has 67 million inhabitants.")
        return str(ctx)

    def test_flagged_answer_exits_2_and_reports(self, tmp_path, capsys):
        """Test flagged answer exits 2 and reports."""
        ctx = self.make_context(tmp_path)
        code = run_main(
            ["--model-path", "stub", "--context-file", ctx],
            {"transcript_path": str(FIXTURE)},
            spans=[{"text": "69 million", "confidence": 0.95, "start": 0, "end": 10}],
        )
        assert code == 2
        assert "69 million" in capsys.readouterr().err

    def test_clean_answer_exits_0(self, tmp_path, capsys):
        """Test clean answer exits 0."""
        ctx = self.make_context(tmp_path)
        code = run_main(
            ["--model-path", "stub", "--context-file", ctx],
            {"transcript_path": str(FIXTURE)},
            spans=[],
        )
        assert code == 0
        assert capsys.readouterr().err == ""

    def test_stop_hook_active_short_circuits(self, tmp_path):
        """Test stop hook active short circuits."""
        ctx = self.make_context(tmp_path)
        code = run_main(
            ["--model-path", "stub", "--context-file", ctx],
            {"transcript_path": str(FIXTURE), "stop_hook_active": True},
            spans=[{"text": "x", "confidence": 0.99}],
        )
        assert code == 0

    def test_missing_context_exits_0(self, tmp_path):
        """Test missing context exits 0."""
        code = run_main(
            ["--model-path", "stub", "--context-file", str(tmp_path / "nope.md")],
            {"transcript_path": str(FIXTURE)},
            spans=[{"text": "x", "confidence": 0.99}],
        )
        assert code == 0

    def test_min_confidence_filters(self, tmp_path):
        """Test min confidence filters."""
        ctx = self.make_context(tmp_path)
        code = run_main(
            ["--model-path", "stub", "--context-file", ctx, "--min-confidence", "0.9"],
            {"transcript_path": str(FIXTURE)},
            spans=[{"text": "x", "confidence": 0.5}],
        )
        assert code == 0

    def test_invalid_stdin_exits_0(self):
        """Test invalid stdin exits 0."""
        with patch("sys.stdin", io.StringIO("not json")):
            assert main(["--model-path", "stub"]) == 0

    def test_api_mode_uses_client(self, tmp_path, capsys):
        """Test api mode uses client."""
        ctx = self.make_context(tmp_path)

        class StubItem:
            """Stub API span item."""

            def model_dump(self):
                return {"text": "69 million", "hallucination_score": 0.97, "start": 0, "end": 10}

        class StubResponse:
            """Stub span response."""

            predictions = (StubItem(),)

        class StubClient:
            """Stub HTTP client."""

            def __init__(self, url):
                self.url = url

            def detect_spans(self, contexts, question, answer):
                assert contexts and answer
                return StubResponse()

        with (
            patch("sys.stdin", io.StringIO(json.dumps({"transcript_path": str(FIXTURE)}))),
            patch("lettucedetect_api.client.LettuceClient", StubClient),
        ):
            code = main(["--api-url", "http://x", "--context-file", ctx])
        assert code == 2
        assert "0.97" in capsys.readouterr().err

    def test_llm_mode_passes_base_url_and_reports_typed_spans(self, tmp_path, capsys):
        """Test llm mode passes base url and reports typed spans."""
        ctx = self.make_context(tmp_path)
        seen = {}

        class StubDetector:
            """Stub LLM detector capturing constructor kwargs."""

            def __init__(self, **kwargs):
                seen.update(kwargs)

            def predict(self, **kwargs):
                return [
                    {
                        "text": "added retry logic",
                        "confidence": 0.9,
                        "category": "unsupported_addition",
                    }
                ]

        with (
            patch("sys.stdin", io.StringIO(json.dumps({"transcript_path": str(FIXTURE)}))),
            patch("lettucedetect.models.inference.HallucinationDetector", StubDetector),
        ):
            code = main(
                [
                    "--llm-model",
                    "KRLabsOrg/lettucedect-v2-qwen-2b",
                    "--llm-base-url",
                    "http://localhost:8001/v1",
                    "--context-file",
                    ctx,
                ]
            )
        assert code == 2
        assert seen == {
            "method": "llm",
            "model": "KRLabsOrg/lettucedect-v2-qwen-2b",
            "base_url": "http://localhost:8001/v1",
        }
        assert "the request did not ask for this" in capsys.readouterr().err


class TestArgs:
    """Argument validation."""

    def test_mode_required(self):
        """Test mode required."""
        with pytest.raises(SystemExit):
            main([])

    def test_modes_mutually_exclusive(self):
        """Test modes mutually exclusive."""
        with pytest.raises(SystemExit):
            main(["--api-url", "http://x", "--llm-model", "y"])
