"""Tests for the dataset-level hallucination-rate reporting script."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_report_module() -> ModuleType:
    """Load the reporting script without importing detector dependencies."""
    script = Path(__file__).parents[1] / "scripts" / "report_hallucination_rates.py"
    spec = importlib.util.spec_from_file_location("report_hallucination_rates", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StubDetector:
    """Network-free detector that flags answers containing the word hallucinated."""

    def __init__(self, *, fail_at: int | None = None) -> None:
        """Initialize the stub and optional one-based failure position."""
        self.calls: list[dict] = []
        self.fail_at = fail_at

    def predict(self, **kwargs):
        """Record the invocation and return a deterministic span list."""
        self.calls.append(kwargs)
        if self.fail_at == len(self.calls):
            raise ValueError("stub detector failed")
        answer = kwargs["answer"]
        if "hallucinated" not in answer:
            return []
        start = answer.index("hallucinated")
        return [
            {
                "start": start,
                "end": start + len("hallucinated"),
                "text": "hallucinated",
                "confidence": 0.9,
                "category": "factual",
                "subcategory": "contradiction",
            }
        ]


def make_record(module, index, group=None):
    """Build a canonical record for pure aggregation tests."""
    return {
        "row_number": index + 1,
        "context": [f"context {index}"],
        "answer": f"answer {index}",
        "question": None,
        "identifier": str(index),
        "group": group,
    }


class TestInputLoading:
    """Input normalization tests for both supported file formats."""

    def test_jsonl_normalizes_string_and_list_context_and_honors_limit(self, tmp_path):
        """JSONL string/list contexts become the same canonical shape."""
        module = load_report_module()
        path = tmp_path / "records.jsonl"
        rows = [
            {"id": 1, "context": "one", "answer": "a", "source": "alpha"},
            {
                "id": 2,
                "context": ["two", "three"],
                "question": "q?",
                "answer": "b",
            },
            {"id": 3, "context": "ignored", "answer": "c"},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

        records = module.load_records(path, group_by="source", limit=2)

        assert [record["context"] for record in records] == [["one"], ["two", "three"]]
        assert records[0]["identifier"] == "1"
        assert records[1]["question"] == "q?"
        assert records[1]["group"] == module.MISSING_GROUP

    def test_csv_accepts_plain_and_json_array_context(self, tmp_path):
        """CSV context cells support plain text and encoded string arrays."""
        module = load_report_module()
        path = tmp_path / "records.csv"
        path.write_text(
            'context,answer,source\nplain context,one,A\n"[""first"", ""second""]",two,B\n',
            encoding="utf-8",
        )

        records = module.load_records(path, group_by="source")

        assert records[0]["context"] == ["plain context"]
        assert records[1]["context"] == ["first", "second"]
        assert [record["group"] for record in records] == ["A", "B"]

    def test_csv_keeps_bracket_prefixed_prose_as_literal_context(self, tmp_path):
        """A bracket-prefixed prose context is not required to be a JSON array."""
        module = load_report_module()
        path = tmp_path / "records.csv"
        path.write_text(
            "context,answer\n[Introduction] The source says hello,answer\n",
            encoding="utf-8",
        )

        records = module.load_records(path)

        assert records[0]["context"] == ["[Introduction] The source says hello"]

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            ('{"context": ["ok", 3], "answer": "a"}\n', "every context item"),
            ('{"context": "ok"}\n', "missing required field.*answer"),
            ("not json\n", "invalid JSON"),
            ("\n", "blank JSONL"),
        ],
    )
    def test_jsonl_errors_include_line_number(self, tmp_path, content, match):
        """Malformed rows fail loudly with their physical input line."""
        module = load_report_module()
        path = tmp_path / "bad.jsonl"
        path.write_text(content, encoding="utf-8")

        with pytest.raises(ValueError, match=match) as error:
            module.load_records(path)

        assert f"{path}:1" in str(error.value)

    def test_csv_rejects_non_string_context_array_members(self, tmp_path):
        """A JSON-like CSV array cannot smuggle non-string passages."""
        module = load_report_module()
        path = tmp_path / "bad.csv"
        path.write_text('context,answer\n"[""ok"", 2]",answer\n', encoding="utf-8")

        with pytest.raises(ValueError, match="every context item") as error:
            module.load_records(path)

        assert f"{path}:2" in str(error.value)


class TestAggregation:
    """Pure aggregation tests independent of any detector or model."""

    def test_rates_histograms_typed_counts_and_groups_reconcile(self):
        """Every requested count uses the scored-row population consistently."""
        module = load_report_module()
        pairs = [
            {"record": make_record(module, 0, "A"), "spans": []},
            {
                "record": make_record(module, 1, "A"),
                "spans": [
                    {
                        "text": "x",
                        "confidence": 0.0,
                        "category": "factual",
                        "subcategory": "contradiction",
                    }
                ],
            },
            {
                "record": make_record(module, 2, module.MISSING_GROUP),
                "spans": [
                    {"text": "y", "confidence": 0.1, "category": "factual"},
                    {"text": "z", "confidence": 1.0, "subcategory": "fabrication"},
                ],
            },
        ]

        report = module.aggregate_results(pairs, group_by="source", top_n=5)

        assert report["summary"] == {
            "rows_in": 3,
            "rows_scored": 3,
            "flagged_answers": 2,
            "hallucination_rate": {"numerator": 2, "denominator": 3, "value": 2 / 3},
        }
        assert report["groups"]["rates"]["A"]["denominator"] == 2
        assert report["groups"]["rates"]["A"]["numerator"] == 1
        assert report["groups"]["rates"][module.MISSING_GROUP]["denominator"] == 1
        assert report["groups"]["denominator_sum"] == 3
        assert report["span_count_histogram"] == {"0": 1, "1": 1, "2": 1}
        assert report["span_confidence_histogram"]["bins"]["[0.0, 0.1)"] == 1
        assert report["span_confidence_histogram"]["bins"]["[0.1, 0.2)"] == 1
        assert report["span_confidence_histogram"]["bins"]["[0.9, 1.0]"] == 1
        assert report["span_confidence_histogram"]["all_spans"] == 3
        assert report["category_counts"] == {"factual": 2}
        assert report["subcategory_counts"] == {"contradiction": 1, "fabrication": 1}

    def test_confidence_accounting_handles_missing_and_invalid_values(self):
        """Bad confidence values are counted separately and remain valid JSON."""
        module = load_report_module()
        pairs = [
            {
                "record": make_record(module, 0),
                "spans": [
                    {"text": "a"},
                    {"text": "b", "confidence": None},
                    {"text": "c", "confidence": float("nan")},
                    {"text": "d", "confidence": True},
                    {"text": "e", "confidence": 1.2},
                    {"text": "f", "confidence": 0.55},
                ],
            }
        ]

        report = module.aggregate_results(pairs, top_n=1)
        histogram = report["span_confidence_histogram"]

        assert histogram["valid_confidence_spans"] == 1
        assert histogram["missing_confidence_spans"] == 2
        assert histogram["invalid_confidence_spans"] == 3
        assert histogram["all_spans"] == 6
        assert report["top_flagged_examples"][0]["spans"][2]["confidence"] is None
        json.dumps(report, allow_nan=False)

    def test_top_n_ranks_confidence_stably_then_keeps_missing_confidence(self):
        """Flagged examples use maximum confidence and stable source-order ties."""
        module = load_report_module()
        pairs = [
            {
                "record": make_record(module, 0),
                "spans": [{"confidence": 0.6}, {"confidence": 0.9}],
            },
            {"record": make_record(module, 1), "spans": [{"confidence": 0.9}]},
            {"record": make_record(module, 2), "spans": [{"text": "no score"}]},
            {"record": make_record(module, 3), "spans": []},
        ]

        report = module.aggregate_results(pairs, top_n=99)
        examples = report["top_flagged_examples"]

        assert [example["id"] for example in examples] == ["0", "1", "2"]
        assert examples[2]["max_confidence"] is None
        assert module.aggregate_results(pairs, top_n=0)["top_flagged_examples"] == []

    def test_markdown_uses_the_same_report_values(self):
        """Markdown exposes the canonical JSON counts, denominator, and examples."""
        module = load_report_module()
        pair = {
            "record": make_record(module, 0, "docs"),
            "spans": [{"text": "answer", "confidence": 0.8, "category": "factual"}],
        }
        report = module.aggregate_results([pair], group_by="source", top_n=1)

        markdown = module.render_markdown(report)

        assert "100.00% (1/1)" in markdown
        assert "| docs | 1 | 1 | 100.00% |" in markdown
        assert "| `[0.8, 0.9)` | 1 |" in markdown
        assert "factual" in markdown
        assert "answer" in markdown


class TestRunnerAndCli:
    """Network-free scoring and CLI-shaped integration tests."""

    def test_score_records_forwards_every_required_predict_argument(self):
        """Each selected row produces exactly one span prediction."""
        module = load_report_module()
        detector = StubDetector()
        records = [make_record(module, index) for index in range(3)]

        scored = module.score_records(detector, records, min_confidence=0.42, progress=False)

        assert len(scored) == 3
        assert len(detector.calls) == 3
        assert all(call["output_format"] == "spans" for call in detector.calls)
        assert all(call["min_confidence"] == 0.42 for call in detector.calls)
        assert detector.calls[0]["context"] == ["context 0"]

    def test_ten_row_cli_shaped_run_writes_json_and_markdown(self, tmp_path, caplog):
        """The accepted command shape completes over ten rows without a model."""
        module = load_report_module()
        input_path = tmp_path / "sample.jsonl"
        output_json = tmp_path / "nested" / "report.json"
        output_markdown = tmp_path / "nested" / "report.md"
        rows = [
            {
                "id": index,
                "context": [f"source passage {index}"],
                "question": f"question {index}",
                "answer": (
                    f"hallucinated answer {index}" if index % 2 else f"supported answer {index}"
                ),
                "source": "odd" if index % 2 else "even",
            }
            for index in range(10)
        ]
        input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        detector = StubDetector()

        with caplog.at_level(module.logging.INFO, logger=module.__name__):
            result = module.main(
                [
                    "--input",
                    str(input_path),
                    "--method",
                    "transformer",
                    "--model-path",
                    "unused-stub-model",
                    "--min-confidence",
                    "0.7",
                    "--group-by",
                    "source",
                    "--out-json",
                    str(output_json),
                    "--out-md",
                    str(output_markdown),
                ],
                detector=detector,
            )

        assert result == 0
        assert len(detector.calls) == 10
        report = json.loads(output_json.read_text(encoding="utf-8"))
        assert report["summary"]["rows_in"] == 10
        assert report["summary"]["rows_scored"] == 10
        assert report["summary"]["hallucination_rate"]["denominator"] == 10
        assert report["summary"]["hallucination_rate"]["numerator"] == 5
        assert report["groups"]["denominator_sum"] == 10
        assert "Hallucination rate report" in output_markdown.read_text(encoding="utf-8")
        assert caplog.messages[-1] == "Scored 10/10 records"

    def test_detector_failure_does_not_leave_reports(self, tmp_path):
        """A failed scoring run never writes a successful-looking partial artifact."""
        module = load_report_module()
        input_path = tmp_path / "sample.jsonl"
        output_json = tmp_path / "report.json"
        output_markdown = tmp_path / "report.md"
        input_path.write_text(
            "\n".join(
                json.dumps({"context": "c", "answer": f"answer {index}"}) for index in range(3)
            ),
            encoding="utf-8",
        )

        with pytest.raises(SystemExit) as error:
            module.main(
                [
                    "--input",
                    str(input_path),
                    "--model-path",
                    "unused",
                    "--out-json",
                    str(output_json),
                    "--out-md",
                    str(output_markdown),
                ],
                detector=StubDetector(fail_at=2),
            )

        assert error.value.code == 1
        assert not output_json.exists()
        assert not output_markdown.exists()

    @pytest.mark.parametrize("value", ["-0.1", "1.1", "nan", "inf"])
    def test_parser_rejects_invalid_min_confidence(self, value):
        """The CLI validates the detector threshold before any work begins."""
        module = load_report_module()

        with pytest.raises(SystemExit):
            module.build_parser().parse_args(
                [
                    "--input",
                    "in.jsonl",
                    "--model-path",
                    "model",
                    "--min-confidence",
                    value,
                    "--out-json",
                    "out.json",
                    "--out-md",
                    "out.md",
                ]
            )

    @pytest.mark.parametrize(("flag", "value"), [("--limit", "0"), ("--top-n", "-1")])
    def test_parser_rejects_invalid_count_options(self, flag, value):
        """Limit is positive and top-N is non-negative by contract."""
        module = load_report_module()

        with pytest.raises(SystemExit):
            module.build_parser().parse_args(
                [
                    "--input",
                    "in.jsonl",
                    "--model-path",
                    "model",
                    flag,
                    value,
                    "--out-json",
                    "out.json",
                    "--out-md",
                    "out.md",
                ]
            )

    def test_create_detector_rejects_transformer_only_taxonomy_head(self):
        """Detector construction owns validation of method-specific flags."""
        module = load_report_module()

        with pytest.raises(ValueError, match="only supported"):
            module.create_detector("llm", "model", "en", "head")

    def test_main_surfaces_detector_configuration_value_error(self, tmp_path, monkeypatch):
        """A detector configuration ValueError becomes a clean CLI exit code 1."""
        module = load_report_module()
        input_path = tmp_path / "sample.jsonl"
        input_path.write_text('{"context": "c", "answer": "a"}\n', encoding="utf-8")
        monkeypatch.setattr(
            module,
            "create_detector",
            lambda *_args: (_ for _ in ()).throw(ValueError("invalid detector configuration")),
        )

        with pytest.raises(SystemExit) as error:
            module.main(
                [
                    "--input",
                    str(input_path),
                    "--method",
                    "llm",
                    "--model-path",
                    "model",
                    "--taxonomy-head",
                    "head",
                    "--out-json",
                    str(tmp_path / "out.json"),
                    "--out-md",
                    str(tmp_path / "out.md"),
                ],
            )

        assert error.value.code == 1
