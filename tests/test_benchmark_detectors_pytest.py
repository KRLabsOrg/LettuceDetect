"""Tests for the detector benchmark helper script."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest


def load_benchmark_module() -> ModuleType:
    """Load the benchmark helper as a module."""
    script = Path(__file__).parents[1] / "scripts" / "benchmark_detectors.py"
    spec = importlib.util.spec_from_file_location("benchmark_detectors", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestDetectorBenchmark:
    """Tests for the detector benchmark helper script."""

    def test_run_benchmark_reports_latency_throughput_device_and_memory(self):
        """Benchmark helper measures each case after warmup and records metadata."""
        module = load_benchmark_module()
        detector = MagicMock()
        detector.predict.return_value = []
        detector.detector.device = "cpu"
        cases = [
            {
                "name": "short",
                "context": ["France is in Europe."],
                "question": "Where is France?",
                "answer": "France is in Europe.",
                "context_tokens": 4,
            },
            {
                "name": "medium",
                "context": ["Paris is the capital of France."],
                "question": "What is the capital?",
                "answer": "Paris is the capital of France.",
                "context_tokens": 6,
            },
        ]

        result = module.run_benchmark(
            detector,
            cases,
            method="transformer",
            model_path="dummy-model",
            warmup=1,
            repeats=2,
            output_format="spans",
        )

        assert detector.predict.call_count == 6
        assert result.method == "transformer"
        assert result.device == "cpu"
        assert result.cases == 2
        assert result.context_lengths == [4, 6]
        assert result.repeats == 2
        assert result.latency_mean_ms >= 0
        assert result.throughput_cases_per_second > 0
        assert result.peak_memory_bytes > 0
        assert result.peak_memory_source == "resource.getrusage(RUSAGE_SELF).ru_maxrss"
        assert [case.name for case in result.case_results] == ["short", "medium"]

    def test_run_benchmark_resets_and_reports_cuda_peak_memory(self):
        """GPU benchmarks reset and read torch CUDA peak memory counters."""
        module = load_benchmark_module()
        detector = MagicMock()
        detector.predict.return_value = []
        detector.detector.device = "cuda"
        cases = [module.make_synthetic_case(8)]

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.max_memory_allocated.return_value = 123456

        with patch.object(module, "import_torch", return_value=fake_torch):
            result = module.run_benchmark(
                detector,
                cases,
                method="transformer",
                model_path="dummy-model",
                warmup=0,
                repeats=1,
                output_format="spans",
            )

        fake_torch.cuda.reset_peak_memory_stats.assert_called_once_with("cuda")
        fake_torch.cuda.max_memory_allocated.assert_called_once_with("cuda")
        assert result.peak_memory_bytes == 123456
        assert result.peak_memory_source == "torch.cuda.max_memory_allocated"

    def test_load_cases_requires_context_and_answer(self, tmp_path):
        """JSONL benchmark cases must include the fields used by predict()."""
        module = load_benchmark_module()
        cases_path = tmp_path / "cases.jsonl"
        cases_path.write_text('{"context": ["only context"]}\n', encoding="utf-8")

        with pytest.raises(ValueError, match="answer"):
            module.load_cases(cases_path)

    def test_default_cases_cover_requested_context_lengths(self):
        """Built-in synthetic cases cover the issue's ~512/2k/8k sweep."""
        module = load_benchmark_module()

        cases = module.load_cases(None)

        assert [case["context_tokens"] for case in cases] == [512, 2048, 8192]
        assert [case["name"] for case in cases] == [
            "synthetic-512-tokens",
            "synthetic-2048-tokens",
            "synthetic-8192-tokens",
        ]
        assert all(case["context"] and case["answer"] for case in cases)

    def test_parse_context_lengths_rejects_empty_or_non_positive_values(self):
        """CLI parser validates synthetic context-length configuration."""
        module = load_benchmark_module()

        with pytest.raises(module.argparse.ArgumentTypeError):
            module.parse_context_lengths("")
        with pytest.raises(module.argparse.ArgumentTypeError):
            module.parse_context_lengths("512,0")

    def test_text_output_prints_each_context_length(self, capsys):
        """Plain output includes the per-case latency table requested by the benchmark issue."""
        module = load_benchmark_module()
        detector = MagicMock()
        detector.predict.return_value = []
        detector.detector.device = "cpu"

        result = module.run_benchmark(
            detector,
            [module.make_synthetic_case(512), module.make_synthetic_case(2048)],
            method="transformer",
            model_path="dummy-model",
            warmup=0,
            repeats=1,
            output_format="spans",
        )
        module.print_text_result(result)

        output = capsys.readouterr().out
        assert "Case | context tokens | mean ms | median ms | p95 ms" in output
        assert "synthetic-512-tokens | 512 |" in output
        assert "synthetic-2048-tokens | 2048 |" in output

    def test_context_length_help_explains_whitespace_tokens_and_truncation(self):
        """CLI help documents how synthetic context lengths are counted."""
        module = load_benchmark_module()

        help_text = module.build_parser().format_help()

        assert "whitespace-token lengths" in help_text
        assert "may measure truncation" in help_text

    def test_main_rejects_device_for_non_transformer_methods(self):
        """Device selection is only threaded to transformer detectors."""
        module = load_benchmark_module()

        with pytest.raises(SystemExit):
            module.main(["--method", "llm", "--device", "cpu"])
