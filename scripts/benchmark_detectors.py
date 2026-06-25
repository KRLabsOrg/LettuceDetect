"""Latency and throughput benchmark for LettuceDetect detectors."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict, cast

DEFAULT_CONTEXT = ["France is a country in Europe. The capital of France is Paris."]
DEFAULT_QUESTION = "What is the capital of France?"
DEFAULT_ANSWER = "The capital of France is Paris."


class BenchmarkCase(TypedDict):
    """Input case for detector benchmarking."""

    context: list[str]
    answer: str
    question: NotRequired[str | None]


class PredictDetector(Protocol):
    """Detector interface needed by the benchmark runner."""

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
    ) -> list:
        """Predict hallucination tokens or spans."""
        ...


@dataclass(frozen=True)
class BenchmarkResult:
    """Summary statistics for a detector benchmark run."""

    method: str
    model_path: str | None
    cases: int
    warmup: int
    repeats: int
    output_format: str
    total_seconds: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    throughput_cases_per_second: float


def load_cases(path: Path | None) -> list[BenchmarkCase]:
    """Load benchmark cases from JSONL or return a small built-in sample."""
    if path is None:
        return [
            {
                "context": DEFAULT_CONTEXT,
                "question": DEFAULT_QUESTION,
                "answer": DEFAULT_ANSWER,
            }
        ]

    cases: list[BenchmarkCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        missing = {"context", "answer"} - item.keys()
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"{path}:{line_number} is missing required field(s): {fields}")
        cases.append(cast(BenchmarkCase, item))

    if not cases:
        raise ValueError(f"{path} does not contain any benchmark cases")
    return cases


def run_benchmark(
    detector: PredictDetector,
    cases: list[BenchmarkCase],
    *,
    method: str,
    model_path: str | None,
    warmup: int,
    repeats: int,
    output_format: str,
) -> BenchmarkResult:
    """Run warmup and measured detector predictions over the supplied cases."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be zero or greater")

    for _ in range(warmup):
        for case in cases:
            detector.predict(
                context=case["context"],
                question=case.get("question"),
                answer=case["answer"],
                output_format=output_format,
            )

    latencies: list[float] = []
    total_start = time.perf_counter()
    for _ in range(repeats):
        for case in cases:
            start = time.perf_counter()
            detector.predict(
                context=case["context"],
                question=case.get("question"),
                answer=case["answer"],
                output_format=output_format,
            )
            latencies.append(time.perf_counter() - start)
    total_seconds = time.perf_counter() - total_start

    latencies_ms = [value * 1000 for value in latencies]
    sorted_latencies = sorted(latencies_ms)
    p95_index = min(len(sorted_latencies) - 1, int(len(sorted_latencies) * 0.95))

    return BenchmarkResult(
        method=method,
        model_path=model_path,
        cases=len(cases),
        warmup=warmup,
        repeats=repeats,
        output_format=output_format,
        total_seconds=total_seconds,
        latency_mean_ms=statistics.fmean(latencies_ms),
        latency_median_ms=statistics.median(latencies_ms),
        latency_p95_ms=sorted_latencies[p95_index],
        throughput_cases_per_second=len(latencies) / total_seconds if total_seconds else 0.0,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", default="transformer", choices=["transformer", "llm", "rag_fact_checker"]
    )
    parser.add_argument("--model-path", help="Model path passed to HallucinationDetector")
    parser.add_argument(
        "--cases", type=Path, help="JSONL file with context, question, and answer fields"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Warmup iterations before measurement"
    )
    parser.add_argument("--repeats", type=int, default=5, help="Measured iterations over all cases")
    parser.add_argument("--output-format", default="spans", choices=["tokens", "spans"])
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the detector benchmark."""
    parser = build_parser()
    args = parser.parse_args(argv)

    from lettucedetect.models.inference import HallucinationDetector

    detector_kwargs: dict[str, str] = {}
    if args.model_path:
        detector_kwargs["model_path"] = args.model_path
    detector = HallucinationDetector(method=args.method, **detector_kwargs)
    result = run_benchmark(
        detector,
        load_cases(args.cases),
        method=args.method,
        model_path=args.model_path,
        warmup=args.warmup,
        repeats=args.repeats,
        output_format=args.output_format,
    )

    payload = asdict(result)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "Latency mean: "
            f"{result.latency_mean_ms:.2f} ms, "
            f"p95: {result.latency_p95_ms:.2f} ms, "
            f"throughput: {result.throughput_cases_per_second:.2f} cases/s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
