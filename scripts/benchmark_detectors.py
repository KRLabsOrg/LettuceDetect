"""Latency, throughput, and memory benchmark for LettuceDetect detectors."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NotRequired, Protocol, TypedDict, cast

DEFAULT_QUESTION = "What is the capital of France?"
DEFAULT_ANSWER = "The capital of France is Paris."
DEFAULT_CONTEXT_LENGTHS = (512, 2048, 8192)
SYNTHETIC_SENTENCE = "France is a country in Europe and Paris is its capital city."


class CudaModule(Protocol):
    """Small CUDA API surface used for benchmark memory reporting."""

    def is_available(self) -> bool:
        """Return whether CUDA is available."""
        ...

    def reset_peak_memory_stats(self, device: str) -> None:
        """Reset peak memory stats for a device."""
        ...

    def max_memory_allocated(self, device: str) -> int:
        """Return the max allocated memory for a device."""
        ...


class TorchModule(Protocol):
    """Small torch API surface used by this script."""

    cuda: CudaModule


class BenchmarkCase(TypedDict):
    """Input case for detector benchmarking."""

    context: list[str]
    answer: str
    question: NotRequired[str | None]
    name: NotRequired[str]
    context_tokens: NotRequired[int]


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
class CaseBenchmarkResult:
    """Summary statistics for one benchmark case."""

    name: str
    context_tokens: int
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    throughput_cases_per_second: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Summary statistics for a detector benchmark run."""

    method: str
    model_path: str | None
    device: str
    cases: int
    context_lengths: list[int]
    warmup: int
    repeats: int
    output_format: str
    total_seconds: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    throughput_cases_per_second: float
    peak_memory_bytes: int
    peak_memory_source: str
    case_results: list[CaseBenchmarkResult]


def estimate_context_tokens(context: list[str]) -> int:
    """Estimate context length with whitespace tokens for reproducible case metadata."""
    return sum(len(item.split()) for item in context)


def make_synthetic_case(target_tokens: int) -> BenchmarkCase:
    """Create a deterministic synthetic case near the requested context-token length."""
    if target_tokens < 1:
        raise ValueError("context lengths must be positive integers")

    sentence_tokens = SYNTHETIC_SENTENCE.split()
    repeats = (target_tokens + len(sentence_tokens) - 1) // len(sentence_tokens)
    words = (sentence_tokens * repeats)[:target_tokens]
    context = " ".join(words)
    return {
        "name": f"synthetic-{target_tokens}-tokens",
        "context": [context],
        "question": DEFAULT_QUESTION,
        "answer": DEFAULT_ANSWER,
        "context_tokens": target_tokens,
    }


def parse_context_lengths(value: str) -> list[int]:
    """Parse a comma-separated context-length list."""
    try:
        lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "context lengths must be comma-separated integers"
        ) from exc
    if not lengths:
        raise argparse.ArgumentTypeError("at least one context length is required")
    if any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError("context lengths must be positive integers")
    return lengths


def load_cases(path: Path | None, context_lengths: list[int] | None = None) -> list[BenchmarkCase]:
    """Load benchmark cases from JSONL or return built-in synthetic context sweeps."""
    if path is None:
        lengths = context_lengths or list(DEFAULT_CONTEXT_LENGTHS)
        return [make_synthetic_case(length) for length in lengths]

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


def resolve_detector_device(detector: PredictDetector) -> str:
    """Return a readable device for the benchmark output."""
    inner_detector = getattr(detector, "detector", detector)
    device = getattr(inner_detector, "device", None)
    return str(device) if device is not None else "unknown"


def import_torch() -> TorchModule | None:
    """Import torch lazily so JSONL validation and tests do not require model deps."""
    try:
        import torch
    except ImportError:
        return None
    return cast(TorchModule, torch)


def reset_peak_memory(device: str) -> None:
    """Reset GPU peak-memory counters before the measured loop when available."""
    torch = import_torch()
    if torch is not None and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device: str) -> tuple[int, str]:
    """Return peak memory usage and the measurement source."""
    torch = import_torch()
    if torch is not None and device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device), "torch.cuda.max_memory_allocated"

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(usage), "resource.getrusage(RUSAGE_SELF).ru_maxrss"
    return int(usage) * 1024, "resource.getrusage(RUSAGE_SELF).ru_maxrss"


def percentile_95(values: list[float]) -> float:
    """Return the nearest-rank p95 for a non-empty list."""
    sorted_values = sorted(values)
    p95_index = min(len(sorted_values) - 1, int(len(sorted_values) * 0.95))
    return sorted_values[p95_index]


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

    device = resolve_detector_device(detector)
    reset_peak_memory(device)

    latencies: list[float] = []
    case_latencies: dict[int, list[float]] = {index: [] for index in range(len(cases))}
    total_start = time.perf_counter()
    for _ in range(repeats):
        for index, case in enumerate(cases):
            start = time.perf_counter()
            detector.predict(
                context=case["context"],
                question=case.get("question"),
                answer=case["answer"],
                output_format=output_format,
            )
            latency = time.perf_counter() - start
            latencies.append(latency)
            case_latencies[index].append(latency)
    total_seconds = time.perf_counter() - total_start
    memory_bytes, memory_source = peak_memory_bytes(device)

    latencies_ms = [value * 1000 for value in latencies]
    case_results: list[CaseBenchmarkResult] = []
    for index, case in enumerate(cases):
        values_ms = [value * 1000 for value in case_latencies[index]]
        case_total = sum(case_latencies[index])
        context_tokens = int(case.get("context_tokens", estimate_context_tokens(case["context"])))
        case_results.append(
            CaseBenchmarkResult(
                name=case.get("name", f"case-{index + 1}"),
                context_tokens=context_tokens,
                latency_mean_ms=statistics.fmean(values_ms),
                latency_median_ms=statistics.median(values_ms),
                latency_p95_ms=percentile_95(values_ms),
                throughput_cases_per_second=len(values_ms) / case_total if case_total else 0.0,
            )
        )

    return BenchmarkResult(
        method=method,
        model_path=model_path,
        device=device,
        cases=len(cases),
        context_lengths=[case.context_tokens for case in case_results],
        warmup=warmup,
        repeats=repeats,
        output_format=output_format,
        total_seconds=total_seconds,
        latency_mean_ms=statistics.fmean(latencies_ms),
        latency_median_ms=statistics.median(latencies_ms),
        latency_p95_ms=percentile_95(latencies_ms),
        throughput_cases_per_second=len(latencies) / total_seconds if total_seconds else 0.0,
        peak_memory_bytes=memory_bytes,
        peak_memory_source=memory_source,
        case_results=case_results,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", default="transformer", choices=["transformer", "llm", "rag_fact_checker"]
    )
    parser.add_argument("--model-path", help="Model path passed to HallucinationDetector")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for transformer benchmarks; auto uses the detector default",
    )
    parser.add_argument(
        "--cases", type=Path, help="JSONL file with context, question, and answer fields"
    )
    parser.add_argument(
        "--context-lengths",
        type=parse_context_lengths,
        default=list(DEFAULT_CONTEXT_LENGTHS),
        help=(
            "Comma-separated synthetic whitespace-token lengths used when --cases is omitted; "
            "lengths above a model's subword-token window may measure truncation"
        ),
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="Warmup iterations before measurement"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Measured iterations over all cases; higher values make p95 more stable",
    )
    parser.add_argument("--output-format", default="spans", choices=["tokens", "spans"])
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    return parser


def print_text_result(result: BenchmarkResult) -> None:
    """Print the aggregate benchmark summary and per-case latency table."""
    print(
        f"Device: {result.device}, "
        f"contexts: {result.context_lengths}, "
        f"latency mean: {result.latency_mean_ms:.2f} ms, "
        f"p95: {result.latency_p95_ms:.2f} ms, "
        f"throughput: {result.throughput_cases_per_second:.2f} cases/s, "
        f"peak memory: {result.peak_memory_bytes} bytes"
    )
    print("Case | context tokens | mean ms | median ms | p95 ms | throughput cases/s")
    for case in result.case_results:
        print(
            f"{case.name} | {case.context_tokens} | {case.latency_mean_ms:.2f} | "
            f"{case.latency_median_ms:.2f} | {case.latency_p95_ms:.2f} | "
            f"{case.throughput_cases_per_second:.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the detector benchmark."""
    parser = build_parser()
    args = parser.parse_args(argv)

    detector_kwargs: dict[str, str] = {}
    if args.model_path:
        detector_kwargs["model_path"] = args.model_path
    if args.device != "auto":
        if args.method != "transformer":
            parser.error("--device is only supported with --method transformer")
        detector_kwargs["device"] = args.device

    from lettucedetect.models.inference import HallucinationDetector

    detector = HallucinationDetector(method=args.method, **detector_kwargs)
    result = run_benchmark(
        detector,
        load_cases(args.cases, args.context_lengths),
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
        print_text_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
