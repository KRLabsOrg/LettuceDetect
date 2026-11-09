"""Report hallucination rates for a JSONL or CSV dataset.

Each input row must contain ``context`` (a string or a list of strings) and
``answer``. ``question`` and an identifier such as ``id`` are optional. CSV
contexts may be plain text or JSON-encoded string arrays. Use ``--group-by``
to name any additional column whose values should receive separate rates;
blank or missing values are reported as ``<missing>``.

Example::

    python scripts/report_hallucination_rates.py \
        --input sample.jsonl \
        --method transformer \
        --model-path KRLabsOrg/lettucedect-v2-mmbert-base \
        --taxonomy-head KRLabsOrg/lettucedect-v2-taxonomy-head \
        --min-confidence 0.7 \
        --group-by source \
        --out-json report.json \
        --out-md report.md

The reported hallucination rate is the fraction of scored answers with at
least one flagged span. It is not a gold-label detector-quality metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

MISSING_GROUP = "<missing>"
CONFIDENCE_BIN_COUNT = 10
logger = logging.getLogger(__name__)


class InputRecord(TypedDict):
    """Normalized input row passed to a detector."""

    row_number: int
    context: list[str]
    answer: str
    question: str | None
    identifier: str | None
    group: str | None


class PredictDetector(Protocol):
    """Detector surface needed by the reporting runner."""

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return hallucination predictions for one record."""
        ...


class ScoredRecord(TypedDict):
    """One normalized input row paired with its predicted spans."""

    record: InputRecord
    spans: list[dict[str, Any]]


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def confidence_threshold(value: str) -> float:
    """Parse a finite confidence threshold in the closed interval [0, 1]."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number between 0 and 1")
    return parsed


def _normalize_context(value: object, location: str, *, from_csv: bool) -> list[str]:
    """Normalize a context string or string array with a location-aware error."""
    if from_csv and isinstance(value, str) and value.lstrip().startswith("["):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass

    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{location}: context must not be blank")
        return [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}: context must be a non-empty string or list of strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{location}: every context item must be a non-blank string")
    return list(value)


def _normalize_optional_text(value: object, location: str, field: str) -> str | None:
    """Normalize an optional scalar text field."""
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"{location}: {field} must be a string when present")
    return value


def _normalize_row(
    row: Mapping[str, Any],
    *,
    path: Path,
    row_number: int,
    group_by: str | None,
    from_csv: bool,
) -> InputRecord:
    """Validate and normalize one JSONL or CSV input row."""
    location = f"{path}:{row_number}"
    missing = [field for field in ("context", "answer") if field not in row]
    if missing:
        raise ValueError(f"{location}: missing required field(s): {', '.join(missing)}")

    answer = row["answer"]
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{location}: answer must be a non-blank string")

    raw_group = row.get(group_by) if group_by is not None else None
    if raw_group is None or raw_group == "":
        group = MISSING_GROUP if group_by is not None else None
    elif isinstance(raw_group, (str, int, float, bool)):
        group = str(raw_group)
    else:
        raise ValueError(f"{location}: grouping field {group_by!r} must be a scalar value")

    raw_identifier = row.get("id")
    if raw_identifier is None or raw_identifier == "":
        identifier = None
    elif isinstance(raw_identifier, (str, int, float, bool)):
        identifier = str(raw_identifier)
    else:
        raise ValueError(f"{location}: id must be a scalar value when present")

    return {
        "row_number": row_number,
        "context": _normalize_context(row["context"], location, from_csv=from_csv),
        "answer": answer,
        "question": _normalize_optional_text(row.get("question"), location, "question"),
        "identifier": identifier,
        "group": group,
    }


def load_records(
    path: Path, *, group_by: str | None = None, limit: int | None = None
) -> list[InputRecord]:
    """Load and normalize JSONL or CSV records, optionally stopping after ``limit`` rows."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    suffix = path.suffix.lower()
    if suffix not in {".jsonl", ".csv"}:
        raise ValueError(f"{path}: input must have a .jsonl or .csv extension")

    records: list[InputRecord] = []
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_number}: blank JSONL rows are not allowed")
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(raw, dict):
                    raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
                records.append(
                    _normalize_row(
                        raw,
                        path=path,
                        row_number=line_number,
                        group_by=group_by,
                        from_csv=False,
                    )
                )
                if limit is not None and len(records) >= limit:
                    break
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: CSV must have a header row")
            missing_columns = {"context", "answer"} - set(reader.fieldnames)
            if missing_columns:
                fields = ", ".join(sorted(missing_columns))
                raise ValueError(f"{path}: CSV is missing required column(s): {fields}")
            if group_by is not None and group_by not in reader.fieldnames:
                raise ValueError(f"{path}: CSV is missing grouping column {group_by!r}")
            for row_number, raw in enumerate(reader, start=2):
                records.append(
                    _normalize_row(
                        raw,
                        path=path,
                        row_number=row_number,
                        group_by=group_by,
                        from_csv=True,
                    )
                )
                if limit is not None and len(records) >= limit:
                    break

    if not records:
        raise ValueError(f"{path}: input does not contain any records")
    return records


def score_records(
    detector: PredictDetector,
    records: Sequence[InputRecord],
    *,
    min_confidence: float,
    progress: bool = True,
) -> list[ScoredRecord]:
    """Score records sequentially and return validated span lists."""
    if not math.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be a finite number between 0 and 1")

    scored: list[ScoredRecord] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        predictions = detector.predict(
            context=record["context"],
            answer=record["answer"],
            question=record["question"],
            output_format="spans",
            min_confidence=min_confidence,
        )
        if not isinstance(predictions, list) or any(
            not isinstance(span, Mapping) for span in predictions
        ):
            raise TypeError(f"record {index}: detector must return a list of span objects")
        scored.append({"record": record, "spans": [dict(span) for span in predictions]})
        if progress:
            logger.info("Scored %d/%d records", index, total)
    return scored


def _rate(numerator: int, denominator: int) -> dict[str, int | float]:
    """Build a rate object with its explicit numerator and denominator."""
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else 0.0,
    }


def _valid_confidence(value: object) -> float | None:
    """Return a valid finite confidence in [0, 1], excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def _confidence_bin(value: float) -> str:
    """Return a stable 0.1-wide confidence-bin label."""
    index = min(int(value * CONFIDENCE_BIN_COUNT), CONFIDENCE_BIN_COUNT - 1)
    lower = index / CONFIDENCE_BIN_COUNT
    upper = (index + 1) / CONFIDENCE_BIN_COUNT
    closing = "]" if index == CONFIDENCE_BIN_COUNT - 1 else ")"
    return f"[{lower:.1f}, {upper:.1f}{closing}"


def _preview(value: str, length: int = 240) -> str:
    """Collapse whitespace and truncate text for a compact report example."""
    compact = " ".join(value.split())
    return compact if len(compact) <= length else compact[: length - 1] + "…"


def _report_span(span: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a span while replacing invalid confidence values with JSON-safe null."""
    result = dict(span)
    if "confidence" in result and result["confidence"] is not None:
        if _valid_confidence(result["confidence"]) is None:
            result["confidence"] = None
            result["confidence_valid"] = False
    return result


def aggregate_results(
    pairs: Sequence[ScoredRecord],
    *,
    group_by: str | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Aggregate detector output without loading or invoking a model."""
    if top_n < 0:
        raise ValueError("top_n must be zero or greater")

    total = len(pairs)
    flagged = sum(bool(pair["spans"]) for pair in pairs)
    span_counts = Counter(len(pair["spans"]) for pair in pairs)
    category_counts: Counter[str] = Counter()
    subcategory_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    missing_confidence = 0
    invalid_confidence = 0
    valid_confidences = 0

    bin_labels = [_confidence_bin(index / CONFIDENCE_BIN_COUNT) for index in range(10)]
    confidence_counts.update({label: 0 for label in bin_labels})

    for pair in pairs:
        for span in pair["spans"]:
            category = span.get("category")
            if isinstance(category, str) and category.strip():
                category_counts[category] += 1
            subcategory = span.get("subcategory")
            if isinstance(subcategory, str) and subcategory.strip():
                subcategory_counts[subcategory] += 1

            if "confidence" not in span or span["confidence"] is None:
                missing_confidence += 1
                continue
            confidence = _valid_confidence(span["confidence"])
            if confidence is None:
                invalid_confidence += 1
                continue
            valid_confidences += 1
            confidence_counts[_confidence_bin(confidence)] += 1

    groups: dict[str, Any] | None = None
    if group_by is not None:
        group_totals: Counter[str] = Counter()
        group_flagged: Counter[str] = Counter()
        for pair in pairs:
            group = pair["record"]["group"] or MISSING_GROUP
            group_totals[group] += 1
            if pair["spans"]:
                group_flagged[group] += 1
        rates = {
            group: _rate(group_flagged[group], denominator)
            for group, denominator in sorted(group_totals.items())
        }
        groups = {
            "column": group_by,
            "missing_value_label": MISSING_GROUP,
            "denominator_sum": sum(group_totals.values()),
            "rates": rates,
        }

    ranked: list[tuple[int, float | None, ScoredRecord]] = []
    for index, pair in enumerate(pairs):
        if not pair["spans"]:
            continue
        confidences = [
            confidence
            for span in pair["spans"]
            if (confidence := _valid_confidence(span.get("confidence"))) is not None
        ]
        ranked.append((index, max(confidences) if confidences else None, pair))
    ranked.sort(key=lambda item: (item[1] is None, -(item[1] or 0.0), item[0]))

    examples: list[dict[str, Any]] = []
    for _, max_confidence, pair in ranked[:top_n]:
        record = pair["record"]
        examples.append(
            {
                "row_number": record["row_number"],
                "id": record["identifier"],
                "group": record["group"],
                "question": record["question"],
                "answer_preview": _preview(record["answer"]),
                "context_preview": _preview("\n".join(record["context"])),
                "span_count": len(pair["spans"]),
                "max_confidence": max_confidence,
                "spans": [_report_span(span) for span in pair["spans"]],
            }
        )

    return {
        "schema_version": 1,
        "summary": {
            "rows_in": total,
            "rows_scored": total,
            "flagged_answers": flagged,
            "hallucination_rate": _rate(flagged, total),
        },
        "groups": groups,
        "span_count_histogram": {
            str(count): frequency for count, frequency in sorted(span_counts.items())
        },
        "span_confidence_histogram": {
            "bins": dict(confidence_counts),
            "valid_confidence_spans": valid_confidences,
            "missing_confidence_spans": missing_confidence,
            "invalid_confidence_spans": invalid_confidence,
            "all_spans": sum(span_counts[count] * count for count in span_counts),
        },
        "category_counts": dict(sorted(category_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "top_flagged_examples": examples,
    }


def _markdown_cell(value: object) -> str:
    """Escape an arbitrary scalar for one Markdown table cell."""
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a human-readable Markdown view of a canonical report object."""
    summary = report["summary"]
    rate = summary["hallucination_rate"]
    lines = [
        "# Hallucination rate report",
        "",
        "The rate is the fraction of scored answers with at least one flagged span.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Rows in | {summary['rows_in']} |",
        f"| Rows scored | {summary['rows_scored']} |",
        f"| Flagged answers | {summary['flagged_answers']} |",
        f"| Hallucination rate | {rate['value']:.2%} ({rate['numerator']}/{rate['denominator']}) |",
    ]

    groups = report["groups"]
    if groups is not None:
        lines.extend(
            [
                "",
                f"## Rates by `{groups['column']}`",
                "",
                "| Group | Flagged | Denominator | Rate |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for group, group_rate in groups["rates"].items():
            lines.append(
                f"| {_markdown_cell(group)} | {group_rate['numerator']} | "
                f"{group_rate['denominator']} | {group_rate['value']:.2%} |"
            )
        lines.append(f"\nGroup denominators sum to **{groups['denominator_sum']}**.")

    lines.extend(
        [
            "",
            "## Span count histogram",
            "",
            "| Spans per answer | Answers |",
            "| ---: | ---: |",
        ]
    )
    for count, frequency in report["span_count_histogram"].items():
        lines.append(f"| {count} | {frequency} |")

    confidence = report["span_confidence_histogram"]
    lines.extend(
        [
            "",
            "## Span confidence histogram",
            "",
            "Bins are left-inclusive and right-exclusive, except `[0.9, 1.0]`.",
            "",
            "| Confidence | Spans |",
            "| --- | ---: |",
        ]
    )
    for label, count in confidence["bins"].items():
        lines.append(f"| `{label}` | {count} |")
    lines.extend(
        [
            f"| Missing confidence | {confidence['missing_confidence_spans']} |",
            f"| Invalid confidence | {confidence['invalid_confidence_spans']} |",
            f"| **All spans** | **{confidence['all_spans']}** |",
        ]
    )

    for title, key in (
        ("Category counts", "category_counts"),
        ("Subcategory counts", "subcategory_counts"),
    ):
        lines.extend(["", f"## {title}", ""])
        counts = report[key]
        if not counts:
            lines.append("No typed spans were reported.")
        else:
            lines.extend(["| Label | Spans |", "| --- | ---: |"])
            for label, count in counts.items():
                lines.append(f"| {_markdown_cell(label)} | {count} |")

    lines.extend(["", "## Top flagged examples", ""])
    examples = report["top_flagged_examples"]
    if not examples:
        lines.append("No answers were flagged.")
    else:
        lines.extend(
            [
                "| Row | ID | Group | Max confidence | Spans | Answer preview |",
                "| ---: | --- | --- | ---: | ---: | --- |",
            ]
        )
        for example in examples:
            maximum = (
                "—" if example["max_confidence"] is None else f"{example['max_confidence']:.3f}"
            )
            lines.append(
                f"| {example['row_number']} | {_markdown_cell(example['id'])} | "
                f"{_markdown_cell(example['group'])} | {maximum} | {example['span_count']} | "
                f"{_markdown_cell(example['answer_preview'])} |"
            )
            lines.append("")
            lines.append("```json")
            lines.append(
                json.dumps(example["spans"], ensure_ascii=False, indent=2, allow_nan=False)
            )
            lines.append("```")

    return "\n".join(lines).rstrip() + "\n"


def _atomic_write(path: Path, content: str) -> None:
    """Replace a text output atomically after its full content is ready."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_reports(report: Mapping[str, Any], json_path: Path, markdown_path: Path) -> None:
    """Write JSON and Markdown representations of the same report object."""
    json_content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    markdown_content = render_markdown(report)
    _atomic_write(json_path, json_content)
    _atomic_write(markdown_path, markdown_content)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input .jsonl or .csv file.")
    parser.add_argument(
        "--method", choices=["transformer", "llm"], default="transformer", help="Detector type."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Transformer model path/HF id, or the model name for --method llm.",
    )
    parser.add_argument(
        "--taxonomy-head", help="Optional transformer taxonomy-head path or Hugging Face id."
    )
    parser.add_argument(
        "--min-confidence", type=confidence_threshold, default=0.0, help="Span threshold in [0, 1]."
    )
    parser.add_argument(
        "--lang", default="en", help="Detector prompt/model language (default: en)."
    )
    parser.add_argument("--group-by", help="Optional input column to aggregate rates by.")
    parser.add_argument("--limit", type=positive_int, help="Score only the first N records.")
    parser.add_argument(
        "--top-n",
        type=non_negative_int,
        default=10,
        help="Number of flagged examples (default: 10).",
    )
    parser.add_argument("--out-json", type=Path, required=True, help="JSON report destination.")
    parser.add_argument("--out-md", type=Path, required=True, help="Markdown report destination.")
    return parser


def create_detector(
    method: str, model_path: str, lang: str, taxonomy_head: str | None
) -> PredictDetector:
    """Create a detector lazily with method-appropriate constructor arguments."""
    if method == "llm" and taxonomy_head is not None:
        raise ValueError("--taxonomy-head is only supported with --method transformer")

    from lettucedetect.models.inference import HallucinationDetector

    kwargs: dict[str, Any] = {"lang": lang}
    if method == "transformer":
        kwargs["model_path"] = model_path
        if taxonomy_head is not None:
            kwargs["taxonomy_head"] = taxonomy_head
    else:
        kwargs["model"] = model_path
    return cast(PredictDetector, HallucinationDetector(method=method, **kwargs))


def main(argv: Sequence[str] | None = None, *, detector: PredictDetector | None = None) -> int:
    """Run dataset scoring and write both report formats."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        records = load_records(args.input, group_by=args.group_by, limit=args.limit)
        active_detector = detector or create_detector(
            args.method, args.model_path, args.lang, args.taxonomy_head
        )
        scored = score_records(
            active_detector, records, min_confidence=args.min_confidence, progress=True
        )
        report = aggregate_results(scored, group_by=args.group_by, top_n=args.top_n)
        report["run"] = {
            "input": str(args.input),
            "method": args.method,
            "model_path": args.model_path,
            "taxonomy_head": args.taxonomy_head,
            "min_confidence": args.min_confidence,
            "lang": args.lang,
            "group_by": args.group_by,
            "limit": args.limit,
        }
        write_reports(report, args.out_json, args.out_md)
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
