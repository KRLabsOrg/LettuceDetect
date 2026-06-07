"""Phase 9: Quality checks and validation report."""

import ast
import difflib
import json
from collections import Counter

# Reuse the canonical span/coverage helpers from the injector so the generation
# and validation passes never drift apart.
from lettucedetect.generation.injection import (
    _extract_code_regions,
    _max_allowed_coverage,
    _span_is_in_code,
)

from .config import DATASET_PATH, FORMATS_PATH, METADATA_PATH, VALIDATION_REPORT_PATH

LEAKY_TERMS = (
    "bug",
    "wrong",
    "incorrect",
    "incorrectly",
    "deprecated",
    "hallucination",
    "helper method",
    "should be replaced",
)


def validate_spans(samples: list[dict]) -> list[str]:
    """Check span boundary validity."""
    issues = []
    for i, sample in enumerate(samples):
        answer_len = len(sample["answer"])
        previous_end = -1
        seen = set()
        for label in sample.get("labels", []):
            start = label.get("start", 0)
            end = label.get("end", 0)
            if start < 0 or end < 0:
                issues.append(f"Sample {i}: negative span offset ({start}, {end})")
            if end <= start:
                issues.append(f"Sample {i}: empty/inverted span ({start}, {end})")
            if end > answer_len:
                issues.append(f"Sample {i}: span exceeds answer length ({end} > {answer_len})")
            if start < previous_end:
                issues.append(f"Sample {i}: unsorted/overlapping spans ({start} < {previous_end})")
            if (start, end, label.get("label")) in seen:
                issues.append(f"Sample {i}: duplicate span ({start}, {end})")
            seen.add((start, end, label.get("label")))
            previous_end = end
    return issues


def _is_whitespace_only_diff(original_text: str, hallucinated_text: str) -> bool:
    """Treat pure whitespace edits as ignorable when checking diff coverage."""
    return (original_text or "").strip() == "" and (hallucinated_text or "").strip() == ""


def _diff_outside_labels(
    original_answer: str, hallucinated_answer: str, labels: list[dict]
) -> list[dict]:
    """Return meaningful diffs not covered by any labeled hallucinated span."""
    label_ranges = [(lab["start"], lab["end"]) for lab in labels]

    def is_covered(start: int, end: int) -> bool:
        return any(
            not (end <= lab_start or start >= lab_end) for lab_start, lab_end in label_ranges
        )

    uncovered = []
    matcher = difflib.SequenceMatcher(a=original_answer, b=hallucinated_answer)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        original_chunk = original_answer[i1:i2]
        hallucinated_chunk = hallucinated_answer[j1:j2]
        if _is_whitespace_only_diff(original_chunk, hallucinated_chunk):
            continue

        if j1 == j2:
            continue

        if not is_covered(j1, j2):
            uncovered.append(
                {
                    "tag": tag,
                    "start": j1,
                    "end": j2,
                    "original": original_chunk[:80],
                    "hallucinated": hallucinated_chunk[:80],
                }
            )

    return uncovered


def check_label_quality(samples: list[dict], metadata: list[dict]) -> dict:
    """Report common synthetic-label issues that should be filtered before training."""
    issues = Counter()
    original_answers = {}
    if FORMATS_PATH.exists():
        with open(FORMATS_PATH) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                original_answers[entry.get("instance_id")] = entry.get("answer", "")

    for sample, meta in zip(samples, metadata):
        if not sample.get("labels"):
            continue

        answer = sample["answer"]
        coverage = sum(label["end"] - label["start"] for label in sample["labels"]) / max(
            len(answer), 1
        )
        if coverage > _max_allowed_coverage(len(answer), 0.30):
            issues["coverage_over_30pct"] += 1

        for label in sample["labels"]:
            span_text = answer[label["start"] : label["end"]]
            if any(term in span_text.lower() for term in LEAKY_TERMS):
                issues["labels_with_leakage_terms"] += 1
                break

        if meta.get("format_type") == "code_with_explanation":
            if any(
                not _span_is_in_code(answer, label["start"], label["end"])
                for label in sample["labels"]
            ):
                issues["code_with_explanation_label_outside_code"] += 1

        original_answer = original_answers.get(meta.get("instance_id"))
        if original_answer:
            uncovered_diffs = _diff_outside_labels(original_answer, answer, sample["labels"])
            if uncovered_diffs:
                issues["diff_outside_labels"] += 1
                if any(
                    diff["tag"] == "insert" or len(diff["hallucinated"]) >= 20
                    for diff in uncovered_diffs
                ):
                    issues["large_diff_outside_labels"] += 1

    return dict(issues)


def check_span_coverage(samples: list[dict]) -> dict:
    """Report span coverage distribution for hallucinated samples."""
    coverages = []
    for sample in samples:
        if not sample.get("labels"):
            continue
        answer_len = len(sample["answer"])
        if answer_len == 0:
            continue
        total_span = sum(label["end"] - label["start"] for label in sample["labels"])
        coverage = total_span / answer_len
        coverages.append(coverage)

    if not coverages:
        return {"count": 0}

    return {
        "count": len(coverages),
        "min": min(coverages),
        "max": max(coverages),
        "mean": sum(coverages) / len(coverages),
        "low_coverage": sum(1 for c in coverages if c < 0.02),
        "high_coverage": sum(1 for c in coverages if c > 0.80),
    }


def check_distributions(metadata: list[dict]) -> dict:
    """Report format/type/LLM/repo distributions."""
    format_counts = Counter(m.get("format_type") for m in metadata if m.get("format_type"))
    type_counts = Counter(
        m.get("hallucination_type") for m in metadata if m.get("hallucination_type")
    )
    model_counts = Counter(m.get("injector_model") for m in metadata if m.get("injector_model"))
    repo_counts = Counter(m.get("repo") for m in metadata)
    split_counts = Counter(m.get("split") for m in metadata)

    return {
        "format": dict(format_counts),
        "hallucination_type": dict(type_counts),
        "injector_model": dict(model_counts),
        "repos": len(repo_counts),
        "top_repos": dict(repo_counts.most_common(10)),
        "split": dict(split_counts),
    }


def check_near_duplicates(samples: list[dict], threshold: float = 0.95) -> int:
    """Simple near-duplicate check via answer Jaccard similarity (sampled)."""
    import random

    rng = random.Random(42)

    n = min(500, len(samples))
    sample_indices = rng.sample(range(len(samples)), n)

    duplicates = 0
    for i in range(len(sample_indices)):
        for j in range(i + 1, min(i + 5, len(sample_indices))):
            a = set(samples[sample_indices[i]]["answer"].split())
            b = set(samples[sample_indices[j]]["answer"].split())
            if not a or not b:
                continue
            jaccard = len(a & b) / len(a | b)
            if jaccard > threshold:
                duplicates += 1

    return duplicates


def check_ast_parseability(samples: list[dict], metadata: list[dict]) -> dict:
    """Check AST parseability for complete_function format samples."""
    total = 0
    parseable = 0

    for sample, meta in zip(samples, metadata):
        if meta.get("format_type") != "complete_function":
            continue
        total += 1
        try:
            ast.parse(sample["answer"])
            parseable += 1
        except SyntaxError:
            pass

    return {
        "total": total,
        "parseable": parseable,
        "rate": parseable / max(total, 1),
    }


def run(samples: list[dict] = None, metadata: list[dict] = None):
    """Run Phase 9: Validation."""
    print("=" * 60)
    print("Phase 9: Validation")
    print("=" * 60)

    if samples is None:
        with open(DATASET_PATH) as f:
            samples = json.load(f)
    if metadata is None:
        with open(METADATA_PATH) as f:
            metadata = json.load(f)

    report_lines = []

    def report(text):
        print(text)
        report_lines.append(text)

    report(f"Total samples: {len(samples)}")
    n_clean = sum(1 for s in samples if not s["labels"])
    n_hall = sum(1 for s in samples if s["labels"])
    report(f"Clean: {n_clean}, Hallucinated: {n_hall}")
    report("")

    # 1. Span validity
    report("=== Span Validity ===")
    span_issues = validate_spans(samples)
    report(f"Issues found: {len(span_issues)}")
    for issue in span_issues[:10]:
        report(f"  {issue}")
    report("")

    # 2. Span coverage
    report("=== Span Coverage ===")
    coverage = check_span_coverage(samples)
    for k, v in coverage.items():
        report(f"  {k}: {v}")
    report("")

    # 3. Distributions
    report("=== Distributions ===")
    dists = check_distributions(metadata)
    for k, v in dists.items():
        report(f"  {k}: {v}")
    report("")

    # 4. Near duplicates
    report("=== Near Duplicates ===")
    n_dup = check_near_duplicates(samples)
    report(f"Near duplicates (sampled): {n_dup}")
    report("")

    # 5. Label quality
    report("=== Label Quality ===")
    label_quality = check_label_quality(samples, metadata)
    for k, v in label_quality.items():
        report(f"  {k}: {v}")
    report("")

    # 6. AST parseability
    report("=== AST Parseability ===")
    ast_check = check_ast_parseability(samples, metadata)
    for k, v in ast_check.items():
        report(f"  {k}: {v}")
    report("")

    # 7. Length stats
    report("=== Length Statistics ===")
    prompt_lens = [len(s["prompt"]) for s in samples]
    answer_lens = [len(s["answer"]) for s in samples]
    if prompt_lens:
        report(
            f"  Prompt chars - min: {min(prompt_lens):,}, max: {max(prompt_lens):,}, avg: {sum(prompt_lens) // len(prompt_lens):,}"
        )
        report(
            f"  Answer chars - min: {min(answer_lens):,}, max: {max(answer_lens):,}, avg: {sum(answer_lens) // len(answer_lens):,}"
        )

    # Save report
    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to {VALIDATION_REPORT_PATH}")
    return report_lines


if __name__ == "__main__":
    run()
