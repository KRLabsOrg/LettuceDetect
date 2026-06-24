"""Command-line entry point for LettuceDetect.

Exposes the ``lettucedetect`` console script that wraps
:class:`~lettucedetect.models.inference.HallucinationDetector` so users can
run hallucination detection from a terminal without writing Python.

Example usage::

    lettucedetect --model KRLabsOrg/lettucedect-base-modernbert-en-v1 \
        --context context.txt \
        --question "Who founded Wikipedia?" \
        --answer answer.txt \
        --format spans

    # Read context from stdin, answer from a file
    echo "Paris is the capital of France." | \
        lettucedetect --model KRLabsOrg/lettucedect-base-modernbert-en-v1 \
        --context - --answer answer.txt --format spans
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read_source(value: str, stdin_text: str | None = None) -> str:
    """Read text from a file path, stdin sentinel, or return the value as a literal string.

    :param value: A file path, the sentinel ``"-"`` for stdin, or a literal
        text string.  If the value is ``"-"`` *stdin_text* is returned.
        If the value is an existing file path the file contents are returned.
        Otherwise the raw string is used as-is.
    :param stdin_text: Pre-read stdin content to use when *value* is ``"-"``.
        If ``None`` and *value* is ``"-"``, stdin is read at call time.
    :return: The resolved text content.
    :rtype: str
    """
    if value == "-":
        if stdin_text is not None:
            return stdin_text
        return sys.stdin.read()
    path = Path(value)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return value


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser for the ``lettucedetect`` CLI.

    :return: Configured :class:`argparse.ArgumentParser` instance.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="lettucedetect",
        description=(
            "Detect hallucinations in a RAG answer using LettuceDetect. "
            "Context and answer accept file paths, literal strings, or '-' for stdin."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Use files
  lettucedetect --model KRLabsOrg/lettucedect-base-modernbert-en-v1 \\
      --context context.txt --answer answer.txt --format spans

  # Pass text inline
  lettucedetect --model ./my-local-model \\
      --context "The Eiffel Tower is in Paris." \\
      --question "Where is the Eiffel Tower?" \\
      --answer "The Eiffel Tower is in Berlin." \\
      --format spans

  # Read context from stdin
  cat context.txt | lettucedetect --model KRLabsOrg/lettucedect-base-modernbert-en-v1 \\
      --context - --answer answer.txt --format spans
""",
    )

    parser.add_argument(
        "--model",
        required=True,
        metavar="MODEL",
        help=(
            "HuggingFace model ID or local path to the transformer detector "
            "(e.g. KRLabsOrg/lettucedect-base-modernbert-en-v1)."
        ),
    )
    parser.add_argument(
        "--context",
        required=True,
        metavar="CONTEXT",
        help=(
            "Path to a plain-text file, '-' to read from stdin, or a literal "
            "context string. Multiple passages are separated by a blank line."
        ),
    )
    parser.add_argument(
        "--answer",
        required=True,
        metavar="ANSWER",
        help=(
            "Path to a plain-text file, '-' to read from stdin, or a literal "
            "answer string."
        ),
    )
    parser.add_argument(
        "--question",
        metavar="QUESTION",
        help="Optional question string. Improves detection accuracy when provided.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["spans", "tokens"],
        default="spans",
        help="Output format: 'spans' (default) for grouped span objects, 'tokens' for per-token predictions.",
    )
    parser.add_argument(
        "--method",
        choices=["transformer", "llm"],
        default="transformer",
        help="Detection method: 'transformer' (default) or 'llm' (requires OPENAI_API_KEY).",
    )
    return parser


def main() -> None:
    """Entry point for the ``lettucedetect`` console script.

    Parses CLI arguments, runs hallucination detection via
    :class:`~lettucedetect.models.inference.HallucinationDetector`, and
    prints the result as JSON to stdout.  Exits with code 1 on any error.

    :raises SystemExit: On argument errors or runtime failures.
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.context == "-" and args.answer == "-":
        parser.error("--context and --answer cannot both be '-' (stdin) at the same time.")

    # Read stdin once if needed, before any other I/O
    stdin_text: str | None = None
    if args.context == "-" or args.answer == "-":
        stdin_text = sys.stdin.read()

    # Resolve context, split on blank lines to support multiple passages
    raw_context = _read_source(args.context, stdin_text)
    context_passages = [p.strip() for p in raw_context.split("\n\n") if p.strip()]
    if not context_passages:
        parser.error("--context resolved to empty text.")

    answer = _read_source(args.answer, stdin_text).strip()
    if not answer:
        parser.error("--answer resolved to empty text.")

    # Lazy import so `--help` works even without heavy ML deps installed
    try:
        from lettucedetect.models.inference import HallucinationDetector
    except ImportError as exc:
        sys.exit(f"Failed to import LettuceDetect — is it installed? {exc}")

    try:
        detector = HallucinationDetector(method=args.method, model_path=args.model)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Failed to load model '{args.model}': {exc}")

    try:
        predictions = detector.predict(
            context=context_passages,
            answer=answer,
            question=args.question,
            output_format=args.output_format,
        )
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Detection failed: {exc}")

    print(json.dumps(predictions, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()