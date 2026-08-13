"""Claude Code Stop-hook that checks the agent's final answer for hallucinations.

Reads the hook event JSON from stdin (a ``Stop`` event carries ``transcript_path``),
extracts the last assistant message and the last user message from the transcript,
and checks the answer against user-supplied grounding context.

Two detector modes:

  --api-url http://127.0.0.1:8000
      Uses the running LettuceDetect web API (``python scripts/start_api.py dev``).
      Recommended: the model stays loaded in the server, so the hook costs one
      HTTP round trip.

  --model-path KRLabsOrg/lettucedect-base-modernbert-en-v1
      In-process detector. Simplest setup, but the model is loaded on every hook
      invocation. Use ``KRLabsOrg/lettucedect-v2-mmbert-base`` (optionally with
      ``--taxonomy-head KRLabsOrg/lettucedect-v2-taxonomy-head``) for code and
      tool-output answers; the taxonomy head types each span, so additions the
      request never asked for are reported as ``unsupported_addition``.

  --llm-model KRLabsOrg/lettucedect-v2-qwen-2b --llm-base-url http://localhost:8001/v1
      Generative detector through an OpenAI-compatible endpoint (e.g. vLLM
      serving the qwen model). Emits typed spans in one pass. Without
      ``--llm-base-url`` the configured provider default is used, so plain LLM
      judges (``--llm-model gpt-4.1-mini``) work too.

Exit contract (Claude Code hooks): exit 2 with the report on stderr feeds the
report back to the agent; exit 0 means nothing to report. When the event says
``stop_hook_active`` the hook exits 0 immediately to avoid a feedback loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

UNSUPPORTED_ADDITION_NOTE = "the request did not ask for this"


def parse_transcript(transcript_path: str | Path) -> tuple[str | None, str | None]:
    """Extract (last user message, last assistant message) from a transcript JSONL.

    Transcript lines are JSON objects; conversation entries carry a ``message``
    with ``role`` and ``content`` (a string or a list of content blocks).
    Unparseable lines are skipped.
    """
    question = None
    answer = None
    path = Path(transcript_path)
    if not path.is_file():
        return None, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        text = _message_text(message)
        if not text:
            continue
        if message.get("role") == "user":
            question = text
        elif message.get("role") == "assistant":
            answer = text
    return question, answer


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p).strip()
    return ""


def load_context(context_files: list[str]) -> list[str]:
    """Read grounding passages; missing files are skipped."""
    contexts = []
    for name in context_files:
        path = Path(name)
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                contexts.append(text)
    return contexts


def format_report(spans: list[dict]) -> str:
    """Human-readable report for flagged spans; handles typed and untyped spans."""
    lines = [f"LettuceDetect flagged {len(spans)} unsupported span(s) in the answer:"]
    for span in spans:
        confidence = span.get("confidence", span.get("hallucination_score"))
        conf = f"confidence {confidence:.2f}" if confidence is not None else "confidence n/a"
        category = span.get("category")
        if category:
            label = category
            if span.get("subcategory"):
                label += f"/{span['subcategory']}"
            if category == "unsupported_addition":
                label += f" — {UNSUPPORTED_ADDITION_NOTE}"
            lines.append(f'- "{span.get("text", "").strip()}" ({conf}, {label})')
        else:
            lines.append(f'- "{span.get("text", "").strip()}" ({conf})')
    lines.append(
        "Revise the flagged parts so every claim is supported by the provided context, "
        "or state explicitly that they are not grounded in it."
    )
    return "\n".join(lines)


def detect_spans_api(api_url: str, contexts: list[str], question: str, answer: str) -> list[dict]:
    """Detect spans via a running LettuceDetect web API."""
    from lettucedetect_api.client import LettuceClient

    client = LettuceClient(api_url)
    response = client.detect_spans(contexts, question, answer)
    return [item.model_dump() for item in response.predictions]


def detect_spans_local(
    model_path: str, taxonomy_head: str | None, contexts: list[str], question: str, answer: str
) -> list[dict]:
    """Detect spans with an in-process detector (loads the model)."""
    from lettucedetect.models.inference import HallucinationDetector

    kwargs = {"method": "transformer", "model_path": model_path}
    if taxonomy_head:
        kwargs["taxonomy_head"] = taxonomy_head
    detector = HallucinationDetector(**kwargs)
    return detector.predict(
        context=contexts, question=question, answer=answer, output_format="spans"
    )


def detect_spans_llm(
    llm_model: str, base_url: str | None, contexts: list[str], question: str, answer: str
) -> list[dict]:
    """Detect spans with the LLM detector (generative lettucedect-v2 models or LLM judges)."""
    from lettucedetect.models.inference import HallucinationDetector

    kwargs = {"method": "llm", "model": llm_model}
    if base_url:
        kwargs["base_url"] = base_url
    detector = HallucinationDetector(**kwargs)
    return detector.predict(
        context=contexts, question=question, answer=answer, output_format="spans"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the hook: parse stdin event, check the answer, report flagged spans."""
    parser = argparse.ArgumentParser(
        description="Claude Code Stop-hook: check the final answer with LettuceDetect."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--api-url", help="Base URL of a running LettuceDetect web API")
    mode.add_argument("--model-path", help="HF model id or local path for in-process detection")
    mode.add_argument(
        "--llm-model",
        help="Generative detector (e.g. KRLabsOrg/lettucedect-v2-qwen-2b via vLLM) or LLM judge",
    )
    parser.add_argument(
        "--llm-base-url",
        help="OpenAI-compatible endpoint for --llm-model (e.g. a vLLM server)",
    )
    parser.add_argument(
        "--taxonomy-head",
        help="Optional span-typing head for --model-path (typed spans in the report)",
    )
    parser.add_argument(
        "--context-file",
        action="append",
        default=None,
        help="Grounding passage file; repeatable (default: context.md in the working directory)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Report only spans at or above this confidence (default 0.0)",
    )
    args = parser.parse_args(argv)

    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if event.get("stop_hook_active"):
        return 0

    transcript_path = event.get("transcript_path")
    if not transcript_path:
        return 0
    question, answer = parse_transcript(transcript_path)
    if not answer:
        return 0

    contexts = load_context(args.context_file or ["context.md"])
    if not contexts:
        return 0

    if args.api_url:
        spans = detect_spans_api(args.api_url, contexts, question or "", answer)
    elif args.llm_model:
        spans = detect_spans_llm(
            args.llm_model, args.llm_base_url, contexts, question or "", answer
        )
    else:
        spans = detect_spans_local(
            args.model_path, args.taxonomy_head, contexts, question or "", answer
        )

    flagged = [
        s
        for s in spans
        if (s.get("confidence", s.get("hallucination_score")) or 0.0) >= args.min_confidence
    ]
    if not flagged:
        return 0
    print(format_report(flagged), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
