# LettuceDetect + Claude Code

A [Claude Code hook](https://docs.claude.com/en/docs/claude-code/hooks) that checks
the agent's final answer against grounding context you provide, and feeds flagged
spans back to the agent. With the typed-span models it reports not only unsupported
claims but also `unsupported_addition` — behavior the request never asked for.

## Installation

```bash
pip install lettucedetect
```

## Try it (5 lines)

```bash
# 1. Put your grounding passages in context.md in the project root.
# 2. Start the detection server once (keeps the model loaded):
python scripts/start_api.py dev
# 3. Run the hook against a transcript fixture:
echo '{"transcript_path": "tests/fixtures/claude_code_transcript.jsonl"}' | \
  python -m lettucedetect.integrations.claude_code.check_answer --api-url http://127.0.0.1:8000
```

Exit code 0 means the answer is supported; exit code 2 prints a span report on
stderr, which Claude Code feeds back to the agent.

## Hook configuration

Paste into your project's `.claude/settings.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python -m lettucedetect.integrations.claude_code.check_answer --api-url http://127.0.0.1:8000 --min-confidence 0.5"
          }
        ]
      }
    ]
  }
}
```

Every time the agent finishes a reply, the hook extracts the final answer from the
session transcript, checks it against `context.md`, and if unsupported spans are
found the agent receives them and revises.

## Modes

| Flag | Behavior |
|---|---|
| `--api-url URL` | Uses the running web API (recommended: model stays loaded, one HTTP round trip per check) |
| `--model-path ID` | In-process detection; simplest setup, but loads the model on every invocation |
| `--taxonomy-head ID` | With `--model-path`: types each span (`unsupported_addition`, `contradiction`, ...) |
| `--context-file F` | Grounding passage file, repeatable; default `context.md` |
| `--min-confidence X` | Report only spans at or above this confidence |

Models: `KRLabsOrg/lettucedect-base-modernbert-en-v1` for prose RAG answers;
`KRLabsOrg/lettucedect-v2-mmbert-base` plus
`KRLabsOrg/lettucedect-v2-taxonomy-head` for code and tool-output answers with
typed spans.

## Conventions and limits

- You supply the grounding context (`context.md` or `--context-file`); the hook
  does not reconstruct retrieved context from the session.
- If the context file is missing or the transcript has no assistant message, the
  hook exits 0 silently.
- When the event carries `stop_hook_active`, the hook exits 0 immediately, so the
  agent's revision after a flagged answer is not re-blocked in a loop.
