"""Streamlit viewer for code-agent hallucination samples.

Browse generated samples — the developer request, the grounded context, and the
answer with hallucinated spans highlighted by category — to spot-check quality.

Run::

    streamlit run demo/code_hallucination_viewer.py

Point it at a generated directory (``data/v2/code_agent``) or a single JSONL file
via the sidebar.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

CATEGORY_STYLE = {
    "contradiction": ("#ffd6d6", "#cc0000"),
    "unsupported_addition": ("#ffe9c7", "#cc7700"),
    "fabricated_reference": ("#e7d6ff", "#7a00cc"),
}
DEFAULT_DIR = "data/v2/code_agent"


@st.cache_data(show_spinner=False)
def load_samples(path: str) -> list[dict]:
    """Load samples from a JSONL file or a directory of ``*.jsonl`` splits."""
    p = Path(path)
    files = (
        [f for f in sorted(p.glob("*.jsonl")) if not f.name.endswith(".failures.jsonl")]
        if p.is_dir()
        else [p]
    )
    samples: list[dict] = []
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            meta = s.get("metadata")
            s["metadata"] = json.loads(meta) if isinstance(meta, str) else (meta or {})
            s["_file"] = f.name
            samples.append(s)
    return samples


def _highlight_line(text: str, line_start: int, labels: list[dict]) -> str:
    """Escape one line and wrap the parts overlapped by a label span."""
    events = []
    for label in labels:
        a = max(label["start"], line_start) - line_start
        b = min(label["end"], line_start + len(text)) - line_start
        if a < b:
            events.append((a, b, label))
    if not events:
        return html.escape(text)
    events.sort()
    out: list[str] = []
    pos = 0
    for a, b, label in events:
        if a < pos:
            continue
        out.append(html.escape(text[pos:a]))
        bg, border = CATEGORY_STYLE.get(label.get("category", ""), ("#eeeeee", "#888888"))
        tip = html.escape(label.get("explanation", "") or label.get("category", ""))
        out.append(
            f'<span title="{tip}" style="background:{bg};color:#111;'
            f'border-bottom:2px solid {border};border-radius:3px;">{html.escape(text[a:b])}</span>'
        )
        pos = b
    out.append(html.escape(text[pos:]))
    return "".join(out)


_HEADER = ("in file ", "replace:", "with:", "add:")
_CODE = (
    "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:0.82rem;"
    "white-space:pre-wrap;word-break:break-word;padding:2px 10px;margin:0;"
)


def render_answer(answer: str, labels: list[dict]) -> str:
    """Render the answer: edit-style answers as a before→after diff, else a code block."""
    is_edit = "```" in answer and any(m in answer.lower() for m in ("replace:", ", add:"))
    rows: list[str] = []
    pos = 0
    in_code = False
    side = "to"  # current fenced block is removed ('from') or added ('to')
    for line in answer.split("\n"):
        start, stripped = pos, line.strip()
        pos += len(line) + 1
        if is_edit and stripped.startswith("```"):
            in_code = not in_code
            continue
        if is_edit and not in_code and (
            stripped.lower().startswith("in file ") or stripped.lower() in ("replace:", "with:")
        ):
            side = "from" if "replace" in stripped.lower() else "to"
            rows.append(f"<div style='{_CODE}color:#8b949e;font-weight:600;'>{html.escape(line)}</div>")
            continue
        content = _highlight_line(line, start, labels)
        if is_edit and in_code:
            bg, mark = ("#3a1d1d", "-") if side == "from" else ("#16301c", "+")
            rows.append(
                f"<div style='{_CODE}background:{bg};color:#e6edf3;'>"
                f"<span style='color:#6e7681'>{mark} </span>{content}</div>"
            )
        else:
            rows.append(f"<div style='{_CODE}color:#e6edf3;'>{content}</div>")
    return (
        "<div style='background:#0d1117;border-radius:8px;padding:10px 4px;overflow-x:auto;'>"
        + "".join(rows)
        + "</div>"
    )


def legend() -> str:
    """Return an HTML legend mapping each category to its highlight colour."""
    chips = []
    for cat, (bg, border) in CATEGORY_STYLE.items():
        chips.append(
            f'<span style="background:{bg};border-bottom:2px solid {border};'
            f'border-radius:3px;padding:1px 6px;margin-right:8px;">{cat}</span>'
        )
    return "<div style='margin:4px 0 12px'>" + "".join(chips) + "</div>"


def main() -> None:
    """Run the Streamlit viewer."""
    st.set_page_config(page_title="Code Hallucination Viewer", layout="wide")
    st.title("Code-Agent Hallucination Viewer")

    with st.sidebar:
        st.header("Source")
        default = DEFAULT_DIR if Path(DEFAULT_DIR).exists() else ""
        choices = sorted({str(f.parent) for f in Path("data/v2").glob("*/*.jsonl")})
        path = st.selectbox("Directory", choices, index=choices.index(default) if default in choices else 0) if choices else ""
        path = st.text_input("…or path", value=path or DEFAULT_DIR)

    samples = load_samples(path) if path else []
    if not samples:
        st.info(f"No samples found at `{path}`. Generate some, or point to a JSONL file/dir.")
        return

    with st.sidebar:
        st.header("Filter")
        only = st.radio("Show", ["all", "hallucinated", "clean"], horizontal=True)
        modes = sorted({s["metadata"].get("hallucination_mode") for s in samples if s["labels"]} - {None})
        mode = st.selectbox("Mode", ["any", *modes])
        cats = sorted({label["category"] for s in samples for label in s["labels"]})
        cat = st.selectbox("Category", ["any", *cats])
        repos = sorted({s["metadata"].get("instance_id", "").split("__")[0] for s in samples})
        repo = st.selectbox("Repo", ["any", *repos])
        query = st.text_input("Search request/answer")

    def keep(s: dict) -> bool:
        if only == "hallucinated" and not s["labels"]:
            return False
        if only == "clean" and s["labels"]:
            return False
        if mode != "any" and s["metadata"].get("hallucination_mode") != mode:
            return False
        if cat != "any" and not any(label["category"] == cat for label in s["labels"]):
            return False
        if repo != "any" and not s["metadata"].get("instance_id", "").startswith(repo):
            return False
        if query and query.lower() not in (s.get("question", "") + s.get("answer", "")).lower():
            return False
        return True

    filtered = [s for s in samples if keep(s)]
    st.caption(
        f"{len(filtered)} / {len(samples)} samples · "
        f"{sum(1 for s in filtered if s['labels'])} hallucinated"
    )
    if not filtered:
        st.warning("No samples match the filters.")
        return

    idx = st.number_input("Sample", 0, len(filtered) - 1, 0, 1)
    s = filtered[int(idx)]
    meta = s["metadata"]

    chips = " · ".join(
        x for x in [
            f"**{meta.get('instance_id', '?')}**",
            f"`{s['_file']}`",
            f"mode: `{meta.get('hallucination_mode', '—')}`" if s["labels"] else "**clean**",
            f"style: `{meta.get('answer_style', '?')}`",
        ] if x
    )
    st.markdown(chips)

    st.subheader("Developer request")
    st.markdown(f"> {s.get('question', '')}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Answer")
        st.markdown(legend(), unsafe_allow_html=True)
        st.markdown(render_answer(s.get("answer", ""), s["labels"]), unsafe_allow_html=True)
        if s["labels"]:
            st.subheader("Labels")
            for label in sorted(s["labels"], key=lambda label: label["start"]):
                seg = s["answer"][label["start"]:label["end"]]
                st.markdown(
                    f"- **{label['category']}** / `{label.get('subcategory')}` — "
                    f"`{seg[:80]}`  \n  {label.get('explanation', '')}"
                )
    with right:
        st.subheader("Context")
        st.code(s.get("context", ""), language="python")


if __name__ == "__main__":
    main()
