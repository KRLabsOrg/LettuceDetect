#!/usr/bin/env python3
r"""Collect README files from popular GitHub repositories via the REST API.

Searches popular repos across several languages (for diversity), fetches each
repo's README, filters to substantial structured docs, and writes a JSONL
corpus that the markdown/README data adapter consumes. Re-runnable: existing
repos in the output are skipped, so it resumes and can be expanded later.

Requires a GitHub token (env ``GITHUB_TOKEN`` or ``--token``) for a usable rate
limit (5000 requests/hour authenticated vs. 60 unauthenticated).

Usage::

    GITHUB_TOKEN=ghp_xxx python scripts/collect_github_readmes.py \\
        --out data/readmes/github_readmes.jsonl --max-repos 1500
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests

API = "https://api.github.com"
DEFAULT_LANGUAGES = [
    "python", "javascript", "typescript", "go", "rust", "java",
    "c++", "c", "ruby", "php", "c#", "kotlin", "swift", "scala",
]
MIN_README_CHARS = 600
MAX_README_CHARS = 40000


def _session(token: str | None) -> requests.Session:
    s = requests.Session()
    s.headers["Accept"] = "application/vnd.github+json"
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def _respect_rate_limit(resp: requests.Response) -> None:
    """Sleep until reset if the response indicates the rate limit is exhausted."""
    if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
        reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
        wait = max(reset - int(time.time()), 1) + 1
        print(f"  rate limit hit; sleeping {wait}s")
        time.sleep(wait)


def search_repos(session: requests.Session, query: str, max_pages: int) -> list[dict]:
    """Return repo dicts for a search query (paginated, 100 per page)."""
    repos: list[dict] = []
    for page in range(1, max_pages + 1):
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": 100, "page": page}
        resp = session.get(f"{API}/search/repositories", params=params, timeout=30)
        if resp.status_code != 200:
            _respect_rate_limit(resp)
            break
        items = resp.json().get("items", [])
        if not items:
            break
        repos.extend(items)
        time.sleep(2.5)  # search is limited to ~30/min
    return repos


def fetch_readme(session: requests.Session, full_name: str) -> str | None:
    """Fetch a repo's README as raw text, or None if missing/unsuitable."""
    resp = session.get(
        f"{API}/repos/{full_name}/readme",
        headers={"Accept": "application/vnd.github.raw"},
        timeout=30,
    )
    if resp.status_code == 403:
        _respect_rate_limit(resp)
        return None
    if resp.status_code != 200:
        return None
    return resp.text


def is_good_readme(text: str) -> bool:
    """Keep substantial, structured READMEs (has a heading, reasonable length)."""
    if not (MIN_README_CHARS <= len(text) <= MAX_README_CHARS):
        return False
    return any(line.lstrip().startswith("#") for line in text.splitlines())


def load_done(path: Path) -> set[str]:
    """Return repos already in the output file (for resume)."""
    done: set[str] = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["repo"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main() -> None:
    """Collect READMEs from popular repos across languages into a JSONL corpus."""
    ap = argparse.ArgumentParser(description="Collect GitHub READMEs into a JSONL corpus.")
    ap.add_argument("--out", default="data/readmes/github_readmes.jsonl")
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    ap.add_argument("--max-repos", type=int, default=1500)
    ap.add_argument("--min-stars", type=int, default=2000)
    ap.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    ap.add_argument("--pages-per-language", type=int, default=2, help="100 repos per page.")
    args = ap.parse_args()

    if not args.token:
        print("WARNING: no GITHUB_TOKEN — limited to 60 requests/hour.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    print(f"Already collected: {len(done)} repos")

    session = _session(args.token)
    seen = set(done)
    kept = 0

    with open(out_path, "a") as f:
        for lang in args.languages:
            if kept + len(done) >= args.max_repos:
                break
            query = f"stars:>{args.min_stars} language:{lang}"
            for repo in search_repos(session, query, args.pages_per_language):
                full = repo["full_name"]
                if full in seen:
                    continue
                seen.add(full)
                if kept + len(done) >= args.max_repos:
                    break
                readme = fetch_readme(session, full)
                if not readme or not is_good_readme(readme):
                    continue
                f.write(
                    json.dumps(
                        {
                            "repo": full,
                            "stars": repo.get("stargazers_count"),
                            "language": repo.get("language"),
                            "readme": readme,
                        }
                    )
                    + "\n"
                )
                f.flush()
                kept += 1
                if kept % 50 == 0:
                    print(f"  collected {kept} new ({lang})")

    print(f"Done. Collected {kept} new READMEs -> {out_path}")


if __name__ == "__main__":
    main()
