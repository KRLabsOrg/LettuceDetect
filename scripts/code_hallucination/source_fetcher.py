"""Phase 2: Clone repos and fetch source files via git show."""

import ast
import json
import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

import requests

from .config import MAX_FILE_CHARS, MAX_PROMPT_CHARS, REPOS_DIR, SOURCE_CACHE_DIR


def build_source_context(source_data: dict) -> str:
    """Build the source-code context string from cached source data (truncated)."""
    parts = [
        f"File: {filepath}\n```python\n{content}\n```"
        for filepath, content in source_data.get("source_files", {}).items()
    ]
    context = "\n\n".join(parts)
    return context[:MAX_PROMPT_CHARS] if len(context) > MAX_PROMPT_CHARS else context


GITHUB_RAW_BASE = "https://raw.githubusercontent.com"


def truncate_around_patch(
    full_content: str, patch: str, filepath: str, max_chars: int = MAX_FILE_CHARS
) -> str:
    """Truncate a source file keeping the region around the patch.

    Instead of taking the first N chars (which may miss the patched region),
    find where the patch applies and keep a window around it, plus the file header
    (imports/class definitions).
    """
    if len(full_content) <= max_chars:
        return full_content

    # Find the hunk start lines from the patch for this file
    hunk_lines = []
    in_file = False
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            in_file = match is not None and match.group(2) == filepath
        elif in_file and line.startswith("@@"):
            hunk_match = re.match(r"@@ -(\d+)", line)
            if hunk_match:
                hunk_lines.append(int(hunk_match.group(1)))

    if not hunk_lines:
        # Can't find patch location, fall back to first N chars
        return full_content[:max_chars]

    lines = full_content.split("\n")

    # Always keep the header (imports, class defs) — first 50 lines or until first function
    header_end = min(50, len(lines))
    for i, line in enumerate(lines[:200]):
        if line.strip().startswith("def ") or line.strip().startswith("class "):
            if i > 20:
                header_end = i
                break

    header = "\n".join(lines[:header_end])
    header_chars = len(header)
    remaining_budget = max_chars - header_chars - 100  # 100 for separator

    if remaining_budget <= 0:
        return full_content[:max_chars]

    # Build a window around the patch hunks
    min_hunk = min(hunk_lines) - 1  # Convert to 0-based
    max_hunk = max(hunk_lines) - 1

    # Expand window to use the remaining budget
    lines_budget = remaining_budget // 80  # Rough estimate: 80 chars per line
    padding = max(lines_budget // 2, 30)

    window_start = max(header_end, min_hunk - padding)
    window_end = min(len(lines), max_hunk + padding)

    window = "\n".join(lines[window_start:window_end])

    if window_start > header_end:
        result = header + "\n\n# ... (truncated) ...\n\n" + window
    else:
        result = header + "\n" + window

    return result[:max_chars]


def extract_changed_files(patch: str) -> list[str]:
    """Extract file paths from a unified diff using anchored regex.

    Fixed: Uses re.match on 'diff --git' lines instead of re.search(r"b/(.+)$")
    which was ambiguous on paths containing 'b/'.
    """
    files = []
    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            if match:
                files.append(match.group(2))
    return files


def clone_repo(repo: str, repos_dir: Path = REPOS_DIR) -> Path | None:
    """Clone a repo if not already present. Returns repo directory path."""
    repos_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = repos_dir / repo.replace("/", "__")

    if repo_dir.exists():
        return repo_dir

    print(f"  Cloning {repo}...")
    try:
        result = subprocess.run(
            ["git", "clone", "--bare", f"https://github.com/{repo}.git", str(repo_dir)],
            capture_output=True,
            text=True,
            timeout=60,  # 1 min timeout, fall back to GitHub API
        )
        if result.returncode != 0:
            print(f"  ERROR cloning {repo}: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT cloning {repo} (>30 min)")
        return None

    return repo_dir


class TransientFetchError(Exception):
    """A fetch failure that may succeed on retry (timeout, rate limit, 5xx)."""


def fetch_file_from_github(repo: str, commit: str, filepath: str) -> str | None:
    """Fetch a file from GitHub raw.

    Works unauthenticated for public repos; a ``GITHUB_TOKEN`` env var, when set,
    is sent to raise the request rate limit for large runs.

    Returns None only on a definitive miss (404 etc.); raises
    :class:`TransientFetchError` on timeouts, rate limits, and server errors so
    callers don't treat (or cache) a transient failure as a permanent miss.
    """
    url = f"{GITHUB_RAW_BASE}/{repo}/{commit}/{filepath}"
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = requests.get(url, headers=headers, timeout=15)
    except requests.RequestException as exc:
        raise TransientFetchError(f"{url}: {exc}") from exc
    if r.status_code == 200:
        return r.text
    if r.status_code in (403, 429) or r.status_code >= 500:
        raise TransientFetchError(f"{url}: HTTP {r.status_code}")
    return None


def fetch_file_at_commit(repo_dir: Path, commit: str, filepath: str) -> str | None:
    """Fetch a file's contents at a specific commit using git show."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{filepath}"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"    Error fetching {filepath}@{commit[:8]}: {e}")
        return None


def apply_patch_and_get_file(repo_dir: Path, commit: str, patch: str, filepath: str) -> str | None:
    """Apply a patch to get the post-fix version of a file.

    Uses a temporary worktree to apply the patch cleanly.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temporary worktree at the base commit
            result = subprocess.run(
                ["git", "worktree", "add", tmpdir, commit],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                return None

            # Apply the patch
            result = subprocess.run(
                ["git", "apply", "--allow-empty", "-"],
                input=patch,
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return None

            # Read the patched file
            patched_path = Path(tmpdir) / filepath
            if patched_path.exists():
                return patched_path.read_text()

            # Clean up worktree
            subprocess.run(
                ["git", "worktree", "remove", "--force", tmpdir],
                cwd=str(repo_dir),
                capture_output=True,
                timeout=30,
            )
            return None
    except Exception as e:
        print(f"    Error applying patch for {filepath}: {e}")
        return None


def extract_modified_functions(original_source: str, patched_source: str) -> list[dict]:
    """Extract functions that were modified between original and patched source.

    Returns list of dicts with 'name', 'original', 'patched' function bodies.
    """

    def get_functions(source: str) -> dict[str, str]:
        """Parse source and extract function name -> source mapping."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source)
        except SyntaxError:
            return {}

        funcs = {}
        lines = source.split("\n")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno - 1
                end = node.end_lineno
                func_source = "\n".join(lines[start:end])
                funcs[node.name] = func_source
        return funcs

    orig_funcs = get_functions(original_source)
    patched_funcs = get_functions(patched_source)

    modified = []
    for name, patched_body in patched_funcs.items():
        orig_body = orig_funcs.get(name)
        if orig_body and orig_body != patched_body:
            modified.append(
                {
                    "name": name,
                    "original": orig_body,
                    "patched": patched_body,
                }
            )
        elif not orig_body:
            # New function added by patch
            modified.append(
                {
                    "name": name,
                    "original": None,
                    "patched": patched_body,
                }
            )

    return modified


def apply_patch_in_memory(original_source: str, patch: str, filepath: str) -> str | None:
    """Apply a unified diff patch to source code in memory, without git.

    Parses the unified diff to extract hunks for the target file,
    then applies them line-by-line to produce the patched source.

    :param original_source: The original file content.
    :param patch: The full unified diff (may contain multiple files).
    :param filepath: The specific file to extract and apply hunks for.
    :return: Patched source string, or None if application fails.
    """
    # Split patch into per-file sections
    file_patch_lines = []
    in_target_file = False

    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            in_target_file = match is not None and match.group(2) == filepath
            continue
        if in_target_file:
            file_patch_lines.append(line)

    if not file_patch_lines:
        return None

    # Parse hunks from the file-specific patch lines
    hunks = []
    current_hunk = None

    for line in file_patch_lines:
        if line.startswith("@@"):
            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            hunk_match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if hunk_match:
                current_hunk = {
                    "old_start": int(hunk_match.group(1)),
                    "lines": [],
                }
                hunks.append(current_hunk)
        elif line.startswith("---") or line.startswith("+++"):
            continue
        elif current_hunk is not None:
            current_hunk["lines"].append(line)

    if not hunks:
        return None

    # Apply hunks to original source
    original_lines = original_source.split("\n")
    result_lines = []
    orig_idx = 0  # 0-based index into original_lines

    try:
        for hunk in hunks:
            hunk_start = hunk["old_start"] - 1  # Convert to 0-based

            # Copy unchanged lines before this hunk
            while orig_idx < hunk_start and orig_idx < len(original_lines):
                result_lines.append(original_lines[orig_idx])
                orig_idx += 1

            # Apply hunk lines
            for line in hunk["lines"]:
                if line.startswith("+"):
                    result_lines.append(line[1:])
                elif line.startswith("-"):
                    orig_idx += 1  # Skip the removed line
                elif line.startswith(" "):
                    result_lines.append(line[1:])
                    orig_idx += 1
                elif line == "":
                    # Empty line in diff context — treat as context
                    if orig_idx < len(original_lines):
                        result_lines.append(original_lines[orig_idx])
                        orig_idx += 1

        # Copy remaining lines after last hunk
        while orig_idx < len(original_lines):
            result_lines.append(original_lines[orig_idx])
            orig_idx += 1

        return "\n".join(result_lines)
    except (IndexError, ValueError):
        return None


def extract_code_from_patch(patch: str) -> str:
    """Extract added/changed lines from a unified diff as code fragment.

    Fixed: Uses removeprefix("b/") instead of lstrip("b/").
    """
    added_lines = []
    in_hunk = False

    for line in patch.split("\n"):
        if line.startswith("@@"):
            in_hunk = True
            continue
        if line.startswith("diff --git") or line.startswith("---") or line.startswith("+++"):
            continue
        if in_hunk:
            if line.startswith("+"):
                added_lines.append(line[1:])
            elif line.startswith("-"):
                continue
            else:
                if line.startswith(" "):
                    added_lines.append(line[1:])
                else:
                    added_lines.append(line)

    return "\n".join(added_lines)


def build_edit_style_answer(patch: str, changed_files: list[str]) -> str | None:
    """Build an edit-style answer from a patch.

    Format:
        In file path/to/file.py, replace:
        ```python
        old code
        ```
        with:
        ```python
        new code
        ```
    """
    edits = []

    current_file = None
    old_lines = []
    new_lines = []
    in_hunk = False

    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            # Flush previous hunk
            if current_file and (old_lines or new_lines):
                edits.append(
                    {
                        "file": current_file,
                        "old": "\n".join(old_lines),
                        "new": "\n".join(new_lines),
                    }
                )
                old_lines = []
                new_lines = []
            match = re.match(r"diff --git a/(.+?) b/(.+)$", line)
            if match:
                current_file = match.group(2)
            in_hunk = False
            continue

        if line.startswith("@@"):
            # Flush previous hunk in same file
            if old_lines or new_lines:
                edits.append(
                    {
                        "file": current_file,
                        "old": "\n".join(old_lines),
                        "new": "\n".join(new_lines),
                    }
                )
                old_lines = []
                new_lines = []
            in_hunk = True
            continue

        if line.startswith("---") or line.startswith("+++"):
            continue

        if in_hunk:
            if line.startswith("-"):
                old_lines.append(line[1:])
            elif line.startswith("+"):
                new_lines.append(line[1:])
            # Context lines are skipped in edit-style

    # Flush last hunk
    if current_file and (old_lines or new_lines):
        edits.append(
            {
                "file": current_file,
                "old": "\n".join(old_lines),
                "new": "\n".join(new_lines),
            }
        )

    if not edits:
        return None

    parts = []
    for edit in edits:
        if edit["old"] and edit["new"]:
            parts.append(
                f"In file {edit['file']}, replace:\n```python\n{edit['old']}\n```\nwith:\n```python\n{edit['new']}\n```"
            )
        elif edit["new"]:
            parts.append(f"In file {edit['file']}, add:\n```python\n{edit['new']}\n```")

    return "\n\n".join(parts) if parts else None


def extract_definition(source: str, name: str, max_chars: int = 2000) -> str | None:
    """Extract a specific function or class definition from source code using AST.

    Returns the source lines for that definition (including docstring), or None if
    the definition is not found or source cannot be parsed.

    Args:
        source: Python source code to search.
        name: Function or class name to extract.
        max_chars: Maximum number of characters to return (truncates if necessary).

    Returns:
        Source string for the definition, or None if not found.

    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return None

    lines = source.split("\n")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                start = node.lineno - 1
                end = node.end_lineno
                definition = "\n".join(lines[start:end])
                if len(definition) > max_chars:
                    definition = definition[:max_chars]
                return definition
    return None


def find_definitions_in_repo(
    names: set[str],
    repo_dir: Path,
    commit: str,
    max_names: int = 20,
    max_candidates_per_name: int = 3,
    file_cache: dict[str, str | None] | None = None,
) -> dict[str, dict]:
    """Search a bare git repo for function/class definitions by name.

    Issues a single ``git grep`` with all names batched into one regex, then
    fetches candidate files and extracts definitions with AST.  This is much
    faster than one grep call per name on large repos.

    Returns dict mapping name -> {"file": filepath, "source": definition_source}.
    """
    if not names:
        return {}

    # Use caller-supplied cache or a local one
    if file_cache is None:
        file_cache = {}

    results: dict[str, dict] = {}
    processed = 0

    for name in list(names)[:max_names]:
        if processed >= max_names:
            break
        processed += 1

        # Per-name grep with -l (list files only): fast because git stops at
        # first match per file.  Alternation patterns force PCRE which is ~20x
        # slower on large repos, so we search one name at a time.
        try:
            proc = subprocess.run(
                ["git", f"--git-dir={repo_dir}", "grep", "-l", f"def {name}", "HEAD"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (subprocess.TimeoutExpired, Exception):
            continue

        if proc.returncode != 0 or not proc.stdout.strip():
            continue

        candidate_files = [
            line.strip().removeprefix("HEAD:").lstrip("/")
            for line in proc.stdout.splitlines()
            if line.strip() and line.strip().removeprefix("HEAD:").lstrip("/").endswith(".py")
        ]

        for filepath in candidate_files[:max_candidates_per_name]:
            if filepath not in file_cache:
                file_cache[filepath] = fetch_file_at_commit(repo_dir, commit, filepath)
            content = file_cache[filepath]
            if content is None:
                continue
            definition = extract_definition(content, name)
            if definition:
                results[name] = {"file": filepath, "source": definition}
                break

    return results


def fetch_import_dependencies(
    source_files: dict[str, str],
    repo_dir: Path,
    commit: str,
) -> dict[str, str]:
    """Pass A — resolve relative imports and return referenced definitions.

    For each file in *source_files*:
      1. Parse AST for relative imports (``from . import X``,
         ``from .module import X``).
      2. Resolve the import to a file path relative to the changed file's
         directory.
      3. Fetch that file at *commit*.
      4. Extract only the definitions (functions/classes) that are actually
         referenced in the original *source_files* content.

    Args:
        source_files: Mapping of filepath -> source content (patch-touched files).
        repo_dir: Path to a bare git repository.
        commit: Git commit SHA to fetch dependency files at.

    Returns:
        Dict mapping ``filepath -> definition_source`` for all resolved
        dependency definitions.  Returns an empty dict if anything goes wrong.

    """
    if not source_files:
        return {}

    # Concatenate all original source content for reference lookups
    all_source = "\n".join(source_files.values())

    # Collect resolved dependency file paths to fetch
    dep_file_paths: set[str] = set()
    # Map dep file path -> set of names actually imported from it
    dep_file_names: dict[str, set[str]] = {}

    for filepath, content in source_files.items():
        file_dir = filepath.rsplit("/", 1)[0] if "/" in filepath else ""

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(content)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue

            imported_names = {alias.name for alias in node.names}

            if node.level == 0:
                # Absolute import — try to resolve as an internal repo path.
                # e.g. "from astropy.table.column import Column"
                #   → try "astropy/table/column.py" in the repo.
                # We verify existence lazily (fetch_file_at_commit returns None if absent).
                if node.module:
                    candidate = node.module.replace(".", "/") + ".py"
                    dep_file_paths.add(candidate)
                    dep_file_names.setdefault(candidate, set()).update(imported_names)
                continue

            # Relative import: resolve module path
            # level=1 -> same directory, level=2 -> parent, etc.
            parts = file_dir.split("/") if file_dir else []
            parent_parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts

            module = node.module or ""
            if module:
                module_path = "/".join(parent_parts + [module.replace(".", "/")]) + ".py"
            else:
                # ``from . import X`` — the names come from the package __init__
                module_path = "/".join(parent_parts + ["__init__.py"])

            dep_file_paths.add(module_path)
            dep_file_names.setdefault(module_path, set()).update(imported_names)

    if not dep_file_paths:
        return {}

    dependency_definitions: dict[str, str] = {}

    for dep_filepath in dep_file_paths:
        if dep_filepath in source_files:
            # Already in context — skip
            continue

        content = fetch_file_at_commit(repo_dir, commit, dep_filepath)
        if content is None:
            continue

        # Extract only definitions that are referenced in the original source
        names_from_file = dep_file_names.get(dep_filepath, set())
        definitions_found = []
        for name in names_from_file:
            # Only include if the name actually appears in the original source content
            if name not in all_source:
                continue
            definition = extract_definition(content, name)
            if definition:
                definitions_found.append(f"# {name}\n{definition}")

        if definitions_found:
            dependency_definitions[dep_filepath] = "\n\n".join(definitions_found)

    return dependency_definitions


def fetch_source_for_instance(
    instance: dict, repos_dir: Path = REPOS_DIR, use_github_api: bool = False
) -> dict | None:
    """Fetch all source files for a single SWE-bench instance.

    Args:
        instance: SWE-bench instance dict
        repos_dir: Directory for cloned repos
        use_github_api: If True, use GitHub raw API instead of git clone

    Returns dict with:
        instance_id, changed_files, source_files, patch_code, edit_style,
        modified_functions
    Or None if fetching failed.

    """
    repo = instance["repo"]
    commit = instance["base_commit"]
    patch = instance["patch"]

    # Extract changed files from patch
    changed_files = extract_changed_files(patch)
    if not changed_files:
        return None

    # Fetch source files
    source_files = {}
    repo_dir = None

    if use_github_api:
        # Use GitHub raw API (slower but no cloning needed)
        import time

        for filepath in changed_files:
            try:
                content = fetch_file_from_github(repo, commit, filepath)
            except TransientFetchError as exc:
                print(f"    transient fetch failure, skipping {filepath}: {exc}")
                content = None
            if content:
                source_files[filepath] = content
            time.sleep(0.3)
    else:
        # Clone repo and use git show (fast for bulk)
        repo_dir = clone_repo(repo, repos_dir)
        if repo_dir is None:
            # Fallback to GitHub API
            import time

            print(f"    Falling back to GitHub API for {repo}")
            for filepath in changed_files:
                try:
                    content = fetch_file_from_github(repo, commit, filepath)
                except TransientFetchError as exc:
                    print(f"    transient fetch failure, skipping {filepath}: {exc}")
                    content = None
                if content:
                    source_files[filepath] = content
                time.sleep(0.3)
        else:
            for filepath in changed_files:
                content = fetch_file_at_commit(repo_dir, commit, filepath)
                if content:
                    source_files[filepath] = content

    if not source_files:
        return None

    # Build different answer formats
    # Fragment format
    patch_code = extract_code_from_patch(patch)

    # Edit-style format
    edit_style = build_edit_style_answer(patch, changed_files)

    # Complete function format — extract modified functions (needs full content)
    modified_functions = []
    for filepath in changed_files:
        if filepath not in source_files:
            continue
        patched_source = apply_patch_in_memory(source_files[filepath], patch, filepath)
        if patched_source:
            funcs = extract_modified_functions(source_files[filepath], patched_source)
            for func in funcs:
                func["file"] = filepath
            modified_functions.extend(funcs)

    # Smart truncation AFTER patch application: keep header + patch-relevant regions
    # instead of blind first-N-chars truncation
    for filepath in list(source_files.keys()):
        source_files[filepath] = truncate_around_patch(source_files[filepath], patch, filepath)

    # Pass A: fetch import-based dependency definitions
    dependency_files: dict[str, str] = {}
    if repo_dir is not None:
        try:
            dependency_files = fetch_import_dependencies(source_files, repo_dir, commit)
        except Exception as e:
            print(
                f"    Warning: fetch_import_dependencies failed for {instance['instance_id']}: {e}"
            )

    return {
        "instance_id": instance["instance_id"],
        "changed_files": changed_files,
        "source_files": source_files,
        "patch_code": patch_code,
        "edit_style": edit_style,
        "modified_functions": modified_functions,
        "dependency_files": dependency_files,
    }


def run(instances: list[dict], use_github_api: bool = False):
    """Run Phase 2: Fetch source files for all instances.

    Args:
        instances: List of SWE-bench instances
        use_github_api: If True, use GitHub raw API instead of cloning repos

    """
    print("=" * 60)
    print("Phase 2: Source File Fetching")
    print("=" * 60)

    # Suppress SyntaxWarning from ast.parse on third-party source files
    warnings.filterwarnings("ignore", category=SyntaxWarning)

    SOURCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPOS_DIR.mkdir(parents=True, exist_ok=True)

    if use_github_api:
        print("Using GitHub raw API (no cloning)")

    # Track repos that failed to clone so we don't retry
    clone_failed_repos: set[str] = set()

    # Fetch sources per instance
    results = []
    failed = 0

    for i, instance in enumerate(instances):
        if (i + 1) % 100 == 0 or (i + 1) == len(instances):
            print(f"  Phase 2: {i + 1}/{len(instances)} ({len(results)} success, {failed} failed)")

        # Skip if already cached
        cache_path = SOURCE_CACHE_DIR / f"{instance['instance_id']}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                results.append(json.load(f))
            continue

        # Try clone first, fall back to GitHub API
        repo = instance["repo"]
        use_api_for_this = use_github_api
        if not use_api_for_this and repo not in clone_failed_repos:
            repo_dir = clone_repo(repo)
            if repo_dir is None:
                clone_failed_repos.add(repo)
                use_api_for_this = True
                print(f"  Falling back to GitHub API for {repo}")

        result = fetch_source_for_instance(
            instance, use_github_api=use_api_for_this or repo in clone_failed_repos
        )
        if result:
            results.append(result)
            cache_path = SOURCE_CACHE_DIR / f"{instance['instance_id']}.json"
            with open(cache_path, "w") as f:
                json.dump(result, f)
        else:
            failed += 1

    print(f"\nDone: {len(results)} success, {failed} failed out of {len(instances)}")

    # Report format availability
    n_fragment = sum(1 for r in results if r.get("patch_code", "").strip())
    n_edit = sum(1 for r in results if r.get("edit_style"))
    n_function = sum(1 for r in results if r.get("modified_functions"))
    print("Format availability:")
    print(f"  Fragment: {n_fragment}")
    print(f"  Edit-style: {n_edit}")
    print(f"  Complete function: {n_function}")

    return results


if __name__ == "__main__":
    from .swebench_loader import load_instances

    instances = load_instances()
    run(instances)
