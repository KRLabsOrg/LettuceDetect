"""Ground an answer's code references against the repository.

A generated solution may call methods or use imported names whose definitions
were truncated out of the source context. Left ungrounded, those references look
like fabrications to a detector. This resolves such references to their real
definitions at the base commit (via GitHub raw) so the context supports them.
"""

from __future__ import annotations

import re
import time
from functools import lru_cache

from .source_fetcher import TransientFetchError, extract_definition, fetch_file_from_github

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SELF_CALL = re.compile(r"\bself\.([A-Za-z_]\w*)\s*\(")
_BARE_CALL = re.compile(r"(?<![.\w])([A-Za-z_]\w*)\s*\(")
_DEF = re.compile(r"\bdef\s+([A-Za-z_]\w*)")
_FROM_IMPORT = re.compile(r"^\s*from\s+(\.*)([\w.]*)\s+import\s+(.+?)$", re.MULTILINE)


def _import_names(piece_group: str) -> list[str]:
    """Split an ``import a, b as c`` clause into the bound names."""
    names = []
    for piece in piece_group.split(","):
        name = piece.strip().split(" as ")[0].strip().strip("()")
        if name and name != "*":
            names.append(name)
    return names


_BUILTINS = frozenset(
    "len str int float bool list dict tuple set print range enumerate sorted map filter "
    "zip open isinstance getattr setattr hasattr super type repr min max sum abs all any "
    "format join split strip startswith endswith replace encode decode get items keys "
    "values update append extend pop remove add insert index count find lower upper "
    "super property staticmethod classmethod next iter reversed round bytes bytearray "
    "frozenset object Exception ValueError TypeError KeyError IndexError "
    # typing / abc / common stdlib names that aren't fabricated references
    "Optional List Dict Tuple Set Union Any Callable Iterable Iterator Sequence Mapping "
    "Type Annotated Literal cast TYPE_CHECKING ABC ABCMeta abstractmethod dataclass field "
    "partial namedtuple defaultdict OrderedDict Counter Path datetime timedelta "
    # python-2 builtins and common stdlib helpers (not fabricated references)
    "long unicode basestring xrange unichr raw_input quote unquote urlparse urlencode "
    "urlopen urljoin warning info debug error exception critical log "
    "callable vars dir hash id slice bin hex oct ord chr divmod pow input eval exec "
    # exception/builtin types and keywords commonly mis-flagged as references
    "issubclass NotImplemented NotImplementedError RuntimeError AssertionError ImportError "
    "AttributeError StopIteration GeneratorExit KeyboardInterrupt OSError IOError RuntimeWarning "
    "FileNotFoundError PermissionError ConnectionError TimeoutError RecursionError NameError "
    "OverflowError ZeroDivisionError ArithmeticError LookupError UnicodeError DeprecationWarning "
    "UserWarning Warning FutureWarning StopAsyncIteration globals locals assert except elif yield "
    "return raise pass break continue lambda assert_called assertEqual assertTrue assertFalse "
    "assertRaises assertIn assertIsNone assertIsInstance assertListEqual assertDictEqual "
    # common stdlib helpers
    "contextmanager wraps reduce StringIO BytesIO Enum IntEnum IntFlag auto deepcopy copy "
    "unicode_literals print_function division absolute_import annotations TODO FIXME XXX".split()
)

_BOUND_ASSIGN = re.compile(r"(?m)^\s*([A-Za-z_]\w*)\s*=(?!=)")
_BOUND_AS = re.compile(r"\b(?:for|as)\s+([A-Za-z_]\w*)")
_DEF_PARAMS = re.compile(r"\bdef\s+\w+\s*\(([^)]*)\)")


def _locally_bound(answer: str) -> set[str]:
    """Names the answer defines/binds itself (def, class, assignment, loop, params)."""
    bound = set(_DEF.findall(answer))
    bound.update(re.findall(r"\bclass\s+([A-Za-z_]\w*)", answer))
    bound.update(_BOUND_ASSIGN.findall(answer))
    bound.update(_BOUND_AS.findall(answer))
    for params in _DEF_PARAMS.findall(answer):
        for piece in params.split(","):
            name = piece.split(":")[0].split("=")[0].strip().lstrip("*")
            if name.isidentifier():
                bound.add(name)
    return bound


def _strip_noncode(text: str) -> str:
    """Remove comments and string/docstring contents so prose words aren't read as references."""
    text = re.sub(r"'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"", " ", text)  # triple-quoted
    text = re.sub(r"#[^\n]*", " ", text)  # line comments
    return re.sub(r"\"[^\"\n]*\"|'[^'\n]*'", " ", text)  # single-line strings


def _referenced_names(answer: str) -> set[str]:
    """Collect callable/imported names the answer refers to (ignoring comments and strings)."""
    code = _strip_noncode(answer)
    names: set[str] = set()
    names.update(_SELF_CALL.findall(code))
    names.update(_BARE_CALL.findall(code))
    for _dots, _module, group in _FROM_IMPORT.findall(code):
        names.update(_import_names(group))
    return names


@lru_cache(maxsize=4096)
def _fetch_cached(repo: str, commit: str, path: str) -> str | None:
    """Fetch a repo file at a commit, cached across samples (same repo reuses modules).

    Only definitive results are cached: a :class:`TransientFetchError` raised by
    the fetch propagates (lru_cache does not cache exceptions), so a timeout or
    rate-limit window never becomes a permanent miss.
    """
    return fetch_file_from_github(repo, commit, path)


def _fetch_with_retry(repo: str, commit: str, path: str) -> str | None:
    """Fetch via the cache with one retry on transient failure; log and skip after that."""
    for attempt in (1, 2):
        try:
            return _fetch_cached(repo, commit, path)
        except TransientFetchError as exc:
            if attempt == 2:
                print(f"  grounding fetch failed (transient, skipped): {exc}")
                return None
            time.sleep(2.0)
    return None


def _internal_module_paths(text: str, from_file: str) -> list[str]:
    """Candidate repo .py paths for repo-internal modules imported in ``text``.

    Resolves relative imports against ``from_file``'s package and absolute imports
    that share its top-level package; third-party imports are skipped.
    """
    pkg_parts = from_file.rsplit("/", 1)[0].split("/") if "/" in from_file else []
    top = pkg_parts[0] if pkg_parts else ""
    paths: list[str] = []
    for dots, module in re.findall(r"^\s*from\s+(\.*)([\w.]*)\s+import", text, re.MULTILINE):
        if dots:
            keep = pkg_parts[: len(pkg_parts) - (len(dots) - 1)] if len(dots) > 1 else pkg_parts
            base = "/".join([*keep, *(module.split(".") if module else [])])
        elif module and module.split(".")[0] == top:
            base = module.replace(".", "/")
        else:
            continue
        if base:
            paths.append(base + ".py")
            paths.append(base + "/__init__.py")
    return paths


def _import_targets(answer: str, changed_files: list[str]) -> dict[str, list[str]]:
    """Map each imported name in the answer to candidate repo paths for its module.

    Resolves both absolute-internal imports (``from pkg.mod import x`` →
    ``pkg/mod.py``) and relative ones (``from .mod import x``) against the package
    directory of the first changed file. Third-party modules simply won't resolve.
    """
    pkg_parts: list[str] = []
    if changed_files and "/" in changed_files[0]:
        pkg_parts = changed_files[0].rsplit("/", 1)[0].split("/")
    out: dict[str, list[str]] = {}
    for dots, module, group in _FROM_IMPORT.findall(answer):
        if dots:
            keep = pkg_parts[: len(pkg_parts) - (len(dots) - 1)] if len(dots) > 1 else pkg_parts
            base = "/".join([*keep, *(module.split(".") if module else [])])
        elif module:
            base = module.replace(".", "/")
        else:
            continue
        if not base:
            continue
        paths = [base + ".py", base + "/__init__.py"]
        for name in _import_names(group):
            out[name] = paths
    return out


def _ungrounded_targets(answer: str, context: str) -> set[str]:
    """Referenced names absent from the context and not bound by the answer itself."""
    ctx_names = set(_IDENT.findall(context))
    bound = _locally_bound(answer)
    return {
        name
        for name in _referenced_names(answer)
        if len(name) > 2
        and not name.startswith("__")
        and name not in _BUILTINS
        and name not in bound
        and name not in ctx_names
    }


def resolve_definitions(
    answer: str,
    context: str,
    *,
    repo: str,
    commit: str,
    changed_files: list[str],
    modified_functions: list[dict],
    max_defs: int = 8,
    max_modules: int = 25,
) -> dict[str, str]:
    """Return ``{name: definition_source}`` for answer references found in the repo.

    Tiers, cheapest first: the patch's modified functions; the full changed files;
    modules the answer imports; and modules the changed files import (which reach
    base-class mixins and sibling modules, grounding ``self.method`` references).
    Only names that resolve to a real definition are returned.
    """
    targets = _ungrounded_targets(answer, context)
    if not targets:
        return {}

    def fetch(path: str) -> str | None:
        return _fetch_with_retry(repo, commit, path)

    def absorb(content: str | None) -> None:
        if not content:
            return
        for name in list(targets):
            definition = extract_definition(content, name)
            if definition:
                grounded[name] = definition
                targets.discard(name)

    grounded: dict[str, str] = {}
    for func in modified_functions:
        name = func.get("name")
        if name in targets and func.get("patched"):
            grounded[name] = func["patched"]
            targets.discard(name)

    # Tier 2: the full changed files.
    changed_contents = {}
    for filepath in changed_files:
        if not targets or len(grounded) >= max_defs:
            break
        changed_contents[filepath] = fetch(filepath)
        absorb(changed_contents[filepath])

    # Tier 3: modules the answer explicitly imports.
    imports = _import_targets(answer, changed_files)
    for name in list(targets):
        if len(grounded) >= max_defs:
            break
        for path in imports.get(name, []):
            definition = extract_definition(fetch(path) or "", name)
            if definition:
                grounded[name] = definition
                targets.discard(name)
                break

    # Tier 4: modules imported by the changed files (base mixins, sibling modules).
    if targets and len(grounded) < max_defs:
        module_paths: list[str] = []
        for filepath, content in changed_contents.items():
            if content:
                module_paths += _internal_module_paths(content, filepath)
        seen: set[str] = set()
        for path in module_paths:
            if not targets or len(grounded) >= max_defs or len(seen) >= max_modules:
                break
            if path in seen:
                continue
            seen.add(path)
            absorb(fetch(path))

    return grounded


_SECTION = frozenset(
    {"description", "usage", "parameters", "returns", "example", "examples", "note"}
)


def _signatures(doc: str, max_lines: int = 24) -> str:
    """Reduce a Context7 doc to API names, call signatures, and parameter names."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in doc.splitlines():
        s = raw.strip()
        if not s or s.startswith("```") or s.lower().startswith("source:"):
            continue
        if s.startswith("#"):  # API name header
            name = s.lstrip("# ").strip()
            if name and name.lower() not in _SECTION:
                keep = name
            else:
                continue
        elif s.startswith(("*", "-")):  # parameter bullet -> "name (type)"
            keep = s.lstrip("*- ").split(" - ")[0].replace("**", "").strip()
        elif "(" in s and ")" in s and not s.endswith(":") and "=" not in s.split("(", 1)[0]:
            keep = s  # a call / usage line
        else:
            continue
        if keep and keep not in seen:
            seen.add(keep)
            out.append(keep)
        if len(out) >= max_lines:
            break
    return "\n".join(out)


def ground_fabricated_apis(changes: list[dict], *, max_calls: int = 3) -> str:
    """Fetch Context7 signatures for the *real* third-party APIs a structural edit replaced.

    Each structural change's ``original`` holds the real call (e.g.
    ``torch.cuda.set_device(...)``) before the fabrication. Querying Context7 for
    that exact dotted path retrieves the real signature, so the fabricated member
    becomes verifiable from the context. Returns a context block, or "".
    """
    from .context7_docs import PATH_TO_LIB, fetch_context7_docs

    refs: list[tuple[str, str]] = []  # (library, exact dotted path)
    seen: set[str] = set()
    for change in changes:
        original = change.get("original", "")
        for mod, lib in PATH_TO_LIB.items():
            for match in re.finditer(rf"\b({re.escape(mod)}(?:\.\w+)+)", original):
                path = match.group(1)
                if path not in seen:
                    seen.add(path)
                    refs.append((lib, path))

    blocks: list[str] = []
    for lib, query in refs[:max_calls]:
        doc = fetch_context7_docs(lib, query, max_chars=4000)
        sigs = _signatures(doc) if doc else ""
        if sigs:
            blocks.append(f"## {query}\n{sigs}")
    if not blocks:
        return ""
    return "Library signatures:\n" + "\n\n".join(blocks)


def render_definitions(definitions: dict[str, str]) -> str:
    """Render resolved definitions as a context block."""
    if not definitions:
        return ""
    body = "\n\n".join(f"# {name}\n{source}" for name, source in definitions.items())
    return "Referenced definitions:\n```python\n" + body + "\n```"


def remaining_ungrounded(answer: str, context: str) -> set[str]:
    """Names the answer references that are still absent from the context.

    Names the answer itself imports are not counted: the import statement is
    evidence the name exists (stdlib/third-party modules aren't in the repo, so
    they can't be grounded from it; repo-internal imports are grounded
    separately by :func:`resolve_definitions`).
    """
    targets = _ungrounded_targets(answer, context)
    imported: set[str] = set()
    for _dots, _module, group in _FROM_IMPORT.findall(_strip_noncode(answer)):
        imported.update(_import_names(group))
    return targets - imported
