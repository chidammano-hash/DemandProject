#!/usr/bin/env python3
"""Lint for f-string SQL that interpolates identifiers as values.

Project rule: psycopg3 uses ``%s`` placeholders and parameterised queries.
F-strings embedded in ``cur.execute(...)`` calls are only acceptable when
they build SQL *shape* (column lists, identifiers, literal table names) from
trusted sources. They are *never* acceptable for user input or dynamic values.

This tool scans ``.py`` files and flags any ``cur.execute(f"...`` /
``cur.execute(f'''...`` call whose f-string body contains ``{...}`` expressions
that look like runtime variables.

Works on both the legacy (pre-3.12) tokenizer, which emits f-strings as a single
STRING token, and the modern tokenizer (3.12+) which emits
FSTRING_START/FSTRING_MIDDLE/... /FSTRING_END.

Exit codes:
- 0 if no offending f-string executes were found
- 1 if at least one was found (so this can wire into CI / pre-commit)

Usage:
    python scripts/tools/check_fstring_sql.py [paths...]

With no arguments, scans ``api/``, ``common/``, ``scripts/``.

See CLAUDE.md "Critical Rules" for the parameterised-query requirement.
"""
from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import tokenize
from pathlib import Path

logger = logging.getLogger(__name__)

# A placeholder is "safe" (not a value) if its expression is empty, pure
# whitespace, or only a format-spec (``:>5``).
_SAFE_PLACEHOLDER = re.compile(r"^\s*(:[^}]*)?$")

# Python 3.12+ tokenizer emits these; on earlier versions the attributes are
# absent and we fall back to STRING-token parsing.
_FSTRING_START = getattr(tokenize, "FSTRING_START", None)
_FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", None)
_FSTRING_END = getattr(tokenize, "FSTRING_END", None)


def _find_variable_placeholders_in_body(body: str) -> list[str]:
    """Return ``{...}`` placeholder expressions from an old-style STRING literal body."""
    variables: list[str] = []
    depth = 0
    buf: list[str] = []
    in_expr = False
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "{" and i + 1 < len(body) and body[i + 1] == "{":
            i += 2
            continue
        if ch == "}" and i + 1 < len(body) and body[i + 1] == "}":
            i += 2
            continue
        if ch == "{":
            if depth == 0:
                buf = []
                in_expr = True
            depth += 1
            i += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0 and in_expr:
                expr = "".join(buf).strip()
                if not _SAFE_PLACEHOLDER.match(expr):
                    variables.append(expr)
                in_expr = False
            i += 1
            continue
        if in_expr and depth > 0:
            buf.append(ch)
        i += 1
    return variables


def _strip_quotes(literal: str) -> str:
    """Remove surrounding quote characters (triple or single) from a string literal body."""
    for q in ('"""', "'''", '"', "'"):
        if literal.startswith(q) and literal.endswith(q) and len(literal) >= 2 * len(q):
            return literal[len(q):-len(q)]
    return literal


def _iter_execute_calls(tokens: list[tokenize.TokenInfo]):
    """Yield (line_no, snippet, fstring_token_start_index) for each ``cur.execute(f"..."``.

    The caller inspects the tokens after ``fstring_token_start_index`` to collect the
    interpolation expressions.
    """
    for idx, tok in enumerate(tokens):
        if tok.type != tokenize.NAME or tok.string != "execute":
            continue
        if idx < 2:
            continue
        prev = tokens[idx - 1]
        prev2 = tokens[idx - 2]
        if prev.type != tokenize.OP or prev.string != ".":
            continue
        if prev2.type != tokenize.NAME or prev2.string not in {"cur", "cursor"}:
            continue
        # Advance to `(`
        j = idx + 1
        while j < len(tokens) and tokens[j].type in {tokenize.NL, tokenize.NEWLINE}:
            j += 1
        if j >= len(tokens) or tokens[j].type != tokenize.OP or tokens[j].string != "(":
            continue
        # Skip NL, then expect f-string start (either FSTRING_START or STRING starting with f-prefix)
        k = j + 1
        while k < len(tokens) and tokens[k].type in {tokenize.NL, tokenize.NEWLINE}:
            k += 1
        if k >= len(tokens):
            continue
        first_arg = tokens[k]
        line_no = first_arg.start[0]
        snippet = f"{prev2.string}.execute(f"

        if _FSTRING_START is not None and first_arg.type == _FSTRING_START:
            yield (line_no, snippet, "new", k)
            continue
        if first_arg.type == tokenize.STRING:
            s = first_arg.string
            if re.match(r"^(?:rf|fr|RF|FR|f|F)", s):
                yield (line_no, snippet, "legacy", k)


def _collect_new_style_variables(tokens: list[tokenize.TokenInfo], start_idx: int) -> list[str]:
    """Collect interpolation expressions between FSTRING_START and FSTRING_END.

    Every ``{`` (OP) after FSTRING_START that is matched by a ``}`` (OP) — before
    FSTRING_END — wraps an expression. We concatenate the token strings inside
    as a best-effort representation of the interpolated identifier.
    """
    variables: list[str] = []
    depth = 0
    expr_tokens: list[str] = []
    i = start_idx + 1
    while i < len(tokens):
        tok = tokens[i]
        if _FSTRING_END is not None and tok.type == _FSTRING_END:
            return variables
        if tok.type == tokenize.OP and tok.string == "{":
            if depth == 0:
                expr_tokens = []
            depth += 1
            i += 1
            continue
        if tok.type == tokenize.OP and tok.string == "}":
            depth -= 1
            if depth == 0:
                expr = " ".join(expr_tokens).strip()
                if expr and not _SAFE_PLACEHOLDER.match(expr):
                    variables.append(expr)
            i += 1
            continue
        if depth > 0:
            # Stop capturing once we hit a format-spec delimiter (":") at depth==1.
            if tok.type == tokenize.OP and tok.string == ":" and depth == 1:
                # remainder is format spec — consume until matching "}"
                j = i + 1
                inner = 0
                while j < len(tokens):
                    t2 = tokens[j]
                    if t2.type == tokenize.OP and t2.string == "{":
                        inner += 1
                    elif t2.type == tokenize.OP and t2.string == "}":
                        if inner == 0:
                            break
                        inner -= 1
                    j += 1
                # Treat what we've collected as the expression
                expr = " ".join(expr_tokens).strip()
                if expr and not _SAFE_PLACEHOLDER.match(expr):
                    variables.append(expr)
                depth -= 1
                i = j + 1 if j < len(tokens) else j
                continue
            if tok.string.strip():
                expr_tokens.append(tok.string)
        i += 1
    return variables


def check_file(path: Path) -> list[tuple[int, str, list[str]]]:
    """Return (line_no, snippet, variables) for offending execute calls."""
    text = path.read_text(encoding="utf-8", errors="replace")
    offenders: list[tuple[int, str, list[str]]] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenizeError, SyntaxError, IndentationError) as exc:
        logger.warning("tokenize failed for %s: %s", path, exc)
        return offenders

    for line_no, snippet, style, idx in _iter_execute_calls(tokens):
        if style == "legacy":
            literal = tokens[idx].string
            m = re.match(r"^(?:rf|fr|RF|FR|f|F)", literal)
            body = _strip_quotes(literal[m.end():]) if m else literal
            variables = _find_variable_placeholders_in_body(body)
        else:
            variables = _collect_new_style_variables(tokens, idx)
        if variables:
            offenders.append((line_no, snippet, variables))
    return offenders


def scan_paths(paths: list[Path]) -> dict[Path, list[tuple[int, str, list[str]]]]:
    results: dict[Path, list[tuple[int, str, list[str]]]] = {}
    for root in paths:
        if root.is_file() and root.suffix == ".py":
            candidates = [root]
        else:
            candidates = sorted(root.rglob("*.py"))
        for py in candidates:
            rel = str(py)
            if "__pycache__" in rel or "/.venv/" in rel or "/archive/" in rel:
                continue
            try:
                offenders = check_file(py)
            except OSError as exc:
                logger.warning("skipping %s: %s", py, exc)
                continue
            if offenders:
                results[py] = offenders
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to scan (default: api/ common/ scripts/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only emit offenders; suppress the success banner",
    )
    args = parser.parse_args()

    roots = args.paths or [
        Path("api"),
        Path("common"),
        Path("scripts"),
    ]
    roots = [r for r in roots if r.exists()]
    if not roots:
        print("No paths to scan", file=sys.stderr)
        return 0

    results = scan_paths(roots)
    if not results:
        if not args.quiet:
            print("OK: no f-string SQL with variable placeholders found.")
        return 0

    total = 0
    for path, offenders in results.items():
        for line_no, snippet, variables in offenders:
            total += 1
            vars_str = ", ".join(f"{{{v}}}" for v in variables)
            print(f"{path}:{line_no}: {snippet}...) - interpolates {vars_str}")
    print(
        f"\nFAIL: {total} f-string SQL execute call(s) with variable interpolation. "
        "Parameterised queries (%s placeholders) are required. See CLAUDE.md.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
