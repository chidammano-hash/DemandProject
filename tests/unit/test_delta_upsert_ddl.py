"""DDL guard for the delta loader's ON CONFLICT requirement.

``scripts/etl/load.py::_safe_upsert`` resolves its ON CONFLICT target from the
target table's non-primary unique indexes (``_resolve_conflict_target``). If a
migration drops the last such index — as sql/172 did to fact_sales_monthly
(restored by sql/199) — every ``--mode delta`` load of that domain fails at
runtime with "no usable unique constraint ... for ON CONFLICT".

These tests replay the sql/ DDL in file order and assert every delta-loaded
domain table ends up with at least one surviving non-primary unique constraint
or unique index, turning that class of failure into a test failure instead of
a broken pipeline job.
"""
from __future__ import annotations

import re

import yaml

from common.core.domain_specs import get_spec
from common.core.paths import PROJECT_ROOT

SQL_DIR = PROJECT_ROOT / "sql"
ETL_CONFIG = PROJECT_ROOT / "config" / "etl" / "etl_config.yaml"

_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\(", re.IGNORECASE,
)
_CONSTRAINT_LINE_RE = re.compile(r"^\s*CONSTRAINT\s+(\w+)\s+UNIQUE\s*\(([^)]*)\)", re.IGNORECASE)
_INLINE_UNIQUE_RE = re.compile(r"^\s*(\w+)\s+\w+.*\bUNIQUE\b", re.IGNORECASE)
_CREATE_UNIQUE_INDEX_RE = re.compile(
    r"CREATE\s+UNIQUE\s+INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(\w+)\s+ON\s+(?:ONLY\s+)?(\w+)[^(]*\(([^)]*)\)",
    re.IGNORECASE,
)
_ADD_CONSTRAINT_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?(\w+)\s+"
    r"ADD\s+CONSTRAINT\s+(\w+)\s+UNIQUE\s*\(([^)]*)\)",
    re.IGNORECASE,
)
_DROP_CONSTRAINT_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?(\w+)\s+"
    r"DROP\s+CONSTRAINT\s+(?:IF\s+EXISTS\s+)?(\w+)",
    re.IGNORECASE,
)
_DROP_INDEX_RE = re.compile(
    r"DROP\s+INDEX\s+(?:CONCURRENTLY\s+)?(?:IF\s+EXISTS\s+)?(\w+)", re.IGNORECASE,
)
_DROP_TABLE_RE = re.compile(r"DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\w+)", re.IGNORECASE)
_RENAME_TABLE_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(\w+)\s+RENAME\s+TO\s+(\w+)", re.IGNORECASE,
)


def _delta_tables() -> list[str]:
    domains = yaml.safe_load(ETL_CONFIG.read_text())["domain_order"]
    return [get_spec(d).table for d in domains]


def _strip_comments(sql: str) -> str:
    return re.sub(r"--[^\n]*", "", sql)


def _cols(col_list: str) -> frozenset[str]:
    """First identifier of each comma-separated entry (drops opclass/DESC)."""
    out = set()
    for piece in col_list.split(","):
        m = re.match(r"\s*(\w+)", piece)
        if m:
            out.add(m.group(1).lower())
    return frozenset(out)


def _table_body(text: str, open_paren: int) -> str:
    """Return the paren-balanced CREATE TABLE body starting at ``open_paren``."""
    depth = 0
    for i in range(open_paren, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1 : i]
    return text[open_paren + 1 :]


def _create_table_uniques(table: str, body: str) -> dict[str, frozenset[str]]:
    found: dict[str, frozenset[str]] = {}
    for line in body.splitlines():
        m = _CONSTRAINT_LINE_RE.match(line)
        if m:
            found[m.group(1)] = _cols(m.group(2))
            continue
        m = _INLINE_UNIQUE_RE.match(line)
        is_constraint_line = line.lstrip().lower().startswith("constraint")
        if m and "primary" not in line.lower() and not is_constraint_line:
            col = m.group(1)
            found[f"{table}_{col}_key"] = frozenset({col.lower()})
    return found


def _surviving_uniques() -> dict[str, dict[str, frozenset[str]]]:
    """Replay sql/*.sql in order; return table -> {unique name -> columns}."""
    uniques: dict[str, dict[str, frozenset[str]]] = {}
    index_owner: dict[str, str] = {}

    def add(table: str, name: str, cols: frozenset[str]) -> None:
        uniques.setdefault(table, {})[name] = cols
        index_owner[name] = table

    # p.is_file() guard: sql/ contains a directory named 009_create_chat_embeddings.sql.
    for path in sorted(
        (p for p in SQL_DIR.glob("*.sql") if p.is_file()), key=lambda p: p.name,
    ):
        text = _strip_comments(path.read_text())
        events: list[tuple[int, str, tuple]] = []
        for m in _CREATE_TABLE_RE.finditer(text):
            body = _table_body(text, m.end() - 1)
            events.append((m.start(), "create_table", (m.group(1), body)))
        for m in _CREATE_UNIQUE_INDEX_RE.finditer(text):
            events.append((m.start(), "add", (m.group(2), m.group(1), _cols(m.group(3)))))
        for m in _ADD_CONSTRAINT_RE.finditer(text):
            events.append((m.start(), "add", (m.group(1), m.group(2), _cols(m.group(3)))))
        for m in _DROP_CONSTRAINT_RE.finditer(text):
            events.append((m.start(), "drop", (m.group(1), m.group(2))))
        for m in _DROP_INDEX_RE.finditer(text):
            events.append((m.start(), "drop_index", (m.group(1),)))
        for m in _DROP_TABLE_RE.finditer(text):
            events.append((m.start(), "drop_table", (m.group(1),)))
        for m in _RENAME_TABLE_RE.finditer(text):
            events.append((m.start(), "rename", (m.group(1), m.group(2))))

        for _pos, kind, payload in sorted(events, key=lambda e: e[0]):
            if kind == "create_table":
                table, body = payload
                for name, cols in _create_table_uniques(table, body).items():
                    add(table, name, cols)
            elif kind == "add":
                table, name, cols = payload
                add(table, name, cols)
            elif kind == "drop":
                table, name = payload
                uniques.get(table, {}).pop(name, None)
            elif kind == "drop_index":
                (name,) = payload
                owner = index_owner.get(name)
                if owner:
                    uniques.get(owner, {}).pop(name, None)
            elif kind == "drop_table":
                (table,) = payload
                uniques.pop(table, None)
            elif kind == "rename":
                old, new = payload
                if old in uniques:
                    uniques[new] = uniques.pop(old)
                    for name in uniques[new]:
                        index_owner[name] = new
    return uniques


def test_every_delta_domain_table_keeps_a_non_pk_unique_index():
    uniques = _surviving_uniques()
    missing = [t for t in _delta_tables() if not uniques.get(t)]
    assert not missing, (
        f"Delta-loaded tables without any surviving non-primary unique "
        f"constraint/index in sql/ DDL: {missing}. The delta loader's "
        f"INSERT ... ON CONFLICT needs one (see scripts/etl/load.py "
        f"_resolve_conflict_target); dropping the last one breaks every "
        f"`--mode delta` load of that domain. If a perf audit flags such an "
        f"index as unused, remember ON CONFLICT arbiter lookups do not "
        f"increment idx_scan (see sql/172 vs sql/199)."
    )


def test_sales_ck_unique_survives_the_ddl_replay():
    """Regression: sql/172 dropped the sales_ck unique; sql/199 restores it."""
    uniques = _surviving_uniques()
    sales = uniques.get("fact_sales_monthly", {})
    assert any("sales_ck" in cols for cols in sales.values()), (
        f"fact_sales_monthly has no surviving unique on sales_ck "
        f"(found: {sales}) — delta sales loads upsert ON CONFLICT (sales_ck)."
    )
