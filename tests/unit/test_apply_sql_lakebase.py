"""Unit tests for the Lakebase DDL applier (scripts/db/apply_sql_lakebase.py)."""
from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

from scripts.db.apply_sql_lakebase import TOLERATE_FAIL, apply_files, select_files


def _paths(*names: str) -> list[Path]:
    return [Path("sql") / n for n in names]


# ---------------------------------------------------------------------------
# select_files — pure filtering
# ---------------------------------------------------------------------------
def test_select_files_no_filter_returns_all():
    files = _paths("001_a.sql", "002_b.sql", "170_x.sql")
    assert select_files(files) == files


def test_select_files_only_substring():
    files = _paths("001_dim_item.sql", "002_fact_sales.sql", "003_dim_sku.sql")
    assert select_files(files, only="dim_") == _paths("001_dim_item.sql", "003_dim_sku.sql")


def test_select_files_start_from_is_inclusive():
    files = _paths("001_a.sql", "100_b.sql", "196_c.sql")
    assert select_files(files, start_from="100_") == _paths("100_b.sql", "196_c.sql")


def test_select_files_combines_start_from_and_only():
    files = _paths("001_dim.sql", "100_dim.sql", "100_fact.sql")
    assert select_files(files, only="dim", start_from="100_") == _paths("100_dim.sql")


# ---------------------------------------------------------------------------
# apply_files — execution / tolerance behavior (with a fake connection)
# ---------------------------------------------------------------------------
class _FakeCursor:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc) -> None:
        return None

    def execute(self, sql_text: str) -> None:
        self._conn.executed.append(sql_text)
        if any(token in sql_text for token in self._conn.fail_on):
            raise psycopg.OperationalError("permission denied")


class _FakeConn:
    """Minimal psycopg.Connection stand-in; fails execute when a marker is in the SQL."""

    def __init__(self, fail_on: tuple[str, ...] = ()) -> None:
        self.fail_on = fail_on
        self.executed: list[str] = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def test_apply_files_dry_run_does_not_touch_db():
    files = _paths("001_a.sql", "002_b.sql")
    applied, warnings = apply_files(
        conn=None,  # dry-run path never dereferences conn
        files=files,
        continue_on_error=False,
        tolerate=frozenset(),
        dry_run=True,
    )
    assert applied == 2
    assert warnings == []


def test_apply_files_commits_each_file(tmp_path):
    f1 = tmp_path / "001_a.sql"
    f1.write_text("CREATE TABLE a();")
    f2 = tmp_path / "002_b.sql"
    f2.write_text("CREATE TABLE b();")
    conn = _FakeConn()
    applied, warnings = apply_files(
        conn, [f1, f2], continue_on_error=False, tolerate=frozenset(), dry_run=False
    )
    assert applied == 2
    assert warnings == []
    assert conn.commits == 2
    assert conn.rollbacks == 0


def test_apply_files_tolerates_listed_failure(tmp_path):
    good = tmp_path / "001_ok.sql"
    good.write_text("CREATE TABLE ok();")
    ext = tmp_path / "170_enable_pg_stat_statements.sql"
    ext.write_text("CREATE EXTENSION pg_stat_statements;")
    conn = _FakeConn(fail_on=("EXTENSION",))
    applied, warnings = apply_files(
        conn,
        [good, ext],
        continue_on_error=False,
        tolerate=frozenset({"170_enable_pg_stat_statements.sql"}),
        dry_run=False,
    )
    assert applied == 1  # only the good file
    assert len(warnings) == 1
    assert "170_enable_pg_stat_statements.sql" in warnings[0]
    assert conn.rollbacks == 1  # failed file rolled back


def test_apply_files_raises_on_untolerated_failure(tmp_path):
    bad = tmp_path / "050_bad.sql"
    bad.write_text("CREATE TABLE boom();")
    conn = _FakeConn(fail_on=("boom",))
    with pytest.raises(psycopg.Error):
        apply_files(conn, [bad], continue_on_error=False, tolerate=frozenset(), dry_run=False)
    assert conn.rollbacks == 1


def test_apply_files_continue_on_error_tolerates_everything(tmp_path):
    bad = tmp_path / "050_bad.sql"
    bad.write_text("CREATE TABLE boom();")
    conn = _FakeConn(fail_on=("boom",))
    applied, warnings = apply_files(
        conn, [bad], continue_on_error=True, tolerate=frozenset(), dry_run=False
    )
    assert applied == 0
    assert len(warnings) == 1


def test_pg_stat_statements_is_tolerated_by_default():
    assert "170_enable_pg_stat_statements.sql" in TOLERATE_FAIL
