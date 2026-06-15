"""Unit tests for ``common.services.integration_runner.IntegrationRunner``.

The pool, cursor and ``subprocess.run`` are all mocked — no real DB, no real
subprocess invocation. The ThreadPoolExecutor is patched so that ``submit()``
does not actually spawn a background job.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import psycopg

from common.services.integration_runner import IntegrationRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_pool(fetchone=None, fetchall=None, *, description=None):
    """Return ``(pool, cursor)`` mocks wired with the standard ctx-manager dance."""
    cursor = MagicMock()
    cursor.fetchone.return_value = fetchone
    cursor.fetchall.return_value = fetchall if fetchall is not None else []
    cursor.description = description or [
        ("id",), ("domain",), ("mode",), ("slice",), ("file_path",),
        ("status",), ("rows_loaded",), ("error_message",),
        ("started_at",), ("completed_at",), ("duration_ms",), ("triggered_by",),
    ]

    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    pool = MagicMock()
    pool.connection.return_value = conn
    return pool, cursor


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------
def test_get_returns_dict_when_row_exists():
    row = (
        "abc-123", "sales", "onetime", None, None,
        "success", 100, None, None, None, 1234, "api",
    )
    pool, cursor = _make_pool(fetchone=row)
    runner = IntegrationRunner(pool)

    result = runner.get("abc-123")
    assert isinstance(result, dict)
    for key in (
        "id", "domain", "mode", "slice", "file_path", "status",
        "rows_loaded", "error_message", "started_at", "completed_at",
        "duration_ms", "triggered_by",
    ):
        assert key in result
    assert result["id"] == "abc-123"
    assert result["status"] == "success"
    assert result["rows_loaded"] == 100


def test_get_returns_none_when_no_row():
    pool, _ = _make_pool(fetchone=None)
    runner = IntegrationRunner(pool)
    assert runner.get("missing-id") is None


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------
def test_list_with_domain_filter():
    rows = [
        ("id1", "sales", "delta", None, None, "success", 10, None, None, None, 1, "api"),
        ("id2", "sales", "delta", None, None, "failed",   0, "x",  None, None, 2, "api"),
    ]
    pool, cursor = _make_pool(fetchall=rows)
    runner = IntegrationRunner(pool)

    items = runner.list(domain="sales")
    assert len(items) == 2

    sql = cursor.execute.call_args.args[0]
    params = cursor.execute.call_args.args[1]
    assert "WHERE domain" in sql
    assert params[0] == "sales"


def test_list_default_limit_50():
    pool, cursor = _make_pool(fetchall=[])
    runner = IntegrationRunner(pool)
    runner.list()

    params = cursor.execute.call_args.args[1]
    # No domain filter → tuple of (limit,)
    assert params == (50,)


def test_list_respects_custom_limit():
    pool, cursor = _make_pool(fetchall=[])
    runner = IntegrationRunner(pool)
    runner.list(limit=7)
    assert cursor.execute.call_args.args[1] == (7,)


def test_list_with_domain_and_custom_limit():
    pool, cursor = _make_pool(fetchall=[])
    runner = IntegrationRunner(pool)
    runner.list(domain="forecast", limit=11)
    assert cursor.execute.call_args.args[1] == ("forecast", 11)


# ---------------------------------------------------------------------------
# US17b — reads come from the unified view (writes still hit the base table)
# ---------------------------------------------------------------------------
def test_list_reads_unified_view():
    pool, cursor = _make_pool(fetchall=[])
    runner = IntegrationRunner(pool)
    runner.list()
    sql = cursor.execute.call_args.args[0]
    assert "integration_job_unified" in sql


def test_get_reads_unified_view():
    row = (
        "abc-123", "sales", "onetime", None, None,
        "success", 100, None, None, None, 1234, "api",
    )
    pool, cursor = _make_pool(fetchone=row)
    runner = IntegrationRunner(pool)
    runner.get("abc-123")
    sql = cursor.execute.call_args.args[0]
    assert "integration_job_unified" in sql


def test_writes_still_target_base_table():
    # purge / reap must never write to the view.
    pool, cursor = _make_pool(fetchall=[])
    runner = IntegrationRunner(pool)
    runner.purge()
    sql = cursor.execute.call_args.args[0]
    assert "integration_job_unified" not in sql
    assert "integration_job" in sql


# ---------------------------------------------------------------------------
# health()
# ---------------------------------------------------------------------------
def test_health_pool_ok_table_ok():
    pool, cursor = _make_pool(fetchone=(True,))
    runner = IntegrationRunner(pool)

    h = runner.health()
    assert h == {"pool": "ok", "table": "ok"}


def test_health_pool_degraded():
    """If the connection raises, pool status must report ``degraded``."""
    pool = MagicMock()
    pool.connection.side_effect = psycopg.Error("connect refused")
    runner = IntegrationRunner(pool)

    h = runner.health()
    assert h["pool"] == "degraded"
    assert h["table"] == "missing"  # table check also fails because pool fails


def test_health_table_missing():
    """to_regclass returning False yields table='missing'."""
    pool, cursor = _make_pool(fetchone=(False,))
    runner = IntegrationRunner(pool)

    h = runner.health()
    assert h["pool"] == "ok"
    assert h["table"] == "missing"


def test_health_table_check_handles_db_error():
    """Errors during the table-existence query degrade gracefully."""
    pool, cursor = _make_pool(fetchone=(True,))
    cursor.execute.side_effect = psycopg.Error("relation error")
    runner = IntegrationRunner(pool)

    h = runner.health()
    assert h["table"] == "missing"


