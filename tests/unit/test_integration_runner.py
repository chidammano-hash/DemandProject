"""Unit tests for ``common.services.integration_runner.IntegrationRunner``.

The pool, cursor and ``subprocess.run`` are all mocked — no real DB, no real
subprocess invocation. The ThreadPoolExecutor is patched so that ``submit()``
does not actually spawn a background job.
"""
from __future__ import annotations

import subprocess
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch
from uuid import uuid4

import psycopg
import pytest

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
# submit()
# ---------------------------------------------------------------------------
def test_submit_validates_mode():
    pool, _ = _make_pool()
    runner = IntegrationRunner(pool)
    with pytest.raises(ValueError) as exc:
        runner.submit(domain="sales", mode="bogus")
    assert "invalid mode" in str(exc.value)


@pytest.mark.parametrize("mode", ["onetime", "delta", "file"])
def test_submit_accepts_valid_modes(mode):
    job_uuid = uuid4()
    pool, cursor = _make_pool(fetchone=(job_uuid,))
    runner = IntegrationRunner(pool)

    with patch.object(IntegrationRunner, "_executor", new=MagicMock()):
        job_id = runner.submit(domain="sales", mode=mode)
    assert isinstance(job_id, str)


def test_submit_inserts_queued_row_and_returns_job_id():
    job_uuid = uuid4()
    pool, cursor = _make_pool(fetchone=(job_uuid,))
    runner = IntegrationRunner(pool)

    fake_executor = MagicMock()
    with patch.object(IntegrationRunner, "_executor", new=fake_executor):
        job_id = runner.submit(
            domain="sales",
            mode="onetime",
            triggered_by="api",
        )

    # Returned id is the stringified UUID
    assert job_id == str(job_uuid)

    # INSERT ran, with 'queued' literal in the SQL.
    cursor.execute.assert_called_once()
    sql_arg = cursor.execute.call_args.args[0]
    params_arg = cursor.execute.call_args.args[1]
    assert "INSERT INTO integration_job" in sql_arg
    assert "'queued'" in sql_arg
    assert params_arg == ("sales", "onetime", None, None, "api")

    # Background job was scheduled on the executor.
    fake_executor.submit.assert_called_once()
    submitted_callable = fake_executor.submit.call_args.args[0]
    assert callable(submitted_callable)


def test_submit_passes_slice_and_file():
    job_uuid = uuid4()
    pool, cursor = _make_pool(fetchone=(job_uuid,))
    runner = IntegrationRunner(pool)

    with patch.object(IntegrationRunner, "_executor", new=MagicMock()):
        runner.submit(
            domain="customer_demand",
            mode="file",
            slice="2026-04",
            file="/tmp/foo.csv",
            triggered_by="scheduler",
        )

    params = cursor.execute.call_args.args[1]
    assert params == ("customer_demand", "file", "2026-04", "/tmp/foo.csv", "scheduler")


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


# ---------------------------------------------------------------------------
# _run_job — exercise the threaded subprocess path directly
# ---------------------------------------------------------------------------
def _completed(returncode: int, stdout: str = "", stderr: str = "") -> CompletedProcess:
    return CompletedProcess(args=["uv"], returncode=returncode, stdout=stdout, stderr=stderr)


def _capture_final_update(cursor) -> tuple[str, ...]:
    """Return the params of the LAST UPDATE call (the finalization step)."""
    update_calls = [
        c for c in cursor.execute.call_args_list
        if "UPDATE integration_job" in c.args[0] and "completed_at" in c.args[0]
    ]
    assert update_calls, "no finalization UPDATE found"
    return update_calls[-1].args[1]


def test_run_job_success_path():
    pool, cursor = _make_pool()
    runner = IntegrationRunner(pool)

    fake = _completed(0, stdout='{"rows_loaded": 42}\n')
    with patch("subprocess.run", return_value=fake):
        runner._run_job("job-1", "sales", "onetime", None, None)

    params = _capture_final_update(cursor)
    status, rows_loaded, *_rest, error_message, duration_ms, job_id = params
    assert status == "success"
    assert rows_loaded == 42
    assert error_message is None
    assert isinstance(duration_ms, int)
    assert job_id == "job-1"


def test_run_job_skipped_path():
    pool, cursor = _make_pool()
    runner = IntegrationRunner(pool)

    fake = _completed(2, stdout='{"rows_loaded": 0}\n')
    with patch("subprocess.run", return_value=fake):
        runner._run_job("job-2", "sales", "delta", None, None)

    params = _capture_final_update(cursor)
    status, rows_loaded, *_diff_cols, error_message, _duration_ms, _job_id = params
    assert status == "skipped"
    assert rows_loaded == 0


def test_run_job_failed_path_uses_stderr_when_no_json_error():
    pool, cursor = _make_pool()
    runner = IntegrationRunner(pool)

    fake = _completed(1, stdout="", stderr="boom: load crashed")
    with patch("subprocess.run", return_value=fake):
        runner._run_job("job-3", "sales", "onetime", None, None)

    params = _capture_final_update(cursor)
    status, rows_loaded, *_diff_cols, error_message, _duration_ms, _job_id = params
    assert status == "failed"
    assert rows_loaded == 0
    assert error_message is not None
    assert "boom" in error_message


def test_run_job_failed_path_uses_json_error_field():
    pool, cursor = _make_pool()
    runner = IntegrationRunner(pool)

    fake = _completed(1, stdout='{"rows_loaded": 0, "error": "schema mismatch"}\n')
    with patch("subprocess.run", return_value=fake):
        runner._run_job("job-4", "sales", "onetime", None, None)

    params = _capture_final_update(cursor)
    status, _rows, *_diff_cols, error_message, _duration_ms, _job_id = params
    assert status == "failed"
    assert error_message == "schema mismatch"


def test_run_job_subprocess_error():
    """A SubprocessError during subprocess.run -> status='failed'."""
    pool, cursor = _make_pool()
    runner = IntegrationRunner(pool)

    with patch("subprocess.run", side_effect=subprocess.SubprocessError("spawn failed")):
        runner._run_job("job-5", "sales", "onetime", None, None)

    params = _capture_final_update(cursor)
    status, _rows, *_diff_cols, error_message, _duration_ms, _job_id = params
    assert status == "failed"
    assert error_message is not None
    assert "subprocess error" in error_message


def test_run_job_timeout_marks_failed():
    pool, cursor = _make_pool()
    runner = IntegrationRunner(pool)

    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="uv", timeout=3600),
    ):
        runner._run_job("job-6", "sales", "onetime", None, None)

    params = _capture_final_update(cursor)
    status, _rows, *_diff_cols, error_message, _duration_ms, _job_id = params
    assert status == "failed"
    assert error_message is not None
    assert "timed out" in error_message


def test_run_job_includes_slice_and_file_in_command():
    pool, _cursor = _make_pool()
    runner = IntegrationRunner(pool)

    fake = _completed(0, stdout='{"rows_loaded": 1}\n')
    with patch("subprocess.run", return_value=fake) as mocked_run:
        runner._run_job("job-7", "customer_demand", "file", "2026-04", "/tmp/x.csv")

    cmd = mocked_run.call_args.args[0]
    assert "--domain" in cmd and "customer_demand" in cmd
    assert "--mode" in cmd and "file" in cmd
    assert "--slice" in cmd and "2026-04" in cmd
    assert "--file" in cmd and "/tmp/x.csv" in cmd


# ---------------------------------------------------------------------------
# _parse_final_json
# ---------------------------------------------------------------------------
_EMPTY = {"rows_loaded": 0, "rows_inserted": None, "rows_updated": None,
          "rows_deleted": None, "error": None}


@pytest.mark.parametrize(
    "stdout,expected",
    [
        ("", _EMPTY),
        ("\n\n", _EMPTY),
        ("not json at all\n", _EMPTY),
        ('{"rows_loaded": 42}\n', {**_EMPTY, "rows_loaded": 42}),
        ('{"rows_loaded": "17"}\n', {**_EMPTY, "rows_loaded": 17}),
        ('{"rows_loaded": null}\n', _EMPTY),
        ('{"rows_loaded": 5, "error": "oops"}\n',
         {**_EMPTY, "rows_loaded": 5, "error": "oops"}),
        ('garbage line\n{"rows_loaded": 9}\n', {**_EMPTY, "rows_loaded": 9}),
        ('[1,2,3]\n', _EMPTY),
        ('{"rows_loaded": 1, "rows_inserted": 1, "rows_updated": 0, "rows_deleted": 0}\n',
         {**_EMPTY, "rows_loaded": 1, "rows_inserted": 1,
          "rows_updated": 0, "rows_deleted": 0}),
    ],
)
def test_parse_final_json(stdout, expected):
    assert IntegrationRunner._parse_final_json(stdout) == expected
