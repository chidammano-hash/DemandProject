"""Unit tests for ``common.services.pg_queue`` (Item 22 pilot).

All DB calls are mocked via ``patch('common.services.pg_queue._connect')``.
The mock pattern follows ``test_job_registry.py`` — a context-manager
connection that yields a context-manager cursor.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import psycopg
import pytest

from common.services import pg_queue

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_conn() -> MagicMock:
    """Build a MagicMock that mimics psycopg.Connection + Cursor context managers."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)

    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn._cursor = cursor  # convenience handle for assertions
    return conn


@pytest.fixture
def mock_conn():
    """Patch ``pg_queue._connect`` to return a MagicMock connection."""
    conn = _make_conn()
    with patch("common.services.pg_queue._connect", return_value=conn):
        yield conn


# ---------------------------------------------------------------------------
# enqueue_job
# ---------------------------------------------------------------------------


class TestEnqueueJob:
    def test_returns_inserted_id(self, mock_conn):
        mock_conn._cursor.fetchone.return_value = (42,)
        result = pg_queue.enqueue_job("refresh_intramonth", {"foo": "bar"})
        assert result == 42

    def test_commits_after_insert(self, mock_conn):
        mock_conn._cursor.fetchone.return_value = (1,)
        pg_queue.enqueue_job("refresh_intramonth")
        mock_conn.commit.assert_called_once()

    def test_default_params_serialised_as_empty_dict(self, mock_conn):
        mock_conn._cursor.fetchone.return_value = (1,)
        pg_queue.enqueue_job("refresh_intramonth")
        # Inspect the params positional argument of execute(...)
        args, _ = mock_conn._cursor.execute.call_args
        bound = args[1]
        # bound = (job_type, params_json, status, priority, run_at, max_attempts)
        assert bound[0] == "refresh_intramonth"
        assert json.loads(bound[1]) == {}
        assert bound[2] == pg_queue.STATUS_PENDING
        assert bound[3] == 100  # default priority
        assert bound[5] == 3    # default max_attempts

    def test_custom_priority_and_max_attempts(self, mock_conn):
        mock_conn._cursor.fetchone.return_value = (7,)
        pg_queue.enqueue_job(
            "refresh_intramonth", priority=10, max_attempts=5,
        )
        args, _ = mock_conn._cursor.execute.call_args
        bound = args[1]
        assert bound[3] == 10
        assert bound[5] == 5

    def test_empty_job_type_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            pg_queue.enqueue_job("")

    def test_zero_max_attempts_raises(self):
        with pytest.raises(ValueError, match="max_attempts"):
            pg_queue.enqueue_job("foo", max_attempts=0)


# ---------------------------------------------------------------------------
# claim_next_job
# ---------------------------------------------------------------------------


class TestClaimNextJob:
    def test_returns_none_when_queue_empty(self, mock_conn):
        mock_conn._cursor.fetchone.return_value = None
        result = pg_queue.claim_next_job(worker_id="w1")
        assert result is None
        mock_conn.rollback.assert_called_once()

    def test_claims_pending_job(self, mock_conn):
        # SELECT ... FOR UPDATE returns one row, then UPDATE succeeds.
        mock_conn._cursor.fetchone.return_value = (
            5, "refresh_intramonth", {"foo": "bar"}, 0, 3,
        )
        result = pg_queue.claim_next_job(worker_id="w1")
        assert result is not None
        assert result["id"] == 5
        assert result["job_type"] == "refresh_intramonth"
        assert result["params"] == {"foo": "bar"}
        assert result["attempts"] == 0
        assert result["max_attempts"] == 3
        assert result["worker_id"] == "w1"
        # Two execute calls: SELECT FOR UPDATE then UPDATE to claimed.
        assert mock_conn._cursor.execute.call_count == 2
        mock_conn.commit.assert_called_once()

    def test_claim_with_job_types_filter(self, mock_conn):
        mock_conn._cursor.fetchone.return_value = None
        pg_queue.claim_next_job(worker_id="w1", job_types=["refresh_intramonth"])
        # First execute is the SELECT — verify its bind parameters end with our list
        first_call_args = mock_conn._cursor.execute.call_args_list[0]
        bound = first_call_args[0][1]
        assert ["refresh_intramonth"] in bound

    def test_claim_parses_json_string_params(self, mock_conn):
        # When psycopg returns params as a JSON string instead of dict.
        mock_conn._cursor.fetchone.return_value = (
            9, "refresh_intramonth", '{"k": "v"}', 1, 3,
        )
        result = pg_queue.claim_next_job(worker_id="w1")
        assert result["params"] == {"k": "v"}

    def test_claim_rollback_on_db_error(self, mock_conn):
        mock_conn._cursor.execute.side_effect = psycopg.Error("boom")
        with pytest.raises(psycopg.Error):
            pg_queue.claim_next_job(worker_id="w1")
        mock_conn.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestStateTransitions:
    def test_mark_running_increments_attempts(self, mock_conn):
        pg_queue.mark_running(123)
        args, _ = mock_conn._cursor.execute.call_args
        sql_text = args[0]
        bound = args[1]
        assert "attempts = attempts + 1" in sql_text
        assert pg_queue.STATUS_RUNNING in bound
        assert pg_queue.STATUS_CLAIMED in bound  # WHERE clause guard
        assert 123 in bound
        mock_conn.commit.assert_called_once()

    def test_mark_completed_serialises_result(self, mock_conn):
        pg_queue.mark_completed(7, {"output": "ok"})
        args, _ = mock_conn._cursor.execute.call_args
        bound = args[1]
        assert pg_queue.STATUS_COMPLETED in bound
        # Result JSON is the third positional binding — find any string value
        # that successfully parses as JSON and matches the expected payload.
        def _safe_loads(value: object) -> object:
            if not isinstance(value, str):
                return None
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return None
        assert any(_safe_loads(v) == {"output": "ok"} for v in bound)
        mock_conn.commit.assert_called_once()

    def test_mark_completed_with_none_result(self, mock_conn):
        pg_queue.mark_completed(7, None)
        args, _ = mock_conn._cursor.execute.call_args
        bound = args[1]
        # result payload is None when no result given
        assert None in bound

    def test_mark_failed_truncates_long_error(self, mock_conn):
        big = "x" * 20000
        pg_queue.mark_failed(11, big)
        args, _ = mock_conn._cursor.execute.call_args
        bound = args[1]
        # Find the truncated string in the bind tuple
        truncated = next(v for v in bound if isinstance(v, str) and v.startswith("x"))
        assert len(truncated) == 8000

    def test_mark_failed_handles_empty_string(self, mock_conn):
        # Should not crash on empty error.
        pg_queue.mark_failed(11, "")
        mock_conn.commit.assert_called_once()


# ---------------------------------------------------------------------------
# requeue_failed_with_backoff
# ---------------------------------------------------------------------------


class TestRequeueWithBackoff:
    def test_requeue_when_under_max_attempts(self, mock_conn):
        # SELECT returns (attempts=1, max_attempts=3) → under cap → requeue.
        mock_conn._cursor.fetchone.return_value = (1, 3)
        result = pg_queue.requeue_failed_with_backoff(99)
        assert result is True
        # Two executes: SELECT FOR UPDATE + UPDATE to pending.
        assert mock_conn._cursor.execute.call_count == 2
        mock_conn.commit.assert_called_once()
        # Inspect the UPDATE binds
        update_args = mock_conn._cursor.execute.call_args_list[1][0]
        bound = update_args[1]
        assert pg_queue.STATUS_PENDING in bound

    def test_no_requeue_when_max_attempts_reached(self, mock_conn):
        mock_conn._cursor.fetchone.return_value = (3, 3)
        result = pg_queue.requeue_failed_with_backoff(99)
        assert result is False
        # Only the SELECT ran — no UPDATE.
        assert mock_conn._cursor.execute.call_count == 1
        mock_conn.rollback.assert_called_once()

    def test_no_requeue_when_job_missing(self, mock_conn):
        mock_conn._cursor.fetchone.return_value = None
        result = pg_queue.requeue_failed_with_backoff(404)
        assert result is False
        mock_conn.rollback.assert_called_once()

    def test_backoff_grows_exponentially(self, mock_conn):
        # attempts=4 → delay = 60 * 2^4 = 960s, capped at 3600s.
        mock_conn._cursor.fetchone.return_value = (4, 10)
        pg_queue.requeue_failed_with_backoff(50)
        update_args = mock_conn._cursor.execute.call_args_list[1][0]
        bound = update_args[1]
        next_run = next(v for v in bound if isinstance(v, datetime))
        delta = (next_run - datetime.now(UTC)).total_seconds()
        # Allow some tolerance for clock drift during the test.
        assert 950 <= delta <= 970

    def test_backoff_is_capped(self, mock_conn):
        # attempts=20 → 60 * 2^20 way overflows; should cap at _BACKOFF_CAP_SECONDS.
        mock_conn._cursor.fetchone.return_value = (20, 100)
        pg_queue.requeue_failed_with_backoff(60)
        update_args = mock_conn._cursor.execute.call_args_list[1][0]
        bound = update_args[1]
        next_run = next(v for v in bound if isinstance(v, datetime))
        delta = (next_run - datetime.now(UTC)).total_seconds()
        assert delta <= pg_queue._BACKOFF_CAP_SECONDS + 5


# ---------------------------------------------------------------------------
# get_queue_depth
# ---------------------------------------------------------------------------


class TestQueueDepth:
    def test_empty_queue_returns_zero_counts(self, mock_conn):
        mock_conn._cursor.fetchall.return_value = []
        result = pg_queue.get_queue_depth()
        assert result == {
            "pending": 0, "claimed": 0, "running": 0,
            "completed": 0, "failed": 0,
        }

    def test_populated_counts(self, mock_conn):
        mock_conn._cursor.fetchall.return_value = [
            ("pending", 3),
            ("running", 1),
            ("completed", 12),
        ]
        result = pg_queue.get_queue_depth()
        assert result["pending"] == 3
        assert result["running"] == 1
        assert result["completed"] == 12
        assert result["failed"] == 0  # untouched default

    def test_filter_by_job_type(self, mock_conn):
        mock_conn._cursor.fetchall.return_value = [("pending", 2)]
        pg_queue.get_queue_depth(job_type="refresh_intramonth")
        args, _ = mock_conn._cursor.execute.call_args
        bound = args[1]
        assert bound == ("refresh_intramonth",)


# ---------------------------------------------------------------------------
# Worker default identifier
# ---------------------------------------------------------------------------


class TestDefaultWorkerId:
    def test_uses_env_override(self, monkeypatch):
        monkeypatch.setenv("PG_QUEUE_WORKER_ID", "custom-worker-1")
        assert pg_queue._default_worker_id() == "custom-worker-1"

    def test_falls_back_to_host_pid(self, monkeypatch):
        monkeypatch.delenv("PG_QUEUE_WORKER_ID", raising=False)
        wid = pg_queue._default_worker_id()
        # Format: <host>:<pid>
        assert ":" in wid
        host, pid_str = wid.rsplit(":", 1)
        assert pid_str.isdigit()
        assert host  # non-empty
