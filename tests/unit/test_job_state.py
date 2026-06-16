"""Unit tests for common/services/job_state.py — _run_subprocess with PID tracking,
cancellation, and log streaming.

All DB and subprocess operations are mocked so no running database or process
is needed.
"""
from __future__ import annotations

import signal
from threading import Event
from unittest.mock import MagicMock, patch, call

import pytest

from common.services.job_state import (
    _run_subprocess,
    _store_pid,
    _clear_pid,
    _append_log,
    get_job_log,
    get_job_pid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_conn():
    """Create a mock connection that works as a context manager."""
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    return mock_conn


# ---------------------------------------------------------------------------
# Tests: PID helpers
# ---------------------------------------------------------------------------

class TestStorePid:
    def test_stores_pid_in_db(self):
        mock_conn = _make_mock_conn()
        with patch("common.services.job_state._get_conn", return_value=mock_conn):
            _store_pid("job_123", 42)
        mock_conn.execute.assert_called_once()
        args = mock_conn.execute.call_args
        assert "UPDATE job_history SET pid = %s" in args[0][0]
        assert args[0][1] == (42, "job_123")

    def test_no_op_when_no_job_id(self):
        with patch("common.services.job_state._get_conn") as mock_get:
            _store_pid(None, 42)
        mock_get.assert_not_called()

    def test_swallows_db_errors(self):
        mock_conn = _make_mock_conn()
        mock_conn.execute.side_effect = Exception("DB down")
        with patch("common.services.job_state._get_conn", return_value=mock_conn):
            _store_pid("job_123", 42)  # should not raise


class TestClearPid:
    def test_clears_pid_in_db(self):
        mock_conn = _make_mock_conn()
        with patch("common.services.job_state._get_conn", return_value=mock_conn):
            _clear_pid("job_123")
        mock_conn.execute.assert_called_once()
        args = mock_conn.execute.call_args
        assert "pid = NULL" in args[0][0]


class TestAppendLog:
    def test_appends_text_to_db(self):
        mock_conn = _make_mock_conn()
        with patch("common.services.job_state._get_conn", return_value=mock_conn):
            _append_log("job_123", "line1\nline2\n")
        mock_conn.execute.assert_called_once()

    def test_no_op_when_empty_text(self):
        with patch("common.services.job_state._get_conn") as mock_get:
            _append_log("job_123", "")
        mock_get.assert_not_called()


class TestGetJobLog:
    def test_returns_log_text(self):
        mock_conn = _make_mock_conn()
        mock_conn.execute.return_value.fetchone.return_value = ("hello world",)
        with patch("common.services.job_state._get_conn", return_value=mock_conn):
            result = get_job_log("job_123")
        assert result == "hello world"

    def test_returns_empty_string_on_not_found(self):
        mock_conn = _make_mock_conn()
        mock_conn.execute.return_value.fetchone.return_value = None
        with patch("common.services.job_state._get_conn", return_value=mock_conn):
            result = get_job_log("missing")
        assert result == ""


class TestGetJobPid:
    def test_returns_pid(self):
        mock_conn = _make_mock_conn()
        mock_conn.execute.return_value.fetchone.return_value = (1234,)
        with patch("common.services.job_state._get_conn", return_value=mock_conn):
            result = get_job_pid("job_123")
        assert result == 1234

    def test_returns_none_when_null(self):
        mock_conn = _make_mock_conn()
        mock_conn.execute.return_value.fetchone.return_value = (None,)
        with patch("common.services.job_state._get_conn", return_value=mock_conn):
            result = get_job_pid("job_123")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _run_subprocess
# ---------------------------------------------------------------------------

class TestRunSubprocess:
    def test_success_returns_stdout(self):
        """A successful subprocess should return its output and store/clear PID."""
        mock_proc = MagicMock()
        mock_proc.pid = 9999
        mock_proc.stdout = iter(["line1\n", "line2\n"])
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0

        with patch("common.services.job_state.subprocess.Popen", return_value=mock_proc), \
             patch("common.services.job_state._store_pid") as m_store, \
             patch("common.services.job_state._clear_pid") as m_clear, \
             patch("common.services.job_state._append_log") as m_log:
            result = _run_subprocess(["echo", "hi"], job_id="j1")

        assert "line1" in result
        assert "line2" in result
        m_store.assert_called_once_with("j1", 9999)
        m_clear.assert_called_once_with("j1")

    def test_start_new_session(self):
        """Subprocess should be started with start_new_session=True."""
        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.stdout = iter([])
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0

        with patch("common.services.job_state.subprocess.Popen", return_value=mock_proc) as m_popen, \
             patch("common.services.job_state._store_pid"), \
             patch("common.services.job_state._clear_pid"), \
             patch("common.services.job_state._append_log"):
            _run_subprocess(["test"])

        _, kwargs = m_popen.call_args
        assert kwargs["start_new_session"] is True

    def test_failure_raises_runtime_error(self):
        """Non-zero exit code should raise RuntimeError."""
        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.stdout = iter(["error output\n"])
        mock_proc.wait.return_value = None
        mock_proc.returncode = 1

        with patch("common.services.job_state.subprocess.Popen", return_value=mock_proc), \
             patch("common.services.job_state._store_pid"), \
             patch("common.services.job_state._clear_pid"), \
             patch("common.services.job_state._append_log"):
            with pytest.raises(RuntimeError, match="Command failed"):
                _run_subprocess(["false"], job_id="j1")

    def test_cancel_kills_process_group(self):
        """When cancel_event is set, the process group should be killed."""
        cancel = Event()
        cancel.set()  # pre-set so it cancels on first check

        mock_proc = MagicMock()
        mock_proc.pid = 5555
        # Simulate one line of output before cancel check
        mock_proc.stdout = iter(["output\n"])
        mock_proc.wait.return_value = None
        mock_proc.returncode = -15  # SIGTERM

        with patch("common.services.job_state.subprocess.Popen", return_value=mock_proc), \
             patch("common.services.job_state._store_pid"), \
             patch("common.services.job_state._clear_pid"), \
             patch("common.services.job_state._append_log"), \
             patch("common.services.job_state.os.killpg") as m_killpg, \
             patch("common.services.job_state.os.getpgid", return_value=5555):
            with pytest.raises(RuntimeError, match="cancelled"):
                _run_subprocess(["sleep", "100"], cancel_event=cancel, job_id="j1")

        m_killpg.assert_called_with(5555, signal.SIGTERM)

    def test_pid_cleared_on_error(self):
        """PID should be cleared even on error (via finally block)."""
        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.stdout = iter([])
        mock_proc.wait.return_value = None
        mock_proc.returncode = 1

        with patch("common.services.job_state.subprocess.Popen", return_value=mock_proc), \
             patch("common.services.job_state._store_pid"), \
             patch("common.services.job_state._clear_pid") as m_clear, \
             patch("common.services.job_state._append_log"):
            with pytest.raises(RuntimeError):
                _run_subprocess(["false"], job_id="j1")

        m_clear.assert_called_once_with("j1")

    def test_log_flushed_to_db(self):
        """Output should be flushed to the log column."""
        mock_proc = MagicMock()
        mock_proc.pid = 1
        # Generate enough lines to trigger flush
        mock_proc.stdout = iter([f"line{i}\n" for i in range(25)])
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0

        with patch("common.services.job_state.subprocess.Popen", return_value=mock_proc), \
             patch("common.services.job_state._store_pid"), \
             patch("common.services.job_state._clear_pid"), \
             patch("common.services.job_state._append_log") as m_log, \
             patch("common.services.job_state.time.time", side_effect=[0.0] + [0.0] * 100):
            _run_subprocess(["test"], job_id="j1")

        # Should have flushed at least once (20 line threshold)
        assert m_log.call_count >= 1

    def test_progress_cb_called(self):
        """progress_cb should be called with step_msg and stdout lines."""
        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.stdout = iter(["hello\n"])
        mock_proc.wait.return_value = None
        mock_proc.returncode = 0

        cb = MagicMock()
        with patch("common.services.job_state.subprocess.Popen", return_value=mock_proc), \
             patch("common.services.job_state._store_pid"), \
             patch("common.services.job_state._clear_pid"), \
             patch("common.services.job_state._append_log"):
            _run_subprocess(["test"], progress_cb=cb, step_msg="Starting")

        cb.assert_any_call(msg="Starting")
        cb.assert_any_call(msg="hello")


# ---------------------------------------------------------------------------
# Tests: _run_compute_sku_features param mapping
# ---------------------------------------------------------------------------

class TestComputeSkuFeaturesHandler:
    """The job/params schema exposes ``time_window_months`` but run_pipeline's
    kwarg is ``time_window`` — the handler must map between them (regression)."""

    def test_maps_time_window_months_param_to_run_pipeline_time_window(self):
        from common.services.job_state import _run_compute_sku_features

        with patch("scripts.ml.compute_sku_features.run_pipeline") as mock_run:
            mock_run.return_value = {"skus_processed": 5}
            result = _run_compute_sku_features({"time_window_months": 24})

        mock_run.assert_called_once_with(time_window=24)
        assert result == {"skus_processed": 5}

    def test_defaults_time_window_to_36_when_param_absent(self):
        from common.services.job_state import _run_compute_sku_features

        with patch("scripts.ml.compute_sku_features.run_pipeline") as mock_run:
            mock_run.return_value = {}
            _run_compute_sku_features({})

        mock_run.assert_called_once_with(time_window=36)
