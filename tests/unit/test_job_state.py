"""Unit tests for common/services/job_state.py — _run_subprocess with PID tracking,
cancellation, and log streaming.

All DB and subprocess operations are mocked so no running database or process
is needed.
"""
from __future__ import annotations

import signal
from threading import Event
from unittest.mock import MagicMock, patch

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
             patch("common.services.job_state._append_log"):
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


# ---------------------------------------------------------------------------
# Tests: _run_refresh_customer_analytics (Recalculate button handler)
# ---------------------------------------------------------------------------

class TestRefreshCustomerAnalyticsHandler:
    """Refreshes the 6 CA materialized views CONCURRENTLY off the request thread."""

    def _mock_conn_with_cursor(self):
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn = _make_mock_conn()
        mock_conn.cursor = MagicMock(return_value=mock_cur)
        return mock_conn, mock_cur

    def test_refreshes_all_six_mvs_concurrently(self):
        from common.services.job_state import _run_refresh_customer_analytics

        mock_conn, mock_cur = self._mock_conn_with_cursor()
        with patch("common.services.job_state._get_conn", return_value=mock_conn):
            result = _run_refresh_customer_analytics({})

        assert result["failed"] == []
        assert result["refreshed"] == [
            "mv_customer_activity_monthly",
            "mv_customer_filter_options",
            "mv_ca_segment_trends",
            "mv_ca_demand_at_risk",
            "mv_ca_order_patterns",
            "mv_ca_item_state",
        ]
        # activity rollup must be refreshed first
        executed = " ".join(str(c.args[0]) for c in mock_cur.execute.call_args_list)
        assert "CONCURRENTLY" in executed

    def test_continues_past_a_failing_mv(self):
        import psycopg

        from common.services.job_state import _run_refresh_customer_analytics

        mock_conn, mock_cur = self._mock_conn_with_cursor()

        # First REFRESH (after the SET) raises; the rest succeed.
        calls = {"n": 0}

        def _execute(stmt, *a, **k):
            text = str(stmt)
            if "REFRESH" in text:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise psycopg.Error("lock timeout")
            return MagicMock()

        mock_cur.execute.side_effect = _execute
        with patch("common.services.job_state._get_conn", return_value=mock_conn):
            result = _run_refresh_customer_analytics({})

        assert len(result["failed"]) == 1
        assert len(result["refreshed"]) == 5

    def test_raises_when_cancelled(self):
        from threading import Event

        from common.services.job_state import _run_refresh_customer_analytics

        mock_conn, _ = self._mock_conn_with_cursor()
        cancel = Event()
        cancel.set()
        with patch("common.services.job_state._get_conn", return_value=mock_conn):
            with pytest.raises(RuntimeError, match="cancelled"):
                _run_refresh_customer_analytics({}, cancel_event=cancel)


class TestJobScriptPathsExist:
    """Guard against script-path rot: every script a job launches must exist.

    The backtest/pipeline scripts were reorganized into domain subdirs
    (scripts/ml, scripts/inventory, ...). job_state.py held flat scripts/<name>.py
    paths that no longer resolved, so UI-launched jobs failed with
    'No such file or directory'. These tests parse the real path literals from
    the module source so a future move can't silently break the Jobs tab again.
    """

    def _module_src(self) -> str:
        from pathlib import Path

        import common.services.job_state as job_state

        return Path(job_state.__file__).read_text()

    def test_all_referenced_script_files_exist(self):
        import re

        from common.core.paths import PROJECT_ROOT

        refs = sorted(set(re.findall(r'"(scripts/[A-Za-z0-9_./-]+\.py)"', self._module_src())))
        assert refs, "expected job_state to reference at least one script path"
        missing = [r for r in refs if not (PROJECT_ROOT / r).is_file()]
        assert not missing, f"job_state references missing script files: {missing}"

    def test_foundation_module_paths_resolve(self):
        import re

        from common.core.paths import PROJECT_ROOT

        mods = sorted(set(re.findall(r'"(scripts\.ml\.[A-Za-z0-9_.]+)"', self._module_src())))
        missing = [m for m in mods if not (PROJECT_ROOT / (m.replace(".", "/") + ".py")).is_file()]
        assert not missing, f"job_state references missing foundation modules: {missing}"


# ---------------------------------------------------------------------------
# Tests: _run_generate_production_forecast — horizon + CI flag threading
# ---------------------------------------------------------------------------

class TestGenerateProductionForecastCmd:
    """The generate handler must thread horizon + CI into the subprocess cmd."""

    def _run(self, params):
        from common.services.job_state import _run_generate_production_forecast
        with patch(
            "common.services.job_state._run_subprocess", return_value="ok"
        ) as m_sub:
            _run_generate_production_forecast(params)
        return m_sub.call_args[0][0]  # first positional arg = cmd list

    def test_threads_horizon_and_model_id(self):
        cmd = self._run({"model_id": "lgbm_cluster", "horizon": 9})
        assert "--horizon" in cmd
        assert cmd[cmd.index("--horizon") + 1] == "9"
        assert "--model-id" in cmd
        assert cmd[cmd.index("--model-id") + 1] == "lgbm_cluster"

    def test_confidence_intervals_true_adds_flag(self):
        cmd = self._run({"model_id": "lgbm_cluster", "confidence_intervals": True})
        assert "--confidence-intervals" in cmd
        assert "--no-confidence-intervals" not in cmd

    def test_confidence_intervals_false_adds_negated_flag(self):
        cmd = self._run({"model_id": "lgbm_cluster", "confidence_intervals": False})
        assert "--no-confidence-intervals" in cmd
        assert "--confidence-intervals" not in cmd

    def test_unset_params_omit_optional_flags(self):
        cmd = self._run({"model_id": "lgbm_cluster"})
        assert "--horizon" not in cmd
        assert "--confidence-intervals" not in cmd
        assert "--no-confidence-intervals" not in cmd


# ---------------------------------------------------------------------------
# Tests: auto-load after a successful backtest (no manual Load needed)
# ---------------------------------------------------------------------------

_MOD = "common.services.job_state"


class TestAutoLoadBacktest:
    """_auto_load_backtest loads predictions on completion, best-effort."""

    def test_calls_loader_when_predictions_exist(self):
        from common.services.job_state import _auto_load_backtest
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(f"{_MOD}._run_load_backtest_model") as m_load,
        ):
            _auto_load_backtest("lgbm", 7)
        m_load.assert_called_once()
        args, _ = m_load.call_args
        # First positional arg is the params dict: full dir model_id + run_id.
        assert args[0] == {"model_id": "lgbm_cluster", "run_id": 7}

    def test_skips_when_no_predictions(self):
        from common.services.job_state import _auto_load_backtest
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch(f"{_MOD}._run_load_backtest_model") as m_load,
        ):
            _auto_load_backtest("catboost", 3)
        m_load.assert_not_called()

    def test_swallows_loader_errors(self):
        from common.services.job_state import _auto_load_backtest
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(f"{_MOD}._run_load_backtest_model", side_effect=RuntimeError("boom")),
        ):
            # Must not raise — a load failure cannot fail a completed backtest.
            _auto_load_backtest("xgboost", 9)

    def test_non_tree_model_uses_identity_dir(self):
        from common.services.job_state import _auto_load_backtest
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(f"{_MOD}._run_load_backtest_model") as m_load,
        ):
            _auto_load_backtest("chronos_bolt", 1)
        assert m_load.call_args[0][0] == {"model_id": "chronos_bolt", "run_id": 1}

    def test_customer_enriched_tree_model_uses_own_output_dir(self):
        from common.services.job_state import _auto_load_backtest
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(f"{_MOD}._run_load_backtest_model") as m_load,
        ):
            _auto_load_backtest("xgboost_cust_enriched", 11)
        assert m_load.call_args[0][0] == {
            "model_id": "xgboost_cust_enriched",
            "run_id": 11,
        }


class TestRunBacktestAutoLoadsBeforeCompletion:
    """_run_backtest auto-loads BEFORE marking the run completed, so the UI sees
    is_loaded_to_db set by the time status flips to 'completed'."""

    def test_auto_load_runs_before_completion_update(self):
        from common.services.job_state import _run_backtest
        manager = MagicMock()
        with (
            patch(f"{_MOD}._run_subprocess", return_value="ok"),
            patch(f"{_MOD}._get_conn"),
            patch(f"{_MOD}._auto_load_backtest") as m_auto,
            patch(f"{_MOD}._update_backtest_run_on_completion") as m_update,
        ):
            manager.attach_mock(m_auto, "auto")
            manager.attach_mock(m_update, "update")
            _run_backtest("lgbm", {"backtest_run_id": 5})

        m_auto.assert_called_once()
        # auto-load got (model, run_id)
        assert m_auto.call_args[0][0] == "lgbm"
        assert m_auto.call_args[0][1] == 5
        # ordering: auto-load before completion update
        names = [c[0] for c in manager.mock_calls]
        assert names.index("auto") < names.index("update")

    def test_customer_enriched_tree_model_id_threads_to_subprocess_and_outputs(self):
        from common.services.job_state import _run_backtest
        with (
            patch(f"{_MOD}._run_subprocess", return_value="ok") as m_run,
            patch(f"{_MOD}._get_conn"),
            patch(f"{_MOD}._auto_load_backtest") as m_auto,
            patch(f"{_MOD}._update_backtest_run_on_completion") as m_update,
        ):
            _run_backtest(
                "catboost",
                {"backtest_run_id": 8, "model_id": "catboost_cust_enriched"},
            )

        cmd = m_run.call_args[0][0]
        assert cmd[cmd.index("--model-id") + 1] == "catboost_cust_enriched"
        assert m_auto.call_args[0][0] == "catboost_cust_enriched"
        assert m_update.call_args[0][1] == "catboost_cust_enriched"
