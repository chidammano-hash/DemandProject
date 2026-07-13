"""Unit tests for common/services/job_state.py — _run_subprocess with PID tracking,
cancellation, and log streaming.

All DB and subprocess operations are mocked so no running database or process
is needed.
"""
from __future__ import annotations

import signal
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from common.services.job_state import (
    _append_log,
    _clear_pid,
    _run_subprocess,
    _store_pid,
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
    @pytest.fixture(autouse=True)
    def _process_identity(self):
        identity = {
            "start_marker": "Sun Jul 12 03:00:00 2026",
            "command_marker": "test-command",
        }
        with (
            patch(
                "common.services.job_state.capture_process_identity",
                return_value=identity,
            ),
            patch("common.services.job_state._store_process_identity"),
        ):
            yield

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
    """Delegates to the central MV-refresh service keyed on the CA tables.

    The refresh mechanics (CONCURRENTLY fallback, failure isolation,
    cancellation) are covered by tests/unit/test_mv_refresh.py — here we pin
    the delegation contract only.
    """

    def test_delegates_to_central_service_with_ca_tables(self):
        from threading import Event

        from common.services.job_state import _run_refresh_customer_analytics

        cancel = Event()
        sentinel = {"refreshed": ["mv_customer_activity_monthly"], "failed": [], "missing": []}
        progress = MagicMock()
        with patch(
            "common.core.mv_refresh.refresh_for_tables", return_value=sentinel
        ) as refresh:
            result = _run_refresh_customer_analytics(
                {}, progress_cb=progress, cancel_event=cancel
            )

        assert result is sentinel
        refresh.assert_called_once()
        call = refresh.call_args
        assert call.args[0] == ["fact_customer_demand_monthly", "dim_customer"]
        assert call.kwargs["progress_cb"] is progress
        assert call.kwargs["cancel_event"] is cancel

    def test_ca_tables_cover_all_six_ca_mvs(self):
        # The derived set must include everything the CA tab reads.
        from common.core.mv_refresh import mvs_for_tables

        derived = set(mvs_for_tables(["fact_customer_demand_monthly", "dim_customer"]))
        assert {
            "mv_customer_activity_monthly",
            "mv_customer_filter_options",
            "mv_ca_segment_trends",
            "mv_ca_demand_at_risk",
            "mv_ca_order_patterns",
            "mv_ca_item_state",
        } <= derived


class TestRefreshForecastViewsHandler:
    """Delegates to the central MV-refresh service keyed on the forecast tables."""

    def test_delegates_with_forecast_tables(self):
        from common.services.job_state import _run_refresh_forecast_views

        sentinel = {"refreshed": [], "failed": [], "missing": []}
        with patch(
            "common.core.mv_refresh.refresh_for_tables", return_value=sentinel
        ) as refresh:
            result = _run_refresh_forecast_views({})

        assert result is sentinel
        assert refresh.call_args.args[0] == [
            "fact_external_forecast_monthly", "backtest_lag_archive",
        ]

    def test_forecast_tables_cover_naive_scale_and_by_dfu(self):
        # These two were the historically skipped MVs — pin their coverage.
        from common.core.mv_refresh import mvs_for_tables

        derived = set(
            mvs_for_tables(["fact_external_forecast_monthly", "backtest_lag_archive"])
        )
        assert "agg_dfu_naive_scale" in derived
        assert "agg_accuracy_by_dfu" in derived


class TestRefreshAllMvsHandler:
    """Refreshes every known MV; skip_heavy excludes HEAVY_MVS."""

    def test_refreshes_all_mvs(self):
        from common.core.mv_refresh import all_mvs
        from common.services.job_state import _run_refresh_all_mvs

        with patch(
            "common.core.mv_refresh.refresh_materialized_views",
            return_value={"refreshed": [], "failed": [], "missing": []},
        ) as refresh:
            _run_refresh_all_mvs({})
        assert refresh.call_args.args[0] == all_mvs()

    def test_skip_heavy_excludes_heavy_mvs(self):
        from common.core.mv_refresh import HEAVY_MVS
        from common.services.job_state import _run_refresh_all_mvs

        with patch(
            "common.core.mv_refresh.refresh_materialized_views",
            return_value={"refreshed": [], "failed": [], "missing": []},
        ) as refresh:
            _run_refresh_all_mvs({"skip_heavy": True})
        assert not (HEAVY_MVS & set(refresh.call_args.args[0]))


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

    def test_allocates_run_id_and_threads_generation_purpose(self):
        fixed = "00000000-0000-0000-0000-000000000099"
        from common.services.job_state import _run_generate_production_forecast

        with (
            patch("common.services.job_state.uuid.uuid4", return_value=fixed),
            patch("common.services.job_state._run_subprocess", return_value="ok") as m_sub,
        ):
            result = _run_generate_production_forecast(
                {"generation_purpose": "snapshot_contender"}
            )

        cmd = m_sub.call_args[0][0]
        assert cmd[cmd.index("--run-id") + 1] == fixed
        assert cmd[cmd.index("--generation-purpose") + 1] == "snapshot_contender"
        assert result["run_id"] == fixed

    def test_uses_configured_production_generation_timeout(self):
        from common.services.job_state import _run_generate_production_forecast

        with (
            patch(
                "common.services.job_state._subprocess_timeout_seconds",
                return_value=28_800,
            ) as timeout,
            patch("common.services.job_state._run_subprocess", return_value="ok") as run,
        ):
            _run_generate_production_forecast({})

        timeout.assert_called_once_with("production_generation")
        assert run.call_args.kwargs["timeout_seconds"] == 28_800

    def test_isolates_mixed_lightgbm_and_pytorch_openmp_threads(self):
        from common.services.job_state import _run_generate_production_forecast

        with patch(
            "common.services.job_state._run_subprocess", return_value="ok"
        ) as run:
            _run_generate_production_forecast({})

        assert run.call_args.kwargs["env_overrides"] == {"OMP_NUM_THREADS": "1"}


class TestTrainProductionModelCmd:
    """Production training must always choose a valid CLI mode."""

    def test_empty_params_train_all_forecastable_tree_models(self):
        from common.services.job_state import _run_train_production_model

        with patch(
            "common.services.job_state._run_subprocess", return_value="ok"
        ) as m_sub:
            result = _run_train_production_model({})

        cmd = m_sub.call_args[0][0]
        assert "--all" in cmd
        assert "--model" not in cmd
        assert result["all_models"] is True

    def test_explicit_model_uses_single_model_mode(self):
        from common.services.job_state import _run_train_production_model

        with patch(
            "common.services.job_state._run_subprocess", return_value="ok"
        ) as m_sub:
            result = _run_train_production_model({"model_id": "lgbm_cluster"})

        cmd = m_sub.call_args[0][0]
        assert cmd[cmd.index("--model") + 1] == "lgbm_cluster"
        assert "--all" not in cmd
        assert result["all_models"] is False

    def test_uses_configured_production_training_timeout(self):
        from common.services.job_state import _run_train_production_model

        with (
            patch(
                "common.services.job_state._subprocess_timeout_seconds",
                return_value=28_800,
            ) as timeout,
            patch("common.services.job_state._run_subprocess", return_value="ok") as run,
        ):
            _run_train_production_model({})

        timeout.assert_called_once_with("production_training")
        assert run.call_args.kwargs["timeout_seconds"] == 28_800


def test_completed_backtest_run_cannot_be_reopened() -> None:
    from common.services.job_state import _mark_backtest_run_running

    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = None
    with patch("common.services.job_state._get_conn", return_value=conn):
        with pytest.raises(RuntimeError, match="not eligible to start"):
            _mark_backtest_run_running(72)

    sql = conn.execute.call_args.args[0]
    assert "status IN ('queued', 'failed', 'running')" in sql


def test_snapshot_contender_preparation_uses_configured_timeout():
    from common.services.job_state import _run_prepare_forecast_snapshot_contenders

    with (
        patch(
            "common.services.job_state._subprocess_timeout_seconds",
            return_value=28_800,
        ) as timeout,
        patch("common.services.job_state._run_subprocess", return_value="ok") as run,
    ):
        _run_prepare_forecast_snapshot_contenders({})

    timeout.assert_called_once_with("snapshot_contenders")
    assert run.call_args.kwargs["timeout_seconds"] == 28_800


def test_stale_cluster_tuning_uses_configured_timeout():
    from common.services.job_state import _run_tune_stale_clusters

    with (
        patch(
            "common.services.job_state._subprocess_timeout_seconds",
            return_value=28_800,
        ) as timeout,
        patch("common.services.job_state._run_subprocess", return_value="ok") as run,
        patch("common.core.utils.reset_config") as reset_config,
    ):
        _run_tune_stale_clusters({})

    timeout.assert_called_once_with("stale_cluster_tuning")
    assert run.call_args.kwargs["timeout_seconds"] == 28_800
    reset_config.assert_called_once_with("cluster_tuning_profiles.yaml")


def test_failed_stale_cluster_tuning_keeps_cached_last_good_config():
    from common.services.job_state import _run_tune_stale_clusters

    with (
        patch(
            "common.services.job_state._run_subprocess",
            side_effect=RuntimeError("tuning failed"),
        ),
        patch("common.core.utils.reset_config") as reset_config,
        pytest.raises(RuntimeError, match="tuning failed"),
    ):
        _run_tune_stale_clusters({})

    reset_config.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: auto-load after a successful backtest (no manual Load needed)
# ---------------------------------------------------------------------------

_MOD = "common.services.job_state"


class TestAutoLoadBacktest:
    """_auto_load_backtest makes database loading part of successful completion."""

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

    def test_fails_when_no_predictions(self):
        from common.services.job_state import _auto_load_backtest
        with (
            patch("pathlib.Path.exists", return_value=False),
            patch(f"{_MOD}._run_load_backtest_model") as m_load,
        ):
            with pytest.raises(RuntimeError, match="predictions file is missing"):
                _auto_load_backtest("unknown_model", 3)
        m_load.assert_not_called()

    def test_propagates_loader_errors(self):
        from common.services.job_state import _auto_load_backtest
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(f"{_MOD}._run_load_backtest_model", side_effect=RuntimeError("boom")),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                _auto_load_backtest("lgbm", 9)

    def test_non_tree_model_uses_identity_dir(self):
        from common.services.job_state import _auto_load_backtest
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch(f"{_MOD}._run_load_backtest_model") as m_load,
        ):
            _auto_load_backtest("chronos2_enriched", 1)
        assert m_load.call_args[0][0] == {"model_id": "chronos2_enriched", "run_id": 1}


class TestRunBacktestAutoLoadsBeforeCompletion:
    """_run_backtest auto-loads BEFORE marking the run completed, so the UI sees
    is_loaded_to_db set by the time status flips to 'completed'."""

    def test_auto_load_runs_before_completion_update(self):
        from common.services.job_state import _run_backtest
        manager = MagicMock()
        with (
            patch(f"{_MOD}._run_subprocess", return_value="ok"),
            patch(f"{_MOD}._get_conn"),
            patch(f"{_MOD}.record_backtest_artifact_identity"),
            patch(f"{_MOD}.verify_backtest_artifact_identity"),
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

    def test_pipeline_backtest_creates_tracking_run_and_loads_results(self):
        """Jobs submitted by named pipelines must be tracked and auto-loaded."""
        from common.services.job_state import _run_backtest

        with (
            patch(f"{_MOD}._run_subprocess", return_value="ok"),
            patch(f"{_MOD}._reserve_backtest_run", return_value=71) as reserve,
            patch(f"{_MOD}._get_conn"),
            patch(f"{_MOD}.record_backtest_artifact_identity"),
            patch(f"{_MOD}.verify_backtest_artifact_identity"),
            patch(f"{_MOD}._auto_load_backtest") as auto_load,
            patch(f"{_MOD}._update_backtest_run_on_completion") as update_run,
        ):
            result = _run_backtest("mstl", {}, job_id="job_pipeline_1")

        reserve.assert_called_once_with("mstl", "job_pipeline_1")
        auto_load.assert_called_once_with(
            "mstl", 71, None, None, "job_pipeline_1"
        )
        update_run.assert_called_once_with(71, "mstl")
        assert result["backtest_run_id"] == 71

    def test_unknown_model_is_rejected_before_creating_tracking_run(self):
        from common.services.job_state import _run_backtest

        with patch(f"{_MOD}._reserve_backtest_run") as reserve:
            with pytest.raises(ValueError, match="Unknown backtest model"):
                _run_backtest("retired_model", {}, job_id="job_bad")

        reserve.assert_not_called()

    def test_auto_load_failure_marks_backtest_failed(self):
        from common.services.job_state import _run_backtest

        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = False
        with (
            patch(f"{_MOD}._run_subprocess", return_value="ok"),
            patch(f"{_MOD}._get_conn", return_value=conn),
            patch(f"{_MOD}.record_backtest_artifact_identity"),
            patch(f"{_MOD}.verify_backtest_artifact_identity"),
            patch(f"{_MOD}._auto_load_backtest", side_effect=RuntimeError("load failed")),
            patch(f"{_MOD}._update_backtest_run_on_completion") as m_update,
        ):
            with pytest.raises(RuntimeError, match="load failed"):
                _run_backtest("lgbm", {"backtest_run_id": 5})

        m_update.assert_not_called()
        assert any("status = 'failed'" in call.args[0] for call in conn.execute.call_args_list)

    def test_mstl_installs_statistical_extra_and_removes_stale_artifacts(self):
        from common.services.job_state import _run_backtest

        with (
            patch(f"{_MOD}._run_subprocess", return_value="ok") as run,
            patch(f"{_MOD}._reserve_backtest_run", return_value=72),
            patch(f"{_MOD}._get_conn"),
            patch(f"{_MOD}.record_backtest_artifact_identity"),
            patch(f"{_MOD}.verify_backtest_artifact_identity"),
            patch(f"{_MOD}._auto_load_backtest"),
            patch(f"{_MOD}._update_backtest_run_on_completion"),
            patch("pathlib.Path.unlink") as unlink,
        ):
            _run_backtest("mstl", {})

        assert run.call_args.args[0][1:4] == ["run", "--extra", "statistical"]
        assert unlink.call_count == 3

    def test_governed_backtest_stamps_one_stable_source_lineage(self):
        from common.services.job_state import _run_backtest

        lineage = {
            "source_sales_batch_id": 301,
            "data_checksum": "a" * 64,
            "cluster_experiment_id": 35,
            "cluster_assignment_count": 13_968,
            "cluster_assignment_checksum": "b" * 64,
        }
        with (
            patch(f"{_MOD}._load_governed_backtest_lineage", side_effect=[lineage, lineage]),
            patch(f"{_MOD}._run_subprocess", return_value="ok"),
            patch(f"{_MOD}._get_conn"),
            patch(f"{_MOD}.record_backtest_artifact_identity") as record,
            patch(f"{_MOD}.verify_backtest_artifact_identity"),
            patch(f"{_MOD}._auto_load_backtest"),
            patch(f"{_MOD}._update_backtest_run_on_completion"),
        ):
            _run_backtest("lgbm", {"backtest_run_id": 5, "governed": True})

        assert record.call_args.kwargs["governed_lineage"] == lineage

    def test_governed_lineage_uses_transaction_for_server_cursor(self):
        from common.services.job_state import _load_governed_backtest_lineage

        conn = _make_mock_conn()
        with (
            patch(f"{_MOD}._get_conn", return_value=conn),
            patch(
                "common.services.sales_lineage.load_completed_sales_lineage",
                return_value=SimpleNamespace(batch_id=301, source_hash="a" * 64),
            ),
            patch(
                "common.services.cluster_lineage.load_promoted_cluster_population",
                return_value=SimpleNamespace(
                    experiment_id=35,
                    assignment_count=13_968,
                    assignment_checksum="b" * 64,
                ),
            ),
        ):
            lineage = _load_governed_backtest_lineage()

        conn.transaction.assert_called_once_with()
        assert lineage["source_sales_batch_id"] == 301
        assert lineage["cluster_assignment_count"] == 13_968

    def test_governed_backtest_rejects_lineage_change_before_auto_load(self):
        from common.services.job_state import _run_backtest

        before = {
            "source_sales_batch_id": 301,
            "data_checksum": "a" * 64,
            "cluster_experiment_id": 35,
            "cluster_assignment_count": 13_968,
            "cluster_assignment_checksum": "b" * 64,
        }
        after = {**before, "source_sales_batch_id": 302}
        with (
            patch(f"{_MOD}._load_governed_backtest_lineage", side_effect=[before, after]),
            patch(f"{_MOD}._run_subprocess", return_value="ok"),
            patch(f"{_MOD}._get_conn"),
            patch(f"{_MOD}._auto_load_backtest") as auto_load,
            patch(f"{_MOD}._mark_backtest_run_failed") as mark_failed,
        ):
            with pytest.raises(RuntimeError, match="lineage changed"):
                _run_backtest("lgbm", {"backtest_run_id": 5, "governed": True})

        auto_load.assert_not_called()
        mark_failed.assert_called_once_with(5)
