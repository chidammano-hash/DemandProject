"""Tests for the _run_model_tuning_experiment job callable.

Validates the full lifecycle of a model tuning experiment job:
- Registry presence
- Subprocess invocation with correct arguments
- DB status updates (running, completed, failed)
- Result registration via tuning_tracker
- Lag and cluster breakdown inserts
- Temp config cleanup
- Progress callback invocation
- Cancel event handling
- Log streaming
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from common.core.paths import PROJECT_ROOT as _PROJECT_ROOT
# ===========================================================================
# 1. Job type registration
# ===========================================================================
def test_job_type_registered():
    """model_tuning_run exists in JOB_TYPE_REGISTRY with correct fields."""
    from common.services.job_registry import JOB_TYPE_REGISTRY

    assert "model_tuning_run" in JOB_TYPE_REGISTRY
    type_def = JOB_TYPE_REGISTRY["model_tuning_run"]
    assert type_def.type_id == "model_tuning_run"
    assert type_def.label == "Model Tuning Experiment"
    assert type_def.group == "tuning"
    assert callable(type_def.callable)
    # params_schema must contain run_id, model, config_path, run_label
    for key in ("run_id", "model", "config_path", "run_label"):
        assert key in type_def.params_schema, f"Missing params_schema key: {key}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SAMPLE_METADATA = {
    "accuracy_pct": 70.5,
    "wape": 29.5,
    "bias": 0.02,
    "n_predictions": 10000,
    "n_dfus": 500,
    "timeframes": [
        {"timeframe": "2024-01", "accuracy_pct": 71.0, "wape": 29.0, "bias": 0.01,
         "n_predictions": 2000, "n_dfus": 500},
        {"timeframe": "2024-02", "accuracy_pct": 70.0, "wape": 30.0, "bias": 0.03,
         "n_predictions": 2000, "n_dfus": 500},
    ],
}

_SAMPLE_PARAMS = {
    "run_id": 42,
    "model": "lgbm",
    "config_path": "/tmp/tuning_test/forecast_pipeline_config.yaml",
    "run_label": "test_experiment",
}


def _make_mock_conn():
    """Build a mock psycopg connection with cursor context-manager support."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_cursor.fetchall.return_value = []

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    return mock_conn, mock_cursor


# ===========================================================================
# 2. Job creates temp config
# ===========================================================================
@patch("common.services.job_state._cleanup_temp_config")
@patch("common.ml.tuning_tracker.register_cluster_month_breakdowns")
@patch("common.ml.tuning_tracker.register_timeframes")
@patch("common.ml.tuning_tracker.complete_run")
@patch("common.services.job_state._run_subprocess", return_value="backtest output")
@patch("common.services.job_state._get_conn")
def test_job_creates_temp_config(
    mock_get_conn, mock_subprocess, mock_complete, mock_timeframes,
    mock_cluster, mock_cleanup,
):
    """Verify the job passes the config_path from params to subprocess."""
    from common.services.job_state import _run_model_tuning_experiment

    mock_conn, _ = _make_mock_conn()
    mock_get_conn.return_value = mock_conn

    result = _run_model_tuning_experiment(
        params=_SAMPLE_PARAMS.copy(),
        progress_cb=None,
        cancel_event=None,
        job_id="job-001",
    )

    # The subprocess should have received the config_path argument
    call_args = mock_subprocess.call_args
    cmd = call_args[0][0]  # first positional arg is the cmd list
    assert "--config" in cmd
    config_idx = cmd.index("--config")
    assert cmd[config_idx + 1] == _SAMPLE_PARAMS["config_path"]

    assert result["run_id"] == 42
    assert result["model"] == "lgbm"


# ===========================================================================
# 3. Job runs backtest subprocess with correct command args
# ===========================================================================
@patch("common.services.job_state._cleanup_temp_config")
@patch("common.ml.tuning_tracker.register_cluster_month_breakdowns")
@patch("common.ml.tuning_tracker.register_timeframes")
@patch("common.ml.tuning_tracker.complete_run")
@patch("common.services.job_state._run_subprocess", return_value="ok")
@patch("common.services.job_state._get_conn")
def test_job_runs_backtest_subprocess(
    mock_get_conn, mock_subprocess, mock_complete, mock_timeframes,
    mock_cluster, mock_cleanup,
):
    """Subprocess invoked with 'uv run python scripts/run_backtest.py --model lgbm --config <path>'."""
    from common.services.job_state import _run_model_tuning_experiment

    mock_conn, _ = _make_mock_conn()
    mock_get_conn.return_value = mock_conn

    _run_model_tuning_experiment(
        params=_SAMPLE_PARAMS.copy(),
        progress_cb=MagicMock(),
        cancel_event=None,
        job_id="job-002",
    )

    mock_subprocess.assert_called_once()
    cmd = mock_subprocess.call_args[0][0]
    assert "uv" == cmd[0]
    assert "run" == cmd[1]
    assert "python" == cmd[2]
    assert cmd[3].endswith("run_backtest.py")
    assert "--model" in cmd
    assert "lgbm" in cmd
    assert "--config" in cmd


# ===========================================================================
# 4. Job registers results on success
# ===========================================================================
@patch("common.services.job_state._cleanup_temp_config")
@patch("common.ml.tuning_tracker.register_cluster_month_breakdowns")
@patch("common.ml.tuning_tracker.register_timeframes")
@patch("common.ml.tuning_tracker.complete_run")
@patch("common.services.job_state._run_subprocess", return_value="backtest done")
@patch("common.services.job_state._get_conn")
def test_job_registers_results_on_success(
    mock_get_conn, mock_subprocess, mock_complete, mock_timeframes,
    mock_cluster, mock_cleanup,
):
    """On success, complete_run and register_timeframes are called with correct run_id."""
    from common.services.job_state import _run_model_tuning_experiment

    mock_conn, _ = _make_mock_conn()
    mock_get_conn.return_value = mock_conn

    _run_model_tuning_experiment(
        params=_SAMPLE_PARAMS.copy(),
        progress_cb=None,
        cancel_event=None,
        job_id="job-003",
    )

    # complete_run called with run_id and the metadata path
    mock_complete.assert_called_once()
    args = mock_complete.call_args[0]
    assert args[0] == 42  # run_id
    assert str(args[1]).endswith("backtest_metadata.json")

    # register_timeframes called with run_id and metadata path
    mock_timeframes.assert_called_once()
    tf_args = mock_timeframes.call_args[0]
    assert tf_args[0] == 42


# ===========================================================================
# 5. Job inserts lag breakdowns via register_timeframes
# ===========================================================================
@patch("common.services.job_state._cleanup_temp_config")
@patch("common.ml.tuning_tracker.register_cluster_month_breakdowns")
@patch("common.ml.tuning_tracker.register_timeframes")
@patch("common.ml.tuning_tracker.complete_run")
@patch("common.services.job_state._run_subprocess", return_value="ok")
@patch("common.services.job_state._get_conn")
def test_job_inserts_lag_breakdowns(
    mock_get_conn, mock_subprocess, mock_complete, mock_timeframes,
    mock_cluster, mock_cleanup,
):
    """register_timeframes is called once to insert per-lag data."""
    from common.services.job_state import _run_model_tuning_experiment

    mock_conn, _ = _make_mock_conn()
    mock_get_conn.return_value = mock_conn

    _run_model_tuning_experiment(
        params=_SAMPLE_PARAMS.copy(),
        progress_cb=None,
        cancel_event=None,
        job_id="job-004",
    )

    mock_timeframes.assert_called_once()
    # First arg is run_id, second is metadata path
    assert mock_timeframes.call_args[0][0] == 42
    meta_path = Path(mock_timeframes.call_args[0][1])
    assert meta_path.name == "backtest_metadata.json"
    assert "lgbm_cluster" in str(meta_path)


# ===========================================================================
# 6. Job inserts cluster breakdowns when predictions file exists
# ===========================================================================
@patch("common.services.job_state._cleanup_temp_config")
@patch("common.ml.tuning_tracker.register_cluster_month_breakdowns")
@patch("common.ml.tuning_tracker.register_timeframes")
@patch("common.ml.tuning_tracker.complete_run")
@patch("common.services.job_state._run_subprocess", return_value="ok")
@patch("common.services.job_state._get_conn")
def test_job_inserts_cluster_breakdowns(
    mock_get_conn, mock_subprocess, mock_complete, mock_timeframes,
    mock_cluster, mock_cleanup,
):
    """register_cluster_month_breakdowns called when predictions CSV exists."""
    from common.services.job_state import _run_model_tuning_experiment

    mock_conn, _ = _make_mock_conn()
    mock_get_conn.return_value = mock_conn

    # Patch Path.exists to return True for the predictions file
    with (
        patch.object(Path, "exists", return_value=True),
        patch("common.ml.tuning_tracker.register_lag_breakdowns"),
    ):
        _run_model_tuning_experiment(
            params=_SAMPLE_PARAMS.copy(),
            progress_cb=None,
            cancel_event=None,
            job_id="job-005",
        )

    mock_cluster.assert_called_once()
    assert mock_cluster.call_args[0][0] == 42
    pred_path = Path(mock_cluster.call_args[0][1])
    assert pred_path.name == "backtest_predictions.csv"
    assert "lgbm_cluster" in str(pred_path)


# ===========================================================================
# 7. Job handles subprocess failure
# ===========================================================================
@patch("common.services.job_state._cleanup_temp_config")
@patch("common.ml.tuning_tracker.fail_run")
@patch("common.services.job_state._run_subprocess", side_effect=RuntimeError("Backtest failed"))
@patch("common.services.job_state._get_conn")
def test_job_handles_subprocess_failure(
    mock_get_conn, mock_subprocess, mock_fail_run, mock_cleanup,
):
    """On subprocess failure, status is set to failed via fail_run."""
    from common.services.job_state import _run_model_tuning_experiment

    mock_conn, _ = _make_mock_conn()
    mock_get_conn.return_value = mock_conn

    with pytest.raises(RuntimeError, match="Backtest failed"):
        _run_model_tuning_experiment(
            params=_SAMPLE_PARAMS.copy(),
            progress_cb=None,
            cancel_event=None,
            job_id="job-006",
        )

    # fail_run should have been called with the run_id and error message
    mock_fail_run.assert_called_once()
    assert mock_fail_run.call_args[0][0] == 42
    assert "Backtest failed" in mock_fail_run.call_args[0][1]


# ===========================================================================
# 8. Job cleans up temp config after completion
# ===========================================================================
@patch("common.services.job_state._cleanup_temp_config")
@patch("common.ml.tuning_tracker.register_cluster_month_breakdowns")
@patch("common.ml.tuning_tracker.register_timeframes")
@patch("common.ml.tuning_tracker.complete_run")
@patch("common.services.job_state._run_subprocess", return_value="ok")
@patch("common.services.job_state._get_conn")
def test_job_cleans_up_temp_config(
    mock_get_conn, mock_subprocess, mock_complete, mock_timeframes,
    mock_cluster, mock_cleanup,
):
    """Temp config file is deleted after successful completion."""
    from common.services.job_state import _run_model_tuning_experiment

    mock_conn, _ = _make_mock_conn()
    mock_get_conn.return_value = mock_conn

    _run_model_tuning_experiment(
        params=_SAMPLE_PARAMS.copy(),
        progress_cb=None,
        cancel_event=None,
        job_id="job-007",
    )

    mock_cleanup.assert_called_once_with(_SAMPLE_PARAMS["config_path"])


# ===========================================================================
# 9. Job updates progress callback
# ===========================================================================
@patch("common.services.job_state._cleanup_temp_config")
@patch("common.ml.tuning_tracker.register_cluster_month_breakdowns")
@patch("common.ml.tuning_tracker.register_timeframes")
@patch("common.ml.tuning_tracker.complete_run")
@patch("common.services.job_state._run_subprocess", return_value="ok")
@patch("common.services.job_state._get_conn")
def test_job_updates_progress(
    mock_get_conn, mock_subprocess, mock_complete, mock_timeframes,
    mock_cluster, mock_cleanup,
):
    """progress_cb is called multiple times with increasing pct and descriptive messages."""
    from common.services.job_state import _run_model_tuning_experiment

    mock_conn, _ = _make_mock_conn()
    mock_get_conn.return_value = mock_conn
    progress_cb = MagicMock()

    _run_model_tuning_experiment(
        params=_SAMPLE_PARAMS.copy(),
        progress_cb=progress_cb,
        cancel_event=None,
        job_id="job-008",
    )

    assert progress_cb.call_count >= 3, "Expected at least 3 progress updates"

    # First call should be pct=0 (starting)
    first_call = progress_cb.call_args_list[0]
    assert first_call.kwargs.get("pct") == 0 or first_call[1].get("pct") == 0

    # Last call should be pct=100 (completed)
    last_call = progress_cb.call_args_list[-1]
    assert last_call.kwargs.get("pct") == 100 or last_call[1].get("pct") == 100


# ===========================================================================
# 10. Job handles missing metadata gracefully
# ===========================================================================
@patch("common.services.job_state._cleanup_temp_config")
@patch("common.ml.tuning_tracker.register_timeframes")
@patch("common.ml.tuning_tracker.complete_run", side_effect=FileNotFoundError("backtest_metadata.json not found"))
@patch("common.services.job_state._run_subprocess", return_value="ok")
@patch("common.services.job_state._get_conn")
def test_job_handles_missing_metadata(
    mock_get_conn, mock_subprocess, mock_complete, mock_timeframes, mock_cleanup,
):
    """If backtest_metadata.json is not found, complete_run raises FileNotFoundError."""
    from common.services.job_state import _run_model_tuning_experiment

    mock_conn, _ = _make_mock_conn()
    mock_get_conn.return_value = mock_conn

    with pytest.raises(FileNotFoundError, match="backtest_metadata.json"):
        _run_model_tuning_experiment(
            params=_SAMPLE_PARAMS.copy(),
            progress_cb=None,
            cancel_event=None,
            job_id="job-009",
        )


# ===========================================================================
# 11. Job cancel event checked — terminates early
# ===========================================================================
@patch("common.services.job_state._cleanup_temp_config")
@patch("common.services.job_state._run_subprocess", return_value="ok")
@patch("common.services.job_state._get_conn")
def test_job_cancel_event_checked(
    mock_get_conn, mock_subprocess, mock_cleanup,
):
    """cancel_event.is_set() terminates the job early with RuntimeError."""
    from common.services.job_state import _run_model_tuning_experiment

    mock_conn, _ = _make_mock_conn()
    mock_get_conn.return_value = mock_conn

    cancel_event = threading.Event()
    cancel_event.set()  # Set immediately — should cancel before subprocess

    with pytest.raises(RuntimeError, match="cancelled"):
        _run_model_tuning_experiment(
            params=_SAMPLE_PARAMS.copy(),
            progress_cb=None,
            cancel_event=cancel_event,
            job_id="job-010",
        )


# ===========================================================================
# 12. Job logs streamed via subprocess
# ===========================================================================
@patch("common.services.job_state._cleanup_temp_config")
@patch("common.ml.tuning_tracker.register_cluster_month_breakdowns")
@patch("common.ml.tuning_tracker.register_timeframes")
@patch("common.ml.tuning_tracker.complete_run")
@patch("common.services.job_state._run_subprocess", return_value="line1\nline2\nline3")
@patch("common.services.job_state._get_conn")
def test_job_logs_streamed(
    mock_get_conn, mock_subprocess, mock_complete, mock_timeframes,
    mock_cluster, mock_cleanup,
):
    """Subprocess stdout is captured and returned in output_log."""
    from common.services.job_state import _run_model_tuning_experiment

    mock_conn, _ = _make_mock_conn()
    mock_get_conn.return_value = mock_conn

    result = _run_model_tuning_experiment(
        params=_SAMPLE_PARAMS.copy(),
        progress_cb=None,
        cancel_event=None,
        job_id="job-011",
    )

    assert "output_log" in result
    assert "line1" in result["output_log"]
    assert "line2" in result["output_log"]


# ===========================================================================
# Bonus: Verify the retained model uses its canonical output directory
# ===========================================================================
@patch("common.services.job_state._cleanup_temp_config")
@patch("common.ml.tuning_tracker.register_cluster_month_breakdowns")
@patch("common.ml.tuning_tracker.register_timeframes")
@patch("common.ml.tuning_tracker.complete_run")
@patch("common.services.job_state._run_subprocess", return_value="ok")
@patch("common.services.job_state._get_conn")
def test_job_model_output_directory(
    mock_get_conn, mock_subprocess, mock_complete, mock_timeframes,
    mock_cluster, mock_cleanup,
):
    """LightGBM uses the canonical cluster output directory."""
    from common.services.job_state import _run_model_tuning_experiment

    mock_conn, _ = _make_mock_conn()
    mock_get_conn.return_value = mock_conn
    model = "lgbm"
    expected_dir = "lgbm_cluster"

    params = {
        "run_id": 99,
        "model": model,
        "config_path": "/tmp/test_config.yaml",
        "run_label": f"test_{model}",
    }

    _run_model_tuning_experiment(
        params=params,
        progress_cb=None,
        cancel_event=None,
        job_id=f"job-{model}",
    )

    # Verify complete_run received the correct metadata path
    meta_path = Path(mock_complete.call_args[0][1])
    assert expected_dir in str(meta_path)
    assert meta_path.name == "backtest_metadata.json"

    # Verify subprocess received the correct model arg
    cmd = mock_subprocess.call_args[0][0]
    model_idx = cmd.index("--model")
    assert cmd[model_idx + 1] == model


# ===========================================================================
# Bonus: Unknown model type raises ValueError
# ===========================================================================
@patch("common.services.job_state._get_conn")
def test_job_unknown_model_raises(mock_get_conn):
    """Unknown model type raises ValueError immediately."""
    from common.services.job_state import _run_model_tuning_experiment

    mock_conn, _ = _make_mock_conn()
    mock_get_conn.return_value = mock_conn

    params = {
        "run_id": 1,
        "model": "invalid_model",
        "config_path": "/tmp/test.yaml",
        "run_label": "bad_model",
    }

    with pytest.raises(ValueError, match="Unknown model type"):
        _run_model_tuning_experiment(
            params=params,
            progress_cb=None,
            cancel_event=None,
            job_id="job-bad",
        )


# ===========================================================================
# Bonus: Cleanup called even on failure
# ===========================================================================
@patch("common.services.job_state._cleanup_temp_config")
@patch("common.ml.tuning_tracker.fail_run")
@patch("common.services.job_state._run_subprocess", side_effect=OSError("Process error"))
@patch("common.services.job_state._get_conn")
def test_job_cleans_up_on_failure(
    mock_get_conn, mock_subprocess, mock_fail_run, mock_cleanup,
):
    """Temp config is cleaned up even when the subprocess fails."""
    from common.services.job_state import _run_model_tuning_experiment

    mock_conn, _ = _make_mock_conn()
    mock_get_conn.return_value = mock_conn

    with pytest.raises(OSError, match="Process error"):
        _run_model_tuning_experiment(
            params=_SAMPLE_PARAMS.copy(),
            progress_cb=None,
            cancel_event=None,
            job_id="job-cleanup-fail",
        )

    mock_cleanup.assert_called_once_with(_SAMPLE_PARAMS["config_path"])
