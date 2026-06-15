"""Unit tests for common/job_registry.py — JobManager class and registry.

Tests the singleton pattern, job submission, group concurrency, queueing,
pipeline support, cancellation, and recovery logic. All DB operations are
mocked so no running database is needed.
"""
from __future__ import annotations

import json
import signal
import threading
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from common.services.job_state import JobTypeDef, _row_to_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_callable(params: dict, progress_cb) -> dict:
    """Dummy job callable that returns success."""
    progress_cb(pct=50, msg="Halfway")
    return {"status": "done"}


def _failing_callable(params: dict, progress_cb) -> dict:
    """Job callable that always raises."""
    raise RuntimeError("Simulated failure")


def _make_type_def(
    type_id: str = "test_job",
    group: str = "test_group",
    callable_fn=None,
) -> JobTypeDef:
    return JobTypeDef(
        type_id=type_id,
        label=f"Test {type_id}",
        description="A test job",
        group=group,
        callable=callable_fn or _dummy_callable,
        params_schema={},
    )


# We need to reset the singleton between tests, so we use a fresh class each time
@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the JobManager singleton between tests."""
    from common.services.job_registry import JobManager
    JobManager._instance = None
    yield
    JobManager._instance = None


@pytest.fixture
def mock_db():
    """Mock all DB operations on JobManager."""
    with patch("common.services.job_registry._get_conn") as mock_conn_fn:
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_result.fetchone.return_value = None
        mock_result.fetchall.return_value = []
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn_fn.return_value = mock_conn
        yield mock_conn


@pytest.fixture
def mock_scheduler():
    """Mock APScheduler and prevent real scheduler from starting."""
    with patch("common.services.job_registry.make_scheduler") as mock_make:
        scheduler = MagicMock()
        scheduler.get_job.return_value = None
        mock_make.return_value = scheduler
        yield scheduler


# ---------------------------------------------------------------------------
# Tests: Registry
# ---------------------------------------------------------------------------

class TestJobTypeRegistry:
    """Tests for the JOB_TYPE_REGISTRY constant."""

    def test_registry_contains_expected_types(self):
        from common.services.job_registry import JOB_TYPE_REGISTRY
        expected = {
            "cluster_scenario", "cluster_pipeline", "seasonality_pipeline",
            "backtest_lgbm", "backtest_catboost", "backtest_xgboost",
            "champion_select", "generate_ai_insights",
            "generate_production_forecast", "compute_replenishment_plan",
            "generate_storyboard", "compute_safety_stock", "compute_eoq",
            "assign_policies", "generate_exceptions", "classify_abc_xyz",
            "compute_variability", "compute_demand_signals",
            "compute_investment", "refresh_health_scores",
            "refresh_intramonth", "run_ss_simulation",
            "data_quality", "tuning_backtest",
        }
        assert expected.issubset(set(JOB_TYPE_REGISTRY.keys()))

    def test_load_domain_job_registered(self):
        # US17c: per-domain integration loads run as a JobManager job in the
        # 'etl' group (one ingestion run at a time, shared with etl_pipeline).
        from common.services.job_registry import JOB_TYPE_REGISTRY
        assert "load_domain" in JOB_TYPE_REGISTRY
        td = JOB_TYPE_REGISTRY["load_domain"]
        assert td.group == "etl"
        assert callable(td.callable)

    def test_all_registry_entries_are_job_type_def(self):
        from common.services.job_registry import JOB_TYPE_REGISTRY
        for key, val in JOB_TYPE_REGISTRY.items():
            assert isinstance(val, JobTypeDef), f"{key} is not a JobTypeDef"
            assert val.type_id == key
            assert val.group
            assert val.label
            assert callable(val.callable)

    def test_groups_are_valid_strings(self):
        from common.services.job_registry import JOB_TYPE_REGISTRY
        for key, val in JOB_TYPE_REGISTRY.items():
            assert isinstance(val.group, str)
            assert len(val.group) > 0


# ---------------------------------------------------------------------------
# Tests: JobManager singleton
# ---------------------------------------------------------------------------

class TestJobManagerSingleton:
    """Tests for singleton pattern."""

    def test_singleton_returns_same_instance(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        a = JobManager()
        b = JobManager()
        assert a is b

    def test_singleton_reset_creates_new_instance(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        a = JobManager()
        a._ensure_init()
        JobManager._instance = None
        b = JobManager()
        assert a is not b


# ---------------------------------------------------------------------------
# Tests: _row_to_dict helper
# ---------------------------------------------------------------------------

class TestRowToDict:
    """Tests for the _row_to_dict helper from job_state."""

    def test_basic_mapping(self):
        cols = ("a", "b", "c")
        row = (1, "two", 3.0)
        result = _row_to_dict(cols, row)
        assert result == {"a": 1, "b": "two", "c": 3.0}

    def test_datetime_conversion(self):
        cols = ("submitted_at", "started_at")
        dt = datetime(2026, 1, 1, 12, 0, 0)
        row = (dt, None)
        result = _row_to_dict(cols, row)
        assert result["submitted_at"] == dt.isoformat()
        assert result["started_at"] is None

    def test_params_dict_passthrough(self):
        """JSON params should be parsed if string, or passed through if dict."""
        cols = ("params", "result")
        row = ('{"key": "val"}', None)
        result = _row_to_dict(cols, row)
        assert result["params"] == {"key": "val"}
        assert result["result"] is None

    def test_params_already_dict(self):
        cols = ("params",)
        row = ({"key": "val"},)
        result = _row_to_dict(cols, row)
        assert result["params"] == {"key": "val"}

    def test_logs_string_parsed(self):
        cols = ("logs",)
        row = ('[{"ts": "12:00:00", "pct": 50, "msg": "test"}]',)
        result = _row_to_dict(cols, row)
        assert result["logs"] == [{"ts": "12:00:00", "pct": 50, "msg": "test"}]

    def test_logs_list_passthrough(self):
        cols = ("logs",)
        row = ([{"ts": "12:00:00", "pct": 50, "msg": "test"}],)
        result = _row_to_dict(cols, row)
        assert result["logs"] == [{"ts": "12:00:00", "pct": 50, "msg": "test"}]

    def test_logs_null_returns_empty_list(self):
        cols = ("logs",)
        row = (None,)
        result = _row_to_dict(cols, row)
        assert result["logs"] == []


# ---------------------------------------------------------------------------
# Tests: submit_job
# ---------------------------------------------------------------------------

class TestSubmitJob:
    """Tests for job submission."""

    def test_submit_unknown_type_raises(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        with pytest.raises(ValueError, match="Unknown job type"):
            mgr.submit_job("nonexistent_type")

    def test_submit_valid_type_returns_job_id(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager, JOB_TYPE_REGISTRY
        mgr = JobManager()
        # Use a real registered type
        job_id = mgr.submit_job("cluster_pipeline", params={"k_range": [3, 8]})
        assert job_id.startswith("job_")
        # Verify scheduler was called
        mock_scheduler.add_job.assert_called_once()

    def test_submit_generates_unique_ids(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        id1 = mgr.submit_job("cluster_pipeline")
        id2 = mgr.submit_job("seasonality_pipeline")
        assert id1 != id2

    def test_submit_queues_when_group_busy(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        mgr._ensure_init()

        # First job becomes active
        id1 = mgr.submit_job("backtest_lgbm")
        assert id1 in mgr._active_jobs

        # Second job in same group should be queued
        id2 = mgr.submit_job("backtest_catboost")
        assert id2 not in mgr._active_jobs
        assert len(mgr._pending_queues.get("backtest", [])) == 1

    def test_submit_different_groups_not_blocked(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        mgr._ensure_init()

        id1 = mgr.submit_job("backtest_lgbm")
        id2 = mgr.submit_job("cluster_pipeline")
        # Both should be active since different groups
        assert id1 in mgr._active_jobs
        assert id2 in mgr._active_jobs

    def test_submit_with_custom_label(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        job_id = mgr.submit_job("cluster_pipeline", label="Custom Label")
        assert job_id.startswith("job_")


# ---------------------------------------------------------------------------
# Tests: cancel_job
# ---------------------------------------------------------------------------

class TestCancelJob:
    """Tests for job cancellation."""

    def test_cancel_nonexistent_job(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        mgr._ensure_init()
        # DB returns None for unknown job
        assert mgr.cancel_job("nonexistent") is False

    def test_cancel_running_job(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        mgr._ensure_init()

        # Mock DB to return a running job (14 cols including pid)
        mock_db.execute.return_value.fetchone.return_value = (
            "job_123", "cluster_pipeline", "Test", "running", "{}", None,
            None, None, None, None, 0, "", "[]", None
        )
        with patch("common.services.job_registry.get_job_pid", return_value=None):
            result = mgr.cancel_job("job_123")
        assert result is True

    def test_cancel_running_job_kills_pid(self, mock_db, mock_scheduler):
        """Cancel should send SIGTERM to the process group by PID."""
        from common.services.job_registry import JobManager
        mgr = JobManager()
        mgr._ensure_init()

        mock_db.execute.return_value.fetchone.return_value = (
            "job_123", "cluster_pipeline", "Test", "running", "{}", None,
            None, None, None, None, 0, "", "[]", 5555
        )
        with patch("common.services.job_registry.get_job_pid", return_value=5555), \
             patch("common.services.job_registry.os.killpg") as m_killpg, \
             patch("common.services.job_registry.os.getpgid", return_value=5555):
            mgr.cancel_job("job_123")
        m_killpg.assert_called_once_with(5555, signal.SIGTERM)

    def test_cancel_completed_job_fails(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        mgr._ensure_init()

        # Mock DB to return a completed job (14 cols including pid)
        mock_db.execute.return_value.fetchone.return_value = (
            "job_123", "cluster_pipeline", "Test", "completed", "{}", None,
            None, None, None, None, 100, "Done", "[]", None
        )
        result = mgr.cancel_job("job_123")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: get_types
# ---------------------------------------------------------------------------

class TestGetTypes:
    """Tests for listing registered job types."""

    def test_get_types_returns_all(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager, JOB_TYPE_REGISTRY
        mgr = JobManager()
        types = mgr.get_types()
        assert len(types) == len(JOB_TYPE_REGISTRY)
        for t in types:
            assert "type_id" in t
            assert "label" in t
            assert "description" in t
            assert "group" in t

    def test_get_types_structure(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        types = mgr.get_types()
        first = types[0]
        assert isinstance(first["type_id"], str)
        assert isinstance(first["label"], str)
        assert isinstance(first["group"], str)


# ---------------------------------------------------------------------------
# Tests: pipeline
# ---------------------------------------------------------------------------

class TestSubmitPipeline:
    """Tests for pipeline (chained job) submission."""

    def test_pipeline_empty_raises(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        with pytest.raises(ValueError, match="at least one step"):
            mgr.submit_pipeline(steps=[])

    def test_pipeline_single_step(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        pipe_id = mgr.submit_pipeline(
            steps=[{"job_type": "cluster_pipeline", "params": {}}],
            label="Test Pipeline",
        )
        assert pipe_id.startswith("pipe_")

    def test_pipeline_multi_step(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        pipe_id = mgr.submit_pipeline(
            steps=[
                {"job_type": "cluster_pipeline", "params": {}},
                {"job_type": "seasonality_pipeline", "params": {}},
            ],
            label="Multi-step",
        )
        assert pipe_id.startswith("pipe_")

    def test_pipeline_unknown_type_raises(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        with pytest.raises((ValueError, KeyError)):
            mgr.submit_pipeline(
                steps=[{"job_type": "nonexistent", "params": {}}],
            )


# ---------------------------------------------------------------------------
# Tests: schedule_recurring
# ---------------------------------------------------------------------------

class TestScheduleRecurring:
    """Tests for recurring schedule creation."""

    def test_schedule_cron(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        with patch("common.services.job_registry.make_trigger") as mock_trigger:
            mock_trigger.return_value = MagicMock()
            mgr = JobManager()
            sched_id = mgr.schedule_recurring(
                "cluster_pipeline", cron="0 2 * * *"
            )
        assert sched_id.startswith("sched_")

    def test_schedule_interval(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        with patch("common.services.job_registry.make_trigger") as mock_trigger:
            mock_trigger.return_value = MagicMock()
            mgr = JobManager()
            sched_id = mgr.schedule_recurring(
                "seasonality_pipeline", interval_minutes=360
            )
        assert sched_id.startswith("sched_")

    def test_schedule_no_trigger_raises(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        with pytest.raises(ValueError, match="Must specify"):
            mgr.schedule_recurring("cluster_pipeline")

    def test_schedule_unknown_type_raises(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        with pytest.raises(ValueError, match="Unknown job type"):
            mgr.schedule_recurring("nonexistent", cron="* * * * *")


# ---------------------------------------------------------------------------
# Tests: remove_schedule
# ---------------------------------------------------------------------------

class TestRemoveSchedule:
    """Tests for schedule removal."""

    def test_remove_existing_schedule(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mock_db.execute.return_value.rowcount = 1
        mgr = JobManager()
        result = mgr.remove_schedule("sched_abc12345")
        assert result is True

    def test_remove_nonexistent_schedule(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mock_db.execute.return_value.rowcount = 0
        mgr = JobManager()
        result = mgr.remove_schedule("sched_nonexist")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: _execute_job
# ---------------------------------------------------------------------------

class TestExecuteJob:
    """Tests for the internal job execution method."""

    def test_successful_execution(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        mgr._ensure_init()

        type_def = _make_type_def(callable_fn=_dummy_callable)

        # Pre-register the job as active
        mgr._active_jobs["test_job_1"] = "test_group"
        mgr._cancel_flags["test_job_1"] = threading.Event()

        mgr._execute_job("test_job_1", type_def, {"key": "val"})

        # Job should be removed from active after execution
        assert "test_job_1" not in mgr._active_jobs

    def test_failed_execution_no_retry(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        mgr._ensure_init()

        type_def = _make_type_def(callable_fn=_failing_callable)

        mgr._active_jobs["test_job_2"] = "test_group"
        mgr._cancel_flags["test_job_2"] = threading.Event()

        # Should not raise — failure is caught internally
        mgr._execute_job("test_job_2", type_def, {}, max_retries=0)
        assert "test_job_2" not in mgr._active_jobs

    def test_dispatch_next_after_completion(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        mgr._ensure_init()

        type_def = _make_type_def(callable_fn=_dummy_callable, group="grp1")

        # Queue a pending job
        pending_type_def = _make_type_def(type_id="pending_job", group="grp1")
        mgr._pending_queues["grp1"] = [
            ("pending_id", pending_type_def, {}, 0, None)
        ]

        # Register current job as active
        mgr._active_jobs["active_id"] = "grp1"
        mgr._cancel_flags["active_id"] = threading.Event()

        mgr._execute_job("active_id", type_def, {})

        # After active job completes, the pending job should be dispatched
        assert "pending_id" in mgr._active_jobs


# ---------------------------------------------------------------------------
# Tests: recover_stale_jobs
# ---------------------------------------------------------------------------

class TestRecoverStaleJobs:
    """Tests for recovery on startup."""

    def test_recover_marks_running_as_failed(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mock_db.execute.return_value.rowcount = 2
        mock_db.execute.return_value.fetchall.return_value = []
        mgr = JobManager()
        mgr._ensure_init()
        # recover_stale_jobs is called during _ensure_init, so just check it ran
        # The mock_db.execute should have been called with UPDATE ... WHERE status = 'running'
        assert mock_db.execute.called


# ---------------------------------------------------------------------------
# Tests: _generate_id
# ---------------------------------------------------------------------------

class TestGenerateId:
    """Tests for ID generation."""

    def test_id_format(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        job_id = mgr._generate_id()
        assert job_id.startswith("job_")
        parts = job_id.split("_")
        assert len(parts) >= 3  # job_YYYYMMDD_HHMMSS_hex

    def test_ids_are_unique(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        ids = {mgr._generate_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# Tests: start_job_in_background (backward compat)
# ---------------------------------------------------------------------------

class TestStartJobInBackground:
    """Tests for the backward-compatible start method."""

    def test_noop_does_not_raise(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        # Should not raise
        mgr.start_job_in_background("any_job_id")


# ---------------------------------------------------------------------------
# Tests: list_schedules
# ---------------------------------------------------------------------------

class TestListSchedules:
    """Tests for schedule listing."""

    def test_list_schedules_empty(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mock_db.execute.return_value.fetchall.return_value = []
        mgr = JobManager()
        schedules = mgr.list_schedules()
        assert schedules == []

    def test_list_schedules_with_data(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        now = datetime(2026, 3, 1, 12, 0, 0)
        mock_db.execute.return_value.fetchall.return_value = [
            ("sched_1", "cluster_pipeline", "Clustering", "0 2 * * *", None,
             '{}', True, now, None, now, 5),
        ]
        mgr = JobManager()
        schedules = mgr.list_schedules()
        assert len(schedules) == 1
        assert schedules[0]["schedule_id"] == "sched_1"
        assert schedules[0]["job_type"] == "cluster_pipeline"
        assert schedules[0]["run_count"] == 5

    def test_list_schedules_handles_missing_table(self, mock_db, mock_scheduler):
        """If job_schedule table doesn't exist after init, should return empty list."""
        from common.services.job_registry import JobManager
        mgr = JobManager()
        mgr._ensure_init()
        # Now set the side_effect after init has completed
        mock_db.execute.side_effect = Exception("relation does not exist")
        schedules = mgr.list_schedules()
        assert schedules == []


# ---------------------------------------------------------------------------
# Tests: _is_group_busy
# ---------------------------------------------------------------------------

class TestIsGroupBusy:
    """Tests for group concurrency check."""

    def test_empty_group_not_busy(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        mgr._ensure_init()
        assert mgr._is_group_busy("test") is False

    def test_group_with_active_job_is_busy(self, mock_db, mock_scheduler):
        from common.services.job_registry import JobManager
        mgr = JobManager()
        mgr._ensure_init()
        mgr._active_jobs["job_1"] = "backtest"
        assert mgr._is_group_busy("backtest") is True
        assert mgr._is_group_busy("clustering") is False


# ---------------------------------------------------------------------------
# Tests: _run_data_quality callable
# ---------------------------------------------------------------------------

class TestRunDataQuality:
    """Tests for the _run_data_quality job callable."""

    def test_run_data_quality_calls_engine(self):
        from common.services.job_state import _run_data_quality
        mock_results = [
            {"check_name": "c1", "status": "pass"},
            {"check_name": "c2", "status": "fail"},
            {"check_name": "c3", "status": "pass"},
        ]
        with patch("common.engines.dq_engine.DQEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.run_all_checks.return_value = mock_results
            progress = MagicMock()

            result = _run_data_quality({}, progress)

        instance.run_all_checks.assert_called_once_with(domain=None)
        assert result["total_checks"] == 3
        assert result["passed"] == 2
        assert result["failed"] == 1
        progress.assert_any_call(pct=10, msg="Running data quality checks")
        progress.assert_any_call(pct=100, msg="Data quality checks complete")

    def test_run_data_quality_with_domain_filter(self):
        from common.services.job_state import _run_data_quality
        with patch("common.engines.dq_engine.DQEngine") as MockEngine:
            instance = MockEngine.return_value
            instance.run_all_checks.return_value = []
            progress = MagicMock()

            result = _run_data_quality({"domain": "sales"}, progress)

        instance.run_all_checks.assert_called_once_with(domain="sales")
        assert result["total_checks"] == 0
        assert result["passed"] == 0
        assert result["failed"] == 0

    def test_data_quality_in_platform_group(self):
        from common.services.job_registry import JOB_TYPE_REGISTRY
        entry = JOB_TYPE_REGISTRY["data_quality"]
        assert entry.group == "platform"
        assert entry.type_id == "data_quality"
