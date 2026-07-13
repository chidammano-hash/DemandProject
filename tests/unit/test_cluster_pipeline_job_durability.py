"""Durability regressions for the managed clustering pipeline job."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from common.services.job_registry import JobManager
from common.services.job_state import (
    JobTypeDef,
    _run_cluster_pipeline,
    bind_job_attempt,
    reconcile_cluster_pipeline_experiment,
    reset_job_attempt,
)


def _manager() -> JobManager:
    manager = object.__new__(JobManager)
    manager._initialized = True
    manager._scheduler = MagicMock()
    manager._state_lock = threading.Lock()
    manager._active_jobs = {}
    manager._cancel_flags = {}
    manager._pending_queues = {}
    return manager


def _type_def(type_id: str, callable_fn: Callable[..., dict[str, Any]]) -> JobTypeDef:
    return JobTypeDef(
        type_id=type_id,
        label="Cluster durability test",
        description="Cluster durability test",
        group="clustering",
        callable=callable_fn,
    )


def test_cluster_pipeline_is_one_exact_managed_subprocess() -> None:
    """The worker must delegate the whole lifecycle to one recoverable child."""
    cancel_event = threading.Event()
    context = bind_job_attempt("attempt-cluster-17")
    try:
        with (
            patch(
                "common.services.job_state._run_subprocess",
                return_value="pipeline complete",
            ) as run,
            patch(
                "common.services.job_state.verify_cluster_pipeline_completion",
                return_value={
                    "experiment_id": 17,
                    "scenario_id": "sc_20260712_120000_abcd",
                    "status": "completed",
                    "is_promoted": True,
                },
            ) as verify,
        ):
            result = _run_cluster_pipeline(
                {
                    "feature_params": {"time_window_months": 24},
                    "model_params": {"k_range": [9, 12]},
                    "label": "Nightly clustering",
                    "auto_promote": True,
                },
                cancel_event=cancel_event,
                job_id="job-cluster-17",
            )
    finally:
        reset_job_attempt(context)

    command = run.call_args.args[0]
    assert command[:3] == ["uv", "run", "python"]
    assert command[3].endswith("scripts/ml/run_cluster_pipeline.py")
    assert command[command.index("--job-id") + 1] == "job-cluster-17"
    assert command[command.index("--attempt-token") + 1] == "attempt-cluster-17"
    payload = json.loads(command[command.index("--params-json") + 1])
    assert payload == {
        "auto_promote": True,
        "feature_params": {"time_window_months": 24},
        "label": "Nightly clustering",
        "model_params": {"k_range": [9, 12]},
    }
    assert run.call_count == 1
    assert run.call_args.kwargs["cancel_event"] is cancel_event
    assert run.call_args.kwargs["job_id"] == "job-cluster-17"
    verify.assert_called_once_with("job-cluster-17", require_promoted=True)
    assert result["experiment_id"] == 17


def test_cluster_pipeline_requires_exact_managed_attempt_identity() -> None:
    with pytest.raises(RuntimeError, match="attempt token"):
        _run_cluster_pipeline({}, job_id="job-without-bound-attempt")


def test_cluster_cli_params_are_strict_json() -> None:
    from scripts.ml.run_cluster_pipeline import parse_pipeline_params

    assert parse_pipeline_params(
        '{"feature_params":{"time_window_months":24},"auto_promote":false}'
    ) == {
        "feature_params": {"time_window_months": 24},
        "auto_promote": False,
    }
    with pytest.raises(ValueError, match="Unsupported cluster pipeline parameter"):
        parse_pipeline_params('{"unknown":true}')
    with pytest.raises(ValueError, match="model_params must be an object"):
        parse_pipeline_params('{"model_params":[]}')


def test_experiment_and_exact_job_binding_commit_together() -> None:
    from scripts.ml.run_cluster_pipeline import _create_pipeline_experiment

    connection = MagicMock()
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    connection.cursor.return_value = cursor
    cursor.fetchone.return_value = (41,)
    cursor.rowcount = 1

    with patch("psycopg.connect", return_value=connection):
        experiment_id = _create_pipeline_experiment(
            {"host": "db"},
            scenario_id="sc_20260712_120000_abcd",
            label="Managed",
            job_id="job-cluster-41",
            attempt_token="attempt-cluster-41",
            feature_params={"time_window_months": 36},
            model_params={"k_range": [9, 18]},
            label_params={"volume_high": 0.75},
        )

    assert experiment_id == 41
    assert cursor.execute.call_count == 2
    binding_sql, binding_args = cursor.execute.call_args_list[1].args
    assert json.loads(binding_args[0]) == {
        "cluster_experiment_id": 41,
        "cluster_scenario_id": "sc_20260712_120000_abcd",
    }
    assert "attempt_token = %s" in str(binding_sql)
    assert binding_args[-3:] == (
        "job-cluster-41",
        "attempt-cluster-41",
        "running",
    )
    connection.commit.assert_called_once_with()


def test_experiment_creation_rolls_back_when_attempt_binding_is_stale() -> None:
    from scripts.ml.run_cluster_pipeline import _create_pipeline_experiment

    connection = MagicMock()
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    connection.cursor.return_value = cursor
    cursor.fetchone.return_value = (42,)
    cursor.rowcount = 0

    with (
        patch("psycopg.connect", return_value=connection),
        pytest.raises(RuntimeError, match="exact running job attempt"),
    ):
        _create_pipeline_experiment(
            {},
            scenario_id="sc_20260712_120001_abcd",
            label="Stale",
            job_id="job-cluster-42",
            attempt_token="attempt-stale",
            feature_params={},
            model_params={},
            label_params={},
        )

    connection.commit.assert_not_called()


def test_cluster_terminal_reconciliation_targets_current_exact_experiment() -> None:
    connection = MagicMock()
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)
    connection.execute.return_value.rowcount = 1

    with patch("common.services.job_state._get_conn", return_value=connection):
        assert (
            reconcile_cluster_pipeline_experiment(
                "job-cluster-9",
                "cancelled",
            )
            is True
        )

    sql, args = connection.execute.call_args.args
    assert "job.params ->> 'cluster_experiment_id'" in str(sql)
    assert "experiment.job_id = job.job_id" in str(sql)
    assert args == ("cancelled", "job-cluster-9")


def test_callable_completion_marker_precedes_terminal_job_status() -> None:
    """A crash after parent-side postwork must not replay or skip it ambiguously."""
    manager = _manager()
    manager._active_jobs["job-callable"] = "forecast"
    manager._cancel_flags["job-callable"] = threading.Event()
    manager._db_update_status = MagicMock(return_value=True)
    manager._dispatch_next = MagicMock()

    manager._execute_job(
        "job-callable",
        _type_def("forecast-test", MagicMock(return_value={"done": True})),
        {},
    )

    writes = manager._db_update_status.call_args_list
    started = writes[0]
    attempt_token = started.kwargs["attempt_token"]
    marker_index = next(
        index
        for index, call in enumerate(writes)
        if call.kwargs.get("attempt_callable_completion") is not None
    )
    completed_index = next(
        index for index, call in enumerate(writes) if call.args[1] == "completed"
    )
    marker = writes[marker_index].kwargs["attempt_callable_completion"]
    assert marker == {"attempt_token": attempt_token}
    assert marker_index < completed_index


def test_recovery_fails_closed_for_uncovered_parent_postwork() -> None:
    manager = _manager()
    with pytest.raises(RuntimeError, match="no durable recovery finalizer"):
        manager._finalize_recovered_job(
            "job-tuning-chat",
            _type_def("tuning_backtest", MagicMock()),
            {"params": {"run_id": 12, "session_id": "session-12"}},
        )


def test_recovery_trusts_only_exact_callable_completion_marker() -> None:
    """Finished parent postwork may skip replay only for the same attempt token."""
    manager = _manager()
    manager._db_update_status = MagicMock(return_value=True)
    manager._finalize_recovered_job = MagicMock(
        side_effect=AssertionError("durable postwork must not replay")
    )
    job = {
        "params": {
            "__attempt_command_digest": "digest-12",
            "__attempt_callable_completion": {
                "attempt_token": "attempt-12",
            },
        },
        "attempt_token": "attempt-12",
        "attempt_result": {
            "attempt_token": "attempt-12",
            "exit_code": 0,
            "completed_at": "2026-07-12T12:00:00+00:00",
            "command_digest": "digest-12",
        },
        "retry_count": 0,
        "max_retries": 0,
    }

    with patch("common.services.job_registry._clear_pid"):
        released = manager._reconcile_recovered_attempt(
            "job-tuning-chat",
            _type_def("tuning_backtest", MagicMock()),
            job,
            execution_group="tuning",
        )

    assert released is True
    manager._finalize_recovered_job.assert_not_called()
    assert manager._db_update_status.call_args.args[1] == "completed"


def test_cluster_cancel_reconciles_its_exact_experiment() -> None:
    manager = _manager()
    manager._db_get = MagicMock(
        return_value={
            "job_id": "job-cluster-cancel",
            "job_type": "cluster_pipeline",
            "status": "running",
            "pid": 987,
            "attempt_token": "attempt-cancel",
        }
    )
    manager._kill_process = MagicMock(return_value=True)
    manager._db_update_status = MagicMock(return_value=True)

    with patch(
        "common.services.job_registry.reconcile_cluster_pipeline_experiment",
        return_value=True,
    ) as reconcile:
        assert manager.cancel_job("job-cluster-cancel") is True

    reconcile.assert_called_once_with("job-cluster-cancel", "cancelled")


def test_recovered_cluster_completion_rechecks_exact_experiment_lineage() -> None:
    manager = _manager()
    with patch("common.services.job_registry.verify_cluster_pipeline_completion") as verify:
        manager._finalize_recovered_job(
            "job-cluster-recovered",
            _type_def("cluster_pipeline", MagicMock()),
            {"params": {"auto_promote": False}},
        )

    verify.assert_called_once_with(
        "job-cluster-recovered",
        require_promoted=False,
    )


def test_ambiguous_cluster_restart_fails_its_exact_experiment() -> None:
    manager = _manager()
    manager._db_update_status = MagicMock(return_value=True)

    with patch(
        "common.services.job_registry.reconcile_cluster_pipeline_experiment",
        return_value=True,
    ) as reconcile:
        manager._quarantine_recovered_job(
            "job-cluster-ambiguous",
            "Process exited without an exact attempt result",
            attempt_token="attempt-ambiguous",
            type_def=_type_def("cluster_pipeline", MagicMock()),
        )

    reconcile.assert_called_once_with("job-cluster-ambiguous", "failed")
    quarantine = manager._db_update_status.call_args
    assert quarantine.args[1] == "failed"
    assert quarantine.kwargs["recovery_quarantine_reason"].startswith("Process exited")
