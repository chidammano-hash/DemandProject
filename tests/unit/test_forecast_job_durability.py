"""RED regressions for restart-safe forecasting jobs and pipelines."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from common.services.job_registry import JobManager
from common.services.job_state import (
    JobTypeDef,
    _run_backtest,
    _run_cleanup_forecast_staging,
    _run_subprocess,
    capture_process_identity,
    reconcile_backtest_run,
)
from common.services.pipeline_presets import get_pipeline_preset, preset_steps


def _bare_manager() -> JobManager:
    """Build a JobManager without starting APScheduler or touching Postgres."""
    manager = object.__new__(JobManager)
    manager._initialized = True
    manager._scheduler = MagicMock()
    manager._state_lock = threading.Lock()
    manager._active_jobs = {}
    manager._cancel_flags = {}
    manager._pending_queues = {}
    return manager


def test_linux_process_identity_uses_boot_id_and_start_ticks(
    tmp_path,
) -> None:
    proc_root = tmp_path / "proc"
    process_dir = proc_root / "123"
    process_dir.mkdir(parents=True)
    (proc_root / "sys/kernel/random").mkdir(parents=True)
    (proc_root / "sys/kernel/random/boot_id").write_text("boot-123\n")
    # Fields after the parenthesized command start at proc field 3; index 19 is
    # field 22 (starttime).
    (process_dir / "stat").write_text(
        "123 (worker name) S " + " ".join(["0"] * 18 + ["98765"] + ["0"] * 4)
    )
    (process_dir / "cmdline").write_bytes(b"python\0worker.py\0")

    real_path = Path

    def redirected_path(value):
        path = real_path(value)
        if path == real_path("/proc"):
            return proc_root
        if path == real_path("/proc/sys/kernel/random/boot_id"):
            return proc_root / "sys/kernel/random/boot_id"
        return path

    with patch("common.services.job_state.Path", side_effect=redirected_path):
        identity = capture_process_identity(123)

    assert identity is not None
    assert identity["start_marker"] == "linux:boot-123:98765"


def test_backtest_reconciliation_uses_exact_durable_job_lineage() -> None:
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = False
    connection.execute.return_value.rowcount = 1

    with patch("common.services.job_state._get_conn", return_value=connection):
        reconciled = reconcile_backtest_run("job-backtest-17", "failed")

    assert reconciled is True
    query, params = connection.execute.call_args.args
    assert "run.job_id = job.job_id" in query
    assert "job.params ->> 'backtest_run_id'" in query
    assert params == ("failed", "job-backtest-17")


def test_ambiguous_backtest_restart_fails_its_exact_tracking_run() -> None:
    manager = _bare_manager()
    manager._db_update_status = MagicMock(return_value=True)
    type_def = JobTypeDef(
        type_id="backtest_lgbm",
        label="LightGBM",
        description="test",
        group="backtest",
        callable=MagicMock(),
    )

    with patch(
        "common.services.job_registry.reconcile_backtest_run",
        return_value=True,
    ) as reconcile:
        manager._quarantine_recovered_job(
            "job-backtest-ambiguous",
            "Process exited without an exact attempt result",
            attempt_token="attempt-ambiguous",
            type_def=type_def,
        )

    reconcile.assert_called_once_with("job-backtest-ambiguous", "failed")


def test_named_pipeline_reuses_active_pipeline_under_database_lock() -> None:
    manager = _bare_manager()
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = False
    advisory_result = MagicMock()
    active_result = MagicMock()
    active_result.fetchone.return_value = ("pipe-active",)
    connection.execute.side_effect = [advisory_result, active_result]

    with (
        patch("common.services.job_registry._get_conn", return_value=connection),
        patch.object(manager, "submit_pipeline") as submit,
    ):
        result = manager.submit_named_pipeline(
            [{"job_type": "backtest_mstl", "params": {}}],
            label="model-refresh",
        )

    assert result == ("pipe-active", False)
    submit.assert_not_called()
    assert "pg_advisory_xact_lock" in connection.execute.call_args_list[0].args[0]
    assert "completed_at > NOW() - INTERVAL '5 minutes'" in (
        connection.execute.call_args_list[1].args[0]
    )


def test_named_pipeline_creates_once_when_no_active_pipeline_exists() -> None:
    manager = _bare_manager()
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = False
    advisory_result = MagicMock()
    inactive_result = MagicMock()
    inactive_result.fetchone.return_value = None
    connection.execute.side_effect = [advisory_result, inactive_result]

    with (
        patch("common.services.job_registry._get_conn", return_value=connection),
        patch.object(manager, "submit_pipeline", return_value="pipe-new") as submit,
    ):
        result = manager.submit_named_pipeline(
            [{"job_type": "backtest_mstl", "params": {}}],
            label="model-refresh",
            triggered_by="api",
        )

    assert result == ("pipe-new", True)
    submit.assert_called_once_with(
        steps=[{"job_type": "backtest_mstl", "params": {}}],
        label="model-refresh",
        triggered_by="api",
    )


def test_deleting_quarantined_job_releases_group_and_dispatches_next() -> None:
    manager = _bare_manager()
    manager._active_jobs["quarantine:job-q"] = "forecast"

    with (
        patch.object(manager, "_db_delete", return_value=("forecast", "ambiguous exit")),
        patch.object(manager, "_dispatch_next") as dispatch,
    ):
        assert manager.delete_job("job-q") is True

    assert "quarantine:job-q" not in manager._active_jobs
    dispatch.assert_called_once_with("forecast")


def test_bulk_purge_preserves_quarantines_for_single_job_acknowledgement() -> None:
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    conn.execute.return_value.rowcount = 3

    with patch("common.services.job_registry._get_conn", return_value=conn):
        assert JobManager.purge_history(status="failed") == 3

    sql = conn.execute.call_args.args[0]
    assert "recovery_quarantine_reason IS NULL" in sql


def _type_def(
    callable_fn: Callable[..., dict[str, Any]],
    *,
    type_id: str = "forecast-test",
    group: str = "forecast",
) -> JobTypeDef:
    return JobTypeDef(
        type_id=type_id,
        label="Forecast durability test",
        description="Forecast durability test",
        group=group,
        callable=callable_fn,
    )


def test_silent_subprocess_cancellation_finishes_promptly() -> None:
    """Cancellation must not wait for a silent child to emit a stdout line."""
    cancel_event = threading.Event()
    process_started = threading.Event()
    captured_processes: list[subprocess.Popen[str]] = []
    outcome: dict[str, object] = {}
    real_popen = subprocess.Popen

    def capture_process(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        process = real_popen(*args, **kwargs)
        captured_processes.append(process)
        process_started.set()
        return process

    def run_silent_child() -> None:
        try:
            outcome["result"] = _run_subprocess(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cancel_event=cancel_event,
            )
        except RuntimeError as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run_silent_child, daemon=True)
    started = False
    finished_after_cancel = False
    try:
        with (
            patch(
                "common.services.job_state.subprocess.Popen",
                side_effect=capture_process,
            ),
            patch("common.services.job_state._store_pid"),
            patch("common.services.job_state._clear_pid"),
        ):
            worker.start()
            started = process_started.wait(timeout=2)
            cancel_event.set()
            worker.join(timeout=1.5)
            finished_after_cancel = not worker.is_alive()
    finally:
        for process in captured_processes:
            if process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)
        worker.join(timeout=3)

    assert started, "silent child did not start"
    assert finished_after_cancel, "cancellation waited on silent stdout"
    assert isinstance(outcome.get("error"), RuntimeError)
    assert "cancel" in str(outcome["error"]).lower()


def test_terminal_cancellation_is_not_retried_or_overwritten() -> None:
    """A cancellation token is terminal even when retries were requested."""
    manager = _bare_manager()
    job_id = "cancelled-forecast"
    cancel_event = threading.Event()
    cancel_event.set()
    manager._active_jobs[job_id] = "forecast"
    manager._cancel_flags[job_id] = cancel_event
    manager._db_update_status = MagicMock()
    manager._dispatch_next = MagicMock()
    attempts = 0

    def cancelled_callable(
        _params: dict[str, Any],
        _progress_cb: Callable[..., None],
        *,
        cancel_event: threading.Event | None,
        job_id: str,
    ) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        if cancel_event and cancel_event.is_set():
            raise RuntimeError(f"Job {job_id} cancelled by user")
        return {"unexpected": "cancelled job resumed"}

    with patch("common.services.job_registry.time.sleep", return_value=None):
        manager._execute_job(
            job_id,
            _type_def(cancelled_callable),
            {},
            max_retries=2,
        )

    statuses = [call.args[1] for call in manager._db_update_status.call_args_list]
    assert attempts == 1
    assert statuses[-1] == "cancelled"
    assert "failed" not in statuses
    assert "completed" not in statuses


@pytest.mark.parametrize(
    "job_type",
    [
        "generate_customer_forecast",
        "generate_customer_forecast_backtest",
        "generate_customer_forecast_blend",
    ],
)
def test_late_customer_cancel_after_child_commit_keeps_job_completed(
    job_type: str,
) -> None:
    """A cancellation observed after a successful child commit must lose the race."""
    manager = _bare_manager()
    job_id = f"late-cancel-{job_type}"
    cancel_event = threading.Event()
    manager._active_jobs[job_id] = "forecast"
    manager._cancel_flags[job_id] = cancel_event
    manager._db_update_status = MagicMock(return_value=True)
    manager._dispatch_next = MagicMock()

    def committed_callable(
        _params: dict[str, Any],
        _progress_cb: Callable[..., None],
        *,
        cancel_event: threading.Event | None,
        job_id: str,
    ) -> dict[str, Any]:
        assert cancel_event is not None
        cancel_event.set()
        return {"run_id": job_id}

    with patch(
        "common.services.job_registry.finalize_customer_forecast_job_cancellation",
        return_value="completed",
    ) as finalize_cancellation:
        manager._execute_job(
            job_id,
            _type_def(committed_callable, type_id=job_type),
            {},
            max_retries=2,
        )

    statuses = [call.args[1] for call in manager._db_update_status.call_args_list]
    assert "cancelled" not in statuses
    assert "failed" not in statuses
    finalize_cancellation.assert_called_once_with(
        job_id,
        job_type,
        expected_status="running",
        expected_attempt_token=manager._db_update_status.call_args_list[0].kwargs[
            "attempt_token"
        ],
        retry_count=0,
    )


def test_cancel_endpoint_keeps_late_customer_success_completed() -> None:
    """The synchronous PID cancellation path honors an already-ready blend run."""
    manager = _bare_manager()
    job_id = "late-cancel-ready-blend"
    manager._db_get = MagicMock(
        return_value={
            "job_id": job_id,
            "job_type": "generate_customer_forecast_blend",
            "status": "running",
            "pid": 4321,
            "attempt_token": "attempt-ready",
        }
    )
    manager._kill_process = MagicMock(return_value=True)
    manager._db_update_status = MagicMock(return_value=True)
    manager._reconcile_job_terminal_state = MagicMock()

    with patch(
        "common.services.job_registry.finalize_customer_forecast_job_cancellation",
        return_value="completed",
    ) as finalize_cancellation:
        assert manager.cancel_job(job_id) is False

    finalize_cancellation.assert_called_once_with(
        job_id,
        "generate_customer_forecast_blend",
        expected_status="running",
        expected_attempt_token="attempt-ready",
        retry_count=None,
    )
    manager._db_update_status.assert_not_called()
    manager._reconcile_job_terminal_state.assert_not_called()


@pytest.mark.parametrize(
    ("job_type", "manifest_status"),
    [
        ("generate_customer_forecast", "completed"),
        ("generate_customer_forecast_backtest", "completed"),
        ("generate_customer_forecast_blend", "ready"),
        ("generate_customer_forecast_blend", "promoted"),
        ("generate_customer_forecast_blend", "archived"),
    ],
)
def test_atomic_customer_cancel_locks_manifest_before_success_wins(
    job_type: str,
    manifest_status: str,
) -> None:
    """A child commit serialized ahead of cancellation completes the managed job."""
    from common.services.job_state import (
        finalize_customer_forecast_job_cancellation,
    )

    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = False
    transaction = connection.transaction.return_value
    transaction.__enter__.return_value = transaction
    transaction.__exit__.return_value = False
    locked = MagicMock()
    locked.fetchone.return_value = ("manifest-run", manifest_status)
    updated = MagicMock(rowcount=1)
    results = iter([locked, updated])

    def execute_in_transaction(*_args: Any) -> MagicMock:
        assert transaction.__enter__.called
        assert not transaction.__exit__.called
        return next(results)

    connection.execute.side_effect = execute_in_transaction

    with (
        patch("common.services.job_state._get_conn", return_value=connection),
        patch("common.services.job_state._invalidate_customer_forecast_cache"),
    ):
        terminal_state = finalize_customer_forecast_job_cancellation(
            "job-late-success",
            job_type,
            expected_status="running",
            expected_attempt_token="attempt-17",
            retry_count=1,
        )

    assert terminal_state == "completed"
    transaction.__enter__.assert_called_once_with()
    lock_sql = connection.execute.call_args_list[0].args[0]
    assert "FOR UPDATE OF job" in lock_sql
    assert "job.params ->> 'run_id'" in lock_sql
    assert connection.execute.call_args_list[-1].args[1][0] == "completed"
    assert len(connection.execute.call_args_list) == 2


@pytest.mark.parametrize(
    ("job_type", "manifest_table"),
    [
        ("generate_customer_forecast", "customer_forecast_run"),
        (
            "generate_customer_forecast_backtest",
            "customer_forecast_backtest_run",
        ),
        ("generate_customer_forecast_blend", "forecast_generation_run"),
    ],
)
def test_atomic_customer_cancel_updates_manifest_and_job_under_one_lock(
    job_type: str,
    manifest_table: str,
) -> None:
    """Cancellation winning the row lock makes both terminal writes one transaction."""
    from common.services.job_state import (
        finalize_customer_forecast_job_cancellation,
    )

    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = False
    transaction = connection.transaction.return_value
    transaction.__enter__.return_value = transaction
    transaction.__exit__.return_value = False
    locked = MagicMock()
    locked.fetchone.return_value = ("manifest-run", "generating")
    manifest_updated = MagicMock(rowcount=1)
    job_updated = MagicMock(rowcount=1)
    results = iter([locked, manifest_updated, job_updated])

    def execute_in_transaction(*_args: Any) -> MagicMock:
        assert transaction.__enter__.called
        assert not transaction.__exit__.called
        return next(results)

    connection.execute.side_effect = execute_in_transaction

    with (
        patch("common.services.job_state._get_conn", return_value=connection),
        patch("common.services.job_state._invalidate_customer_forecast_cache"),
    ):
        terminal_state = finalize_customer_forecast_job_cancellation(
            "job-cancel-wins",
            job_type,
            expected_status="running",
            expected_attempt_token="attempt-18",
        )

    assert terminal_state == "cancelled"
    transaction.__enter__.assert_called_once_with()
    manifest_sql = connection.execute.call_args_list[1].args[0]
    assert f"UPDATE {manifest_table}" in manifest_sql
    job_sql, job_params = connection.execute.call_args_list[2].args
    assert "UPDATE job_history" in job_sql
    assert "'msg', %s::text" in job_sql
    assert job_params[0] == "cancelled"


def test_atomic_customer_cancel_rolls_back_if_job_transition_is_lost() -> None:
    """A manifest cancellation cannot commit without its matching job transition."""
    from common.services.job_state import (
        finalize_customer_forecast_job_cancellation,
    )

    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = False
    transaction = connection.transaction.return_value
    transaction.__enter__.return_value = transaction
    transaction.__exit__.return_value = False
    locked = MagicMock()
    locked.fetchone.return_value = ("manifest-run", "generating")
    manifest_updated = MagicMock(rowcount=1)
    job_update_lost = MagicMock(rowcount=0)
    connection.execute.side_effect = [locked, manifest_updated, job_update_lost]

    with (
        patch("common.services.job_state._get_conn", return_value=connection),
        patch("common.services.job_state._invalidate_customer_forecast_cache"),
    ):
        terminal_state = finalize_customer_forecast_job_cancellation(
            "job-transition-lost",
            "generate_customer_forecast",
            expected_status="running",
            expected_attempt_token="attempt-lost",
        )

    assert terminal_state is None
    transaction.__exit__.assert_called_once()
    assert transaction.__exit__.call_args.args[0] is RuntimeError


def test_retry_retains_group_and_cancel_token_until_terminal_exit() -> None:
    """A transient retry must keep its concurrency lease and cancellation token."""
    manager = _bare_manager()
    job_id = "retry-forecast"
    cancel_event = threading.Event()
    manager._active_jobs[job_id] = "forecast"
    manager._cancel_flags[job_id] = cancel_event
    manager._db_update_status = MagicMock()
    attempt_state: list[tuple[bool, bool]] = []
    dispatch_after_attempt: list[int] = []

    def flaky_callable(
        _params: dict[str, Any],
        _progress_cb: Callable[..., None],
        *,
        cancel_event: threading.Event | None,
        job_id: str,
    ) -> dict[str, Any]:
        attempt_state.append(
            (
                job_id in manager._active_jobs,
                cancel_event is original_cancel_event
                and manager._cancel_flags.get(job_id) is original_cancel_event,
            )
        )
        if len(attempt_state) == 1:
            raise RuntimeError("transient forecast dependency failure")
        return {"status": "completed"}

    manager._dispatch_next = MagicMock(
        side_effect=lambda _group: dispatch_after_attempt.append(len(attempt_state))
    )
    original_cancel_event = cancel_event
    with (
        patch("common.services.job_registry.time.sleep", return_value=None),
        patch.object(cancel_event, "wait", return_value=False),
    ):
        manager._execute_job(
            job_id,
            _type_def(flaky_callable),
            {},
            max_retries=1,
        )

    assert attempt_state == [(True, True), (True, True)]
    assert dispatch_after_attempt == [2]
    assert job_id not in manager._active_jobs
    assert job_id not in manager._cancel_flags


def test_attempt_claim_and_terminal_write_share_one_exact_token() -> None:
    """A stale callback cannot complete the row claimed by another attempt."""
    manager = _bare_manager()
    job_id = "token-cas"
    manager._active_jobs[job_id] = "forecast"
    manager._cancel_flags[job_id] = threading.Event()
    manager._db_update_status = MagicMock(return_value=True)
    manager._dispatch_next = MagicMock()

    manager._execute_job(
        job_id,
        _type_def(MagicMock(return_value={"ok": True})),
        {},
        execution_group="forecast",
    )

    running_claim = manager._db_update_status.call_args_list[0]
    attempt_token = running_claim.kwargs["attempt_token"]
    assert attempt_token
    assert running_claim.kwargs["attempt_failure_recorded"] is False
    completed = next(
        call
        for call in manager._db_update_status.call_args_list
        if call.args[1] == "completed"
    )
    assert completed.kwargs["expected_attempt_token"] == attempt_token


def test_live_job_is_readopted_with_exact_persisted_group() -> None:
    """Recovery monitors a surviving wrapper without recomputing its group."""
    manager = _bare_manager()
    connection = MagicMock()
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)

    def execute(query: str, *_args: Any) -> MagicMock:
        result = MagicMock()
        result.rowcount = 0
        sql = str(query)
        if "UPDATE job_history running" in sql:
            result.fetchall.return_value = [
                (
                        "live-generation",
                        "generate_production_forecast",
                        4321,
                        {
                            "__process_identity": {
                                "start_marker": "Sun Jul 12 03:00:00 2026",
                                "command_marker": "generation",
                            },
                            "__attempt_command_digest": "generation-command",
                        },
                    0,
                    "pipe-release",
                    0,
                    "forecast-release-42",
                    "attempt-generation",
                    None,
                    False,
                )
            ]
        else:
            result.fetchall.return_value = []
        return result

    connection.execute.side_effect = execute
    manager._is_pid_alive = MagicMock(return_value=True)
    manager._readopt_job = MagicMock()

    with (
        patch("common.services.job_registry._get_conn", return_value=connection),
        patch(
            "common.services.job_registry.process_identity_matches",
            return_value=True,
        ),
        patch("common.services.job_registry.time.sleep", return_value=None),
    ):
        manager.recover_stale_jobs()

    manager._readopt_job.assert_called_once()
    readopt = manager._readopt_job.call_args
    assert readopt.kwargs["execution_group"] == "forecast-release-42"
    assert readopt.kwargs["attempt_token"] == "attempt-generation"


def test_named_pipeline_persists_backtest_identity() -> None:
    """Restart recovery must see the exact named-pipeline backtest run."""
    manager = _bare_manager()
    manager._ensure_init = MagicMock()
    manager._db_insert = MagicMock()

    with (
        patch(
            "common.services.job_registry._reserve_backtest_run",
            return_value=71,
            create=True,
        ),
        patch(
            "common.services.job_state._reserve_backtest_run",
            return_value=71,
        ),
    ):
        manager.submit_pipeline(
            [{"job_type": "backtest_mstl", "params": {}}],
            label="model-refresh",
        )

    persisted = {
        call.args[1]: call.args[3] for call in manager._db_insert.call_args_list
    }
    assert persisted["backtest_mstl"]["backtest_run_id"] == 71


def test_named_pipeline_persists_generation_identity() -> None:
    """Restart recovery must reuse one immutable production generation UUID."""
    manager = _bare_manager()
    manager._ensure_init = MagicMock()
    manager._db_insert = MagicMock()
    manager.submit_pipeline(
        [{"job_type": "generate_production_forecast", "params": {}}],
        label="forecast-publish",
    )

    persisted = {
        call.args[1]: call.args[3] for call in manager._db_insert.call_args_list
    }
    generation_run_id = persisted["generate_production_forecast"]["run_id"]
    assert str(UUID(str(generation_run_id))) == str(generation_run_id)


def test_recovered_completed_pipeline_step_advances_next_exactly_once() -> None:
    """Repeated recovery observation must create one successor, not zero or two."""
    manager = _bare_manager()
    remaining = [{"job_type": "champion_select", "params": {}}]
    params = {
        "backtest_run_id": 41,
        "model_id": "mstl",
        "__pipeline_step": 2,
        "__pipeline_total_steps": 3,
        "__pipeline_label": "model-refresh",
        "__pipeline_remaining": remaining,
        "__attempt_command_digest": "mstl",
    }
    job = {
        "status": "running",
        "pipeline_id": "pipe-model-refresh",
        "params": params,
        "attempt_token": "attempt-mstl",
        "attempt_result": {
            "attempt_token": "attempt-mstl",
            "exit_code": 0,
            "completed_at": "2026-07-12T09:00:00+00:00",
            "command_digest": "mstl",
        },
        "retry_count": 0,
        "max_retries": 0,
    }
    manager._is_pid_alive = MagicMock(return_value=False)
    manager._db_get = MagicMock(side_effect=lambda _job_id: job)
    manager._finalize_recovered_job = MagicMock()
    manager._advance_pipeline_step_once = MagicMock(return_value=True)
    manager._dispatch_next = MagicMock()

    def update_status(
        _job_id: str,
        status: str,
        *,
        expected_status: str | None = None,
        **_kwargs: Any,
    ) -> bool:
        if expected_status is not None and job["status"] != expected_status:
            return False
        job["status"] = status
        return True

    manager._db_update_status = MagicMock(side_effect=update_status)

    class InlineThread:
        def __init__(self, *, target: Callable[[], None], **_kwargs: Any) -> None:
            self._target = target

        def start(self) -> None:
            self._target()

    type_def = _type_def(
        MagicMock(),
        type_id="backtest_mstl",
        group="backtest",
    )
    with patch("common.services.job_registry.threading.Thread", InlineThread):
        manager._readopt_job(
            "recovered-mstl",
            type_def,
            12345,
            attempt_token="attempt-mstl",
        )
        manager._readopt_job(
            "recovered-mstl",
            type_def,
            12345,
            attempt_token="attempt-mstl",
        )

    manager._advance_pipeline_step_once.assert_called_once_with(
        "recovered-mstl",
        "pipe-model-refresh",
        remaining,
        params,
    )


def test_pipeline_step_lock_observes_existing_successor_on_repeat() -> None:
    """Two continuation observations submit at most one next-step row."""
    manager = _bare_manager()
    remaining = [{"job_type": "champion_select", "params": {}}]
    params = {"__pipeline_step": 2, "__pipeline_remaining": remaining}
    successor_exists = False
    connection = MagicMock()
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)

    def execute(query: str, *_args: Any) -> MagicMock:
        result = MagicMock()
        sql = str(query)
        if "UPDATE job_history completed" in sql:
            result.fetchone.return_value = None if successor_exists else (1,)
        elif "FROM job_history" in sql:
            result.fetchone.return_value = (1,) if successor_exists else None
        return result

    def submit_successor(*_args: Any) -> str:
        nonlocal successor_exists
        successor_exists = True
        return "next-job"

    connection.execute.side_effect = execute
    manager._trigger_next_pipeline_step = MagicMock(side_effect=submit_successor)
    with patch("common.services.job_registry._get_conn", return_value=connection):
        assert manager._advance_pipeline_step_once(
            "completed-mstl", "pipe-model-refresh", remaining, params
        )
        assert manager._advance_pipeline_step_once(
            "completed-mstl", "pipe-model-refresh", remaining, params
        )

    manager._trigger_next_pipeline_step.assert_called_once()


def test_snapshot_bundle_performs_archive_gated_cleanup() -> None:
    """The bundle cleans staging only after its preceding archive reconciliation."""
    steps = preset_steps(get_pipeline_preset("forecast-snapshot-bundle"))
    cleanup = next(
        step for step in steps if step["job_type"] == "cleanup_forecast_staging"
    )
    assert cleanup["params"]["dry_run"] is False


def test_cancel_failure_does_not_persist_terminal_cancelled() -> None:
    """A surviving child must keep the durable job non-terminal."""
    manager = _bare_manager()
    manager._db_get = MagicMock(
        return_value={"job_id": "live", "status": "running"}
    )
    manager._cancel_flags["live"] = threading.Event()
    manager._kill_process = MagicMock(return_value=False)
    manager._db_update_status = MagicMock(return_value=True)

    assert manager.cancel_job("live") is False
    terminal_updates = [
        call
        for call in manager._db_update_status.call_args_list
        if len(call.args) > 1 and call.args[1] == "cancelled"
    ]
    assert terminal_updates == []


def test_queued_cancel_race_retains_worker_lease_and_token() -> None:
    """If a queued worker starts concurrently, it must still observe cancellation."""
    manager = _bare_manager()
    job_id = "queued-race"
    cancel_event = threading.Event()
    manager._active_jobs[job_id] = "forecast"
    manager._cancel_flags[job_id] = cancel_event
    manager._db_get = MagicMock(
        return_value={"job_id": job_id, "status": "queued", "pid": None}
    )
    manager._kill_process = MagicMock(return_value=True)
    manager._db_update_status = MagicMock(return_value=False)
    manager._dispatch_next = MagicMock()

    assert manager.cancel_job(job_id) is True
    assert cancel_event.is_set()
    assert manager._active_jobs[job_id] == "forecast"
    assert manager._cancel_flags[job_id] is cancel_event
    manager._dispatch_next.assert_not_called()


@pytest.mark.parametrize(
    "job_type",
    [
        "generate_customer_forecast",
        "generate_customer_forecast_backtest",
        "generate_customer_forecast_blend",
    ],
)
def test_queued_customer_cancel_reconciles_reserved_manifest(job_type: str) -> None:
    """A queued cancellation must still terminate its reserved customer manifest."""
    manager = _bare_manager()
    job_id = f"queued-{job_type}"
    manager._db_get = MagicMock(
        return_value={
            "job_id": job_id,
            "job_type": job_type,
            "status": "queued",
            "pid": None,
            "attempt_token": None,
        }
    )
    manager._kill_process = MagicMock(return_value=True)
    manager._db_update_status = MagicMock(return_value=True)
    manager._reconcile_job_terminal_state = MagicMock()
    manager._dispatch_next = MagicMock()

    with patch(
        "common.services.job_registry.finalize_customer_forecast_job_cancellation",
        return_value="cancelled",
    ) as finalize_cancellation:
        assert manager.cancel_job(job_id) is True

    finalize_cancellation.assert_called_once_with(
        job_id,
        job_type,
        expected_status="queued",
        expected_attempt_token=None,
        retry_count=None,
    )
    manager._reconcile_job_terminal_state.assert_not_called()
    manager._db_update_status.assert_not_called()


def test_recovered_process_death_honors_pending_cancellation() -> None:
    """A re-adopt monitor must not finalize after cancellation killed the PID."""
    manager = _bare_manager()
    job_id = "recovered-cancel"
    manager._is_pid_alive = MagicMock(return_value=False)
    manager._db_get = MagicMock(
        return_value={
            "status": "running",
            "pipeline_id": "pipe-release",
            "params": {
                "backtest_run_id": 41,
                "__pipeline_remaining": [
                    {"job_type": "champion_select", "params": {}}
                ],
            },
        }
    )
    manager._db_update_status = MagicMock(return_value=True)
    manager._finalize_recovered_job = MagicMock()
    manager._advance_pipeline_step_once = MagicMock()
    manager._dispatch_next = MagicMock()

    class InlineThread:
        def __init__(self, *, target: Callable[[], None], **_kwargs: Any) -> None:
            self._target = target

        def start(self) -> None:
            manager._cancel_flags[job_id].set()
            self._target()

    type_def = _type_def(
        MagicMock(),
        type_id="backtest_mstl",
        group="backtest",
    )
    with patch("common.services.job_registry.threading.Thread", InlineThread):
        manager._readopt_job(job_id, type_def, 12345)

    statuses = [call.args[1] for call in manager._db_update_status.call_args_list]
    assert statuses == ["cancelled"]
    manager._finalize_recovered_job.assert_not_called()
    manager._advance_pipeline_step_once.assert_not_called()


def test_pid_reuse_is_never_signalled() -> None:
    """A start-marker mismatch proves the recorded process already exited."""
    expected = {
        "start_marker": "Sun Jul 12 03:00:00 2026",
        "command_marker": "expected",
    }
    with (
        patch("common.services.job_registry.get_job_pid", return_value=4321),
        patch(
            "common.services.job_registry.get_job_process_identity",
            return_value=expected,
            create=True,
        ),
        patch(
            "common.services.job_registry.process_identity_matches",
            return_value=False,
            create=True,
        ),
        patch("common.services.job_registry.os.getpgid", return_value=4321),
        patch("common.services.job_registry.os.killpg") as kill_group,
    ):
        assert JobManager._kill_process("reused-pid") is True

    kill_group.assert_not_called()


def test_subprocess_persists_start_and_command_identity() -> None:
    """Recovery metadata must identify more than a recyclable integer PID."""
    process = MagicMock()
    process.pid = 9876
    process.stdout = iter([])
    process.poll.return_value = 0
    process.wait.return_value = None
    process.returncode = 0
    identity = {
        "start_marker": "Sun Jul 12 03:00:00 2026",
        "command_marker": "command-sha256",
    }
    with (
        patch("common.services.job_state.subprocess.Popen", return_value=process),
        patch("common.services.job_state._store_pid"),
        patch("common.services.job_state._clear_pid"),
        patch("common.services.job_state._append_log"),
        patch(
            "common.services.job_state.capture_process_identity",
            return_value=identity,
            create=True,
        ),
        patch(
            "common.services.job_state._store_process_identity",
            create=True,
        ) as store_identity,
    ):
        _run_subprocess(["uv", "run", "python", "forecast.py"], job_id="job-9")

    store_identity.assert_called_once_with("job-9", 9876, identity)


def test_progress_failure_does_not_leave_an_orphan_child() -> None:
    """Any manager-side exception must terminate the still-running child."""
    process = MagicMock()
    process.pid = 9877
    process.stdout = iter(["first line\n"])
    process.poll.return_value = None
    process.wait.return_value = None
    process.returncode = 0
    with (
        patch("common.services.job_state.subprocess.Popen", return_value=process),
        patch("common.services.job_state.capture_process_identity", return_value=None),
        patch("common.services.job_state._store_pid"),
        patch("common.services.job_state._clear_pid"),
        patch("common.services.job_state._append_log"),
        patch("common.services.job_state._terminate_subprocess") as terminate,
    ):
        with pytest.raises(RuntimeError, match="progress backend failed"):
            _run_subprocess(
                ["uv", "run", "python", "forecast.py"],
                progress_cb=MagicMock(
                    side_effect=RuntimeError("progress backend failed")
                ),
            )

    terminate.assert_called_once_with(process)


def test_unverifiable_pid_is_never_signalled_or_declared_dead() -> None:
    """Missing identity metadata must fail closed."""
    with (
        patch("common.services.job_registry.get_job_pid", return_value=4321),
        patch(
            "common.services.job_registry.get_job_process_identity",
            return_value=None,
            create=True,
        ),
        patch("common.services.job_registry.os.killpg") as kill_group,
    ):
        assert JobManager._kill_process("unknown-pid") is False

    kill_group.assert_not_called()


def test_readopt_monitor_quarantines_reused_pid_without_exit_result() -> None:
    """A recycled PID is never treated as successful completion evidence."""
    manager = _bare_manager()
    identity = {
        "start_marker": "Sun Jul 12 03:00:00 2026",
        "command_marker": "mstl",
    }
    manager._is_pid_alive = MagicMock(return_value=True)
    manager._db_get = MagicMock(
        return_value={
            "status": "running",
            "params": {"backtest_run_id": 41},
            "attempt_token": "attempt-reused",
            "attempt_result": None,
        }
    )
    manager._db_update_status = MagicMock(return_value=True)
    manager._finalize_recovered_job = MagicMock()
    manager._dispatch_next = MagicMock()

    class InlineThread:
        def __init__(self, *, target: Callable[[], None], **_kwargs: Any) -> None:
            self._target = target

        def start(self) -> None:
            self._target()

    type_def = _type_def(
        MagicMock(), type_id="backtest_mstl", group="backtest"
    )
    with (
        patch("common.services.job_registry.threading.Thread", InlineThread),
        patch(
            "common.services.job_registry.process_identity_matches",
            return_value=False,
        ),
    ):
        manager._readopt_job(
            "recovered-reuse",
            type_def,
            12345,
            attempt_token="attempt-reused",
            process_identity=identity,
        )

    manager._finalize_recovered_job.assert_not_called()
    quarantines = [
        call
        for call in manager._db_update_status.call_args_list
        if call.kwargs.get("recovery_quarantine_reason")
    ]
    assert len(quarantines) == 1
    assert quarantines[0].args[1] == "failed"


def test_restart_preserves_consumed_retry_budget() -> None:
    """A restart must resume with the persisted retry count, not reset to zero."""
    manager = _bare_manager()
    connection = MagicMock()
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)

    def execute(query: str, *_args: Any) -> MagicMock:
        result = MagicMock()
        result.rowcount = 0
        sql = str(query)
        if "WHERE status = 'running'" in sql and "SELECT job_id" in sql:
            result.fetchall.return_value = []
        elif "UPDATE job_history queued" in sql:
            result.fetchall.return_value = [
                (
                    "retry-mstl",
                    "backtest_mstl",
                    {"backtest_run_id": 91},
                    3,
                    2,
                    "pipe-model-refresh",
                    "backtest",
                )
            ]
        elif "status = 'completed'" in sql:
            result.fetchall.return_value = []
        else:
            result.fetchall.return_value = []
        return result

    connection.execute.side_effect = execute
    with patch("common.services.job_registry._get_conn", return_value=connection):
        manager.recover_stale_jobs()

    dispatch = manager._scheduler.add_job.call_args
    assert dispatch is not None
    assert dispatch.kwargs["args"][-2] == 2


def test_direct_backtest_submission_persists_tracking_identity() -> None:
    """Generic /jobs launches need the same restart-safe ID as named pipelines."""
    manager = _bare_manager()
    manager._ensure_init = MagicMock()
    manager._db_insert = MagicMock()

    with patch(
        "common.services.job_registry._reserve_backtest_run",
        return_value=71,
    ) as reserve:
        manager.submit_job("backtest_mstl", params={})

    reserve.assert_called_once()
    persisted_params = manager._db_insert.call_args.args[3]
    assert persisted_params["backtest_run_id"] == 71


def test_startup_reconciles_completed_pipeline_without_successor() -> None:
    """The completed-before-submit crash window must heal on startup."""
    manager = _bare_manager()
    manager._advance_pipeline_step_once = MagicMock(return_value=True)
    connection = MagicMock()
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)
    params = {
        "__pipeline_step": 2,
        "__pipeline_total_steps": 3,
        "__pipeline_remaining": [
            {"job_type": "champion_select", "params": {}}
        ],
    }

    def execute(query: str, *_args: Any) -> MagicMock:
        result = MagicMock()
        result.rowcount = 0
        sql = str(query)
        if "status = 'completed'" in sql and "SELECT" in sql:
            result.fetchall.return_value = [
                ("completed-mstl", "pipe-model-refresh", params)
            ]
        else:
            result.fetchall.return_value = []
        return result

    connection.execute.side_effect = execute
    with patch("common.services.job_registry._get_conn", return_value=connection):
        manager.recover_stale_jobs()

    manager._advance_pipeline_step_once.assert_called_once_with(
        "completed-mstl",
        "pipe-model-refresh",
        params["__pipeline_remaining"],
        params,
    )


def test_cleanup_job_defaults_to_non_destructive_dry_run() -> None:
    """An omitted cleanup parameter is never implicit deletion approval."""
    with patch(
        "common.services.job_state._run_subprocess",
        return_value="previewed",
    ) as run:
        _run_cleanup_forecast_staging({})

    assert "--dry-run" in run.call_args.args[0]
    from common.services.job_registry import JOB_TYPE_REGISTRY

    assert (
        JOB_TYPE_REGISTRY["cleanup_forecast_staging"].params_schema["dry_run"]
        is True
    )


def test_chronos_backtest_timeout_covers_documented_runtime() -> None:
    """The ~6h Chronos job must not inherit the generic two-hour timeout."""
    connection = MagicMock()
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)
    with (
        patch("common.services.job_state._get_conn", return_value=connection),
        patch("common.services.job_state._run_subprocess", return_value="ok") as run,
        patch("common.services.job_state._auto_load_backtest"),
        patch("common.services.job_state._update_backtest_run_on_completion"),
        patch("common.services.job_state.record_backtest_artifact_identity"),
        patch("common.services.job_state.verify_backtest_artifact_identity"),
    ):
        _run_backtest(
            "chronos2_enriched",
            {"backtest_run_id": 17, "model_id": "chronos2_enriched"},
        )

    assert run.call_args.kwargs["timeout_seconds"] >= 6 * 60 * 60


def test_readopted_tuning_finalization_failure_is_terminal() -> None:
    """A failed durable tuning registration must not be marked completed."""
    manager = _bare_manager()
    manager._db_get = MagicMock(
        return_value={"params": {"run_id": 17, "model": "lgbm"}}
    )

    with patch(
        "common.ml.tuning_tracker.complete_run",
        side_effect=RuntimeError("registration failed"),
    ):
        with pytest.raises(RuntimeError, match="registration failed"):
            manager._finalize_tuning_run("recovered-tuning")


def test_subprocess_kills_child_when_attempt_identity_cannot_be_persisted(
    tmp_path,
) -> None:
    """The domain command stays gated until its exact attempt lease is durable."""
    from common.services.job_state import (
        bind_job_attempt,
        reset_job_attempt,
    )

    process = MagicMock()
    process.pid = 9911
    process.stdout = iter([])
    process.poll.return_value = None
    identity = {
        "start_marker": "Sun Jul 12 04:00:00 2026",
        "command_marker": "wrapper-command",
    }
    result_path = tmp_path / "result.json"
    gate_path = tmp_path / "gate"
    context_token = bind_job_attempt("attempt-exact")
    try:
        with (
            patch("common.services.job_state.subprocess.Popen", return_value=process),
            patch(
                "common.services.job_state._prepare_attempt_files",
                return_value=(result_path, gate_path),
            ),
            patch(
                "common.services.job_state.capture_process_identity",
                return_value=identity,
            ),
            patch(
                "common.services.job_state._store_attempt_process",
                return_value=False,
            ),
            patch("common.services.job_state._terminate_subprocess") as terminate,
        ):
            with pytest.raises(RuntimeError, match="persist attempt identity"):
                _run_subprocess([sys.executable, "-c", "pass"], job_id="job-lease")
    finally:
        reset_job_attempt(context_token)

    terminate.assert_called_once_with(process)
    assert not gate_path.exists(), "the child command gate must remain closed"


def test_attempt_pid_clear_and_result_write_are_exact_token_guarded() -> None:
    """A stale callback cannot clear or overwrite a newer process attempt."""
    from common.services.job_state import _clear_pid, _store_attempt_result

    connection = MagicMock()
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)
    connection.execute.return_value.rowcount = 1
    result = {
        "attempt_token": "attempt-new",
        "exit_code": 0,
        "completed_at": "2026-07-12T09:00:00+00:00",
        "command_digest": "abc123",
    }

    with patch("common.services.job_state._get_conn", return_value=connection):
        assert _store_attempt_result("job-token", "attempt-new", result) is True
        _clear_pid("job-token", "attempt-new")

    result_sql, result_args = connection.execute.call_args_list[0].args
    clear_sql, clear_args = connection.execute.call_args_list[1].args
    assert "attempt_token = %s" in result_sql
    assert result_args[-2:] == ("job-token", "attempt-new")
    assert "attempt_token = %s" in clear_sql
    assert clear_args[-2:] == ("job-token", "attempt-new")


def test_managed_wrapper_records_exact_token_exit_and_command_digest(tmp_path) -> None:
    """The outliving wrapper leaves deterministic evidence for recovery."""
    from common.services.job_state import (
        _command_digest,
        bind_job_attempt,
        reset_job_attempt,
    )

    command = [sys.executable, "-c", "print('durable-wrapper-ok')"]
    stored_results: list[dict[str, Any]] = []
    context_token = bind_job_attempt("attempt-wrapper")
    try:
        with (
            patch("common.services.job_state._DATA_DIR", tmp_path),
            patch(
                "common.services.job_state.capture_process_identity",
                return_value={
                    "start_marker": "Sun Jul 12 04:00:00 2026",
                    "command_marker": "wrapper",
                },
            ),
            patch(
                "common.services.job_state._store_attempt_process",
                return_value=True,
            ) as store_process,
            patch(
                "common.services.job_state._store_attempt_result",
                side_effect=lambda _job, _token, result: (
                    stored_results.append(result) or True
                ),
            ),
            patch("common.services.job_state._clear_pid"),
        ):
            output = _run_subprocess(command, job_id="job-wrapper")
    finally:
        reset_job_attempt(context_token)

    assert output == "durable-wrapper-ok"
    assert stored_results == [
        {
            "attempt_token": "attempt-wrapper",
            "exit_code": 0,
            "completed_at": stored_results[0]["completed_at"],
            "command_digest": _command_digest(command),
        }
    ]
    assert stored_results[0]["completed_at"]
    assert store_process.call_args.args[-2:] == (
        "attempt-wrapper",
        _command_digest(command),
    )


def test_submit_persists_and_dispatches_exact_execution_group() -> None:
    """A group override must survive restart instead of being recomputed."""
    manager = _bare_manager()
    manager._ensure_init = MagicMock()
    manager._db_insert = MagicMock()

    manager.submit_job(
        "model_tuning_run",
        params={"model": "lgbm"},
        group_override="tuning_cluster_17",
    )

    assert manager._db_insert.call_args.kwargs["execution_group"] == "tuning_cluster_17"
    scheduled_args = manager._scheduler.add_job.call_args.kwargs["args"]
    assert scheduled_args[-1] == "tuning_cluster_17"


def test_recovery_leases_and_deduplicates_queued_job_ids() -> None:
    """Concurrent/repeated startup scans may enqueue each durable row once."""
    manager = _bare_manager()
    connection = MagicMock()
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)

    def execute(query: str, *_args: Any) -> MagicMock:
        result = MagicMock()
        result.rowcount = 0
        sql = str(query)
        if "UPDATE job_history queued" in sql:
            row = (
                "queued-mstl",
                "backtest_mstl",
                {"backtest_run_id": 91},
                2,
                0,
                "pipe-model-refresh",
                "backtest_exact",
            )
            result.fetchall.return_value = [row, row]
        elif "recovery_quarantine_reason IS NOT NULL" in sql:
            result.fetchall.return_value = []
        elif "WHERE status = 'running'" in sql and "SELECT job_id" in sql:
            result.fetchall.return_value = []
        elif "status = 'completed'" in sql:
            result.fetchall.return_value = []
        else:
            result.fetchall.return_value = []
        return result

    connection.execute.side_effect = execute
    with patch("common.services.job_registry._get_conn", return_value=connection):
        manager.recover_stale_jobs()

    assert manager._scheduler.add_job.call_count == 1
    scheduled_args = manager._scheduler.add_job.call_args.kwargs["args"]
    assert scheduled_args[-1] == "backtest_exact"
    lease_sql = next(
        str(call.args[0])
        for call in connection.execute.call_args_list
        if "UPDATE job_history queued" in str(call.args[0])
    )
    assert "recovery_lease_owner" in lease_sql


def test_pid_reuse_without_exact_attempt_result_is_quarantined() -> None:
    """PID disappearance/reuse is not evidence that a forecast succeeded."""
    manager = _bare_manager()
    manager._is_pid_alive = MagicMock(return_value=True)
    manager._db_get = MagicMock(
        return_value={
            "status": "running",
            "params": {},
            "attempt_token": "attempt-old",
            "attempt_result": None,
            "execution_group": "backtest_exact",
        }
    )
    manager._db_update_status = MagicMock(return_value=True)
    manager._finalize_recovered_job = MagicMock()
    manager._dispatch_next = MagicMock()

    class InlineThread:
        def __init__(self, *, target: Callable[[], None], **_kwargs: Any) -> None:
            self._target = target

        def start(self) -> None:
            self._target()

    with (
        patch("common.services.job_registry.threading.Thread", InlineThread),
        patch(
            "common.services.job_registry.process_identity_matches",
            return_value=False,
        ),
        patch(
            "common.services.job_registry.load_attempt_result",
            return_value=None,
        ),
    ):
        manager._readopt_job(
            "recovered-reuse",
            _type_def(MagicMock(), type_id="backtest_mstl", group="backtest"),
            12345,
            execution_group="backtest_exact",
            attempt_token="attempt-old",
            process_identity={
                "start_marker": "Sun Jul 12 03:00:00 2026",
                "command_marker": "mstl",
            },
        )

    manager._finalize_recovered_job.assert_not_called()
    quarantines = [
        call
        for call in manager._db_update_status.call_args_list
        if call.kwargs.get("recovery_quarantine_reason")
    ]
    assert len(quarantines) == 1
    assert quarantines[0].args[1] == "failed"
    manager._dispatch_next.assert_not_called()


def test_recovered_nonzero_attempt_result_consumes_retry_without_finalizing() -> None:
    """A durable nonzero exit is a failed attempt, not an ambiguous restart."""
    manager = _bare_manager()
    job = {
        "status": "running",
        "params": {
            "backtest_run_id": 91,
            "__attempt_command_digest": "cmd",
        },
        "attempt_token": "attempt-failed",
        "attempt_result": {
            "attempt_token": "attempt-failed",
            "exit_code": 2,
            "completed_at": "2026-07-12T09:00:00+00:00",
            "command_digest": "cmd",
        },
        "retry_count": 0,
        "max_retries": 1,
        "attempt_failure_recorded": False,
        "execution_group": "backtest_exact",
    }
    manager._db_get = MagicMock(return_value=job)
    manager._db_update_status = MagicMock(return_value=True)
    manager._finalize_recovered_job = MagicMock()
    manager._dispatch_next = MagicMock()

    manager._reconcile_recovered_attempt(
        "recovered-failure",
        _type_def(MagicMock(), type_id="backtest_mstl", group="backtest"),
        job,
        execution_group="backtest_exact",
    )

    manager._finalize_recovered_job.assert_not_called()
    requeue = manager._db_update_status.call_args
    assert requeue.args[1] == "queued"
    assert requeue.kwargs["retry_count"] == 1
    assert requeue.kwargs["expected_attempt_token"] == "attempt-failed"


def test_recovery_does_not_double_consume_a_persisted_retry_failure() -> None:
    """A crash during retry backoff resumes the retry already accounted for."""
    manager = _bare_manager()
    job = {
        "status": "running",
        "params": {"__attempt_command_digest": "cmd"},
        "attempt_token": "attempt-recorded",
        "attempt_result": {
            "attempt_token": "attempt-recorded",
            "exit_code": 2,
            "completed_at": "2026-07-12T09:00:00+00:00",
            "command_digest": "cmd",
        },
        "retry_count": 1,
        "max_retries": 1,
        "attempt_failure_recorded": True,
    }
    manager._db_update_status = MagicMock(return_value=True)
    manager._finalize_recovered_job = MagicMock()

    with patch("common.services.job_registry._clear_pid"):
        released = manager._reconcile_recovered_attempt(
            "retry-backoff-crash",
            _type_def(MagicMock(), type_id="backtest_mstl", group="backtest"),
            job,
            execution_group="backtest",
        )

    assert released is True
    requeue = manager._db_update_status.call_args
    assert requeue.args[1] == "queued"
    assert requeue.kwargs["retry_count"] == 1
    assert requeue.kwargs["attempt_failure_recorded"] is False


def test_recovered_ready_generation_finalizes_without_duplicate_rerun() -> None:
    """A committed ready run_id is reconciled, never inserted a second time."""
    manager = _bare_manager()
    generation_callable = MagicMock()
    run_id = "a35d902f-462d-45e2-8830-eec73c686b52"
    job = {
        "status": "running",
        "params": {
            "run_id": run_id,
            "model_id": "champion",
            "__attempt_command_digest": "generation-command",
        },
        "attempt_token": "attempt-generation-ready",
        "attempt_result": {
            "attempt_token": "attempt-generation-ready",
            "exit_code": 0,
            "completed_at": "2026-07-12T09:00:00+00:00",
            "command_digest": "generation-command",
        },
        "retry_count": 0,
        "max_retries": 0,
        "execution_group": "forecast-release",
    }
    manager._db_update_status = MagicMock(return_value=True)
    manager._advance_pipeline_step_once = MagicMock()
    connection = MagicMock()
    connection.__enter__ = MagicMock(return_value=connection)
    connection.__exit__ = MagicMock(return_value=False)
    connection.execute.return_value.fetchone.return_value = (
        "ready",
        120,
        "a" * 64,
    )

    with (
        patch("common.services.job_registry._get_conn", return_value=connection),
        patch("common.services.job_registry._clear_pid"),
    ):
        released = manager._reconcile_recovered_attempt(
            "generation-ready",
            _type_def(
                generation_callable,
                type_id="generate_production_forecast",
                group="forecast",
            ),
            job,
            execution_group="forecast-release",
        )

    assert released is True
    generation_callable.assert_not_called()
    terminal = manager._db_update_status.call_args
    assert terminal.args[1] == "completed"
    assert terminal.kwargs["expected_attempt_token"] == (
        "attempt-generation-ready"
    )
    assert "forecast_generation_run" in str(connection.execute.call_args.args[0])


def test_recovered_champion_load_replays_mandatory_lineage_only(tmp_path) -> None:
    """Exit-zero recovery restores the audit transaction without rerunning load."""
    champion_dir = tmp_path / "champion"
    champion_dir.mkdir()
    winners = champion_dir / "experiment_17_winners.csv"
    winners.write_text("item_id,loc,model_id\nA,L,mstl\n", encoding="utf-8")
    manager = _bare_manager()
    type_def = _type_def(
        MagicMock(),
        type_id="champion_results_load",
        group="champion",
    )

    with (
        patch("common.core.paths.DATA_DIR", tmp_path),
        patch(
            "common.services.job_registry._finalize_champion_results_lineage"
        ) as finalize_lineage,
    ):
        manager._finalize_recovered_job(
            "champion-load-17",
            type_def,
            {"params": {"experiment_id": 17}},
        )

    finalize_lineage.assert_called_once_with(
        17,
        "champion-load-17",
        winners,
    )
    type_def.callable.assert_not_called()


def test_recovered_backtest_rejects_stale_artifact_identity() -> None:
    """Auto-load must consume artifacts bound to this exact tracking run."""
    manager = _bare_manager()
    type_def = _type_def(MagicMock(), type_id="backtest_mstl", group="backtest")
    job = {"params": {"backtest_run_id": 53, "model_id": "mstl"}}

    with (
        patch(
            "common.services.job_registry.verify_backtest_artifact_identity",
            side_effect=RuntimeError("artifact belongs to backtest run 52"),
        ),
        patch("common.services.job_state._auto_load_backtest") as auto_load,
    ):
        with pytest.raises(RuntimeError, match="backtest run 52"):
            manager._finalize_recovered_job("job-53", type_def, job)

    auto_load.assert_not_called()


def test_backtest_metadata_is_atomically_bound_to_current_tracking_run(
    tmp_path,
) -> None:
    """A later run cannot auto-load an older run's shared artifact files."""
    from common.services.job_state import (
        record_backtest_artifact_identity,
        verify_backtest_artifact_identity,
    )

    metadata_path = tmp_path / "backtest" / "mstl" / "backtest_metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        '{"accuracy_at_execution_lag": {"wape": 0.2}}',
        encoding="utf-8",
    )

    with patch("common.services.job_state._DATA_DIR", tmp_path):
        governed_lineage = {
            "source_sales_batch_id": 301,
            "data_checksum": "a" * 64,
            "cluster_experiment_id": 35,
            "cluster_assignment_count": 13_968,
            "cluster_assignment_checksum": "b" * 64,
        }
        record_backtest_artifact_identity(
            "mstl",
            53,
            "job-53",
            governed_lineage=governed_lineage,
        )
        verify_backtest_artifact_identity("mstl", 53, "job-53")
        with pytest.raises(RuntimeError, match="expected 54"):
            verify_backtest_artifact_identity("mstl", 54, "job-54")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["governed_lineage"] == governed_lineage

    persisted = metadata_path.read_text(encoding="utf-8")
    assert '"backtest_run_id": 53' in persisted
    assert '"job_id": "job-53"' in persisted
