"""US17d — pure adapters mapping a JobManager pipeline to the integration chain
shape. A chain is a pipeline of load_domain steps: chain_id == pipeline_id,
step == pipeline_step. These functions are PURE (dict in, dict out) — no DB.
"""
from __future__ import annotations

import common.services.chain_shape as chain_shape
from common.services.chain_shape import (
    PLAN_KEY,
    chain_jobs,
    chain_summary,
    to_load_domain_steps,
)


def test_to_load_domain_steps_builds_pipeline_and_plan() -> None:
    specs = [
        {"domain": "item", "mode": "onetime"},
        {"domain": "sales", "mode": "delta", "slice": "2026-04"},
    ]
    steps, plan = to_load_domain_steps(specs)
    assert [s["job_type"] for s in steps] == ["load_domain", "load_domain"]
    assert steps[0]["params"]["domain"] == "item"
    assert steps[1]["params"]["slice"] == "2026-04"
    # full plan is carried in step 1 so the chain renders before later steps run
    assert steps[0]["params"][PLAN_KEY] == plan
    assert [p["step"] for p in plan] == [1, 2]
    assert plan[1]["domain"] == "sales"


def _actual(step, status, **kw):
    base = {
        "step": step, "job_id": f"job-{step}", "domain": "d", "mode": "delta",
        "slice": None, "status": status, "rows_loaded": 0, "rows_inserted": None,
        "rows_updated": None, "rows_deleted": None, "error_message": None,
        "started_at": None, "completed_at": None, "duration_ms": None,
    }
    base.update(kw)
    return base


def test_chain_jobs_merges_plan_with_actuals() -> None:
    plan = [{"step": 1, "domain": "item", "mode": "onetime", "slice": None},
            {"step": 2, "domain": "sales", "mode": "delta", "slice": None}]
    actual = {1: _actual(1, "success", domain="item", rows_loaded=10)}
    jobs = chain_jobs("pipe_x", plan, actual)
    assert len(jobs) == 2
    assert jobs[0]["status"] == "success"
    assert jobs[0]["job_id"] == "job-1"
    # step 2 not yet submitted -> pending, synthetic unique id (for React keys)
    assert jobs[1]["status"] == "queued"
    assert jobs[1]["job_id"] == "pipe_x#2"
    assert jobs[1]["domain"] == "sales"


def test_chain_jobs_halt_cancels_downstream_steps() -> None:
    plan = [{"step": 1, "domain": "item", "mode": "onetime", "slice": None},
            {"step": 2, "domain": "sales", "mode": "delta", "slice": None},
            {"step": 3, "domain": "forecast", "mode": "delta", "slice": None}]
    actual = {
        1: _actual(1, "success"),
        2: _actual(2, "failed", error_message="boom"),
    }
    jobs = chain_jobs("pipe_y", plan, actual)
    assert jobs[1]["status"] == "failed"
    assert jobs[1]["error_message"] == "boom"
    # step after the failure is cancelled (matches legacy halt semantics)
    assert jobs[2]["status"] == "failed"
    assert "halted at step 2" in jobs[2]["error_message"]


def test_chain_summary_success() -> None:
    plan = [{"step": 1, "domain": "item", "mode": "onetime", "slice": None},
            {"step": 2, "domain": "sales", "mode": "delta", "slice": None}]
    actual = {
        1: _actual(1, "success", started_at="2026-06-01T10:00:00", completed_at="2026-06-01T10:01:00"),
        2: _actual(2, "success", started_at="2026-06-01T10:01:00", completed_at="2026-06-01T10:02:00"),
    }
    s = chain_summary("pipe_z", plan, actual, triggered_by="ui")
    assert s["id"] == "pipe_z"
    assert s["status"] == "success"
    assert s["total_steps"] == 2
    assert s["completed_steps"] == 2
    assert s["failed_step"] is None
    assert s["triggered_by"] == "ui"


def test_chain_summary_halted_reports_failed_step() -> None:
    plan = [{"step": 1, "domain": "item", "mode": "onetime", "slice": None},
            {"step": 2, "domain": "sales", "mode": "delta", "slice": None}]
    actual = {1: _actual(1, "success"), 2: _actual(2, "failed", error_message="x")}
    s = chain_summary("pipe_h", plan, actual)
    assert s["status"] == "halted"
    assert s["failed_step"] == 2
    assert s["completed_steps"] == 2  # both reached a terminal state


def test_chain_summary_running_and_queued() -> None:
    plan = [{"step": 1, "domain": "item", "mode": "onetime", "slice": None},
            {"step": 2, "domain": "sales", "mode": "delta", "slice": None}]
    running = chain_summary("p1", plan, {1: _actual(1, "running")})
    assert running["status"] == "running"
    assert running["completed_steps"] == 0
    queued = chain_summary("p2", plan, {})
    assert queued["status"] == "queued"
    assert queued["total_steps"] == 2


def test_pure_no_db() -> None:
    assert not hasattr(chain_shape, "psycopg")
    assert not hasattr(chain_shape, "_get_conn")
