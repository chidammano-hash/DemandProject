"""Pure adapters mapping a JobManager pipeline to the integration chain shape (US17d).

An integration chain runs as a JobManager pipeline of ``load_domain`` steps:

    chain_id == pipeline_id
    step     == pipeline_step

submit_pipeline creates one job row at a time (the next step is submitted when
the previous one completes), so the *full* step plan is stashed in the first
step's params under ``__pipeline_plan`` — that lets ``GET /integration/chains``
render every step (and compute ``total_steps``) before later steps are
submitted, and lets a halted chain mark its downstream steps as cancelled,
matching the legacy IntegrationChainRunner semantics.

Hard rule: pure (dict in, dict out) — no DB, no network. Status values here are
the integration vocabulary (``completed`` already mapped to ``success`` by
``common/services/job_shape.py`` upstream).
"""
from __future__ import annotations

from typing import Any

# Stored in step 1's params. Starts with ``__pipeline`` so JobManager strips it
# from the params handed to the load_domain callable (see _execute_job).
PLAN_KEY = "__pipeline_plan"


def to_load_domain_steps(
    job_specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build ``submit_pipeline`` steps and the full plan from chain job specs.

    Returns ``(steps, plan)``. The plan is embedded in ``steps[0].params`` under
    :data:`PLAN_KEY` so it survives in ``job_history`` for the read adapters.
    """
    plan: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for idx, spec in enumerate(job_specs, start=1):
        domain = spec["domain"]
        mode = spec["mode"]
        slice_ = spec.get("slice")
        file = spec.get("file")
        plan.append({"step": idx, "domain": domain, "mode": mode,
                     "slice": slice_, "file": file})
        steps.append({
            "job_type": "load_domain",
            "params": {"domain": domain, "mode": mode, "slice": slice_, "file": file},
            "label": f"Load {domain} ({mode})",
        })
    if steps:
        steps[0]["params"][PLAN_KEY] = plan
    return steps, plan


def _real_failed_step(actual_by_step: dict[int, dict[str, Any]]) -> int | None:
    """Lowest step whose actual job row genuinely failed (not a cancel marker)."""
    for step in sorted(actual_by_step):
        if actual_by_step[step].get("status") == "failed":
            return step
    return None


def chain_jobs(
    chain_id: str,
    plan: list[dict[str, Any]] | None,
    actual_by_step: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge the plan with the actual per-step job rows into ordered ChainJobs.

    Steps that have a real job row use it; steps not yet submitted are rendered
    as ``queued`` placeholders with a synthetic but unique ``job_id`` (so the UI
    can key a list on it). If the chain halted at step N, steps after N are shown
    as cancelled (``failed`` with a "halted at step N" message), matching the
    legacy chain runner.
    """
    plan_by_step = {p["step"]: p for p in (plan or [])}
    failed_step = _real_failed_step(actual_by_step)
    steps = sorted(set(plan_by_step) | set(actual_by_step))
    jobs: list[dict[str, Any]] = []
    for step in steps:
        if step in actual_by_step:
            jobs.append({**actual_by_step[step], "step": step})
            continue
        p = plan_by_step.get(step, {})
        if failed_step is not None and step > failed_step:
            status = "failed"
            error = f"cancelled: chain halted at step {failed_step}"
        else:
            status = "queued"
            error = None
        jobs.append({
            "step": step,
            "job_id": f"{chain_id}#{step}",  # synthetic placeholder, unique
            "domain": p.get("domain", ""),
            "mode": p.get("mode", ""),
            "slice": p.get("slice"),
            "status": status,
            "rows_loaded": None,
            "rows_inserted": None,
            "rows_updated": None,
            "rows_deleted": None,
            "error_message": error,
            "started_at": None,
            "completed_at": None,
            "duration_ms": None,
        })
    return jobs


def chain_summary(
    chain_id: str,
    plan: list[dict[str, Any]] | None,
    actual_by_step: dict[int, dict[str, Any]],
    triggered_by: str = "api",
) -> dict[str, Any]:
    """Aggregate a chain's lifecycle envelope from its actual step rows."""
    total_steps = len(plan) if plan else (max(actual_by_step) if actual_by_step else 0)
    statuses = [j.get("status") for j in actual_by_step.values()]
    terminal = [s for s in statuses if s in ("success", "failed")]
    completed_steps = len(terminal)
    failed_step = _real_failed_step(actual_by_step)

    if failed_step is not None:
        status = "halted"
    elif total_steps > 0 and sum(1 for s in statuses if s == "success") >= total_steps:
        status = "success"
    elif any(s in ("running", "success") for s in statuses):
        status = "running"
    else:
        status = "queued"

    started = [j.get("started_at") for j in actual_by_step.values() if j.get("started_at")]
    started_at = min(started) if started else None
    completed_at = None
    duration_ms = None
    if status in ("success", "halted"):
        finished = [j.get("completed_at") for j in actual_by_step.values() if j.get("completed_at")]
        completed_at = max(finished) if finished else None

    return {
        "id": chain_id,
        "status": status,
        "total_steps": total_steps,
        "completed_steps": completed_steps,
        "failed_step": failed_step,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": duration_ms,
        "triggered_by": triggered_by,
    }
