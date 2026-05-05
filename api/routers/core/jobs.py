"""Job scheduler API endpoints (Feature 39).

Provides endpoints to submit, monitor, list, cancel, and delete
long-running background jobs (clustering, backtesting, seasonality, etc.).

Also supports:
- Cron/interval scheduling for recurring automation
- Job pipelines (sequential chaining)
- Dashboard statistics
- Foundation for agentic AI automation
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.auth import require_api_key

router = APIRouter(tags=["jobs"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SubmitJobRequest(BaseModel):
    job_type: str
    params: dict[str, Any] = {}
    label: str | None = None
    max_retries: int = 0


class ScheduleJobRequest(BaseModel):
    job_type: str
    params: dict[str, Any] = {}
    label: str | None = None
    cron: str | None = None
    interval_minutes: int | None = None


class PipelineStep(BaseModel):
    job_type: str
    params: dict[str, Any] = {}
    label: str | None = None


class SubmitPipelineRequest(BaseModel):
    steps: list[PipelineStep]
    label: str = "Pipeline"


# ---------------------------------------------------------------------------
# Lazy-init JobManager to avoid import-time DB access
# ---------------------------------------------------------------------------
_manager = None


def _get_manager():
    global _manager
    if _manager is None:
        from common.services.job_registry import JobManager
        _manager = JobManager()
    return _manager


# ---------------------------------------------------------------------------
# Endpoints — Core CRUD
# ---------------------------------------------------------------------------

@router.get("/jobs/types")
def list_job_types():
    """List all available job types with metadata."""
    mgr = _get_manager()
    return {"types": mgr.get_types()}


@router.get("/jobs/stats")
def get_job_stats():
    """Get aggregate job statistics for the dashboard."""
    mgr = _get_manager()
    return mgr.get_stats()


@router.post("/jobs", dependencies=[Depends(require_api_key)])
def submit_job(req: SubmitJobRequest):
    """Submit a new background job. Returns 202 with job_id.

    Jobs are dispatched to APScheduler's managed thread pool for execution.
    Per-group concurrency control ensures only one job runs per group at a time.
    """
    mgr = _get_manager()

    from common.services.job_registry import JOB_TYPE_REGISTRY
    if req.job_type not in JOB_TYPE_REGISTRY:
        raise HTTPException(status_code=422, detail=f"Unknown job type: {req.job_type}")

    job_id = mgr.submit_job(
        req.job_type, req.params, req.label,
        triggered_by="api",
        max_retries=req.max_retries,
    )

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued"},
    )


@router.get("/jobs/active")
def get_active_jobs():
    """Get all currently running/queued jobs."""
    mgr = _get_manager()
    return {"jobs": mgr.get_active_jobs()}


@router.get("/jobs")
def list_jobs(
    status: str | None = Query(default=None),
    job_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List jobs with optional filters and pagination."""
    mgr = _get_manager()
    jobs, total = mgr.list_jobs(status=status, job_type=job_type, limit=limit, offset=offset)
    return {"jobs": jobs, "total": total, "limit": limit, "offset": offset}


# ---------------------------------------------------------------------------
# Endpoints — Scheduling (recurring jobs)
# IMPORTANT: These must be registered BEFORE /jobs/{job_id} so that
# FastAPI does not match "schedules" or "schedule" as a job_id.
# ---------------------------------------------------------------------------

@router.post("/jobs/schedule", dependencies=[Depends(require_api_key)])
def schedule_recurring_job(req: ScheduleJobRequest):
    """Schedule a recurring job via cron expression or interval.

    This endpoint enables agentic AI automation: schedule jobs to run
    on a regular cadence (e.g., daily backtest, weekly clustering refresh).
    """
    mgr = _get_manager()

    from common.services.job_registry import JOB_TYPE_REGISTRY
    if req.job_type not in JOB_TYPE_REGISTRY:
        raise HTTPException(status_code=422, detail=f"Unknown job type: {req.job_type}")
    if not req.cron and not req.interval_minutes:
        raise HTTPException(status_code=422, detail="Must specify either cron or interval_minutes")

    try:
        schedule_id = mgr.schedule_recurring(
            req.job_type, req.params, req.label,
            cron=req.cron, interval_minutes=req.interval_minutes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))  # controlled validation message
    except RuntimeError:
        raise HTTPException(status_code=409, detail="Schedule conflict or scheduler unavailable.")

    return JSONResponse(
        status_code=201,
        content={"schedule_id": schedule_id, "status": "active"},
    )


@router.get("/jobs/schedules")
def list_schedules():
    """List all active recurring schedules."""
    mgr = _get_manager()
    return {"schedules": mgr.list_schedules()}


@router.delete("/jobs/schedules/{schedule_id}", dependencies=[Depends(require_api_key)])
def remove_schedule(schedule_id: str):
    """Remove a recurring schedule."""
    mgr = _get_manager()
    success = mgr.remove_schedule(schedule_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Schedule '{schedule_id}' not found")
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Endpoints — Pipelines (job chaining)
# IMPORTANT: Must be registered BEFORE /jobs/{job_id} to avoid path conflict.
# ---------------------------------------------------------------------------

@router.post("/jobs/pipeline", dependencies=[Depends(require_api_key)])
def submit_pipeline(req: SubmitPipelineRequest):
    """Submit a pipeline of chained jobs to run sequentially.

    Jobs run one after another. If a step fails, the pipeline stops.
    This enables complex automation workflows for agentic AI.
    """
    mgr = _get_manager()

    from common.services.job_registry import JOB_TYPE_REGISTRY
    for step in req.steps:
        if step.job_type not in JOB_TYPE_REGISTRY:
            raise HTTPException(status_code=422, detail=f"Unknown job type: {step.job_type}")

    steps_dicts = [
        {"job_type": s.job_type, "params": s.params, "label": s.label}
        for s in req.steps
    ]

    pipeline_id = mgr.submit_pipeline(
        steps=steps_dicts,
        label=req.label,
        triggered_by="api",
    )

    return JSONResponse(
        status_code=202,
        content={"pipeline_id": pipeline_id, "status": "running", "steps": len(req.steps)},
    )


# ---------------------------------------------------------------------------
# Endpoints — Single job CRUD (parameterized {job_id} routes)
# IMPORTANT: These MUST come after all /jobs/<literal> routes above,
# otherwise FastAPI would match "schedules", "pipeline", etc. as a job_id.
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}/logs")
def get_job_logs(
    job_id: str,
    offset: int = Query(default=0, ge=0),
):
    """Get the persistent execution log for a job.

    Supports incremental fetching via `offset` — the frontend can poll for
    new log content by passing the last known total_length as offset.
    """
    mgr = _get_manager()
    job = mgr.get_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    full_log = mgr.get_job_logs(job_id)
    return {
        "job_id": job_id,
        "log": full_log[offset:],
        "total_length": len(full_log),
        "offset": offset,
    }


@router.get("/jobs/{job_id}")
def get_job_detail(job_id: str):
    """Get a single job's status and result."""
    mgr = _get_manager()
    job = mgr.get_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@router.post("/jobs/{job_id}/cancel", dependencies=[Depends(require_api_key)])
def cancel_job(job_id: str):
    """Cancel a running or queued job."""
    mgr = _get_manager()
    success = mgr.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found or not cancellable")
    return {"job_id": job_id, "status": "cancelled"}


@router.delete("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def delete_job(job_id: str):
    """Delete a completed/failed/cancelled job from history."""
    mgr = _get_manager()
    success = mgr.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found or still active")
    return {"deleted": True}


@router.delete("/jobs", dependencies=[Depends(require_api_key)])
def purge_jobs(
    older_than_hours: int | None = Query(
        default=None, ge=0,
        description="Only purge jobs whose submitted_at is older than N hours. Omit for no age filter.",
    ),
    status: str | None = Query(
        default=None,
        description="Restrict to one status (completed / failed / cancelled). Omit for all terminal statuses.",
    ),
    job_type: str | None = Query(
        default=None,
        description="Restrict to one job_type. Omit for all types.",
    ),
):
    """Bulk-delete terminal jobs. Running/queued jobs are always preserved."""
    from common.services.job_registry import JobManager
    deleted = JobManager.purge_history(
        older_than_hours=older_than_hours,
        status=status,
        job_type=job_type,
    )
    return {"deleted": deleted}
