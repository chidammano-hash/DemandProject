"""Customer-analytics recalculation endpoint.

Backs the "Recalculate" button on the Customer Analytics tab. Submits a
background job that refreshes the materialized views the tab reads from, off
the request thread (the refresh exceeds the request ``statement_timeout`` at
40x scale). Returns a 202 + job_id pollable via ``GET /jobs/{job_id}``.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.auth import require_api_key

router = APIRouter(tags=["customer-analytics"])


@router.post("/customer-analytics/recalculate", dependencies=[Depends(require_api_key)])
def recalculate_customer_analytics():
    """Submit a background job to refresh the customer-analytics MVs.

    Returns 202 with a job_id that can be polled via GET /jobs/{job_id}.
    """
    from common.services.job_registry import JobManager

    mgr = JobManager()
    job_id = mgr.submit_job(
        "refresh_customer_analytics",
        {},
        label="Recalculate Customer Analytics",
        triggered_by="api",
    )

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued"},
    )
