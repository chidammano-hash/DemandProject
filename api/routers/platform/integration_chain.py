"""Integration chain endpoints — directory scan + sequential job-chain submission.

Backed by ``common.services.integration_scanner.scan_input_dir`` and
``common.services.integration_chain_runner.IntegrationChainRunner``.

The single-job submission path lives in ``api.routers.platform.integration``;
this module mirrors its conventions (router prefix, Pydantic v2 models,
``_get_runner``-style dependency, ``require_api_key`` on writes).

Endpoints:
- ``GET    /integration/scan``                 directory scan + proposed chain (read-only)
- ``POST   /integration/chains``               submit a multi-step chain (api key required)
- ``GET    /integration/chains``               list recent chains
- ``GET    /integration/chains/{chain_id}``    fetch a single chain with all jobs
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.core import _get_pool
from common.services.integration_chain_runner import IntegrationChainRunner
from common.services.integration_scanner import scan_input_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/integration", tags=["integration-chain"])

# Hardcoded set of domains the chain runner accepts. Kept in sync with
# api/routers/platform/integration.py KNOWN_DOMAINS — both lists must move
# together when a new domain is added to the ETL pipeline.
KNOWN_DOMAINS: list[str] = [
    "item", "location", "customer", "time", "sku", "sales", "forecast",
    "inventory", "customer_demand", "sourcing", "purchase_order",
]


# ---------------------------------------------------------------------------
# Pydantic v2 models — scan. ``extra='allow'`` mirrors the scanner's dict
# shape so new metadata (file hashes, byte counts, etc.) flows through to the
# UI without a coordinated router release.
# ---------------------------------------------------------------------------
class DomainChangeModel(BaseModel):
    """A single change detected by the directory scanner."""
    model_config = {"extra": "allow"}

    domain: str = Field(..., description="Domain name the change belongs to.")
    kind: str = Field(..., description="Change kind (e.g. 'added', 'modified', 'removed').")
    file: str | None = Field(default=None, description="Source file path, when applicable.")
    slice: str | None = Field(default=None, description="Partition slice (e.g. '2026-04'), when applicable.")


class ChainStepModel(BaseModel):
    """A single proposed step in the chain emitted by the scanner."""
    model_config = {"extra": "allow"}

    step: int | None = Field(default=None, description="1-based step index in the proposed chain.")
    domain: str = Field(..., description="Target domain to load at this step.")
    mode: str = Field(..., description="Load mode (onetime / delta / file).")
    slice: str | None = Field(default=None, description="Partition slice, when applicable.")
    file: str | None = Field(default=None, description="Explicit file path, when applicable.")


class ScanResponse(BaseModel):
    """Response body for ``GET /integration/scan``."""

    scanned_at: str = Field(..., description="ISO-8601 timestamp of when the scan ran.")
    changes: list[DomainChangeModel] = Field(
        default_factory=list,
        description="Per-domain changes detected against the last successful load.",
    )
    proposed_chain: list[ChainStepModel] = Field(
        default_factory=list,
        description="Ordered chain of jobs the runner suggests to bring the warehouse up to date.",
    )


# ---------------------------------------------------------------------------
# Pydantic v2 models — chain submission
# ---------------------------------------------------------------------------
class JobSpecModel(BaseModel):
    """One job within a submitted chain."""

    domain: str = Field(..., description="Target domain to load (must be in KNOWN_DOMAINS).")
    mode: Literal["onetime", "delta", "file"] = Field(
        ..., description="Load mode: 'onetime' full reload, 'delta' incremental, 'file' single partition.",
    )
    slice: str | None = Field(
        default=None,
        description="Partition slice (e.g. '2026-04'). Required for partitioned domains in 'file' mode.",
    )
    file: str | None = Field(default=None, description="Explicit file path to load (used in 'file' mode).")


class SubmitChainRequest(BaseModel):
    """Request body for ``POST /integration/chains``."""

    jobs: list[JobSpecModel] = Field(
        ...,
        min_length=1,
        description="Ordered list of jobs to run sequentially. Must contain at least one job.",
    )
    triggered_by: str | None = Field(
        default=None,
        description="Optional caller tag (e.g. 'ui', 'scheduler'). Defaults to 'api'.",
    )


class SubmittedJob(BaseModel):
    """One queued job summary returned by the chain submission endpoint."""
    model_config = {"extra": "allow"}

    job_id: str = Field(..., description="UUID of the queued job.")
    step: int = Field(..., description="1-based step index within the chain.")
    domain: str = Field(..., description="Target domain for this job.")
    mode: str = Field(..., description="Load mode for this job.")


class SubmitChainResponse(BaseModel):
    """Response body for ``POST /integration/chains``."""

    chain_id: str = Field(..., description="UUID of the queued chain.")
    jobs: list[SubmittedJob] = Field(
        default_factory=list, description="Per-step queued job summaries (in chain order).",
    )
    status: Literal["queued"] = Field(
        default="queued", description="Initial chain status — always 'queued' on accept.",
    )


# ---------------------------------------------------------------------------
# Pydantic v2 models — chain reads
# ChainSummary holds the lifecycle/status fields shared with ChainDetail; the
# detail model just adds ``jobs``. Inheriting saves duplicating ~9 fields with
# identical descriptions while keeping both names in the OpenAPI schema.
# ---------------------------------------------------------------------------
class ChainSummary(BaseModel):
    """Summary row for ``GET /integration/chains``."""
    model_config = {"extra": "allow"}

    id: str = Field(..., description="Chain UUID.")
    status: str = Field(..., description="Current status (queued / running / success / failed).")
    total_steps: int = Field(..., description="Total number of jobs in the chain.")
    completed_steps: int = Field(..., description="Number of jobs that have reached a terminal state.")
    failed_step: int | None = Field(default=None, description="1-based step index that failed, if any.")
    started_at: str | None = Field(default=None, description="ISO-8601 timestamp when the chain started running.")
    completed_at: str | None = Field(default=None, description="ISO-8601 timestamp when the chain finished.")
    duration_ms: int | None = Field(default=None, description="Total chain duration in milliseconds, when known.")
    triggered_by: str = Field(default="api", description="Source that triggered this chain (e.g. 'api', 'scheduler').")


class ChainListResponse(BaseModel):
    """Response body for ``GET /integration/chains``."""

    items: list[ChainSummary] = Field(default_factory=list, description="Recent chains (newest first).")


class ChainJob(BaseModel):
    """One job's detailed record inside a chain detail response."""
    model_config = {"extra": "allow"}

    step: int = Field(..., description="1-based step index in the chain.")
    job_id: str = Field(..., description="Job UUID.")
    domain: str = Field(..., description="Target domain for this job.")
    mode: str = Field(..., description="Load mode for this job.")
    slice: str | None = Field(default=None, description="Partition slice, when applicable.")
    status: str = Field(..., description="Job status (queued / running / success / failed).")
    rows_loaded: int | None = Field(default=None, description="Headline rows changed by this job.")
    rows_inserted: int | None = Field(default=None, description="Rows newly INSERTed by this job.")
    rows_updated: int | None = Field(default=None, description="Rows UPDATEd in place by this job.")
    rows_deleted: int | None = Field(default=None, description="Rows DELETEd by this job.")
    error_message: str | None = Field(default=None, description="Error message if the job failed; otherwise null.")
    started_at: str | None = Field(default=None, description="ISO-8601 timestamp when the job started running.")
    completed_at: str | None = Field(default=None, description="ISO-8601 timestamp when the job finished.")
    duration_ms: int | None = Field(default=None, description="Job duration in milliseconds, when known.")


class ChainDetail(ChainSummary):
    """Response body for ``GET /integration/chains/{chain_id}`` — summary + jobs."""

    jobs: list[ChainJob] = Field(default_factory=list, description="Detailed per-step jobs in chain order.")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def _get_chain_runner(pool=Depends(_get_pool)) -> IntegrationChainRunner:
    """Construct an IntegrationChainRunner bound to the shared connection pool."""
    return IntegrationChainRunner(pool)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/scan", response_model=ScanResponse)
def scan(pool=Depends(_get_pool)) -> ScanResponse:
    """Scan the input directory and return detected changes + a proposed chain.

    Read-only — no auth required. Delegates entirely to
    :func:`common.services.integration_scanner.scan_input_dir`.
    """
    payload: dict[str, Any] = scan_input_dir(pool) or {}
    return ScanResponse(
        scanned_at=str(payload.get("scanned_at", "")),
        changes=[DomainChangeModel(**c) for c in payload.get("changes", [])],
        proposed_chain=[ChainStepModel(**s) for s in payload.get("proposed_chain", [])],
    )


@router.post(
    "/chains",
    status_code=202,
    response_model=SubmitChainResponse,
    dependencies=[Depends(require_api_key)],
)
def submit_chain(
    req: SubmitChainRequest,
    runner: IntegrationChainRunner = Depends(_get_chain_runner),
) -> SubmitChainResponse:
    """Submit a multi-step integration chain.

    The runner enqueues every job atomically and returns a chain id that the
    UI can poll via ``GET /integration/chains/{chain_id}``. Domain-whitelist
    checks fail fast at the API boundary so the UI gets a clean 422 with the
    full known-domains list embedded in the message.
    """
    for idx, job in enumerate(req.jobs, start=1):
        if job.domain not in KNOWN_DOMAINS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown domain '{job.domain}' at step {idx}. Known domains: {KNOWN_DOMAINS}",
            )

    job_dicts = [job.model_dump() for job in req.jobs]
    try:
        result: dict[str, Any] = runner.submit_chain(
            jobs=job_dicts,
            triggered_by=req.triggered_by or "api",
        )
    except ValueError as exc:
        # Runner raises ValueError for malformed chains the API didn't catch
        # (e.g. partition slice missing for a partitioned domain). Surface as
        # 422 so the UI can render the message inline rather than a 500.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SubmitChainResponse(
        chain_id=str(result["chain_id"]),
        jobs=[SubmittedJob(**j) for j in result.get("jobs", [])],
        status="queued",
    )


@router.get("/chains", response_model=ChainListResponse)
def list_chains(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of chains to return (1-100)."),
    runner: IntegrationChainRunner = Depends(_get_chain_runner),
) -> ChainListResponse:
    """List recent integration chains, newest first."""
    items = runner.list_chains(limit=limit) or []
    return ChainListResponse(items=[ChainSummary(**item) for item in items])


@router.get("/chains/{chain_id}", response_model=ChainDetail)
def get_chain(
    chain_id: str,
    runner: IntegrationChainRunner = Depends(_get_chain_runner),
) -> ChainDetail:
    """Fetch a single chain with all per-step job details."""
    record = runner.get_chain(chain_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Chain '{chain_id}' not found")

    jobs = [ChainJob(**j) for j in (record.get("jobs", []) or [])]
    return ChainDetail(
        id=str(record["id"]),
        status=str(record.get("status", "queued")),
        total_steps=int(record.get("total_steps", len(jobs))),
        completed_steps=int(record.get("completed_steps", 0)),
        failed_step=record.get("failed_step"),
        started_at=record.get("started_at"),
        completed_at=record.get("completed_at"),
        duration_ms=record.get("duration_ms"),
        triggered_by=str(record.get("triggered_by", "api")),
        jobs=jobs,
    )
