"""Integration chain endpoints — directory scan + sequential job-chain submission.

Backed by ``common.services.integration_scanner.scan_input_dir`` and
``common.services.integration_chain_jobs.ChainJobRunner`` (US17d — chains on
the unified JobManager backend).

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
from common.ai.integration_scan.planner import run_scan_planner
from common.services.integration_chain_jobs import ChainJobRunner
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


class PlannerAnswerModel(BaseModel):
    """One answer returned from the UI's follow-up question loop."""

    question_id: str = Field(..., description="Stable question identifier from the planner response.")
    answer: str = Field(..., description="User response text.")


class PlannerQuestionModel(BaseModel):
    """One question the AI planner wants the user to answer."""

    id: str = Field(..., description="Stable question identifier.")
    prompt: str = Field(..., description="Question text shown to the user.")
    answer_type: Literal["text", "choice", "boolean"] = Field(
        default="text",
        description="How the UI should capture the answer.",
    )
    options: list[str] = Field(default_factory=list, description="Choice options for select-style questions.")
    required: bool = Field(default=True, description="Whether the question must be answered.")
    reason: str | None = Field(default=None, description="Why the planner asked the question.")


class PlannerEvidenceModel(BaseModel):
    """One evidence item the UI can render alongside the plan."""

    kind: Literal["scan", "job", "batch"] = Field(..., description="Evidence source bucket.")
    label: str = Field(..., description="Short evidence label.")
    value: str = Field(..., description="Human-readable evidence value.")


class ScanPlanRequest(BaseModel):
    """Request body for ``POST /integration/scan/plan``."""

    answers: list[PlannerAnswerModel] = Field(default_factory=list, description="Optional follow-up answers from the user.")


class ScanPlanResponse(ScanResponse):
    """Response body for ``POST /integration/scan/plan``."""

    plan_id: str = Field(..., description="Stable identifier for this planning turn.")
    provider: str = Field(..., description="Planner provider used for this turn (e.g. ollama/openai).")
    model: str = Field(..., description="Model id used for the planning turn.")
    status: Literal["questions", "planned", "fallback"] = Field(
        ..., description="Whether the planner needs more information or has a final sequence."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Planner confidence in the returned sequence.")
    explanation: str = Field(..., description="Short rationale for the recommendation.")
    risk_flags: list[str] = Field(default_factory=list, description="Ambiguities or cautions to surface in the UI.")
    questions: list[PlannerQuestionModel] = Field(default_factory=list, description="Optional clarifying questions.")
    recommended_chain: list[ChainStepModel] = Field(
        default_factory=list,
        description="Final AI-recommended execution chain (or deterministic fallback if the model is unavailable).",
    )
    evidence: list[PlannerEvidenceModel] = Field(
        default_factory=list,
        description="Supporting scan / job / batch facts used to justify the planner output.",
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
def _get_chain_runner() -> ChainJobRunner:
    """Construct a ChainJobRunner bound to the shared connection pool.

    US17d: chains run as JobManager pipelines of ``load_domain`` steps; this
    runner submits/reads them (with legacy ``integration_chain`` read-fallback).
    Resolves the pool via direct ``_get_pool()`` call rather than
    ``Depends(_get_pool)`` (per CLAUDE.md "DB & API Patterns" — FastAPI's
    inspection of MagicMock signatures raises 422 in tests).
    """
    return ChainJobRunner(_get_pool())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/scan", response_model=ScanResponse)
def scan() -> ScanResponse:
    """Scan the input directory and return detected changes + a proposed chain.

    Read-only — no auth required. Delegates entirely to
    :func:`common.services.integration_scanner.scan_input_dir`. Resolves the
    pool via direct ``_get_pool()`` call rather than ``Depends(_get_pool)``
    (per CLAUDE.md "DB & API Patterns").
    """
    payload: dict[str, Any] = scan_input_dir(_get_pool()) or {}
    return ScanResponse(
        scanned_at=str(payload.get("scanned_at", "")),
        changes=[DomainChangeModel(**c) for c in payload.get("changes", [])],
        proposed_chain=[ChainStepModel(**s) for s in payload.get("proposed_chain", [])],
    )


@router.post(
    "/scan/plan",
    response_model=ScanPlanResponse,
    dependencies=[Depends(require_api_key)],
)
def scan_plan(req: ScanPlanRequest) -> ScanPlanResponse:
    """Run the deterministic scan, then ask the AI planner for the safest sequence.

    The planner rescan happens server-side so the UI can simply POST user
    answers back to the same endpoint and receive a fresh recommendation.
    """
    payload: dict[str, Any] = run_scan_planner(
        _get_pool(),
        answers=[a.model_dump() for a in req.answers],
    ) or {}
    return ScanPlanResponse(
        scanned_at=str(payload.get("scanned_at", "")),
        changes=[DomainChangeModel(**c) for c in payload.get("changes", [])],
        proposed_chain=[ChainStepModel(**s) for s in payload.get("proposed_chain", [])],
        plan_id=str(payload.get("plan_id", "")),
        provider=str(payload.get("provider", "")),
        model=str(payload.get("model", "")),
        status=str(payload.get("status", "fallback")),
        confidence=float(payload.get("confidence", 0.0)),
        explanation=str(payload.get("explanation", "")),
        risk_flags=[str(v) for v in payload.get("risk_flags", [])],
        questions=[PlannerQuestionModel(**q) for q in payload.get("questions", [])],
        recommended_chain=[ChainStepModel(**s) for s in payload.get("recommended_chain", [])],
        evidence=[PlannerEvidenceModel(**e) for e in payload.get("evidence", [])],
    )


@router.post(
    "/chains",
    status_code=202,
    response_model=SubmitChainResponse,
    dependencies=[Depends(require_api_key)],
)
def submit_chain(
    req: SubmitChainRequest,
    runner: ChainJobRunner = Depends(_get_chain_runner),
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
    runner: ChainJobRunner = Depends(_get_chain_runner),
) -> ChainListResponse:
    """List recent integration chains, newest first."""
    items = runner.list_chains(limit=limit) or []
    return ChainListResponse(items=[ChainSummary(**item) for item in items])


@router.get("/chains/{chain_id}", response_model=ChainDetail)
def get_chain(
    chain_id: str,
    runner: ChainJobRunner = Depends(_get_chain_runner),
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
