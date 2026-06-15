"""Platform integration endpoints — unified ETL job submission and monitoring.

Backed by :class:`common.services.integration_runner.IntegrationRunner`.

Endpoints:
- ``POST   /integration/jobs``         submit a new load job (api key required)
- ``GET    /integration/jobs``         list recent jobs (optionally per domain)
- ``GET    /integration/jobs/{id}``    fetch a single job
- ``GET    /integration/domains``      describe known domains + partition info
- ``GET    /integration/health``       runner health snapshot
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.core import _get_pool
from common.core.domain_partition import (
    PARTITION_SPECS,
    get_partition,
    is_partitioned,
)
from common.services.integration_runner import IntegrationRunner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/integration", tags=["integration"])

# Restrict client-supplied --file paths to this allowlisted directory tree.
# Override with INTEGRATION_DATA_ROOT for tests / non-default layouts.
_DATA_ROOT = Path(
    os.environ.get(
        "INTEGRATION_DATA_ROOT",
        str(Path(__file__).resolve().parents[3] / "data"),
    )
).resolve()


def _validate_file_path(path: str) -> None:
    """Reject file paths that escape the project data directory.

    Why: ``file`` flows from the HTTP request body straight into a subprocess
    argv. Even though we never use ``shell=True``, an attacker could otherwise
    point the loader at arbitrary readable files on disk. Resolving + checking
    ``is_relative_to`` blocks ``../`` traversal and absolute paths outside
    ``_DATA_ROOT``.
    """
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_relative_to(_DATA_ROOT):
        raise HTTPException(
            status_code=422,
            detail=f"file path must be inside {_DATA_ROOT}; got {candidate}",
        )

# Hardcoded set of domains the integration runner accepts.  Kept in sync with
# CLAUDE.md "Domain-Driven Generic Design" + the ETL pipeline domain order.
KNOWN_DOMAINS: list[str] = [
    "item",
    "location",
    "customer",
    "time",
    "sku",
    "sales",
    "forecast",
    "inventory",
    "customer_demand",
    "sourcing",
    "purchase_order",
]


# ---------------------------------------------------------------------------
# Pydantic v2 models
# ---------------------------------------------------------------------------
class SubmitJobRequest(BaseModel):
    """Request body for ``POST /integration/jobs``."""

    domain: str = Field(..., description="Target domain to load (must be in KNOWN_DOMAINS).")
    mode: Literal["onetime", "delta", "file"] = Field(
        ..., description="Load mode: 'onetime' full reload, 'delta' incremental, 'file' single file/partition."
    )
    slice: str | None = Field(
        default=None,
        description="Partition slice (e.g. '2026-04'). Required for partitioned domains in 'file' mode without an explicit file.",
    )
    file: str | None = Field(default=None, description="Explicit file path to load (used in 'file' mode).")
    confirm_destructive: bool = Field(
        default=False,
        description="Must be true to run 'onetime' on a domain whose TRUNCATE would CASCADE to fact tables.",
    )
    reindex: bool = Field(
        default=False,
        description="Run REINDEX TABLE after a successful upsert (defragments indexes; slow — opt-in for large bulk loads).",
    )
    triggered_by: str | None = Field(
        default=None,
        max_length=32,
        pattern=r"^[a-z0-9_-]{1,32}$",
        description="Optional caller tag (e.g. 'ui', 'scheduler', 'test', 'dev_test', 'ci'). Defaults to 'api'.",
    )


class SubmitJobResponse(BaseModel):
    """Response body for ``POST /integration/jobs``."""

    job_id: str = Field(..., description="UUID of the queued integration job.")
    status: Literal["queued"] = Field(default="queued", description="Initial job status — always 'queued' on accept.")


class PipelineRunRequest(BaseModel):
    """Request body for ``POST /integration/pipeline`` (whole-pipeline run)."""

    mode: Literal["full", "refresh"] = Field(
        default="refresh",
        description="'full' = reload everything; 'refresh' = change-detected incremental load.",
    )
    domains: list[str] | None = Field(
        default=None,
        description="Optional subset of domains to run (default: all). Each must be in KNOWN_DOMAINS.",
    )
    parallel: bool = Field(
        default=False,
        description="Parallelize normalize/load/MV refresh (full mode).",
    )


class PipelineRunResponse(BaseModel):
    """Response body for ``POST /integration/pipeline``."""

    job_id: str = Field(..., description="ID of the queued etl_pipeline job (poll via /jobs/{id}).")
    mode: Literal["full", "refresh"] = Field(..., description="Mode the pipeline was started in.")
    status: Literal["queued"] = Field(default="queued", description="Initial status — always 'queued'.")


class Job(BaseModel):
    """Integration job record returned by the runner."""

    id: str = Field(..., description="Job UUID.")
    domain: str = Field(..., description="Target domain for this load job.")
    mode: str = Field(..., description="Load mode (onetime / delta / file).")
    slice: str | None = Field(default=None, description="Partition slice, if applicable.")
    file_path: str | None = Field(default=None, description="Explicit file path if provided at submission.")
    status: str = Field(..., description="Current status (queued / running / success / failed).")
    rows_loaded: int = Field(default=0, description="Headline rows changed (inserted + updated for delta; full count for onetime).")
    rows_inserted: int | None = Field(default=None, description="Rows newly INSERTed by this job (delta path; null for legacy paths).")
    rows_updated: int | None = Field(default=None, description="Rows UPDATEd in place by this job (delta path).")
    rows_deleted: int | None = Field(default=None, description="Rows DELETEd by this job (file-slice / onetime).")
    error_message: str | None = Field(default=None, description="Error message if the job failed; otherwise null.")
    started_at: str | None = Field(default=None, description="ISO-8601 timestamp when the job started running.")
    completed_at: str | None = Field(default=None, description="ISO-8601 timestamp when the job reached a terminal state.")
    duration_ms: int | None = Field(default=None, description="Total job duration in milliseconds, when known.")
    triggered_by: str = Field(default="api", description="Source that triggered this job (e.g. 'api', 'scheduler').")


class JobListResponse(BaseModel):
    """Response body for ``GET /integration/jobs``."""

    items: list[Job] = Field(default_factory=list, description="Recent integration jobs (newest first).")


class DomainInfo(BaseModel):
    """Metadata about a known integration domain."""

    name: str = Field(..., description="Domain name.")
    partitioned: bool = Field(..., description="Whether the domain uses time-based partitioning.")
    partition_format: str | None = Field(
        default=None, description="strftime-style partition format (e.g. '%Y-%m'); null if not partitioned."
    )
    partition_field: str | None = Field(
        default=None, description="Column used to derive the partition key; null if not partitioned."
    )
    onetime_cascades: bool = Field(
        default=False,
        description="True when TRUNCATE on this domain's table would CASCADE to fact tables — onetime mode is destructive.",
    )
    cascade_targets: list[str] = Field(
        default_factory=list,
        description="Tables that would be wiped if onetime is run on this domain (empty when safe).",
    )


class DomainListResponse(BaseModel):
    """Response body for ``GET /integration/domains``."""

    items: list[DomainInfo] = Field(
        default_factory=list, description="Known integration domains with partitioning metadata."
    )


class HealthStatus(BaseModel):
    """Response body for ``GET /integration/health``.

    Mirrors ``IntegrationRunner.health()`` — kept permissive (extra='allow')
    so runner-side additions surface to the UI without router changes.
    """

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def _get_runner() -> IntegrationRunner:
    """Construct an IntegrationRunner bound to the shared connection pool.

    Resolves the pool by calling ``_get_pool()`` directly rather than via
    ``Depends(_get_pool)`` — FastAPI inspects MagicMock signatures of injected
    dependencies and raises 422 in tests (per CLAUDE.md "DB & API Patterns").
    """
    return IntegrationRunner(_get_pool())


def _cascade_map(pool: Any) -> dict[str, list[str]]:
    """Map domain name -> list of dependent tables that onetime would orphan.

    Two layers of protection:
      1. **Enforced FKs** — pg_constraint join finds tables that PG would
         CASCADE-truncate (e.g., fact_sales_monthly -> dim_item).
      2. **Dimension safety net** — any dim_* table with no enforced FK still
         has logical dependents (facts reference it by id without a constraint).
         We populate cascade_targets with the actual fact_* tables in the DB so
         the user sees concretely what would break.

    Returns {} on DB error so the UI degrades gracefully.
    """
    try:
        from common.core.domain_specs import get_spec
    except ImportError:
        return {}
    domain_to_table: dict[str, str] = {}
    for name in KNOWN_DOMAINS:
        try:
            domain_to_table[name] = get_spec(name).table
        except (KeyError, ValueError, AttributeError):
            continue
    if not domain_to_table:
        return {}
    out: dict[str, list[str]] = {d: [] for d in domain_to_table}
    fk_sql = """
        SELECT confrelid::regclass::text AS parent,
               conrelid::regclass::text  AS child
        FROM pg_constraint
        WHERE contype = 'f'
          AND confrelid::regclass::text = ANY(%s)
    """
    facts_sql = """
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public' AND tablename LIKE 'fact_%'
        ORDER BY tablename
    """
    try:
        import psycopg
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(fk_sql, (list(domain_to_table.values()),))
            fk_rows = cur.fetchall()
            cur.execute(facts_sql)
            all_facts = [r[0] for r in cur.fetchall()]
    except (psycopg.Error, OSError) as exc:
        logger.warning("cascade_map probe failed: %s", exc)
        return {}
    table_to_domain = {v: k for k, v in domain_to_table.items()}
    for parent, child in fk_rows:
        domain = table_to_domain.get(parent)
        if domain and child not in out[domain]:
            out[domain].append(child)
    # Dimension safety net: dims without enforced FKs still have logical dependents
    # (facts reference them by id without constraints). Cap the displayed list so the
    # error message stays readable — full warehouse can have 100+ fact tables.
    _DISPLAY_CAP = 5
    for domain, table in domain_to_table.items():
        if table.startswith("dim_") and not out[domain]:
            facts = [f for f in all_facts if f != table and not _is_partition(f)]
            if len(facts) > _DISPLAY_CAP:
                out[domain] = facts[:_DISPLAY_CAP] + [f"…and {len(facts) - _DISPLAY_CAP} more"]
            else:
                out[domain] = facts
    return out


def _is_partition(table: str) -> bool:
    """Filter out monthly partition tables (e.g. fact_inventory_snapshot_2026_03).

    Including partitions in cascade messages is noisy — the parent table already
    represents the dependency. Heuristic: name ends with _YYYY_MM or _default.
    """
    import re
    return bool(re.search(r"_(\d{4}_\d{2}|default)$", table))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post(
    "/jobs",
    status_code=202,
    response_model=SubmitJobResponse,
    dependencies=[Depends(require_api_key)],
)
def submit_job(
    req: SubmitJobRequest,
    runner: IntegrationRunner = Depends(_get_runner),
) -> SubmitJobResponse:
    """Submit a new integration job for the given domain."""
    if req.domain not in KNOWN_DOMAINS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown domain '{req.domain}'. Known domains: {KNOWN_DOMAINS}",
        )

    if (
        req.mode == "file"
        and is_partitioned(req.domain)
        and not req.slice
        and not req.file
    ):
        raise HTTPException(
            status_code=422,
            detail="slice required for partitioned domain in file mode",
        )

    if req.file is not None:
        _validate_file_path(req.file)

    # Cascade guard: onetime + file modes still go through the destructive
    # loader path (TRUNCATE...CASCADE) for non-partitioned domains. Delta is
    # now safe — it uses the dispatcher's _safe_upsert (COPY + ON CONFLICT),
    # which never truncates the target.
    if req.mode in ("onetime", "file") and not req.confirm_destructive:
        targets = _cascade_map(runner.pool).get(req.domain, [])
        if targets:
            display = targets[:5]
            more = "" if len(targets) <= 5 else f" (+{len(targets)-5} more)"
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{req.mode} on '{req.domain}' would TRUNCATE...CASCADE and wipe "
                    f"downstream tables: {display}{more}. "
                    "Use mode=delta (safe upsert), or re-submit with "
                    "confirm_destructive=true to proceed."
                ),
            )

    job_id = runner.submit(
        domain=req.domain,
        mode=req.mode,
        slice=req.slice,
        file=req.file,
        triggered_by=req.triggered_by or "api",
        reindex=req.reindex,
    )
    return SubmitJobResponse(job_id=job_id, status="queued")


@router.post(
    "/pipeline",
    status_code=202,
    response_model=PipelineRunResponse,
    dependencies=[Depends(require_api_key)],
)
def run_pipeline(req: PipelineRunRequest) -> PipelineRunResponse:
    """Run the whole ingestion pipeline (full reload or incremental refresh).

    Submits the managed ``etl_pipeline`` job (JobManager); poll status/logs via
    the unified ``/jobs/{id}`` endpoints.
    """
    if req.domains:
        unknown = [d for d in req.domains if d not in KNOWN_DOMAINS]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown domains: {unknown}. Known domains: {KNOWN_DOMAINS}",
            )

    from common.services.job_registry import JobManager
    try:
        job_id = JobManager().submit_job(
            "etl_pipeline",
            params={"mode": req.mode, "domains": req.domains, "parallel": req.parallel},
            label=f"ETL pipeline ({req.mode})",
            triggered_by="ui",
        )
    except ValueError as exc:
        logger.warning("etl_pipeline submit rejected: %s", exc)
        raise HTTPException(status_code=422, detail="invalid pipeline job request") from exc
    except (psycopg.Error, OSError):
        logger.exception("failed to submit etl_pipeline job")
        raise HTTPException(status_code=500, detail="could not start pipeline") from None
    return PipelineRunResponse(job_id=job_id, mode=req.mode, status="queued")


@router.get("/jobs", response_model=JobListResponse)
def list_jobs(
    domain: str | None = Query(
        default=None, description="Optional domain filter."
    ),
    limit: int = Query(
        default=50, ge=1, le=200, description="Maximum number of jobs to return."
    ),
    runner: IntegrationRunner = Depends(_get_runner),
) -> JobListResponse:
    """List recent integration jobs, newest first."""
    if domain is not None and domain not in KNOWN_DOMAINS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown domain '{domain}'. Known domains: {KNOWN_DOMAINS}",
        )
    items = runner.list(domain=domain, limit=limit)
    return JobListResponse(items=[Job(**item) for item in items])


@router.get("/jobs/{job_id}", response_model=Job)
def get_job(
    job_id: str,
    runner: IntegrationRunner = Depends(_get_runner),
) -> Job:
    """Fetch a single integration job by id."""
    record = runner.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return Job(**record)


class PurgeResponse(BaseModel):
    deleted: int = Field(..., description="Number of integration_job rows deleted.")


@router.delete(
    "/jobs",
    response_model=PurgeResponse,
    dependencies=[Depends(require_api_key)],
)
def purge_jobs(
    older_than_hours: int | None = Query(
        default=None,
        ge=0,
        description="Delete only jobs whose started_at is older than N hours. Omit to ignore the age filter.",
    ),
    status: str | None = Query(
        default=None,
        description="Restrict purge to jobs with this status (success, failed, skipped). Omit for all terminal statuses.",
    ),
    domain: str | None = Query(
        default=None, description="Restrict purge to a single domain."
    ),
    runner: IntegrationRunner = Depends(_get_runner),
) -> PurgeResponse:
    """Delete integration_job rows. NEVER deletes queued/running jobs."""
    if domain is not None and domain not in KNOWN_DOMAINS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown domain '{domain}'. Known domains: {KNOWN_DOMAINS}",
        )
    statuses = [status] if status else None
    deleted = runner.purge(
        older_than_hours=older_than_hours,
        statuses=statuses,
        domain=domain,
        keep_running=True,
    )
    return PurgeResponse(deleted=deleted)


@router.get("/domains", response_model=DomainListResponse)
def list_domains() -> DomainListResponse:
    """List known integration domains with partitioning + cascade-risk metadata.

    Resolves the pool via direct ``_get_pool()`` call rather than
    ``Depends(_get_pool)`` (per CLAUDE.md "DB & API Patterns" — FastAPI's
    inspection of MagicMock signatures raises 422 in tests).
    """
    _ = PARTITION_SPECS  # surface import errors at route registration
    pool = _get_pool()
    cascades = _cascade_map(pool)
    infos: list[DomainInfo] = []
    for name in KNOWN_DOMAINS:
        partitioned = is_partitioned(name)
        spec = get_partition(name) if partitioned else None
        targets = cascades.get(name, [])
        infos.append(
            DomainInfo(
                name=name,
                partitioned=partitioned,
                partition_format=getattr(spec, "format", None) if spec else None,
                partition_field=getattr(spec, "field", None) if spec else None,
                onetime_cascades=bool(targets),
                cascade_targets=targets,
            )
        )
    return DomainListResponse(items=infos)


@router.get("/health", response_model=HealthStatus)
def health(
    runner: IntegrationRunner = Depends(_get_runner),
) -> HealthStatus:
    """Return the integration runner's health snapshot."""
    snapshot: dict[str, Any] = runner.health() or {}
    return HealthStatus(**snapshot)
