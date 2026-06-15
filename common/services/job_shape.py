"""Pure shape/status adapters between job_history and integration_job (US17a).

These functions translate a ``job_history`` row dict (as produced by
``JobManager._db_get`` / ``_db_list`` via ``job_state._row_to_dict``) into the
field set the ``/integration/jobs`` ``Job`` API model expects, so that a
JobManager-backed ingestion job (``etl_pipeline`` / ``load_domain``) can be
surfaced through the existing integration UI without changing its contract.

Hard rule: this module is **pure** — no DB, no network, no I/O. Importing it
must not open a connection. Keep the field list in lockstep with
``api/routers/platform/integration.py::Job``.

Status vocabulary single source of truth (the only divergence between the two
backends):

    job_history 'completed'  <->  integration 'success'

All other statuses pass through unchanged in both directions.
"""
from __future__ import annotations

from typing import Any

# The single point where the two status vocabularies diverge.
_JH_TO_INTEGRATION: dict[str, str] = {"completed": "success"}
_INTEGRATION_TO_JH: dict[str, str] = {"success": "completed"}


def to_integration_status(status: str | None) -> str | None:
    """Map a ``job_history`` status to the integration vocabulary."""
    if status is None:
        return None
    return _JH_TO_INTEGRATION.get(status, status)


def to_job_history_status(status: str | None) -> str | None:
    """Map an integration status back to the ``job_history`` vocabulary."""
    if status is None:
        return None
    return _INTEGRATION_TO_JH.get(status, status)


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce a params/result value to a dict (tolerate None / non-dict)."""
    return value if isinstance(value, dict) else {}


def _coerce_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _derive_domain(params: dict[str, Any]) -> str:
    """Resolve a display domain from a job's params.

    - ``load_domain`` stores a single ``domain``.
    - ``etl_pipeline`` stores ``domains`` (a list, or null meaning "all").
    """
    domain = params.get("domain")
    if domain:
        return str(domain)
    domains = params.get("domains")
    if isinstance(domains, list) and domains:
        return ",".join(str(d) for d in domains)
    if isinstance(domains, str) and domains:
        return domains
    # etl_pipeline with no explicit domains == whole pipeline (all domains).
    return "pipeline"


def job_history_to_integration_job(row: dict[str, Any]) -> dict[str, Any]:
    """Translate a ``job_history`` row dict into the ``/integration/jobs`` shape.

    Returns every field the ``Job`` API model declares so the caller can do
    ``Job(**job_history_to_integration_job(row))`` directly. Missing
    params/result default gracefully (``rows_loaded=0``); a non-ETL row maps to
    sane defaults without raising.
    """
    params = _as_dict(row.get("params"))
    result = _as_dict(row.get("result"))

    rows_loaded = _coerce_int(result.get("rows_loaded"), None)
    if rows_loaded is None:
        rows_loaded = _coerce_int(result.get("loaded"), 0)

    return {
        "id": row.get("job_id"),
        "domain": _derive_domain(params),
        "mode": str(params.get("mode") or ""),
        "slice": params.get("slice"),
        "file_path": params.get("file"),
        "status": to_integration_status(row.get("status")),
        "rows_loaded": rows_loaded if rows_loaded is not None else 0,
        "rows_inserted": _coerce_int(result.get("rows_inserted"), None),
        "rows_updated": _coerce_int(result.get("rows_updated"), None),
        "rows_deleted": _coerce_int(result.get("rows_deleted"), None),
        "error_message": row.get("error"),
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "duration_ms": _coerce_int(result.get("duration_ms"), None),
        "triggered_by": row.get("triggered_by") or "api",
    }
