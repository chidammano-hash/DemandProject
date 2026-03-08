"""Demand Planner Storyboard endpoints — Feature 40: Exception-Based Value Workflow.

Router mounted at /storyboard in api/main.py.

Endpoints:
  GET  /storyboard/exceptions          — paginated exception list with filters
  GET  /storyboard/exceptions/summary  — KPI counts by type/status, top items
  GET  /storyboard/exceptions/{id}     — single exception detail + decision history
  PUT  /storyboard/exceptions/{id}/status — update exception status (auth required)
  POST /storyboard/exceptions/{id}/decide — add planner decision (auth required)
  GET  /storyboard/decisions           — recent planner decisions log
  POST /storyboard/generate            — trigger exception generation (auth required)

All endpoints use get_conn() directly (not Depends(_get_pool)) — same pattern
as health and exception endpoints in inv_planning.py.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import get_conn, set_cache

router = APIRouter(tags=["storyboard"])


def _f(v: Any) -> float | None:
    return float(v) if v is not None else None


def _s(v: Any) -> str | None:
    return str(v) if v is not None else None


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class ExceptionStatusUpdate(BaseModel):
    status: str  # open | investigating | resolved | dismissed
    assigned_to: Optional[str] = None


class PlannerDecision(BaseModel):
    decision_type: str  # override_forecast | accept_exception | escalate | dismiss | request_info
    decision_value: Optional[dict] = None  # {"new_forecast": 500} or {"reason": "..."}
    rationale: Optional[str] = None
    decided_by: Optional[str] = "planner"


class GenerateRequest(BaseModel):
    month: Optional[str] = None         # YYYY-MM (default: current month)
    dry_run: Optional[bool] = False
    exception_type: Optional[str] = None  # filter to one type


# ---------------------------------------------------------------------------
# GET /storyboard/exceptions
# ---------------------------------------------------------------------------

@router.get("/storyboard/exceptions")
def list_exceptions(
    response: FastAPIResponse,
    status: str = Query(default="open", max_length=50),
    exception_type: str = Query(default="", max_length=50),
    severity_min: float = Query(default=0.0, ge=0.0, le=1.0),
    item: str = Query(default="", max_length=100),
    loc: str = Query(default="", max_length=100),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Paginated list of exceptions, sorted by severity descending.

    Filters: status, exception_type, severity_min, item (partial match), loc (partial match).
    Cache: 30s.
    """
    set_cache(response, max_age=30)

    where_parts: list[str] = []
    params: list[Any] = []

    if status.strip():
        where_parts.append("status = %s")
        params.append(status.strip())

    if exception_type.strip():
        where_parts.append("exception_type = %s")
        params.append(exception_type.strip())

    if severity_min > 0:
        where_parts.append("severity >= %s")
        params.append(severity_min)

    if item.strip():
        where_parts.append("item_no ILIKE %s")
        params.append(f"%{item.strip()}%")

    if loc.strip():
        where_parts.append("loc ILIKE %s")
        params.append(f"%{loc.strip()}%")

    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    count_sql = f"SELECT COUNT(*) FROM exception_queue {where_clause}"

    rows_sql = f"""
        SELECT
            exception_id,
            exception_type,
            item_no,
            loc,
            severity,
            financial_impact,
            headline,
            supporting_data,
            status,
            assigned_to,
            generated_at,
            expires_at,
            month_start
        FROM exception_queue
        {where_clause}
        ORDER BY severity DESC, generated_at DESC
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = cur.fetchone()[0]

            cur.execute(rows_sql, params + [limit, offset])
            col_names = [d[0] for d in cur.description]
            raw_rows = cur.fetchall()

    rows = []
    for r in raw_rows:
        row = dict(zip(col_names, r))
        row["severity"] = _f(row.get("severity"))
        row["financial_impact"] = _f(row.get("financial_impact"))
        row["generated_at"] = row["generated_at"].isoformat() if row.get("generated_at") else None
        row["expires_at"] = row["expires_at"].isoformat() if row.get("expires_at") else None
        row["month_start"] = row["month_start"].isoformat() if row.get("month_start") else None
        row["exception_id"] = _s(row.get("exception_id"))
        # supporting_data is already dict/jsonb from psycopg
        rows.append(row)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# GET /storyboard/exceptions/summary
# ---------------------------------------------------------------------------

@router.get("/storyboard/exceptions/summary")
def exceptions_summary(response: FastAPIResponse) -> dict:
    """KPI summary: counts by type and status, total open, avg severity, top 5 items.

    Cache: 30s.
    """
    set_cache(response, max_age=30)

    summary_sql = """
        SELECT
            COUNT(*) FILTER (WHERE status = 'open')              AS open_count,
            COUNT(*) FILTER (WHERE status = 'investigating')     AS investigating_count,
            COUNT(*) FILTER (WHERE status = 'resolved')          AS resolved_count,
            COUNT(*) FILTER (WHERE status = 'dismissed')         AS dismissed_count,
            COUNT(*) FILTER (WHERE status = 'open' AND severity >= 0.75) AS critical_open,
            COUNT(*) FILTER (WHERE status = 'open' AND severity >= 0.50 AND severity < 0.75) AS high_open,
            SUM(financial_impact) FILTER (WHERE status = 'open') AS total_impact_open,
            AVG(severity) FILTER (WHERE status = 'open')         AS avg_severity_open,
            -- By type
            COUNT(*) FILTER (WHERE exception_type = 'forecast_bias' AND status = 'open')   AS forecast_bias_count,
            COUNT(*) FILTER (WHERE exception_type = 'stockout_risk' AND status = 'open')   AS stockout_risk_count,
            COUNT(*) FILTER (WHERE exception_type = 'accuracy_drop' AND status = 'open')   AS accuracy_drop_count,
            COUNT(*) FILTER (WHERE exception_type = 'excess_risk'   AND status = 'open')   AS excess_risk_count,
            COUNT(*) FILTER (WHERE exception_type = 'model_drift'   AND status = 'open')   AS model_drift_count,
            COUNT(*) FILTER (WHERE exception_type = 'new_item'      AND status = 'open')   AS new_item_count
        FROM exception_queue
    """

    top_items_sql = """
        SELECT
            item_no,
            loc,
            COUNT(*) AS exception_count,
            MAX(severity) AS max_severity,
            SUM(COALESCE(financial_impact, 0)) AS total_impact
        FROM exception_queue
        WHERE status = 'open'
        GROUP BY item_no, loc
        ORDER BY max_severity DESC, exception_count DESC
        LIMIT 5
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(summary_sql)
            row = cur.fetchone()
            col_names = [d[0] for d in cur.description]
            summary = dict(zip(col_names, row)) if row else {}

            cur.execute(top_items_sql)
            top_cols = [d[0] for d in cur.description]
            top_rows = cur.fetchall()

    top_items = [
        {
            "item_no": r[0],
            "loc": r[1],
            "exception_count": int(r[2] or 0),
            "max_severity": _f(r[3]),
            "total_impact": _f(r[4]),
        }
        for r in top_rows
    ]

    return {
        "open_count": int(summary.get("open_count") or 0),
        "investigating_count": int(summary.get("investigating_count") or 0),
        "resolved_count": int(summary.get("resolved_count") or 0),
        "dismissed_count": int(summary.get("dismissed_count") or 0),
        "critical_open": int(summary.get("critical_open") or 0),
        "high_open": int(summary.get("high_open") or 0),
        "total_financial_impact_open": _f(summary.get("total_impact_open")),
        "avg_severity_open": _f(summary.get("avg_severity_open")),
        "by_type": {
            "forecast_bias": int(summary.get("forecast_bias_count") or 0),
            "stockout_risk": int(summary.get("stockout_risk_count") or 0),
            "accuracy_drop": int(summary.get("accuracy_drop_count") or 0),
            "excess_risk": int(summary.get("excess_risk_count") or 0),
            "model_drift": int(summary.get("model_drift_count") or 0),
            "new_item": int(summary.get("new_item_count") or 0),
        },
        "top_items": top_items,
    }


# ---------------------------------------------------------------------------
# GET /storyboard/exceptions/{exception_id}
# ---------------------------------------------------------------------------

@router.get("/storyboard/exceptions/{exception_id}")
def get_exception_detail(exception_id: str) -> dict:
    """Single exception detail with full supporting_data and decision history."""

    exc_sql = """
        SELECT
            exception_id,
            exception_type,
            item_no,
            loc,
            severity,
            financial_impact,
            headline,
            supporting_data,
            status,
            assigned_to,
            generated_at,
            expires_at,
            month_start,
            load_ts
        FROM exception_queue
        WHERE exception_id = %s::uuid
    """

    decisions_sql = """
        SELECT
            decision_id,
            decision_type,
            decision_value,
            rationale,
            decided_by,
            decided_at
        FROM planner_decisions
        WHERE exception_id = %s::uuid
        ORDER BY decided_at DESC
        LIMIT 50
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(exc_sql, (exception_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Exception not found")
            col_names = [d[0] for d in cur.description]
            exc = dict(zip(col_names, row))

            cur.execute(decisions_sql, (exception_id,))
            d_cols = [d[0] for d in cur.description]
            d_rows = cur.fetchall()

    exc["severity"] = _f(exc.get("severity"))
    exc["financial_impact"] = _f(exc.get("financial_impact"))
    exc["generated_at"] = exc["generated_at"].isoformat() if exc.get("generated_at") else None
    exc["expires_at"] = exc["expires_at"].isoformat() if exc.get("expires_at") else None
    exc["month_start"] = exc["month_start"].isoformat() if exc.get("month_start") else None
    exc["load_ts"] = exc["load_ts"].isoformat() if exc.get("load_ts") else None
    exc["exception_id"] = _s(exc.get("exception_id"))

    decisions = []
    for dr in d_rows:
        d = dict(zip(d_cols, dr))
        d["decision_id"] = _s(d.get("decision_id"))
        d["decided_at"] = d["decided_at"].isoformat() if d.get("decided_at") else None
        decisions.append(d)

    exc["decisions"] = decisions
    return exc


# ---------------------------------------------------------------------------
# PUT /storyboard/exceptions/{exception_id}/status
# ---------------------------------------------------------------------------

VALID_STATUSES = {"open", "investigating", "resolved", "dismissed"}


@router.post("/storyboard/exceptions/{exception_id}/status")
def update_exception_status(
    exception_id: str,
    body: ExceptionStatusUpdate,
    _key: str = Depends(require_api_key),
) -> dict:
    """Update exception status and optionally assign to a planner.

    Auth required (X-API-Key header when API_KEY env var is set).
    Valid statuses: open, investigating, resolved, dismissed.
    """
    if body.status not in VALID_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status '{body.status}'. Must be one of: {sorted(VALID_STATUSES)}",
        )

    update_sql = """
        UPDATE exception_queue
        SET status = %s,
            assigned_to = COALESCE(%s, assigned_to)
        WHERE exception_id = %s::uuid
        RETURNING exception_id, status, assigned_to
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(update_sql, (body.status, body.assigned_to, exception_id))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Exception not found")
        conn.commit()

    return {
        "exception_id": _s(row[0]),
        "status": row[1],
        "assigned_to": row[2],
    }


# ---------------------------------------------------------------------------
# POST /storyboard/exceptions/{exception_id}/decide
# ---------------------------------------------------------------------------

VALID_DECISION_TYPES = {
    "override_forecast",
    "accept_exception",
    "escalate",
    "dismiss",
    "request_info",
}


@router.post("/storyboard/exceptions/{exception_id}/decide")
def add_decision(
    exception_id: str,
    body: PlannerDecision,
    _key: str = Depends(require_api_key),
) -> dict:
    """Record a planner decision for an exception and update its status.

    decision_type options:
      - accept_exception: marks exception resolved
      - override_forecast: marks resolved, stores new_forecast in decision_value
      - escalate: marks investigating, logs note
      - dismiss: marks dismissed
      - request_info: marks investigating

    Auth required (X-API-Key header when API_KEY env var is set).
    """
    if body.decision_type not in VALID_DECISION_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid decision_type '{body.decision_type}'. "
                   f"Must be one of: {sorted(VALID_DECISION_TYPES)}",
        )

    # Map decision type to new exception status
    status_map = {
        "accept_exception": "resolved",
        "override_forecast": "resolved",
        "escalate": "investigating",
        "dismiss": "dismissed",
        "request_info": "investigating",
    }
    new_status = status_map[body.decision_type]

    insert_decision_sql = """
        INSERT INTO planner_decisions
            (exception_id, item_no, loc, decision_type, decision_value, rationale, decided_by)
        SELECT
            %s::uuid,
            item_no,
            loc,
            %s,
            %s::jsonb,
            %s,
            %s
        FROM exception_queue
        WHERE exception_id = %s::uuid
        RETURNING decision_id, item_no, loc
    """

    update_status_sql = """
        UPDATE exception_queue
        SET status = %s
        WHERE exception_id = %s::uuid
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(insert_decision_sql, (
                exception_id,
                body.decision_type,
                json.dumps(body.decision_value) if body.decision_value else "{}",
                body.rationale,
                body.decided_by or "planner",
                exception_id,
            ))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Exception not found")

            decision_id, item_no, loc = row

            cur.execute(update_status_sql, (new_status, exception_id))

        conn.commit()

    return {
        "decision_id": _s(decision_id),
        "exception_id": exception_id,
        "decision_type": body.decision_type,
        "new_exception_status": new_status,
        "item_no": item_no,
        "loc": loc,
    }


# ---------------------------------------------------------------------------
# GET /storyboard/decisions
# ---------------------------------------------------------------------------

@router.get("/storyboard/decisions")
def list_decisions(
    response: FastAPIResponse,
    item: str = Query(default="", max_length=100),
    loc: str = Query(default="", max_length=100),
    decision_type: str = Query(default="", max_length=50),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Recent planner decisions audit log.

    Filters: item, loc, decision_type.
    Cache: 30s.
    """
    set_cache(response, max_age=30)

    where_parts: list[str] = []
    params: list[Any] = []

    if item.strip():
        where_parts.append("item_no ILIKE %s")
        params.append(f"%{item.strip()}%")

    if loc.strip():
        where_parts.append("loc ILIKE %s")
        params.append(f"%{loc.strip()}%")

    if decision_type.strip():
        where_parts.append("decision_type = %s")
        params.append(decision_type.strip())

    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    count_sql = f"SELECT COUNT(*) FROM planner_decisions {where_clause}"
    rows_sql = f"""
        SELECT
            decision_id,
            exception_id,
            item_no,
            loc,
            decision_type,
            decision_value,
            rationale,
            decided_by,
            decided_at
        FROM planner_decisions
        {where_clause}
        ORDER BY decided_at DESC
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = cur.fetchone()[0]

            cur.execute(rows_sql, params + [limit, offset])
            col_names = [d[0] for d in cur.description]
            raw_rows = cur.fetchall()

    rows = []
    for r in raw_rows:
        row = dict(zip(col_names, r))
        row["decision_id"] = _s(row.get("decision_id"))
        row["exception_id"] = _s(row.get("exception_id"))
        row["decided_at"] = row["decided_at"].isoformat() if row.get("decided_at") else None
        rows.append(row)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# POST /storyboard/generate
# ---------------------------------------------------------------------------

@router.post("/storyboard/generate")
def generate_exceptions(
    body: GenerateRequest,
    _key: str = Depends(require_api_key),
) -> dict:
    """Trigger on-demand exception generation for a given month.

    Auth required (X-API-Key header when API_KEY env var is set).
    Returns counts: detected, inserted, skipped_dedupe.
    """
    import scripts.generate_storyboard_exceptions as gen_module

    result = gen_module.run(
        month_str=body.month,
        dry_run=body.dry_run or False,
        exception_type=body.exception_type,
    )

    return result
