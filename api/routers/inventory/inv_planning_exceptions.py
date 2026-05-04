"""Inventory Planning — IPfeature7: Exception Queue & Replenishment Recommendations endpoints."""
from __future__ import annotations

import logging
from typing import Optional

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import _f, add_cross_dim_filters, get_conn, set_cache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inv-planning"])


# ---------------------------------------------------------------------------
# Exception lifecycle writer — Gen-4 Roadmap 1.9
# ---------------------------------------------------------------------------
def _record_lifecycle_transition(
    cur,
    *,
    exception_id: str,
    from_state: str | None,
    to_state: str,
    actor: str | None,
    notes: str | None,
    exception_type: str | None = None,
    severity: str | None = None,
    item_id: str | None = None,
    loc: str | None = None,
    financial_impact: float | None = None,
) -> None:
    """Append a row to fact_exception_lifecycle.

    Best-effort: the table may not exist yet on older deployments, so we
    catch psycopg.Error and log without failing the parent transaction's
    user-visible outcome. Callers MUST commit the parent transaction.
    """
    try:
        cur.execute(
            """
            INSERT INTO fact_exception_lifecycle
                (exception_id, from_state, to_state, actor, notes,
                 exception_type, severity, item_id, loc, financial_impact)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                exception_id, from_state, to_state, actor, notes,
                exception_type, severity, item_id, loc, financial_impact,
            ],
        )
    except psycopg.Error as exc:
        # Table may be missing (older env). Log + continue so the business
        # endpoint still succeeds.
        logger.exception(
            "Failed to write fact_exception_lifecycle row for %s: %s", exception_id, exc
        )


_VALID_EXCEPTION_TYPES = {
    "below_rop", "below_rop_critical", "below_ss", "stockout", "excess", "zero_velocity"
}
_VALID_SEVERITIES = {"critical", "high", "medium", "low"}
_VALID_STATUSES = {"open", "acknowledged", "ordered", "resolved"}
_EXCEPTION_SORT_COLS = {
    "severity", "exception_date", "recommended_order_by",
    "current_qty_on_hand", "item_id", "loc", "financial_impact_total",
}
_SEVERITY_ORDER = "CASE severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 ELSE 4 END"


class ExceptionAcknowledgeBody(BaseModel):
    acknowledged_by: str
    notes: Optional[str] = None


class ExceptionStatusBody(BaseModel):
    status: str
    notes: Optional[str] = None




@router.get("/inv-planning/exceptions")
def get_exceptions(
    response: FastAPIResponse,
    exception_type: Optional[str] = None,
    severity: Optional[str] = None,
    status: str = "open",
    item: Optional[str] = None,
    location: Optional[str] = None,
    brand: Optional[str] = None,
    category: Optional[str] = None,
    market: Optional[str] = None,
    sort_by: str = "severity",
    sort_dir: str = "asc",
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Paginated exception queue with filters.

    Default: open exceptions sorted by severity (critical first), then exception_date asc.
    Cache: 60s.
    """
    set_cache(response, max_age=60)

    wheres: list[str] = []
    params: list = []

    if status and status in _VALID_STATUSES:
        wheres.append("status = %s")
        params.append(status)
    if exception_type and exception_type in _VALID_EXCEPTION_TYPES:
        wheres.append("exception_type = %s")
        params.append(exception_type)
    if severity and severity in _VALID_SEVERITIES:
        wheres.append("severity = %s")
        params.append(severity)
    if item:
        wheres.append("item_id ILIKE %s")
        params.append(f"%{item}%")
    if location:
        wheres.append("loc ILIKE %s")
        params.append(f"%{location}%")
    add_cross_dim_filters(wheres, params, brand=brand, category=category, market=market)

    where_clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""

    col = sort_by if sort_by in _EXCEPTION_SORT_COLS else "severity"
    direction = "ASC" if sort_dir.lower() == "asc" else "DESC"

    if col == "severity":
        order_clause = f"{_SEVERITY_ORDER} {direction}, exception_date ASC"
    else:
        order_clause = f"{col} {direction} NULLS LAST"

    count_sql = f"SELECT COUNT(*) FROM fact_replenishment_exceptions t {where_clause}"
    rows_sql = f"""
        SELECT
            exception_id, item_id, loc, exception_date, exception_type, severity,
            current_qty_on_hand, current_dos, ss_combined, reorder_point,
            recommended_order_qty, recommended_order_by, expected_receipt_date,
            estimated_order_value, policy_id, status, acknowledged_by, notes,
            unit_cost, unit_margin, daily_demand_rate,
            loss_of_sales_7d, loss_of_sales_30d,
            monthly_holding_cost, financial_impact_total
        FROM fact_replenishment_exceptions t
        {where_clause}
        ORDER BY {order_clause}
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, params)
            total = (cur.fetchone() or (0,))[0]
            cur.execute(rows_sql, params + [limit, offset])
            rows = cur.fetchall()

    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "rows": [
            {
                "exception_id":           r[0],
                "item_id":                r[1],
                "loc":                    r[2],
                "exception_date":         r[3].isoformat() if r[3] else None,
                "exception_type":         r[4],
                "severity":               r[5],
                "current_qty_on_hand":    _f(r[6]),
                "current_dos":            _f(r[7]),
                "ss_combined":            _f(r[8]),
                "reorder_point":          _f(r[9]),
                "recommended_order_qty":  _f(r[10]),
                "recommended_order_by":   r[11].isoformat() if r[11] else None,
                "expected_receipt_date":  r[12].isoformat() if r[12] else None,
                "estimated_order_value":  _f(r[13]),
                "policy_id":              r[14],
                "status":                 r[15],
                "acknowledged_by":        r[16],
                "notes":                  r[17],
                "unit_cost":              _f(r[18]),
                "unit_margin":            _f(r[19]),
                "daily_demand_rate":      _f(r[20]),
                "loss_of_sales_7d":       _f(r[21]),
                "loss_of_sales_30d":      _f(r[22]),
                "monthly_holding_cost":   _f(r[23]),
                "financial_impact_total":  _f(r[24]),
            }
            for r in rows
        ],
    }


@router.get("/inv-planning/exceptions/summary")
def get_exception_summary(
    response: FastAPIResponse,
    status: str = "open",
    brand: Optional[str] = None,
    category: Optional[str] = None,
    market: Optional[str] = None,
) -> dict:
    """Aggregate exception counts by type and severity.

    Cache: 60s.
    """
    set_cache(response, max_age=60)

    wheres: list[str] = []
    params: list = []

    if status in _VALID_STATUSES:
        wheres.append("status = %s")
        params.append(status)
    add_cross_dim_filters(wheres, params, brand=brand, category=category, market=market)

    where = ("WHERE " + " AND ".join(wheres)) if wheres else ""

    sql = f"""
        SELECT
            COUNT(*)                                                               AS open_count,
            SUM(CASE WHEN exception_type = 'below_rop'          THEN 1 ELSE 0 END) AS below_rop,
            SUM(CASE WHEN exception_type = 'below_rop_critical'  THEN 1 ELSE 0 END) AS below_rop_critical,
            SUM(CASE WHEN exception_type = 'below_ss'           THEN 1 ELSE 0 END) AS below_ss,
            SUM(CASE WHEN exception_type = 'stockout'           THEN 1 ELSE 0 END) AS stockout,
            SUM(CASE WHEN exception_type = 'excess'             THEN 1 ELSE 0 END) AS excess,
            SUM(CASE WHEN exception_type = 'zero_velocity'      THEN 1 ELSE 0 END) AS zero_velocity,
            SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) AS critical_count,
            SUM(CASE WHEN severity = 'high'     THEN 1 ELSE 0 END) AS high_count,
            SUM(CASE WHEN severity = 'medium'   THEN 1 ELSE 0 END) AS medium_count,
            SUM(CASE WHEN severity = 'low'      THEN 1 ELSE 0 END) AS low_count,
            COALESCE(SUM(estimated_order_value), 0)                  AS total_recommended_order_value,
            COALESCE(
                MAX(EXTRACT(DAY FROM NOW() - exception_date::TIMESTAMPTZ))::INT,
                0
            ) AS oldest_open_days,
            COALESCE(SUM(financial_impact_total), 0)                 AS total_financial_impact,
            COALESCE(SUM(loss_of_sales_7d), 0)                       AS total_loss_of_sales_7d,
            COALESCE(SUM(monthly_holding_cost), 0)                   AS total_monthly_holding_cost,
            MAX(load_ts)                                              AS last_generated_at
        FROM fact_replenishment_exceptions t
        {where}
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone() or (0,) * 17

    # row[16] = last_generated_at (TIMESTAMPTZ or None)
    last_gen = row[16]
    last_generated_at = last_gen.isoformat() if hasattr(last_gen, "isoformat") else (str(last_gen) if last_gen else None)

    return {
        "open_count": int(row[0] or 0),
        "by_type": {
            "below_rop":          int(row[1] or 0),
            "below_rop_critical": int(row[2] or 0),
            "below_ss":           int(row[3] or 0),
            "stockout":           int(row[4] or 0),
            "excess":             int(row[5] or 0),
            "zero_velocity":      int(row[6] or 0),
        },
        "by_severity": {
            "critical": int(row[7] or 0),
            "high":     int(row[8] or 0),
            "medium":   int(row[9] or 0),
            "low":      int(row[10] or 0),
        },
        "total_recommended_order_value": float(row[11] or 0),
        "oldest_open_days":              int(row[12] or 0),
        "total_financial_impact":        float(row[13] or 0),
        "total_loss_of_sales_7d":        float(row[14] or 0),
        "total_monthly_holding_cost":    float(row[15] or 0),
        "last_generated_at":             last_generated_at,
    }


@router.post("/inv-planning/exceptions/{exception_id}/acknowledge")
def acknowledge_exception(
    exception_id: str,
    body: ExceptionAcknowledgeBody,
    _: None = Depends(require_api_key),
) -> dict:
    """Acknowledge an open exception (sets status to 'acknowledged')."""
    import datetime as _dt

    sql = """
        UPDATE fact_replenishment_exceptions
        SET
            status           = 'acknowledged',
            acknowledged_by  = %s,
            acknowledged_ts  = %s,
            notes            = COALESCE(%s, notes),
            modified_ts      = %s
        WHERE exception_id = %s
          AND status       = 'open'
        RETURNING
            exception_id, item_id, loc, exception_type, severity,
            status, acknowledged_by, acknowledged_ts, notes
    """
    now = _dt.datetime.now(_dt.timezone.utc)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [body.acknowledged_by, now, body.notes, now, exception_id])
            row = cur.fetchone()
            if row:
                # Gen-4 Roadmap 1.9 — log state transition open -> acknowledged.
                _record_lifecycle_transition(
                    cur,
                    exception_id=str(row[0]),
                    from_state="open",
                    to_state="acknowledged",
                    actor=body.acknowledged_by,
                    notes=body.notes,
                    exception_type=row[3],
                    severity=row[4],
                    item_id=row[1],
                    loc=row[2],
                )
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Exception not found or already acknowledged")

    return {
        "exception_id":    row[0],
        "item_id":         row[1],
        "loc":             row[2],
        "exception_type":  row[3],
        "severity":        row[4],
        "status":          row[5],
        "acknowledged_by": row[6],
        "acknowledged_ts": row[7].isoformat() if row[7] else None,
        "notes":           row[8],
    }


@router.post("/inv-planning/exceptions/{exception_id}/status")
def update_exception_status(
    exception_id: str,
    body: ExceptionStatusBody,
    _: None = Depends(require_api_key),
) -> dict:
    """Advance exception status to 'ordered' or 'resolved'."""
    import datetime as _dt

    if body.status not in ("ordered", "resolved"):
        raise HTTPException(status_code=422, detail="status must be 'ordered' or 'resolved'")

    now = _dt.datetime.now(_dt.timezone.utc)
    ts_col = "ordered_ts" if body.status == "ordered" else "resolved_ts"

    sql = f"""
        UPDATE fact_replenishment_exceptions
        SET
            status      = %s,
            {ts_col}    = %s,
            notes       = COALESCE(%s, notes),
            modified_ts = %s
        WHERE exception_id = %s
        RETURNING
            exception_id, item_id, loc, exception_type, severity,
            status, acknowledged_by, {ts_col}, notes
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [body.status, now, body.notes, now, exception_id])
            row = cur.fetchone()
            if row:
                # Gen-4 Roadmap 1.9 — log transition into ordered / resolved.
                # The UPDATE does not return the pre-update status, so from_state
                # is recorded as NULL (unknown). A future read-modify-write
                # should capture it; keeping the audit entry is the priority.
                _record_lifecycle_transition(
                    cur,
                    exception_id=str(row[0]),
                    from_state=None,
                    to_state=body.status,
                    actor=row[6],  # acknowledged_by (most recent actor)
                    notes=body.notes,
                    exception_type=row[3],
                    severity=row[4],
                    item_id=row[1],
                    loc=row[2],
                )
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Exception not found")

    return {
        "exception_id":   row[0],
        "item_id":        row[1],
        "loc":            row[2],
        "exception_type": row[3],
        "severity":       row[4],
        "status":         row[5],
        "acknowledged_by": row[6],
        ts_col:           row[7].isoformat() if row[7] else None,
        "notes":          row[8],
    }


@router.get("/inv-planning/exceptions/{exception_id}/lifecycle")
def get_exception_lifecycle(
    exception_id: str,
    response: FastAPIResponse,
) -> dict:
    """Return the append-only lifecycle trail for one exception (Gen-4 1.9)."""
    set_cache(response, max_age=30)
    sql = """
        SELECT lifecycle_id, from_state, to_state, transitioned_at, actor, notes
        FROM fact_exception_lifecycle
        WHERE exception_id = %s
        ORDER BY transitioned_at ASC
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql, [exception_id])
                rows = cur.fetchall()
            except psycopg.Error as exc:
                logger.exception("lifecycle query failed: %s", exc)
                return {"exception_id": exception_id, "transitions": []}

    return {
        "exception_id": exception_id,
        "transitions": [
            {
                "lifecycle_id":    int(r[0]),
                "from_state":      r[1],
                "to_state":        r[2],
                "transitioned_at": r[3].isoformat() if r[3] else None,
                "actor":           r[4],
                "notes":           r[5],
            }
            for r in rows
        ],
    }


@router.get("/inv-planning/exceptions/mttr")
def get_exception_mttr(
    response: FastAPIResponse,
    exception_type: Optional[str] = None,
    severity: Optional[str] = None,
) -> dict:
    """Mean time to resolve from the exception lifecycle audit (Gen-4 1.9)."""
    set_cache(response, max_age=300)

    wheres: list[str] = []
    params: list = []
    if exception_type:
        wheres.append("exception_type = %s")
        params.append(exception_type)
    if severity and severity in _VALID_SEVERITIES:
        wheres.append("severity = %s")
        params.append(severity)
    where_sql = ("WHERE " + " AND ".join(wheres)) if wheres else ""

    sql = f"""
        SELECT exception_type, severity, resolved_count,
               mttr_hours_avg, mttr_hours_p50, mttr_hours_p90
        FROM v_exception_mttr_summary
        {where_sql}
        ORDER BY exception_type, severity
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql, params)
                rows = cur.fetchall()
            except psycopg.Error as exc:
                logger.exception("mttr query failed: %s", exc)
                return {"rows": []}

    return {
        "rows": [
            {
                "exception_type": r[0],
                "severity":       r[1],
                "resolved_count": int(r[2] or 0),
                "mttr_hours_avg": _f(r[3]),
                "mttr_hours_p50": _f(r[4]),
                "mttr_hours_p90": _f(r[5]),
            }
            for r in rows
        ],
    }


@router.post("/inv-planning/exceptions/generate")
def generate_exceptions(
    _: None = Depends(require_api_key),
) -> dict:
    """Trigger exception detection and insert new exceptions into the queue."""
    from scripts.inventory.generate_replenishment_exceptions import run as _gen_run

    result = _gen_run(dry_run=False)
    return {
        "generated_count": result["generated_count"],
        "skipped_dedup":   result["skipped_dedup"],
        "by_type":         result["by_type"],
    }
