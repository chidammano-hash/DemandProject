"""Inventory Planning — IPfeature7: Exception Queue & Replenishment Recommendations endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import _f, _s, get_conn, set_cache

router = APIRouter(tags=["inv-planning"])


_VALID_EXCEPTION_TYPES = {
    "below_rop", "below_rop_critical", "below_ss", "stockout", "excess", "zero_velocity"
}
_VALID_SEVERITIES = {"critical", "high", "medium", "low"}
_VALID_STATUSES = {"open", "acknowledged", "ordered", "resolved"}
_EXCEPTION_SORT_COLS = {
    "severity", "exception_date", "recommended_order_by",
    "current_qty_on_hand", "item_no", "loc",
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
        wheres.append("item_no ILIKE %s")
        params.append(f"%{item}%")
    if location:
        wheres.append("loc ILIKE %s")
        params.append(f"%{location}%")
    if brand:
        params.append(brand.split(","))
        wheres.append("EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = t.item_no AND di.brand_name = ANY(%s))")
    if category:
        params.append(category.split(","))
        wheres.append('EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = t.item_no AND di.class_ = ANY(%s))')
    if market:
        params.append(market.split(","))
        wheres.append("EXISTS (SELECT 1 FROM dim_location dl WHERE dl.loc = t.loc AND dl.state_id = ANY(%s))")

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
            exception_id, item_no, loc, exception_date, exception_type, severity,
            current_qty_on_hand, current_dos, ss_combined, reorder_point,
            recommended_order_qty, recommended_order_by, expected_receipt_date,
            estimated_order_value, policy_id, status, acknowledged_by, notes
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
                "item_no":                r[1],
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
    if brand:
        params.append(brand.split(","))
        wheres.append("EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = t.item_no AND di.brand_name = ANY(%s))")
    if category:
        params.append(category.split(","))
        wheres.append('EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = t.item_no AND di.class_ = ANY(%s))')
    if market:
        params.append(market.split(","))
        wheres.append("EXISTS (SELECT 1 FROM dim_location dl WHERE dl.loc = t.loc AND dl.state_id = ANY(%s))")

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
            ) AS oldest_open_days
        FROM fact_replenishment_exceptions t
        {where}
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone() or (0,) * 13

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
            exception_id, item_no, loc, exception_type, severity,
            status, acknowledged_by, acknowledged_ts, notes
    """
    now = _dt.datetime.now(_dt.timezone.utc)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [body.acknowledged_by, now, body.notes, now, exception_id])
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Exception not found or already acknowledged")

    return {
        "exception_id":    row[0],
        "item_no":         row[1],
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
            exception_id, item_no, loc, exception_type, severity,
            status, acknowledged_by, {ts_col}, notes
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [body.status, now, body.notes, now, exception_id])
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Exception not found")

    return {
        "exception_id":   row[0],
        "item_no":        row[1],
        "loc":            row[2],
        "exception_type": row[3],
        "severity":       row[4],
        "status":         row[5],
        "acknowledged_by": row[6],
        ts_col:           row[7].isoformat() if row[7] else None,
        "notes":          row[8],
    }


@router.post("/inv-planning/exceptions/generate")
def generate_exceptions(
    _: None = Depends(require_api_key),
) -> dict:
    """Trigger exception detection and insert new exceptions into the queue."""
    from scripts.generate_replenishment_exceptions import run as _gen_run

    result = _gen_run(dry_run=False)
    return {
        "generated_count": result["generated_count"],
        "skipped_dedup":   result["skipped_dedup"],
        "by_type":         result["by_type"],
    }
