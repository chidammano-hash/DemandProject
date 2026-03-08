"""F2.3 — Consensus Forecasting & Planner Overrides API.

Endpoints:
    GET  /forecast/overrides/summary    — Portfolio-level override summary
    GET  /forecast/overrides            — List overrides with filters
    POST /forecast/overrides            — Submit a new planner override
    PUT  /forecast/overrides/{id}/approve — Approve a pending override
    PUT  /forecast/overrides/{id}/reject  — Reject a pending override
    DELETE /forecast/overrides/{id}       — Soft-delete (supersede) an override
    GET  /forecast/consensus-plan       — Retrieve merged consensus plan
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator

from api.auth import require_api_key
from api.core import get_conn

router = APIRouter(tags=["consensus-plan"])

_cfg = yaml.safe_load(open("config/consensus_config.yaml"))
_VALID_TYPES = set(_cfg["consensus_plan"]["valid_override_types"])
_MULT_MIN = _cfg["consensus_plan"]["multiplier_bounds"]["min"]
_MULT_MAX = _cfg["consensus_plan"]["multiplier_bounds"]["max"]
_THRESH_VALUE = _cfg["consensus_plan"]["approval_required_threshold_value"]
_THRESH_PCT = _cfg["consensus_plan"]["approval_required_threshold_pct"]
_THRESH_UNITS = _cfg["consensus_plan"]["approval_required_threshold_units"]


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class OverrideSubmitRequest(BaseModel):
    item_no: str
    loc: str
    override_month: date
    override_type: str
    override_qty: Optional[float] = None
    override_multiplier: Optional[float] = None
    override_additive_qty: float = 0.0
    is_hard_override: bool = False
    override_reason: str
    override_note: Optional[str] = None
    valid_from: date
    valid_to: date
    created_by: str
    priority_rank: int = 5
    statistical_qty: Optional[float] = None

    @field_validator("override_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in _VALID_TYPES:
            raise ValueError(f"override_type must be one of {sorted(_VALID_TYPES)}")
        return v

    @field_validator("override_multiplier")
    @classmethod
    def validate_multiplier(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (_MULT_MIN <= v <= _MULT_MAX):
            raise ValueError(f"override_multiplier must be in [{_MULT_MIN}, {_MULT_MAX}]")
        return v

    @field_validator("override_month")
    @classmethod
    def validate_future_month(cls, v: date) -> date:
        today = date.today()
        if v < today.replace(day=1):
            raise ValueError("override_month must be current month or future")
        return v


class ApproveRequest(BaseModel):
    approved_by: str
    approval_note: Optional[str] = None


class RejectRequest(BaseModel):
    rejected_by: str
    rejection_reason: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _compute_requires_approval(
    impact_units: float | None,
    impact_value: float | None,
    multiplier: float | None,
    stat_qty: float | None,
) -> bool:
    if impact_value is not None and impact_value > _THRESH_VALUE:
        return True
    if impact_units is not None and impact_units > _THRESH_UNITS:
        return True
    if multiplier is not None and stat_qty is not None and stat_qty > 0:
        pct_change = abs(multiplier - 1.0)
        if pct_change > _THRESH_PCT:
            return True
    return False


def _compute_impact(
    stat_qty: float | None,
    multiplier: float | None,
    additive: float,
    is_hard: bool,
    override_qty: float | None,
    unit_cost: float | None = None,
) -> tuple[float | None, float | None]:
    if stat_qty is None:
        return None, None
    if is_hard and override_qty is not None:
        delta = abs(override_qty - stat_qty)
    else:
        m = multiplier if multiplier is not None else 1.0
        delta = abs(stat_qty * m + additive - stat_qty)
    value = round(delta * unit_cost, 2) if unit_cost else None
    return round(delta, 2), value


# ---------------------------------------------------------------------------
# GET /forecast/overrides/summary  (must come before /forecast/overrides/{id})
# ---------------------------------------------------------------------------

@router.get("/forecast/overrides/summary")
async def get_override_summary():
    """Portfolio-level override summary: counts by status and type."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE status = 'pending_approval') AS pending,
                    COUNT(*) FILTER (WHERE status = 'approved')         AS approved,
                    COUNT(*) FILTER (WHERE status = 'rejected')         AS rejected,
                    COUNT(*) FILTER (WHERE status = 'expired')          AS expired,
                    COUNT(*) FILTER (WHERE status = 'superseded')       AS superseded,
                    COUNT(DISTINCT item_no || '|' || loc)
                        FILTER (WHERE status = 'approved')              AS dfu_count_overridden,
                    COALESCE(SUM(estimated_impact_units)
                        FILTER (WHERE status = 'approved'), 0)          AS total_uplift_units,
                    COALESCE(SUM(estimated_impact_value)
                        FILTER (WHERE status = 'approved'), 0)          AS total_uplift_value
                FROM fact_forecast_overrides
            """)
            row = cur.fetchone()

            cur.execute("""
                SELECT override_type, COUNT(*)
                FROM fact_forecast_overrides
                WHERE status = 'approved'
                GROUP BY override_type
            """)
            type_rows = cur.fetchall()

    by_type = {r[0]: int(r[1]) for r in type_rows}
    return {
        "by_status": {
            "pending_approval": int(row[0]) if row[0] else 0,
            "approved": int(row[1]) if row[1] else 0,
            "rejected": int(row[2]) if row[2] else 0,
            "expired": int(row[3]) if row[3] else 0,
            "superseded": int(row[4]) if row[4] else 0,
        },
        "dfu_count_overridden": int(row[5]) if row[5] else 0,
        "total_uplift_units": float(row[6]) if row[6] else 0.0,
        "total_uplift_value": float(row[7]) if row[7] else 0.0,
        "by_type": by_type,
    }


# ---------------------------------------------------------------------------
# GET /forecast/overrides
# ---------------------------------------------------------------------------

@router.get("/forecast/overrides")
async def list_overrides(
    item_no: str | None = None,
    loc: str | None = None,
    status: str | None = None,
    override_type: str | None = None,
    month_from: str | None = None,
    month_to: str | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """List planner overrides with optional filters."""
    page_size = min(page_size, 200)
    offset = (page - 1) * page_size

    clauses, params = [], []
    if item_no:
        clauses.append("item_no = %s"); params.append(item_no)
    if loc:
        clauses.append("loc = %s"); params.append(loc)
    if status:
        clauses.append("status = %s"); params.append(status)
    if override_type:
        clauses.append("override_type = %s"); params.append(override_type)
    if month_from:
        clauses.append("override_month >= %s"); params.append(month_from)
    if month_to:
        clauses.append("override_month <= %s"); params.append(month_to)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM fact_forecast_overrides {where}", params)
            total = cur.fetchone()[0] or 0

            cur.execute(f"""
                SELECT
                    override_id, item_no, loc, override_month,
                    override_type, override_qty, override_multiplier,
                    override_additive_qty, is_hard_override,
                    override_reason, override_note, created_by, created_at,
                    valid_from, valid_to, approved_by, approved_at,
                    rejected_by, rejected_at, rejection_reason,
                    status, requires_approval, priority_rank,
                    statistical_qty_at_creation,
                    estimated_impact_units, estimated_impact_value, currency
                FROM fact_forecast_overrides
                {where}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """, params + [page_size, offset])
            rows = cur.fetchall()

    def _fmt(r):
        return {
            "override_id": r[0],
            "item_no": r[1],
            "loc": r[2],
            "override_month": r[3].isoformat() if r[3] else None,
            "override_type": r[4],
            "override_qty": float(r[5]) if r[5] is not None else None,
            "override_multiplier": float(r[6]) if r[6] is not None else None,
            "override_additive_qty": float(r[7]) if r[7] is not None else 0.0,
            "is_hard_override": r[8],
            "override_reason": r[9],
            "override_note": r[10],
            "created_by": r[11],
            "created_at": r[12].isoformat() if r[12] else None,
            "valid_from": r[13].isoformat() if r[13] else None,
            "valid_to": r[14].isoformat() if r[14] else None,
            "approved_by": r[15],
            "approved_at": r[16].isoformat() if r[16] else None,
            "rejected_by": r[17],
            "rejected_at": r[18].isoformat() if r[18] else None,
            "rejection_reason": r[19],
            "status": r[20],
            "requires_approval": r[21],
            "priority_rank": r[22],
            "statistical_qty_at_creation": float(r[23]) if r[23] is not None else None,
            "estimated_impact_units": float(r[24]) if r[24] is not None else None,
            "estimated_impact_value": float(r[25]) if r[25] is not None else None,
            "currency": r[26],
        }

    return {"total": int(total), "page": page, "overrides": [_fmt(r) for r in rows]}


# ---------------------------------------------------------------------------
# POST /forecast/overrides
# ---------------------------------------------------------------------------

@router.post("/forecast/overrides", status_code=201)
async def submit_override(body: OverrideSubmitRequest, request: Request):
    """Submit a new planner forecast override."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    impact_units, impact_value = _compute_impact(
        stat_qty=body.statistical_qty,
        multiplier=body.override_multiplier,
        additive=body.override_additive_qty,
        is_hard=body.is_hard_override,
        override_qty=body.override_qty,
    )

    requires_approval = _compute_requires_approval(
        impact_units=impact_units,
        impact_value=impact_value,
        multiplier=body.override_multiplier,
        stat_qty=body.statistical_qty,
    )

    initial_status = "pending_approval" if requires_approval else "approved"

    sql = """
        INSERT INTO fact_forecast_overrides
            (item_no, loc, override_month, override_type,
             override_qty, override_multiplier, override_additive_qty,
             is_hard_override, override_reason, override_note,
             created_by, valid_from, valid_to, priority_rank,
             status, requires_approval,
             statistical_qty_at_creation,
             estimated_impact_units, estimated_impact_value)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING override_id, status, requires_approval,
                  estimated_impact_units, estimated_impact_value
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                body.item_no, body.loc, body.override_month, body.override_type,
                body.override_qty, body.override_multiplier, body.override_additive_qty,
                body.is_hard_override, body.override_reason, body.override_note,
                body.created_by, body.valid_from, body.valid_to, body.priority_rank,
                initial_status, requires_approval,
                body.statistical_qty, impact_units, impact_value,
            ))
            row = cur.fetchone()
            conn.commit()

    msg = ("Override submitted. Pending approval from demand_manager role."
           if requires_approval else "Override approved automatically.")

    return {
        "override_id": row[0],
        "status": row[1],
        "requires_approval": row[2],
        "estimated_impact_units": float(row[3]) if row[3] is not None else None,
        "estimated_impact_value": float(row[4]) if row[4] is not None else None,
        "message": msg,
    }


# ---------------------------------------------------------------------------
# PUT /forecast/overrides/{id}/approve
# ---------------------------------------------------------------------------

@router.put("/forecast/overrides/{override_id}/approve")
async def approve_override(override_id: int, body: ApproveRequest, request: Request):
    """Manager approves a pending override."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    sql = """
        UPDATE fact_forecast_overrides
        SET status = 'approved',
            approved_by = %s,
            approved_at = NOW()
        WHERE override_id = %s
          AND status = 'pending_approval'
        RETURNING override_id, status, approved_by, approved_at
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (body.approved_by, override_id))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404,
                                    detail="Override not found or not in pending_approval state.")
            conn.commit()

    return {
        "override_id": row[0],
        "status": row[1],
        "approved_by": row[2],
        "approved_at": row[3].isoformat() if row[3] else None,
    }


# ---------------------------------------------------------------------------
# PUT /forecast/overrides/{id}/reject
# ---------------------------------------------------------------------------

@router.put("/forecast/overrides/{override_id}/reject")
async def reject_override(override_id: int, body: RejectRequest, request: Request):
    """Manager rejects a pending override."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    sql = """
        UPDATE fact_forecast_overrides
        SET status = 'rejected',
            rejected_by = %s,
            rejected_at = NOW(),
            rejection_reason = %s
        WHERE override_id = %s
          AND status = 'pending_approval'
        RETURNING override_id, status
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (body.rejected_by, body.rejection_reason, override_id))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404,
                                    detail="Override not found or not in pending_approval state.")
            conn.commit()

    return {"override_id": row[0], "status": row[1]}


# ---------------------------------------------------------------------------
# DELETE /forecast/overrides/{id}
# ---------------------------------------------------------------------------

@router.delete("/forecast/overrides/{override_id}")
async def delete_override(override_id: int, request: Request):
    """Soft-delete: sets override status to 'superseded'."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    sql = """
        UPDATE fact_forecast_overrides
        SET status = 'superseded'
        WHERE override_id = %s
          AND status NOT IN ('superseded', 'expired')
        RETURNING override_id, status
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (override_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Override not found or already closed.")
            conn.commit()

    return {"override_id": row[0], "status": row[1]}


# ---------------------------------------------------------------------------
# GET /forecast/consensus-plan
# ---------------------------------------------------------------------------

@router.get("/forecast/consensus-plan")
async def get_consensus_plan(
    item_no: str,
    loc: str,
    plan_version: str | None = None,
    month_from: str | None = None,
    month_to: str | None = None,
):
    """Retrieve the merged consensus plan (statistical baseline + approved overrides)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            if not plan_version:
                cur.execute("""
                    SELECT plan_version FROM fact_consensus_plan
                    WHERE item_no = %s AND loc = %s
                    ORDER BY generated_at DESC LIMIT 1
                """, [item_no, loc])
                row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No consensus plan found for {item_no}/{loc}."
                    )
                plan_version = row[0]

            params = [item_no, loc, plan_version]
            date_filters = ""
            if month_from:
                date_filters += " AND plan_month >= %s"; params.append(month_from)
            if month_to:
                date_filters += " AND plan_month <= %s"; params.append(month_to)

            cur.execute(f"""
                SELECT
                    plan_month, statistical_qty, statistical_p10, statistical_p90,
                    override_qty, consensus_qty, consensus_p10, consensus_p90,
                    override_applied, override_type, override_multiplier,
                    is_hard_override, overrider, approver, uplift_pct
                FROM fact_consensus_plan
                WHERE item_no = %s AND loc = %s AND plan_version = %s
                {date_filters}
                ORDER BY plan_month
            """, params)
            rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No consensus plan rows for {item_no}/{loc} version {plan_version}."
        )

    return {
        "plan_version": plan_version,
        "item_no": item_no,
        "loc": loc,
        "months": [
            {
                "plan_month": r[0].isoformat() if r[0] else None,
                "statistical_qty": float(r[1]) if r[1] is not None else None,
                "statistical_p10": float(r[2]) if r[2] is not None else None,
                "statistical_p90": float(r[3]) if r[3] is not None else None,
                "override_qty": float(r[4]) if r[4] is not None else 0.0,
                "consensus_qty": float(r[5]) if r[5] is not None else None,
                "consensus_p10": float(r[6]) if r[6] is not None else None,
                "consensus_p90": float(r[7]) if r[7] is not None else None,
                "override_applied": r[8],
                "override_type": r[9],
                "override_multiplier": float(r[10]) if r[10] is not None else None,
                "is_hard_override": r[11],
                "overrider": r[12],
                "approver": r[13],
                "uplift_pct": float(r[14]) if r[14] is not None else 0.0,
            }
            for r in rows
        ],
    }
