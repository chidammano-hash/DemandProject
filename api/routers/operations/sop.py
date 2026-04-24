"""
F4.2 — Sales & Operations Planning (S&OP) Module API endpoints.

Endpoints:
    GET  /sop/cycles                         — List S&OP cycles
    GET  /sop/cycles/{cycle_id}              — Full cycle detail
    POST /sop/cycles/{cycle_id}/advance      — Advance to next stage (auth)
    POST /sop/cycles/{cycle_id}/approve      — Executive approval + publish plan (auth)
    GET  /sop/cycles/{cycle_id}/gaps         — Gap analysis
    GET  /sop/approved-plan                  — Locked approved demand
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from api.core import get_conn
from api.auth import require_api_key

router = APIRouter(tags=["sop"])

STAGE_ORDER = [
    "demand_review",
    "supply_review",
    "pre_sop",
    "executive_sop",
    "approved",
    "closed",
]


def _next_stage(current: str) -> str:
    """Return the next S&OP stage after the current one."""
    try:
        idx = STAGE_ORDER.index(current)
    except ValueError:
        raise ValueError(f"Unknown stage: {current}")
    if idx >= len(STAGE_ORDER) - 1:
        raise ValueError(f"Stage '{current}' is already the final stage")
    return STAGE_ORDER[idx + 1]


class _AdvanceRequest(BaseModel):
    facilitated_by: str
    notes: Optional[str] = None


class _ApproveRequest(BaseModel):
    approved_by: str
    plan_version: str


@router.get("/sop/cycles")
async def list_sop_cycles(page: int = 1, page_size: int = 20):
    """List all S&OP cycles with status."""
    page_size = max(1, min(page_size, 100))
    offset = (max(1, page) - 1) * page_size

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM fact_sop_cycles")
            total = cur.fetchone()[0]
            cur.execute(
                """
                SELECT cycle_id, cycle_month, status, facilitated_by,
                       approved_by, approved_plan_version, created_at, updated_at
                FROM fact_sop_cycles
                ORDER BY cycle_month DESC
                LIMIT %s OFFSET %s
                """,
                (page_size, offset),
            )
            rows = cur.fetchall()

    cols = [
        "cycle_id", "cycle_month", "current_stage", "facilitated_by",
        "approved_by", "approved_plan_version", "created_at", "updated_at",
    ]
    cycles = []
    for r in rows:
        d = dict(zip(cols, r))
        for field in ("cycle_month", "created_at", "updated_at"):
            if d.get(field) and hasattr(d[field], "isoformat"):
                d[field] = d[field].isoformat()
        cycles.append(d)

    return {"total": total, "page": page, "cycles": cycles}


@router.get("/sop/cycles/{cycle_id}")
async def get_sop_cycle(cycle_id: int):
    """Full cycle detail including demand review, supply constraints, and gaps."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT cycle_id, cycle_month, status, facilitated_by,
                       approved_by, approved_plan_version, created_at, updated_at
                FROM fact_sop_cycles WHERE cycle_id = %s
                """,
                (cycle_id,),
            )
            cycle_row = cur.fetchone()
            if not cycle_row:
                raise HTTPException(404, f"Cycle {cycle_id} not found")

            cur.execute(
                """
                SELECT item_category, statistical_demand_qty, commercial_demand_qty,
                       consensus_demand_qty, review_status
                FROM fact_sop_demand_review WHERE cycle_id = %s
                ORDER BY item_category
                """,
                (cycle_id,),
            )
            demand_rows = cur.fetchall()

            cur.execute(
                """
                SELECT constraint_type, supplier_id, impact_qty, impact_period, mitigation_status
                FROM fact_sop_supply_constraints WHERE cycle_id = %s
                ORDER BY constraint_type
                """,
                (cycle_id,),
            )
            constraint_rows = cur.fetchall()

    cycle_cols = [
        "cycle_id", "cycle_month", "current_stage", "facilitated_by",
        "approved_by", "approved_plan_version", "created_at", "updated_at",
    ]
    cycle = dict(zip(cycle_cols, cycle_row))
    for field in ("cycle_month", "created_at", "updated_at"):
        if cycle.get(field) and hasattr(cycle[field], "isoformat"):
            cycle[field] = cycle[field].isoformat()

    demand_cols = [
        "item_category", "statistical_demand_qty", "commercial_demand_qty",
        "consensus_demand_qty", "review_status",
    ]
    cycle["demand_review"] = [dict(zip(demand_cols, r)) for r in demand_rows]

    constraint_cols = [
        "constraint_type", "supplier_id", "impact_qty", "impact_period", "mitigation_status",
    ]
    cycle["supply_constraints"] = [dict(zip(constraint_cols, r)) for r in constraint_rows]

    return cycle


@router.post("/sop/cycles/{cycle_id}/advance")
async def advance_sop_cycle(cycle_id: int, body: _AdvanceRequest, request: Request):
    """Advance cycle to next stage (auth required)."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM fact_sop_cycles WHERE cycle_id = %s", (cycle_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, f"Cycle {cycle_id} not found")
            current_status = row[0]
            try:
                next_status = _next_stage(current_status)
            except ValueError as e:
                raise HTTPException(400, str(e))

            cur.execute(
                """
                UPDATE fact_sop_cycles
                SET status = %s, facilitated_by = %s, updated_at = NOW()
                WHERE cycle_id = %s
                """,
                (next_status, body.facilitated_by, cycle_id),
            )
        conn.commit()

    return {"cycle_id": cycle_id, "previous_status": current_status, "new_status": next_status}


@router.post("/sop/cycles/{cycle_id}/approve")
async def approve_sop_cycle(cycle_id: int, body: _ApproveRequest, request: Request):
    """Executive approval: lock the plan and publish to approved-plan table (auth required)."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM fact_sop_cycles WHERE cycle_id = %s", (cycle_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, f"Cycle {cycle_id} not found")

            cur.execute(
                """
                UPDATE fact_sop_cycles
                SET status = 'approved', approved_by = %s,
                    approved_plan_version = %s, updated_at = NOW()
                WHERE cycle_id = %s
                """,
                (body.approved_by, body.plan_version, cycle_id),
            )
        conn.commit()

    return {
        "cycle_id": cycle_id,
        "status": "approved",
        "approved_by": body.approved_by,
        "plan_version": body.plan_version,
    }


@router.get("/sop/cycles/{cycle_id}/gaps")
async def get_sop_gaps(
    cycle_id: int,
    severity: str | None = None,
    resolution_status: str = "open",
):
    """Supply-demand gap analysis for a cycle."""
    conditions = ["cycle_id = %s", "resolution_status = %s"]
    params: list = [cycle_id, resolution_status]
    if severity:
        conditions.append("severity = %s")
        params.append(severity)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT gap_id, gap_type, gap_qty, gap_value, severity,
                       resolution_options, resolution_status
                FROM fact_sop_gaps
                WHERE {where}
                ORDER BY CASE severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2 ELSE 3 END
                """,
                params,
            )
            rows = cur.fetchall()

    cols = ["gap_id", "gap_type", "gap_qty", "gap_value", "severity",
            "resolution_options", "resolution_status"]
    return {"cycle_id": cycle_id, "gaps": [dict(zip(cols, r)) for r in rows]}


@router.get("/sop/approved-plan")
async def get_approved_plan(
    cycle_id: int | None = None,
    item_id: str | None = None,
    loc: str | None = None,
    category: str | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """Locked approved demand from the most recent approved S&OP cycle."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    conditions = ["1=1"]
    params: list = []
    if cycle_id:
        conditions.append("cycle_id = %s")
        params.append(cycle_id)
    if item_id:
        conditions.append("item_id = %s")
        params.append(item_id)
    if loc:
        conditions.append("loc = %s")
        params.append(loc)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM fact_sop_approved_plan WHERE {where}", params
            )
            total = cur.fetchone()[0]
            cur.execute(
                f"""
                SELECT cycle_id, item_id, loc, plan_month, approved_qty,
                       statistical_qty, override_qty, source, locked
                FROM fact_sop_approved_plan
                WHERE {where}
                ORDER BY plan_month, item_id, loc
                LIMIT %s OFFSET %s
                """,
                params + [page_size, offset],
            )
            rows = cur.fetchall()

    cols = [
        "cycle_id", "item_id", "loc", "plan_month", "approved_qty",
        "statistical_qty", "override_qty", "source", "locked",
    ]
    items = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("plan_month") and hasattr(d["plan_month"], "isoformat"):
            d["plan_month"] = d["plan_month"].isoformat()
        items.append(d)

    return {"total": total, "page": page, "approved_plan": items}
