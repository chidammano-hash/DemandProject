"""Inventory Planning — Rebalancing: Cross-location transfer optimization endpoints."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import _f, get_conn, set_cache

router = APIRouter(tags=["Inventory Rebalancing"])

_VALID_PLAN_STATUSES = {
    "draft", "pending_approval", "approved", "partially_approved",
    "executing", "completed", "cancelled",
}
_VALID_TRANSFER_STATUSES = {
    "recommended", "approved", "rejected", "hold",
    "in_transit", "received", "cancelled",
}
_URGENCY_ORDER = "CASE urgency WHEN 'critical' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 ELSE 4 END"


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------
class LaneCreateBody(BaseModel):
    source_loc: str
    dest_loc: str
    transfer_mode: str = "truck"
    cost_per_unit: float
    handling_cost: float = 0
    freight_cost: float = 0
    receiving_cost: float = 0
    fixed_cost_per_shipment: float = 0
    transfer_lt_days: int = 3
    min_transfer_qty: int = 1
    max_transfer_qty: Optional[int] = None
    batch_size: int = 1
    max_shipments_per_week: int = 5
    max_receiving_units_per_period: Optional[int] = None


class TransferApproveBody(BaseModel):
    approved_by: str
    approved_qty: Optional[float] = None
    notes: Optional[str] = None


class TransferRejectBody(BaseModel):
    rejection_reason: str
    notes: Optional[str] = None


class ComputeBody(BaseModel):
    solver: str = "greedy"
    horizon_weeks: int = 4
    budget_cap: Optional[float] = None


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
@router.get("/inv-planning/rebalancing/kpis")
def get_rebalancing_kpis(response: FastAPIResponse) -> dict:
    """Network balance KPIs from mv_network_balance."""
    set_cache(response, max_age=120)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                                              AS total_items,
                    AVG(dos_cv)                                           AS avg_dos_cv,
                    SUM(CASE WHEN excess_loc_count > 0 AND shortage_loc_count > 0 THEN 1 ELSE 0 END)
                                                                          AS imbalanced_items,
                    SUM(excess_loc_count)                                  AS total_excess_locs,
                    SUM(shortage_loc_count)                                AS total_shortage_locs
                FROM mv_network_balance
            """)
            r = cur.fetchone() or (0, None, 0, 0, 0)

            # Latest plan summary
            cur.execute("""
                SELECT plan_id, total_transfer_qty, total_transfer_cost,
                       total_avoided_stockout_value, net_roi, items_rebalanced,
                       status, computation_date
                FROM fact_rebalancing_plan
                ORDER BY computation_date DESC, created_ts DESC
                LIMIT 1
            """)
            plan = cur.fetchone()

    kpis = {
        "total_multi_loc_items": int(r[0] or 0),
        "avg_dos_cv": _f(r[1]),
        "network_balance_score": round((1 - float(r[1] or 0)) * 100, 1) if r[1] is not None else None,
        "imbalanced_items": int(r[2] or 0),
        "total_excess_locs": int(r[3] or 0),
        "total_shortage_locs": int(r[4] or 0),
    }

    if plan:
        kpis["latest_plan"] = {
            "plan_id": plan[0],
            "total_transfer_qty": _f(plan[1]),
            "total_transfer_cost": _f(plan[2]),
            "total_avoided_stockout_value": _f(plan[3]),
            "net_roi": _f(plan[4]),
            "items_rebalanced": int(plan[5] or 0),
            "status": plan[6],
            "computation_date": plan[7].isoformat() if plan[7] else None,
        }
    else:
        kpis["latest_plan"] = None

    return kpis


# ---------------------------------------------------------------------------
# Network topology
# ---------------------------------------------------------------------------
@router.get("/inv-planning/rebalancing/network")
def list_lanes(
    response: FastAPIResponse,
    source_loc: Optional[str] = None,
    dest_loc: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """List active transfer lanes."""
    set_cache(response, max_age=300)

    wheres = ["is_active = TRUE"]
    params: list = []
    if source_loc:
        wheres.append("source_loc ILIKE %s")
        params.append(f"%{source_loc}%")
    if dest_loc:
        wheres.append("dest_loc ILIKE %s")
        params.append(f"%{dest_loc}%")

    where = " AND ".join(wheres)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM dim_transfer_lane WHERE {where}", params)
            total = (cur.fetchone() or (0,))[0]
            cur.execute(f"""
                SELECT lane_id, source_loc, dest_loc, transfer_mode,
                       cost_per_unit, handling_cost, freight_cost, receiving_cost,
                       fixed_cost_per_shipment, transfer_lt_days,
                       min_transfer_qty, max_transfer_qty, batch_size
                FROM dim_transfer_lane
                WHERE {where}
                ORDER BY source_loc, dest_loc
                LIMIT %s OFFSET %s
            """, params + [limit, offset])
            rows = cur.fetchall()

    return {
        "total": int(total),
        "rows": [
            {
                "lane_id": r[0], "source_loc": r[1], "dest_loc": r[2],
                "transfer_mode": r[3], "cost_per_unit": _f(r[4]),
                "handling_cost": _f(r[5]), "freight_cost": _f(r[6]),
                "receiving_cost": _f(r[7]), "fixed_cost_per_shipment": _f(r[8]),
                "transfer_lt_days": int(r[9]),
                "min_transfer_qty": int(r[10] or 1),
                "max_transfer_qty": int(r[11]) if r[11] else None,
                "batch_size": int(r[12] or 1),
            }
            for r in rows
        ],
    }


@router.post("/inv-planning/rebalancing/network")
def create_lane(body: LaneCreateBody, _: None = Depends(require_api_key)) -> dict:
    """Create or update a transfer lane."""
    sql = """
        INSERT INTO dim_transfer_lane (
            source_loc, dest_loc, transfer_mode,
            cost_per_unit, handling_cost, freight_cost, receiving_cost,
            fixed_cost_per_shipment, transfer_lt_days,
            min_transfer_qty, max_transfer_qty, batch_size,
            max_shipments_per_week, max_receiving_units_per_period
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (source_loc, dest_loc, transfer_mode)
        DO UPDATE SET
            cost_per_unit = EXCLUDED.cost_per_unit,
            handling_cost = EXCLUDED.handling_cost,
            freight_cost = EXCLUDED.freight_cost,
            receiving_cost = EXCLUDED.receiving_cost,
            fixed_cost_per_shipment = EXCLUDED.fixed_cost_per_shipment,
            transfer_lt_days = EXCLUDED.transfer_lt_days,
            min_transfer_qty = EXCLUDED.min_transfer_qty,
            max_transfer_qty = EXCLUDED.max_transfer_qty,
            batch_size = EXCLUDED.batch_size,
            max_shipments_per_week = EXCLUDED.max_shipments_per_week,
            max_receiving_units_per_period = EXCLUDED.max_receiving_units_per_period,
            modified_ts = NOW()
        RETURNING lane_id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [
                body.source_loc, body.dest_loc, body.transfer_mode,
                body.cost_per_unit, body.handling_cost, body.freight_cost,
                body.receiving_cost, body.fixed_cost_per_shipment, body.transfer_lt_days,
                body.min_transfer_qty, body.max_transfer_qty, body.batch_size,
                body.max_shipments_per_week, body.max_receiving_units_per_period,
            ])
            row = cur.fetchone()
        conn.commit()

    return {"lane_id": row[0] if row else None, "status": "created"}


@router.delete("/inv-planning/rebalancing/network/{lane_id}")
def deactivate_lane(lane_id: str, _: None = Depends(require_api_key)) -> dict:
    """Soft-delete a transfer lane."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE dim_transfer_lane SET is_active = FALSE, modified_ts = NOW() WHERE lane_id = %s RETURNING lane_id",
                [lane_id],
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Lane not found")
    return {"lane_id": row[0], "status": "deactivated"}


# ---------------------------------------------------------------------------
# Imbalance detection
# ---------------------------------------------------------------------------
@router.get("/inv-planning/rebalancing/imbalances")
def get_imbalances(
    response: FastAPIResponse,
    item: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Detect items with simultaneous excess and shortage across locations."""
    set_cache(response, max_age=60)

    wheres = ["excess_loc_count > 0", "shortage_loc_count > 0"]
    params: list = []
    if item:
        wheres.append("item_id ILIKE %s")
        params.append(f"%{item}%")

    where = " AND ".join(wheres)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM mv_network_balance WHERE {where}", params)
            total = (cur.fetchone() or (0,))[0]
            cur.execute(f"""
                SELECT item_id, location_count, avg_on_hand, avg_dos, dos_cv,
                       excess_loc_count, shortage_loc_count
                FROM mv_network_balance
                WHERE {where}
                ORDER BY dos_cv DESC NULLS LAST
                LIMIT %s OFFSET %s
            """, params + [limit, offset])
            rows = cur.fetchall()

    return {
        "total": int(total),
        "rows": [
            {
                "item_id": r[0], "location_count": int(r[1]),
                "avg_on_hand": _f(r[2]), "avg_dos": _f(r[3]),
                "dos_cv": _f(r[4]),
                "excess_loc_count": int(r[5] or 0),
                "shortage_loc_count": int(r[6] or 0),
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Compute (async)
# ---------------------------------------------------------------------------
@router.post("/inv-planning/rebalancing/compute", status_code=202)
def trigger_compute(
    body: ComputeBody,
    _: None = Depends(require_api_key),
) -> dict:
    """Trigger rebalancing computation in background. Returns 202."""
    from concurrent.futures import ThreadPoolExecutor

    _executor = ThreadPoolExecutor(max_workers=1)

    def _run():
        from scripts.compute_rebalancing import run as compute_run
        compute_run(
            solver=body.solver,
            horizon=body.horizon_weeks,
            budget_cap=body.budget_cap,
        )

    _executor.submit(_run)
    return {"status": "accepted", "message": "Rebalancing computation started"}


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
@router.get("/inv-planning/rebalancing/plans")
def list_plans(
    response: FastAPIResponse,
    status: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
) -> dict:
    """List rebalancing plans by date."""
    set_cache(response, max_age=30)

    wheres: list[str] = []
    params: list = []
    if status and status in _VALID_PLAN_STATUSES:
        wheres.append("status = %s")
        params.append(status)

    where = ("WHERE " + " AND ".join(wheres)) if wheres else ""

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM fact_rebalancing_plan {where}", params)
            total = (cur.fetchone() or (0,))[0]
            cur.execute(f"""
                SELECT plan_id, computation_date, solver_method, objective,
                       total_transfer_qty, total_transfer_cost,
                       total_avoided_stockout_value, net_roi,
                       items_rebalanced, lanes_used, status, solver_runtime_ms,
                       created_ts
                FROM fact_rebalancing_plan {where}
                ORDER BY computation_date DESC, created_ts DESC
                LIMIT %s OFFSET %s
            """, params + [limit, offset])
            rows = cur.fetchall()

    return {
        "total": int(total),
        "rows": [
            {
                "plan_id": r[0],
                "computation_date": r[1].isoformat() if r[1] else None,
                "solver_method": r[2], "objective": r[3],
                "total_transfer_qty": _f(r[4]),
                "total_transfer_cost": _f(r[5]),
                "total_avoided_stockout_value": _f(r[6]),
                "net_roi": _f(r[7]),
                "items_rebalanced": int(r[8] or 0),
                "lanes_used": int(r[9] or 0),
                "status": r[10],
                "solver_runtime_ms": int(r[11] or 0),
                "created_ts": r[12].isoformat() if r[12] else None,
            }
            for r in rows
        ],
    }


@router.get("/inv-planning/rebalancing/plans/{plan_id}")
def get_plan_detail(plan_id: str, response: FastAPIResponse) -> dict:
    """Get a single plan with summary KPIs."""
    set_cache(response, max_age=30)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT plan_id, computation_date, horizon_weeks, solver_method, objective,
                       total_transfer_qty, total_transfer_cost,
                       total_avoided_stockout_value, net_roi,
                       network_balance_before, network_balance_after,
                       items_rebalanced, lanes_used, status,
                       approved_by, approved_ts, solver_runtime_ms, created_ts
                FROM fact_rebalancing_plan WHERE plan_id = %s
            """, [plan_id])
            r = cur.fetchone()

    if not r:
        raise HTTPException(status_code=404, detail="Plan not found")

    return {
        "plan_id": r[0],
        "computation_date": r[1].isoformat() if r[1] else None,
        "horizon_weeks": int(r[2] or 4),
        "solver_method": r[3], "objective": r[4],
        "total_transfer_qty": _f(r[5]),
        "total_transfer_cost": _f(r[6]),
        "total_avoided_stockout_value": _f(r[7]),
        "net_roi": _f(r[8]),
        "network_balance_before": _f(r[9]),
        "network_balance_after": _f(r[10]),
        "items_rebalanced": int(r[11] or 0),
        "lanes_used": int(r[12] or 0),
        "status": r[13],
        "approved_by": r[14],
        "approved_ts": r[15].isoformat() if r[15] else None,
        "solver_runtime_ms": int(r[16] or 0),
        "created_ts": r[17].isoformat() if r[17] else None,
    }


@router.get("/inv-planning/rebalancing/plans/{plan_id}/transfers")
def list_plan_transfers(
    plan_id: str,
    response: FastAPIResponse,
    urgency: Optional[str] = None,
    status: Optional[str] = None,
    item: Optional[str] = None,
    sort_by: str = "priority_score",
    sort_dir: str = "desc",
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """Paginated list of transfers for a plan."""
    set_cache(response, max_age=30)

    wheres = ["plan_id = %s"]
    params: list = [plan_id]
    if urgency and urgency in {"critical", "high", "medium", "low"}:
        wheres.append("urgency = %s")
        params.append(urgency)
    if status and status in _VALID_TRANSFER_STATUSES:
        wheres.append("status = %s")
        params.append(status)
    if item:
        wheres.append("item_id ILIKE %s")
        params.append(f"%{item}%")

    where = " AND ".join(wheres)

    valid_sort = {"priority_score", "recommended_qty", "transfer_cost", "net_benefit", "roi", "urgency", "item_id"}
    col = sort_by if sort_by in valid_sort else "priority_score"
    direction = "ASC" if sort_dir.lower() == "asc" else "DESC"
    if col == "urgency":
        order = f"{_URGENCY_ORDER} {direction}"
    else:
        order = f"{col} {direction} NULLS LAST"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM fact_rebalancing_transfer WHERE {where}", params)
            total = (cur.fetchone() or (0,))[0]
            cur.execute(f"""
                SELECT transfer_id, item_id, source_loc, dest_loc, transfer_mode,
                       recommended_qty, approved_qty,
                       source_on_hand, source_dos, source_ss_target, source_excess_qty,
                       dest_on_hand, dest_dos, dest_ss_target, dest_shortage_qty,
                       transfer_cost, carrying_cost_saved, stockout_cost_avoided,
                       net_benefit, roi,
                       planned_ship_date, expected_arrival_date, transfer_lt_days,
                       priority_score, abc_class, urgency, status,
                       approved_by, rejection_reason, notes
                FROM fact_rebalancing_transfer
                WHERE {where}
                ORDER BY {order}
                LIMIT %s OFFSET %s
            """, params + [limit, offset])
            rows = cur.fetchall()

    return {
        "total": int(total),
        "rows": [
            {
                "transfer_id": r[0], "item_id": r[1],
                "source_loc": r[2], "dest_loc": r[3],
                "transfer_mode": r[4],
                "recommended_qty": _f(r[5]), "approved_qty": _f(r[6]),
                "source_on_hand": _f(r[7]), "source_dos": _f(r[8]),
                "source_ss_target": _f(r[9]), "source_excess_qty": _f(r[10]),
                "dest_on_hand": _f(r[11]), "dest_dos": _f(r[12]),
                "dest_ss_target": _f(r[13]), "dest_shortage_qty": _f(r[14]),
                "transfer_cost": _f(r[15]),
                "carrying_cost_saved": _f(r[16]),
                "stockout_cost_avoided": _f(r[17]),
                "net_benefit": _f(r[18]), "roi": _f(r[19]),
                "planned_ship_date": r[20].isoformat() if r[20] else None,
                "expected_arrival_date": r[21].isoformat() if r[21] else None,
                "transfer_lt_days": int(r[22] or 0),
                "priority_score": _f(r[23]),
                "abc_class": r[24], "urgency": r[25], "status": r[26],
                "approved_by": r[27], "rejection_reason": r[28], "notes": r[29],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Transfer workflow
# ---------------------------------------------------------------------------
@router.post("/inv-planning/rebalancing/transfers/{transfer_id}/approve")
def approve_transfer(
    transfer_id: str,
    body: TransferApproveBody,
    _: None = Depends(require_api_key),
) -> dict:
    """Approve a transfer recommendation."""
    import datetime as _dt

    now = _dt.datetime.now(_dt.timezone.utc)
    sql = """
        UPDATE fact_rebalancing_transfer
        SET status = 'approved',
            approved_by = %s,
            approved_qty = COALESCE(%s, recommended_qty),
            approved_ts = %s,
            notes = COALESCE(%s, notes),
            modified_ts = %s
        WHERE transfer_id = %s AND status = 'recommended'
        RETURNING transfer_id, item_id, source_loc, dest_loc, approved_qty, status
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [body.approved_by, body.approved_qty, now, body.notes, now, transfer_id])
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Transfer not found or already processed")

    return {
        "transfer_id": row[0], "item_id": row[1],
        "source_loc": row[2], "dest_loc": row[3],
        "approved_qty": _f(row[4]), "status": row[5],
    }


@router.post("/inv-planning/rebalancing/transfers/{transfer_id}/reject")
def reject_transfer(
    transfer_id: str,
    body: TransferRejectBody,
    _: None = Depends(require_api_key),
) -> dict:
    """Reject a transfer with reason."""
    import datetime as _dt

    if not body.rejection_reason or not body.rejection_reason.strip():
        raise HTTPException(status_code=422, detail="rejection_reason is required")

    now = _dt.datetime.now(_dt.timezone.utc)
    sql = """
        UPDATE fact_rebalancing_transfer
        SET status = 'rejected',
            rejection_reason = %s,
            notes = COALESCE(%s, notes),
            modified_ts = %s
        WHERE transfer_id = %s AND status IN ('recommended', 'hold')
        RETURNING transfer_id, item_id, source_loc, dest_loc, status
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [body.rejection_reason, body.notes, now, transfer_id])
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=404, detail="Transfer not found or already processed")

    return {
        "transfer_id": row[0], "item_id": row[1],
        "source_loc": row[2], "dest_loc": row[3], "status": row[4],
    }


@router.post("/inv-planning/rebalancing/plans/{plan_id}/approve-all")
def approve_all_transfers(
    plan_id: str,
    body: TransferApproveBody,
    _: None = Depends(require_api_key),
) -> dict:
    """Bulk approve all recommended transfers in a plan."""
    import datetime as _dt

    now = _dt.datetime.now(_dt.timezone.utc)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE fact_rebalancing_transfer
                SET status = 'approved',
                    approved_by = %s,
                    approved_qty = recommended_qty,
                    approved_ts = %s,
                    modified_ts = %s
                WHERE plan_id = %s AND status = 'recommended'
            """, [body.approved_by, now, now, plan_id])
            count = cur.rowcount

            cur.execute("""
                UPDATE fact_rebalancing_plan
                SET status = 'approved',
                    approved_by = %s,
                    approved_ts = %s,
                    modified_ts = %s
                WHERE plan_id = %s
            """, [body.approved_by, now, now, plan_id])
        conn.commit()

    return {"plan_id": plan_id, "approved_count": count, "status": "approved"}
