"""F1.2 — Forward Inventory Projection API endpoints.

Endpoints:
    GET  /inv-planning/projection              — DFU projection curves (all 3 scenarios)
    GET  /inv-planning/projection/at-risk      — DFUs with near-term stockout risk
    POST /inv-planning/projection/refresh      — On-demand recompute for one DFU
"""
from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import get_conn

router = APIRouter(tags=["inv-planning-projection"])

_SCENARIOS = ("no_order", "with_open_po", "with_planned_orders")


# ---------------------------------------------------------------------------
# GET /inv-planning/projection
# ---------------------------------------------------------------------------

@router.get("/inv-planning/projection")
async def get_projection(
    item_no: str,
    loc: str,
    horizon_days: int = 90,
    scenario: str = "all",
):
    """Return day-by-day projection curves for a single DFU."""
    horizon_days = max(1, min(horizon_days, 365))

    scenarios_filter = _SCENARIOS if scenario == "all" else (scenario,)

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Latest run_id for this DFU
            cur.execute("""
                SELECT DISTINCT projection_run_id, forecast_source, plan_version
                FROM fact_inventory_projection
                WHERE item_no = %s AND loc = %s
                ORDER BY 1 DESC
                LIMIT 1
            """, [item_no, loc])
            run_row = cur.fetchone()
            if not run_row:
                raise HTTPException(
                    status_code=404,
                    detail=f"No projection found for {item_no}/{loc}. "
                           f"Run 'make projection-compute-dfu ITEM={item_no} LOC={loc}'."
                )
            run_id, forecast_source, plan_version = run_row

            # Key dates from summary view
            cur.execute("""
                SELECT scenario, reorder_trigger_date, stockout_date,
                       excess_date, days_until_stockout, last_computed_at
                FROM mv_inventory_projection_summary
                WHERE item_no = %s AND loc = %s AND projection_run_id = %s
                  AND scenario = ANY(%s::varchar[])
            """, [item_no, loc, str(run_id), list(scenarios_filter)])
            summary_rows = cur.fetchall()

            # Current inventory snapshot
            cur.execute("""
                SELECT qty_on_hand
                FROM fact_inventory_snapshot
                WHERE item_no = %s AND loc = %s
                ORDER BY snapshot_date DESC LIMIT 1
            """, [item_no, loc])
            inv_row = cur.fetchone()
            current_qty = float(inv_row[0] or 0) if inv_row else 0.0

            # Safety stock
            cur.execute("""
                SELECT ss_combined FROM fact_safety_stock_targets
                WHERE item_no = %s AND loc = %s
                ORDER BY computed_at DESC LIMIT 1
            """, [item_no, loc])
            ss_row = cur.fetchone()
            safety_stock = float(ss_row[0] or 0) if ss_row else 0.0

            # Check if open PO data available
            cur.execute("SELECT COUNT(*) FROM fact_open_purchase_orders")
            po_count = (cur.fetchone() or (0,))[0]
            open_po_data_available = (po_count or 0) > 0

            # Projection data — pivot scenarios into columns per date
            cur.execute("""
                SELECT projection_date, scenario, projected_qty, daily_demand_rate,
                       receipts_expected, reorder_triggered, stockout_risk, excess_risk
                FROM fact_inventory_projection
                WHERE item_no = %s AND loc = %s AND projection_run_id = %s
                  AND scenario = ANY(%s::varchar[])
                  AND projection_date <= CURRENT_DATE + %s
                ORDER BY projection_date, scenario
            """, [item_no, loc, str(run_id), list(scenarios_filter), horizon_days])
            proj_rows = cur.fetchall()

    # Build key_dates dict
    key_dates = {}
    computed_at = None
    for r in summary_rows:
        sce, rt, so, ex, dus, ca = r
        key_dates[sce] = {
            "reorder_trigger_date": rt.isoformat() if rt else None,
            "stockout_date": so.isoformat() if so else None,
            "days_until_stockout": int(dus) if dus is not None else None,
            "excess_date": ex.isoformat() if ex else None,
        }
        if ca and not computed_at:
            computed_at = ca.isoformat()

    # Pivot projection rows by date
    by_date: dict[date, dict] = {}
    for r in proj_rows:
        proj_date, sce, qty, dr, rcpt, reorder, so, excess = r
        if proj_date not in by_date:
            by_date[proj_date] = {
                "projection_date": proj_date.isoformat(),
                "daily_demand_rate": float(dr or 0),
                "receipts_expected": float(rcpt or 0),
            }
        by_date[proj_date][f"{sce}_qty"] = float(qty or 0)
        by_date[proj_date][f"{sce}_stockout_risk"] = bool(so)
        by_date[proj_date][f"{sce}_reorder_triggered"] = bool(reorder)

    projection = sorted(by_date.values(), key=lambda x: x["projection_date"])

    return {
        "item_no": item_no,
        "loc": loc,
        "current_qty_on_hand": current_qty,
        "safety_stock": safety_stock,
        "reorder_point": safety_stock,
        "forecast_source": forecast_source,
        "plan_version": plan_version,
        "open_po_data_available": open_po_data_available,
        "computed_at": computed_at,
        "key_dates": key_dates,
        "projection": projection,
    }


# ---------------------------------------------------------------------------
# GET /inv-planning/projection/at-risk
# ---------------------------------------------------------------------------

@router.get("/inv-planning/projection/at-risk")
async def get_projection_at_risk(
    horizon_days: int = 30,
    page: int = 1,
    page_size: int = 50,
):
    """Return DFUs with stockout risk within N days in the with_open_po scenario."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(DISTINCT (s.item_no, s.loc))
                FROM mv_inventory_projection_summary s
                WHERE s.scenario = 'with_open_po'
                  AND s.stockout_date IS NOT NULL
                  AND s.days_until_stockout <= %s
                  AND s.days_until_stockout >= 0
            """, [horizon_days])
            total = (cur.fetchone() or (0,))[0] or 0

            cur.execute("""
                SELECT
                    s.item_no,
                    s.loc,
                    s.stockout_date,
                    s.days_until_stockout,
                    s.reorder_trigger_date,
                    i.qty_on_hand AS current_qty,
                    ss.ss_combined AS safety_stock,
                    CASE
                        WHEN s.days_until_stockout <= 7  THEN 'critical'
                        WHEN s.days_until_stockout <= 14 THEN 'high'
                        ELSE 'medium'
                    END AS severity
                FROM mv_inventory_projection_summary s
                LEFT JOIN LATERAL (
                    SELECT qty_on_hand FROM fact_inventory_snapshot
                    WHERE item_no = s.item_no AND loc = s.loc
                    ORDER BY snapshot_date DESC LIMIT 1
                ) i ON TRUE
                LEFT JOIN LATERAL (
                    SELECT ss_combined FROM fact_safety_stock_targets
                    WHERE item_no = s.item_no AND loc = s.loc
                    ORDER BY computed_at DESC LIMIT 1
                ) ss ON TRUE
                WHERE s.scenario = 'with_open_po'
                  AND s.stockout_date IS NOT NULL
                  AND s.days_until_stockout <= %s
                  AND s.days_until_stockout >= 0
                ORDER BY s.days_until_stockout ASC, s.item_no
                LIMIT %s OFFSET %s
            """, [horizon_days, page_size, offset])
            rows = cur.fetchall()

    return {
        "total": int(total),
        "horizon_days": horizon_days,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "item_no": r[0],
                "loc": r[1],
                "stockout_date": r[2].isoformat() if r[2] else None,
                "days_until_stockout": int(r[3]) if r[3] is not None else None,
                "reorder_trigger_date": r[4].isoformat() if r[4] else None,
                "current_qty": float(r[5] or 0),
                "safety_stock": float(r[6] or 0),
                "severity": r[7],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# POST /inv-planning/projection/refresh
# ---------------------------------------------------------------------------

class ProjectionRefreshRequest(BaseModel):
    item_no: str
    loc: str
    horizon_days: int = 90


@router.post("/inv-planning/projection/refresh")
async def refresh_projection(body: ProjectionRefreshRequest, api_key: str = Depends(require_api_key)):
    """Trigger a synchronous projection recompute for one DFU."""
    import yaml
    from pathlib import Path
    from scripts.compute_inventory_projection import (
        compute_dfu_projection, refresh_summary_view,
    )

    try:
        with open("config/projection_config.yaml") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {
            "projection": {"horizon_days": 90, "scenarios": ["no_order", "with_open_po", "with_planned_orders"]},
            "thresholds": {"reorder_point_source": "safety_stock", "excess_coverage_months": 6},
        }

    with get_conn() as conn:
        written, run_id = compute_dfu_projection(
            body.item_no, body.loc,
            body.horizon_days, config, conn,
            dry_run=False,
        )
        try:
            refresh_summary_view(conn)
        except Exception:
            pass

    return {
        "status": "ok",
        "item_no": body.item_no,
        "loc": body.loc,
        "rows_written": written,
        "run_id": run_id,
    }
