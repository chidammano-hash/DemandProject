"""
F3.4 — Demand Sensing Integration (Blended Forecast) API endpoints.

Endpoints:
    GET /forecast/blended        — Weekly blended forecast for a DFU
    GET /forecast/blended/summary — Portfolio sensing status
    GET /forecast/sensing-active  — DFUs where sensing currently overrides statistical
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.core import get_conn

router = APIRouter(tags=["blended-forecast"])


@router.get("/forecast/blended")
async def get_blended_forecast(
    item_no: str | None = None,
    loc: str | None = None,
    weeks: int = 8,
    plan_version: str | None = None,
):
    """Weekly blended forecast combining demand sensing signal + statistical model."""
    if not item_no or not loc:
        raise HTTPException(400, "item_no and loc are required")

    weeks = max(1, min(weeks, 52))
    conditions = ["item_no = %s", "loc = %s"]
    params: list = [item_no, loc]
    if plan_version:
        conditions.append("plan_version = %s"); params.append(plan_version)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT week_start, alpha_weight, sensing_signal_qty,
                       statistical_forecast_qty, blended_qty,
                       velocity_spike_ratio, is_outlier_capped, plan_version
                FROM fact_blended_demand_plan
                WHERE {where}
                ORDER BY week_start
                LIMIT %s
                """,
                params + [weeks],
            )
            rows = cur.fetchall()

    cols = [
        "week_start", "alpha_weight", "sensing_signal_qty",
        "statistical_forecast_qty", "blended_qty",
        "velocity_spike_ratio", "is_outlier_capped", "plan_version",
    ]
    weeks_data = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("week_start"):
            d["week_start"] = d["week_start"].isoformat()
        weeks_data.append(d)

    monthly_total = sum(
        float(d.get("blended_qty") or 0) for d in weeks_data
    )
    return {
        "item_no": item_no,
        "loc": loc,
        "weeks": weeks,
        "plan_version": plan_version,
        "weekly_forecast": weeks_data,
        "monthly_total_blended": round(monthly_total, 2),
    }


@router.get("/forecast/blended/summary")
async def get_blended_summary():
    """Portfolio-level blended forecast status: active overrides, top spikes."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(DISTINCT item_no || '@' || loc)          AS total_dfus,
                    SUM(CASE WHEN alpha_weight > 0.3 THEN 1 ELSE 0 END) AS sensing_active_count,
                    AVG(velocity_spike_ratio)                      AS avg_spike_ratio,
                    COUNT(CASE WHEN is_outlier_capped THEN 1 END)  AS capped_count
                FROM fact_blended_demand_plan
                WHERE week_start >= CURRENT_DATE
            """)
            row = cur.fetchone()

    if not row:
        return {"total_dfus": 0}

    return {
        "total_dfus": row[0] or 0,
        "sensing_active_count": row[1] or 0,
        "avg_spike_ratio": float(row[2]) if row[2] is not None else None,
        "outlier_capped_count": row[3] or 0,
    }


@router.get("/forecast/sensing-active")
async def get_sensing_active(page: int = 1, page_size: int = 50):
    """DFUs where demand sensing alpha > 0.5 (sensing dominates this week)."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM mv_sensing_overrides_active"
            )
            total_row = cur.fetchone()
            total = total_row[0] if total_row else 0
            cur.execute(
                """
                SELECT item_no, loc, week_start, alpha_weight,
                       sensing_signal_qty, statistical_forecast_qty, blended_qty,
                       velocity_spike_ratio
                FROM mv_sensing_overrides_active
                ORDER BY velocity_spike_ratio DESC NULLS LAST
                LIMIT %s OFFSET %s
                """,
                (page_size, offset),
            )
            rows = cur.fetchall()

    cols = [
        "item_no", "loc", "week_start", "alpha_weight",
        "sensing_signal_qty", "statistical_forecast_qty", "blended_qty",
        "velocity_spike_ratio",
    ]
    items = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("week_start"):
            d["week_start"] = d["week_start"].isoformat()
        items.append(d)

    return {"total": total, "page": page, "active_overrides": items}
