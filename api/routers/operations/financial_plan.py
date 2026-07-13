"""
F4.1 — Financial Inventory Plan (Budget vs. Actuals) API endpoints.

Endpoints:
    GET  /finance/inventory-plan          — Summary + by-category breakdown
    GET  /finance/budget-status           — Budget utilization across all periods
    GET  /finance/working-capital-trend   — Actual + projected WC timeline
    GET  /finance/excess-value            — Top excess SKU-locations
    POST /finance/budget                  — Create new budget period (auth)
    PUT  /finance/budget/{budget_id}      — Update budget cap (auth)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from api.core import get_conn
from api.auth import require_api_key

router = APIRouter(tags=["financial-plan"])


class _BudgetCreate(BaseModel):
    scope_type: str
    scope_value: str
    period_type: str = "monthly"
    budget_start: str
    budget_end: str
    budget_cap: float
    carrying_cost_pct: float = 0.25


class _BudgetUpdate(BaseModel):
    budget_cap: float
    notes: Optional[str] = None


@router.get("/finance/inventory-plan")
async def get_inventory_plan(
    horizon: int = 6,
    plan_version: str = "latest",
):
    """Summary financial inventory plan with by-category breakdown."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(DISTINCT item_id || '@' || loc)   AS sku_loc_count,
                    SUM(projected_inventory_value)          AS total_projected_value,
                    SUM(planned_order_value)                AS total_order_value,
                    SUM(carrying_cost_monthly)              AS total_carrying_cost,
                    SUM(excess_value)                       AS total_excess_value,
                    SUM(CASE WHEN within_budget = FALSE THEN 1 ELSE 0 END) AS budget_breach_count,
                    MAX(plan_month)                         AS latest_plan_month
                FROM fact_financial_inventory_plan
                WHERE plan_version = %s
                """,
                (plan_version,),
            )
            row = cur.fetchone()

            cur.execute(
                """
                SELECT item_category, plan_month,
                       SUM(projected_inventory_value) AS projected_value,
                       SUM(planned_order_value)        AS order_value,
                       SUM(excess_value)               AS excess_value
                FROM fact_financial_inventory_plan
                JOIN dim_item USING (item_id)
                WHERE plan_version = %s
                GROUP BY item_category, plan_month
                ORDER BY plan_month, item_category
                """,
                (plan_version,),
            )
            cat_rows = cur.fetchall()

    summary = {}
    if row:
        summary = {
            "sku_loc_count": row[0] or 0,
            "total_projected_value": float(row[1]) if row[1] else 0,
            "total_order_value": float(row[2]) if row[2] else 0,
            "total_carrying_cost": float(row[3]) if row[3] else 0,
            "total_excess_value": float(row[4]) if row[4] else 0,
            "budget_breach_count": row[5] or 0,
            "latest_plan_month": row[6].isoformat() if row[6] else None,
        }

    by_category = []
    for r in cat_rows:
        by_category.append({
            "item_category": r[0],
            "plan_month": r[1].isoformat() if r[1] else None,
            "projected_value": float(r[2]) if r[2] else 0,
            "order_value": float(r[3]) if r[3] else 0,
            "excess_value": float(r[4]) if r[4] else 0,
        })

    return {"plan_version": plan_version, "horizon": horizon, "summary": summary, "by_category": by_category}


@router.get("/finance/budget-status")
async def get_budget_status(
    period_start: str | None = None,
    period_end: str | None = None,
):
    """Budget utilization across all active budget periods."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT b.budget_id, b.scope_type, b.scope_value,
                       b.budget_start, b.budget_end, b.budget_cap,
                       COALESCE(SUM(f.planned_order_value), 0)  AS consumed,
                       b.budget_cap - COALESCE(SUM(f.planned_order_value), 0) AS remaining,
                       CASE WHEN b.budget_cap > 0
                            THEN COALESCE(SUM(f.planned_order_value), 0) / b.budget_cap
                            ELSE 0 END                           AS utilization_pct
                FROM fact_budget_periods b
                LEFT JOIN fact_financial_inventory_plan f
                    ON  f.plan_month BETWEEN b.budget_start AND b.budget_end
                GROUP BY b.budget_id, b.scope_type, b.scope_value,
                         b.budget_start, b.budget_end, b.budget_cap
                ORDER BY utilization_pct DESC
            """)
            rows = cur.fetchall()

    cols = [
        "budget_id", "scope_type", "scope_value",
        "budget_start", "budget_end", "budget_cap",
        "consumed", "remaining", "utilization_pct",
    ]
    items = []
    for r in rows:
        d = dict(zip(cols, r))
        for field in ("budget_start", "budget_end"):
            if d.get(field):
                d[field] = d[field].isoformat()
        items.append(d)

    return {"budgets": items}


@router.get("/finance/working-capital-trend")
async def get_wc_trend(
    months_history: int = 6,
    months_forward: int = 6,
):
    """Working capital trend: actual (history) + projected (forward)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT plan_month,
                       SUM(projected_inventory_value) AS inventory_value,
                       SUM(planned_order_value)        AS order_value,
                       SUM(carrying_cost_monthly)      AS carrying_cost,
                       SUM(excess_value)               AS excess_value
                FROM fact_financial_inventory_plan
                WHERE plan_month >= CURRENT_DATE - INTERVAL '1 month' * %s
                  AND plan_month <= CURRENT_DATE + INTERVAL '1 month' * %s
                GROUP BY plan_month
                ORDER BY plan_month
                """,
                (months_history, months_forward),
            )
            rows = cur.fetchall()

    cols = [
        "plan_month", "inventory_value", "order_value", "carrying_cost", "excess_value",
    ]
    trend = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("plan_month"):
            d["plan_month"] = d["plan_month"].isoformat()
        trend.append(d)

    return {"months_history": months_history, "months_forward": months_forward, "trend": trend}


@router.get("/finance/excess-value")
async def get_excess_value(
    limit: int = 20,
    min_excess_value: float = 0,
    plan_version: str = "latest",
):
    """Top excess SKU-locations ranked by excess inventory value."""
    limit = max(1, min(limit, 100))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT item_id, loc, plan_month, excess_qty, excess_value,
                       projected_inventory_value, budget_cap, within_budget
                FROM fact_financial_inventory_plan
                WHERE plan_version = %s AND excess_value >= %s
                ORDER BY excess_value DESC
                LIMIT %s
                """,
                (plan_version, min_excess_value, limit),
            )
            rows = cur.fetchall()

    cols = [
        "item_id", "loc", "plan_month", "excess_qty", "excess_value",
        "projected_inventory_value", "budget_cap", "within_budget",
    ]
    items = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("plan_month"):
            d["plan_month"] = d["plan_month"].isoformat()
        items.append(d)

    return {"plan_version": plan_version, "excess_items": items}


@router.post("/finance/budget", status_code=201)
async def create_budget(body: _BudgetCreate, request: Request):
    """Create a new budget period (auth required)."""
    await require_api_key(
        x_api_key=request.headers.get("x-api-key"),
        authorization=request.headers.get("authorization"),
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fact_budget_periods
                    (scope_type, scope_value, period_type, budget_start, budget_end,
                     budget_cap, carrying_cost_pct)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING budget_id
                """,
                (
                    body.scope_type, body.scope_value, body.period_type,
                    body.budget_start, body.budget_end,
                    body.budget_cap, body.carrying_cost_pct,
                ),
            )
            budget_id = cur.fetchone()[0]
        conn.commit()

    return {"budget_id": budget_id, "status": "created"}


@router.put("/finance/budget/{budget_id}")
async def update_budget(budget_id: int, body: _BudgetUpdate, request: Request):
    """Update a budget period's cap (auth required)."""
    await require_api_key(
        x_api_key=request.headers.get("x-api-key"),
        authorization=request.headers.get("authorization"),
    )

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE fact_budget_periods
                SET budget_cap = %s, updated_at = NOW()
                WHERE budget_id = %s
                RETURNING budget_id
                """,
                (body.budget_cap, budget_id),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(404, f"Budget {budget_id} not found")

    return {"budget_id": budget_id, "budget_cap": body.budget_cap, "status": "updated"}
