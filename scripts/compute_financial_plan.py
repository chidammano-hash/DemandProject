"""
compute_financial_plan.py — F4.1 Financial Inventory Plan

Computes the financial inventory plan by valuing on-hand inventory, projecting
forward inventory value, and tracking planned order spend against budget caps.

Usage:
    uv run python scripts/compute_financial_plan.py
    uv run python scripts/compute_financial_plan.py --plan-date 2026-04-01
    uv run python scripts/compute_financial_plan.py --dry-run

Config: config/financial_plan_config.yaml
"""

from __future__ import annotations

import argparse
import yaml
import psycopg
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Optional

from common.db import get_db_params
from common.planning_date import get_planning_date

CONFIG_PATH = "config/financial_plan_config.yaml"

DEFAULT_CARRYING_COST_PCT = 0.25   # 25% annually
DEFAULT_MONTHS_AHEAD = 6


def load_config(path: str = CONFIG_PATH) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f).get("financial_plan", {})
    except FileNotFoundError:
        return {
            "carrying_cost_pct": DEFAULT_CARRYING_COST_PCT,
            "months_ahead": DEFAULT_MONTHS_AHEAD,
            "excess_dos_threshold": 180,
            "budget_breach_alert_pct": 0.90,
        }


def compute_inventory_value(qty: float, unit_cost: float) -> float:
    """Compute inventory value from quantity and unit cost."""
    return max(0.0, qty * unit_cost)


def compute_carrying_cost(
    inventory_value: float,
    carrying_cost_pct: float = DEFAULT_CARRYING_COST_PCT,
    months: float = 1.0,
) -> float:
    """
    Compute monthly carrying cost for held inventory.

    Monthly carrying cost = inventory_value × (carrying_cost_pct / 12) × months

    Args:
        inventory_value: Current inventory value in $
        carrying_cost_pct: Annual carrying cost fraction (e.g., 0.25 = 25%)
        months: Number of months to project (default 1)

    Returns:
        Carrying cost in $
    """
    return inventory_value * (carrying_cost_pct / 12.0) * months


def compute_excess_value(
    qty_on_hand: float,
    avg_daily_demand: float,
    unit_cost: float,
    excess_dos_threshold: int = 180,
) -> float:
    """
    Compute excess inventory value (stock beyond the DOS threshold).

    excess_qty = max(0, on_hand - avg_daily_demand × threshold_days)
    excess_value = excess_qty × unit_cost

    Args:
        qty_on_hand: Current on-hand quantity
        avg_daily_demand: Historical average daily demand
        unit_cost: Unit cost in $
        excess_dos_threshold: Days of supply threshold above which is "excess"

    Returns:
        Excess inventory value in $ (0 if within threshold)
    """
    if avg_daily_demand <= 0 or unit_cost <= 0:
        return 0.0
    max_normal_qty = avg_daily_demand * excess_dos_threshold
    excess_qty = max(0.0, qty_on_hand - max_normal_qty)
    return excess_qty * unit_cost


def compute_budget_utilization(
    committed_spend: float,
    budget_cap: float,
) -> tuple[float, bool]:
    """
    Compute budget utilization % and breach flag.

    Returns:
        (utilization_pct, is_breached)
    """
    if budget_cap <= 0:
        return 0.0, False
    utilization = committed_spend / budget_cap * 100.0
    return round(utilization, 2), utilization >= 100.0


def fetch_inventory_with_costs(
    conn: psycopg.Connection,
    plan_date: date,
) -> list[dict]:
    """
    Fetch current inventory on-hand joined with item costs.

    Returns rows with item_no, loc, qty_on_hand, unit_cost, avg_daily_sales.
    """
    sql = """
        SELECT
            inv.item_no,
            inv.loc,
            inv.qty_on_hand,
            COALESCE(ic.unit_cost, 0)        AS unit_cost,
            COALESCE(inv.avg_daily_sales, 0) AS avg_daily_demand
        FROM (
            SELECT DISTINCT ON (item_no, loc)
                item_no, loc, qty_on_hand, avg_daily_sales
            FROM fact_inventory_snapshot
            WHERE snapshot_date <= %s
            ORDER BY item_no, loc, snapshot_date DESC
        ) inv
        LEFT JOIN (
            SELECT DISTINCT ON (item_no, loc)
                item_no, loc, unit_cost
            FROM dim_item_cost
            WHERE effective_from <= %s
              AND (effective_to IS NULL OR effective_to >= %s)
            ORDER BY item_no, loc, effective_from DESC
        ) ic ON ic.item_no = inv.item_no AND ic.loc = inv.loc
        WHERE ic.unit_cost > 0
    """
    with conn.cursor() as cur:
        cur.execute(sql, (plan_date, plan_date, plan_date))
        rows = cur.fetchall()
    return [
        {
            "item_no": r[0],
            "loc": r[1],
            "qty_on_hand": float(r[2]) if r[2] else 0.0,
            "unit_cost": float(r[3]),
            "avg_daily_demand": float(r[4]),
        }
        for r in rows
    ]


def fetch_planned_order_spend(
    conn: psycopg.Connection,
    plan_date: date,
    months_ahead: int,
) -> list[dict]:
    """
    Fetch committed planned order spend from fact_replenishment_exceptions.
    Summarised by category (abc_vol from dim_dfu).
    """
    horizon_date = plan_date + relativedelta(months=months_ahead)
    sql = """
        SELECT
            COALESCE(d.abc_vol, 'X')     AS abc_class,
            COALESCE(d.ml_cluster, 'unknown') AS cluster,
            e.item_no,
            e.loc,
            e.recommended_order_qty,
            COALESCE(ic.unit_cost, 0)    AS unit_cost
        FROM fact_replenishment_exceptions e
        LEFT JOIN dim_dfu d ON d.dmdunit = e.item_no AND d.loc = e.loc
        LEFT JOIN (
            SELECT DISTINCT ON (item_no, loc) item_no, loc, unit_cost
            FROM dim_item_cost
            WHERE effective_to IS NULL OR effective_to >= %s
            ORDER BY item_no, loc, effective_from DESC
        ) ic ON ic.item_no = e.item_no AND ic.loc = e.loc
        WHERE e.recommended_reorder_date <= %s
          AND e.status NOT IN ('rejected', 'closed')
    """
    with conn.cursor() as cur:
        cur.execute(sql, (plan_date, horizon_date))
        rows = cur.fetchall()
    return [
        {
            "abc_class": r[0],
            "cluster": r[1],
            "item_no": r[2],
            "loc": r[3],
            "recommended_order_qty": float(r[4]) if r[4] else 0.0,
            "unit_cost": float(r[5]),
        }
        for r in rows
    ]


def fetch_budget_caps(conn: psycopg.Connection, plan_date: date) -> list[dict]:
    """Fetch active budget periods from fact_budget_periods."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT budget_id, budget_name, budget_period_start, budget_period_end,
                   category_filter, budget_cap_value, currency
            FROM fact_budget_periods
            WHERE budget_period_start <= %s AND budget_period_end >= %s
              AND active = TRUE
            """,
            (plan_date, plan_date),
        )
        rows = cur.fetchall()
    return [
        {
            "budget_id": r[0],
            "budget_name": r[1],
            "period_start": r[2],
            "period_end": r[3],
            "category_filter": r[4],
            "budget_cap": float(r[5]) if r[5] else 0.0,
            "currency": r[6],
        }
        for r in rows
    ]


def upsert_financial_plan(conn: psycopg.Connection, rows: list[dict]) -> int:
    sql = """
        INSERT INTO fact_financial_inventory_plan (
            item_no, loc, plan_date, unit_cost,
            on_hand_qty, on_hand_value, carrying_cost_monthly,
            excess_qty, excess_value,
            planned_order_qty, planned_order_value,
            budget_utilization_pct, is_budget_breached
        ) VALUES (
            %(item_no)s, %(loc)s, %(plan_date)s, %(unit_cost)s,
            %(on_hand_qty)s, %(on_hand_value)s, %(carrying_cost_monthly)s,
            %(excess_qty)s, %(excess_value)s,
            %(planned_order_qty)s, %(planned_order_value)s,
            %(budget_utilization_pct)s, %(is_budget_breached)s
        )
        ON CONFLICT (item_no, loc, plan_date)
        DO UPDATE SET
            unit_cost               = EXCLUDED.unit_cost,
            on_hand_qty             = EXCLUDED.on_hand_qty,
            on_hand_value           = EXCLUDED.on_hand_value,
            carrying_cost_monthly   = EXCLUDED.carrying_cost_monthly,
            excess_qty              = EXCLUDED.excess_qty,
            excess_value            = EXCLUDED.excess_value,
            planned_order_qty       = EXCLUDED.planned_order_qty,
            planned_order_value     = EXCLUDED.planned_order_value,
            budget_utilization_pct  = EXCLUDED.budget_utilization_pct,
            is_budget_breached      = EXCLUDED.is_budget_breached,
            computed_at             = NOW()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def run(
    plan_date: Optional[date] = None,
    dry_run: bool = False,
) -> dict:
    """Main entry point: compute financial inventory plan."""
    cfg = load_config()
    if plan_date is None:
        plan_date = get_planning_date()

    carrying_cost_pct = cfg.get("carrying_cost_pct", DEFAULT_CARRYING_COST_PCT)
    excess_dos_threshold = cfg.get("excess_dos_threshold", 180)
    months_ahead = cfg.get("months_ahead", DEFAULT_MONTHS_AHEAD)

    rows_to_write: list[dict] = []

    with psycopg.connect(**get_db_params()) as conn:
        inventory_rows = fetch_inventory_with_costs(conn, plan_date)
        planned_orders = fetch_planned_order_spend(conn, plan_date, months_ahead)
        budgets = fetch_budget_caps(conn, plan_date)

        # Build planned order lookup: (item_no, loc) → qty
        po_lookup: dict = {}
        for po in planned_orders:
            key = (po["item_no"], po["loc"])
            if key not in po_lookup:
                po_lookup[key] = {"qty": 0.0, "value": 0.0}
            po_lookup[key]["qty"] += po["recommended_order_qty"]
            po_lookup[key]["value"] += po["recommended_order_qty"] * po["unit_cost"]

        total_on_hand_value = 0.0
        total_excess_value = 0.0

        for row in inventory_rows:
            item = row["item_no"]
            loc = row["loc"]
            qty = row["qty_on_hand"]
            cost = row["unit_cost"]
            daily_demand = row["avg_daily_demand"]

            on_hand_value = compute_inventory_value(qty, cost)
            carrying_cost = compute_carrying_cost(on_hand_value, carrying_cost_pct)
            excess_val = compute_excess_value(qty, daily_demand, cost, excess_dos_threshold)

            excess_qty = 0.0
            if daily_demand > 0 and cost > 0:
                max_normal_qty = daily_demand * excess_dos_threshold
                excess_qty = max(0.0, qty - max_normal_qty)

            po_data = po_lookup.get((item, loc), {"qty": 0.0, "value": 0.0})
            po_qty = po_data["qty"]
            po_value = po_data["value"]

            # Budget utilization (simplified: check total planned spend vs first applicable budget)
            budget_util = 0.0
            is_breached = False
            for budget in budgets:
                if budget["budget_cap"] > 0:
                    util, breached = compute_budget_utilization(po_value, budget["budget_cap"])
                    budget_util = util
                    is_breached = breached
                    break

            total_on_hand_value += on_hand_value
            total_excess_value += excess_val

            rows_to_write.append({
                "item_no": item,
                "loc": loc,
                "plan_date": plan_date,
                "unit_cost": round(cost, 4),
                "on_hand_qty": round(qty, 2),
                "on_hand_value": round(on_hand_value, 2),
                "carrying_cost_monthly": round(carrying_cost, 2),
                "excess_qty": round(excess_qty, 2),
                "excess_value": round(excess_val, 2),
                "planned_order_qty": round(po_qty, 2),
                "planned_order_value": round(po_value, 2),
                "budget_utilization_pct": budget_util,
                "is_budget_breached": is_breached,
            })

        rows_written = 0
        if not dry_run and rows_to_write:
            rows_written = upsert_financial_plan(conn, rows_to_write)
            conn.commit()

    return {
        "items_processed": len(rows_to_write),
        "rows_written": rows_written,
        "total_on_hand_value": round(total_on_hand_value, 2),
        "total_excess_value": round(total_excess_value, 2),
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute financial inventory plan")
    parser.add_argument("--plan-date", help="Plan date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan_date = date.fromisoformat(args.plan_date) if args.plan_date else None
    result = run(plan_date, args.dry_run)
    print(result)
