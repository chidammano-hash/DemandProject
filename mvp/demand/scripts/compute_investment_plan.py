"""
IPfeature13 — Capital & Space Investment Optimization

Computes per-DFU investment ranking and efficient frontier for
capital allocation decisions.

Usage:
    uv run python scripts/compute_investment_plan.py [--budget AMOUNT] [--target-csl FLOAT]
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg
from common.db import get_db_params  # noqa: E402

# Base CSL at SS=0 (median service level)
BASE_CSL = 0.5


def compute_marginal_roi(csl_increment: float, investment_increment: float) -> float:
    """CSL gain per dollar invested."""
    return csl_increment / max(investment_increment, 1.0)


def estimate_current_csl(
    current_ss_qty: float,
    recommended_ss_qty: float,
    recommended_csl: float,
) -> float:
    """Linear interpolation of CSL based on SS coverage ratio."""
    if recommended_ss_qty <= 0 or current_ss_qty >= recommended_ss_qty:
        return recommended_csl
    coverage = current_ss_qty / recommended_ss_qty
    return BASE_CSL + (recommended_csl - BASE_CSL) * coverage


def run(budget_constraint: float | None = None, target_csl: float | None = None) -> dict:
    plan_id = str(uuid.uuid4())
    computation_date = date.today()

    # ABC class → service level mapping
    abc_sl_map = {"A": 0.98, "B": 0.95, "C": 0.90}

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            # Load SS targets
            cur.execute("""
                SELECT s.item_no, s.loc, s.ss_combined, NULL AS unit_cost,
                       d.abc_vol, d.abc_xyz_segment
                FROM fact_safety_stock_targets s
                LEFT JOIN dim_dfu d ON s.item_no = d.dmdunit AND s.loc = d.loc
                WHERE s.policy_version = 'v1'
                  AND s.ss_combined IS NOT NULL
            """)
            ss_rows = cur.fetchall()

            # Load current on-hand (latest EOM)
            cur.execute("""
                SELECT item_no, loc, eom_qty_on_hand
                FROM agg_inventory_monthly
                WHERE month_start = (SELECT MAX(month_start) FROM agg_inventory_monthly)
            """)
            eom_map: dict[tuple, float] = {
                (r[0], r[1]): float(r[2]) for r in cur.fetchall()
            }

    items = []
    for row in ss_rows:
        item_no, loc, ss_combined, unit_cost, abc_vol, abc_xyz_segment = row
        rec_ss = float(ss_combined)
        unit_cost_f = float(unit_cost) if unit_cost else 1.0
        current_ss = eom_map.get((item_no, loc), 0.0)

        rec_csl = abc_sl_map.get(abc_vol or "", 0.90)
        if target_csl is not None:
            rec_csl = target_csl

        current_csl = estimate_current_csl(current_ss, rec_ss, rec_csl)
        ss_increment = max(0.0, rec_ss - current_ss)
        inv_increment = ss_increment * unit_cost_f
        csl_increment = max(0.0, rec_csl - current_csl)
        roi = compute_marginal_roi(csl_increment, inv_increment)

        items.append({
            "item_no": item_no,
            "loc": loc,
            "abc_vol": abc_vol,
            "abc_xyz_segment": abc_xyz_segment,
            "unit_cost": unit_cost_f,
            "current_ss_qty": current_ss,
            "current_ss_value": current_ss * unit_cost_f,
            "current_csl": current_csl,
            "recommended_ss_qty": rec_ss,
            "recommended_ss_value": rec_ss * unit_cost_f,
            "recommended_csl": rec_csl,
            "ss_increment_qty": ss_increment,
            "investment_increment": inv_increment,
            "csl_increment": csl_increment,
            "marginal_roi": roi,
        })

    # Sort by marginal_roi desc, then investment_increment asc (tiebreak: cheapest first)
    items.sort(key=lambda x: (-x["marginal_roi"], x["investment_increment"]))

    # Assign ranks and cumulative investment
    cumulative = 0.0
    frontier_rows = []
    funded_csl_sum = 0.0
    total_items = len(items)
    portfolio_current_csl = sum(x["current_csl"] for x in items) / max(total_items, 1)

    for rank, item in enumerate(items, start=1):
        item["investment_rank"] = rank
        cumulative += item["investment_increment"]
        item["cumulative_investment"] = cumulative

        # Build efficient frontier point
        funded_csl_sum += item["recommended_csl"] - item["current_csl"]
        achievable_csl = min(1.0, portfolio_current_csl + funded_csl_sum / total_items)
        frontier_rows.append({
            "plan_id": plan_id,
            "budget_point": cumulative,
            "items_funded": rank,
            "achievable_csl": achievable_csl,
            "marginal_item": item["item_no"],
        })

    insert_plan_sql = """
        INSERT INTO fact_inventory_investment_plan (
            plan_id, item_no, loc, computation_date,
            current_ss_qty, current_ss_value, current_csl,
            recommended_ss_qty, recommended_ss_value, recommended_csl,
            ss_increment_qty, investment_increment, csl_increment, marginal_roi,
            investment_rank, cumulative_investment,
            abc_vol, abc_xyz_segment, unit_cost
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s
        )
        ON CONFLICT (plan_id, item_no, loc) DO NOTHING
    """

    insert_frontier_sql = """
        INSERT INTO fact_efficient_frontier (plan_id, budget_point, items_funded, achievable_csl, marginal_item)
        VALUES (%s, %s, %s, %s, %s)
    """

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            for item in items:
                cur.execute(insert_plan_sql, [
                    plan_id, item["item_no"], item["loc"], computation_date,
                    item["current_ss_qty"], item["current_ss_value"], item["current_csl"],
                    item["recommended_ss_qty"], item["recommended_ss_value"], item["recommended_csl"],
                    item["ss_increment_qty"], item["investment_increment"], item["csl_increment"],
                    item["marginal_roi"], item["investment_rank"], item["cumulative_investment"],
                    item["abc_vol"], item["abc_xyz_segment"], item["unit_cost"],
                ])
            for fr in frontier_rows:
                cur.execute(insert_frontier_sql, [
                    fr["plan_id"], fr["budget_point"], fr["items_funded"],
                    fr["achievable_csl"], fr["marginal_item"],
                ])
        conn.commit()

    total_inv = sum(x["investment_increment"] for x in items)
    print(f"Investment plan {plan_id} computed.")
    print(f"  {total_items} DFUs analyzed, total investment gap: ${total_inv:,.0f}")
    return {"plan_id": plan_id, "total_items": total_items, "total_investment_gap": total_inv}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute capital investment optimization plan")
    parser.add_argument("--budget", type=float, help="Budget constraint in dollars")
    parser.add_argument("--target-csl", type=float, help="Target portfolio CSL (0-1)")
    args = parser.parse_args()
    run(budget_constraint=args.budget, target_csl=args.target_csl)
