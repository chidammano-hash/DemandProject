"""Generate replenishment exceptions for the Exception Queue (IPfeature7).

Scans the latest inventory position and detects 6 exception types:
  - stockout: current_qty <= 0
  - below_ss: current_qty <= ss_combined
  - below_rop: current_qty <= reorder_point (but > ss_combined)
  - excess: current_dos > target_dos_max * 1.5
  - zero_velocity: avg_daily_sls == 0 and current_qty > 0

Deduplication: skips item-loc-type if an open exception exists within last 7 days.

Usage:
    uv run python scripts/generate_replenishment_exceptions.py
    uv run python scripts/generate_replenishment_exceptions.py --dry-run
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
from collections import defaultdict

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.db import get_db_params
from common.planning_date import get_planning_date
from common.services.perf_profiler import profiled_section


# ---------------------------------------------------------------------------
# Exception detection helpers (pure functions — also used by unit tests)
# ---------------------------------------------------------------------------

_DEFAULT_UNIT_COST = 10.0        # fallback when no cost data available (USD)
_MARGIN_RATE = 0.30              # assumed gross margin (30%) when no margin data
_ANNUAL_HOLDING_RATE = 0.25      # annual inventory carrying cost rate (25%)
_DAYS_PER_MONTH = 30.44          # average days per month


def detect_exception_type(
    current_qty: float,
    ss_combined: float | None,
    reorder_point: float | None,
    current_dos: float | None,
    target_dos_max: float | None,
    avg_daily_sls: float,
) -> tuple[str | None, str | None]:
    """Return (exception_type, severity) or (None, None) if no exception."""
    ss = ss_combined or 0.0
    rop = reorder_point or 0.0

    if current_qty <= 0:
        return "stockout", "critical"

    if ss > 0 and current_qty <= ss:
        coverage_ratio = current_qty / ss
        severity = "critical" if coverage_ratio < 0.5 else "high"
        return "below_ss", severity

    if rop > 0 and current_qty <= rop:
        return "below_rop", "high"

    if (
        target_dos_max is not None
        and target_dos_max > 0
        and current_dos is not None
        and current_dos > target_dos_max * 1.5
    ):
        severity = "low" if current_dos >= 180 else "medium"
        return "excess", severity

    if avg_daily_sls == 0 and current_qty > 0:
        return "zero_velocity", "low"

    return None, None


def compute_financial_impact(
    exception_type: str,
    current_qty: float,
    ss_combined: float | None,
    unit_cost: float | None,
    demand_mean_monthly: float | None,
    current_dos: float | None,
    lead_time_mean_days: float | None,
) -> dict:
    """Compute financial impact metrics for an exception.

    Returns dict with keys: unit_cost, unit_margin, daily_demand_rate,
    loss_of_sales_7d, loss_of_sales_30d, monthly_holding_cost, financial_impact_total.
    """
    cost = unit_cost if unit_cost and unit_cost > 0 else _DEFAULT_UNIT_COST
    margin = cost * _MARGIN_RATE
    monthly_demand = demand_mean_monthly if demand_mean_monthly and demand_mean_monthly > 0 else 0.0
    daily_demand = monthly_demand / _DAYS_PER_MONTH

    loss_7d = 0.0
    loss_30d = 0.0
    monthly_holding = 0.0
    impact_total = 0.0

    if exception_type in ("stockout", "below_ss", "below_rop"):
        # Days at risk: how many days before lead time replenishment arrives
        lt = lead_time_mean_days or 0.0
        dos = current_dos if current_dos is not None else 0.0
        if lt > 0:
            # Known lead time: risk = lead time minus current days of supply
            days_at_risk = max(0.0, lt - dos)
        elif dos <= 0 and daily_demand > 0:
            # No lead time data but already stocked out: assume 7-day risk window
            days_at_risk = 7.0
        else:
            days_at_risk = 0.0
        loss_7d = round(daily_demand * margin * min(7.0, days_at_risk), 2)
        loss_30d = round(daily_demand * margin * min(30.0, days_at_risk), 2)
        impact_total = loss_7d

    elif exception_type == "excess":
        ss = ss_combined or 0.0
        excess_qty = max(0.0, current_qty - ss * 2.0)  # qty above 2x safety stock
        monthly_holding = round(excess_qty * cost * _ANNUAL_HOLDING_RATE / 12.0, 2)
        impact_total = monthly_holding

    # zero_velocity: no direct financial impact computed (impact_total stays 0)

    return {
        "unit_cost": round(cost, 2),
        "unit_margin": round(margin, 2),
        "daily_demand_rate": round(daily_demand, 4),
        "loss_of_sales_7d": loss_7d,
        "loss_of_sales_30d": loss_30d,
        "monthly_holding_cost": monthly_holding,
        "financial_impact_total": impact_total,
    }


def compute_recommendation(
    exception_type: str,
    severity: str,
    current_qty: float,
    ss_combined: float | None,
    effective_eoq: float | None,
    demand_mean_monthly: float | None,
    review_cycle_days: int | None,
    lead_time_mean_days: float | None,
    max_eoq_months_supply: float,
    today: datetime.date,
) -> tuple[float, datetime.date | None, datetime.date | None]:
    """Return (recommended_order_qty, recommended_order_by, expected_receipt_date)."""
    if exception_type in ("below_rop", "below_ss", "stockout"):
        eoq = effective_eoq or 1.0
        ss = ss_combined or 0.0
        gap = max(0.0, ss - current_qty)
        qty = max(eoq, gap + eoq / 2.0)
        # Cap at max_eoq_months_supply months of demand
        if demand_mean_monthly and demand_mean_monthly > 0:
            cap = max_eoq_months_supply * demand_mean_monthly
            qty = min(qty, cap)

        # Order-by date
        cycle = review_cycle_days or 7
        if severity == "critical":
            order_by = today
        else:
            order_by = today + datetime.timedelta(days=cycle)

        # Expected receipt
        lt = int(lead_time_mean_days or 0)
        receipt = order_by + datetime.timedelta(days=lt)
        return round(qty, 4), order_by, receipt

    # excess / zero_velocity — no order recommended
    return 0.0, None, None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> dict:
    """Detect exceptions and insert into fact_replenishment_exceptions."""
    import psycopg

    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    eoq_cfg_path = os.path.join(config_dir, "eoq_config.yaml")
    with open(eoq_cfg_path) as fh:
        eoq_config = yaml.safe_load(fh)
    max_months = float(eoq_config.get("max_eoq_months_supply", 6))

    db_params = get_db_params()
    today = get_planning_date()

    with psycopg.connect(**db_params) as conn:
        with conn.cursor() as cur:
            # ----------------------------------------------------------------
            # 1. Latest inventory position from agg_inventory_monthly
            # ----------------------------------------------------------------
            with profiled_section("fetch_inventory_position"):
                try:
                    cur.execute("""
                        SELECT
                            item_id,
                            loc,
                            eom_qty_on_hand      AS current_qty,
                            avg_daily_sls        AS avg_daily_sls
                        FROM (
                            SELECT
                                item_id,
                                loc,
                                eom_qty_on_hand,
                                avg_daily_sls,
                                ROW_NUMBER() OVER (
                                    PARTITION BY item_id, loc
                                    ORDER BY month_start DESC
                                ) AS rn
                            FROM agg_inventory_monthly
                        ) t
                        WHERE rn = 1
                    """)
                    inv_rows = cur.fetchall()
                except Exception:
                    conn.rollback()
                    inv_rows = []

            inv_map: dict[tuple, dict] = {}
            for r in inv_rows:
                key = (r[0], r[1])
                inv_map[key] = {
                    "current_qty": float(r[2] or 0),
                    "avg_daily_sls": float(r[3] or 0),
                }

            # ----------------------------------------------------------------
            # 2. Safety stock targets (stub until IPfeature3)
            # ----------------------------------------------------------------
            with profiled_section("fetch_safety_stock_targets"):
                try:
                    cur.execute("""
                        SELECT
                            item_id, loc,
                            ss_combined, reorder_point,
                            target_max_qty,
                            target_dos_max, demand_mean_monthly
                        FROM fact_safety_stock_targets
                    """)
                    ss_rows = cur.fetchall()
                except Exception:
                    conn.rollback()
                    ss_rows = []

            ss_map: dict[tuple, dict] = {}
            for r in ss_rows:
                key = (r[0], r[1])
                ss_map[key] = {
                    "ss_combined": float(r[2]) if r[2] is not None else None,
                    "reorder_point": float(r[3]) if r[3] is not None else None,
                    "effective_eoq": float(r[4]) if r[4] is not None else None,
                    "target_dos_max": float(r[5]) if r[5] is not None else None,
                    "unit_cost": None,
                    "demand_mean_monthly": float(r[6]) if r[6] is not None else None,
                }

            # ----------------------------------------------------------------
            # 2b. Unit cost lookup from fact_eoq_targets (batch load once)
            # ----------------------------------------------------------------
            with profiled_section("fetch_unit_costs"):
                try:
                    cur.execute("""
                        SELECT item_id, loc, unit_cost
                        FROM fact_eoq_targets
                        WHERE unit_cost IS NOT NULL AND unit_cost > 0
                    """)
                    cost_rows = cur.fetchall()
                except Exception:
                    conn.rollback()
                    cost_rows = []

            cost_map: dict[tuple, float] = {}
            for r in cost_rows:
                cost_map[(r[0], r[1])] = float(r[2])

            # Merge cost data into ss_map for items that have it
            for key, cost in cost_map.items():
                if key in ss_map:
                    ss_map[key]["unit_cost"] = cost
                else:
                    # Create a minimal entry so cost is available even without SS targets
                    ss_map[key] = {
                        "ss_combined": None,
                        "reorder_point": None,
                        "effective_eoq": None,
                        "target_dos_max": None,
                        "unit_cost": cost,
                        "demand_mean_monthly": None,
                    }

            # ----------------------------------------------------------------
            # 3. Policy assignments (review_cycle_days from IPfeature5)
            # ----------------------------------------------------------------
            with profiled_section("fetch_policy_assignments"):
                try:
                    cur.execute("""
                        SELECT a.item_id, a.loc, a.policy_id, p.review_cycle_days
                        FROM fact_dfu_policy_assignment a
                        JOIN dim_replenishment_policy p ON p.policy_id = a.policy_id
                    """)
                    policy_rows = cur.fetchall()
                except Exception:
                    conn.rollback()
                    policy_rows = []

            policy_map: dict[tuple, dict] = {}
            for r in policy_rows:
                key = (r[0], r[1])
                policy_map[key] = {
                    "policy_id": r[2],
                    "review_cycle_days": int(r[3]) if r[3] is not None else 7,
                }

            # ----------------------------------------------------------------
            # 4. Lead time profile (stub until IPfeature2)
            # ----------------------------------------------------------------
            with profiled_section("fetch_lead_time_profile"):
                try:
                    cur.execute("""
                        SELECT item_id, loc, lt_mean_days
                        FROM dim_item_lead_time_profile
                    """)
                    lt_rows = cur.fetchall()
                    lt_map = {(r[0], r[1]): float(r[2] or 0) for r in lt_rows}
                except Exception:
                    conn.rollback()
                    lt_map = {}

            # ----------------------------------------------------------------
            # 5. Load existing open exceptions (dedup: last 7 days)
            # ----------------------------------------------------------------
            with profiled_section("fetch_existing_exceptions"):
                dedup_cutoff = today - datetime.timedelta(days=7)
                try:
                    cur.execute("""
                        SELECT item_id, loc, exception_type
                        FROM fact_replenishment_exceptions
                        WHERE status = 'open'
                          AND exception_date >= %s
                    """, [dedup_cutoff])
                    existing = {(r[0], r[1], r[2]) for r in cur.fetchall()}
                except Exception:
                    conn.rollback()
                    existing = set()

        # ----------------------------------------------------------------
        # 6. Detect exceptions
        # ----------------------------------------------------------------
        with profiled_section("detect_exceptions"):
            to_insert: list[dict] = []
            counts_by_type: dict[str, int] = defaultdict(int)
            skipped_dedup = 0

            for (item_id, loc), inv in inv_map.items():
                current_qty = inv["current_qty"]
                avg_daily_sls = inv["avg_daily_sls"]

                ss_data = ss_map.get((item_id, loc), {})
                ss_combined = ss_data.get("ss_combined")
                reorder_point = ss_data.get("reorder_point")
                effective_eoq = ss_data.get("effective_eoq")
                target_dos_max = ss_data.get("target_dos_max")
                unit_cost = ss_data.get("unit_cost")
                demand_mean_monthly = ss_data.get("demand_mean_monthly")

                current_dos: float | None = None
                if avg_daily_sls > 0:
                    current_dos = current_qty / avg_daily_sls

                exc_type, severity = detect_exception_type(
                    current_qty, ss_combined, reorder_point,
                    current_dos, target_dos_max, avg_daily_sls,
                )
                if exc_type is None:
                    continue

                # Deduplication check
                if (item_id, loc, exc_type) in existing:
                    skipped_dedup += 1
                    continue

                pol = policy_map.get((item_id, loc), {})
                policy_id = pol.get("policy_id")
                review_cycle_days = pol.get("review_cycle_days", 7)
                lead_time = lt_map.get((item_id, loc))

                rec_qty, order_by, receipt = compute_recommendation(
                    exc_type, severity, current_qty, ss_combined, effective_eoq,
                    demand_mean_monthly, review_cycle_days, lead_time, max_months, today,
                )

                est_value = round(rec_qty * (unit_cost or 0.0), 2)

                # Financial impact computation
                fin = compute_financial_impact(
                    exception_type=exc_type,
                    current_qty=current_qty,
                    ss_combined=ss_combined,
                    unit_cost=unit_cost,
                    demand_mean_monthly=demand_mean_monthly,
                    current_dos=current_dos,
                    lead_time_mean_days=lead_time,
                )

                to_insert.append({
                    "item_id": item_id,
                    "loc": loc,
                    "exception_date": today,
                    "exception_type": exc_type,
                    "severity": severity,
                    "current_qty_on_hand": current_qty,
                    "current_dos": current_dos,
                    "ss_combined": ss_combined,
                    "reorder_point": reorder_point,
                    "recommended_order_qty": rec_qty,
                    "recommended_order_by": order_by,
                    "expected_receipt_date": receipt,
                    "estimated_order_value": est_value,
                    "policy_id": policy_id,
                    "lead_time_mean_days": lead_time,
                    "unit_cost": fin["unit_cost"],
                    "unit_margin": fin["unit_margin"],
                    "daily_demand_rate": fin["daily_demand_rate"],
                    "loss_of_sales_7d": fin["loss_of_sales_7d"],
                    "loss_of_sales_30d": fin["loss_of_sales_30d"],
                    "monthly_holding_cost": fin["monthly_holding_cost"],
                    "financial_impact_total": fin["financial_impact_total"],
                })
                counts_by_type[exc_type] += 1

        # ----------------------------------------------------------------
        # 7. Insert (skip on dry-run)
        # ----------------------------------------------------------------
        with profiled_section("insert_exceptions"):
            if not dry_run and to_insert:
                with conn.cursor() as cur:
                    cur.executemany("""
                        INSERT INTO fact_replenishment_exceptions (
                            item_id, loc, exception_date, exception_type, severity,
                            current_qty_on_hand, current_dos, ss_combined, reorder_point,
                            recommended_order_qty, recommended_order_by, expected_receipt_date,
                            estimated_order_value, policy_id, lead_time_mean_days,
                            unit_cost, unit_margin, daily_demand_rate,
                            loss_of_sales_7d, loss_of_sales_30d,
                            monthly_holding_cost, financial_impact_total
                        ) VALUES (
                            %(item_id)s, %(loc)s, %(exception_date)s, %(exception_type)s, %(severity)s,
                            %(current_qty_on_hand)s, %(current_dos)s, %(ss_combined)s, %(reorder_point)s,
                            %(recommended_order_qty)s, %(recommended_order_by)s, %(expected_receipt_date)s,
                            %(estimated_order_value)s, %(policy_id)s, %(lead_time_mean_days)s,
                            %(unit_cost)s, %(unit_margin)s, %(daily_demand_rate)s,
                            %(loss_of_sales_7d)s, %(loss_of_sales_30d)s,
                            %(monthly_holding_cost)s, %(financial_impact_total)s
                        )
                        ON CONFLICT DO NOTHING
                    """, to_insert)
                conn.commit()

    generated_count = len(to_insert)
    result = {
        "generated_count": generated_count,
        "skipped_dedup": skipped_dedup,
        "dry_run": dry_run,
        "by_type": dict(counts_by_type),
    }

    prefix = "[DRY RUN] " if dry_run else ""
    print(f"{prefix}Generated: {generated_count} exceptions, skipped (dedup): {skipped_dedup}")
    for exc_type, count in sorted(counts_by_type.items()):
        print(f"  {exc_type}: {count}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate replenishment exceptions")
    parser.add_argument("--dry-run", action="store_true", help="Preview without inserting")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
