"""
F1.2 — compute_inventory_projection.py

Computes day-by-day forward inventory projections for all active DFUs (or a single DFU).
Writes results to fact_inventory_projection and refreshes mv_inventory_projection_summary.

Usage:
    uv run python scripts/compute_inventory_projection.py [--horizon 90] [--dfu ITEM LOC] [--dry-run]
"""

from __future__ import annotations

import argparse
import calendar
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

import psycopg
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.db import get_db_params

CONFIG_PATH = "config/projection_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data retrieval helpers
# ---------------------------------------------------------------------------

def get_active_dfus(conn) -> list[tuple[str, str]]:
    """Return all (item_no, loc) pairs that have recent inventory snapshots."""
    sql = """
        SELECT DISTINCT item_no, loc
        FROM fact_inventory_snapshot
        ORDER BY item_no, loc
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return [(r[0], r[1]) for r in cur.fetchall()]


def get_current_inventory(item_no: str, loc: str, conn) -> dict:
    """Pull the latest snapshot values for this DFU."""
    sql = """
        SELECT qty_on_hand, lead_time_days
        FROM fact_inventory_snapshot
        WHERE item_no = %s AND loc = %s
        ORDER BY snapshot_date DESC
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (item_no, loc))
        row = cur.fetchone()
    if not row:
        return {"qty_on_hand": 0.0, "lead_time_days": 14}
    return {"qty_on_hand": float(row[0] or 0), "lead_time_days": int(row[1] or 14)}


def get_safety_stock(item_no: str, loc: str, conn) -> float:
    """Return the current safety stock target for this DFU, or 0 if not computed."""
    sql = """
        SELECT ss_combined
        FROM fact_safety_stock_targets
        WHERE item_no = %s AND loc = %s
        ORDER BY computed_at DESC
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (item_no, loc))
        row = cur.fetchone()
    return float(row[0] or 0) if row else 0.0


def get_daily_demand_rates(
    item_no: str,
    loc: str,
    start_date: date,
    horizon_days: int,
    conn,
) -> tuple[dict[date, float], str]:
    """
    Pull production forecast and disaggregate to daily rates.
    Falls back to 3-month average actuals if no production forecast exists.
    Returns ({date: daily_qty}, source_label).
    """
    # Try production forecast first
    sql = """
        SELECT forecast_month, forecast_qty, plan_version
        FROM fact_production_forecast
        WHERE item_no = %s AND loc = %s
          AND plan_version = (
              SELECT plan_version FROM fact_production_forecast
              WHERE item_no = %s AND loc = %s
              ORDER BY generated_at DESC LIMIT 1
          )
        ORDER BY forecast_month
    """
    with conn.cursor() as cur:
        cur.execute(sql, (item_no, loc, item_no, loc))
        rows = cur.fetchall()

    if rows:
        source = "production_forecast"
        plan_version = rows[0][2]
        monthly = {r[0]: float(r[1] or 0) for r in rows}
    else:
        # Fallback: 3-month average actuals
        source = "fallback_avg"
        plan_version = None
        sql_fallback = """
            SELECT AVG(qty)
            FROM fact_sales_monthly
            WHERE dmdunit = %s AND loc = %s AND type = 1
              AND startdate >= CURRENT_DATE - INTERVAL '90 days'
        """
        with conn.cursor() as cur:
            cur.execute(sql_fallback, (item_no, loc))
            row = cur.fetchone()
        avg_monthly = float(row[0] or 0.0) if row else 0.0
        daily_rate = avg_monthly / 30.0
        return (
            {start_date + timedelta(days=i): daily_rate for i in range(horizon_days)},
            source,
            plan_version,
        )

    # Disaggregate monthly to daily
    daily: dict[date, float] = {}
    for i in range(horizon_days):
        d = start_date + timedelta(days=i)
        month_start = d.replace(day=1)
        days_in_month = calendar.monthrange(d.year, d.month)[1]
        monthly_qty = monthly.get(month_start, 0.0)
        daily[d] = monthly_qty / days_in_month

    return daily, source, plan_version


def get_open_po_receipts(
    item_no: str,
    loc: str,
    start_date: date,
    horizon_days: int,
    conn,
) -> dict[date, float]:
    """
    Return {date: qty} of confirmed PO receipts within [start_date, start_date+horizon_days].
    """
    end_date = start_date + timedelta(days=horizon_days)
    sql = """
        SELECT effective_delivery_date, SUM(open_qty) AS qty
        FROM fact_open_purchase_orders
        WHERE item_no = %s AND loc = %s
          AND line_status IN ('open', 'partially_received')
          AND effective_delivery_date BETWEEN %s AND %s
        GROUP BY effective_delivery_date
    """
    with conn.cursor() as cur:
        cur.execute(sql, (item_no, loc, start_date, end_date))
        rows = cur.fetchall()
    return {r[0]: float(r[1] or 0) for r in rows if r[0] is not None}


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_projection_scenario(
    current_qty: float,
    demand_by_day: dict[date, float],
    receipts_by_day: dict[date, float],
    safety_stock: float,
    max_coverage_qty: float,
    horizon_days: int,
    start_date: date,
    scenario: str,
    forecast_source: str,
    plan_version: str | None,
    projection_run_id: str,
    item_no: str,
    loc: str,
) -> list[dict]:
    """Simulate inventory day by day for a single scenario."""
    rows = []
    qty = current_qty
    cumulative_demand = 0.0

    for i in range(horizon_days):
        d = start_date + timedelta(days=i)
        daily_demand = demand_by_day.get(d, 0.0)

        # No receipts in 'no_order' scenario
        if scenario == "no_order":
            daily_receipts = 0.0
        else:
            daily_receipts = receipts_by_day.get(d, 0.0)

        qty = max(0.0, qty + daily_receipts - daily_demand)
        cumulative_demand += daily_demand
        dos = qty / daily_demand if daily_demand > 0 else 9999.0

        rows.append({
            "projection_run_id": projection_run_id,
            "item_no": item_no,
            "loc": loc,
            "projection_date": d,
            "scenario": scenario,
            "projected_qty": round(qty, 2),
            "projected_dos": round(min(dos, 9999.0), 2),
            "forecast_qty_consumed": round(cumulative_demand, 2),
            "receipts_expected": round(daily_receipts, 2),
            "reorder_triggered": qty <= safety_stock,
            "stockout_risk": qty <= 0,
            "excess_risk": qty > max_coverage_qty,
            "daily_demand_rate": round(daily_demand, 4),
            "forecast_source": forecast_source,
            "plan_version": plan_version,
        })

    return rows


def write_projection_rows(rows: list[dict], dry_run: bool, conn) -> int:
    """Bulk-insert projection rows, deleting prior rows for this item/loc first."""
    if dry_run or not rows:
        return len(rows)

    item_no = rows[0]["item_no"]
    loc = rows[0]["loc"]

    sql_delete = """
        DELETE FROM fact_inventory_projection
        WHERE item_no = %s AND loc = %s
    """
    sql_insert = """
        INSERT INTO fact_inventory_projection
            (projection_run_id, item_no, loc, projection_date, scenario,
             projected_qty, projected_dos, forecast_qty_consumed, receipts_expected,
             reorder_triggered, stockout_risk, excess_risk, daily_demand_rate,
             forecast_source, plan_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    with conn.cursor() as cur:
        cur.execute(sql_delete, (item_no, loc))
        cur.executemany(sql_insert, [
            (
                r["projection_run_id"], r["item_no"], r["loc"],
                r["projection_date"], r["scenario"],
                r["projected_qty"], r["projected_dos"], r["forecast_qty_consumed"],
                r["receipts_expected"], r["reorder_triggered"], r["stockout_risk"],
                r["excess_risk"], r["daily_demand_rate"], r["forecast_source"],
                r["plan_version"],
            )
            for r in rows
        ])
    conn.commit()
    return len(rows)


def refresh_summary_view(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY mv_inventory_projection_summary")
    conn.commit()


# ---------------------------------------------------------------------------
# Per-DFU projection runner
# ---------------------------------------------------------------------------

def compute_dfu_projection(
    item_no: str,
    loc: str,
    horizon_days: int,
    config: dict,
    conn,
    dry_run: bool = False,
) -> tuple[int, str]:
    """Compute and write projection for a single DFU. Returns (rows_written, run_id)."""
    start_date = date.today()
    run_id = str(uuid.uuid4())

    # Inputs
    inventory = get_current_inventory(item_no, loc, conn)
    current_qty = inventory["qty_on_hand"]
    safety_stock = get_safety_stock(item_no, loc, conn)

    daily_demand, source, plan_version = get_daily_demand_rates(
        item_no, loc, start_date, horizon_days, conn
    )
    po_receipts = get_open_po_receipts(item_no, loc, start_date, horizon_days, conn)

    # Max coverage qty: 6 months of average daily demand
    avg_daily = sum(daily_demand.values()) / max(len(daily_demand), 1)
    excess_months = config["thresholds"]["excess_coverage_months"]
    max_coverage_qty = avg_daily * 30 * excess_months

    scenarios = config["projection"]["scenarios"]
    all_rows: list[dict] = []

    for scenario in scenarios:
        receipts = po_receipts if scenario != "no_order" else {}
        rows = run_projection_scenario(
            current_qty=current_qty,
            demand_by_day=daily_demand,
            receipts_by_day=receipts,
            safety_stock=safety_stock,
            max_coverage_qty=max_coverage_qty,
            horizon_days=horizon_days,
            start_date=start_date,
            scenario=scenario,
            forecast_source=source,
            plan_version=plan_version,
            projection_run_id=run_id,
            item_no=item_no,
            loc=loc,
        )
        all_rows.extend(rows)

    written = write_projection_rows(all_rows, dry_run, conn)
    return written, run_id


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute forward inventory projections")
    parser.add_argument("--horizon", type=int, default=None, help="Days to project (default: from config)")
    parser.add_argument("--dfu", nargs=2, metavar=("ITEM", "LOC"), help="Single DFU to project")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    config = load_config()
    horizon_days = args.horizon or config["projection"]["horizon_days"]
    dry_run = args.dry_run

    with psycopg.connect(**get_db_params()) as conn:
        if args.dfu:
            item_no, loc = args.dfu
            dfus = [(item_no, loc)]
        else:
            dfus = get_active_dfus(conn)

        print(f"Computing inventory projections for {len(dfus)} DFUs, horizon={horizon_days} days")
        total_rows = 0
        for item_no, loc in dfus:
            try:
                written, run_id = compute_dfu_projection(
                    item_no, loc, horizon_days, config, conn, dry_run=dry_run
                )
                total_rows += written
            except Exception as e:
                print(f"  WARN: {item_no}/{loc} failed: {e}")

        if not dry_run and not args.dfu:
            print("Refreshing mv_inventory_projection_summary...")
            try:
                refresh_summary_view(conn)
            except Exception as e:
                print(f"  WARN: Could not refresh summary view: {e}")

    print(f"Done. {total_rows} rows {'would be' if dry_run else ''} written.")


if __name__ == "__main__":
    main()
