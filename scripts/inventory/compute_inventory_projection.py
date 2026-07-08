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
import logging
import sys
import uuid
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.core.db import get_db_params
from common.core.mv_refresh import refresh_for_tables
from common.core.planning_date import get_planning_date
from common.services.perf_profiler import profiled_section
from common.core.utils import load_config as _load_config

log = logging.getLogger(__name__)


def load_config() -> dict:
    cfg = _load_config("inventory_planning_config")
    return cfg.get("projection", {})


# ---------------------------------------------------------------------------
# Data retrieval helpers (single-DFU, kept for --dfu mode)
# ---------------------------------------------------------------------------

def get_active_dfus(conn) -> list[tuple[str, str]]:
    """Return all (item_id, loc) pairs that have recent inventory snapshots."""
    sql = """
        SELECT DISTINCT item_id, loc
        FROM fact_inventory_snapshot
        ORDER BY item_id, loc
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return [(r[0], r[1]) for r in cur.fetchall()]


def get_current_inventory(item_id: str, loc: str, conn) -> dict:
    """Pull the latest snapshot values for this DFU."""
    sql = """
        SELECT qty_on_hand, lead_time_days
        FROM fact_inventory_snapshot
        WHERE item_id = %s AND loc = %s
        ORDER BY snapshot_date DESC
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (item_id, loc))
        row = cur.fetchone()
    if not row:
        return {"qty_on_hand": 0.0, "lead_time_days": 14}
    return {"qty_on_hand": float(row[0] or 0), "lead_time_days": int(row[1] or 14)}


def get_safety_stock(item_id: str, loc: str, conn) -> float:
    """Return the current safety stock target for this DFU, or 0 if not computed."""
    sql = """
        SELECT ss_combined
        FROM fact_safety_stock_targets
        WHERE item_id = %s AND loc = %s
        ORDER BY computed_at DESC
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (item_id, loc))
        row = cur.fetchone()
    return float(row[0] or 0) if row else 0.0


def get_daily_demand_rates(
    item_id: str,
    loc: str,
    start_date: date,
    horizon_days: int,
    conn,
    config: dict | None = None,
    forecast_source: str = "production",
    staging_model_id: str | None = None,
) -> tuple[dict[date, float], str]:
    """
    Pull demand rates for projection, in priority order:
      1. fact_production_forecast (F1.1 forward forecast, latest plan_version)
         — OR fact_production_forecast_staging when forecast_source="staging"
      2. fact_external_forecast_monthly model_id='champion' lag=0 (most recent months)
      3. Fallback: N-month average of actual sales from fact_sales_monthly

    When forecast_source="staging", priority 1 queries the staging table filtered
    by staging_model_id. Priorities 2 and 3 remain as fallback.

    Returns ({date: daily_qty}, source_label, plan_version).
    """
    fallback_months = (config or {}).get("projection", {}).get("fallback_history_months", 3)

    # --- Priority 1: production or staging forecast ---
    if forecast_source == "staging" and staging_model_id:
        sql_prod = """
            SELECT forecast_month, forecast_qty
            FROM fact_production_forecast_staging
            WHERE item_id = %s AND loc = %s AND model_id = %s
            ORDER BY forecast_month
        """
        with conn.cursor() as cur:
            cur.execute(sql_prod, (item_id, loc, staging_model_id))
            prod_rows = cur.fetchall()

        if prod_rows:
            source = f"staging:{staging_model_id}"
            plan_version = None
            monthly = {r[0]: float(r[1] or 0) for r in prod_rows}
            daily: dict[date, float] = {}
            for i in range(horizon_days):
                d = start_date + timedelta(days=i)
                month_start = d.replace(day=1)
                days_in_month = calendar.monthrange(d.year, d.month)[1]
                daily[d] = monthly.get(month_start, 0.0) / days_in_month
            return daily, source, plan_version
    else:
        sql_prod = """
            SELECT forecast_month, forecast_qty, plan_version
            FROM fact_production_forecast
            WHERE item_id = %s AND loc = %s
              AND plan_version = (
                  SELECT plan_version FROM fact_production_forecast
                  WHERE item_id = %s AND loc = %s
                  ORDER BY generated_at DESC LIMIT 1
              )
            ORDER BY forecast_month
        """
        with conn.cursor() as cur:
            cur.execute(sql_prod, (item_id, loc, item_id, loc))
            prod_rows = cur.fetchall()

        if prod_rows:
            source = "production_forecast"
            plan_version = prod_rows[0][2]
            monthly = {r[0]: float(r[1] or 0) for r in prod_rows}
            daily: dict[date, float] = {}
            for i in range(horizon_days):
                d = start_date + timedelta(days=i)
                month_start = d.replace(day=1)
                days_in_month = calendar.monthrange(d.year, d.month)[1]
                daily[d] = monthly.get(month_start, 0.0) / days_in_month
            return daily, source, plan_version

    # --- Priority 2: champion model forecasts (most recent lag=0 months) ---
    sql_champion = """
        SELECT startdate, basefcst_pref
        FROM fact_external_forecast_monthly
        WHERE item_id = %s AND loc = %s AND model_id = 'champion' AND lag = 0
        ORDER BY startdate DESC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql_champion, (item_id, loc, fallback_months))
        champ_rows = cur.fetchall()

    if champ_rows:
        source = "champion_forecast"
        plan_version = None
        avg_monthly = sum(float(r[1] or 0) for r in champ_rows) / len(champ_rows)
        daily_rate = avg_monthly / 30.0
        return (
            {start_date + timedelta(days=i): daily_rate for i in range(horizon_days)},
            source,
            plan_version,
        )

    # --- Priority 3: historical sales average fallback ---
    source = "fallback_avg"
    plan_version = None
    sql_fallback = """
        SELECT AVG(qty)
        FROM fact_sales_monthly
        WHERE item_id = %s AND loc = %s AND type = 1
          AND startdate >= (
              SELECT MAX(startdate) - INTERVAL '1 month' * %s
              FROM fact_sales_monthly
              WHERE item_id = %s AND loc = %s AND type = 1
          )
    """
    with conn.cursor() as cur:
        cur.execute(sql_fallback, (item_id, loc, fallback_months, item_id, loc))
        row = cur.fetchone()
    avg_monthly = float(row[0] or 0.0) if row else 0.0
    daily_rate = avg_monthly / 30.0
    return (
        {start_date + timedelta(days=i): daily_rate for i in range(horizon_days)},
        source,
        plan_version,
    )


def get_open_po_receipts(
    item_id: str,
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
        WHERE item_id = %s AND loc = %s
          AND line_status IN ('open', 'partially_received')
          AND effective_delivery_date BETWEEN %s AND %s
        GROUP BY effective_delivery_date
    """
    with conn.cursor() as cur:
        cur.execute(sql, (item_id, loc, start_date, end_date))
        rows = cur.fetchall()
    return {r[0]: float(r[1] or 0) for r in rows if r[0] is not None}


# ---------------------------------------------------------------------------
# Batch data retrieval helpers (all DFUs in one query each)
# ---------------------------------------------------------------------------

def _load_all_inventory(conn) -> dict[tuple[str, str], dict]:
    """Batch-load latest inventory snapshot for every DFU.

    Returns dict keyed by (item_id, loc) with {qty_on_hand, lead_time_days}.
    """
    sql = """
        SELECT DISTINCT ON (item_id, loc)
            item_id, loc, qty_on_hand, lead_time_days
        FROM fact_inventory_snapshot
        ORDER BY item_id, loc, snapshot_date DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    result: dict[tuple[str, str], dict] = {}
    for r in rows:
        result[(r[0], r[1])] = {
            "qty_on_hand": float(r[2] or 0),
            "lead_time_days": int(r[3] or 14),
        }
    return result


def _load_all_safety_stock(conn) -> dict[tuple[str, str], float]:
    """Batch-load latest safety stock target for every DFU.

    Returns dict keyed by (item_id, loc) with ss_combined float.
    """
    sql = """
        SELECT DISTINCT ON (item_id, loc)
            item_id, loc, ss_combined
        FROM fact_safety_stock_targets
        ORDER BY item_id, loc, computed_at DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return {(r[0], r[1]): float(r[2] or 0) for r in rows}


def _load_all_production_forecasts(conn) -> dict[tuple[str, str], list[tuple]]:
    """Batch-load production forecasts (latest plan_version per DFU).

    Returns dict keyed by (item_id, loc) with list of (forecast_month, forecast_qty, plan_version).
    """
    sql = """
        SELECT pf.item_id, pf.loc, pf.forecast_month, pf.forecast_qty, pf.plan_version
        FROM fact_production_forecast pf
        INNER JOIN (
            SELECT item_id, loc, plan_version
            FROM (
                SELECT item_id, loc, plan_version,
                       ROW_NUMBER() OVER (PARTITION BY item_id, loc ORDER BY generated_at DESC) AS rn
                FROM fact_production_forecast
            ) sub
            WHERE rn = 1
        ) latest ON pf.item_id = latest.item_id
                 AND pf.loc = latest.loc
                 AND pf.plan_version = latest.plan_version
        ORDER BY pf.item_id, pf.loc, pf.forecast_month
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    result: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for r in rows:
        result[(r[0], r[1])].append((r[2], r[3], r[4]))
    return dict(result)


def _load_all_champion_forecasts(conn, fallback_months: int) -> dict[tuple[str, str], list[tuple]]:
    """Batch-load champion forecasts (most recent lag=0 months per DFU).

    Returns dict keyed by (item_id, loc) with list of (startdate, basefcst_pref).
    """
    sql = """
        SELECT item_id, loc, startdate, basefcst_pref
        FROM (
            SELECT item_id, loc, startdate, basefcst_pref,
                   ROW_NUMBER() OVER (PARTITION BY item_id, loc ORDER BY startdate DESC) AS rn
            FROM fact_external_forecast_monthly
            WHERE model_id = 'champion' AND lag = 0
        ) sub
        WHERE rn <= %s
        ORDER BY item_id, loc, startdate DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (fallback_months,))
        rows = cur.fetchall()
    result: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for r in rows:
        result[(r[0], r[1])].append((r[2], r[3]))
    return dict(result)


def _load_all_sales_avg(conn, fallback_months: int) -> dict[tuple[str, str], float]:
    """Batch-load average monthly sales for fallback demand rates.

    Returns dict keyed by (item_id, loc) with avg monthly qty.
    """
    sql = """
        SELECT s.item_id, s.loc, AVG(s.qty)
        FROM fact_sales_monthly s
        INNER JOIN (
            SELECT item_id, loc, MAX(startdate) - INTERVAL '1 month' * %s AS cutoff
            FROM fact_sales_monthly
            WHERE type = 1
            GROUP BY item_id, loc
        ) mx ON s.item_id = mx.item_id AND s.loc = mx.loc
            AND s.startdate >= mx.cutoff
        WHERE s.type = 1
        GROUP BY s.item_id, s.loc
    """
    with conn.cursor() as cur:
        cur.execute(sql, (fallback_months,))
        rows = cur.fetchall()
    return {(r[0], r[1]): float(r[2] or 0) for r in rows}


def _load_all_open_pos(
    conn, start_date: date, horizon_days: int,
) -> dict[tuple[str, str], dict[date, float]]:
    """Batch-load open PO receipts for all DFUs within the projection window.

    Returns dict keyed by (item_id, loc) with {date: qty}.
    """
    end_date = start_date + timedelta(days=horizon_days)
    sql = """
        SELECT item_id, loc, effective_delivery_date, SUM(open_qty) AS qty
        FROM fact_open_purchase_orders
        WHERE line_status IN ('open', 'partially_received')
          AND effective_delivery_date BETWEEN %s AND %s
        GROUP BY item_id, loc, effective_delivery_date
    """
    with conn.cursor() as cur:
        cur.execute(sql, (start_date, end_date))
        rows = cur.fetchall()
    result: dict[tuple[str, str], dict[date, float]] = defaultdict(dict)
    for r in rows:
        if r[2] is not None:
            result[(r[0], r[1])][r[2]] = float(r[3] or 0)
    return dict(result)


def _build_demand_from_batch(
    key: tuple[str, str],
    start_date: date,
    horizon_days: int,
    prod_map: dict[tuple[str, str], list[tuple]],
    champ_map: dict[tuple[str, str], list[tuple]],
    sales_avg_map: dict[tuple[str, str], float],
) -> tuple[dict[date, float], str, str | None]:
    """Resolve demand rates for a single DFU using pre-loaded batch data.

    Priority: production forecast > champion forecast > sales average fallback.
    Returns (daily_demand_dict, source_label, plan_version).
    """
    # Priority 1: production forecast
    prod_rows = prod_map.get(key)
    if prod_rows:
        source = "production_forecast"
        plan_version = prod_rows[0][2]
        monthly = {r[0]: float(r[1] or 0) for r in prod_rows}
        daily: dict[date, float] = {}
        for i in range(horizon_days):
            d = start_date + timedelta(days=i)
            month_start = d.replace(day=1)
            days_in_month = calendar.monthrange(d.year, d.month)[1]
            daily[d] = monthly.get(month_start, 0.0) / days_in_month
        return daily, source, plan_version

    # Priority 2: champion forecast
    champ_rows = champ_map.get(key)
    if champ_rows:
        source = "champion_forecast"
        avg_monthly = sum(float(r[1] or 0) for r in champ_rows) / len(champ_rows)
        daily_rate = avg_monthly / 30.0
        return (
            {start_date + timedelta(days=i): daily_rate for i in range(horizon_days)},
            source,
            None,
        )

    # Priority 3: sales average fallback
    avg_monthly = sales_avg_map.get(key, 0.0)
    daily_rate = avg_monthly / 30.0
    return (
        {start_date + timedelta(days=i): daily_rate for i in range(horizon_days)},
        "fallback_avg",
        None,
    )


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
    item_id: str,
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
            "item_id": item_id,
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

    item_id = rows[0]["item_id"]
    loc = rows[0]["loc"]

    sql_delete = """
        DELETE FROM fact_inventory_projection
        WHERE item_id = %s AND loc = %s
    """
    sql_insert = """
        INSERT INTO fact_inventory_projection
            (projection_run_id, item_id, loc, projection_date, scenario,
             projected_qty, projected_dos, forecast_qty_consumed, receipts_expected,
             reorder_triggered, stockout_risk, excess_risk, daily_demand_rate,
             forecast_source, plan_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    with conn.cursor() as cur:
        cur.execute(sql_delete, (item_id, loc))
        cur.executemany(sql_insert, [
            (
                r["projection_run_id"], r["item_id"], r["loc"],
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


def write_all_projection_rows(all_rows: list[dict], dry_run: bool, conn) -> int:
    """Bulk-insert all projection rows in a single transaction.

    Performs one DELETE covering all DFUs then one executemany for all rows,
    instead of per-DFU DELETE+INSERT round-trips.
    """
    if dry_run or not all_rows:
        return len(all_rows)

    # Collect distinct (item_id, loc) keys for bulk delete
    dfu_keys = {(r["item_id"], r["loc"]) for r in all_rows}

    sql_delete = """
        DELETE FROM fact_inventory_projection
        WHERE item_id = %s AND loc = %s
    """
    sql_insert = """
        INSERT INTO fact_inventory_projection
            (projection_run_id, item_id, loc, projection_date, scenario,
             projected_qty, projected_dos, forecast_qty_consumed, receipts_expected,
             reorder_triggered, stockout_risk, excess_risk, daily_demand_rate,
             forecast_source, plan_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    insert_tuples = [
        (
            r["projection_run_id"], r["item_id"], r["loc"],
            r["projection_date"], r["scenario"],
            r["projected_qty"], r["projected_dos"], r["forecast_qty_consumed"],
            r["receipts_expected"], r["reorder_triggered"], r["stockout_risk"],
            r["excess_risk"], r["daily_demand_rate"], r["forecast_source"],
            r["plan_version"],
        )
        for r in all_rows
    ]
    batch_size = 50_000
    with conn.cursor() as cur:
        cur.executemany(sql_delete, list(dfu_keys))
        for i in range(0, len(insert_tuples), batch_size):
            cur.executemany(sql_insert, insert_tuples[i:i + batch_size])
            log.info("Wrote batch %d–%d of %d rows",
                     i, min(i + batch_size, len(insert_tuples)), len(insert_tuples))
    conn.commit()
    return len(all_rows)


def refresh_summary_view(conn) -> None:
    # conn unused: the service opens its own autocommit connection
    # (CONCURRENTLY is illegal inside a transaction block).
    refresh_for_tables(["fact_inventory_projection"])


# ---------------------------------------------------------------------------
# Per-DFU projection runner
# ---------------------------------------------------------------------------

def compute_dfu_projection(
    item_id: str,
    loc: str,
    horizon_days: int,
    config: dict,
    conn,
    dry_run: bool = False,
    forecast_source: str = "production",
    staging_model_id: str | None = None,
) -> tuple[int, str]:
    """Compute and write projection for a single DFU. Returns (rows_written, run_id)."""
    start_date = get_planning_date()
    run_id = str(uuid.uuid4())

    # Inputs
    inventory = get_current_inventory(item_id, loc, conn)
    current_qty = inventory["qty_on_hand"]
    safety_stock = get_safety_stock(item_id, loc, conn)

    daily_demand, source, plan_version = get_daily_demand_rates(
        item_id, loc, start_date, horizon_days, conn, config=config,
        forecast_source=forecast_source, staging_model_id=staging_model_id,
    )
    po_receipts = get_open_po_receipts(item_id, loc, start_date, horizon_days, conn)

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
            item_id=item_id,
            loc=loc,
        )
        all_rows.extend(rows)

    written = write_projection_rows(all_rows, dry_run, conn)
    return written, run_id


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Compute forward inventory projections")
    parser.add_argument("--horizon", type=int, default=None, help="Days to project (default: from config)")
    parser.add_argument("--dfu", nargs=2, metavar=("ITEM", "LOC"), help="Single DFU to project")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--forecast-source", choices=["production", "staging"],
                        default="production", help="Forecast source table")
    parser.add_argument("--model-id", default=None,
                        help="When --forecast-source=staging, which model_id to use")
    args = parser.parse_args()

    if args.forecast_source == "staging" and not args.model_id:
        parser.error("--model-id is required when --forecast-source=staging")

    if args.horizon is not None and args.horizon <= 0:
        parser.error("--horizon must be positive")

    with profiled_section("load_config"):
        config = load_config()
    horizon_days = args.horizon or config["projection"]["horizon_days"]
    dry_run = args.dry_run

    with psycopg.connect(**get_db_params()) as conn:
        if args.dfu:
            # Single-DFU mode: use per-DFU queries (original path)
            item_id, loc = args.dfu
            dfus = [(item_id, loc)]
            log.info("Computing inventory projections for 1 DFU, horizon=%d days", horizon_days)
            total_rows = 0
            for item_id, loc in dfus:
                try:
                    written, run_id = compute_dfu_projection(
                        item_id, loc, horizon_days, config, conn, dry_run=dry_run,
                        forecast_source=args.forecast_source,
                        staging_model_id=args.model_id,
                    )
                    total_rows += written
                except psycopg.Error as e:
                    log.warning("%s/%s failed: %s", item_id, loc, e)
        else:
            # Batch mode: load ALL data upfront, then iterate without DB calls
            with profiled_section("load_active_dfus"):
                dfus = get_active_dfus(conn)
            log.info("Computing inventory projections for %d DFUs, horizon=%d days", len(dfus), horizon_days)

            start_date = get_planning_date()
            fallback_months = config.get("projection", {}).get("fallback_history_months", 3)

            with profiled_section("batch_load_inputs"):
                log.info("Batch-loading inventory snapshots...")
                inv_map = _load_all_inventory(conn)
                log.info("Batch-loading safety stock targets...")
                ss_map = _load_all_safety_stock(conn)
                log.info("Batch-loading production forecasts...")
                prod_map = _load_all_production_forecasts(conn)
                log.info("Batch-loading champion forecasts...")
                champ_map = _load_all_champion_forecasts(conn, fallback_months)
                log.info("Batch-loading sales averages...")
                sales_avg_map = _load_all_sales_avg(conn, fallback_months)
                log.info("Batch-loading open PO receipts...")
                po_map = _load_all_open_pos(conn, start_date, horizon_days)

            scenarios = config["projection"]["scenarios"]
            excess_months = config["thresholds"]["excess_coverage_months"]
            all_projection_rows: list[dict] = []
            total_rows = 0
            projected_count = 0
            skipped_count = 0

            with profiled_section("run_projections"):
                for item_id, loc in dfus:
                    try:
                        key = (item_id, loc)
                        run_id = str(uuid.uuid4())

                        inventory = inv_map.get(key, {"qty_on_hand": 0.0, "lead_time_days": 14})
                        current_qty = inventory["qty_on_hand"]
                        safety_stock = ss_map.get(key, 0.0)

                        daily_demand, source, plan_version = _build_demand_from_batch(
                            key, start_date, horizon_days, prod_map, champ_map, sales_avg_map
                        )
                        po_receipts = po_map.get(key, {})

                        avg_daily = sum(daily_demand.values()) / max(len(daily_demand), 1)
                        max_coverage_qty = avg_daily * 30 * excess_months

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
                                item_id=item_id,
                                loc=loc,
                            )
                            all_projection_rows.extend(rows)
                        projected_count += 1
                    except psycopg.Error as e:
                        skipped_count += 1
                        log.warning("%s/%s failed: %s", item_id, loc, e)

            log.info("Projected %d DFUs, skipped %d (no inventory/forecast data or DB error)", projected_count, skipped_count)

            with profiled_section("bulk_insert"):
                total_rows = write_all_projection_rows(all_projection_rows, dry_run, conn)

        if not dry_run and not args.dfu:
            with profiled_section("refresh_mv"):
                log.info("Refreshing mv_inventory_projection_summary...")
                try:
                    refresh_summary_view(conn)
                except psycopg.Error as e:
                    log.warning("Could not refresh summary view: %s", e)

    log.info("Done. %d rows %swritten.", total_rows, "would be " if dry_run else "")


if __name__ == "__main__":
    main()
