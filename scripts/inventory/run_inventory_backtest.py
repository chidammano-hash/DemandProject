"""Inventory Planning Backtest — simulate historical inventory outcomes per forecast algorithm.

For each (model_id, eval_month, item_id, loc):
  1. Get the backtest prediction from backtest_lag_archive
  2. Get the actual demand from fact_sales_monthly
  3. Get actual inventory from agg_inventory_monthly
  4. Compute what safety stock WOULD have been
  5. Determine if that SS level would have prevented stockouts

Usage:
    uv run python scripts/run_inventory_backtest.py
    uv run python scripts/run_inventory_backtest.py --models lgbm_cluster,nbeats --months 6
    uv run python scripts/run_inventory_backtest.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params
from common.planning_date import get_planning_date
from common.services.perf_profiler import profiled_section
from common.utils import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_ss_config() -> dict:
    """Load safety stock config for z-scores and service levels."""
    cfg = load_config("safety_stock_config")
    return cfg


def _get_z_score(abc_vol: str, cfg: dict) -> float:
    """Get z-score for a given ABC class from config."""
    service_levels = cfg.get("service_levels", {"A": 0.98, "B": 0.95, "C": 0.90})
    z_table = cfg.get("z_table", {
        "0.98": 2.054, "0.95": 1.645, "0.90": 1.282, "0.85": 1.036,
    })
    sl = service_levels.get(abc_vol, 0.90)
    return z_table.get(str(sl), 1.282)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_backtest_predictions(conn, model_ids: list[str] | None, months: int) -> pd.DataFrame:
    """Load backtest predictions from backtest_lag_archive."""
    planning_dt = get_planning_date()
    cutoff = planning_dt.replace(day=1) - pd.DateOffset(months=months)

    sql = """
        SELECT model_id, item_id, loc, startdate AS eval_month,
               basefcst_pref AS forecast_qty
        FROM backtest_lag_archive
        WHERE timeframe = 'J'
          AND lag = 0
          AND startdate >= %s
    """
    params: list = [cutoff.date() if hasattr(cutoff, 'date') else cutoff]

    if model_ids:
        placeholders = ",".join(["%s"] * len(model_ids))
        sql += f" AND model_id IN ({placeholders})"
        params.extend(model_ids)

    df = pd.read_sql(sql, conn, params=params)
    if not df.empty:
        df["eval_month"] = pd.to_datetime(df["eval_month"])
        df["forecast_qty"] = pd.to_numeric(df["forecast_qty"], errors="coerce").fillna(0)
    logger.info("Backtest predictions loaded: %s rows, %s models",
                f"{len(df):,}", df["model_id"].nunique() if not df.empty else 0)
    return df


def load_actual_sales(conn, months: int) -> pd.DataFrame:
    """Load actual sales for comparison."""
    planning_dt = get_planning_date()
    cutoff = planning_dt.replace(day=1) - pd.DateOffset(months=months)

    sql = """
        SELECT item_id, loc, startdate, SUM(qty) AS actual_demand
        FROM fact_sales_monthly
        WHERE type = 1 AND startdate >= %s
        GROUP BY item_id, loc, startdate
    """
    df = pd.read_sql(sql, conn, params=[cutoff.date() if hasattr(cutoff, 'date') else cutoff])
    if not df.empty:
        df["startdate"] = pd.to_datetime(df["startdate"])
        df["actual_demand"] = pd.to_numeric(df["actual_demand"], errors="coerce").fillna(0)
    logger.info("Actual sales loaded: %s rows", f"{len(df):,}")
    return df


def load_actual_inventory(conn, months: int) -> pd.DataFrame:
    """Load actual inventory positions."""
    planning_dt = get_planning_date()
    cutoff = planning_dt.replace(day=1) - pd.DateOffset(months=months)

    sql = """
        SELECT item_id, loc, month_start, eom_qty_on_hand, monthly_sales, avg_daily_sls
        FROM agg_inventory_monthly
        WHERE month_start >= %s
    """
    df = pd.read_sql(sql, conn, params=[cutoff.date() if hasattr(cutoff, 'date') else cutoff])
    if not df.empty:
        df["month_start"] = pd.to_datetime(df["month_start"])
    logger.info("Actual inventory loaded: %s rows", f"{len(df):,}")
    return df


def load_lead_time_profiles(conn) -> dict[tuple, dict]:
    """Load lead time stats keyed by (item_id, loc)."""
    sql = "SELECT item_id, loc, lt_mean_days, lt_std_days FROM dim_item_lead_time_profile"
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    result = {}
    for r in rows:
        result[(r[0], r[1])] = {"lt_mean": float(r[2] or 14), "lt_std": float(r[3] or 3)}
    logger.info("Lead time profiles loaded: %s SKUs", f"{len(result):,}")
    return result


def load_abc_classes(conn) -> dict[tuple, str]:
    """Load ABC classification from dim_sku."""
    sql = "SELECT item_id, loc, abc_vol FROM dim_sku"
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return {(r[0], r[1]): (r[2] or "C") for r in rows}


# ---------------------------------------------------------------------------
# Backtest computation
# ---------------------------------------------------------------------------

def compute_backtest(
    predictions: pd.DataFrame,
    actuals_sales: pd.DataFrame,
    actuals_inv: pd.DataFrame,
    lt_profiles: dict,
    abc_map: dict,
    ss_config: dict,
) -> list[dict]:
    """Compute inventory backtest outcomes for all (model, DFU, month) combos."""
    from datetime import datetime, timezone

    ts_now = datetime.now(timezone.utc)
    all_rows = []
    default_lt_count = 0

    # Index actuals for O(1) lookup
    sales_idx = {}
    for _, r in actuals_sales.iterrows():
        key = (r["item_id"], r["loc"], r["startdate"])
        sales_idx[key] = float(r["actual_demand"])

    inv_idx = {}
    for _, r in actuals_inv.iterrows():
        key = (r["item_id"], r["loc"], r["month_start"])
        inv_idx[key] = {
            "eom_on_hand": float(r["eom_qty_on_hand"] or 0),
            "monthly_sales": float(r["monthly_sales"] or 0),
        }

    for _, row in predictions.iterrows():
        model_id = row["model_id"]
        item_id = row["item_id"]
        loc = row["loc"]
        eval_month = row["eval_month"]
        forecast_qty = float(row["forecast_qty"])

        # Get actual demand for this month
        actual = sales_idx.get((item_id, loc, eval_month), 0.0)

        # Get actual inventory
        inv = inv_idx.get((item_id, loc, eval_month), {})
        eom_on_hand = inv.get("eom_on_hand", 0.0)
        monthly_sales = inv.get("monthly_sales", 0.0)

        # Lead time
        lt = lt_profiles.get((item_id, loc), {"lt_mean": 14, "lt_std": 3})
        if (item_id, loc) not in lt_profiles:
            default_lt_count += 1
        lt_mean = lt["lt_mean"]
        lt_std = lt["lt_std"]

        # ABC and z-score
        abc = abc_map.get((item_id, loc), "C")
        z = _get_z_score(abc, ss_config)

        # Compute simulated SS from this algorithm's forecast
        forecast_daily = forecast_qty / 30.44
        # Use abs forecast error as demand variability proxy
        error = abs(forecast_qty - actual)
        demand_std_daily = error / 30.44 if error > 0 else max(forecast_daily * 0.2, 0.1)

        ss = z * math.sqrt(
            lt_mean * demand_std_daily**2 + forecast_daily**2 * lt_std**2
        )
        ss = max(0, ss)
        rop = forecast_daily * lt_mean + ss

        # Simulated outcomes
        effective_stock = eom_on_hand + ss
        would_stockout = actual > 0 and effective_stock < actual
        fill_rate = min(1.0, effective_stock / actual) if actual > 0 else 1.0
        excess = max(0, effective_stock - actual)

        all_rows.append({
            "model_id": model_id,
            "item_id": item_id,
            "loc": loc,
            "eval_month": eval_month.date() if hasattr(eval_month, 'date') else eval_month,
            "forecast_qty": round(forecast_qty, 2),
            "actual_demand": round(actual, 2),
            "forecast_error": round(forecast_qty - actual, 2),
            "abs_error": round(abs(forecast_qty - actual), 2),
            "simulated_ss": round(ss, 4),
            "simulated_rop": round(rop, 4),
            "actual_eom_on_hand": round(eom_on_hand, 2),
            "actual_monthly_sales": round(monthly_sales, 2),
            "would_have_stocked_out": would_stockout,
            "simulated_fill_rate": round(fill_rate, 4),
            "excess_inventory": round(excess, 2),
            "abc_vol": abc,
            "computed_at": ts_now,
        })

        if len(all_rows) % 10000 == 0 and len(all_rows) > 0:
            logger.info("  ... processed %s rows", f"{len(all_rows):,}")

    if default_lt_count > 0:
        logger.warning(
            "%d DFUs used default lead time (14 days) — run lead time profiling first",
            default_lt_count,
        )
    logger.info("Backtest computed: %s rows", f"{len(all_rows):,}")
    return all_rows


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

def write_backtest(rows: list[dict], conn, dry_run: bool = False) -> int:
    """Upsert backtest rows into fact_inventory_backtest."""
    if not rows or dry_run:
        logger.info("[DRY RUN] Would write %s rows", f"{len(rows):,}" if rows else "0")
        return len(rows) if rows else 0

    sql = """
        INSERT INTO fact_inventory_backtest
            (model_id, item_id, loc, eval_month,
             forecast_qty, actual_demand, forecast_error, abs_error,
             simulated_ss, simulated_rop,
             actual_eom_on_hand, actual_monthly_sales,
             would_have_stocked_out, simulated_fill_rate, excess_inventory,
             abc_vol, computed_at)
        VALUES
            (%(model_id)s, %(item_id)s, %(loc)s, %(eval_month)s,
             %(forecast_qty)s, %(actual_demand)s, %(forecast_error)s, %(abs_error)s,
             %(simulated_ss)s, %(simulated_rop)s,
             %(actual_eom_on_hand)s, %(actual_monthly_sales)s,
             %(would_have_stocked_out)s, %(simulated_fill_rate)s, %(excess_inventory)s,
             %(abc_vol)s, %(computed_at)s)
        ON CONFLICT (model_id, item_id, loc, eval_month)
        DO UPDATE SET
            forecast_qty = EXCLUDED.forecast_qty,
            actual_demand = EXCLUDED.actual_demand,
            forecast_error = EXCLUDED.forecast_error,
            abs_error = EXCLUDED.abs_error,
            simulated_ss = EXCLUDED.simulated_ss,
            simulated_rop = EXCLUDED.simulated_rop,
            actual_eom_on_hand = EXCLUDED.actual_eom_on_hand,
            actual_monthly_sales = EXCLUDED.actual_monthly_sales,
            would_have_stocked_out = EXCLUDED.would_have_stocked_out,
            simulated_fill_rate = EXCLUDED.simulated_fill_rate,
            excess_inventory = EXCLUDED.excess_inventory,
            abc_vol = EXCLUDED.abc_vol,
            computed_at = EXCLUDED.computed_at
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# API router helper
# ---------------------------------------------------------------------------

def get_backtest_summary(conn) -> list[dict]:
    """Per-model aggregate for API."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT model_id,
                   COUNT(*) AS n_rows,
                   COUNT(DISTINCT item_id || '|' || loc) AS n_dfus,
                   AVG(simulated_fill_rate)::numeric(6,4) AS avg_fill_rate,
                   SUM(CASE WHEN would_have_stocked_out THEN 1 ELSE 0 END)::float
                       / NULLIF(COUNT(*), 0) AS stockout_rate,
                   AVG(simulated_ss)::numeric(10,2) AS avg_ss,
                   AVG(excess_inventory)::numeric(10,2) AS avg_excess,
                   AVG(abs_error)::numeric(10,2) AS avg_abs_error
            FROM fact_inventory_backtest
            GROUP BY model_id
            ORDER BY avg_fill_rate DESC
        """)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate inventory planning outcomes using historical backtest predictions"
    )
    parser.add_argument("--models", default=None,
                        help="Comma-separated model_ids (default: all in backtest_lag_archive)")
    parser.add_argument("--months", type=int, default=12,
                        help="How many recent months to evaluate (default: 12)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    model_ids = args.models.split(",") if args.models else None

    logger.info("Inventory Backtest — models=%s, months=%d", model_ids or "all", args.months)
    t_start = time.time()

    db = get_db_params()
    ss_config = _load_ss_config()

    with psycopg.connect(**db) as conn:
        with profiled_section("load_data"):
            predictions = load_backtest_predictions(conn, model_ids, args.months)
            if predictions.empty:
                logger.info("No backtest predictions found. Run backtests first.")
                return
            actuals_sales = load_actual_sales(conn, args.months)
            actuals_inv = load_actual_inventory(conn, args.months)
            lt_profiles = load_lead_time_profiles(conn)
            abc_map = load_abc_classes(conn)

        with profiled_section("compute_backtest"):
            rows = compute_backtest(
                predictions, actuals_sales, actuals_inv,
                lt_profiles, abc_map, ss_config,
            )

        with profiled_section("write_results"):
            written = write_backtest(rows, conn, dry_run=args.dry_run)

        logger.info("Written: %s rows", f"{written:,}")

        # Print summary
        if not args.dry_run and rows:
            summary = get_backtest_summary(conn)
            logger.info("=== Backtest Summary ===")
            for s in summary:
                logger.info(
                    "  %s: fill_rate=%.3f, stockout_rate=%.3f, avg_ss=%.0f, avg_error=%.0f",
                    s["model_id"], float(s["avg_fill_rate"] or 0),
                    float(s["stockout_rate"] or 0), float(s["avg_ss"] or 0),
                    float(s["avg_abs_error"] or 0),
                )

    elapsed = time.time() - t_start
    logger.info("Inventory backtest complete in %.0fs", elapsed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
