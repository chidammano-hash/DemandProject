"""
compute_blended_forecast.py — F3.4 Demand Sensing Integration

Blends short-horizon demand sensing signals with statistical forecasts
using a linearly decaying alpha weight over the sensing horizon.

Usage:
    uv run python scripts/compute_blended_forecast.py
    uv run python scripts/compute_blended_forecast.py --item-no 100320 --loc 1401-BULK
    uv run python scripts/compute_blended_forecast.py --dry-run

Config: config/quantile_forecast_config.yaml
"""

from __future__ import annotations

import argparse
import yaml
import psycopg
from datetime import date, timedelta
from typing import Optional

from common.db import get_db_params
from common.planning_date import get_planning_date

CONFIG_PATH = "config/quantile_forecast_config.yaml"
SENSING_HORIZON_WEEKS = 4
OUTLIER_THRESHOLD = 3.0


def load_config(path: str = CONFIG_PATH) -> dict:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
            return data.get("sensing", {})
    except FileNotFoundError:
        return {
            "sensing_horizon_weeks": SENSING_HORIZON_WEEKS,
            "outlier_threshold": OUTLIER_THRESHOLD,
            "min_days_elapsed": 3,
        }


def compute_alpha(week_offset: int, sensing_horizon_weeks: int = SENSING_HORIZON_WEEKS) -> float:
    """
    Linearly decay alpha from 1.0 (week 1) to 0.0 (at/beyond horizon).

    Args:
        week_offset: 0-indexed week number from today (0 = current week)
        sensing_horizon_weeks: weeks over which sensing applies

    Returns:
        float alpha in [0.0, 1.0]

    Examples:
        compute_alpha(0, 4) → 1.0   (pure sensing, current week)
        compute_alpha(3, 4) → 0.25  (mostly statistical)
        compute_alpha(4, 4) → 0.0   (pure statistical)
    """
    if sensing_horizon_weeks <= 0:
        return 0.0
    alpha = 1.0 - (week_offset / sensing_horizon_weeks)
    return max(0.0, min(1.0, alpha))


def compute_velocity_signal(
    mtd_sales: float,
    days_elapsed: int,
    days_in_month: int,
    historical_daily_avg: float,
    outlier_threshold: float = OUTLIER_THRESHOLD,
) -> tuple[float, float, float, bool]:
    """
    Project monthly demand from MTD velocity.

    Args:
        mtd_sales:            Sales so far this month
        days_elapsed:         Days into the month (1-based)
        days_in_month:        Calendar days in the month
        historical_daily_avg: Historical average daily demand
        outlier_threshold:    Cap spike_ratio at this (to prevent extreme orders)

    Returns:
        (projected_monthly, daily_run_rate, spike_ratio, is_capped)

    Examples:
        mtd=40, elapsed=10, days=30, hist_avg=4 → (120, 4.0, 1.0, False)
        mtd=120, elapsed=10, days=30, hist_avg=4 → (360, 12.0, 3.0, True)
    """
    if days_elapsed <= 0:
        return historical_daily_avg * days_in_month, historical_daily_avg, 1.0, False

    daily_run_rate = mtd_sales / days_elapsed
    spike_ratio = daily_run_rate / historical_daily_avg if historical_daily_avg > 0 else 1.0
    is_capped = spike_ratio > outlier_threshold
    if is_capped:
        daily_run_rate = historical_daily_avg * outlier_threshold
    projected_monthly = daily_run_rate * days_in_month
    return projected_monthly, daily_run_rate, spike_ratio, is_capped


def monthly_to_weekly(monthly_qty: float, n_weeks_in_month: float = 4.33) -> float:
    """Convert a monthly quantity to a weekly quantity."""
    return monthly_qty / n_weeks_in_month if n_weeks_in_month > 0 else 0.0


def apply_dow_factor(weekly_qty: float, dow_factor: float = 1.0) -> float:
    """Scale weekly qty by day-of-week delivery pattern factor."""
    return weekly_qty * dow_factor


def upsert_blend_rows(conn: psycopg.Connection, rows: list[dict]) -> int:
    sql = """
        INSERT INTO fact_blended_demand_plan (
            item_no, loc, week_start, plan_version, alpha_weight,
            sensing_signal_qty, statistical_forecast_qty, blended_qty,
            velocity_spike_ratio, is_outlier_capped
        ) VALUES (
            %(item_no)s, %(loc)s, %(week_start)s, %(plan_version)s, %(alpha_weight)s,
            %(sensing_signal_qty)s, %(statistical_forecast_qty)s, %(blended_qty)s,
            %(velocity_spike_ratio)s, %(is_outlier_capped)s
        )
        ON CONFLICT (item_no, loc, week_start, plan_version)
        DO UPDATE SET
            alpha_weight             = EXCLUDED.alpha_weight,
            sensing_signal_qty       = EXCLUDED.sensing_signal_qty,
            statistical_forecast_qty = EXCLUDED.statistical_forecast_qty,
            blended_qty              = EXCLUDED.blended_qty,
            velocity_spike_ratio     = EXCLUDED.velocity_spike_ratio,
            is_outlier_capped        = EXCLUDED.is_outlier_capped,
            computed_at              = NOW()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def fetch_sensing_data(
    conn: psycopg.Connection,
    item_no: Optional[str],
    loc: Optional[str],
) -> list[dict]:
    """Fetch latest demand signals from fact_demand_signals."""
    conditions = ["signal_date >= CURRENT_DATE - INTERVAL '7 days'"]
    params: list = []
    if item_no:
        conditions.append("item_no = %s"); params.append(item_no)
    if loc:
        conditions.append("loc = %s"); params.append(loc)
    where = " AND ".join(conditions)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT item_no, loc, mtd_sales, avg_daily_sales_hist, signal_date
            FROM fact_demand_signals
            WHERE {where}
            ORDER BY signal_date DESC
            """,
            params,
        )
        rows = cur.fetchall()
    return [
        {
            "item_no": r[0], "loc": r[1],
            "mtd_sales": float(r[2]) if r[2] else 0,
            "daily_avg": float(r[3]) if r[3] else 0,
        }
        for r in rows
    ]


def fetch_statistical_forecast(
    conn: psycopg.Connection,
    item_no: Optional[str],
    loc: Optional[str],
    months_ahead: int = 2,
) -> list[dict]:
    """Fetch latest statistical champion forecast."""
    conditions = [
        "model_id = 'champion'",
        "startdate >= CURRENT_DATE",
        f"startdate <= CURRENT_DATE + INTERVAL '1 month' * %s",
    ]
    params: list = [months_ahead]
    if item_no:
        conditions.append("dmdunit = %s"); params.append(item_no)
    if loc:
        conditions.append("loc = %s"); params.append(loc)
    where = " AND ".join(conditions)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT dmdunit, loc, startdate, basefcst_pref
            FROM fact_external_forecast_monthly
            WHERE {where}
            ORDER BY dmdunit, loc, startdate
            """,
            params,
        )
        rows = cur.fetchall()
    return [
        {
            "item_no": r[0], "loc": r[1],
            "month": r[2], "stat_qty": float(r[3]) if r[3] else 0,
        }
        for r in rows
    ]


def run(
    item_no: Optional[str] = None,
    loc: Optional[str] = None,
    dry_run: bool = False,
    plan_version: str = "latest",
) -> dict:
    cfg = load_config()
    horizon_weeks = cfg.get("sensing_horizon_weeks", SENSING_HORIZON_WEEKS)
    outlier_threshold = cfg.get("outlier_threshold", OUTLIER_THRESHOLD)

    today = get_planning_date()
    # Week starts on Monday
    week_start_base = today - timedelta(days=today.weekday())

    rows_to_write: list[dict] = []

    with psycopg.connect(**get_db_params()) as conn:
        signals = fetch_sensing_data(conn, item_no, loc)
        forecasts = fetch_statistical_forecast(conn, item_no, loc)

        # Build lookup: (item_no, loc) → signal
        signal_lookup = {(s["item_no"], s["loc"]): s for s in signals}

        # Build lookup: (item_no, loc) → list of monthly forecast
        from collections import defaultdict
        stat_lookup: dict = defaultdict(list)
        for f in forecasts:
            stat_lookup[(f["item_no"], f["loc"])].append(f)

        # Generate blended forecast for each DFU × week
        all_dfus = set(signal_lookup.keys()) | set(stat_lookup.keys())
        for dfu_key in all_dfus:
            item, location = dfu_key
            signal = signal_lookup.get(dfu_key, {})
            stat_months = stat_lookup.get(dfu_key, [])

            if not stat_months:
                continue

            stat_weekly = sum(f["stat_qty"] for f in stat_months) / max(len(stat_months), 1)
            stat_weekly_wk = monthly_to_weekly(stat_weekly)

            daily_avg = signal.get("daily_avg", stat_weekly / 7)
            mtd_sales = signal.get("mtd_sales", 0)
            days_elapsed = (today - today.replace(day=1)).days + 1
            import calendar
            days_in_month = calendar.monthrange(today.year, today.month)[1]

            proj_monthly, _, spike_ratio, is_capped = compute_velocity_signal(
                mtd_sales, days_elapsed, days_in_month, daily_avg, outlier_threshold
            )
            sensing_weekly = monthly_to_weekly(proj_monthly)

            for week_offset in range(horizon_weeks + 4):
                wk_start = week_start_base + timedelta(weeks=week_offset)
                alpha = compute_alpha(week_offset, horizon_weeks)
                blended = alpha * sensing_weekly + (1.0 - alpha) * stat_weekly_wk

                rows_to_write.append({
                    "item_no": item,
                    "loc": location,
                    "week_start": wk_start,
                    "plan_version": plan_version,
                    "alpha_weight": round(alpha, 3),
                    "sensing_signal_qty": round(sensing_weekly, 2),
                    "statistical_forecast_qty": round(stat_weekly_wk, 2),
                    "blended_qty": round(max(0.0, blended), 2),
                    "velocity_spike_ratio": round(spike_ratio, 3),
                    "is_outlier_capped": is_capped,
                })

        rows_written = 0
        if not dry_run and rows_to_write:
            rows_written = upsert_blend_rows(conn, rows_to_write)
            conn.commit()

    return {
        "dfus_processed": len(all_dfus) if 'all_dfus' in locals() else 0,
        "rows_computed": len(rows_to_write),
        "rows_written": rows_written,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-no")
    parser.add_argument("--loc")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(args.item_no, args.loc, args.dry_run)
    print(result)
