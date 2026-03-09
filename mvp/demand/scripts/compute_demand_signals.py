"""
IPfeature9 — Demand Sensing & Short-Horizon Signal Integration

Computes daily demand velocity signals from fact_inventory_snapshot MTD data,
comparing projected monthly demand against champion forecast.

Writes results to fact_demand_signals.

Usage:
    uv run python scripts/compute_demand_signals.py [--signal-date YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import argparse
import calendar
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg
from common.db import get_db_params  # noqa: E402
from common.planning_date import get_planning_date  # noqa: E402

# Minimum day of month before computing signals (insufficient data before day 5)
MIN_DAY_OF_MONTH = 5

# Signal classification thresholds (%)
ABOVE_PLAN_THRESHOLD = 10.0
BELOW_PLAN_THRESHOLD = -10.0
WATCH_THRESHOLD = 20.0


def compute_projected_monthly(
    mtd_actual: float,
    day_of_month: int,
    days_in_month: int,
) -> float | None:
    """Project monthly total from MTD actual."""
    if day_of_month < MIN_DAY_OF_MONTH or day_of_month == 0:
        return None
    return mtd_actual * (days_in_month / day_of_month)


def compute_demand_vs_forecast_pct(
    projected_monthly: float | None,
    forecast_monthly: float | None,
) -> float | None:
    """Compute % deviation of projected demand vs champion forecast."""
    if projected_monthly is None:
        return None
    if forecast_monthly is None or forecast_monthly == 0:
        return None
    return (projected_monthly - forecast_monthly) / forecast_monthly * 100.0


def classify_signal_type(demand_vs_forecast_pct: float | None) -> str:
    """Classify demand deviation into above_plan / below_plan / on_plan."""
    if demand_vs_forecast_pct is None:
        return "on_plan"
    if demand_vs_forecast_pct > ABOVE_PLAN_THRESHOLD:
        return "above_plan"
    if demand_vs_forecast_pct < BELOW_PLAN_THRESHOLD:
        return "below_plan"
    return "on_plan"


def compute_signal_strength(demand_vs_forecast_pct: float | None) -> float:
    """Signal strength = |demand_vs_forecast_pct| / 100."""
    if demand_vs_forecast_pct is None:
        return 0.0
    return abs(demand_vs_forecast_pct) / 100.0


def classify_alert_priority(
    projected_stockout: bool,
    is_below_ss: bool,
    demand_vs_forecast_pct: float | None,
) -> str:
    """Classify alert priority: urgent | watch | none."""
    if projected_stockout and is_below_ss:
        return "urgent"
    if demand_vs_forecast_pct is not None and abs(demand_vs_forecast_pct) > WATCH_THRESHOLD:
        return "watch"
    return "none"


def compute_projected_stockout(
    projected_daily_demand: float,
    days_remaining: int,
    current_on_hand: float,
) -> bool:
    """True if projected demand for remaining days exceeds on-hand inventory."""
    if days_remaining <= 0 or current_on_hand is None:
        return False
    return (projected_daily_demand * days_remaining) > current_on_hand


def run(signal_date: date | None = None, dry_run: bool = False) -> dict:
    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            # Determine signal_date
            if signal_date is None:
                cur.execute(
                    "SELECT MAX(snapshot_date) FROM fact_inventory_snapshot"
                )
                row = cur.fetchone()
                signal_date = row[0] if row and row[0] else get_planning_date()

            month_start = signal_date.replace(day=1)
            day_of_month = signal_date.day
            days_in_month = calendar.monthrange(signal_date.year, signal_date.month)[1]
            days_remaining = days_in_month - day_of_month

            print(f"Computing demand signals for {signal_date} (day {day_of_month}/{days_in_month})")

            if day_of_month < MIN_DAY_OF_MONTH:
                print(f"  Skipping: day_of_month={day_of_month} < {MIN_DAY_OF_MONTH} (insufficient data)")
                return {"inserted": 0, "skipped": 0, "reason": "insufficient_days"}

            # Load MTD snapshot data for this month
            cur.execute("""
                SELECT
                    item_no,
                    loc,
                    MAX(mtd_sales)   AS mtd_actual,
                    MAX(qty_on_hand) FILTER (WHERE snapshot_date = %s) AS current_on_hand
                FROM fact_inventory_snapshot
                WHERE snapshot_date >= %s
                  AND snapshot_date <= %s
                GROUP BY item_no, loc
            """, [signal_date, month_start, signal_date])
            snapshot_rows = cur.fetchall()
            print(f"  Loaded {len(snapshot_rows)} item-loc snapshots")

            # Load champion forecast for this month
            cur.execute("""
                SELECT dmdunit AS item_no, loc, SUM(basefcst_pref) AS forecast_monthly
                FROM fact_external_forecast_monthly
                WHERE startdate = %s
                  AND model_id = 'champion'
                GROUP BY dmdunit, loc
            """, [month_start])
            forecast_map: dict[tuple, float] = {
                (r[0], r[1]): float(r[2]) for r in cur.fetchall()
            }

            # Load safety stock data
            cur.execute("""
                SELECT item_no, loc, ss_combined, is_below_ss
                FROM fact_safety_stock_targets
                WHERE policy_version = 'v1'
            """)
            ss_map: dict[tuple, tuple] = {
                (r[0], r[1]): (float(r[2]) if r[2] is not None else 0.0, bool(r[3]))
                for r in cur.fetchall()
            }

        # Compute signals
        signals = []
        skipped = 0

        for row in snapshot_rows:
            item_no, loc, mtd_actual_raw, current_on_hand_raw = row
            if mtd_actual_raw is None:
                skipped += 1
                continue

            mtd_actual = float(mtd_actual_raw)
            current_on_hand = float(current_on_hand_raw) if current_on_hand_raw is not None else 0.0

            projected_monthly = compute_projected_monthly(mtd_actual, day_of_month, days_in_month)
            if projected_monthly is None:
                skipped += 1
                continue

            key = (item_no, loc)
            forecast_monthly = forecast_map.get(key)
            ss_combined, is_below_ss = ss_map.get(key, (0.0, False))

            demand_vs_forecast_pct = compute_demand_vs_forecast_pct(projected_monthly, forecast_monthly)
            signal_type = classify_signal_type(demand_vs_forecast_pct)
            signal_strength = compute_signal_strength(demand_vs_forecast_pct)

            projected_daily_demand = mtd_actual / day_of_month if day_of_month > 0 else 0.0
            proj_stockout = compute_projected_stockout(projected_daily_demand, days_remaining, current_on_hand)
            proj_excess = projected_monthly is not None and forecast_monthly is not None and projected_monthly < 0.5 * forecast_monthly

            alert_priority = classify_alert_priority(proj_stockout, is_below_ss, demand_vs_forecast_pct)

            # mtd_expected = forecast_monthly / days_in_month * days_elapsed
            mtd_expected = (
                (float(forecast_monthly) / days_in_month) * day_of_month
                if forecast_monthly is not None else None
            )

            signals.append((
                item_no, loc,
                signal_date, month_start,
                day_of_month, day_of_month, days_remaining,
                mtd_actual, mtd_expected, projected_monthly, None, forecast_monthly,
                demand_vs_forecast_pct, None,
                signal_type, signal_strength,
                current_on_hand, ss_combined, is_below_ss,
                proj_stockout, proj_excess, alert_priority,
            ))

        print(f"  Computed {len(signals)} signals, skipped {skipped}")

        if dry_run:
            print("  Dry run — no changes written.")
            return {"inserted": 0, "skipped": skipped, "dry_run": True}

        upsert_sql = """
            INSERT INTO fact_demand_signals (
                item_no, loc,
                signal_date, month_start,
                day_of_month, days_elapsed, days_remaining,
                mtd_actual, mtd_expected, projected_monthly, historical_avg_monthly, forecast_monthly,
                demand_vs_forecast_pct, demand_acceleration,
                signal_type, signal_strength,
                current_on_hand, ss_combined, is_below_ss,
                projected_stockout, projected_excess, alert_priority
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (item_no, loc, signal_date)
            DO UPDATE SET
                mtd_actual             = EXCLUDED.mtd_actual,
                mtd_expected           = EXCLUDED.mtd_expected,
                projected_monthly      = EXCLUDED.projected_monthly,
                forecast_monthly       = EXCLUDED.forecast_monthly,
                demand_vs_forecast_pct = EXCLUDED.demand_vs_forecast_pct,
                signal_type            = EXCLUDED.signal_type,
                signal_strength        = EXCLUDED.signal_strength,
                current_on_hand        = EXCLUDED.current_on_hand,
                is_below_ss            = EXCLUDED.is_below_ss,
                projected_stockout     = EXCLUDED.projected_stockout,
                projected_excess       = EXCLUDED.projected_excess,
                alert_priority         = EXCLUDED.alert_priority,
                load_ts                = NOW()
        """

        with psycopg.connect(**get_db_params()) as conn:
            with conn.cursor() as cur:
                for sig in signals:
                    cur.execute(upsert_sql, list(sig))
            conn.commit()

        print(f"  Upserted {len(signals)} signals for {signal_date}.")
        return {"inserted": len(signals), "skipped": skipped}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute demand sensing signals")
    parser.add_argument("--signal-date", help="Signal date YYYY-MM-DD (default: latest snapshot date)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    signal_date = date.fromisoformat(args.signal_date) if args.signal_date else None
    run(signal_date=signal_date, dry_run=args.dry_run)
