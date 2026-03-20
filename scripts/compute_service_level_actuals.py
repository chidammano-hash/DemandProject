"""
compute_service_level_actuals.py — F3.2 Service Level Actuals vs. Targets

Computes fill-rate actuals vs. targets for each DFU, tracks streaks,
classifies miss reasons, and writes to fact_service_level_performance.

Usage:
    uv run python scripts/compute_service_level_actuals.py --month 2026-03-01
    uv run python scripts/compute_service_level_actuals.py --month 2026-03-01 --dry-run

Config: config/service_level_config.yaml
"""

from __future__ import annotations

import argparse
import yaml
import psycopg
from datetime import date
from typing import Optional

from common.db import get_db_params

CONFIG_PATH = "config/service_level_config.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            "default_target_by_class": {"A": 0.975, "B": 0.950, "C": 0.900},
            "on_target_tolerance": 0.002,
            "streak_miss_threshold": 2,
        }


def resolve_target(
    targets: dict,
    item_no: str,
    loc: str,
    abc_class: str,
) -> float:
    """
    Resolve service level target with precedence:
    item+loc override > class default.

    Args:
        targets: {(item_no, loc): target, abc_class: target, ...}
        item_no, loc, abc_class: DFU identity
    Returns:
        float service level target [0.0, 1.0]
    """
    key_item_loc = (item_no, loc)
    if key_item_loc in targets:
        return targets[key_item_loc]
    if abc_class in targets:
        return targets[abc_class]
    return 0.95  # global default


def classify_miss_reason(
    gap: float,
    stockout_days: int,
    lt_variance_days: float,
    demand_spike_ratio: float,
) -> tuple[str, float]:
    """
    Classify the primary reason for a service level miss.

    Args:
        gap: actual_fill_rate - target (negative means miss)
        stockout_days: days with zero stock in the month
        lt_variance_days: supplier LT variance vs planned
        demand_spike_ratio: actual_demand / avg_demand (>1.5 = spike)

    Returns:
        (primary_miss_reason, confidence_score)
    """
    if lt_variance_days > 7:
        return "lead_time_variance", 0.90
    if demand_spike_ratio > 1.5:
        return "demand_spike", 0.85
    if stockout_days > 5:
        return "insufficient_ss", 0.80
    if gap < -0.02:
        return "insufficient_ss", 0.60
    return "data_gap", 0.30


def compute_miss_streak(
    conn: psycopg.Connection,
    item_no: str,
    loc: str,
    current_month: date,
) -> int:
    """Return number of consecutive months (ending at current_month) where gap < 0."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT gap, perf_month
            FROM fact_service_level_performance
            WHERE item_no = %s AND loc = %s
              AND perf_month < %s
            ORDER BY perf_month DESC
            LIMIT 12
            """,
            (item_no, loc, current_month),
        )
        rows = cur.fetchall()

    streak = 0
    for gap, _ in rows:
        if gap is not None and float(gap) < 0:
            streak += 1
        else:
            break
    return streak


def fetch_fill_rate_actuals(conn: psycopg.Connection, month: date) -> list[dict]:
    """Load fill rate actuals from mv_fill_rate_monthly for the given month."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.item_no, f.loc, f.month_start,
                   f.fill_rate_pct,
                   d.abc_class,
                   COALESCE(f.stockout_days, 0) AS stockout_days
            FROM mv_fill_rate_monthly f
            LEFT JOIN dim_dfu d ON d.dmdunit = f.item_no AND d.loc = f.loc
            WHERE f.month_start = %s
            """,
            (month,),
        )
        rows = cur.fetchall()
    return [
        {
            "item_no": r[0],
            "loc": r[1],
            "month": r[2],
            "fill_rate": float(r[3]) / 100.0 if r[3] is not None else None,
            "abc_class": r[4] or "C",
            "stockout_days": int(r[5]) if r[5] is not None else 0,
        }
        for r in rows
    ]


def fetch_sl_targets(conn: psycopg.Connection) -> dict:
    """Load all SL targets into a lookup dict."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT abc_class, item_no, loc, target_fill_rate
            FROM fact_service_level_targets
            ORDER BY
                CASE WHEN item_no IS NOT NULL AND loc IS NOT NULL THEN 1 ELSE 2 END
            """
        )
        rows = cur.fetchall()
    targets = {}
    for abc_class, item_no, loc, target in rows:
        if item_no and loc:
            targets[(item_no, loc)] = float(target)
        else:
            targets[abc_class] = float(target)
    return targets


def upsert_performance_row(conn: psycopg.Connection, row: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO fact_service_level_performance (
                item_no, loc, perf_month, abc_class,
                actual_fill_rate, target_fill_rate, gap, gap_direction,
                stockout_events, miss_streak_months, primary_miss_reason,
                flagged_for_review, computed_at
            ) VALUES (
                %(item_no)s, %(loc)s, %(perf_month)s, %(abc_class)s,
                %(actual_fill_rate)s, %(target_fill_rate)s, %(gap)s, %(gap_direction)s,
                %(stockout_events)s, %(miss_streak_months)s, %(primary_miss_reason)s,
                %(flagged_for_review)s, NOW()
            )
            ON CONFLICT (item_no, loc, perf_month)
            DO UPDATE SET
                actual_fill_rate    = EXCLUDED.actual_fill_rate,
                gap                 = EXCLUDED.gap,
                gap_direction       = EXCLUDED.gap_direction,
                miss_streak_months  = EXCLUDED.miss_streak_months,
                primary_miss_reason = EXCLUDED.primary_miss_reason,
                flagged_for_review  = EXCLUDED.flagged_for_review,
                computed_at         = NOW()
            """,
            row,
        )


def run(month_str: str, dry_run: bool = False) -> dict:
    cfg = load_config()
    perf_month = date.fromisoformat(month_str) if isinstance(month_str, str) else month_str

    rows_written = 0
    dfus_processed = 0

    with psycopg.connect(**get_db_params()) as conn:
        targets = fetch_sl_targets(conn)
        actuals = fetch_fill_rate_actuals(conn, perf_month)

        for rec in actuals:
            dfus_processed += 1
            item_no = rec["item_no"]
            loc = rec["loc"]
            abc_class = rec["abc_class"]
            actual_rate = rec["fill_rate"]

            if actual_rate is None:
                continue

            target_rate = resolve_target(targets, item_no, loc, abc_class)
            gap = actual_rate - target_rate
            tol = cfg.get("on_target_tolerance", 0.002)
            if gap >= tol:
                gap_direction = "above_target"
            elif gap <= -tol:
                gap_direction = "below_target"
            else:
                gap_direction = "on_target"

            streak = 0
            if gap < 0 and not dry_run:
                streak = compute_miss_streak(conn, item_no, loc, perf_month)
                if gap < 0:
                    streak += 1

            miss_reason, _ = classify_miss_reason(
                gap,
                rec.get("stockout_days", 0),
                lt_variance_days=0.0,
                demand_spike_ratio=1.0,
            )

            row = {
                "item_no": item_no,
                "loc": loc,
                "perf_month": perf_month,
                "abc_class": abc_class,
                "actual_fill_rate": actual_rate,
                "target_fill_rate": target_rate,
                "gap": round(gap, 6),
                "gap_direction": gap_direction,
                "stockout_events": rec.get("stockout_days", 0),
                "miss_streak_months": streak,
                "primary_miss_reason": miss_reason if gap < 0 else None,
                "flagged_for_review": (streak >= cfg.get("streak_miss_threshold", 3)),
            }

            if not dry_run:
                upsert_performance_row(conn, row)
                rows_written += 1

        if not dry_run:
            conn.commit()

    return {
        "month": str(perf_month),
        "dfus_processed": dfus_processed,
        "rows_written": rows_written,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute service level actuals")
    parser.add_argument("--month", required=True, help="Month to process (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(args.month, args.dry_run)
    print(result)
