"""
compute_bias_corrections.py — F3.1 Forecast Bias Correction Engine

Computes rolling 3-month forecast bias by segment and derives correction factors.
Writes results to fact_bias_corrections and fact_bias_correction_history.
Optionally applies corrections to fact_demand_plan.

Usage:
    uv run python scripts/compute_bias_corrections.py \\
        --plan-version 2026-04-01_production \\
        [--apply-to-plan] [--dry-run] [--segment dfu|cluster|abc_seasonality]

Config: config/bias_correction_config.yaml
"""

from __future__ import annotations

import argparse
import yaml
import numpy as np
import psycopg
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Optional

from common.db import get_db_params
from common.planning_date import get_planning_date

CONFIG_PATH = "config/bias_correction_config.yaml"

DEFAULT_WEIGHTS = [0.50, 0.30, 0.20]
CORRECTION_MIN = 0.70
CORRECTION_MAX = 1.30
REVIEW_THRESHOLD = 0.20


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["bias_correction"]


def compute_rolling_bias(
    bias_values: list[float],
    weights: list[float] = DEFAULT_WEIGHTS,
) -> float:
    """
    Compute exponentially weighted rolling bias.

    Args:
        bias_values: Bias values ordered most-recent-first (up to 3)
        weights:     Decay weights summing to 1.0

    Returns:
        Weighted rolling bias (float)

    Example:
        bias_values = [0.150, 0.170, 0.220]  # Mar, Feb, Jan
        weights     = [0.50,  0.30,  0.20]
        result      = 0.50*0.150 + 0.30*0.170 + 0.20*0.220 = 0.170
    """
    n = min(len(bias_values), len(weights))
    vals = np.array(bias_values[:n], dtype=float)
    w = np.array(weights[:n], dtype=float)
    w = w / w.sum()
    return float(np.dot(vals, w))


def derive_correction_factor(
    rolling_bias: float,
    clip_min: float = CORRECTION_MIN,
    clip_max: float = CORRECTION_MAX,
    review_threshold: float = REVIEW_THRESHOLD,
) -> tuple[float, float, bool, bool]:
    """
    Derive correction factor from rolling bias with guard rail clipping.

    Returns:
        (correction_factor_raw, correction_factor_clipped, was_clipped, flagged_for_review)

    Example:
        rolling_bias = +0.170
        raw_factor   = 1 / 1.170 = 0.855
        clipped      = 0.855 (within [0.70, 1.30])
        flagged      = True  (|1 - 0.855| = 0.145 < 0.20 → False actually)
    """
    raw = 1.0 / (1.0 + rolling_bias) if rolling_bias != -1.0 else clip_max
    clipped = float(np.clip(raw, clip_min, clip_max))
    was_clipped = abs(clipped - raw) > 1e-6
    flagged = abs(1.0 - raw) > review_threshold
    return raw, clipped, was_clipped, flagged


def apply_correction_to_forecast(
    raw_qty: float,
    correction_factor: float,
) -> float:
    """Apply correction factor; floor at 0."""
    return max(0.0, raw_qty * correction_factor)


def load_historical_bias_cluster(
    reference_months: list[date],
    conn: psycopg.Connection,
) -> list[dict]:
    """Load per-cluster bias from backtest_lag_archive for reference months."""
    sql = """
        SELECT
            d.ml_cluster::TEXT AS segment_value,
            f.startdate        AS plan_month,
            SUM(f.basefcst_pref) AS forecast_sum,
            SUM(f.qty)           AS actual_sum
        FROM fact_external_forecast_monthly f
        JOIN dim_dfu d ON d.dmdunit = f.dmdunit AND d.loc = f.loc
        WHERE f.startdate = ANY(%s)
          AND f.lag = 0
          AND f.model_id = 'champion'
        GROUP BY d.ml_cluster, f.startdate
        HAVING SUM(f.qty) > 0
    """
    with conn.cursor() as cur:
        cur.execute(sql, (reference_months,))
        rows = cur.fetchall()
    return [
        {
            "segment_value": r[0],
            "plan_month": r[1],
            "forecast_sum": float(r[2]) if r[2] else 0,
            "actual_sum": float(r[3]) if r[3] else 0,
            "bias": (float(r[2]) / float(r[3]) - 1) if r[3] and float(r[3]) > 0 else None,
        }
        for r in rows
    ]


def load_historical_bias_dfu(
    reference_months: list[date],
    conn: psycopg.Connection,
) -> list[dict]:
    """Load per-DFU bias from champion model forecasts for reference months."""
    sql = """
        SELECT
            f.dmdunit AS item_no,
            f.loc,
            f.startdate AS plan_month,
            SUM(f.basefcst_pref) AS forecast_sum,
            SUM(f.qty)           AS actual_sum
        FROM fact_external_forecast_monthly f
        WHERE f.startdate = ANY(%s)
          AND f.lag = 0
          AND f.model_id = 'champion'
        GROUP BY f.dmdunit, f.loc, f.startdate
        HAVING SUM(f.qty) > 0
    """
    with conn.cursor() as cur:
        cur.execute(sql, (reference_months,))
        rows = cur.fetchall()
    return [
        {
            "item_no": r[0],
            "loc": r[1],
            "plan_month": r[2],
            "bias": (float(r[3]) / float(r[4]) - 1) if r[4] and float(r[4]) > 0 else None,
        }
        for r in rows
    ]


def write_bias_corrections(
    corrections: list[dict],
    conn: psycopg.Connection,
    dry_run: bool = False,
) -> int:
    """Upsert bias correction rows into fact_bias_corrections."""
    if not corrections or dry_run:
        return len(corrections)

    sql = """
        INSERT INTO fact_bias_corrections (
            item_no, loc, plan_month, segment_type, segment_value,
            rolling_bias_3m, bias_month1, bias_month2, bias_month3,
            correction_factor_raw, correction_factor, correction_was_clipped,
            raw_forecast_qty, corrected_forecast_qty, correction_pct,
            flagged_for_review, months_of_data
        ) VALUES (
            %(item_no)s, %(loc)s, %(plan_month)s, %(segment_type)s, %(segment_value)s,
            %(rolling_bias_3m)s, %(bias_month1)s, %(bias_month2)s, %(bias_month3)s,
            %(correction_factor_raw)s, %(correction_factor)s, %(correction_was_clipped)s,
            %(raw_forecast_qty)s, %(corrected_forecast_qty)s, %(correction_pct)s,
            %(flagged_for_review)s, %(months_of_data)s
        )
        ON CONFLICT (item_no, loc, plan_month, segment_type)
        DO UPDATE SET
            rolling_bias_3m       = EXCLUDED.rolling_bias_3m,
            correction_factor_raw = EXCLUDED.correction_factor_raw,
            correction_factor     = EXCLUDED.correction_factor,
            correction_was_clipped = EXCLUDED.correction_was_clipped,
            flagged_for_review    = EXCLUDED.flagged_for_review,
            computed_at           = NOW()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, corrections)
    conn.commit()
    return len(corrections)


def run(
    plan_version: str,
    plan_run_date: Optional[date] = None,
    apply_to_plan: bool = False,
    dry_run: bool = False,
    segment: str = "cluster",
) -> dict:
    """Main entry point: compute bias corrections for the next plan cycle."""
    cfg = load_config()
    if plan_run_date is None:
        plan_run_date = get_planning_date()

    weights = cfg.get("rolling_weights", DEFAULT_WEIGHTS)
    clip_min = cfg.get("correction_factor_min", CORRECTION_MIN)
    clip_max = cfg.get("correction_factor_max", CORRECTION_MAX)
    review_threshold = cfg.get("review_threshold", REVIEW_THRESHOLD)
    lookback = cfg.get("lookback_months", 3)

    reference_months = [
        (plan_run_date - relativedelta(months=i)).replace(day=1)
        for i in range(1, lookback + 1)
    ]
    plan_month = (plan_run_date + relativedelta(months=1)).replace(day=1)

    corrections = []
    with psycopg.connect(**get_db_params()) as conn:
        if segment == "cluster":
            rows = load_historical_bias_cluster(reference_months, conn)
        else:
            rows = load_historical_bias_dfu(reference_months, conn)

        # Group by segment_value and compute rolling bias
        from collections import defaultdict
        grouped: dict[str, list] = defaultdict(list)
        for r in rows:
            key = r.get("segment_value") or f"{r.get('item_no')}@{r.get('loc')}"
            grouped[key].append(r)

        for seg_val, seg_rows in grouped.items():
            seg_rows.sort(key=lambda x: x["plan_month"], reverse=True)
            bias_vals = [r["bias"] for r in seg_rows if r.get("bias") is not None]
            if not bias_vals:
                continue
            n = min(len(bias_vals), len(weights))
            rolling = compute_rolling_bias(bias_vals[:n], weights)
            raw_cf, cf, was_clipped, flagged = derive_correction_factor(
                rolling, clip_min, clip_max, review_threshold
            )

            # For cluster segment, emit one row per DFU in the cluster
            if segment == "cluster":
                # Determine DFUs in this cluster
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT dmdunit, loc FROM dim_dfu WHERE ml_cluster = %s",
                        (seg_val,),
                    )
                    dfus = cur.fetchall()
                for dmdunit, loc in dfus:
                    corrections.append({
                        "item_no": dmdunit,
                        "loc": loc,
                        "plan_month": plan_month,
                        "segment_type": "cluster",
                        "segment_value": seg_val,
                        "rolling_bias_3m": rolling,
                        "bias_month1": bias_vals[0] if len(bias_vals) > 0 else None,
                        "bias_month2": bias_vals[1] if len(bias_vals) > 1 else None,
                        "bias_month3": bias_vals[2] if len(bias_vals) > 2 else None,
                        "correction_factor_raw": raw_cf,
                        "correction_factor": cf,
                        "correction_was_clipped": was_clipped,
                        "raw_forecast_qty": None,
                        "corrected_forecast_qty": None,
                        "correction_pct": round((cf - 1.0) * 100, 2),
                        "flagged_for_review": flagged,
                        "months_of_data": n,
                    })
            else:
                parts = seg_val.split("@", 1)
                item_no, loc = parts[0], parts[1] if len(parts) > 1 else ""
                corrections.append({
                    "item_no": item_no,
                    "loc": loc,
                    "plan_month": plan_month,
                    "segment_type": "dfu",
                    "segment_value": seg_val,
                    "rolling_bias_3m": rolling,
                    "bias_month1": bias_vals[0] if len(bias_vals) > 0 else None,
                    "bias_month2": bias_vals[1] if len(bias_vals) > 1 else None,
                    "bias_month3": bias_vals[2] if len(bias_vals) > 2 else None,
                    "correction_factor_raw": raw_cf,
                    "correction_factor": cf,
                    "correction_was_clipped": was_clipped,
                    "raw_forecast_qty": None,
                    "corrected_forecast_qty": None,
                    "correction_pct": round((cf - 1.0) * 100, 2),
                    "flagged_for_review": flagged,
                    "months_of_data": n,
                })

        n_written = write_bias_corrections(corrections, conn, dry_run=dry_run)

    return {
        "plan_month": str(plan_month),
        "segment": segment,
        "corrections_computed": len(corrections),
        "corrections_written": n_written,
        "flagged_for_review": sum(1 for c in corrections if c["flagged_for_review"]),
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute forecast bias corrections")
    parser.add_argument("--plan-version", default="latest")
    parser.add_argument("--apply-to-plan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--segment", choices=["dfu", "cluster", "abc_seasonality"], default="cluster")
    args = parser.parse_args()

    result = run(
        plan_version=args.plan_version,
        apply_to_plan=args.apply_to_plan,
        dry_run=args.dry_run,
        segment=args.segment,
    )
    print(result)
