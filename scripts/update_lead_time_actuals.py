"""
update_lead_time_actuals.py — F3.3 Supplier Performance & Lead Time Learning

Loads PO receipt data, computes LT statistics per supplier, detects
degradation, and writes to dim_lead_time_profile and fact_lt_review_triggers.

Usage:
    uv run python scripts/update_lead_time_actuals.py --input data/po_receipts.csv
    uv run python scripts/update_lead_time_actuals.py --supplier-id "ABC Trading Co." --dry-run

Config: config/lead_time_config.yaml
"""

from __future__ import annotations

import argparse
import yaml
import math
import psycopg
from typing import Optional

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params

CONFIG_PATH = "config/lead_time_config.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            "window_months": 12,
            "min_sample_size": 3,
            "mean_lt_change_threshold_days": 5,
            "stddev_change_threshold_pct": 0.30,
            "otdr_degradation_threshold": 0.10,
        }


def compute_lt_statistics(lt_days_series: list[float]) -> Optional[dict]:
    """
    Compute lead time distribution statistics from a sample of actual LT days.

    Args:
        lt_days_series: List of actual lead time days (integers or floats)

    Returns:
        Dict with mean, stddev, p50, p90, p95, min, max, sample_size
        or None if the series is empty.

    Examples:
        >>> compute_lt_statistics([10, 12, 11]) == {'mean': 11.0, 'stddev': 1.0, ...}
    """
    if not lt_days_series:
        return None

    n = len(lt_days_series)
    sorted_vals = sorted(lt_days_series)
    mean = sum(sorted_vals) / n
    variance = sum((x - mean) ** 2 for x in sorted_vals) / max(n - 1, 1)
    stddev = math.sqrt(variance)

    def percentile(p: float) -> float:
        idx = (p / 100.0) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])

    return {
        "mean": round(mean, 2),
        "stddev": round(stddev, 2),
        "p50": round(percentile(50), 2),
        "p90": round(percentile(90), 2),
        "p95": round(percentile(95), 2),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "sample_size": n,
    }


def compute_recalculated_ss(
    mean_demand_daily: float,
    sigma_demand_daily: float,
    mean_lt_days: float,
    sigma_lt_days: float,
    z_score: float,
) -> float:
    """
    Recalculate safety stock using the full demand×LT variance formula.

    SS = Z × sqrt(mean_LT × σ_demand² + mean_demand² × σ_LT²)

    Args:
        mean_demand_daily:  Average daily demand
        sigma_demand_daily: Std dev of daily demand
        mean_lt_days:       Average lead time in days
        sigma_lt_days:      Std dev of lead time in days
        z_score:            Service level z-score (e.g., 1.645 for 95%)

    Returns:
        Recalculated safety stock quantity

    Example:
        mean_demand=10, sigma_demand=3, mean_lt=14, sigma_lt=0, z=1.645
        → SS = 1.645 × sqrt(14 × 9) = 1.645 × sqrt(126) ≈ 18.5
    """
    variance = mean_lt_days * sigma_demand_daily**2 + mean_demand_daily**2 * sigma_lt_days**2
    return z_score * math.sqrt(max(variance, 0))


def detect_ss_review_trigger(
    old_stats: Optional[dict],
    new_stats: dict,
    cfg: dict,
) -> Optional[str]:
    """
    Detect if LT changes warrant a safety stock review.

    Returns:
        Trigger type string or None if no trigger.
    """
    if old_stats is None:
        return None

    mean_threshold = cfg.get("mean_lt_change_threshold_days", 5)
    stddev_pct_threshold = cfg.get("stddev_change_threshold_pct", 0.30)

    mean_change = abs(new_stats.get("mean", 0) - old_stats.get("mean", 0))
    if mean_change >= mean_threshold:
        return "mean_lt_change"

    old_std = old_stats.get("stddev", 0) or 0
    new_std = new_stats.get("stddev", 0) or 0
    if old_std > 0 and abs(new_std - old_std) / old_std >= stddev_pct_threshold:
        return "stddev_change"

    return None


def run(
    input_csv: Optional[str] = None,
    supplier_filter: Optional[str] = None,
    dry_run: bool = False,
    window_months: int = 12,
) -> dict:
    """Main entry: load receipt data, compute stats, detect triggers."""
    cfg = load_config()
    rows_processed = 0
    profiles_updated = 0
    triggers_created = 0

    with psycopg.connect(**get_db_params()) as conn:
        # Fetch LT actuals from DB
        conditions = ["actual_receipt_date >= CURRENT_DATE - INTERVAL '1 month' * %s"]
        params: list = [window_months]
        if supplier_filter:
            conditions.append("supplier_id = %s")
            params.append(supplier_filter)

        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT supplier_id, item_category, loc, lead_time_days_actual,
                       on_time, actual_receipt_date
                FROM fact_lead_time_actuals
                WHERE {" AND ".join(conditions)}
                ORDER BY supplier_id, item_category, loc, actual_receipt_date DESC
                """,
                params,
            )
            raw_rows = cur.fetchall()

        # Group by (supplier_id, item_category, loc)
        from collections import defaultdict
        grouped: dict = defaultdict(list)
        for supplier_id, item_cat, loc, lt_days, on_time, _ in raw_rows:
            if lt_days is not None:
                grouped[(supplier_id or "", item_cat or "", loc or "")].append(
                    {"lt": float(lt_days), "on_time": bool(on_time)}
                )
            rows_processed += 1

        for (supplier_id, item_cat, loc), records in grouped.items():
            lt_series = [r["lt"] for r in records]
            on_time_count = sum(1 for r in records if r["on_time"])

            stats = compute_lt_statistics(lt_series)
            if not stats or stats["sample_size"] < cfg.get("min_sample_size", 3):
                continue

            otdr = on_time_count / len(records) if records else 0.0

            # Fetch prior stats
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT mean_lt_days, stddev_lt_days, on_time_delivery_rate
                    FROM dim_lead_time_profile
                    WHERE supplier_id = %s AND item_category = %s AND loc = %s
                    """,
                    (supplier_id, item_cat, loc),
                )
                prior_row = cur.fetchone()

            prior_stats = (
                {"mean": float(prior_row[0]), "stddev": float(prior_row[1] or 0)}
                if prior_row
                else None
            )

            trigger_type = detect_ss_review_trigger(prior_stats, stats, cfg)
            flagged = trigger_type is not None

            if not dry_run:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO dim_lead_time_profile (
                            supplier_id, item_category, loc,
                            mean_lt_days, stddev_lt_days, p50_lt_days, p90_lt_days, p95_lt_days,
                            on_time_delivery_rate, sample_size,
                            prior_mean_lt_days, prior_stddev_lt_days, prior_otdr,
                            flagged_for_ss_review, window_months, updated_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, NOW()
                        )
                        ON CONFLICT (supplier_id, item_category, loc)
                        DO UPDATE SET
                            prior_mean_lt_days  = dim_lead_time_profile.mean_lt_days,
                            prior_stddev_lt_days = dim_lead_time_profile.stddev_lt_days,
                            prior_otdr          = dim_lead_time_profile.on_time_delivery_rate,
                            mean_lt_days        = EXCLUDED.mean_lt_days,
                            stddev_lt_days      = EXCLUDED.stddev_lt_days,
                            p50_lt_days         = EXCLUDED.p50_lt_days,
                            p90_lt_days         = EXCLUDED.p90_lt_days,
                            p95_lt_days         = EXCLUDED.p95_lt_days,
                            on_time_delivery_rate = EXCLUDED.on_time_delivery_rate,
                            sample_size         = EXCLUDED.sample_size,
                            flagged_for_ss_review = EXCLUDED.flagged_for_ss_review,
                            updated_at          = NOW()
                        """,
                        (
                            supplier_id, item_cat, loc,
                            stats["mean"], stats["stddev"], stats["p50"],
                            stats["p90"], stats["p95"],
                            otdr, stats["sample_size"],
                            prior_stats["mean"] if prior_stats else None,
                            prior_stats["stddev"] if prior_stats else None,
                            float(prior_row[2]) if prior_row else None,
                            flagged, window_months,
                        ),
                    )
                profiles_updated += 1

                if trigger_type:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO fact_lt_review_triggers (
                                supplier_id, trigger_type,
                                old_mean_lt_days, new_mean_lt_days,
                                old_stddev_lt_days, new_stddev_lt_days,
                                affected_dfu_count, review_status
                            ) VALUES (%s, %s, %s, %s, %s, %s, 0, 'open')
                            """,
                            (
                                supplier_id, trigger_type,
                                prior_stats["mean"] if prior_stats else None,
                                stats["mean"],
                                prior_stats["stddev"] if prior_stats else None,
                                stats["stddev"],
                            ),
                        )
                    triggers_created += 1

        if not dry_run:
            conn.commit()

    return {
        "rows_processed": rows_processed,
        "profiles_updated": profiles_updated,
        "triggers_created": triggers_created,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--supplier-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--window-months", type=int, default=12)
    args = parser.parse_args()
    result = run(args.input, args.supplier_id, args.dry_run, args.window_months)
    print(result)
