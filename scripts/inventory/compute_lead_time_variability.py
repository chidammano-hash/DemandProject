"""IPfeature2: Lead Time Variability Profiling.

Detects lead time change-points in fact_inventory_snapshot, computes
LT distribution stats (mean, std, CV, percentiles) per item-location,
and upserts results into dim_item_lead_time_profile.

Usage:
    uv run python scripts/inventory/compute_lead_time_variability.py
    uv run python scripts/inventory/compute_lead_time_variability.py --config config/inventory/inventory_planning_config.yaml
    uv run python scripts/inventory/compute_lead_time_variability.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params
from common.services.perf_profiler import profiled_section


# ---------------------------------------------------------------------------
# Pure computation helpers (importable by unit tests)
# ---------------------------------------------------------------------------

def extract_lt_change_points(lt_series: list[float]) -> list[float]:
    """Return the LT value at each change-point in an ordered daily LT series.

    The first value is always included.  Subsequent values are included only
    when they differ from the immediately preceding value.
    """
    if not lt_series:
        return []
    observations: list[float] = [lt_series[0]]
    for i in range(1, len(lt_series)):
        if lt_series[i] != lt_series[i - 1]:
            observations.append(lt_series[i])
    return observations


def compute_lt_metrics(observations: list[float], config: dict) -> dict[str, Any] | None:
    """Compute LT distribution stats from a list of change-point observations.

    Returns None if fewer than min_observations are present.
    """
    min_obs = config["change_point"]["min_observations"]
    if len(observations) < min_obs:
        return None

    arr = np.array(observations, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0

    return {
        "lt_mean_days": mean,
        "lt_std_days": std,
        "lt_cv": cv,
        "lt_min_days": float(np.min(arr)),
        "lt_max_days": float(np.max(arr)),
        "lt_p25_days": float(np.percentile(arr, 25)),
        "lt_p50_days": float(np.percentile(arr, 50)),
        "lt_p75_days": float(np.percentile(arr, 75)),
        "lt_p95_days": float(np.percentile(arr, 95)),
        "observation_count": len(observations),
    }


def classify_lt_variability_class(cv: float | None, config: dict) -> str | None:
    """Classify lt_variability_class from CV.

    Returns:
        'stable'   if cv < stable_threshold
        'moderate' if stable_threshold ≤ cv < moderate_threshold
        'volatile' if cv ≥ moderate_threshold
        None       if cv is None
    """
    if cv is None:
        return None
    stable_threshold = config["cv_thresholds"]["stable"]
    moderate_threshold = config["cv_thresholds"]["moderate"]
    if cv < stable_threshold:
        return "stable"
    if cv < moderate_threshold:
        return "moderate"
    return "volatile"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config: dict, dry_run: bool = False) -> dict[str, int]:
    """Query fact_inventory_snapshot, compute LT profiles, upsert dim_item_lead_time_profile.

    Returns a summary dict: {processed, updated, skipped}.
    """
    import psycopg

    history_months = config["history"]["history_months"]
    batch_size = config["batch"]["batch_size"]

    conn_params = get_db_params()

    # -----------------------------------------------------------------------
    # Step 1: Extract LT change-points per item-loc via SQL window function.
    # Using a CTE so Postgres does the heavy lifting on the 190M-row table.
    # -----------------------------------------------------------------------
    extract_sql = """
        WITH daily AS (
            SELECT
                item_id,
                loc,
                snapshot_date,
                lead_time_days,
                LAG(lead_time_days) OVER (
                    PARTITION BY item_id, loc
                    ORDER BY snapshot_date
                ) AS prev_lt
            FROM fact_inventory_snapshot
            WHERE snapshot_date >= CURRENT_DATE - (%(history_months)s || ' months')::INTERVAL
              AND lead_time_days IS NOT NULL
              AND lead_time_days > 0
        ),
        change_points AS (
            SELECT item_id, loc, lead_time_days, snapshot_date
            FROM daily
            WHERE prev_lt IS NULL OR lead_time_days != prev_lt
        )
        SELECT
            item_id,
            loc,
            array_agg(lead_time_days ORDER BY snapshot_date)    AS lt_observations,
            COUNT(*)                                             AS observation_count,
            COUNT(DISTINCT DATE_TRUNC('month', snapshot_date))   AS observation_months
        FROM change_points
        GROUP BY item_id, loc
        HAVING COUNT(*) >= %(min_obs)s
        ORDER BY item_id, loc
    """

    upsert_sql = """
        INSERT INTO dim_item_lead_time_profile (
            item_id, loc,
            lt_mean_days, lt_std_days, lt_cv,
            lt_min_days, lt_max_days,
            lt_p25_days, lt_p50_days, lt_p75_days, lt_p95_days,
            observation_count, observation_months,
            lt_variability_class, computed_at
        ) VALUES (
            %(item_id)s, %(loc)s,
            %(lt_mean_days)s, %(lt_std_days)s, %(lt_cv)s,
            %(lt_min_days)s, %(lt_max_days)s,
            %(lt_p25_days)s, %(lt_p50_days)s, %(lt_p75_days)s, %(lt_p95_days)s,
            %(observation_count)s, %(observation_months)s,
            %(lt_variability_class)s, %(computed_at)s
        )
        ON CONFLICT (item_id, loc) DO UPDATE SET
            lt_mean_days         = EXCLUDED.lt_mean_days,
            lt_std_days          = EXCLUDED.lt_std_days,
            lt_cv                = EXCLUDED.lt_cv,
            lt_min_days          = EXCLUDED.lt_min_days,
            lt_max_days          = EXCLUDED.lt_max_days,
            lt_p25_days          = EXCLUDED.lt_p25_days,
            lt_p50_days          = EXCLUDED.lt_p50_days,
            lt_p75_days          = EXCLUDED.lt_p75_days,
            lt_p95_days          = EXCLUDED.lt_p95_days,
            observation_count    = EXCLUDED.observation_count,
            observation_months   = EXCLUDED.observation_months,
            lt_variability_class = EXCLUDED.lt_variability_class,
            computed_at          = EXCLUDED.computed_at
    """

    processed = 0
    updated = 0
    skipped = 0
    now = datetime.now(timezone.utc)

    with psycopg.connect(**conn_params) as conn:
        with profiled_section("load_data"):
            with conn.cursor() as cur:
                cur.execute(
                    extract_sql,
                    {
                        "history_months": history_months,
                        "min_obs": config["change_point"]["min_observations"],
                    },
                )
                rows = cur.fetchall()

        batch: list[dict] = []

        with profiled_section("compute_variability"):
            for item_id, loc, lt_observations, obs_count, obs_months in rows:
                processed += 1

                # lt_observations is already a list from array_agg
                lt_list = [float(v) for v in lt_observations]
                metrics = compute_lt_metrics(lt_list, config)
                if metrics is None:
                    skipped += 1
                    continue

                lt_class = classify_lt_variability_class(metrics["lt_cv"], config)
                row = {
                    "item_id": item_id,
                    "loc": loc,
                    **metrics,
                    "observation_months": int(obs_months),
                    "lt_variability_class": lt_class,
                    "computed_at": now,
                }
                batch.append(row)

                if len(batch) >= batch_size and not dry_run:
                    with conn.cursor() as cur:
                        cur.executemany(upsert_sql, batch)
                    conn.commit()
                    updated += len(batch)
                    batch.clear()

        with profiled_section("write_results"):
            if batch and not dry_run:
                with conn.cursor() as cur:
                    cur.executemany(upsert_sql, batch)
                conn.commit()
                updated += len(batch)

    return {"processed": processed, "updated": updated, "skipped": skipped}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute lead time variability profiles.")
    parser.add_argument(
        "--config",
        default="config/inventory/inventory_planning_config.yaml",
        help="Path to YAML config (default: config/inventory/inventory_planning_config.yaml)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip DB writes")
    args = parser.parse_args()

    with open(args.config) as fh:
        raw = yaml.safe_load(fh)

    # Support both merged (lead_time section) and legacy flat format
    cfg = raw.get("lead_time", raw)

    summary = run(cfg, dry_run=args.dry_run)
    print(
        f"[lead-time-variability] processed={summary['processed']} "
        f"updated={summary['updated']} skipped={summary['skipped']}"
        + (" (DRY RUN)" if args.dry_run else "")
    )
