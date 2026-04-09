"""IPfeature1: Compute demand variability statistics per DFU.

.. deprecated::
    This script is deprecated.  Use ``scripts/ml/compute_sku_features.py``
    (backed by ``common/ml/sku_features/``) for all new work.  The unified
    module computes seasonality, variability, and lifecycle features in a
    single pass.  This file is kept only for backward-compatible function
    exports consumed by existing tests.

Reads config/forecast_domain_config.yaml (variability section) for all thresholds.
Queries fact_sales_monthly (type=1, 24-month rolling window).
Winsorizes outliers, computes CV/MAD/skewness/kurtosis/intermittency.
Classifies each DFU into variability_class: low | medium | high | lumpy.
Upserts results back into dim_sku.

Usage:
    uv run python scripts/compute_demand_variability.py
    uv run python scripts/compute_demand_variability.py --config config/forecast_domain_config.yaml
    uv run python scripts/compute_demand_variability.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.db import get_db_params
from common.services.perf_profiler import profiled_section

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure computation helpers (importable by unit tests)
# ---------------------------------------------------------------------------

def winsorize(series: np.ndarray, sigma: float) -> np.ndarray:
    """Cap values beyond mean ± sigma * std; returns clipped array."""
    if len(series) < 2:
        return series
    mean = np.mean(series)
    std = np.std(series, ddof=1)
    if std == 0:
        return series
    lo = mean - sigma * std
    hi = mean + sigma * std
    return np.clip(series, lo, hi)


def compute_variability_metrics(
    monthly_qty: pd.Series,
    config: dict,
) -> dict:
    """Compute demand variability statistics from a monthly qty series.

    Args:
        monthly_qty: Series of monthly demand quantities (type=1 sales).
                     May contain zeros. Should be sorted chronologically.
        config: Loaded variability section from forecast_domain_config.yaml.

    Returns:
        Dict with all computed metrics; None values when insufficient data.
    """
    outlier_sigma = config["outlier"]["sigma_threshold"]
    min_months = config["history"]["min_months_history"]
    cv_low = config["cv_thresholds"]["low"]
    cv_medium = config["cv_thresholds"]["medium"]
    cv_high = config["cv_thresholds"]["high"]
    intermittency_thresh = config["intermittency_threshold"]["ratio"]

    total_months = len(monthly_qty)
    zero_months = int((monthly_qty == 0).sum())
    intermittency_ratio = zero_months / total_months if total_months > 0 else 0.0

    result: dict = {
        "demand_mean": None,
        "demand_std": None,
        "demand_cv": None,
        "demand_mad": None,
        "demand_p50": None,
        "demand_p90": None,
        "demand_skewness": None,
        "demand_kurtosis": None,
        "zero_demand_months": zero_months,
        "total_demand_months": total_months,
        "intermittency_ratio": round(intermittency_ratio, 6),
        "variability_class": None,
    }

    if total_months < min_months:
        return result

    arr = monthly_qty.to_numpy(dtype=float)
    arr = winsorize(arr, outlier_sigma)

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv_val = std_val / mean_val if mean_val > 0 else 0.0
    mad_val = float(np.mean(np.abs(arr - np.mean(arr))))
    p50_val = float(np.percentile(arr, 50))
    p90_val = float(np.percentile(arr, 90))

    # Skewness and kurtosis (scipy-style, excess kurtosis)
    n = len(arr)
    if n >= 3 and std_val > 0:
        skewness = float(np.mean(((arr - mean_val) / std_val) ** 3))
        kurtosis = float(np.mean(((arr - mean_val) / std_val) ** 4) - 3)
    else:
        skewness = 0.0
        kurtosis = 0.0

    # Variability class
    if intermittency_ratio >= intermittency_thresh or cv_val >= cv_high:
        variability_class = "lumpy"
    elif cv_val >= cv_medium:
        variability_class = "high"
    elif cv_val >= cv_low:
        variability_class = "medium"
    else:
        variability_class = "low"

    result.update({
        "demand_mean": round(mean_val, 4),
        "demand_std": round(std_val, 4),
        "demand_cv": round(cv_val, 6),
        "demand_mad": round(mad_val, 4),
        "demand_p50": round(p50_val, 4),
        "demand_p90": round(p90_val, 4),
        "demand_skewness": round(skewness, 6),
        "demand_kurtosis": round(kurtosis, 6),
        "variability_class": variability_class,
    })
    return result


def classify_variability_class(cv: float | None, intermittency_ratio: float, config: dict) -> str | None:
    """Classify a DFU into a variability class given pre-computed stats."""
    if cv is None:
        return None
    cv_low = config["cv_thresholds"]["low"]
    cv_medium = config["cv_thresholds"]["medium"]
    cv_high = config["cv_thresholds"]["high"]
    intermittency_thresh = config["intermittency_threshold"]["ratio"]

    if intermittency_ratio >= intermittency_thresh or cv >= cv_high:
        return "lumpy"
    elif cv >= cv_medium:
        return "high"
    elif cv >= cv_low:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f)
    # Support merged forecast_domain_config.yaml (variability section)
    if "variability" in raw:
        return raw["variability"]
    return raw


def run(config: dict, dry_run: bool = False) -> dict:
    """Run the variability computation pipeline.

    Returns summary dict: {processed, updated, skipped}.
    """
    history_months = config["history"]["history_months"]
    db_params = get_db_params()

    with profiled_section("load_data"):
        log.info("Connecting to database...")
        with psycopg.connect(**db_params) as conn:
            # Pre-aggregate basic stats in SQL to reduce data transfer
            log.info("Loading %d months of pre-aggregated demand stats from fact_sales_monthly...", history_months)
            agg_sql = """
                SELECT
                    s.item_id,
                    s.customer_group,
                    s.loc,
                    AVG(COALESCE(s.qty, 0)) AS mean_demand,
                    STDDEV(COALESCE(s.qty, 0)) AS std_demand,
                    COUNT(*) AS n_months,
                    SUM(CASE WHEN COALESCE(s.qty, 0) = 0 THEN 1 ELSE 0 END) AS zero_months,
                    CASE WHEN AVG(COALESCE(s.qty, 0)) > 0 THEN STDDEV(COALESCE(s.qty, 0)) / AVG(COALESCE(s.qty, 0)) ELSE 0 END AS cv,
                    percentile_cont(0.5) WITHIN GROUP (ORDER BY COALESCE(s.qty, 0)) AS p50_demand,
                    percentile_cont(0.9) WITHIN GROUP (ORDER BY COALESCE(s.qty, 0)) AS p90_demand
                FROM fact_sales_monthly s
                WHERE s.type = 1
                  AND s.startdate >= (
                      SELECT MAX(startdate) FROM fact_sales_monthly WHERE type = 1
                  ) - (%(months)s || ' months')::INTERVAL
                GROUP BY s.item_id, s.customer_group, s.loc
            """
            with conn.cursor() as cur:
                cur.execute(agg_sql, {"months": history_months})
                cur.fetchall()  # aggregation executed for side-effect; raw data used below
                # column names not needed — raw-level computation follows

            # Also load raw data for metrics that need row-level computation
            # (winsorization, skewness, kurtosis, MAD)
            log.info("Loading raw sales for advanced metrics (skewness, kurtosis, MAD)...")
            sales_sql = """
                SELECT
                    s.item_id,
                    s.customer_group,
                    s.loc,
                    s.startdate,
                    COALESCE(s.qty, 0) AS qty
                FROM fact_sales_monthly s
                WHERE s.type = 1
                  AND s.startdate >= (
                      SELECT MAX(startdate) FROM fact_sales_monthly WHERE type = 1
                  ) - (%(months)s || ' months')::INTERVAL
                ORDER BY s.item_id, s.customer_group, s.loc, s.startdate
            """
            with conn.cursor() as cur:
                cur.execute(sales_sql, {"months": history_months})
                rows = cur.fetchall()
                colnames = [d[0] for d in cur.description]

    if not rows:
        log.warning("No sales data found.")
        return {"processed": 0, "updated": 0, "skipped": 0}

    df = pd.DataFrame(rows, columns=colnames)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)

    log.info("Loaded %d sales rows for %d DFUs.", len(df), df[["item_id", "customer_group", "loc"]].drop_duplicates().shape[0])

    # Compute metrics per DFU (uses row-level data for winsorization/skewness/kurtosis)
    with profiled_section("compute_variability"):
        results = []
        groups = df.groupby(["item_id", "customer_group", "loc"])
        for (item_id, customer_group, loc), grp in groups:
            metrics = compute_variability_metrics(grp["qty"], config)
            metrics["item_id"] = item_id
            metrics["customer_group"] = customer_group
            metrics["loc"] = loc
            results.append(metrics)

    log.info("Computed variability for %d DFUs.", len(results))

    skipped = sum(1 for r in results if r["variability_class"] is None)
    updated = len(results) - skipped

    if dry_run:
        log.info("[DRY RUN] Would update %d DFUs, skip %d (insufficient history).", updated, skipped)
        return {"processed": len(results), "updated": updated, "skipped": skipped}

    # Upsert into dim_sku
    with profiled_section("write_results"):
        now = datetime.now(timezone.utc)
        upsert_sql = """
            UPDATE dim_sku SET
                demand_mean         = %(demand_mean)s,
                demand_std          = %(demand_std)s,
                demand_cv           = %(demand_cv)s,
                demand_mad          = %(demand_mad)s,
                demand_p50          = %(demand_p50)s,
                demand_p90          = %(demand_p90)s,
                demand_skewness     = %(demand_skewness)s,
                demand_kurtosis     = %(demand_kurtosis)s,
                zero_demand_months  = %(zero_demand_months)s,
                total_demand_months = %(total_demand_months)s,
                intermittency_ratio = %(intermittency_ratio)s,
                variability_class   = %(variability_class)s,
                demand_profile_ts   = %(demand_profile_ts)s
            WHERE item_id = %(item_id)s
              AND customer_group = %(customer_group)s
              AND loc = %(loc)s
        """

        batch_size = 500
        total_written = 0
        with psycopg.connect(**db_params) as conn:
            with conn.cursor() as cur:
                for i in range(0, len(results), batch_size):
                    batch = results[i : i + batch_size]
                    for r in batch:
                        r["demand_profile_ts"] = now
                    cur.executemany(upsert_sql, batch)
                    total_written += len(batch)
            conn.commit()

    log.info("Updated %d DFU rows (%d skipped for insufficient history).", total_written, skipped)
    return {"processed": len(results), "updated": total_written, "skipped": skipped}


def main() -> None:
    """Entry point — delegates to the unified SKU features pipeline.

    .. deprecated::
        This script is deprecated.  Run ``scripts/ml/compute_sku_features.py``
        directly for the unified pipeline.
    """
    import warnings

    warnings.warn(
        "scripts/compute_demand_variability.py is deprecated. "
        "Use scripts/ml/compute_sku_features.py (backed by common/ml/sku_features/) instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    print(
        "WARNING: scripts/compute_demand_variability.py is deprecated.\n"
        "  Delegating to the unified SKU features pipeline "
        "(scripts/ml/compute_sku_features.py).\n"
        "  Please update your workflow to call scripts/ml/compute_sku_features.py directly.\n"
    )

    from scripts.ml.compute_sku_features import run_pipeline

    parser = argparse.ArgumentParser(
        description="[DEPRECATED] Compute demand variability — delegates to unified SKU features pipeline",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="(ignored, kept for backward compat)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute but do not write to DB")
    args = parser.parse_args()

    summary = run_pipeline(dry_run=args.dry_run)
    log.info("Unified pipeline summary: %s", summary)


if __name__ == "__main__":
    main()
