"""
Detect seasonality patterns in DFU monthly sales history.

Computes per-DFU seasonality metrics: strength (CV of monthly means),
year-over-year correlation, autocorrelation at lag 12, peak/trough analysis,
and classifies each DFU into a seasonality profile tier.
"""

import argparse
import logging
import multiprocessing
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
import yaml
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

logger = logging.getLogger(__name__)

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """No-op decorator fallback when numba is not installed."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

logger.info("Seasonality JIT backend: %s", "numba" if _NUMBA_AVAILABLE else "numpy (no JIT)")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params
from common.services.perf_profiler import profiled_section


def load_config(config_path: str = "config/seasonality_config.yaml") -> dict:
    """Load seasonality configuration."""
    path = ROOT / config_path
    with open(path) as f:
        return yaml.safe_load(f)["seasonality"]


@njit(cache=True)
def _acf_lag12_core(series: np.ndarray) -> float:
    """Pure numerical ACF lag-12 computation (numba-accelerated when available)."""
    n = len(series)
    if n < 25:
        return 0.0
    total = 0.0
    for i in range(n):
        total += series[i]
    mean = total / n

    var_sum = 0.0
    for i in range(n):
        diff = series[i] - mean
        var_sum += diff * diff
    var = var_sum / n

    if var == 0.0:
        return 0.0

    cov = 0.0
    for i in range(n - 12):
        cov += (series[i] - mean) * (series[i + 12] - mean)
    cov = cov / n

    return cov / var


@njit(cache=True)
def _cv_of_monthly_means(mm_values: np.ndarray) -> float:
    """Compute coefficient of variation of monthly means (numba-accelerated)."""
    n = len(mm_values)
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        total += mm_values[i]
    mean = total / n
    if mean <= 0.0:
        return 0.0
    var_sum = 0.0
    for i in range(n):
        diff = mm_values[i] - mean
        var_sum += diff * diff
    std = (var_sum / n) ** 0.5
    return std / mean


def compute_acf_lag12(series: np.ndarray) -> float:
    """Compute autocorrelation at lag 12."""
    return float(_acf_lag12_core(series))


def compute_seasonality_metrics(
    dfu_sales: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """Compute all seasonality metrics for a single DFU.

    Parameters
    ----------
    dfu_sales : DataFrame with columns [startdate, qty], sorted by startdate
    config : seasonality config dict

    Returns
    -------
    dict with seasonality metrics and classification
    """
    min_months = config["min_months_history"]
    thresholds = config["thresholds"]
    confirmation = config["confirmation"]
    peak_trough_min = config["peak_trough_min_ratio"]

    qty = dfu_sales["qty"].fillna(0).values.astype(np.float64)
    months_available = len(qty)

    # Insufficient history check
    if months_available < min_months:
        return {
            "seasonality_profile": "insufficient_history",
            "seasonality_strength": None,
            "is_yearly_seasonal": None,
            "peak_month": None,
            "trough_month": None,
            "peak_trough_ratio": None,
            "yoy_correlation": None,
            "acf_lag12": None,
            "months_available": months_available,
        }

    # Step 1: Monthly means
    dfu_sales["month"] = dfu_sales["startdate"].dt.month
    monthly_means = dfu_sales.groupby("month")["qty"].mean()
    # Fill missing months with 0
    monthly_means = monthly_means.reindex(range(1, 13), fill_value=0.0)
    mm_values = monthly_means.values.astype(np.float64)

    # Step 2: Seasonality strength (CV of monthly means — numba-accelerated)
    seasonality_strength = float(_cv_of_monthly_means(mm_values))

    # Step 3: Year-over-year correlation
    dfu_sales["year"] = dfu_sales["startdate"].dt.year
    pivot = dfu_sales.pivot_table(values="qty", index="month", columns="year", aggfunc="mean")
    if pivot.shape[1] >= 2:
        corr_matrix = pivot.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper = corr_matrix.where(mask).stack()
        yoy_correlation = float(upper.mean()) if len(upper) > 0 else 0.0
    else:
        yoy_correlation = 0.0

    # Step 4: Autocorrelation at lag 12
    acf_lag12 = compute_acf_lag12(qty)

    # Step 5: Peak and trough
    peak_month = int(monthly_means.idxmax())
    trough_month = int(monthly_means.idxmin())
    trough_val = mm_values[trough_month - 1]
    peak_val = mm_values[peak_month - 1]
    peak_trough_ratio = float(peak_val / trough_val) if trough_val > 0 else None

    # Step 6: Profile classification
    has_confirmation = (
        yoy_correlation >= confirmation["yoy_correlation"]
        or acf_lag12 >= confirmation["acf_lag12"]
    )

    if seasonality_strength >= thresholds["high"] and has_confirmation:
        profile = "high"
    elif seasonality_strength >= thresholds["medium"] and has_confirmation:
        profile = "medium"
    elif seasonality_strength >= thresholds["low"]:
        profile = "low"
    else:
        profile = "none"

    # Step 7: Yearly seasonal flag
    is_yearly_seasonal = (
        seasonality_strength >= thresholds["low"]
        and has_confirmation
        and (peak_trough_ratio is not None and peak_trough_ratio >= peak_trough_min)
    )

    return {
        "seasonality_profile": profile,
        "seasonality_strength": round(seasonality_strength, 4),
        "is_yearly_seasonal": is_yearly_seasonal,
        "peak_month": peak_month,
        "trough_month": trough_month,
        "peak_trough_ratio": round(peak_trough_ratio, 4) if peak_trough_ratio is not None else None,
        "yoy_correlation": round(yoy_correlation, 4),
        "acf_lag12": round(acf_lag12, 4),
        "months_available": months_available,
    }


def _compute_seasonality_for_group(args: tuple) -> dict[str, Any]:
    """Worker function for parallel seasonality detection."""
    sku_ck, sales_data, config = args
    import pandas as pd  # noqa: F811 — re-import needed in worker process

    dfu_df = pd.DataFrame(sales_data)
    dfu_df["startdate"] = pd.to_datetime(dfu_df["startdate"])
    metrics = compute_seasonality_metrics(dfu_df.sort_values("startdate"), config)
    metrics["sku_ck"] = sku_ck
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect seasonality patterns in DFU sales")
    parser.add_argument("--config", type=str, default="config/seasonality_config.yaml", help="Config file path")
    parser.add_argument("--min-months", type=int, default=None, help="Override minimum months required")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--verbose", action="store_true", help="Print per-DFU diagnostics")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = load_config(args.config)

    if args.min_months is not None:
        config["min_months_history"] = args.min_months

    output_path = ROOT / (args.output or config["output_path"])

    print(f"Seasonality detection (min_months={config['min_months_history']})")
    print(f"Thresholds: low={config['thresholds']['low']}, "
          f"medium={config['thresholds']['medium']}, high={config['thresholds']['high']}")

    db = get_db_params()

    with profiled_section("load_sales_data"):
        with psycopg.connect(**db) as conn:
            print("Loading sales data...")
            sales_df = pd.read_sql(
                """
                SELECT d.sku_ck, s.startdate, s.qty
                FROM fact_sales_monthly s
                INNER JOIN dim_sku d
                    ON d.item_id = s.item_id
                    AND d.customer_group = s.customer_group
                    AND d.loc = s.loc
                WHERE s.qty IS NOT NULL
                ORDER BY d.sku_ck, s.startdate
                """,
                conn,
            )

        sales_df["startdate"] = pd.to_datetime(sales_df["startdate"])
        print(f"Loaded {len(sales_df)} sales records for {sales_df['sku_ck'].nunique()} DFUs")

    # Process each DFU
    with profiled_section("compute_seasonality"):
        grouped = sales_df.groupby("sku_ck", sort=False)
        n_groups = grouped.ngroups

        n_workers = min(multiprocessing.cpu_count(), 8)
        if n_groups > 500 and n_workers > 1 and not args.verbose:
            print(f"  Parallel mode: {n_workers} workers for {n_groups} DFUs")
            work_items = [
                (sku_ck, {"startdate": g["startdate"].values, "qty": g["qty"].values}, config)
                for sku_ck, g in grouped
            ]
            with multiprocessing.Pool(n_workers) as pool:
                results = pool.map(
                    _compute_seasonality_for_group,
                    work_items,
                    chunksize=max(1, n_groups // (n_workers * 4)),
                )
        else:
            # Serial fallback (small datasets or verbose mode)
            results = []
            for idx, (sku_ck, dfu_sales) in enumerate(grouped):
                if (idx + 1) % 2000 == 0 or idx == 0:
                    print(f"  Processing DFU {idx + 1}/{n_groups}...")

                metrics = compute_seasonality_metrics(dfu_sales.sort_values("startdate"), config)
                metrics["sku_ck"] = sku_ck
                results.append(metrics)

                if args.verbose and metrics["seasonality_profile"] in ("high", "medium"):
                    print(f"    {sku_ck}: {metrics['seasonality_profile']} "
                          f"(strength={metrics['seasonality_strength']}, "
                          f"yoy={metrics['yoy_correlation']}, acf12={metrics['acf_lag12']})")

        results_df = pd.DataFrame(results)

    # Save results
    with profiled_section("write_results"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\nSaved {len(results_df)} DFU seasonality profiles to {output_path}")

    # Print summary
    print("\nProfile distribution:")
    profile_counts = results_df["seasonality_profile"].value_counts()
    for profile, count in profile_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {profile}: {count} ({pct:.1f}%)")

    seasonal_count = results_df["is_yearly_seasonal"].sum()
    print(f"\nDFUs with yearly seasonal cycle: {seasonal_count} "
          f"({seasonal_count / len(results_df) * 100:.1f}%)")

    # Top seasonal DFUs
    ranked = results_df.dropna(subset=["seasonality_strength"])
    if len(ranked) > 0:
        top10 = ranked.nlargest(10, "seasonality_strength")
        print("\nTop 10 most seasonal DFUs:")
        for _, row in top10.iterrows():
            print(f"  {row['sku_ck']}: strength={row['seasonality_strength']}, "
                  f"profile={row['seasonality_profile']}, "
                  f"peak={row['peak_month']}, trough={row['trough_month']}")


if __name__ == "__main__":
    main()
