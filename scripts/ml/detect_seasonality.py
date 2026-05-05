"""
Detect seasonality patterns in DFU monthly sales history.

.. deprecated::
    This script is deprecated.  Use ``scripts/ml/compute_sku_features.py``
    (backed by ``common/ml/sku_features/``) for all new work.  The unified
    module computes seasonality, variability, and lifecycle features in a
    single pass.  This file is kept only for backward-compatible function
    exports consumed by existing tests.

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params
from common.services.perf_profiler import profiled_section


def load_config(config_path: str = "config/forecasting/forecast_domain_config.yaml") -> dict:
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
    """Entry point — delegates to the unified SKU features pipeline.

    .. deprecated::
        This script is deprecated.  Run ``scripts/ml/compute_sku_features.py``
        directly for the unified pipeline.
    """
    import warnings

    warnings.warn(
        "scripts/detect_seasonality.py is deprecated. "
        "Use scripts/ml/compute_sku_features.py (backed by common/ml/sku_features/) instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    print(
        "WARNING: scripts/detect_seasonality.py is deprecated.\n"
        "  Delegating to the unified SKU features pipeline "
        "(scripts/ml/compute_sku_features.py).\n"
        "  Please update your workflow to call scripts/ml/compute_sku_features.py directly.\n"
    )

    # Re-use the unified pipeline's CLI so all flags are forwarded
    from scripts.ml.compute_sku_features import run_pipeline

    parser = argparse.ArgumentParser(
        description="[DEPRECATED] Detect seasonality — delegates to unified SKU features pipeline",
    )
    parser.add_argument("--config", type=str, default=None, help="(ignored, kept for backward compat)")
    parser.add_argument("--min-months", type=int, default=None, help="(ignored, kept for backward compat)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--verbose", action="store_true", help="(ignored, kept for backward compat)")
    parser.add_argument("--dry-run", action="store_true", help="Compute but do not write to DB")
    args = parser.parse_args()

    summary = run_pipeline(
        dry_run=args.dry_run,
        output_csv=args.output,
    )
    logger.info("Unified pipeline summary: %s", summary)


if __name__ == "__main__":
    main()
