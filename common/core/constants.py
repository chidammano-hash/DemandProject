"""Shared constants for backtest scripts and feature engineering."""

import logging

from common.utils import load_config

logger = logging.getLogger(__name__)

# Feature engineering constants
CAT_FEATURES = ["ml_cluster", "region", "brand", "abc_vol"]
NUMERIC_SKU_FEATURES = ["execution_lag", "total_lt"]
NUMERIC_ITEM_FEATURES = ["case_weight", "item_proof", "bpc"]

LAG_RANGE = range(1, 13)  # qty_lag_1 .. qty_lag_12
ROLLING_WINDOWS = [3, 6, 12]
CALENDAR_FEATURES = ["month", "quarter",
                     "is_quarter_end", "is_year_end", "days_in_month"]
DERIVED_FEATURES = ["mom_growth", "demand_accel", "volatility_ratio",
                    "lag_ratio_yoy", "lag_ratio_mom", "lag_ratio_3v12",
                    "n_zero_last_6m"]

# Per-DFU time-series profile features (computed from full history per DFU)
TS_PROFILE_FEATURES = [
    "cv_demand", "zero_demand_pct", "trend_slope_norm", "recency_ratio",
    "seasonal_amplitude", "adi", "mean_demand", "yoy_correlation",
]

# Fourier seasonal terms — sub-annual seasonality (quarterly, biannual patterns)
FOURIER_FEATURES: list[str] = []
for _period in [12, 6, 4, 3]:
    FOURIER_FEATURES.extend([f"fourier_sin_{_period}", f"fourier_cos_{_period}"])

# Croston decomposition features — intermittent demand signals
CROSTON_FEATURES = ["croston_demand_size", "croston_demand_interval", "croston_probability"]

# Cross-DFU cluster aggregate features — cluster-level demand signals
CROSS_DFU_FEATURES = ["cluster_mean_lag1", "cluster_total_lag1",
                      "cluster_demand_trend", "cluster_zero_pct"]

# External forecast signal features — optional enrichment
EXTERNAL_FORECAST_FEATURES = ["ext_fcst_ratio", "ext_fcst_lag1_ratio"]

# All enhanced features from the four new feature groups
ENHANCED_FEATURES = FOURIER_FEATURES + CROSTON_FEATURES + CROSS_DFU_FEATURES + EXTERNAL_FORECAST_FEATURES

# Features always kept by SHAP selection (never dropped regardless of SHAP rank).
# These provide essential temporal/categorical context the model needs.
# Fourier terms are calendar-derived (no leakage), so they are also protected.
PROTECTED_FEATURES = {
    "month", "quarter", "ml_cluster",
    *FOURIER_FEATURES,
}

# Output column ordering for fact_external_forecast_monthly
OUTPUT_COLS = [
    "forecast_ck", "item_id", "customer_group", "loc",
    "fcstdate", "startdate", "lag", "execution_lag",
    "basefcst_pref", "tothist_dmd", "model_id",
]

# Output column ordering for backtest_lag_archive
ARCHIVE_COLS = [
    "forecast_ck", "item_id", "customer_group", "loc",
    "fcstdate", "startdate", "lag", "execution_lag",
    "basefcst_pref", "tothist_dmd", "model_id", "timeframe",
]

# Metadata columns excluded from feature set
METADATA_COLS = {"sku_ck", "item_id", "customer_group", "loc", "startdate", "qty", "_k"}

# Maximum archive lag (0-4)
MAX_ARCHIVE_LAG = 4

# Minimum training months required
MIN_TRAINING_MONTHS = 13

# Minimum rows per cluster for per-cluster training (static floor)
MIN_CLUSTER_ROWS = 50


def compute_min_cluster_rows(
    n_features: int,
    samples_per_feature: int | None = None,
    floor: int = 50,
) -> int:
    """Minimum training rows for a cluster, scaled by feature count.

    Uses a ratio of ``samples_per_feature`` per feature (default from
    ``config/algorithm_config.yaml`` ``cluster_sizing.samples_per_feature``,
    falling back to 3), with a *floor* of 50 rows regardless.  Accounts for
    the 80/20 train/val split by inflating by 1.25x so that the *training*
    partition alone has enough rows.

    Parameters
    ----------
    n_features:
        Number of features used for model training.
    samples_per_feature:
        Minimum samples per feature before the split inflation.
        When *None*, reads from ``algorithm_config.yaml``.
    floor:
        Absolute minimum regardless of feature count.
    """
    if samples_per_feature is None:
        cfg = load_config("algorithm_config.yaml")
        samples_per_feature = (
            cfg.get("cluster_sizing", {}).get("samples_per_feature", 3)
        )
    min_for_features = int(n_features * samples_per_feature * 1.25)  # 1.25x for 80/20 split
    result = max(floor, min_for_features)
    if result > MIN_CLUSTER_ROWS:
        logger.info(
            "Cluster min rows scaled to %d (n_features=%d, spf=%d, floor=%d)",
            result, n_features, samples_per_feature, floor,
        )
    return result
