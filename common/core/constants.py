"""Shared constants for backtest scripts and feature engineering."""

import logging

from common.utils import load_config

logger = logging.getLogger(__name__)

# Feature engineering constants
CAT_FEATURES = ["region", "brand", "abc_vol"]
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
    "periodicity",
]

# Fourier seasonal terms — sub-annual seasonality (quarterly, biannual patterns)
FOURIER_FEATURES: list[str] = []
for _period in [12, 6, 4, 3]:
    FOURIER_FEATURES.extend([f"fourier_sin_{_period}", f"fourier_cos_{_period}"])

# Croston decomposition features — intermittent demand signals
CROSTON_FEATURES = ["croston_demand_size", "croston_demand_interval", "croston_probability"]

# Cross-DFU cluster aggregate features — REMOVED (ml_cluster leaks future info
# into backtesting; see docs/KNOWN_GAPS.md §1). Kept as empty list for backward
# compat with code that iterates over it.
CROSS_DFU_FEATURES: list[str] = []

# External forecast signal features — optional enrichment
EXTERNAL_FORECAST_FEATURES = ["ext_fcst_ratio", "ext_fcst_lag1_ratio"]

# Customer-derived features (from customer_features_monthly table)
CUSTOMER_CONCENTRATION_FEATURES = [
    "n_active_cust", "n_active_cust_6m", "hhi_demand",
    "top1_cust_share", "top3_cust_share", "cust_gini",
]
CUSTOMER_DYNAMICS_FEATURES = [
    "new_cust_demand_share", "churned_cust_demand_share",
    "cust_count_mom", "cust_retention_rate", "cust_tenure_mean",
]
CUSTOMER_TRUE_DEMAND_FEATURES = [
    "true_demand_ratio", "oos_rate", "oos_cust_pct",
    "demand_sales_gap_3m", "oos_trend",
    "demand_qty_lag1", "demand_qty_lag3_mean",
]
CUSTOMER_CHANNEL_MIX_FEATURES = [
    "channel_entropy", "dominant_channel_share",
    "channel_mix_shift", "on_premise_share",
]
CUSTOMER_CROSS_FEATURES = [
    "cust_demand_cv_mean", "cust_demand_sync", "max_cust_share_delta",
]
CUSTOMER_ATTRIBUTE_MIX_FEATURES = [
    "store_type_entropy", "dominant_store_type_share",
    "chain_ratio", "top_chain_share",
    "sub_channel_entropy",
    "active_cust_pct", "avg_delivery_freq",
    "on_premise_acct_share", "premise_diversity",
]
CUSTOMER_FEATURE_COLS = (
    CUSTOMER_CONCENTRATION_FEATURES
    + CUSTOMER_DYNAMICS_FEATURES
    + CUSTOMER_TRUE_DEMAND_FEATURES
    + CUSTOMER_CHANNEL_MIX_FEATURES
    + CUSTOMER_CROSS_FEATURES
    + CUSTOMER_ATTRIBUTE_MIX_FEATURES
)

# All enhanced features from the four new feature groups
ENHANCED_FEATURES = FOURIER_FEATURES + CROSTON_FEATURES + CROSS_DFU_FEATURES + EXTERNAL_FORECAST_FEATURES

# Features always kept by SHAP selection (never dropped regardless of SHAP rank).
# These provide essential temporal/categorical context the model needs.
# Fourier terms are calendar-derived (no leakage), so they are also protected.
# NOTE: ml_cluster removed — causes leakage in backtesting (see docs/KNOWN_GAPS.md §1).
PROTECTED_FEATURES = {
    "month", "quarter",
    *FOURIER_FEATURES,
    # Croston decomposition features handle intermittent demand correctly;
    # protect them so SHAP selection cannot strip them (causes bias on sparse SKUs).
    "croston_demand_size", "croston_probability",
    # Customer enrichment — core signals that should never be SHAP-pruned
    "true_demand_ratio", "n_active_cust", "hhi_demand",
    # Core demand signals — highly correlated with each other but each captures
    # a distinct aspect (level, recency, short/medium trend). The correlation
    # filter was dropping these in favour of lower-variance proxies, destroying
    # model accuracy (especially for low-volume clusters).
    "mean_demand", "qty_lag_1", "rolling_mean_3m", "rolling_mean_6m",
    # Lag features essential for recursive prediction — if SHAP drops these,
    # the recursive chain (month N predictions become month N+1 lag inputs)
    # loses signal and accuracy collapses. Protect the first 3 lags which
    # carry the strongest recency signal.
    "qty_lag_2", "qty_lag_3",
    "rolling_mean_12m",
}

# Exact-duplicate feature pairs: alias → canonical name to keep.
# The alias is dropped statically before any per-timeframe computation.
DUPLICATE_FEATURE_ALIASES: dict[str, str] = {
    "year_over_year_correlation": "yoy_correlation",
    "sparsity_score": "zero_demand_pct",
    "growth_rate": "cagr",
    "recent_vs_historical": "recency_ratio",
    "demand_stability": "cv_demand",
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
METADATA_COLS = {"sku_ck", "item_id", "customer_group", "loc", "startdate", "qty", "_k", "ml_cluster"}

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
    ``config/forecast_pipeline_config.yaml``
    ``clustering.samples_per_feature``, falling back to 3), with a *floor*
    of 50 rows regardless.  Accounts for the 80/20 train/val split by
    inflating by 1.25x so that the *training* partition alone has enough
    rows.

    Parameters
    ----------
    n_features:
        Number of features used for model training.
    samples_per_feature:
        Minimum samples per feature before the split inflation.
        When *None*, reads from ``forecast_pipeline_config.yaml``.
    floor:
        Absolute minimum regardless of feature count.
    """
    if samples_per_feature is None:
        from common.utils import load_forecast_pipeline_config
        pcfg = load_forecast_pipeline_config()
        samples_per_feature = (
            pcfg.get("clustering", {}).get("samples_per_feature", 3)
        )
    min_for_features = int(n_features * samples_per_feature * 1.25)  # 1.25x for 80/20 split
    result = max(floor, min_for_features)
    if result > MIN_CLUSTER_ROWS:
        logger.info(
            "Cluster min rows scaled to %d (n_features=%d, spf=%d, floor=%d)",
            result, n_features, samples_per_feature, floor,
        )
    return result
