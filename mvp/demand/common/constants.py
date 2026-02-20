"""Shared constants for backtest scripts and feature engineering."""

# Feature engineering constants
CAT_FEATURES = ["ml_cluster", "region", "brand", "abc_vol"]
NUMERIC_DFU_FEATURES = ["execution_lag", "total_lt"]
NUMERIC_ITEM_FEATURES = ["case_weight", "item_proof", "bpc"]

LAG_RANGE = range(1, 13)  # qty_lag_1 .. qty_lag_12
ROLLING_WINDOWS = [3, 6, 12]

# Output column ordering for fact_external_forecast_monthly
OUTPUT_COLS = [
    "forecast_ck", "dmdunit", "dmdgroup", "loc",
    "fcstdate", "startdate", "lag", "execution_lag",
    "basefcst_pref", "tothist_dmd", "model_id",
]

# Output column ordering for backtest_lag_archive
ARCHIVE_COLS = [
    "forecast_ck", "dmdunit", "dmdgroup", "loc",
    "fcstdate", "startdate", "lag", "execution_lag",
    "basefcst_pref", "tothist_dmd", "model_id", "timeframe",
]

# Metadata columns excluded from feature set
METADATA_COLS = {"dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "qty", "_k"}

# Maximum archive lag (0-4)
MAX_ARCHIVE_LAG = 4

# Minimum training months required
MIN_TRAINING_MONTHS = 13

# Minimum rows per cluster for per-cluster training
MIN_CLUSTER_ROWS = 50
