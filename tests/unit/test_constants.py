"""Tests for common/constants.py."""

from common.constants import (
    CAT_FEATURES,
    CROSTON_FEATURES,
    CROSS_DFU_FEATURES,
    DERIVED_FEATURES,
    ENHANCED_FEATURES,
    EXTERNAL_FORECAST_FEATURES,
    FOURIER_FEATURES,
    NUMERIC_SKU_FEATURES,
    NUMERIC_ITEM_FEATURES,
    LAG_RANGE,
    ROLLING_WINDOWS,
    OUTPUT_COLS,
    ARCHIVE_COLS,
    METADATA_COLS,
    MAX_ARCHIVE_LAG,
    MIN_TRAINING_MONTHS,
    MIN_CLUSTER_ROWS,
    TS_PROFILE_FEATURES,
    PROTECTED_FEATURES,
)


class TestConstants:
    def test_lag_range_bounds(self):
        assert min(LAG_RANGE) == 1
        assert max(LAG_RANGE) == 12
        assert len(list(LAG_RANGE)) == 12

    def test_rolling_windows_sorted(self):
        assert ROLLING_WINDOWS == sorted(ROLLING_WINDOWS)
        assert ROLLING_WINDOWS == [3, 6, 12]

    def test_output_cols_non_empty(self):
        assert len(OUTPUT_COLS) > 0
        assert len(set(OUTPUT_COLS)) == len(OUTPUT_COLS), "No duplicates in OUTPUT_COLS"

    def test_archive_cols_non_empty(self):
        assert len(ARCHIVE_COLS) > 0
        assert len(set(ARCHIVE_COLS)) == len(ARCHIVE_COLS), "No duplicates in ARCHIVE_COLS"

    def test_archive_cols_superset_of_output_cols(self):
        """Archive cols should contain all output cols plus timeframe."""
        output_set = set(OUTPUT_COLS)
        archive_set = set(ARCHIVE_COLS)
        assert output_set.issubset(archive_set)
        assert "timeframe" in archive_set

    def test_cat_features_non_empty(self):
        assert len(CAT_FEATURES) > 0
        assert all(isinstance(f, str) for f in CAT_FEATURES)

    def test_numeric_features_non_empty(self):
        assert len(NUMERIC_SKU_FEATURES) > 0
        assert len(NUMERIC_ITEM_FEATURES) > 0

    def test_metadata_cols_is_set(self):
        assert isinstance(METADATA_COLS, set)
        assert "sku_ck" in METADATA_COLS
        assert "startdate" in METADATA_COLS
        assert "qty" in METADATA_COLS

    def test_max_archive_lag(self):
        assert MAX_ARCHIVE_LAG == 4

    def test_min_training_months(self):
        assert MIN_TRAINING_MONTHS >= 1

    def test_min_cluster_rows(self):
        assert MIN_CLUSTER_ROWS >= 1

    def test_derived_features_includes_lag_ratios(self):
        assert "lag_ratio_yoy" in DERIVED_FEATURES
        assert "lag_ratio_mom" in DERIVED_FEATURES
        assert "lag_ratio_3v12" in DERIVED_FEATURES
        assert "n_zero_last_6m" in DERIVED_FEATURES

    def test_protected_features_includes_temporal(self):
        assert "month" in PROTECTED_FEATURES
        assert "quarter" in PROTECTED_FEATURES
        assert "ml_cluster" in PROTECTED_FEATURES
        # fourier_sin_12/cos_12 replace legacy month_sin/cos
        assert "fourier_sin_12" in PROTECTED_FEATURES
        assert "fourier_cos_12" in PROTECTED_FEATURES
        assert "month_sin" not in PROTECTED_FEATURES
        assert "month_cos" not in PROTECTED_FEATURES
        assert isinstance(PROTECTED_FEATURES, set)

    def test_ts_profile_features_non_empty(self):
        assert len(TS_PROFILE_FEATURES) == 8
        assert all(isinstance(f, str) for f in TS_PROFILE_FEATURES)
        assert "cv_demand" in TS_PROFILE_FEATURES
        assert "seasonal_amplitude" in TS_PROFILE_FEATURES

    def test_fourier_features(self):
        assert len(FOURIER_FEATURES) == 8  # 4 periods × 2 (sin + cos)
        assert all(isinstance(f, str) for f in FOURIER_FEATURES)
        for period in [12, 6, 4, 3]:
            assert f"fourier_sin_{period}" in FOURIER_FEATURES
            assert f"fourier_cos_{period}" in FOURIER_FEATURES

    def test_croston_features(self):
        assert len(CROSTON_FEATURES) == 3
        assert "croston_demand_size" in CROSTON_FEATURES
        assert "croston_demand_interval" in CROSTON_FEATURES
        assert "croston_probability" in CROSTON_FEATURES

    def test_cross_dfu_features(self):
        assert len(CROSS_DFU_FEATURES) == 4
        assert "cluster_mean_lag1" in CROSS_DFU_FEATURES
        assert "cluster_total_lag1" in CROSS_DFU_FEATURES
        assert "cluster_demand_trend" in CROSS_DFU_FEATURES
        assert "cluster_zero_pct" in CROSS_DFU_FEATURES

    def test_external_forecast_features(self):
        assert len(EXTERNAL_FORECAST_FEATURES) == 2
        assert "ext_fcst_ratio" in EXTERNAL_FORECAST_FEATURES
        assert "ext_fcst_lag1_ratio" in EXTERNAL_FORECAST_FEATURES

    def test_enhanced_features_combines_all_groups(self):
        expected = FOURIER_FEATURES + CROSTON_FEATURES + CROSS_DFU_FEATURES + EXTERNAL_FORECAST_FEATURES
        assert ENHANCED_FEATURES == expected
        assert len(ENHANCED_FEATURES) == 8 + 3 + 4 + 2  # 17 total

    def test_protected_features_includes_fourier(self):
        for feat in FOURIER_FEATURES:
            assert feat in PROTECTED_FEATURES, f"{feat} should be protected"
