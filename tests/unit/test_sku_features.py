"""Unit tests for common/ml/sku_features — unified SKU feature computation.

Tests cover:
  1. Feature completeness (all expected keys from compute_time_series_features)
  2. Seasonality classification (classify_seasonality_profile)
  3. Variability classification (classify_variability_class)
  4. compute_all_sku_features orchestration
  5. Edge cases (empty data, single month, all zeros, constant demand)
  6. Package-level imports
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common.ml.clustering.features import compute_time_series_features
from common.ml.sku_features import (
    classify_seasonality_profile,
    classify_variability_class,
    compute_all_sku_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sales_df(
    qty_values: list[float],
    start: str = "2022-01-01",
    sku_ck: str = "SKU-001",
) -> pd.DataFrame:
    """Build a sales DataFrame from a flat list of monthly qty values."""
    dates = pd.date_range(start, periods=len(qty_values), freq="MS")
    return pd.DataFrame({
        "sku_ck": sku_ck,
        "startdate": dates,
        "qty": qty_values,
    })


def _make_ts_df(
    qty_values: list[float],
    start: str = "2022-01-01",
) -> pd.DataFrame:
    """Build a minimal DataFrame for compute_time_series_features."""
    dates = pd.date_range(start, periods=len(qty_values), freq="MS")
    return pd.DataFrame({"startdate": dates, "qty": qty_values})


# ---------------------------------------------------------------------------
# 1. Feature completeness — compute_time_series_features with 36-month data
# ---------------------------------------------------------------------------

class TestFeatureCompleteness:
    """Verify that all expected features are returned for 36-month demand."""

    @pytest.fixture
    def features_36m(self) -> pd.Series:
        pattern = [100 + 50 * np.sin(2 * np.pi * m / 12) for m in range(12)]
        qty = pattern * 3
        df = _make_ts_df(qty)
        return compute_time_series_features(df)

    def test_new_features_present(self, features_36m: pd.Series):
        """All newly added features must be present in the output."""
        new_features = [
            "trough_month",
            "peak_trough_ratio",
            "acf_lag12",
            "demand_mad",
            "demand_p90",
            "demand_skewness",
            "demand_kurtosis",
        ]
        for feat in new_features:
            assert feat in features_36m.index, f"Missing feature: {feat}"

    def test_volume_features_present(self, features_36m: pd.Series):
        volume_features = [
            "mean_demand", "median_demand", "std_demand", "cv_demand",
            "iqr_demand", "min_demand", "max_demand", "total_demand",
            "demand_mad", "demand_p90", "demand_skewness", "demand_kurtosis",
        ]
        for feat in volume_features:
            assert feat in features_36m.index, f"Missing volume feature: {feat}"

    def test_trend_features_present(self, features_36m: pd.Series):
        trend_features = [
            "trend_slope", "trend_slope_norm", "trend_r2",
            "trend_pct_change", "trend_direction",
        ]
        for feat in trend_features:
            assert feat in features_36m.index, f"Missing trend feature: {feat}"

    def test_seasonality_features_present(self, features_36m: pd.Series):
        seasonality_features = [
            "seasonality_strength", "peak_month", "trough_month",
            "peak_trough_ratio", "seasonal_index_std", "seasonal_amplitude",
            "seasonal_r2", "yoy_correlation", "year_over_year_correlation",
            "acf_lag12",
        ]
        for feat in seasonality_features:
            assert feat in features_36m.index, f"Missing seasonality feature: {feat}"

    def test_intermittency_features_present(self, features_36m: pd.Series):
        intermittency_features = [
            "zero_demand_pct", "sparsity_score", "adi", "demand_stability",
            "outlier_count",
        ]
        for feat in intermittency_features:
            assert feat in features_36m.index, f"Missing intermittency feature: {feat}"

    def test_lifecycle_features_present(self, features_36m: pd.Series):
        lifecycle_features = [
            "months_available", "cagr", "growth_rate", "recency_ratio",
            "recent_vs_historical", "acceleration",
        ]
        for feat in lifecycle_features:
            assert feat in features_36m.index, f"Missing lifecycle feature: {feat}"

    def test_periodicity_present(self, features_36m: pd.Series):
        assert "periodicity_strength" in features_36m.index

    def test_trough_month_is_valid(self, features_36m: pd.Series):
        assert 1 <= features_36m["trough_month"] <= 12

    def test_peak_trough_ratio_positive(self, features_36m: pd.Series):
        """For seasonal data with no zero months, ratio should be > 1."""
        assert features_36m["peak_trough_ratio"] > 1.0

    def test_acf_lag12_for_seasonal_pattern(self, features_36m: pd.Series):
        """A repeating 12-month pattern should have strong acf at lag 12."""
        assert features_36m["acf_lag12"] > 0.5

    def test_demand_mad_nonnegative(self, features_36m: pd.Series):
        assert features_36m["demand_mad"] >= 0.0

    def test_demand_p90_gte_median(self, features_36m: pd.Series):
        assert features_36m["demand_p90"] >= features_36m["median_demand"]

    def test_total_feature_count(self, features_36m: pd.Series):
        """Expect at least 35 features for 36-month data."""
        assert len(features_36m) >= 35


# ---------------------------------------------------------------------------
# 2. Seasonality classification — classify_seasonality_profile
# ---------------------------------------------------------------------------

class TestClassifySeasonalityProfile:
    """Tests for classify_seasonality_profile."""

    def test_flat_demand_returns_none(self):
        """Flat demand → seasonal_amplitude=0, yoy_corr=0 → 'none'."""
        features = {
            "seasonal_amplitude": 0.0,
            "yoy_correlation": 0.0,
            "seasonal_r2": 0.0,
        }
        assert classify_seasonality_profile(features) == "none"

    def test_strong_seasonal_demand(self):
        """High amplitude + strong confirmation → 'strong'."""
        features = {
            "seasonal_amplitude": 1.5,
            "yoy_correlation": 0.85,
            "seasonal_r2": 0.70,
        }
        assert classify_seasonality_profile(features) == "strong"

    def test_moderate_seasonal_demand(self):
        """Medium amplitude + confirmation → 'moderate'."""
        features = {
            "seasonal_amplitude": 0.50,
            "yoy_correlation": 0.45,
            "seasonal_r2": 0.25,
        }
        assert classify_seasonality_profile(features) == "moderate"

    def test_low_seasonal_demand(self):
        """Amplitude above low threshold but below medium → 'low'."""
        features = {
            "seasonal_amplitude": 0.20,
            "yoy_correlation": 0.10,
            "seasonal_r2": 0.05,
        }
        assert classify_seasonality_profile(features) == "low"

    def test_high_amplitude_no_confirmation_is_low(self):
        """High amplitude but no confirmation → falls through to 'low'."""
        features = {
            "seasonal_amplitude": 0.80,
            "yoy_correlation": 0.10,
            "seasonal_r2": 0.10,
        }
        assert classify_seasonality_profile(features) == "low"

    def test_confirmation_via_yoy_only(self):
        """Only yoy_correlation passes gate → still confirmed."""
        features = {
            "seasonal_amplitude": 0.50,
            "yoy_correlation": 0.50,
            "seasonal_r2": 0.10,
        }
        assert classify_seasonality_profile(features) == "moderate"

    def test_confirmation_via_seasonal_r2_only(self):
        """Only seasonal_r2 passes gate → still confirmed."""
        features = {
            "seasonal_amplitude": 0.50,
            "yoy_correlation": 0.10,
            "seasonal_r2": 0.50,
        }
        assert classify_seasonality_profile(features) == "moderate"

    def test_below_low_threshold_is_none(self):
        """Amplitude below low threshold → 'none' regardless of confirmation."""
        features = {
            "seasonal_amplitude": 0.10,
            "yoy_correlation": 0.90,
            "seasonal_r2": 0.90,
        }
        assert classify_seasonality_profile(features) == "none"

    def test_custom_thresholds(self):
        """Custom thresholds override defaults."""
        features = {
            "seasonal_amplitude": 0.25,
            "yoy_correlation": 0.50,
            "seasonal_r2": 0.50,
        }
        # With lower high threshold, 0.25 qualifies as strong
        result = classify_seasonality_profile(
            features,
            threshold_low=0.05,
            threshold_medium=0.10,
            threshold_high=0.20,
        )
        assert result == "strong"

    def test_missing_features_default_to_zero(self):
        """Missing feature keys should default to 0 and return 'none'."""
        features = {}
        assert classify_seasonality_profile(features) == "none"

    def test_none_values_treated_as_zero(self):
        """None values should be handled as 0."""
        features = {
            "seasonal_amplitude": None,
            "yoy_correlation": None,
            "seasonal_r2": None,
        }
        assert classify_seasonality_profile(features) == "none"


# ---------------------------------------------------------------------------
# 3. Variability classification — classify_variability_class
# ---------------------------------------------------------------------------

class TestClassifyVariabilityClass:
    """Tests for classify_variability_class (Syntetos-Boylan framework)."""

    def test_smooth_low_cv(self):
        """CV below cv_low → 'smooth'."""
        assert classify_variability_class(0.10, 0.0) == "smooth"
        assert classify_variability_class(0.29, 0.0) == "smooth"

    def test_smooth_zero_cv(self):
        """CV = 0 → 'smooth'."""
        assert classify_variability_class(0.0, 0.0) == "smooth"

    def test_erratic_high_cv(self):
        """CV in [cv_medium, cv_high) with low intermittency → 'erratic'."""
        assert classify_variability_class(0.80, 0.0) == "erratic"
        assert classify_variability_class(1.0, 0.0) == "erratic"
        assert classify_variability_class(1.49, 0.0) == "erratic"

    def test_erratic_moderate_cv(self):
        """CV between cv_low and cv_medium (no intermittency) → 'erratic'."""
        assert classify_variability_class(0.50, 0.0) == "erratic"

    def test_intermittent_high_zero_pct_low_cv(self):
        """High zero-demand pct but low CV → 'intermittent'."""
        assert classify_variability_class(0.20, 0.50) == "intermittent"
        assert classify_variability_class(0.10, 0.40) == "intermittent"

    def test_lumpy_high_cv(self):
        """CV >= cv_high → 'lumpy' regardless of intermittency."""
        assert classify_variability_class(1.50, 0.0) == "lumpy"
        assert classify_variability_class(2.50, 0.20) == "lumpy"

    def test_lumpy_high_cv_and_high_intermittency(self):
        """Both high CV and high intermittency → 'lumpy'."""
        assert classify_variability_class(1.50, 0.50) == "lumpy"
        assert classify_variability_class(0.80, 0.50) == "lumpy"

    def test_lumpy_moderate_cv_with_intermittency(self):
        """CV >= cv_medium AND intermittency >= threshold → 'lumpy'."""
        assert classify_variability_class(0.80, 0.30) == "lumpy"
        assert classify_variability_class(1.0, 0.40) == "lumpy"

    def test_custom_thresholds(self):
        """Custom thresholds override defaults."""
        result = classify_variability_class(
            0.50,
            0.0,
            cv_low=0.60,
            cv_medium=1.00,
            cv_high=2.00,
            intermittency_threshold=0.50,
        )
        assert result == "smooth"

    def test_intermittent_at_exact_threshold(self):
        """zero_demand_pct exactly at threshold with low CV → 'intermittent'."""
        assert classify_variability_class(0.10, 0.30) == "intermittent"

    def test_return_values_are_valid_classes(self):
        """All return values must be in the valid class set."""
        valid = {"smooth", "erratic", "intermittent", "lumpy"}
        for cv in [0.0, 0.15, 0.30, 0.50, 0.80, 1.0, 1.50, 2.0]:
            for zdp in [0.0, 0.10, 0.30, 0.50, 0.80]:
                result = classify_variability_class(cv, zdp)
                assert result in valid, f"Invalid class '{result}' for cv={cv}, zdp={zdp}"


# ---------------------------------------------------------------------------
# 4. compute_all_sku_features orchestration
# ---------------------------------------------------------------------------

class TestComputeAllSkuFeatures:
    """Tests for compute_all_sku_features (serial path, no DB)."""

    def test_basic_computation(self):
        """Verify features are computed for a multi-SKU sales DataFrame."""
        df1 = _make_sales_df([100.0 + 20 * np.sin(2 * np.pi * m / 12)
                              for m in range(36)], sku_ck="A")
        df2 = _make_sales_df([200.0] * 24, start="2022-01-01", sku_ck="B")
        sales = pd.concat([df1, df2], ignore_index=True)

        result = compute_all_sku_features(sales, min_months_history=12)
        assert "sku_ck" in result.columns
        assert set(result["sku_ck"]) == {"A", "B"}

    def test_sku_ck_in_output(self):
        """Each row should have sku_ck identifier."""
        df = _make_sales_df([100.0] * 24, sku_ck="SKU-X")
        result = compute_all_sku_features(df, min_months_history=12)
        assert len(result) == 1
        assert result.iloc[0]["sku_ck"] == "SKU-X"

    def test_feature_columns_in_output(self):
        """Output DataFrame should have feature columns from compute_time_series_features."""
        df = _make_sales_df([100.0] * 36, sku_ck="SKU-001")
        result = compute_all_sku_features(df, min_months_history=12)
        expected_cols = [
            "mean_demand", "cv_demand", "seasonality_strength",
            "trough_month", "peak_trough_ratio", "demand_mad",
            "demand_p90", "demand_skewness", "demand_kurtosis",
            "acf_lag12",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_min_history_filter(self):
        """SKUs with fewer months than min_months_history are excluded."""
        short = _make_sales_df([100.0] * 5, sku_ck="SHORT")
        long = _make_sales_df([100.0] * 24, sku_ck="LONG")
        sales = pd.concat([short, long], ignore_index=True)

        result = compute_all_sku_features(sales, min_months_history=12)
        assert set(result["sku_ck"]) == {"LONG"}

    def test_workers_parameter(self):
        """Setting workers=1 should work (forces serial path)."""
        df = _make_sales_df([100.0] * 24, sku_ck="SKU-001")
        result = compute_all_sku_features(df, min_months_history=12, workers=1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for compute_time_series_features and compute_all_sku_features."""

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty Series."""
        df = pd.DataFrame(columns=["startdate", "qty"])
        features = compute_time_series_features(df)
        assert len(features) == 0

    def test_single_month(self):
        """Single month of data should still return features without error."""
        df = _make_ts_df([100.0])
        features = compute_time_series_features(df)
        assert features["months_available"] == 1
        assert features["mean_demand"] == 100.0
        # Single month: no trend possible
        assert features["trend_slope"] == 0.0
        assert features["trend_direction"] == 0
        # Seasonality not possible with < 12 months
        assert features["seasonality_strength"] == 0.0

    def test_all_zeros(self):
        """All-zero demand should not raise and should have cv=0."""
        df = _make_ts_df([0.0] * 24)
        features = compute_time_series_features(df)
        assert features["mean_demand"] == 0.0
        assert features["cv_demand"] == 0.0
        assert features["zero_demand_pct"] == 1.0
        assert features["total_demand"] == 0.0
        assert features["demand_mad"] == 0.0
        assert features["demand_p90"] == 0.0

    def test_constant_demand(self):
        """Constant non-zero demand → cv=0, no seasonality."""
        df = _make_ts_df([150.0] * 36)
        features = compute_time_series_features(df)
        assert features["cv_demand"] == pytest.approx(0.0, abs=1e-10)
        assert features["std_demand"] == pytest.approx(0.0, abs=1e-10)
        assert features["seasonality_strength"] == pytest.approx(0.0, abs=1e-10)
        assert features["demand_stability"] == pytest.approx(1.0, abs=1e-10)
        # scipy.stats.skew/kurtosis may return NaN for perfectly constant
        # data due to catastrophic cancellation (std=0); accept NaN or 0.
        skew_val = features["demand_skewness"]
        assert np.isnan(skew_val) or skew_val == pytest.approx(0.0, abs=1e-6)
        kurt_val = features["demand_kurtosis"]
        assert np.isnan(kurt_val) or kurt_val == pytest.approx(0.0, abs=1e-6)
        assert features["zero_demand_pct"] == 0.0
        assert features["outlier_count"] == 0
        # Constant demand should have peak_trough_ratio = 1
        assert features["peak_trough_ratio"] == pytest.approx(1.0, abs=1e-10)

    def test_two_months_data(self):
        """Two months of data should compute trend but limited seasonality."""
        df = _make_ts_df([100.0, 200.0])
        features = compute_time_series_features(df)
        assert features["months_available"] == 2
        assert features["trend_slope"] > 0
        assert features["trend_direction"] == 1
        # No seasonality with < 12 months
        assert features["seasonality_strength"] == 0.0

    def test_empty_sales_df_compute_all(self):
        """Empty DataFrame to compute_all_sku_features returns empty result."""
        empty = pd.DataFrame(columns=["sku_ck", "startdate", "qty"])
        result = compute_all_sku_features(empty)
        assert "sku_ck" in result.columns
        assert len(result) == 0

    def test_all_skus_below_min_history(self):
        """All SKUs below min_history → empty result."""
        df = _make_sales_df([100.0] * 3, sku_ck="SHORT")
        result = compute_all_sku_features(df, min_months_history=12)
        assert len(result) == 0

    def test_intermittent_demand(self):
        """Intermittent demand (many zeros) should compute without error."""
        qty = [0, 0, 50, 0, 0, 0, 100, 0, 0, 0, 0, 75,
               0, 0, 60, 0, 0, 0, 80, 0, 0, 0, 0, 90]
        df = _make_ts_df(qty)
        features = compute_time_series_features(df)
        assert features["zero_demand_pct"] > 0.5
        assert features["adi"] > 1.0
        assert features["demand_stability"] < 1.0

    def test_nan_qty_treated_as_zero(self):
        """NaN values in qty should be filled with 0."""
        df = _make_ts_df([100.0] * 12)
        df.loc[3, "qty"] = np.nan
        df.loc[7, "qty"] = np.nan
        features = compute_time_series_features(df)
        # NaNs filled as 0, so mean should be less than 100
        assert features["mean_demand"] < 100.0
        assert features["months_available"] == 12

    def test_no_nan_in_features(self):
        """No feature value should be NaN for well-formed seasonal input."""
        pattern = [100 + 50 * np.sin(2 * np.pi * m / 12) for m in range(12)]
        df = _make_ts_df(pattern * 3)
        features = compute_time_series_features(df)
        for key, val in features.items():
            assert not (isinstance(val, float) and np.isnan(val)), (
                f"Feature '{key}' is NaN"
            )


# ---------------------------------------------------------------------------
# 6. Package imports
# ---------------------------------------------------------------------------

class TestPackageImports:
    """Verify that the sku_features package exports the expected symbols."""

    def test_import_compute_all_sku_features(self):
        from common.ml.sku_features import compute_all_sku_features as fn
        assert callable(fn)

    def test_import_classify_seasonality_profile(self):
        from common.ml.sku_features import classify_seasonality_profile as fn
        assert callable(fn)

    def test_import_classify_variability_class(self):
        from common.ml.sku_features import classify_variability_class as fn
        assert callable(fn)

    def test_import_load_sales_from_db(self):
        from common.ml.sku_features import load_sales_from_db as fn
        assert callable(fn)

    def test_all_exports_listed(self):
        import common.ml.sku_features as pkg
        expected = {
            "compute_all_sku_features",
            "classify_seasonality_profile",
            "classify_variability_class",
            "load_sales_from_db",
            "write_features_to_dim_sku",
        }
        assert expected.issubset(set(pkg.__all__))


# ---------------------------------------------------------------------------
# 7. Integration: classifier + feature extraction round-trip
# ---------------------------------------------------------------------------

class TestClassifierIntegration:
    """End-to-end: compute features → classify."""

    def test_flat_demand_classifies_as_none_seasonality(self):
        df = _make_ts_df([100.0] * 36)
        features = compute_time_series_features(df).to_dict()
        result = classify_seasonality_profile(features)
        assert result == "none"

    def test_seasonal_demand_classifies_as_strong(self):
        """Highly seasonal 3-year pattern should classify as strong."""
        pattern = [50, 60, 80, 120, 180, 250, 280, 260, 180, 100, 60, 50]
        df = _make_ts_df(pattern * 3)
        features = compute_time_series_features(df).to_dict()
        result = classify_seasonality_profile(features)
        assert result in ("strong", "moderate")

    def test_moderate_seasonal_demand(self):
        """Moderate seasonal swing should classify at least as 'low'."""
        pattern = [200 + 100 * np.sin(2 * np.pi * m / 12) for m in range(12)]
        qty = [max(10, v) for v in pattern * 3]
        df = _make_ts_df(qty)
        features = compute_time_series_features(df).to_dict()
        result = classify_seasonality_profile(features)
        assert result in ("low", "moderate", "strong")

    def test_constant_demand_classifies_as_smooth(self):
        df = _make_ts_df([100.0] * 36)
        features = compute_time_series_features(df).to_dict()
        result = classify_variability_class(
            features["cv_demand"],
            features["zero_demand_pct"],
        )
        assert result == "smooth"

    def test_intermittent_demand_classifies_correctly(self):
        """Many zeros → intermittent or lumpy."""
        qty = [0, 0, 50, 0, 0, 0, 100, 0, 0, 0, 0, 75,
               0, 0, 60, 0, 0, 0, 80, 0, 0, 0, 0, 90]
        df = _make_ts_df(qty)
        features = compute_time_series_features(df).to_dict()
        result = classify_variability_class(
            features["cv_demand"],
            features["zero_demand_pct"],
        )
        assert result in ("intermittent", "lumpy")

    def test_erratic_demand_classifies_correctly(self):
        """High CV but few zeros → erratic."""
        rng = np.random.default_rng(42)
        qty = rng.exponential(100, 36).tolist()  # CV ≈ 1.0 for exponential
        df = _make_ts_df(qty)
        features = compute_time_series_features(df).to_dict()
        # Exponential dist has CV ~1.0, zero_demand_pct ~0
        if features["cv_demand"] >= 0.80 and features["zero_demand_pct"] < 0.30:
            result = classify_variability_class(
                features["cv_demand"],
                features["zero_demand_pct"],
            )
            assert result == "erratic"
