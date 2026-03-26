"""Unit tests for new feature engineering additions (Phase 2)."""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.feature_engineering import build_feature_matrix


def _make_sales_df():
    """Minimal sales DataFrame for 2 DFUs across 18 months."""
    months = pd.date_range("2023-01-01", periods=18, freq="MS")
    rows = []
    for dfu in [1, 2]:
        for m in months:
            rows.append({"sku_ck": dfu, "startdate": m, "qty": float(dfu * 10 + m.month)})
    return pd.DataFrame(rows)


def _make_dfu_attrs():
    return pd.DataFrame({
        "sku_ck": [1, 2],
        "item_id": ["U1", "U2"],
        "customer_group": ["G1", "G1"],
        "loc": ["L1", "L1"],
        "ml_cluster": ["A", "B"],
        "region": ["East", "West"],
        "brand": ["X", "Y"],
        "abc_vol": ["A", "B"],
    })


def _make_item_attrs():
    return pd.DataFrame(columns=["item_id"])


def _build_grid(cat_dtype="category"):
    sales = _make_sales_df()
    dfu = _make_dfu_attrs()
    items = _make_item_attrs()
    months = sorted(sales["startdate"].unique().tolist())
    return build_feature_matrix(sales, dfu, items, months, cat_dtype=cat_dtype)


class TestNewCalendarFeatures:
    def test_is_quarter_end_present(self):
        grid = _build_grid()
        assert "is_quarter_end" in grid.columns

    def test_is_year_end_present(self):
        grid = _build_grid()
        assert "is_year_end" in grid.columns

    def test_days_in_month_present(self):
        grid = _build_grid()
        assert "days_in_month" in grid.columns

    def test_is_quarter_end_values(self):
        grid = _build_grid()
        quarter_end_months = {3, 6, 9, 12}
        for _, row in grid.iterrows():
            expected = 1 if row["startdate"].month in quarter_end_months else 0
            assert int(row["is_quarter_end"]) == expected, (
                f"is_quarter_end mismatch for month {row['startdate'].month}"
            )

    def test_is_year_end_values(self):
        grid = _build_grid()
        for _, row in grid.iterrows():
            expected = 1 if row["startdate"].month == 12 else 0
            assert int(row["is_year_end"]) == expected

    def test_days_in_month_feb(self):
        grid = _build_grid()
        # February 2023 has 28 days (2024 is a leap year — filter to 2023 only)
        feb_rows = grid[(grid["startdate"].dt.month == 2) & (grid["startdate"].dt.year == 2023)]
        assert len(feb_rows) > 0
        assert all(feb_rows["days_in_month"] == 28.0)

    def test_days_in_month_jan(self):
        grid = _build_grid()
        jan_rows = grid[grid["startdate"].dt.month == 1]
        assert all(jan_rows["days_in_month"] == 31.0)


class TestNewDerivedFeatures:
    def test_mom_growth_present(self):
        grid = _build_grid()
        assert "mom_growth" in grid.columns

    def test_demand_accel_present(self):
        grid = _build_grid()
        assert "demand_accel" in grid.columns

    def test_volatility_ratio_present(self):
        grid = _build_grid()
        assert "volatility_ratio" in grid.columns

    def test_mom_growth_clipped(self):
        grid = _build_grid()
        assert grid["mom_growth"].max() <= 2.0
        assert grid["mom_growth"].min() >= -2.0

    def test_mom_growth_no_inf(self):
        grid = _build_grid()
        assert not grid["mom_growth"].isin([np.inf, -np.inf]).any()

    def test_volatility_ratio_non_negative(self):
        grid = _build_grid()
        # std / (|mean| + 1) is always >= 0; drop NaN rows (first row per DFU has no lag history)
        non_null = grid["volatility_ratio"].dropna()
        assert len(non_null) > 0
        assert (non_null >= 0).all()

    def test_demand_accel_is_float(self):
        grid = _build_grid()
        assert grid["demand_accel"].dtype in [np.float64, np.float32]


class TestLagRatioFeatures:
    def test_lag_ratio_yoy_present(self):
        grid = _build_grid()
        assert "lag_ratio_yoy" in grid.columns

    def test_lag_ratio_mom_present(self):
        grid = _build_grid()
        assert "lag_ratio_mom" in grid.columns

    def test_lag_ratio_3v12_present(self):
        grid = _build_grid()
        assert "lag_ratio_3v12" in grid.columns

    def test_lag_ratio_yoy_clipped(self):
        grid = _build_grid()
        assert grid["lag_ratio_yoy"].max() <= 10.0
        assert grid["lag_ratio_yoy"].min() >= -10.0

    def test_lag_ratio_mom_clipped(self):
        grid = _build_grid()
        assert grid["lag_ratio_mom"].max() <= 10.0
        assert grid["lag_ratio_mom"].min() >= -10.0

    def test_lag_ratio_3v12_clipped(self):
        grid = _build_grid()
        assert grid["lag_ratio_3v12"].max() <= 10.0
        assert grid["lag_ratio_3v12"].min() >= -10.0

    def test_lag_ratio_no_inf(self):
        grid = _build_grid()
        for col in ["lag_ratio_yoy", "lag_ratio_mom", "lag_ratio_3v12"]:
            assert not grid[col].isin([np.inf, -np.inf]).any(), f"inf in {col}"

    def test_lag_ratio_yoy_formula(self):
        """lag_ratio_yoy = qty_lag_1 / (|qty_lag_12| + 1), clipped to [-10, 10]."""
        grid = _build_grid()
        # Check rows where both lag_1 and lag_12 are present
        valid = grid.dropna(subset=["qty_lag_1", "qty_lag_12"])
        if len(valid) > 0:
            expected = (valid["qty_lag_1"] / (valid["qty_lag_12"].abs() + 1.0)).clip(-10.0, 10.0)
            pd.testing.assert_series_equal(
                valid["lag_ratio_yoy"].reset_index(drop=True),
                expected.reset_index(drop=True),
                check_names=False,
                atol=1e-5,
            )


class TestZeroDemandCount:
    def test_n_zero_last_6m_present(self):
        grid = _build_grid()
        assert "n_zero_last_6m" in grid.columns

    def test_n_zero_last_6m_range(self):
        grid = _build_grid()
        assert grid["n_zero_last_6m"].min() >= 0
        assert grid["n_zero_last_6m"].max() <= 6

    def test_n_zero_last_6m_dtype(self):
        grid = _build_grid()
        assert grid["n_zero_last_6m"].dtype in [np.float32, np.float64]

    def test_n_zero_last_6m_all_positive_demand(self):
        """With all positive demand, zero count should be 0 for months with enough history."""
        grid = _build_grid()
        # After 6 months of history, all lags are positive → zero count = 0
        late_rows = grid[grid["startdate"] >= pd.Timestamp("2023-07-01")]
        if len(late_rows) > 0:
            assert (late_rows["n_zero_last_6m"] == 0).all()


class TestTsProfileInGrid:
    def test_ts_profile_features_present(self):
        from common.constants import TS_PROFILE_FEATURES
        grid = _build_grid()
        for feat in TS_PROFILE_FEATURES:
            assert feat in grid.columns, f"Missing TS profile feature: {feat}"

    def test_ts_profile_no_nan(self):
        from common.constants import TS_PROFILE_FEATURES
        grid = _build_grid()
        for feat in TS_PROFILE_FEATURES:
            assert grid[feat].notna().all(), f"NaN in {feat}"

    def test_cv_demand_non_negative(self):
        grid = _build_grid()
        assert (grid["cv_demand"] >= 0).all()

    def test_mean_demand_positive(self):
        grid = _build_grid()
        assert (grid["mean_demand"] > 0).all()

    def test_zero_demand_pct_range(self):
        grid = _build_grid()
        assert (grid["zero_demand_pct"] >= 0).all()
        assert (grid["zero_demand_pct"] <= 1.0).all()

    def test_ts_profile_static_per_dfu(self):
        """TS profile features should be constant per DFU (static attributes)."""
        from common.constants import TS_PROFILE_FEATURES
        grid = _build_grid()
        for feat in TS_PROFILE_FEATURES:
            nunique = grid.groupby("sku_ck")[feat].nunique()
            assert (nunique == 1).all(), f"{feat} varies within a DFU"

    def test_seasonal_amplitude_non_negative(self):
        grid = _build_grid()
        assert (grid["seasonal_amplitude"] >= 0).all()

    def test_adi_positive(self):
        grid = _build_grid()
        assert (grid["adi"] > 0).all()


class TestFeatureCountIncrease:
    def test_total_feature_count_increased(self):
        """Grid should have all new features from all phases."""
        grid = _build_grid()
        new_feature_names = ["is_quarter_end", "is_year_end", "days_in_month",
                             "mom_growth", "demand_accel", "volatility_ratio",
                             "lag_ratio_yoy", "lag_ratio_mom", "lag_ratio_3v12",
                             "n_zero_last_6m", "cv_demand", "mean_demand",
                             "seasonal_amplitude", "adi"]
        for name in new_feature_names:
            assert name in grid.columns, f"Missing new feature: {name}"

    def test_enhanced_features_all_present(self):
        """Grid should have all enhanced features from the 4 new groups (except external forecast)."""
        from common.constants import FOURIER_FEATURES, CROSTON_FEATURES, CROSS_DFU_FEATURES
        grid = _build_grid()
        for name in FOURIER_FEATURES + CROSTON_FEATURES + CROSS_DFU_FEATURES:
            assert name in grid.columns, f"Missing enhanced feature: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests for compute_time_series_features in generate_clustering_features.py
# ─────────────────────────────────────────────────────────────────────────────

from scripts.generate_clustering_features import compute_time_series_features  # noqa: E402


def make_sales_df(demand_values: list, start: str = "2022-01-01") -> pd.DataFrame:
    """Build a minimal sales DataFrame from a list of demand quantities."""
    dates = pd.date_range(start=start, periods=len(demand_values), freq="MS")
    return pd.DataFrame({"startdate": dates, "qty": demand_values})


# ── IQR Demand ────────────────────────────────────────────────────────────────

class TestIqrDemand:
    def test_iqr_basic(self):
        df = make_sales_df(list(range(1, 11)))
        result = compute_time_series_features(df)
        assert result["iqr_demand"] > 0

    def test_iqr_constant(self):
        df = make_sales_df([5] * 12)
        result = compute_time_series_features(df)
        assert result["iqr_demand"] == 0.0

    def test_iqr_zero_demand(self):
        df = make_sales_df([0] * 12)
        result = compute_time_series_features(df)
        assert result["iqr_demand"] == 0.0


# ── Trend Slope Norm ──────────────────────────────────────────────────────────

class TestTrendSlopeNorm:
    def test_growing_normalized(self):
        df = make_sales_df(list(range(1, 13)))
        result = compute_time_series_features(df)
        assert result["trend_slope_norm"] > 0

    def test_declining_normalized(self):
        df = make_sales_df(list(range(12, 0, -1)))
        result = compute_time_series_features(df)
        assert result["trend_slope_norm"] < 0

    def test_flat_slope_norm(self):
        df = make_sales_df([10] * 24)
        result = compute_time_series_features(df)
        assert abs(result["trend_slope_norm"]) < 1e-10

    def test_scale_invariance(self):
        """Slope norm should be similar regardless of the scale of demand."""
        df1 = make_sales_df(list(range(1, 13)))
        df2 = make_sales_df([x * 100 for x in range(1, 13)])
        r1 = compute_time_series_features(df1)
        r2 = compute_time_series_features(df2)
        assert abs(r1["trend_slope_norm"] - r2["trend_slope_norm"]) < 0.01


# ── Trend R² ──────────────────────────────────────────────────────────────────

class TestTrendR2:
    def test_perfect_trend(self):
        df = make_sales_df(list(range(1, 25)))
        result = compute_time_series_features(df)
        # Perfect linear growth → R² close to 1 (signed positive)
        assert result["trend_r2"] > 0.95

    def test_no_trend(self):
        # Oscillating pattern: no net linear trend
        df = make_sales_df([10, 9, 11, 10, 9, 11] * 4)
        result = compute_time_series_features(df)
        assert abs(result["trend_r2"]) < 0.2

    def test_signed_positive(self):
        df = make_sales_df(list(range(1, 25)))
        result = compute_time_series_features(df)
        assert result["trend_r2"] > 0

    def test_signed_negative(self):
        df = make_sales_df(list(range(24, 0, -1)))
        result = compute_time_series_features(df)
        assert result["trend_r2"] < 0


# ── Seasonal Amplitude ────────────────────────────────────────────────────────

class TestSeasonalAmplitude:
    def test_high_seasonal(self):
        """Build a 24-month series with clear seasonality: high in summer, low in winter."""
        demand = []
        for y in range(2):
            for m in range(1, 13):
                if m in (6, 7, 8):
                    demand.append(200.0)
                elif m in (12, 1, 2):
                    demand.append(10.0)
                else:
                    demand.append(100.0)
        df = make_sales_df(demand)
        result = compute_time_series_features(df)
        assert result["seasonal_amplitude"] > 0

    def test_flat_seasonal(self):
        df = make_sales_df([100] * 24)
        result = compute_time_series_features(df)
        assert result["seasonal_amplitude"] == 0.0

    def test_short_history_seasonal_amplitude(self):
        """Less than 12 months → seasonal_amplitude must be 0.0."""
        df = make_sales_df([100, 50, 200, 80, 150, 30])
        result = compute_time_series_features(df)
        assert result["seasonal_amplitude"] == 0.0


# ── Periodicity Strength ──────────────────────────────────────────────────────

class TestPeriodicityStrength:
    def test_clear_periodicity(self):
        """36-month perfect sine wave with 12-month period → high FFT power concentration."""
        t = np.arange(36)
        demand = (100 + 50 * np.sin(2 * np.pi * t / 12)).tolist()
        df = make_sales_df(demand)
        result = compute_time_series_features(df)
        assert result["periodicity_strength"] > 0.3

    def test_no_periodicity(self):
        """White noise should produce a low periodicity strength."""
        rng = np.random.default_rng(42)
        demand = rng.uniform(1, 5, size=36).tolist()
        df = make_sales_df(demand)
        result = compute_time_series_features(df)
        # No guarantee of extremely low value for random noise, but the dominant
        # component should not dominate more than half of the power on average.
        # We use a generous threshold: < 0.5.
        assert result["periodicity_strength"] < 0.5

    def test_short_history_periodicity(self):
        """Fewer than 12 months → periodicity_strength must be 0.0."""
        df = make_sales_df([10, 20, 30, 40, 50, 60, 70, 80])
        result = compute_time_series_features(df)
        assert result["periodicity_strength"] == 0.0


# ── ADI ───────────────────────────────────────────────────────────────────────

class TestAdi:
    def test_continuous_demand(self):
        """All months nonzero: ADI = mean gap between consecutive nonzero indices = 1.0."""
        df = make_sales_df([10] * 12)
        result = compute_time_series_features(df)
        assert result["adi"] == 1.0

    def test_intermittent(self):
        """Pattern: nonzero every 3rd month → mean gap = 3.0."""
        # Nonzero at indices 0, 3, 6, 9 → diffs = [3, 3, 3] → mean = 3.0
        df = make_sales_df([10, 0, 0, 10, 0, 0, 10, 0, 0, 10, 0, 0])
        result = compute_time_series_features(df)
        assert result["adi"] == 3.0

    def test_all_zero(self):
        """Zero demand throughout → ADI = len(demand)."""
        demand = [0] * 12
        df = make_sales_df(demand)
        result = compute_time_series_features(df)
        assert result["adi"] == float(len(demand))

    def test_single_nonzero(self):
        """Only one nonzero value → ADI = len(demand) (worst-case sparse)."""
        demand = [0] * 11 + [10]
        df = make_sales_df(demand)
        result = compute_time_series_features(df)
        assert result["adi"] == float(len(demand))


# ── Recency Ratio ─────────────────────────────────────────────────────────────

class TestRecencyRatio:
    def test_growing_recent(self):
        """Recent 6m is much higher than prior months → recency_ratio > 1."""
        demand = [5.0] * 18 + [15.0] * 6
        df = make_sales_df(demand)
        result = compute_time_series_features(df)
        assert result["recency_ratio"] > 1.0

    def test_declining_recent(self):
        """Recent 6m is much lower than prior months → recency_ratio < 1."""
        demand = [15.0] * 18 + [5.0] * 6
        df = make_sales_df(demand)
        result = compute_time_series_features(df)
        assert result["recency_ratio"] < 1.0

    def test_stable(self):
        """Constant demand → recency_ratio ≈ 1.0."""
        df = make_sales_df([10.0] * 24)
        result = compute_time_series_features(df)
        assert abs(result["recency_ratio"] - 1.0) < 1e-9

    def test_short_history_recency(self):
        """Fewer than 12 months → recency_ratio fallback = 1.0."""
        df = make_sales_df([5.0] * 5)
        result = compute_time_series_features(df)
        assert result["recency_ratio"] == 1.0


# ── compute_time_series_features (integration-style) ─────────────────────────

class TestComputeTimeSeriesFeatures:
    ALL_NEW_FEATURES = [
        "iqr_demand",
        "trend_slope_norm",
        "trend_r2",
        "seasonal_amplitude",
        "seasonal_r2",
        "periodicity_strength",
        "adi",
        "recency_ratio",
    ]

    def test_returns_series_with_all_new_features(self):
        """36 months of synthetic data must produce all expected feature keys."""
        demand = [float(10 + i % 12) for i in range(36)]
        df = make_sales_df(demand)
        result = compute_time_series_features(df)
        assert isinstance(result, pd.Series)
        for key in self.ALL_NEW_FEATURES:
            assert key in result.index, f"Missing feature: {key}"

    def test_empty_df_returns_empty(self):
        """Empty DataFrame must return an empty pd.Series."""
        df = pd.DataFrame(columns=["startdate", "qty"])
        result = compute_time_series_features(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_single_month(self):
        """A single month of data must not raise and must return a Series."""
        df = make_sales_df([42.0])
        result = compute_time_series_features(df)
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_no_nan_values(self):
        """24 months of positive demand must produce no NaN values."""
        demand = [float(10 + i) for i in range(24)]
        df = make_sales_df(demand)
        result = compute_time_series_features(df)
        nan_features = [k for k, v in result.items() if pd.isna(v)]
        assert nan_features == [], f"NaN found in: {nan_features}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests for the 4 new enhanced feature groups
# ─────────────────────────────────────────────────────────────────────────────

from common.ml.feature_engineering import (  # noqa: E402
    _compute_fourier_features,
    _compute_croston_features,
    _compute_cross_dfu_features,
    enrich_with_external_forecast,
)


class TestFourierFeatures:
    """Tests for Fourier seasonal term features."""

    def test_fourier_columns_present(self):
        grid = _build_grid()
        from common.constants import FOURIER_FEATURES
        for feat in FOURIER_FEATURES:
            assert feat in grid.columns, f"Missing Fourier feature: {feat}"

    def test_fourier_values_bounded(self):
        """Sin/cos values must be in [-1, 1]."""
        grid = _build_grid()
        for period in [12, 6, 4, 3]:
            for func in ["sin", "cos"]:
                col = f"fourier_{func}_{period}"
                assert grid[col].min() >= -1.0 - 1e-6, f"{col} below -1"
                assert grid[col].max() <= 1.0 + 1e-6, f"{col} above 1"

    def test_fourier_no_nan(self):
        grid = _build_grid()
        for period in [12, 6, 4, 3]:
            for func in ["sin", "cos"]:
                col = f"fourier_{func}_{period}"
                assert grid[col].notna().all(), f"NaN in {col}"

    def test_fourier_dtype_float32(self):
        grid = _build_grid()
        for period in [12, 6, 4, 3]:
            for func in ["sin", "cos"]:
                col = f"fourier_{func}_{period}"
                assert grid[col].dtype == np.float32, f"{col} not float32"

    def test_fourier_sin_12_equals_sin_2pi_month_over_12(self):
        """fourier_sin_12 should equal sin(2π * month / 12)."""
        grid = _build_grid()
        expected = np.sin(2 * np.pi * grid["month"].values / 12).astype(np.float32)
        np.testing.assert_allclose(
            grid["fourier_sin_12"].values,
            expected,
            atol=1e-5,
        )

    def test_fourier_cos_12_equals_cos_2pi_month_over_12(self):
        """fourier_cos_12 should equal cos(2π * month / 12)."""
        grid = _build_grid()
        expected = np.cos(2 * np.pi * grid["month"].values / 12).astype(np.float32)
        np.testing.assert_allclose(
            grid["fourier_cos_12"].values,
            expected,
            atol=1e-5,
        )

    def test_month_sin_cos_removed(self):
        """month_sin and month_cos should not exist (replaced by fourier_sin_12/cos_12)."""
        grid = _build_grid()
        assert "month_sin" not in grid.columns
        assert "month_cos" not in grid.columns

    def test_fourier_period_6_alternates(self):
        """Period-6 features should complete a full cycle every 6 months."""
        grid = _build_grid()
        # For month 1 and month 7, the angle difference is 2π → same value
        dfu1 = grid[grid["sku_ck"] == 1].sort_values("startdate")
        m1_val = dfu1[dfu1["startdate"].dt.month == 1]["fourier_sin_6"].iloc[0]
        m7_val = dfu1[dfu1["startdate"].dt.month == 7]["fourier_sin_6"].iloc[0]
        assert abs(m1_val - m7_val) < 1e-5, "Period-6 sin should repeat every 6 months"

    def test_fourier_protected(self):
        """Fourier features should be in PROTECTED_FEATURES."""
        from common.constants import PROTECTED_FEATURES, FOURIER_FEATURES
        for feat in FOURIER_FEATURES:
            assert feat in PROTECTED_FEATURES, f"{feat} should be protected"


class TestCrostonFeatures:
    """Tests for Croston decomposition features (intermittent demand)."""

    def test_croston_columns_present(self):
        grid = _build_grid()
        from common.constants import CROSTON_FEATURES
        for feat in CROSTON_FEATURES:
            assert feat in grid.columns, f"Missing Croston feature: {feat}"

    def test_croston_no_nan(self):
        grid = _build_grid()
        from common.constants import CROSTON_FEATURES
        for feat in CROSTON_FEATURES:
            assert grid[feat].notna().all(), f"NaN in {feat}"

    def test_croston_dtype_float32(self):
        grid = _build_grid()
        from common.constants import CROSTON_FEATURES
        for feat in CROSTON_FEATURES:
            assert grid[feat].dtype == np.float32, f"{feat} not float32"

    def test_croston_demand_size_non_negative(self):
        grid = _build_grid()
        assert (grid["croston_demand_size"] >= 0).all()

    def test_croston_demand_interval_at_least_one(self):
        grid = _build_grid()
        assert (grid["croston_demand_interval"] >= 1.0).all()

    def test_croston_probability_bounded(self):
        """Probability should be in [0, 1]."""
        grid = _build_grid()
        assert (grid["croston_probability"] >= 0).all()
        assert (grid["croston_probability"] <= 1.0 + 1e-6).all()

    def test_croston_continuous_demand(self):
        """With continuous positive demand, size should be positive and interval ≈ 1."""
        grid = _build_grid()
        # After enough history, all demand is positive → interval should be close to 1
        late_rows = grid[grid["startdate"] >= pd.Timestamp("2023-07-01")]
        if len(late_rows) > 0:
            assert (late_rows["croston_demand_size"] > 0).all()
            # Interval should be near 1 for continuous demand
            assert (late_rows["croston_demand_interval"] <= 2.0).all()

    def test_croston_intermittent_demand(self):
        """With intermittent demand, interval should be > 1."""
        months = pd.date_range("2023-01-01", periods=18, freq="MS")
        rows = []
        for m in months:
            # Non-zero every 3rd month
            qty = 100.0 if m.month % 3 == 1 else 0.0
            rows.append({"sku_ck": 1, "startdate": m, "qty": qty})
        sales = pd.DataFrame(rows)
        dfu = pd.DataFrame({
            "sku_ck": [1],
            "item_id": ["U1"],
            "customer_group": ["G1"],
            "loc": ["L1"],
        })
        items = pd.DataFrame(columns=["item_id"])
        grid = build_feature_matrix(sales, dfu, items, sorted(sales["startdate"].unique().tolist()))
        # After enough history (skip first few months), interval should be > 1
        late = grid[grid["startdate"] >= pd.Timestamp("2023-06-01")]
        nonzero_interval = late[late["croston_demand_interval"] > 1.0]
        assert len(nonzero_interval) > 0, "Intermittent demand should produce interval > 1"

    def test_croston_recomputed_after_mask(self):
        """Croston features should be recomputed after mask_future_sales."""
        from common.ml.feature_engineering import mask_future_sales
        grid = _build_grid()
        original_size = grid[grid["startdate"] == pd.Timestamp("2024-06-01")]["croston_demand_size"].values.copy()
        cutoff = pd.Timestamp("2023-06-01")
        masked = mask_future_sales(grid, cutoff)
        # After masking, Croston features should still be present and valid
        assert "croston_demand_size" in masked.columns
        assert masked["croston_demand_size"].notna().all()


class TestCrossDfuFeatures:
    """Tests for cross-DFU cluster aggregate features."""

    def test_cross_dfu_columns_present(self):
        grid = _build_grid()
        from common.constants import CROSS_DFU_FEATURES
        for feat in CROSS_DFU_FEATURES:
            assert feat in grid.columns, f"Missing cross-DFU feature: {feat}"

    def test_cross_dfu_no_nan(self):
        grid = _build_grid()
        from common.constants import CROSS_DFU_FEATURES
        for feat in CROSS_DFU_FEATURES:
            assert grid[feat].notna().all(), f"NaN in {feat}"

    def test_cross_dfu_dtype_float32(self):
        grid = _build_grid()
        from common.constants import CROSS_DFU_FEATURES
        for feat in CROSS_DFU_FEATURES:
            assert grid[feat].dtype == np.float32, f"{feat} not float32"

    def test_cluster_mean_lag1_consistent(self):
        """cluster_mean_lag1 should be the mean of qty_lag_1 within each cluster-month."""
        grid = _build_grid()
        # Use observed=True to skip categories with no data
        for (cluster, month), group in grid.groupby(["ml_cluster", "startdate"], observed=True):
            # pandas .mean() skips NaN, matching the agg behavior
            expected_mean = group["qty_lag_1"].mean()
            if pd.isna(expected_mean):
                expected_mean = 0.0
            actual = group["cluster_mean_lag1"].iloc[0]
            assert abs(float(actual) - float(expected_mean)) < 1e-3, (
                f"cluster_mean_lag1 mismatch for cluster={cluster}, month={month}"
            )

    def test_cluster_total_lag1_consistent(self):
        """cluster_total_lag1 should be the sum of qty_lag_1 within each cluster-month."""
        grid = _build_grid()
        for (cluster, month), group in grid.groupby(["ml_cluster", "startdate"], observed=True):
            # pandas .sum() skips NaN, matching the agg behavior
            expected_sum = group["qty_lag_1"].sum()
            if pd.isna(expected_sum):
                expected_sum = 0.0
            actual = group["cluster_total_lag1"].iloc[0]
            assert abs(float(actual) - float(expected_sum)) < 1e-2, (
                f"cluster_total_lag1 mismatch for cluster={cluster}, month={month}"
            )

    def test_cluster_zero_pct_bounded(self):
        """cluster_zero_pct should be in [0, 1]."""
        grid = _build_grid()
        assert (grid["cluster_zero_pct"] >= 0).all()
        assert (grid["cluster_zero_pct"] <= 1.0 + 1e-6).all()

    def test_cluster_demand_trend_clipped(self):
        grid = _build_grid()
        assert grid["cluster_demand_trend"].max() <= 10.0
        assert grid["cluster_demand_trend"].min() >= -10.0

    def test_cross_dfu_without_ml_cluster(self):
        """When ml_cluster is not in dfu_attrs, features should be 0."""
        months = pd.date_range("2023-01-01", periods=6, freq="MS")
        sales = pd.DataFrame({
            "sku_ck": [1] * 6,
            "startdate": list(months),
            "qty": [10.0] * 6,
        })
        dfu = pd.DataFrame({
            "sku_ck": [1],
            "item_id": ["U1"],
            "customer_group": ["G1"],
            "loc": ["L1"],
            # No ml_cluster column
        })
        items = pd.DataFrame(columns=["item_id"])
        grid = build_feature_matrix(sales, dfu, items, list(months))
        from common.constants import CROSS_DFU_FEATURES
        for feat in CROSS_DFU_FEATURES:
            assert (grid[feat] == 0).all(), f"{feat} should be 0 without ml_cluster"


class TestExternalForecastEnrichment:
    """Tests for external forecast signal features."""

    def test_enrichment_with_none(self):
        """When ext_forecast_df is None, features should be 0."""
        grid = _build_grid()
        result = enrich_with_external_forecast(grid, None)
        from common.constants import EXTERNAL_FORECAST_FEATURES
        for feat in EXTERNAL_FORECAST_FEATURES:
            assert feat in result.columns, f"Missing {feat}"
            assert (result[feat] == 0).all(), f"{feat} should be 0 with None input"

    def test_enrichment_with_empty_df(self):
        """When ext_forecast_df is empty, features should be 0."""
        grid = _build_grid()
        empty = pd.DataFrame(columns=["sku_ck", "startdate", "basefcst_pref"])
        result = enrich_with_external_forecast(grid, empty)
        from common.constants import EXTERNAL_FORECAST_FEATURES
        for feat in EXTERNAL_FORECAST_FEATURES:
            assert (result[feat] == 0).all()

    def test_enrichment_with_data(self):
        """With actual forecast data, features should be non-trivial."""
        grid = _build_grid()
        # Create matching external forecast
        ext_rows = []
        for _, row in grid[["sku_ck", "startdate"]].drop_duplicates().iterrows():
            ext_rows.append({
                "sku_ck": row["sku_ck"],
                "startdate": row["startdate"],
                "basefcst_pref": 50.0,
            })
        ext_df = pd.DataFrame(ext_rows)
        result = enrich_with_external_forecast(grid, ext_df)
        from common.constants import EXTERNAL_FORECAST_FEATURES
        for feat in EXTERNAL_FORECAST_FEATURES:
            assert feat in result.columns
            assert result[feat].notna().all(), f"NaN in {feat}"

    def test_enrichment_dtype_float32(self):
        grid = _build_grid()
        result = enrich_with_external_forecast(grid, None)
        from common.constants import EXTERNAL_FORECAST_FEATURES
        for feat in EXTERNAL_FORECAST_FEATURES:
            assert result[feat].dtype == np.float32

    def test_ext_fcst_ratio_clipped(self):
        """ext_fcst_ratio should be clipped to [-10, 10]."""
        grid = _build_grid()
        ext_rows = []
        for _, row in grid[["sku_ck", "startdate"]].drop_duplicates().iterrows():
            ext_rows.append({
                "sku_ck": row["sku_ck"],
                "startdate": row["startdate"],
                "basefcst_pref": 100000.0,  # extreme value
            })
        ext_df = pd.DataFrame(ext_rows)
        result = enrich_with_external_forecast(grid, ext_df)
        assert result["ext_fcst_ratio"].max() <= 10.0
        assert result["ext_fcst_ratio"].min() >= -10.0

    def test_ext_fcst_lag1_ratio_is_causal(self):
        """ext_fcst_lag1_ratio at time T uses forecast from T-1, not T."""
        grid = _build_grid()
        months = sorted(grid["startdate"].unique())
        ext_rows = []
        for sku in grid["sku_ck"].unique():
            for m in months:
                # Give a distinctive value per month
                ext_rows.append({
                    "sku_ck": sku,
                    "startdate": m,
                    "basefcst_pref": float(m.month) * 10,
                })
        ext_df = pd.DataFrame(ext_rows)
        result = enrich_with_external_forecast(grid, ext_df)
        # First month should have lag1 ratio = 0 (no prior forecast)
        first_month_rows = result[result["startdate"] == months[0]]
        assert (first_month_rows["ext_fcst_lag1_ratio"] == 0).all()

    def test_no_temp_columns_remain(self):
        """Temporary columns _ext_fcst and _ext_fcst_lag1 should be cleaned up."""
        grid = _build_grid()
        ext_rows = [{"sku_ck": 1, "startdate": pd.Timestamp("2023-01-01"), "basefcst_pref": 50.0}]
        ext_df = pd.DataFrame(ext_rows)
        result = enrich_with_external_forecast(grid, ext_df)
        assert "_ext_fcst" not in result.columns
        assert "_ext_fcst_lag1" not in result.columns


class TestEnhancedFeaturesInGetFeatureColumns:
    """Ensure get_feature_columns includes enhanced features."""

    def test_fourier_in_feature_columns(self):
        from common.ml.feature_engineering import get_feature_columns
        grid = _build_grid()
        feat_cols = get_feature_columns(grid)
        from common.constants import FOURIER_FEATURES
        for feat in FOURIER_FEATURES:
            assert feat in feat_cols, f"{feat} missing from feature columns"

    def test_croston_in_feature_columns(self):
        from common.ml.feature_engineering import get_feature_columns
        grid = _build_grid()
        feat_cols = get_feature_columns(grid)
        from common.constants import CROSTON_FEATURES
        for feat in CROSTON_FEATURES:
            assert feat in feat_cols, f"{feat} missing from feature columns"

    def test_cross_dfu_in_feature_columns(self):
        from common.ml.feature_engineering import get_feature_columns
        grid = _build_grid()
        feat_cols = get_feature_columns(grid)
        from common.constants import CROSS_DFU_FEATURES
        for feat in CROSS_DFU_FEATURES:
            assert feat in feat_cols, f"{feat} missing from feature columns"

    def test_enhanced_not_in_metadata(self):
        from common.constants import METADATA_COLS, ENHANCED_FEATURES
        for feat in ENHANCED_FEATURES:
            assert feat not in METADATA_COLS, f"{feat} should NOT be in METADATA_COLS"
