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
            rows.append({"dfu_ck": dfu, "startdate": m, "qty": float(dfu * 10 + m.month)})
    return pd.DataFrame(rows)


def _make_dfu_attrs():
    return pd.DataFrame({
        "dfu_ck": [1, 2],
        "dmdunit": ["U1", "U2"],
        "dmdgroup": ["G1", "G1"],
        "loc": ["L1", "L1"],
        "ml_cluster": ["A", "B"],
        "region": ["East", "West"],
        "brand": ["X", "Y"],
        "abc_vol": ["A", "B"],
    })


def _make_item_attrs():
    return pd.DataFrame(columns=["dmdunit"])


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


class TestFeatureCountIncrease:
    def test_total_feature_count_increased(self):
        """Grid should have at least 6 more columns than before the new features."""
        grid = _build_grid()
        new_feature_names = ["is_quarter_end", "is_year_end", "days_in_month",
                             "mom_growth", "demand_accel", "volatility_ratio"]
        for name in new_feature_names:
            assert name in grid.columns, f"Missing new feature: {name}"


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
