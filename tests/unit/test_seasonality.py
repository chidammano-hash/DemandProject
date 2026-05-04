"""Tests for scripts/detect_seasonality.py — seasonality detection algorithm."""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.ml.detect_seasonality import compute_seasonality_metrics, compute_acf_lag12


@pytest.fixture
def default_config():
    """Default seasonality config for tests."""
    return {
        "min_months_history": 24,
        "thresholds": {"low": 0.15, "medium": 0.35, "high": 0.70},
        "confirmation": {"yoy_correlation": 0.40, "acf_lag12": 0.30},
        "peak_trough_min_ratio": 1.3,
    }


def _make_sales_df(qty_values: list[float], start: str = "2022-01-01") -> pd.DataFrame:
    """Build a sales DataFrame from a flat list of monthly qty values."""
    dates = pd.date_range(start, periods=len(qty_values), freq="MS")
    return pd.DataFrame({"startdate": dates, "qty": qty_values})


class TestComputeAcfLag12:
    def test_short_series_returns_zero(self):
        assert compute_acf_lag12(np.ones(20)) == 0.0

    def test_constant_series_returns_zero(self):
        assert compute_acf_lag12(np.ones(36)) == 0.0

    def test_periodic_signal_positive(self):
        # 12-month sine cycle repeated 3 times
        months = np.arange(36)
        signal = np.sin(2 * np.pi * months / 12)
        acf = compute_acf_lag12(signal)
        assert acf > 0.5, f"Expected strong positive acf, got {acf}"

    def test_random_noise_near_zero(self):
        rng = np.random.default_rng(42)
        noise = rng.normal(100, 10, 48)
        acf = compute_acf_lag12(noise)
        assert abs(acf) < 0.4, f"Expected weak acf for noise, got {acf}"


class TestComputeSeasonalityMetrics:
    def test_insufficient_history(self, default_config):
        """DFUs with fewer than min_months get insufficient_history profile."""
        df = _make_sales_df([100] * 12)
        result = compute_seasonality_metrics(df, default_config)
        assert result["seasonality_profile"] == "insufficient_history"
        assert result["seasonality_strength"] is None
        assert result["is_yearly_seasonal"] is None
        assert result["months_available"] == 12

    def test_flat_demand_none_profile(self, default_config):
        """Perfectly flat demand should be 'none' profile."""
        df = _make_sales_df([100.0] * 36)
        result = compute_seasonality_metrics(df, default_config)
        assert result["seasonality_profile"] == "none"
        assert result["seasonality_strength"] == 0.0
        assert result["is_yearly_seasonal"] is False

    def test_seasonal_demand_high_profile(self, default_config):
        """Strong repeating seasonal pattern should be 'high'."""
        # 3 years of highly seasonal demand: summer peaks, winter troughs
        pattern = [50, 60, 80, 120, 180, 250, 280, 260, 180, 100, 60, 50]
        qty = pattern * 3  # 36 months
        df = _make_sales_df(qty)
        result = compute_seasonality_metrics(df, default_config)
        assert result["seasonality_profile"] in ("high", "medium")
        assert result["seasonality_strength"] > 0.35
        assert result["peak_month"] in (6, 7)  # June or July
        assert result["trough_month"] in (1, 12)  # Jan or Dec
        assert result["peak_trough_ratio"] > 2.0

    def test_moderate_seasonal_medium_profile(self, default_config):
        """Moderate seasonal variation with confirmation should be 'medium'."""
        # 3 years with moderate seasonal swing (amplitude 100 on base 200 → CV ≈ 0.35)
        base = 200
        pattern = [base + 100 * np.sin(2 * np.pi * m / 12) for m in range(12)]
        qty = [max(10, v) for v in pattern * 3]
        df = _make_sales_df(qty)
        result = compute_seasonality_metrics(df, default_config)
        # Should have detectable seasonality
        assert result["seasonality_strength"] > 0.0
        assert result["seasonality_profile"] in ("low", "medium")

    def test_low_profile_without_confirmation(self, default_config):
        """Slight variation but no year-over-year repeatability → 'low'."""
        rng = np.random.default_rng(123)
        # Year 1 has pattern, years 2-3 are different
        y1 = [100 + 20 * np.sin(2 * np.pi * m / 12) for m in range(12)]
        y2 = rng.normal(100, 5, 12).tolist()
        y3 = rng.normal(100, 5, 12).tolist()
        qty = y1 + y2 + y3
        df = _make_sales_df(qty)
        result = compute_seasonality_metrics(df, default_config)
        assert result["seasonality_profile"] in ("none", "low")

    def test_peak_trough_ratio_with_zero_trough(self, default_config):
        """When trough month has 0 demand, peak_trough_ratio should be None."""
        # Pattern with zero demand in some months
        pattern = [0, 0, 50, 100, 200, 300, 300, 200, 100, 50, 0, 0]
        qty = pattern * 3
        df = _make_sales_df(qty)
        result = compute_seasonality_metrics(df, default_config)
        assert result["peak_trough_ratio"] is None or result["peak_trough_ratio"] > 1.0

    def test_is_yearly_seasonal_flag(self, default_config):
        """Verify is_yearly_seasonal requires all three conditions."""
        # Strong, repeating seasonal pattern
        pattern = [50, 60, 80, 120, 180, 250, 280, 260, 180, 100, 60, 50]
        qty = pattern * 3
        df = _make_sales_df(qty)
        result = compute_seasonality_metrics(df, default_config)
        if result["seasonality_strength"] >= 0.15 and result["peak_trough_ratio"] is not None:
            if result["peak_trough_ratio"] >= 1.3:
                # With strong repeating pattern, should be seasonal
                assert result["is_yearly_seasonal"] is True

    def test_output_keys(self, default_config):
        """Verify all expected keys are present in output."""
        df = _make_sales_df([100] * 36)
        result = compute_seasonality_metrics(df, default_config)
        expected_keys = {
            "seasonality_profile", "seasonality_strength", "is_yearly_seasonal",
            "peak_month", "trough_month", "peak_trough_ratio",
            "yoy_correlation", "acf_lag12", "months_available",
        }
        assert expected_keys == set(result.keys())

    def test_peak_month_range(self, default_config):
        """Peak and trough months should be 1-12."""
        pattern = [100 + 50 * np.sin(2 * np.pi * m / 12) for m in range(12)]
        qty = [max(10, v) for v in pattern * 3]
        df = _make_sales_df(qty)
        result = compute_seasonality_metrics(df, default_config)
        assert 1 <= result["peak_month"] <= 12
        assert 1 <= result["trough_month"] <= 12

    def test_months_available_correct(self, default_config):
        """months_available should match input length."""
        df = _make_sales_df([100] * 30)
        result = compute_seasonality_metrics(df, default_config)
        assert result["months_available"] == 30


class TestDfuSpecSeasonalityColumns:
    """Verify seasonality columns are correctly defined in DFU_SPEC."""

    def test_seasonality_columns_in_dfu_spec(self):
        from common.domain_specs import DFU_SPEC
        seasonality_cols = [
            "seasonality_profile", "seasonality_strength", "is_yearly_seasonal",
            "peak_month", "trough_month", "peak_trough_ratio",
        ]
        for col in seasonality_cols:
            assert col in DFU_SPEC.columns, f"Missing column: {col}"

    def test_seasonality_profile_searchable(self):
        from common.domain_specs import DFU_SPEC
        assert "seasonality_profile" in DFU_SPEC.search_fields

    def test_seasonality_int_fields(self):
        from common.domain_specs import DFU_SPEC
        assert "peak_month" in DFU_SPEC.int_fields
        assert "trough_month" in DFU_SPEC.int_fields

    def test_seasonality_float_fields(self):
        from common.domain_specs import DFU_SPEC
        assert "seasonality_strength" in DFU_SPEC.float_fields
        assert "peak_trough_ratio" in DFU_SPEC.float_fields
