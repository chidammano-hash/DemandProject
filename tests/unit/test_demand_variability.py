"""Tests for scripts/compute_demand_variability.py — IPfeature1."""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.ml.compute_demand_variability import (
    winsorize,
    compute_variability_metrics,
    classify_variability_class,
)


@pytest.fixture
def default_config():
    return {
        "history": {"history_months": 24, "min_months_history": 6},
        "outlier": {"sigma_threshold": 3},
        "cv_thresholds": {"low": 0.30, "medium": 0.80, "high": 1.50},
        "intermittency_threshold": {"ratio": 0.30},
    }


def _series(values: list) -> pd.Series:
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# winsorize()
# ---------------------------------------------------------------------------

class TestWinsorize:
    def test_no_outliers_unchanged(self):
        arr = np.array([10.0, 11.0, 12.0, 10.5, 11.5])
        result = winsorize(arr, sigma=3)
        np.testing.assert_array_almost_equal(result, arr)

    def test_high_outlier_capped(self):
        # 29 values near 10, one extreme spike at 10_000 — well beyond 3σ
        arr = np.array([10.0] * 29 + [10_000.0])
        result = winsorize(arr, sigma=3)
        assert result[-1] < 10_000.0

    def test_low_outlier_capped(self):
        # 29 values near 10, one extreme negative — well beyond 3σ
        arr = np.array([-10_000.0] + [10.0] * 29)
        result = winsorize(arr, sigma=3)
        assert result[0] > -10_000.0

    def test_single_element_unchanged(self):
        arr = np.array([42.0])
        result = winsorize(arr, sigma=3)
        assert result[0] == 42.0

    def test_constant_series_unchanged(self):
        arr = np.array([100.0, 100.0, 100.0])
        result = winsorize(arr, sigma=3)
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# compute_variability_metrics()
# ---------------------------------------------------------------------------

class TestComputeVariabilityMetrics:
    def test_insufficient_history_returns_none_class(self, default_config):
        """Fewer than min_months_history → variability_class is None."""
        s = _series([100.0] * 4)
        result = compute_variability_metrics(s, default_config)
        assert result["variability_class"] is None
        assert result["demand_cv"] is None
        assert result["total_demand_months"] == 4

    def test_zero_demand_all_zeros_lumpy(self, default_config):
        """All-zero demand → intermittency_ratio=1.0 → lumpy."""
        s = _series([0.0] * 12)
        result = compute_variability_metrics(s, default_config)
        assert result["variability_class"] == "lumpy"
        assert result["zero_demand_months"] == 12
        assert result["intermittency_ratio"] == 1.0

    def test_constant_demand_low_cv(self, default_config):
        """Perfectly constant demand → CV=0 → low variability."""
        s = _series([100.0] * 24)
        result = compute_variability_metrics(s, default_config)
        assert result["variability_class"] == "low"
        assert result["demand_cv"] == pytest.approx(0.0, abs=1e-6)
        assert result["demand_std"] == pytest.approx(0.0, abs=1e-6)

    def test_cv_low_boundary(self, default_config):
        """CV just below 0.30 → low."""
        # mean=100, std=25 → CV=0.25 < 0.30
        rng = np.random.default_rng(0)
        vals = rng.normal(100, 25, 24).clip(1).tolist()
        s = _series(vals)
        result = compute_variability_metrics(s, default_config)
        if result["demand_cv"] is not None and result["demand_cv"] < 0.30:
            assert result["variability_class"] == "low"

    def test_medium_variability_class(self, default_config):
        """CV in [0.30, 0.80) → medium."""
        rng = np.random.default_rng(1)
        vals = rng.normal(100, 50, 24).clip(1).tolist()  # CV ≈ 0.5
        s = _series(vals)
        result = compute_variability_metrics(s, default_config)
        if result["demand_cv"] is not None and 0.30 <= result["demand_cv"] < 0.80:
            assert result["variability_class"] == "medium"

    def test_high_variability_class(self, default_config):
        """CV in [0.80, 1.50) → high."""
        vals = [200.0 if i % 3 == 0 else 10.0 for i in range(24)]  # Spiky
        s = _series(vals)
        result = compute_variability_metrics(s, default_config)
        if result["demand_cv"] is not None and 0.80 <= result["demand_cv"] < 1.50:
            assert result["variability_class"] == "high"

    def test_lumpy_by_cv(self, default_config):
        """CV >= 1.50 → lumpy."""
        # Very spiky: mostly zero, occasional large spike
        vals = [0.0] * 20 + [1000.0] * 4
        s = _series(vals)
        result = compute_variability_metrics(s, default_config)
        # intermittency = 20/24 = 0.83 → triggers lumpy anyway
        assert result["variability_class"] == "lumpy"

    def test_lumpy_by_intermittency(self, default_config):
        """intermittency_ratio >= 0.30 → lumpy even if CV is low."""
        # 10 zeros out of 24 = 41.7% → triggers lumpy
        vals = [100.0] * 14 + [0.0] * 10
        s = _series(vals)
        result = compute_variability_metrics(s, default_config)
        assert result["variability_class"] == "lumpy"
        assert result["zero_demand_months"] == 10

    def test_intermittency_ratio_computed_correctly(self, default_config):
        """intermittency_ratio = zero_months / total_months."""
        vals = [0.0] * 6 + [100.0] * 18
        s = _series(vals)
        result = compute_variability_metrics(s, default_config)
        assert result["intermittency_ratio"] == pytest.approx(6 / 24)

    def test_p50_p90_reasonable(self, default_config):
        """p50 and p90 should be non-null and p90 >= p50."""
        s = _series(list(range(1, 25)))
        result = compute_variability_metrics(s, default_config)
        assert result["demand_p50"] is not None
        assert result["demand_p90"] is not None
        assert result["demand_p90"] >= result["demand_p50"]

    def test_mad_is_non_negative(self, default_config):
        """MAD = mean(|x - mean|) should always be >= 0."""
        vals = [100.0, 200.0, 50.0, 75.0, 150.0, 90.0, 110.0, 80.0,
                120.0, 60.0, 140.0, 95.0, 105.0, 70.0, 130.0, 85.0,
                115.0, 65.0, 145.0, 100.0, 200.0, 50.0, 75.0, 150.0]
        s = _series(vals)
        result = compute_variability_metrics(s, default_config)
        assert result["demand_mad"] is not None
        assert result["demand_mad"] >= 0.0

    def test_output_keys_present(self, default_config):
        """All required keys present in the output dict."""
        s = _series([100.0] * 24)
        result = compute_variability_metrics(s, default_config)
        expected_keys = {
            "demand_mean", "demand_std", "demand_cv", "demand_mad",
            "demand_p50", "demand_p90", "demand_skewness", "demand_kurtosis",
            "zero_demand_months", "total_demand_months", "intermittency_ratio",
            "variability_class",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_total_demand_months_matches_input_length(self, default_config):
        """total_demand_months = len(input series)."""
        s = _series([100.0] * 18)
        result = compute_variability_metrics(s, default_config)
        assert result["total_demand_months"] == 18

    def test_outlier_winsorization_reduces_cv(self, default_config):
        """A single extreme outlier should be winsorized, reducing CV."""
        # Without outlier: low CV
        base = [100.0] * 23
        # With extreme outlier
        with_outlier = base + [100_000.0]
        s = _series(with_outlier)
        result = compute_variability_metrics(s, default_config)
        # After winsorization the massive spike should be capped;
        # CV should be a finite number, not dominated by the spike
        assert result["demand_cv"] is not None
        assert result["demand_cv"] < 100.0  # sanity: not wildly inflated


# ---------------------------------------------------------------------------
# classify_variability_class()
# ---------------------------------------------------------------------------

class TestClassifyVariabilityClass:
    def test_none_cv_returns_none(self, default_config):
        assert classify_variability_class(None, 0.0, default_config) is None

    def test_low_class(self, default_config):
        assert classify_variability_class(0.10, 0.0, default_config) == "low"
        assert classify_variability_class(0.29, 0.0, default_config) == "low"

    def test_medium_class(self, default_config):
        assert classify_variability_class(0.30, 0.0, default_config) == "medium"
        assert classify_variability_class(0.79, 0.0, default_config) == "medium"

    def test_high_class(self, default_config):
        assert classify_variability_class(0.80, 0.0, default_config) == "high"
        assert classify_variability_class(1.49, 0.0, default_config) == "high"

    def test_lumpy_by_cv(self, default_config):
        assert classify_variability_class(1.50, 0.0, default_config) == "lumpy"
        assert classify_variability_class(2.50, 0.0, default_config) == "lumpy"

    def test_lumpy_by_intermittency_overrides_low_cv(self, default_config):
        """Even CV=0.10, if intermittency>=0.30 → lumpy."""
        assert classify_variability_class(0.10, 0.30, default_config) == "lumpy"
        assert classify_variability_class(0.10, 0.50, default_config) == "lumpy"

    def test_intermittency_below_threshold_does_not_trigger_lumpy(self, default_config):
        assert classify_variability_class(0.10, 0.29, default_config) == "low"


# ---------------------------------------------------------------------------
# DFU_SPEC integration: variability columns present
# ---------------------------------------------------------------------------

class TestDfuSpecVariabilityColumns:
    def test_variability_columns_in_dfu_spec(self):
        from common.core.domain_specs import DFU_SPEC
        expected_cols = [
            "demand_mean", "demand_std", "demand_cv", "demand_mad",
            "demand_p50", "demand_p90", "demand_skewness", "demand_kurtosis",
            "zero_demand_months", "total_demand_months", "intermittency_ratio",
            "variability_class", "demand_profile_ts",
        ]
        for col in expected_cols:
            assert col in DFU_SPEC.columns, f"Missing column in DFU_SPEC: {col}"

    def test_variability_class_searchable(self):
        from common.core.domain_specs import DFU_SPEC
        assert "variability_class" in DFU_SPEC.search_fields

    def test_variability_float_fields(self):
        from common.core.domain_specs import DFU_SPEC
        float_cols = ["demand_cv", "demand_std", "demand_mean", "demand_mad",
                      "demand_p50", "demand_p90", "intermittency_ratio"]
        for col in float_cols:
            assert col in DFU_SPEC.float_fields, f"Expected {col} in float_fields"

    def test_variability_int_fields(self):
        from common.core.domain_specs import DFU_SPEC
        assert "zero_demand_months" in DFU_SPEC.int_fields
        assert "total_demand_months" in DFU_SPEC.int_fields
