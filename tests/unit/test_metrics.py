"""Tests for common/metrics.py — compute_accuracy_metrics."""

import pytest
import pandas as pd
import numpy as np

from common.services.metrics import compute_accuracy_metrics


class TestComputeAccuracyMetrics:
    def test_perfect_forecast(self):
        f = pd.Series([100, 200, 300])
        a = pd.Series([100, 200, 300])
        result = compute_accuracy_metrics(f, a)
        assert result["n_rows"] == 3
        assert result["wape"] == 0
        assert result["bias"] == 0
        assert result["accuracy_pct"] == 100

    def test_over_forecast_50pct(self):
        f = pd.Series([150, 300])
        a = pd.Series([100, 200])
        result = compute_accuracy_metrics(f, a)
        assert result["wape"] == 50.0
        assert result["bias"] == 0.5
        assert result["accuracy_pct"] == 50.0

    def test_under_forecast_50pct(self):
        f = pd.Series([50, 100])
        a = pd.Series([100, 200])
        result = compute_accuracy_metrics(f, a)
        assert result["wape"] == 50.0
        assert result["bias"] == -0.5
        assert result["accuracy_pct"] == 50.0

    def test_all_zeros_actual(self):
        f = pd.Series([100, 200])
        a = pd.Series([0, 0])
        result = compute_accuracy_metrics(f, a)
        assert result["wape"] is None
        assert result["bias"] is None
        assert result["accuracy_pct"] is None

    def test_empty_series(self):
        f = pd.Series([], dtype=float)
        a = pd.Series([], dtype=float)
        result = compute_accuracy_metrics(f, a)
        assert result["n_rows"] == 0
        assert result["wape"] is None
        assert result["bias"] is None
        assert result["accuracy_pct"] is None

    def test_nan_in_data(self):
        f = pd.Series([100, float("nan"), 300])
        a = pd.Series([100, 200, float("nan")])
        result = compute_accuracy_metrics(f, a)
        # Only the first row should survive (both non-NaN)
        assert result["n_rows"] == 1
        assert result["wape"] == 0
        assert result["accuracy_pct"] == 100

    def test_single_row(self):
        f = pd.Series([120])
        a = pd.Series([100])
        result = compute_accuracy_metrics(f, a)
        assert result["n_rows"] == 1
        assert result["wape"] == 20.0
        assert result["bias"] == 0.2
        assert result["accuracy_pct"] == 80.0

    def test_large_numbers(self):
        f = pd.Series([1e12])
        a = pd.Series([1e12])
        result = compute_accuracy_metrics(f, a)
        assert result["wape"] == 0
        assert result["accuracy_pct"] == 100

    def test_mixed_positive_negative_actuals(self):
        """When actuals sum to near zero, abs(total_a) should handle it."""
        f = pd.Series([100, 100])
        a = pd.Series([200, -50])
        result = compute_accuracy_metrics(f, a)
        assert result["n_rows"] == 2
        # total_f=200, total_a=150, abs_error=|100-200|+|100-(-50)|=100+150=250
        # wape = 100*250/150 = 166.67
        assert result["wape"] == pytest.approx(166.67, abs=0.01)

    def test_result_types(self):
        f = pd.Series([120, 180])
        a = pd.Series([100, 200])
        result = compute_accuracy_metrics(f, a)
        assert isinstance(result["wape"], float)
        assert isinstance(result["bias"], float)
        assert isinstance(result["accuracy_pct"], float)
        assert isinstance(result["n_rows"], int)
