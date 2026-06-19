"""Tests for common/metrics.py — compute_accuracy_metrics."""

import pandas as pd
import pytest

from common.services.metrics import (
    compute_accuracy_metrics,
    compute_unweighted_accuracy,
)


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


class TestComputeUnweightedAccuracy:
    """Per-DFU WAPE then mean/median across DFUs (every DFU weighted equally)."""

    def test_basic_mean_and_median(self):
        # per-DFU accuracies: 80, 50, 90  ->  mean 73.33, median 80
        per_dfu = [(100.0, 20.0), (100.0, 50.0), (100.0, 10.0)]
        result = compute_unweighted_accuracy(per_dfu)
        assert result["n_dfus"] == 3
        assert result["n_undefined"] == 0
        assert result["mean_accuracy_pct"] == pytest.approx(73.3333, abs=1e-3)
        assert result["median_accuracy_pct"] == pytest.approx(80.0, abs=1e-3)

    def test_zero_actual_dfu_excluded_not_counted_as_zero(self):
        # The zero-actual DFU is undefined; it must NOT drag the mean to 0.
        per_dfu = [(100.0, 20.0), (0.0, 5.0)]
        result = compute_unweighted_accuracy(per_dfu)
        assert result["n_dfus"] == 2
        assert result["n_undefined"] == 1
        assert result["mean_accuracy_pct"] == pytest.approx(80.0, abs=1e-3)
        assert result["median_accuracy_pct"] == pytest.approx(80.0, abs=1e-3)

    def test_per_dfu_accuracy_clamped_at_zero(self):
        # abs_error > actual would give negative accuracy; clamp to 0 (matches
        # compute_accuracy). One DFU at 0, one at 100 -> mean 50.
        per_dfu = [(100.0, 250.0), (100.0, 0.0)]
        result = compute_unweighted_accuracy(per_dfu)
        assert result["mean_accuracy_pct"] == pytest.approx(50.0, abs=1e-3)

    def test_all_undefined_returns_none(self):
        result = compute_unweighted_accuracy([(0.0, 0.0), (0.0, 3.0)])
        assert result["n_dfus"] == 2
        assert result["n_undefined"] == 2
        assert result["mean_accuracy_pct"] is None
        assert result["median_accuracy_pct"] is None

    def test_empty_input(self):
        result = compute_unweighted_accuracy([])
        assert result["n_dfus"] == 0
        assert result["n_undefined"] == 0
        assert result["mean_accuracy_pct"] is None
        assert result["median_accuracy_pct"] is None
