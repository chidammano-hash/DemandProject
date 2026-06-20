"""Tests for common/metrics.py — compute_accuracy_metrics."""

import pandas as pd
import pytest

from common.services.metrics import (
    compute_accuracy_metrics,
    compute_mase,
    compute_unweighted_accuracy,
    compute_unweighted_mase,
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


class TestComputeMase:
    """MASE = eval MAE / in-sample seasonal-naive MAE (scale-free, naive-relative)."""

    def test_known_value_m1(self):
        # insample diffs: |12-10|, |11-12|, |13-11| = 2, 1, 2 -> q = 5/3
        # eval mae = mean(|12-10|, |8-10|) = mean(2, 2) = 2
        # MASE = 2 / (5/3) = 1.2
        insample = [10.0, 12.0, 11.0, 13.0]
        actuals = [10.0, 10.0]
        forecasts = [12.0, 8.0]
        result = compute_mase(actuals, forecasts, insample, seasonal_period=1)
        assert result == pytest.approx(1.2)

    def test_known_value_seasonal_m12(self):
        # 13 in-sample points; with m=12 only one seasonal diff is available:
        # t=12 -> |insample[12] - insample[0]| = |30 - 20| = 10 -> q = 10
        # eval mae = mean(|55-50|, |40-50|) = mean(5, 10) = 7.5
        # MASE = 7.5 / 10 = 0.75
        insample = [20.0] + [0.0] * 11 + [30.0]
        actuals = [50.0, 50.0]
        forecasts = [55.0, 40.0]
        result = compute_mase(actuals, forecasts, insample, seasonal_period=12)
        assert result == pytest.approx(0.75)

    def test_perfect_forecast_is_zero(self):
        insample = [10.0, 12.0, 11.0, 13.0]
        actuals = [10.0, 20.0, 30.0]
        forecasts = [10.0, 20.0, 30.0]
        assert compute_mase(actuals, forecasts, insample) == 0.0

    def test_flat_insample_returns_none(self):
        # all-equal history -> every naive diff is 0 -> q == 0 -> undefined
        insample = [5.0, 5.0, 5.0, 5.0]
        assert compute_mase([10.0], [12.0], insample) is None

    def test_all_zero_insample_returns_none(self):
        insample = [0.0, 0.0, 0.0, 0.0]
        assert compute_mase([10.0], [12.0], insample) is None

    def test_insample_too_short_for_seasonal_period(self):
        # len(insample) <= seasonal_period -> cannot form one seasonal diff
        assert compute_mase([10.0], [12.0], [3.0], seasonal_period=1) is None
        # exactly equal length is also undefined
        assert compute_mase([10.0], [12.0], [3.0, 4.0], seasonal_period=2) is None

    def test_empty_eval_returns_none(self):
        insample = [10.0, 12.0, 11.0, 13.0]
        assert compute_mase([], [], insample) is None

    def test_length_mismatch_raises(self):
        insample = [10.0, 12.0, 11.0, 13.0]
        with pytest.raises(ValueError, match="Length mismatch"):
            compute_mase([10.0, 20.0], [12.0], insample)

    def test_seasonal_period_below_one_raises(self):
        insample = [10.0, 12.0, 11.0, 13.0]
        with pytest.raises(ValueError, match="seasonal_period"):
            compute_mase([10.0], [12.0], insample, seasonal_period=0)

    def test_scale_depends_only_on_insample(self):
        # Leakage guard: with identical insample the denominator (scale) must be
        # the same regardless of eval window, so MASE1/MASE2 == mae1/mae2.
        insample = [10.0, 12.0, 11.0, 13.0]
        # eval 1: mae1 = mean(|12-10|, |8-10|) = 2
        mase1 = compute_mase([10.0, 10.0], [12.0, 8.0], insample)
        # eval 2: mae2 = mean(|100-50|, |50-50|, |70-50|) = mean(50, 0, 20) = 70/3
        mase2 = compute_mase([50.0, 50.0, 50.0], [100.0, 50.0, 70.0], insample)
        mae1 = 2.0
        mae2 = 70.0 / 3.0
        assert mase1 / mase2 == pytest.approx(mae1 / mae2)


class TestComputeUnweightedMase:
    """Per-DFU MASE then mean/median across DFUs (every DFU weighted equally)."""

    def test_basic_mean_and_median(self):
        # per-DFU MASE: 4/2=2.0, 3/3=1.0, 9/6=1.5 -> mean 1.5, median 1.5
        per_dfu = [(4.0, 2.0), (3.0, 3.0), (9.0, 6.0)]
        result = compute_unweighted_mase(per_dfu)
        assert result["n_dfus"] == 3
        assert result["n_undefined"] == 0
        assert result["mean_mase"] == pytest.approx(1.5)
        assert result["median_mase"] == pytest.approx(1.5)

    def test_zero_scale_dfu_excluded(self):
        # scale_q == 0 DFU is undefined; it must not be folded into the mean.
        per_dfu = [(4.0, 2.0), (1.0, 0.0)]
        result = compute_unweighted_mase(per_dfu)
        assert result["n_dfus"] == 2
        assert result["n_undefined"] == 1
        assert result["mean_mase"] == pytest.approx(2.0)
        assert result["median_mase"] == pytest.approx(2.0)

    def test_negative_scale_excluded_defensively(self):
        per_dfu = [(4.0, 2.0), (5.0, -1.0)]
        result = compute_unweighted_mase(per_dfu)
        assert result["n_dfus"] == 2
        assert result["n_undefined"] == 1
        assert result["mean_mase"] == pytest.approx(2.0)

    def test_all_undefined_returns_none(self):
        result = compute_unweighted_mase([(1.0, 0.0), (2.0, 0.0)])
        assert result["n_dfus"] == 2
        assert result["n_undefined"] == 2
        assert result["mean_mase"] is None
        assert result["median_mase"] is None

    def test_empty_input(self):
        result = compute_unweighted_mase([])
        assert result["n_dfus"] == 0
        assert result["n_undefined"] == 0
        assert result["mean_mase"] is None
        assert result["median_mase"] is None
