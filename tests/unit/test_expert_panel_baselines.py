"""Tests for the expert-panel per-DFU baselines (rolling mean / rolling median).

These exercise the raw-time-series baselines in
``common/ml/expert_panel/baselines.py`` — distinct from the backtest-dispatch
baselines in ``scripts/ml/run_backtest.py`` (covered by
``tests/unit/test_backtest_baselines.py``).
"""

import pandas as pd

from common.core.constants import FORECAST_QTY_COL
from common.ml.expert_panel.baselines import (
    predict_rolling_mean,
    predict_rolling_median,
)


def _make_sales(values: list[float], sku_ck: str = "ck1") -> pd.DataFrame:
    """Build a single-DFU monthly sales frame with len(values) consecutive months."""
    return pd.DataFrame(
        {
            "sku_ck": [sku_ck] * len(values),
            "startdate": pd.date_range("2023-01-01", periods=len(values), freq="MS"),
            "qty": values,
        }
    )


_PREDICT_MONTHS = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]


class TestPredictRollingMedian:
    """Median math + robustness contract for the expert-panel rolling median."""

    def test_median_of_trailing_window(self):
        """Trailing-6 median ignores the outlier spike that the mean chases."""
        # Window [1,2,3,4,5,100]: median = (3+4)/2 = 3.5; mean = 19.166...
        sales = _make_sales([1, 2, 3, 4, 5, 100])
        result = predict_rolling_median(sales, _PREDICT_MONTHS, window=6)

        assert abs(result[FORECAST_QTY_COL].iloc[0] - 3.5) < 0.01
        mean_result = predict_rolling_mean(sales, _PREDICT_MONTHS, window=6)
        assert mean_result[FORECAST_QTY_COL].iloc[0] > 19.0
        # The median is robust: it sits far below the spike-inflated mean.
        assert result[FORECAST_QTY_COL].iloc[0] < mean_result[FORECAST_QTY_COL].iloc[0]

    def test_uses_most_recent_window(self):
        """The window selects the most recent months, not the earliest."""
        sales = _make_sales([100, 100, 100, 100, 100, 100, 200, 200, 200])
        result = predict_rolling_median(sales, _PREDICT_MONTHS, window=3)
        assert abs(result[FORECAST_QTY_COL].iloc[0] - 200.0) < 0.01

    def test_fewer_months_than_window(self):
        """With fewer months than the window, use all available history."""
        sales = _make_sales([100, 300])
        result = predict_rolling_median(sales, _PREDICT_MONTHS, window=6)
        assert abs(result[FORECAST_QTY_COL].iloc[0] - 200.0) < 0.01

    def test_empty_input_returns_empty_with_algorithm_id(self):
        """Empty sales -> empty frame still tagged algorithm_id=rolling_median."""
        empty = pd.DataFrame(
            {
                "sku_ck": pd.Series(dtype="object"),
                "startdate": pd.Series(dtype="datetime64[ns]"),
                "qty": pd.Series(dtype="float64"),
            }
        )
        result = predict_rolling_median(empty, _PREDICT_MONTHS, window=6)
        assert result.empty
        assert (result["algorithm_id"] == "rolling_median").all()
        assert FORECAST_QTY_COL in result.columns

    def test_clamps_negative_to_zero(self):
        """The max(0) clamp prevents negative forecasts."""
        sales = _make_sales([-50, -30, -10])
        result = predict_rolling_median(sales, _PREDICT_MONTHS, window=6)
        assert (result[FORECAST_QTY_COL] >= 0).all()

    def test_flat_across_predict_months(self):
        """The same median value is emitted for every predict month."""
        sales = _make_sales([10, 20, 30, 40, 50, 60])
        result = predict_rolling_median(sales, _PREDICT_MONTHS, window=6)
        per_month = result.groupby("startdate")[FORECAST_QTY_COL].first()
        assert per_month.nunique() == 1
        # Median of [10..60] = 35.0
        assert abs(per_month.iloc[0] - 35.0) < 0.01

    def test_algorithm_id_tag(self):
        """Non-empty output is tagged algorithm_id=rolling_median."""
        sales = _make_sales([10, 20, 30])
        result = predict_rolling_median(sales, _PREDICT_MONTHS, window=6)
        assert (result["algorithm_id"] == "rolling_median").all()

    def test_per_dfu_independence(self):
        """Each DFU gets its own trailing median."""
        a = _make_sales([1, 2, 3, 4, 5, 100], sku_ck="ckA")
        b = _make_sales([10, 10, 10, 10, 10, 10], sku_ck="ckB")
        sales = pd.concat([a, b], ignore_index=True)
        result = predict_rolling_median(sales, [pd.Timestamp("2024-01-01")], window=6)
        jan = result[result["startdate"] == pd.Timestamp("2024-01-01")]
        by_sku = jan.set_index("sku_ck")[FORECAST_QTY_COL]
        assert abs(by_sku["ckA"] - 3.5) < 0.01
        assert abs(by_sku["ckB"] - 10.0) < 0.01
