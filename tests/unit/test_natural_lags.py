"""Tests for assign_natural_lags — correct lag assignment from timeframes.

Verifies that:
  - Each prediction gets the correct natural lag based on its timeframe
  - Only lags 0..4 are kept
  - Different timeframes produce genuinely different lags for the same demand month
  - execution_lag and forecast_ck are correctly assigned
"""

import pandas as pd
import pytest

from common.ml.backtest_framework import assign_natural_lags, generate_timeframes


def _make_predictions(
    timeframes: list[dict],
    n_dfus: int = 2,
) -> pd.DataFrame:
    """Build a synthetic predictions DataFrame from timeframe metadata.

    Each timeframe predicts all months in its [predict_start, predict_end] range.
    """
    rows = []
    for tf in timeframes:
        predict_start = tf["predict_start"]
        predict_end = tf["predict_end"]
        months = pd.date_range(predict_start, predict_end, freq="MS")
        for month in months:
            for i in range(n_dfus):
                rows.append({
                    "sku_ck": f"item{i}_grp_loc{i}",
                    "item_id": f"item{i}",
                    "customer_group": "grp",
                    "loc": f"loc{i}",
                    "startdate": month,
                    "basefcst_pref": 100.0 + tf["index"] * 10 + i,
                    "timeframe": tf["label"],
                    "timeframe_idx": tf["index"],
                    "model_id": "lgbm_cluster",
                })
    return pd.DataFrame(rows)


def _make_exec_lag_map(n_dfus: int = 2) -> dict:
    """Each DFU gets a different execution lag for testing."""
    return {f"item{i}_grp_loc{i}": i % 5 for i in range(n_dfus)}


class TestAssignNaturalLags:
    """Verify natural lag computation from timeframe training cutoff."""

    def test_latest_month_has_all_5_lags(self):
        """The most recent demand month should have predictions at lag 0-4."""
        latest = pd.Timestamp("2025-10-01")
        earliest = pd.Timestamp("2023-01-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        preds = _make_predictions(tfs, n_dfus=1)
        result = assign_natural_lags(preds, tfs, max_lag=4, execution_lag_map={})

        latest_month = result[result["startdate"] == latest]
        lags = sorted(latest_month["lag"].unique())
        assert lags == [0, 1, 2, 3, 4]

    def test_each_lag_has_different_prediction(self):
        """Different lags for the same DFU-month should come from different
        timeframes and have different basefcst_pref values."""
        latest = pd.Timestamp("2025-10-01")
        earliest = pd.Timestamp("2023-01-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        preds = _make_predictions(tfs, n_dfus=1)
        result = assign_natural_lags(preds, tfs, max_lag=4, execution_lag_map={})

        latest_dfu = result[
            (result["startdate"] == latest) &
            (result["item_id"] == "item0")
        ].sort_values("lag")

        # Each lag should come from a different timeframe
        timeframes_used = latest_dfu["timeframe"].tolist()
        assert len(set(timeframes_used)) == 5  # 5 unique timeframes

        # basefcst_pref should differ because each timeframe produces different values
        fcst_values = latest_dfu["basefcst_pref"].tolist()
        assert len(set(fcst_values)) == 5  # 5 unique forecast values

    def test_lag_0_from_latest_timeframe(self):
        """Lag 0 for the latest month should come from the last timeframe (J)."""
        latest = pd.Timestamp("2025-10-01")
        earliest = pd.Timestamp("2023-01-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        preds = _make_predictions(tfs, n_dfus=1)
        result = assign_natural_lags(preds, tfs, max_lag=4, execution_lag_map={})

        lag0_latest = result[
            (result["startdate"] == latest) &
            (result["lag"] == 0)
        ]
        assert (lag0_latest["timeframe"] == "J").all()

    def test_lag_4_from_earlier_timeframe(self):
        """Lag 4 for the latest month should come from timeframe F."""
        latest = pd.Timestamp("2025-10-01")
        earliest = pd.Timestamp("2023-01-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        preds = _make_predictions(tfs, n_dfus=1)
        result = assign_natural_lags(preds, tfs, max_lag=4, execution_lag_map={})

        lag4_latest = result[
            (result["startdate"] == latest) &
            (result["lag"] == 4)
        ]
        assert (lag4_latest["timeframe"] == "F").all()

    def test_lags_above_max_filtered_out(self):
        """Predictions with natural lag > max_lag should not appear."""
        latest = pd.Timestamp("2025-10-01")
        earliest = pd.Timestamp("2023-01-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        preds = _make_predictions(tfs, n_dfus=1)
        result = assign_natural_lags(preds, tfs, max_lag=4, execution_lag_map={})

        assert result["lag"].max() <= 4
        assert result["lag"].min() >= 0

    def test_execution_lag_assigned_from_map(self):
        """execution_lag should come from the DFU dimension map."""
        latest = pd.Timestamp("2025-10-01")
        earliest = pd.Timestamp("2023-01-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        preds = _make_predictions(tfs, n_dfus=3)
        exec_map = {
            "item0_grp_loc0": 0,
            "item1_grp_loc1": 2,
            "item2_grp_loc2": 4,
        }
        result = assign_natural_lags(preds, tfs, max_lag=4, execution_lag_map=exec_map)

        for sku_ck, expected_lag in exec_map.items():
            sku_rows = result[result["sku_ck"] == sku_ck]
            assert (sku_rows["execution_lag"] == expected_lag).all()

    def test_fcstdate_computed_correctly(self):
        """fcstdate should be startdate minus lag months."""
        latest = pd.Timestamp("2025-10-01")
        earliest = pd.Timestamp("2023-01-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        preds = _make_predictions(tfs, n_dfus=1)
        result = assign_natural_lags(preds, tfs, max_lag=4, execution_lag_map={})

        for _, row in result.iterrows():
            expected_fcstdate = row["startdate"] - pd.DateOffset(months=int(row["lag"]))
            assert row["fcstdate"] == expected_fcstdate, (
                f"lag={row['lag']}, startdate={row['startdate']}, "
                f"expected fcstdate={expected_fcstdate}, got {row['fcstdate']}"
            )

    def test_forecast_ck_includes_fcstdate(self):
        """forecast_ck should encode item_id, customer_group, loc, fcstdate, startdate."""
        latest = pd.Timestamp("2025-10-01")
        earliest = pd.Timestamp("2023-01-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        preds = _make_predictions(tfs, n_dfus=1)
        result = assign_natural_lags(preds, tfs, max_lag=4, execution_lag_map={})

        row = result.iloc[0]
        parts = row["forecast_ck"].split("_")
        # Should contain: item_id, customer_group, loc, fcstdate, startdate
        assert len(parts) >= 5

    def test_no_train_end_column_in_output(self):
        """The helper _train_end column should be dropped."""
        latest = pd.Timestamp("2025-10-01")
        earliest = pd.Timestamp("2023-01-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        preds = _make_predictions(tfs, n_dfus=1)
        result = assign_natural_lags(preds, tfs, max_lag=4, execution_lag_map={})
        assert "_train_end" not in result.columns

    def test_natural_lag_formula(self):
        """Verify the lag formula: lag = months(startdate - train_end) - 1."""
        latest = pd.Timestamp("2025-10-01")
        earliest = pd.Timestamp("2023-01-01")
        tfs = generate_timeframes(earliest, latest, n=10)

        # Timeframe J: train_end = Oct 2025 - 1mo = Sep 2025
        # Predicting Oct 2025: lag = (Oct - Sep) - 1 = 0
        tf_j = tfs[9]
        assert tf_j["label"] == "J"

        preds = _make_predictions([tf_j], n_dfus=1)
        result = assign_natural_lags(preds, [tf_j], max_lag=4, execution_lag_map={})

        # Only Oct 2025 is predicted by J, at lag 0
        assert len(result) == 1
        assert result.iloc[0]["lag"] == 0
        assert result.iloc[0]["startdate"] == latest

    def test_n_dfu_months_same_across_lags_for_common_months(self):
        """For months that have all 5 lags, the DFU count should be equal."""
        latest = pd.Timestamp("2025-10-01")
        earliest = pd.Timestamp("2023-01-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        preds = _make_predictions(tfs, n_dfus=3)
        result = assign_natural_lags(preds, tfs, max_lag=4, execution_lag_map={})

        # The last 6 months (latest through latest-5) should have all 5 lags
        # For those months, count DFU-months per lag
        full_coverage_start = latest - pd.DateOffset(months=5)
        full_months = result[result["startdate"] >= full_coverage_start]

        counts_by_lag = full_months.groupby("lag").size()
        # All lags should have the same count
        assert len(counts_by_lag.unique()) == 1

    def test_earlier_months_have_fewer_lags(self):
        """The earliest predicted months may not have all 5 lags."""
        latest = pd.Timestamp("2025-10-01")
        earliest = pd.Timestamp("2023-01-01")
        tfs = generate_timeframes(earliest, latest, n=10)
        preds = _make_predictions(tfs, n_dfus=1)
        result = assign_natural_lags(preds, tfs, max_lag=4, execution_lag_map={})

        # The earliest predicted month (latest - 9mo) should only have lag 0
        earliest_predicted = latest - pd.DateOffset(months=9)
        earliest_rows = result[result["startdate"] == earliest_predicted]
        assert len(earliest_rows["lag"].unique()) == 1
        assert earliest_rows.iloc[0]["lag"] == 0
