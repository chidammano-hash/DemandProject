"""Shared calendar-offset contract for direct forecast adapters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from common.core.constants import FORECAST_QTY_COL


def test_output_window_includes_embargo_months_before_evaluated_dates() -> None:
    from common.ml.forecast_window import build_forecast_output_window

    window = build_forecast_output_window(
        [pd.Timestamp("2026-03-01"), pd.Timestamp("2026-04-01")],
        history_end=pd.Timestamp("2026-01-01"),
        adapter_name="test",
    )

    assert window.output_months == (
        pd.Timestamp("2026-03-01"),
        pd.Timestamp("2026-04-01"),
    )
    assert window.inference_months == (
        pd.Timestamp("2026-02-01"),
        pd.Timestamp("2026-03-01"),
        pd.Timestamp("2026-04-01"),
    )
    assert window.output_offset == 1
    assert window.inference_horizon == 3


def test_mstl_discards_embargo_predictions_before_relabeling_output() -> None:
    from common.ml.mstl import predict_mstl_series

    history = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2025-11-01", periods=3, freq="MS"),
    )
    engine = MagicMock()
    engine.forecast.return_value = pd.DataFrame(
        {
            "unique_id": ["series"] * 3,
            "ds": pd.date_range("2026-02-01", periods=3, freq="MS"),
            "MSTL": [10.0, 20.0, 30.0],
        }
    )
    with (
        patch("common.ml.mstl.StatsForecast", return_value=engine),
        patch("common.ml.mstl.MSTL", return_value=MagicMock()),
    ):
        result = predict_mstl_series(
            history,
            [pd.Timestamp("2026-03-01"), pd.Timestamp("2026-04-01")],
            season_length=12,
            min_history=3,
        )

    assert engine.forecast.call_args.kwargs["h"] == 3
    assert result.index.tolist() == [
        pd.Timestamp("2026-03-01"),
        pd.Timestamp("2026-04-01"),
    ]
    assert result.tolist() == [20.0, 30.0]


def test_neural_discards_embargo_predictions_before_relabeling_output() -> None:
    from common.ml.neural_forecast import FittedNeuralModel, predict_neural_model

    sales = pd.DataFrame(
        {
            "sku_ck": ["sku-1"] * 3,
            "startdate": pd.date_range("2026-04-01", periods=3, freq="MS"),
            "qty": [1.0, 2.0, 3.0],
        }
    )

    class FakeNeuralForecast:
        h = 2

        def __init__(self) -> None:
            self.requested_horizon: int | None = None

        def predict(self, *, df: pd.DataFrame, h: int) -> pd.DataFrame:
            self.requested_horizon = h
            return pd.DataFrame(
                {
                    "unique_id": ["sku-1"] * h,
                    "ds": pd.date_range("2026-07-01", periods=h, freq="MS"),
                    "NHITS": range(1, h + 1),
                }
            )

    runtime = FakeNeuralForecast()
    fitted = FittedNeuralModel(
        neural_forecast=runtime,
        model_id="nhits",
        fitted_horizon=2,
        min_history=3,
        training_dfu_count=1,
    )

    result = predict_neural_model(
        fitted,
        sales,
        [pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01")],
    )

    assert runtime.requested_horizon == 3
    assert result["startdate"].tolist() == [
        pd.Timestamp("2026-08-01"),
        pd.Timestamp("2026-09-01"),
    ]
    assert result[FORECAST_QTY_COL].tolist() == [2.0, 3.0]
    assert result.attrs["output_offset"] == 1
