"""Tests for the canonical MSTL adapter."""

from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import wait as futures_wait
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from common.core.constants import FORECAST_QTY_COL
from common.ml.mstl import predict_mstl_series, run_mstl


def _successful_forecast(
    _history: pd.Series,
    predict_months: list[pd.Timestamp],
    *,
    season_length: int,
    min_history: int,
) -> pd.Series:
    del season_length, min_history
    return pd.Series(5.0, index=predict_months, dtype=float)


def test_predict_mstl_rejects_missing_dependency():
    history = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2025-01-01", periods=3, freq="MS"))
    with (
        patch("common.ml.mstl.StatsForecast", None),
        patch("common.ml.mstl.MSTL", None),
        pytest.raises(RuntimeError, match="statistical dependency"),
    ):
        predict_mstl_series(
            history,
            [pd.Timestamp("2025-04-01")],
            season_length=12,
            min_history=3,
        )


def test_predict_mstl_clips_negative_values():
    history = pd.Series(
        np.arange(24, dtype=float),
        index=pd.date_range("2024-01-01", periods=24, freq="MS"),
    )
    engine = MagicMock()
    engine.forecast.return_value = pd.DataFrame(
        {"unique_id": ["series", "series"], "ds": pd.date_range("2026-01-01", periods=2, freq="MS"), "MSTL": [-2.0, 4.0]}
    )
    with (
        patch("common.ml.mstl.StatsForecast", return_value=engine),
        patch("common.ml.mstl.MSTL", return_value=MagicMock()),
    ):
        result = predict_mstl_series(
            history,
            list(pd.date_range("2026-01-01", periods=2, freq="MS")),
            season_length=12,
            min_history=12,
        )
    assert result.tolist() == [0.0, 4.0]


def test_predict_mstl_forecasts_through_embargo_before_selecting_output_months():
    history = pd.Series(
        np.arange(3, dtype=float),
        index=pd.date_range("2025-01-01", periods=3, freq="MS"),
    )
    engine = MagicMock()
    engine.forecast.return_value = pd.DataFrame(
        {
            "unique_id": ["series"] * 4,
            "ds": pd.date_range("2025-04-01", periods=4, freq="MS"),
            "MSTL": [1.0, 2.0, 30.0, 40.0],
        }
    )
    with (
        patch("common.ml.mstl.StatsForecast", return_value=engine),
        patch("common.ml.mstl.MSTL", return_value=MagicMock()),
    ):
        result = predict_mstl_series(
            history,
            [pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01")],
            season_length=12,
            min_history=3,
        )

    assert engine.forecast.call_args.kwargs["h"] == 4
    assert result.index.tolist() == [
        pd.Timestamp("2025-06-01"),
        pd.Timestamp("2025-07-01"),
    ]
    assert result.tolist() == [30.0, 40.0]


def test_run_mstl_validates_input_columns():
    with pytest.raises(ValueError, match="missing columns"):
        run_mstl(
            pd.DataFrame({"sku_ck": ["A"]}),
            [pd.Timestamp("2026-01-01")],
            season_length=12,
            min_history=12,
            n_workers=1,
        )


def test_run_mstl_emits_only_mstl_rows():
    sales = pd.DataFrame(
        {
            "sku_ck": ["A"] * 3,
            "startdate": pd.date_range("2025-01-01", periods=3, freq="MS"),
            "qty": [1.0, 2.0, 3.0],
        }
    )
    with (
        patch("common.ml.mstl.StatsForecast", MagicMock()),
        patch("common.ml.mstl.MSTL", MagicMock()),
        patch(
            "common.ml.mstl.predict_mstl_series",
            return_value=pd.Series([5.0], index=[pd.Timestamp("2025-04-01")]),
        ),
    ):
        result = run_mstl(
            sales,
            [pd.Timestamp("2025-04-01")],
            season_length=12,
            min_history=3,
            n_workers=1,
        )
    assert result["algorithm_id"].unique().tolist() == ["mstl"]
    assert result[FORECAST_QTY_COL].tolist() == [5.0]


def test_run_mstl_starts_each_calendar_at_its_first_non_null_observation():
    sales = pd.DataFrame(
        {
            "sku_ck": ["EARLY"] * 5 + ["LATE"] * 4,
            "startdate": list(pd.date_range("2025-01-01", periods=5, freq="MS"))
            + list(pd.date_range("2025-01-01", periods=4, freq="MS")),
            "qty": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan, 7.0, np.nan],
        }
    )
    captured: dict[str, pd.Series] = {}

    def capture_history(
        history: pd.Series,
        predict_months: list[pd.Timestamp],
        *,
        season_length: int,
        min_history: int,
    ) -> pd.Series:
        del season_length, min_history
        key = "EARLY" if history.iloc[0] == 1.0 else "LATE"
        captured[key] = history
        return pd.Series(5.0, index=predict_months, dtype=float)

    with (
        patch("common.ml.mstl.StatsForecast", MagicMock()),
        patch("common.ml.mstl.MSTL", MagicMock()),
        patch("common.ml.mstl.predict_mstl_series", side_effect=capture_history),
    ):
        run_mstl(
            sales,
            [pd.Timestamp("2025-06-01")],
            season_length=12,
            min_history=1,
            n_workers=1,
        )

    late_history = captured["LATE"]
    assert late_history.index.tolist() == [
        pd.Timestamp("2025-03-01"),
        pd.Timestamp("2025-04-01"),
        pd.Timestamp("2025-05-01"),
    ]
    assert late_history.tolist() == [7.0, 0.0, 0.0]


def test_run_mstl_uses_spawn_context_and_bounds_parallel_submissions():
    sales = pd.DataFrame(
        {
            "sku_ck": [f"SKU-{index}" for index in range(5)],
            "startdate": [pd.Timestamp("2025-01-01")] * 5,
            "qty": [float(index + 1) for index in range(5)],
        }
    )
    executor_kwargs: dict[str, Any] = {}
    pending_sizes: list[int] = []
    submitted_at_wait: list[int] = []

    class ImmediateExecutor:
        def __init__(self, **kwargs: object) -> None:
            executor_kwargs.update(kwargs)
            self.submitted = 0

        def __enter__(self) -> "ImmediateExecutor":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def submit(
            self,
            function: Callable[[object], list[dict[str, object]]],
            task: object,
        ) -> Future[list[dict[str, object]]]:
            self.submitted += 1
            future: Future[list[dict[str, object]]] = Future()
            future.set_result(function(task))
            return future

    def recording_wait(
        pending: set[Future[list[dict[str, object]]]],
        *,
        return_when: object,
    ) -> tuple[set[Future[list[dict[str, object]]]], set[Future[list[dict[str, object]]]]]:
        pending_sizes.append(len(pending))
        submitted_at_wait.append(executor_kwargs["executor"].submitted)
        return futures_wait(pending, return_when=return_when)

    class RecordingExecutor(ImmediateExecutor):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            executor_kwargs["executor"] = self

    with (
        patch("common.ml.mstl.StatsForecast", MagicMock()),
        patch("common.ml.mstl.MSTL", MagicMock()),
        patch("common.ml.mstl.predict_mstl_series", side_effect=_successful_forecast),
        patch("common.ml.mstl.ProcessPoolExecutor", RecordingExecutor),
        patch("common.ml.mstl.wait", side_effect=recording_wait),
    ):
        result = run_mstl(
            sales,
            [pd.Timestamp("2025-02-01")],
            season_length=12,
            min_history=1,
            n_workers=2,
        )

    assert len(result) == 5
    context = executor_kwargs["mp_context"]
    assert context.get_start_method() == "spawn"
    assert max(pending_sizes) <= 2
    assert submitted_at_wait[0] == 2


def test_run_mstl_propagates_a_per_dfu_failure():
    sales = pd.DataFrame(
        {
            "sku_ck": ["A", "B"],
            "startdate": [pd.Timestamp("2025-01-01")] * 2,
            "qty": [1.0, 2.0],
        }
    )
    with (
        patch("common.ml.mstl.StatsForecast", MagicMock()),
        patch("common.ml.mstl.MSTL", MagicMock()),
        patch(
            "common.ml.mstl.predict_mstl_series",
            side_effect=[
                pd.Series([5.0], index=[pd.Timestamp("2025-02-01")]),
                RuntimeError("DFU fit failed"),
            ],
        ),
        pytest.raises(RuntimeError, match="DFU fit failed"),
    ):
        run_mstl(
            sales,
            [pd.Timestamp("2025-02-01")],
            season_length=12,
            min_history=1,
            n_workers=1,
        )


def test_run_mstl_rejects_incomplete_eligible_dfu_coverage():
    sales = pd.DataFrame(
        {
            "sku_ck": ["A", "A"],
            "startdate": pd.date_range("2025-01-01", periods=2, freq="MS"),
            "qty": [1.0, 2.0],
        }
    )
    with (
        patch("common.ml.mstl.StatsForecast", MagicMock()),
        patch("common.ml.mstl.MSTL", MagicMock()),
        patch(
            "common.ml.mstl.predict_mstl_series",
            return_value=pd.Series([5.0], index=[pd.Timestamp("2025-03-01")]),
        ),
        pytest.raises(RuntimeError, match="incomplete forecast coverage"),
    ):
        run_mstl(
            sales,
            [pd.Timestamp("2025-03-01"), pd.Timestamp("2025-04-01")],
            season_length=12,
            min_history=2,
            n_workers=1,
        )
