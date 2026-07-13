"""MSTL forecasting adapter used by the canonical statistical backtest."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from itertools import islice
from multiprocessing import get_context
from typing import Any

import numpy as np
import pandas as pd

from common.core.constants import FORECAST_QTY_COL
from common.ml.forecast_window import build_forecast_output_window

try:
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL
except ImportError:
    StatsForecast = None
    MSTL = None


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"]
    )


def _complete_monthly_series(
    rows: pd.DataFrame,
    last_month: pd.Timestamp,
) -> pd.Series:
    observations = rows[["startdate", "qty"]].copy()
    observations["qty"] = pd.to_numeric(observations["qty"], errors="coerce")
    first_observation = observations.loc[
        observations["qty"].notna(), "startdate"
    ].min()
    if pd.isna(first_observation):
        sku_ck = str(rows["sku_ck"].iloc[0])
        raise ValueError(f"MSTL DFU {sku_ck} has no non-null observations")

    monthly = observations.groupby("startdate")["qty"].sum(min_count=1)
    months = pd.date_range(first_observation, last_month, freq="MS")
    series = monthly.reindex(months).fillna(0.0)
    series.index = pd.DatetimeIndex(series.index, freq="MS")
    return series.astype(float)


def predict_mstl_series(
    history: pd.Series,
    predict_months: list[pd.Timestamp],
    *,
    season_length: int,
    min_history: int,
) -> pd.Series:
    """Forecast one monthly series with MSTL, returning NaN when it is ineligible."""
    if StatsForecast is None or MSTL is None:
        raise RuntimeError("MSTL requires the statistical dependency group")
    if len(history) < min_history:
        return pd.Series(np.nan, index=predict_months, dtype=float)

    window = build_forecast_output_window(
        predict_months,
        history_end=history.index.max(),
        adapter_name="MSTL",
    )

    frame = pd.DataFrame(
        {"unique_id": "series", "ds": history.index, "y": history.to_numpy()}
    )
    engine = StatsForecast(
        models=[MSTL(season_length=season_length)],
        freq="MS",
        n_jobs=1,
    )
    forecast = engine.forecast(df=frame, h=window.inference_horizon)
    value_columns = [column for column in forecast.columns if column not in {"unique_id", "ds"}]
    if not value_columns:
        raise RuntimeError("MSTL returned no forecast value column")
    forecast_months = tuple(
        pd.to_datetime(forecast["ds"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    if forecast_months != window.inference_months:
        raise RuntimeError("MSTL returned an invalid inference calendar")
    values = pd.to_numeric(forecast[value_columns[0]], errors="coerce").to_numpy()
    if len(values) != window.inference_horizon:
        raise RuntimeError("MSTL returned an incomplete inference horizon")
    selected = values[
        window.output_offset : window.output_offset + len(window.output_months)
    ]
    return pd.Series(
        np.maximum(selected, 0.0),
        index=window.output_months,
        dtype=float,
    )


def _forecast_one_dfu(
    task: tuple[str, pd.Series, list[pd.Timestamp], int, int],
) -> list[dict[str, Any]]:
    sku_ck, history, predict_months, season_length, min_history = task
    if len(history) < min_history:
        return []

    forecast = predict_mstl_series(
        history,
        predict_months,
        season_length=season_length,
        min_history=min_history,
    )
    forecast_months = [
        pd.Timestamp(month).to_period("M").to_timestamp() for month in forecast.index
    ]
    if (
        len(forecast) != len(predict_months)
        or len(set(forecast_months)) != len(forecast_months)
        or set(forecast_months) != set(predict_months)
    ):
        raise RuntimeError(f"MSTL returned incomplete forecast coverage for DFU {sku_ck}")

    values = pd.to_numeric(forecast.to_numpy(), errors="coerce").astype(float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"MSTL returned non-finite predictions for DFU {sku_ck}")

    return [
        {
            "sku_ck": sku_ck,
            "startdate": month,
            FORECAST_QTY_COL: float(value),
            "algorithm_id": "mstl",
        }
        for month, value in zip(forecast_months, values, strict=True)
    ]


def _iter_dfu_tasks(
    frame: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    *,
    season_length: int,
    min_history: int,
) -> Iterator[tuple[str, pd.Series, list[pd.Timestamp], int, int]]:
    last_month = frame["startdate"].max()
    for sku_ck, rows in frame.groupby("sku_ck", sort=False):
        yield (
            str(sku_ck),
            _complete_monthly_series(rows, last_month),
            predict_months,
            season_length,
            min_history,
        )


def _validate_forecast_coverage(
    results: list[dict[str, Any]],
    eligible_ids: set[str],
    predict_months: list[pd.Timestamp],
) -> pd.DataFrame:
    if not eligible_ids:
        if results:
            raise RuntimeError("MSTL returned forecasts for ineligible DFUs")
        return _empty_predictions()
    if not results:
        raise RuntimeError("MSTL returned incomplete forecast coverage for eligible DFUs")

    output = pd.DataFrame(results)
    output["sku_ck"] = output["sku_ck"].astype(str)
    output["startdate"] = (
        pd.to_datetime(output["startdate"]).dt.to_period("M").dt.to_timestamp()
    )
    expected_months = set(predict_months)
    actual_ids = set(output["sku_ck"])
    duplicate_rows = output.duplicated(["sku_ck", "startdate"]).any()
    wrong_month = ~output["startdate"].isin(expected_months)
    row_counts = output.groupby("sku_ck")["startdate"].count()
    incomplete_ids = set(row_counts[row_counts != len(predict_months)].index)
    if (
        actual_ids != eligible_ids
        or duplicate_rows
        or wrong_month.any()
        or incomplete_ids
    ):
        raise RuntimeError("MSTL returned incomplete forecast coverage for eligible DFUs")

    quantities = pd.to_numeric(output[FORECAST_QTY_COL], errors="coerce")
    if not np.isfinite(quantities.to_numpy(dtype=float)).all():
        raise RuntimeError("MSTL returned non-finite forecast quantities")
    if set(output["algorithm_id"]) != {"mstl"}:
        raise RuntimeError("MSTL returned rows with an invalid algorithm_id")
    output[FORECAST_QTY_COL] = quantities.astype(float)
    return output.sort_values(["sku_ck", "startdate"]).reset_index(drop=True)


def run_mstl(
    sales_df: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    *,
    season_length: int,
    min_history: int,
    n_workers: int,
) -> pd.DataFrame:
    """Forecast MSTL for every DFU in a monthly sales frame."""
    required = {"sku_ck", "startdate", "qty"}
    missing = required - set(sales_df.columns)
    if missing:
        raise ValueError(f"MSTL sales input is missing columns: {sorted(missing)}")
    if sales_df.empty or not predict_months:
        return _empty_predictions()
    if StatsForecast is None or MSTL is None:
        raise RuntimeError("MSTL requires `uv sync --extra statistical`")

    frame = sales_df.copy()
    frame["sku_ck"] = frame["sku_ck"].astype(str)
    frame["startdate"] = pd.to_datetime(frame["startdate"]).dt.to_period("M").dt.to_timestamp()
    normalized_months = [
        pd.Timestamp(month).to_period("M").to_timestamp() for month in predict_months
    ]
    if len(set(normalized_months)) != len(normalized_months):
        raise ValueError("MSTL forecast months must be unique")
    tasks = _iter_dfu_tasks(
        frame,
        normalized_months,
        season_length=season_length,
        min_history=min_history,
    )

    results: list[dict[str, Any]] = []
    eligible_ids: set[str] = set()
    sku_count = frame["sku_ck"].nunique()
    if n_workers > 1 and sku_count > 1:
        max_workers = min(n_workers, sku_count)
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=get_context("spawn"),
        ) as executor:
            pending: dict[Future[list[dict[str, Any]]], str] = {}
            for task in islice(tasks, max_workers):
                if len(task[1]) >= min_history:
                    eligible_ids.add(task[0])
                pending[executor.submit(_forecast_one_dfu, task)] = task[0]

            while pending:
                completed, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.pop(future)
                    results.extend(future.result())
                for task in islice(tasks, len(completed)):
                    if len(task[1]) >= min_history:
                        eligible_ids.add(task[0])
                    pending[executor.submit(_forecast_one_dfu, task)] = task[0]
    else:
        for task in tasks:
            if len(task[1]) >= min_history:
                eligible_ids.add(task[0])
            results.extend(_forecast_one_dfu(task))

    return _validate_forecast_coverage(results, eligible_ids, normalized_months)
