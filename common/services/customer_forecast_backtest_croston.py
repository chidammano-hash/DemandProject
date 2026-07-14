"""Croston customer-series backtest batch construction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.ml.croston import croston_forecast
from common.services.customer_forecast import CustomerForecastWindow

_IDENTITY_COLUMNS = ["item_id", "location_id", "customer_no"]
_OUTPUT_COLUMNS = [
    "item_id",
    "loc",
    "forecast_origin",
    "forecast_month",
    "raw_customer_demand_qty",
    "customer_series_count",
]


def build_croston_backtest_batch(
    frame: pd.DataFrame,
    window: CustomerForecastWindow,
    *,
    evaluation_months: int,
    min_train_months: int,
    recent_sales_lookback_months: int,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Produce one-step Croston predictions and aggregate within one source batch."""
    required = {
        *_IDENTITY_COLUMNS,
        "startdate",
        "demand_qty",
        "sales_qty",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Customer backtest history is missing columns: {sorted(missing)}")
    if evaluation_months <= 0 or min_train_months <= 0:
        raise ValueError("Customer backtest windows must be positive")
    if evaluation_months + min_train_months > window.history_months:
        raise ValueError("Customer backtest window exceeds available history")
    if recent_sales_lookback_months <= 0:
        raise ValueError("Customer backtest activity window must be positive")
    if frame.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    history = frame.copy()
    history["startdate"] = pd.to_datetime(history["startdate"], errors="coerce")
    for column in ("demand_qty", "sales_qty"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
        values = history[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Customer backtest history contains invalid {column}")
    if history[_IDENTITY_COLUMNS].isna().any(axis=1).any():
        raise ValueError("Customer backtest history contains an invalid identity")

    calendar = pd.date_range(window.history_start, periods=window.history_months, freq="MS")
    evaluation_start_index = window.history_months - evaluation_months
    records: list[dict[str, Any]] = []
    for identity, group in history.groupby(_IDENTITY_COLUMNS, sort=True, dropna=False):
        monthly = group.groupby("startdate", as_index=True)[["demand_qty", "sales_qty"]].sum()
        dense = monthly.reindex(calendar, fill_value=0.0).astype(float)
        for forecast_index in range(evaluation_start_index, window.history_months):
            if forecast_index < min_train_months:
                continue
            recent_start = max(0, forecast_index - recent_sales_lookback_months)
            if not dense["sales_qty"].iloc[recent_start:forecast_index].gt(0).any():
                continue
            prediction = float(
                croston_forecast(
                    dense["demand_qty"].iloc[:forecast_index].to_numpy(dtype=float),
                    horizon=1,
                    params=params,
                )[0]
            )
            records.append(
                {
                    "item_id": str(identity[0]),
                    "loc": str(identity[1]),
                    "forecast_origin": calendar[forecast_index - 1].date(),
                    "forecast_month": calendar[forecast_index].date(),
                    "raw_customer_demand_qty": prediction,
                    "customer_series_count": 1,
                }
            )
    if not records:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return (
        pd.DataFrame.from_records(records)
        .groupby(
            ["item_id", "loc", "forecast_origin", "forecast_month"],
            as_index=False,
        )
        .agg(
            raw_customer_demand_qty=("raw_customer_demand_qty", "sum"),
            customer_series_count=("customer_series_count", "sum"),
        )
        .sort_values(["item_id", "loc", "forecast_month"], ignore_index=True)
    )
