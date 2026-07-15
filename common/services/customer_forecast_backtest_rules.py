"""Vectorized rule-router backtests for customer demand series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.ml.croston import parse_croston_parameters
from common.ml.customer_forecast_rules import parse_customer_forecast_rule_parameters
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


def _first_demand_by_identity(
    history: pd.DataFrame,
    identity_rows: pd.DataFrame,
) -> np.ndarray:
    first_demand = history[[*_IDENTITY_COLUMNS, "series_first_demand_month"]].drop_duplicates(
        ignore_index=True
    )
    duplicate_metadata = first_demand.duplicated(_IDENTITY_COLUMNS, keep=False)
    if duplicate_metadata.any():
        raise ValueError("Customer backtest history contains inconsistent first demand months")

    observed_first_demand = (
        history.loc[history["demand_qty"] > 0]
        .groupby(_IDENTITY_COLUMNS, sort=False, dropna=False)["startdate"]
        .min()
        .rename("observed_first_demand_month")
        .reset_index()
    )
    aligned = identity_rows.merge(
        first_demand,
        on=_IDENTITY_COLUMNS,
        how="left",
        validate="one_to_one",
    ).merge(
        observed_first_demand,
        on=_IDENTITY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    missing_metadata = aligned["series_first_demand_month"].isna()
    observed_demand = aligned["observed_first_demand_month"].notna()
    metadata_after_observation = (
        observed_demand
        & ~missing_metadata
        & (aligned["series_first_demand_month"] > aligned["observed_first_demand_month"])
    )
    if (missing_metadata & observed_demand).any() or metadata_after_observation.any():
        raise ValueError("Customer backtest first demand month contradicts demand history")
    result: np.ndarray = aligned["series_first_demand_month"].to_numpy(dtype="datetime64[ns]")
    return result


def build_customer_rule_backtest_batch(
    frame: pd.DataFrame,
    window: CustomerForecastWindow,
    *,
    evaluation_months: int,
    min_train_months: int,
    recent_sales_lookback_months: int,
    rule_params: dict[str, Any],
    croston_params: dict[str, Any],
) -> pd.DataFrame:
    """Produce causal one-step rule forecasts and aggregate one source batch."""
    required = {
        *_IDENTITY_COLUMNS,
        "startdate",
        "demand_qty",
        "sales_qty",
        "series_first_demand_month",
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
    raw_first_demand = history["series_first_demand_month"]
    history["series_first_demand_month"] = pd.to_datetime(raw_first_demand, errors="coerce")
    if (raw_first_demand.notna() & history["series_first_demand_month"].isna()).any():
        raise ValueError("Customer backtest history contains an invalid first demand month")
    first_demand_values = history["series_first_demand_month"].dropna()
    if not first_demand_values.eq(first_demand_values.dt.to_period("M").dt.to_timestamp()).all():
        raise ValueError("Customer backtest first demand month must start a month")
    for column in ("demand_qty", "sales_qty"):
        history[column] = pd.to_numeric(history[column], errors="coerce")
        values = history[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Customer backtest history contains invalid {column}")
    if history[_IDENTITY_COLUMNS].isna().any(axis=1).any():
        raise ValueError("Customer backtest history contains an invalid identity")

    rules = parse_customer_forecast_rule_parameters(rule_params)
    croston = parse_croston_parameters(croston_params)
    calendar = pd.date_range(window.history_start, periods=window.history_months, freq="MS")
    identity_rows = (
        history[_IDENTITY_COLUMNS]
        .drop_duplicates()
        .sort_values(_IDENTITY_COLUMNS, ignore_index=True)
    )
    first_demand_months = _first_demand_by_identity(history, identity_rows)
    identity_index = pd.MultiIndex.from_frame(identity_rows)
    series_codes = identity_index.get_indexer(pd.MultiIndex.from_frame(history[_IDENTITY_COLUMNS]))
    month_codes = calendar.get_indexer(history["startdate"])
    valid_rows = (series_codes >= 0) & (month_codes >= 0)
    series_count = len(identity_rows)
    demand = np.zeros((series_count, window.history_months), dtype=float)
    sales = np.zeros_like(demand)
    np.add.at(
        demand,
        (series_codes[valid_rows], month_codes[valid_rows]),
        history.loc[valid_rows, "demand_qty"].to_numpy(dtype=float),
    )
    np.add.at(
        sales,
        (series_codes[valid_rows], month_codes[valid_rows]),
        history.loc[valid_rows, "sales_qty"].to_numpy(dtype=float),
    )

    demand_size = np.zeros(series_count, dtype=float)
    demand_interval = np.zeros(series_count, dtype=float)
    previous_position = np.full(series_count, -1, dtype=int)
    initialized = np.zeros(series_count, dtype=bool)
    evaluation_start_index = window.history_months - evaluation_months
    output_identities = identity_rows.astype(str)
    records: list[pd.DataFrame] = []

    for position in range(window.history_months - 1):
        positive = demand[:, position] > 0
        first = positive & ~initialized
        repeated = positive & initialized
        demand_size[first] = demand[first, position]
        demand_interval[first] = float(position + 1)
        if repeated.any():
            intervals = position - previous_position[repeated]
            demand_size[repeated] += croston.alpha * (
                demand[repeated, position] - demand_size[repeated]
            )
            demand_interval[repeated] += croston.alpha * (intervals - demand_interval[repeated])
        previous_position[positive] = position
        initialized[positive] = True

        forecast_index = position + 1
        if forecast_index < max(evaluation_start_index, min_train_months):
            continue
        recent_sales_start = max(0, forecast_index - recent_sales_lookback_months)
        active = (sales[:, recent_sales_start:forecast_index] > 0).any(axis=1)
        if not active.any():
            continue

        forecast = np.divide(
            demand_size,
            demand_interval,
            out=np.zeros(series_count, dtype=float),
            where=demand_interval > 0,
        )
        if croston.variant == "sba":
            forecast *= 1.0 - croston.alpha / 2.0
        if croston.recursive:
            forecast = (
                croston.recursive_damping * demand[:, position]
                + (1.0 - croston.recursive_damping) * forecast
            )

        recent_demand_start = calendar[forecast_index] - pd.DateOffset(
            months=rules.recent_demand_lookback_months
        )
        recent_route = (
            ~np.isnat(first_demand_months)
            & (first_demand_months >= recent_demand_start.to_datetime64())
            & (first_demand_months < calendar[forecast_index].to_datetime64())
        )
        moving_average = demand[
            :,
            max(0, forecast_index - rules.moving_average_window_months) : forecast_index,
        ].mean(axis=1)
        forecast[recent_route] = moving_average[recent_route]

        complete_repeat_cycle = forecast_index >= rules.repeat_history_lookback_months
        if complete_repeat_cycle:
            repeat_start = forecast_index - rules.repeat_history_lookback_months
            demand_months = np.count_nonzero(
                demand[:, repeat_start:forecast_index] > 0,
                axis=1,
            )
            repeat_route = ~recent_route & (demand_months >= rules.repeat_history_min_demand_months)
            forecast[repeat_route] = demand[
                repeat_route,
                forecast_index - rules.repeat_history_lookback_months,
            ]

        active_rows = np.flatnonzero(active)
        result = output_identities.iloc[active_rows][["item_id", "location_id"]].copy()
        result = result.rename(columns={"location_id": "loc"})
        result["forecast_origin"] = calendar[forecast_index - 1].date()
        result["forecast_month"] = calendar[forecast_index].date()
        result["raw_customer_demand_qty"] = np.maximum(forecast[active_rows], 0.0)
        result["customer_series_count"] = 1
        records.append(result)

    if not records:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return (
        pd.concat(records, ignore_index=True)
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
