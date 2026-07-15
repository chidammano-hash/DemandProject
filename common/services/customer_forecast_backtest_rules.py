"""Vectorized rule-router backtests for customer demand series."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from common.ml.croston import parse_croston_parameters
from common.ml.customer_forecast_rules import (
    parse_customer_forecast_rule_parameters,
    parse_customer_statistical_parameters,
)
from common.services.customer_forecast import CustomerForecastWindow

_IDENTITY_COLUMNS = ["item_id", "location_id", "customer_no"]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
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


def _ses_terminal(matrix: FloatArray, alpha: float) -> FloatArray:
    """Return row-wise SES terminal levels without a series loop."""
    level: FloatArray = matrix[:, 0].astype(np.float64, copy=True)
    for position in range(1, matrix.shape[1]):
        level += alpha * (matrix[:, position] - level)
    return level


def _adida_terminal(
    demand: FloatArray,
    forecast_index: int,
    positive_count: IntArray,
    effective_history_months: IntArray,
    route_mask: BoolArray,
    alpha: float,
) -> FloatArray:
    """Return vectorized ADIDA monthly rates for the selected rows."""
    result: FloatArray = np.zeros(demand.shape[0], dtype=np.float64)
    if not route_mask.any():
        return result
    aggregation_levels = np.maximum(
        1,
        np.rint(
            np.divide(
                effective_history_months,
                positive_count,
                out=np.ones_like(positive_count, dtype=float),
                where=positive_count > 0,
            )
        ).astype(int),
    )
    for history_months in np.unique(effective_history_months[route_mask]):
        history_mask = route_mask & (effective_history_months == history_months)
        for aggregation_level in np.unique(aggregation_levels[history_mask]):
            level_mask = history_mask & (aggregation_levels == aggregation_level)
            rows = demand[
                level_mask,
                forecast_index - int(history_months) : forecast_index,
            ]
            remainder = int(history_months) % int(aggregation_level)
            if remainder:
                rows = np.pad(rows, ((0, 0), (int(aggregation_level) - remainder, 0)))
            aggregated = rows.reshape(rows.shape[0], -1, int(aggregation_level)).sum(axis=2)
            result[level_mask] = _ses_terminal(aggregated, alpha) / int(aggregation_level)
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
    statistical_params: dict[str, Any],
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
    statistics = parse_customer_statistical_parameters(statistical_params)
    calendar = pd.date_range(window.history_start, periods=window.history_months, freq="MS")
    identity_rows = (
        history[_IDENTITY_COLUMNS]
        .drop_duplicates()
        .sort_values(_IDENTITY_COLUMNS, ignore_index=True)
    )
    first_demand_months = _first_demand_by_identity(history, identity_rows)
    first_demand_positions = np.zeros(len(identity_rows), dtype=int)
    first_demand_in_window = ~np.isnat(first_demand_months) & (
        first_demand_months >= calendar[0].to_datetime64()
    )
    if first_demand_in_window.any():
        first_month_ordinals = (
            first_demand_months[first_demand_in_window].astype("datetime64[M]").astype(int)
        )
        calendar_start_ordinal = calendar[0].to_datetime64().astype("datetime64[M]").astype(int)
        first_demand_positions[first_demand_in_window] = (
            first_month_ordinals - calendar_start_ordinal
        )
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
    positive_count = np.zeros(series_count, dtype=int)
    positive_sum = np.zeros(series_count, dtype=float)
    positive_sumsq = np.zeros(series_count, dtype=float)
    tsb_demand_size = np.zeros(series_count, dtype=float)
    tsb_probability = np.zeros(series_count, dtype=float)
    tsb_initialized = np.zeros(series_count, dtype=bool)
    ses_level = np.zeros(series_count, dtype=float)
    holt_level = np.zeros(series_count, dtype=float)
    holt_trend = np.zeros(series_count, dtype=float)
    statistical_state_initialized = np.zeros(series_count, dtype=bool)
    evaluation_start_index = window.history_months - evaluation_months
    output_identities = identity_rows.astype(str)
    records: list[pd.DataFrame] = []

    for position in range(window.history_months - 1):
        positive = demand[:, position] > 0
        positive_count += positive
        positive_sum += np.where(positive, demand[:, position], 0.0)
        positive_sumsq += np.where(positive, demand[:, position] ** 2, 0.0)
        first = positive & ~initialized
        repeated = positive & initialized
        demand_size[first] = demand[first, position]
        demand_interval[first] = np.maximum(
            1,
            position - first_demand_positions[first] + 1,
        )
        if repeated.any():
            intervals = position - previous_position[repeated]
            demand_size[repeated] += croston.alpha * (
                demand[repeated, position] - demand_size[repeated]
            )
            demand_interval[repeated] += croston.alpha * (intervals - demand_interval[repeated])
        previous_position[positive] = position
        initialized[positive] = True

        tsb_first = positive & ~tsb_initialized
        tsb_repeated = positive & tsb_initialized
        tsb_demand_size[tsb_first] = demand[tsb_first, position]
        tsb_demand_size[tsb_repeated] += statistics.tsb_demand_alpha * (
            demand[tsb_repeated, position] - tsb_demand_size[tsb_repeated]
        )
        tsb_probability[tsb_first] = 1.0
        tsb_initialized[positive] = True
        tsb_probability[tsb_initialized] += statistics.tsb_probability_alpha * (
            positive[tsb_initialized].astype(float) - tsb_probability[tsb_initialized]
        )

        starts_now = (~statistical_state_initialized) & (position >= first_demand_positions)
        ses_level[starts_now] = demand[starts_now, position]
        holt_level[starts_now] = demand[starts_now, position]
        statistical_state_initialized[starts_now] = True
        continuing = statistical_state_initialized & ~starts_now
        if continuing.any():
            ses_level[continuing] += statistics.ses_alpha * (
                demand[continuing, position] - ses_level[continuing]
            )
            second_exposure_month = continuing & (position == first_demand_positions + 1)
            holt_trend[second_exposure_month] = (
                demand[second_exposure_month, position] - holt_level[second_exposure_month]
            )
            previous_holt_level = holt_level.copy()
            previous_holt_trend = holt_trend.copy()
            holt_level[continuing] = statistics.holt_level_alpha * demand[continuing, position] + (
                1.0 - statistics.holt_level_alpha
            ) * (
                previous_holt_level[continuing]
                + statistics.holt_damping * previous_holt_trend[continuing]
            )
            holt_trend[continuing] = statistics.holt_trend_alpha * (
                holt_level[continuing] - previous_holt_level[continuing]
            ) + (1.0 - statistics.holt_trend_alpha) * (
                statistics.holt_damping * previous_holt_trend[continuing]
            )

        forecast_index = position + 1
        if forecast_index < max(evaluation_start_index, min_train_months):
            continue
        recent_sales_start = max(0, forecast_index - recent_sales_lookback_months)
        active = (sales[:, recent_sales_start:forecast_index] > 0).any(axis=1)
        if not active.any():
            continue
        effective_history_months = np.clip(
            forecast_index - first_demand_positions,
            1,
            forecast_index,
        )

        croston_forecast = np.divide(
            demand_size,
            demand_interval,
            out=np.zeros(series_count, dtype=float),
            where=demand_interval > 0,
        )
        if croston.variant == "sba":
            croston_forecast *= 1.0 - croston.alpha / 2.0
        if croston.recursive:
            croston_forecast = (
                croston.recursive_damping * demand[:, position]
                + (1.0 - croston.recursive_damping) * croston_forecast
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
        sparse_route = ~recent_route & (positive_count < rules.minimum_positive_demand_months)
        # The current 18-month source cannot provide two annual cycles plus a
        # causal validation holdout, so no seasonal-repeat evidence exists.
        seasonal_route = np.zeros(series_count, dtype=bool)
        unresolved = ~(recent_route | sparse_route | seasonal_route)
        adi = np.divide(
            effective_history_months,
            positive_count,
            out=np.full(series_count, np.inf, dtype=float),
            where=positive_count > 0,
        )
        intermittent = unresolved & (adi >= rules.intermittent_adi_threshold)
        trailing_gap = position - previous_position
        enough_history_for_comparison = effective_history_months >= 12
        if forecast_index >= 12:
            previous_occurrence = np.count_nonzero(
                demand[:, forecast_index - 12 : forecast_index - 6] > 0,
                axis=1,
            )
            recent_occurrence = np.count_nonzero(
                demand[:, forecast_index - 6 : forecast_index] > 0,
                axis=1,
            )
            declining = (
                enough_history_for_comparison
                & (previous_occurrence > 0)
                & (recent_occurrence <= rules.declining_occurrence_ratio * previous_occurrence)
            )
        else:
            declining = np.zeros(series_count, dtype=bool)
        tsb_route = intermittent & (
            (trailing_gap > rules.decay_gap_adi_multiplier * adi) | declining
        )
        positive_mean = np.divide(
            positive_sum,
            positive_count,
            out=np.zeros(series_count, dtype=float),
            where=positive_count > 0,
        )
        positive_variance = np.divide(
            positive_sumsq
            - np.divide(
                positive_sum**2,
                positive_count,
                out=np.zeros(series_count, dtype=float),
                where=positive_count > 0,
            ),
            positive_count - 1,
            out=np.zeros(series_count, dtype=float),
            where=positive_count > 1,
        )
        positive_cv2 = np.divide(
            positive_variance,
            positive_mean**2,
            out=np.zeros(series_count, dtype=float),
            where=positive_mean > 0,
        )
        adida_route = intermittent & ~tsb_route & (positive_cv2 >= rules.lumpy_cv2_threshold)
        croston_route = intermittent & ~(tsb_route | adida_route)
        regular = unresolved & ~intermittent
        if forecast_index >= 12:
            previous_mean = demand[:, forecast_index - 12 : forecast_index - 6].mean(axis=1)
            recent_mean = demand[:, forecast_index - 6 : forecast_index].mean(axis=1)
            relative_change = np.divide(
                np.abs(recent_mean - previous_mean),
                previous_mean,
                out=np.full(series_count, np.inf, dtype=float),
                where=previous_mean > 0,
            )
            material_trend = enough_history_for_comparison & (
                ((previous_mean <= 0) & (recent_mean > 0))
                | (relative_change >= rules.trend_relative_change_threshold)
            )
        else:
            material_trend = np.zeros(series_count, dtype=bool)
        holt_route = regular & material_trend
        ses_route = regular & ~holt_route

        forecast = np.zeros(series_count, dtype=float)
        forecast[recent_route] = moving_average[recent_route]
        trailing_average = demand[
            :,
            max(0, forecast_index - rules.trailing_average_window_months) : forecast_index,
        ].mean(axis=1)
        forecast[sparse_route] = trailing_average[sparse_route]
        forecast[seasonal_route] = demand[
            seasonal_route,
            forecast_index - rules.repeat_history_lookback_months,
        ]
        forecast[tsb_route] = tsb_demand_size[tsb_route] * tsb_probability[tsb_route]
        forecast[adida_route] = _adida_terminal(
            demand,
            forecast_index,
            positive_count,
            effective_history_months,
            adida_route,
            statistics.adida_alpha,
        )[adida_route]
        forecast[croston_route] = croston_forecast[croston_route]
        forecast[ses_route] = ses_level[ses_route]
        forecast[holt_route] = (
            holt_level[holt_route] + statistics.holt_damping * holt_trend[holt_route]
        )

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
