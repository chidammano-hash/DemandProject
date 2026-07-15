"""Rule selection and deterministic forecasts for customer demand series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from common.ml.croston import croston_forecast

CUSTOMER_RULE_ROUTER_MODEL_ID = "customer_rule_router"
MOVING_AVERAGE_ROUTE_ID = "moving_average_3"
SEASONAL_REPEAT_ROUTE_ID = "seasonal_repeat_12"
CROSTON_ROUTE_ID = "croston"
CUSTOMER_FORECAST_ROUTE_IDS = (
    MOVING_AVERAGE_ROUTE_ID,
    SEASONAL_REPEAT_ROUTE_ID,
    CROSTON_ROUTE_ID,
)

_CANONICAL_MOVING_AVERAGE_WINDOW_MONTHS = 3
_CANONICAL_REPEAT_HISTORY_LOOKBACK_MONTHS = 12

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class CustomerForecastRuleParameters:
    """Validated thresholds for the ordered customer forecast rules."""

    recent_demand_lookback_months: int
    moving_average_window_months: int
    repeat_history_lookback_months: int
    repeat_history_min_demand_months: int

    def as_dict(self) -> dict[str, int]:
        return {
            "recent_demand_lookback_months": self.recent_demand_lookback_months,
            "moving_average_window_months": self.moving_average_window_months,
            "repeat_history_lookback_months": self.repeat_history_lookback_months,
            "repeat_history_min_demand_months": self.repeat_history_min_demand_months,
        }


def parse_customer_forecast_rule_parameters(
    raw: dict[str, Any],
) -> CustomerForecastRuleParameters:
    """Parse the complete YAML-backed rule parameter contract."""
    try:
        parsed = CustomerForecastRuleParameters(
            recent_demand_lookback_months=int(raw["recent_demand_lookback_months"]),
            moving_average_window_months=int(raw["moving_average_window_months"]),
            repeat_history_lookback_months=int(raw["repeat_history_lookback_months"]),
            repeat_history_min_demand_months=int(raw["repeat_history_min_demand_months"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Customer forecast rule parameters are incomplete") from exc
    if (
        parsed.recent_demand_lookback_months <= 0
        or parsed.moving_average_window_months != _CANONICAL_MOVING_AVERAGE_WINDOW_MONTHS
        or parsed.moving_average_window_months > parsed.recent_demand_lookback_months
        or parsed.repeat_history_lookback_months != _CANONICAL_REPEAT_HISTORY_LOOKBACK_MONTHS
        or parsed.repeat_history_min_demand_months <= 0
        or parsed.repeat_history_min_demand_months > parsed.repeat_history_lookback_months
    ):
        raise ValueError("Customer forecast rule parameters are invalid")
    return parsed


def _validated_history(history: np.ndarray) -> FloatArray:
    values: FloatArray = np.asarray(history, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Customer forecast history must be a non-empty vector")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Customer forecast history must be finite and non-negative")
    return values


def select_customer_forecast_route(
    history: np.ndarray,
    *,
    demand_started_within_recent_window: bool,
    params: CustomerForecastRuleParameters,
) -> str:
    """Select one route using the configured precedence and causal history."""
    values = _validated_history(history)
    if demand_started_within_recent_window:
        return MOVING_AVERAGE_ROUTE_ID
    if values.size < params.repeat_history_lookback_months:
        return CROSTON_ROUTE_ID
    trailing = values[-params.repeat_history_lookback_months :]
    if int(np.count_nonzero(trailing > 0)) >= params.repeat_history_min_demand_months:
        return SEASONAL_REPEAT_ROUTE_ID
    return CROSTON_ROUTE_ID


def _recursive_moving_average(
    history: FloatArray,
    *,
    horizon: int,
    window_months: int,
) -> FloatArray:
    state = history.tolist()
    forecast: FloatArray = np.empty(horizon, dtype=np.float64)
    for offset in range(horizon):
        value = float(np.mean(state[-window_months:]))
        forecast[offset] = value
        state.append(value)
    return forecast


def _seasonal_repeat(
    history: FloatArray,
    *,
    horizon: int,
    lookback_months: int,
) -> FloatArray:
    if history.size < lookback_months:
        raise ValueError("Seasonal repeat needs a complete history cycle")
    cycle = history[-lookback_months:]
    forecast: FloatArray = np.resize(cycle, horizon).astype(np.float64, copy=False)
    return forecast


def forecast_customer_demand(
    history: np.ndarray,
    *,
    horizon: int,
    route_model_id: str,
    rule_params: CustomerForecastRuleParameters,
    croston_params: dict[str, Any],
) -> FloatArray:
    """Generate one customer forecast with the route selected from causal history."""
    values = _validated_history(history)
    if horizon <= 0:
        raise ValueError("Customer forecast horizon must be positive")
    if route_model_id == MOVING_AVERAGE_ROUTE_ID:
        return _recursive_moving_average(
            values,
            horizon=horizon,
            window_months=rule_params.moving_average_window_months,
        )
    if route_model_id == SEASONAL_REPEAT_ROUTE_ID:
        return _seasonal_repeat(
            values,
            horizon=horizon,
            lookback_months=rule_params.repeat_history_lookback_months,
        )
    if route_model_id == CROSTON_ROUTE_ID:
        forecast: FloatArray = croston_forecast(
            values,
            horizon=horizon,
            params=croston_params,
        )
        return forecast
    raise ValueError(f"Unknown customer forecast route: {route_model_id}")
