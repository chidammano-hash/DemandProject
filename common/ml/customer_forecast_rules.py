"""Rule selection and deterministic forecasts for customer demand series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from common.ml.croston import CrostonParameters, croston_forecast

CUSTOMER_RULE_ROUTER_MODEL_ID = "customer_rule_router_v2"
MOVING_AVERAGE_ROUTE_ID = "moving_average_3"
TRAILING_AVERAGE_ROUTE_ID = "trailing_average_6"
SEASONAL_REPEAT_ROUTE_ID = "seasonal_repeat_12"
TSB_ROUTE_ID = "tsb"
ADIDA_ROUTE_ID = "adida"
CROSTON_ROUTE_ID = "croston"
SES_ROUTE_ID = "ses"
HOLT_DAMPED_ROUTE_ID = "holt_damped"
CUSTOMER_FORECAST_ROUTE_IDS = (
    MOVING_AVERAGE_ROUTE_ID,
    TRAILING_AVERAGE_ROUTE_ID,
    SEASONAL_REPEAT_ROUTE_ID,
    TSB_ROUTE_ID,
    ADIDA_ROUTE_ID,
    CROSTON_ROUTE_ID,
    SES_ROUTE_ID,
    HOLT_DAMPED_ROUTE_ID,
)

_CANONICAL_RECENT_DEMAND_LOOKBACK_MONTHS = 6
_CANONICAL_MOVING_AVERAGE_WINDOW_MONTHS = 3
_CANONICAL_TRAILING_AVERAGE_WINDOW_MONTHS = 6
_CANONICAL_REPEAT_HISTORY_LOOKBACK_MONTHS = 12

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class CustomerForecastRuleParameters:
    """Validated thresholds for the ordered customer forecast rules."""

    recent_demand_lookback_months: int
    moving_average_window_months: int
    trailing_average_window_months: int
    minimum_positive_demand_months: int
    repeat_history_lookback_months: int
    seasonal_min_history_months: int
    seasonal_min_wape_improvement_pct: float
    intermittent_adi_threshold: float
    lumpy_cv2_threshold: float
    decay_gap_adi_multiplier: float
    declining_occurrence_ratio: float
    trend_relative_change_threshold: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "recent_demand_lookback_months": self.recent_demand_lookback_months,
            "moving_average_window_months": self.moving_average_window_months,
            "trailing_average_window_months": self.trailing_average_window_months,
            "minimum_positive_demand_months": self.minimum_positive_demand_months,
            "repeat_history_lookback_months": self.repeat_history_lookback_months,
            "seasonal_min_history_months": self.seasonal_min_history_months,
            "seasonal_min_wape_improvement_pct": (self.seasonal_min_wape_improvement_pct),
            "intermittent_adi_threshold": self.intermittent_adi_threshold,
            "lumpy_cv2_threshold": self.lumpy_cv2_threshold,
            "decay_gap_adi_multiplier": self.decay_gap_adi_multiplier,
            "declining_occurrence_ratio": self.declining_occurrence_ratio,
            "trend_relative_change_threshold": self.trend_relative_change_threshold,
        }


@dataclass(frozen=True)
class CustomerStatisticalParameters:
    """Configured smoothing parameters for customer-only statistical routes."""

    tsb_demand_alpha: float
    tsb_probability_alpha: float
    adida_alpha: float
    ses_alpha: float
    holt_level_alpha: float
    holt_trend_alpha: float
    holt_damping: float

    def as_dict(self) -> dict[str, float]:
        return {
            "tsb_demand_alpha": self.tsb_demand_alpha,
            "tsb_probability_alpha": self.tsb_probability_alpha,
            "adida_alpha": self.adida_alpha,
            "ses_alpha": self.ses_alpha,
            "holt_level_alpha": self.holt_level_alpha,
            "holt_trend_alpha": self.holt_trend_alpha,
            "holt_damping": self.holt_damping,
        }


def parse_customer_forecast_rule_parameters(
    raw: dict[str, Any],
) -> CustomerForecastRuleParameters:
    """Parse the complete YAML-backed routing parameter contract."""
    try:
        parsed = CustomerForecastRuleParameters(
            recent_demand_lookback_months=int(raw["recent_demand_lookback_months"]),
            moving_average_window_months=int(raw["moving_average_window_months"]),
            trailing_average_window_months=int(raw["trailing_average_window_months"]),
            minimum_positive_demand_months=int(raw["minimum_positive_demand_months"]),
            repeat_history_lookback_months=int(raw["repeat_history_lookback_months"]),
            seasonal_min_history_months=int(raw["seasonal_min_history_months"]),
            seasonal_min_wape_improvement_pct=float(raw["seasonal_min_wape_improvement_pct"]),
            intermittent_adi_threshold=float(raw["intermittent_adi_threshold"]),
            lumpy_cv2_threshold=float(raw["lumpy_cv2_threshold"]),
            decay_gap_adi_multiplier=float(raw["decay_gap_adi_multiplier"]),
            declining_occurrence_ratio=float(raw["declining_occurrence_ratio"]),
            trend_relative_change_threshold=float(raw["trend_relative_change_threshold"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Customer forecast rule parameters are incomplete") from exc
    numeric_values = np.asarray(
        [
            parsed.intermittent_adi_threshold,
            parsed.lumpy_cv2_threshold,
            parsed.decay_gap_adi_multiplier,
            parsed.declining_occurrence_ratio,
            parsed.trend_relative_change_threshold,
            parsed.seasonal_min_wape_improvement_pct,
        ],
        dtype=np.float64,
    )
    if (
        parsed.recent_demand_lookback_months != _CANONICAL_RECENT_DEMAND_LOOKBACK_MONTHS
        or parsed.moving_average_window_months != _CANONICAL_MOVING_AVERAGE_WINDOW_MONTHS
        or parsed.trailing_average_window_months != _CANONICAL_TRAILING_AVERAGE_WINDOW_MONTHS
        or parsed.minimum_positive_demand_months < 2
        or parsed.repeat_history_lookback_months != _CANONICAL_REPEAT_HISTORY_LOOKBACK_MONTHS
        or parsed.seasonal_min_history_months < 2 * parsed.repeat_history_lookback_months
        or not 0.0 <= parsed.seasonal_min_wape_improvement_pct <= 100.0
        or not np.isfinite(numeric_values).all()
        or parsed.intermittent_adi_threshold <= 1.0
        or parsed.lumpy_cv2_threshold < 0.0
        or parsed.decay_gap_adi_multiplier <= 0.0
        or not 0.0 < parsed.declining_occurrence_ratio <= 1.0
        or parsed.trend_relative_change_threshold < 0.0
    ):
        raise ValueError("Customer forecast rule parameters are invalid")
    return parsed


def parse_customer_statistical_parameters(
    raw: dict[str, Any],
) -> CustomerStatisticalParameters:
    """Parse smoothing settings for every non-Croston statistical route."""
    try:
        parsed = CustomerStatisticalParameters(
            tsb_demand_alpha=float(raw["tsb_demand_alpha"]),
            tsb_probability_alpha=float(raw["tsb_probability_alpha"]),
            adida_alpha=float(raw["adida_alpha"]),
            ses_alpha=float(raw["ses_alpha"]),
            holt_level_alpha=float(raw["holt_level_alpha"]),
            holt_trend_alpha=float(raw["holt_trend_alpha"]),
            holt_damping=float(raw["holt_damping"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Customer statistical parameters are incomplete") from exc
    alphas = np.asarray(
        [
            parsed.tsb_demand_alpha,
            parsed.tsb_probability_alpha,
            parsed.adida_alpha,
            parsed.ses_alpha,
            parsed.holt_level_alpha,
            parsed.holt_trend_alpha,
        ],
        dtype=np.float64,
    )
    if (
        not np.isfinite(alphas).all()
        or (alphas <= 0.0).any()
        or (alphas > 1.0).any()
        or not np.isfinite(parsed.holt_damping)
        or not 0.0 < parsed.holt_damping < 1.0
    ):
        raise ValueError("Customer statistical parameters are invalid")
    return parsed


def _validated_history(history: np.ndarray) -> FloatArray:
    values: FloatArray = np.asarray(history, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Customer forecast history must be a non-empty vector")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Customer forecast history must be finite and non-negative")
    return values


def _trailing_zero_months(values: FloatArray) -> int:
    positive_positions = np.flatnonzero(values > 0.0)
    if positive_positions.size == 0:
        return int(values.size)
    return int(values.size - int(positive_positions[-1]) - 1)


def _positive_demand_cv2(positive_values: FloatArray) -> float:
    if positive_values.size < 2:
        return 0.0
    mean = float(np.mean(positive_values))
    if mean <= 0.0:
        return 0.0
    return float(np.var(positive_values, ddof=1) / (mean * mean))


def _occurrence_is_declining(values: FloatArray, ratio: float) -> bool:
    if values.size < 12:
        return False
    previous_count = int(np.count_nonzero(values[-12:-6] > 0.0))
    recent_count = int(np.count_nonzero(values[-6:] > 0.0))
    return previous_count > 0 and recent_count <= ratio * previous_count


def _has_material_trend(values: FloatArray, threshold: float) -> bool:
    if values.size < 12:
        return False
    previous_mean = float(np.mean(values[-12:-6]))
    recent_mean = float(np.mean(values[-6:]))
    if previous_mean <= 0.0:
        return recent_mean > 0.0
    return abs(recent_mean - previous_mean) / previous_mean >= threshold


def select_customer_forecast_route(
    history: np.ndarray,
    *,
    demand_started_within_recent_window: bool,
    params: CustomerForecastRuleParameters,
    seasonal_repeat_validated: bool = False,
    effective_history_months: int | None = None,
) -> str:
    """Select one customer-only route from causal history using fixed precedence."""
    values = _validated_history(history)
    if effective_history_months is not None:
        if not 0 < effective_history_months <= values.size:
            raise ValueError("Effective customer history months are invalid")
        values = values[-effective_history_months:]
    if demand_started_within_recent_window:
        return MOVING_AVERAGE_ROUTE_ID

    positive_values = values[values > 0.0]
    positive_count = int(positive_values.size)
    if positive_count < params.minimum_positive_demand_months:
        return TRAILING_AVERAGE_ROUTE_ID

    if values.size >= params.seasonal_min_history_months and seasonal_repeat_validated:
        return SEASONAL_REPEAT_ROUTE_ID

    adi = float(values.size / positive_count)
    if adi >= params.intermittent_adi_threshold:
        decaying = _trailing_zero_months(
            values
        ) > params.decay_gap_adi_multiplier * adi or _occurrence_is_declining(
            values, params.declining_occurrence_ratio
        )
        if decaying:
            return TSB_ROUTE_ID
        if _positive_demand_cv2(positive_values) >= params.lumpy_cv2_threshold:
            return ADIDA_ROUTE_ID
        return CROSTON_ROUTE_ID

    if _has_material_trend(values, params.trend_relative_change_threshold):
        return HOLT_DAMPED_ROUTE_ID
    return SES_ROUTE_ID


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


def _trailing_average(
    history: FloatArray,
    *,
    horizon: int,
    window_months: int,
) -> FloatArray:
    return np.full(horizon, float(np.mean(history[-window_months:])), dtype=np.float64)


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


def _simple_exponential_smoothing(
    history: FloatArray,
    *,
    horizon: int,
    alpha: float,
) -> FloatArray:
    level = float(history[0])
    for observation in history[1:]:
        level += alpha * (float(observation) - level)
    return np.full(horizon, max(level, 0.0), dtype=np.float64)


def _tsb_forecast(
    history: FloatArray,
    *,
    horizon: int,
    demand_alpha: float,
    probability_alpha: float,
) -> FloatArray:
    positive_positions = np.flatnonzero(history > 0.0)
    if positive_positions.size == 0:
        return np.zeros(horizon, dtype=np.float64)
    demand_size = float(history[int(positive_positions[0])])
    probability = 1.0
    initialized = False
    for observation in history:
        occurred = float(observation) > 0.0
        if occurred:
            if initialized:
                demand_size += demand_alpha * (float(observation) - demand_size)
            initialized = True
        if initialized:
            probability += probability_alpha * (float(occurred) - probability)
    horizon_steps = np.arange(horizon, dtype=np.float64)
    occurrence = probability * np.power(1.0 - probability_alpha, horizon_steps)
    return np.maximum(demand_size * occurrence, 0.0)


def _adida_forecast(
    history: FloatArray,
    *,
    horizon: int,
    alpha: float,
) -> FloatArray:
    positive_count = int(np.count_nonzero(history > 0.0))
    if positive_count == 0:
        return np.zeros(horizon, dtype=np.float64)
    aggregation_level = max(1, round(history.size / positive_count))
    remainder = history.size % aggregation_level
    padded = np.pad(history, (aggregation_level - remainder, 0)) if remainder else history
    aggregated = padded.reshape(-1, aggregation_level).sum(axis=1)
    block_level = float(_simple_exponential_smoothing(aggregated, horizon=1, alpha=alpha)[0])
    return np.full(horizon, max(block_level / aggregation_level, 0.0), dtype=np.float64)


def _damped_holt_forecast(
    history: FloatArray,
    *,
    horizon: int,
    level_alpha: float,
    trend_alpha: float,
    damping: float,
) -> FloatArray:
    if history.size == 1:
        return np.full(horizon, float(history[0]), dtype=np.float64)
    level = float(history[0])
    trend = float(history[1] - history[0])
    for observation in history[1:]:
        previous_level = level
        previous_trend = trend
        level = level_alpha * float(observation) + (1.0 - level_alpha) * (
            previous_level + damping * previous_trend
        )
        trend = trend_alpha * (level - previous_level) + (1.0 - trend_alpha) * (
            damping * previous_trend
        )
    steps = np.arange(1, horizon + 1, dtype=np.float64)
    damped_sum = damping * (1.0 - np.power(damping, steps)) / (1.0 - damping)
    return np.maximum(level + trend * damped_sum, 0.0)


def forecast_customer_demand(
    history: np.ndarray,
    *,
    horizon: int,
    route_model_id: str,
    rule_params: CustomerForecastRuleParameters,
    croston_params: dict[str, Any] | CrostonParameters,
    statistical_params: dict[str, Any] | CustomerStatisticalParameters,
    effective_history_months: int | None = None,
) -> FloatArray:
    """Generate one customer forecast with its causally selected route."""
    values = _validated_history(history)
    model_values = values
    if effective_history_months is not None:
        if not 0 < effective_history_months <= values.size:
            raise ValueError("Effective customer history months are invalid")
        model_values = values[-effective_history_months:]
    if horizon <= 0:
        raise ValueError("Customer forecast horizon must be positive")
    statistics = (
        statistical_params
        if isinstance(statistical_params, CustomerStatisticalParameters)
        else parse_customer_statistical_parameters(statistical_params)
    )
    if route_model_id == MOVING_AVERAGE_ROUTE_ID:
        return _recursive_moving_average(
            values,
            horizon=horizon,
            window_months=rule_params.moving_average_window_months,
        )
    if route_model_id == TRAILING_AVERAGE_ROUTE_ID:
        return _trailing_average(
            values,
            horizon=horizon,
            window_months=rule_params.trailing_average_window_months,
        )
    if route_model_id == SEASONAL_REPEAT_ROUTE_ID:
        return _seasonal_repeat(
            values,
            horizon=horizon,
            lookback_months=rule_params.repeat_history_lookback_months,
        )
    if route_model_id == TSB_ROUTE_ID:
        return _tsb_forecast(
            model_values,
            horizon=horizon,
            demand_alpha=statistics.tsb_demand_alpha,
            probability_alpha=statistics.tsb_probability_alpha,
        )
    if route_model_id == ADIDA_ROUTE_ID:
        return _adida_forecast(
            model_values,
            horizon=horizon,
            alpha=statistics.adida_alpha,
        )
    if route_model_id == CROSTON_ROUTE_ID:
        forecast: FloatArray = croston_forecast(
            model_values,
            horizon=horizon,
            params=croston_params,
        )
        return forecast
    if route_model_id == SES_ROUTE_ID:
        return _simple_exponential_smoothing(
            model_values,
            horizon=horizon,
            alpha=statistics.ses_alpha,
        )
    if route_model_id == HOLT_DAMPED_ROUTE_ID:
        return _damped_holt_forecast(
            model_values,
            horizon=horizon,
            level_alpha=statistics.holt_level_alpha,
            trend_alpha=statistics.holt_trend_alpha,
            damping=statistics.holt_damping,
        )
    raise ValueError(f"Unknown customer forecast route: {route_model_id}")
