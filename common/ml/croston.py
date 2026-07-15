"""Configured Croston forecasts for intermittent customer-demand series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CrostonParameters:
    """Validated Croston/SBA generation and recursive rollout parameters."""

    alpha: float
    variant: str
    recursive: bool
    recursive_damping: float


def parse_croston_parameters(params: dict[str, Any]) -> CrostonParameters:
    """Validate the complete configured Croston inference contract."""
    try:
        alpha = float(params["alpha"])
        variant = str(params["variant"])
        recursive = params["recursive"]
        recursive_damping = float(params["recursive_damping"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Croston parameters are incomplete") from exc
    if not 0.0 < alpha <= 1.0:
        raise ValueError("Croston alpha must be in (0, 1]")
    if variant not in {"classic", "sba"}:
        raise ValueError("Croston variant must be classic or sba")
    if not isinstance(recursive, bool):
        raise ValueError("Croston recursive setting must be boolean")
    if not np.isfinite(recursive_damping) or not 0.0 <= recursive_damping < 1.0:
        raise ValueError("Croston recursive damping must be in [0, 1)")
    return CrostonParameters(alpha, variant, recursive, recursive_damping)


def croston_forecast(
    history: np.ndarray,
    *,
    horizon: int,
    params: dict[str, Any],
) -> np.ndarray:
    """Return a configured Croston/SBA forecast for one non-negative series."""
    settings = parse_croston_parameters(params)
    if horizon <= 0:
        raise ValueError("Croston horizon must be positive")

    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Croston history must be a finite one-dimensional array")
    if (values < 0).any():
        raise ValueError("Croston history must be non-negative")

    nonzero_positions = np.flatnonzero(values > 0)
    if len(nonzero_positions) == 0:
        return np.zeros(horizon, dtype=float)

    first_position = int(nonzero_positions[0])
    demand_size = float(values[first_position])
    demand_interval = float(first_position + 1)
    previous_position = first_position
    for position_value in nonzero_positions[1:]:
        position = int(position_value)
        interval = float(position - previous_position)
        demand_size += settings.alpha * (float(values[position]) - demand_size)
        demand_interval += settings.alpha * (interval - demand_interval)
        previous_position = position

    forecast = demand_size / demand_interval
    if settings.variant == "sba":
        forecast *= 1.0 - settings.alpha / 2.0
    long_run_rate = max(forecast, 0.0)
    if not settings.recursive:
        return np.full(horizon, long_run_rate, dtype=float)

    # Closed form of F[h] = d*F[h-1] + (1-d)*rate. This preserves the
    # recursive state contract without adding 18 Python iterations for every
    # one of the millions of customer series in a production run.
    horizon_steps = np.arange(1, horizon + 1, dtype=float)
    damping_weights = np.power(settings.recursive_damping, horizon_steps)
    predictions = long_run_rate + damping_weights * (float(values[-1]) - long_run_rate)
    return np.maximum(predictions, 0.0)
