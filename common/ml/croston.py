"""Configured Croston forecasts for intermittent customer-demand series."""

from __future__ import annotations

from typing import Any

import numpy as np


def croston_forecast(
    history: np.ndarray,
    *,
    horizon: int,
    params: dict[str, Any],
) -> np.ndarray:
    """Return a constant Croston or SBA forecast for one non-negative series."""
    alpha = float(params["alpha"])
    variant = str(params["variant"])
    if not 0.0 < alpha <= 1.0:
        raise ValueError("Croston alpha must be in (0, 1]")
    if variant not in {"classic", "sba"}:
        raise ValueError("Croston variant must be classic or sba")
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
        demand_size += alpha * (float(values[position]) - demand_size)
        demand_interval += alpha * (interval - demand_interval)
        previous_position = position

    forecast = demand_size / demand_interval
    if variant == "sba":
        forecast *= 1.0 - alpha / 2.0
    return np.full(horizon, max(forecast, 0.0), dtype=float)
