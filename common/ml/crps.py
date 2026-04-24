"""Probabilistic-forecast loss functions: CRPS and pinball.

Gen-4 Stream G / AI-2 Phase 1: Replaces WAPE as the champion-selection
metric for models that emit quantile forecasts (Chronos-2, quantile tree
heads, etc.).

Both functions are pure-NumPy — no scipy dependency — and handle the
"quantile matrix" shape ``(n_samples, n_quantiles)`` used by
``fact_candidate_forecast`` quantile reads and the FM spine.

The CRPS here is the **quantile-based approximation** (Laio & Tamea 2007),
equivalent to the integrated pinball loss over the supplied quantile grid.
For a full continuous-distribution CRPS supply a dense quantile grid
(e.g. 99 quantiles from 0.01..0.99) and scale accordingly.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "compute_pinball_loss",
    "compute_crps",
]


def _validate_inputs(
    actuals: ArrayLike,
    forecast_quantiles: ArrayLike,
    quantile_levels: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coerce inputs to numpy and check shape contract."""
    a = np.asarray(actuals, dtype=float).reshape(-1)
    q = np.asarray(forecast_quantiles, dtype=float)
    levels = np.asarray(quantile_levels, dtype=float).reshape(-1)

    if q.ndim == 1:
        # Promote a single forecast's quantile vector to (1, K)
        q = q.reshape(1, -1)
    if q.ndim != 2:
        raise ValueError(
            f"forecast_quantiles must be 1-D or 2-D; got shape {q.shape}"
        )
    if q.shape[0] != a.shape[0]:
        raise ValueError(
            "forecast_quantiles[0] must match actuals length; "
            f"got {q.shape[0]} vs {a.shape[0]}"
        )
    if q.shape[1] != levels.shape[0]:
        raise ValueError(
            "forecast_quantiles[1] must match quantile_levels length; "
            f"got {q.shape[1]} vs {levels.shape[0]}"
        )
    if np.any((levels <= 0.0) | (levels >= 1.0)):
        raise ValueError("quantile_levels must lie strictly in (0, 1)")
    return a, q, levels


def compute_pinball_loss(
    actuals: ArrayLike,
    forecast_quantiles: ArrayLike,
    quantile_levels: ArrayLike,
    *,
    reduce: str = "mean",
) -> float | np.ndarray:
    """Mean pinball (quantile) loss.

    For each observation i and quantile level tau_k with forecast q_ik and
    actual a_i the pinball is:

        L(a, q, tau) = max(tau * (a - q), (tau - 1) * (a - q))

    Args:
        actuals:             shape (N,) or (N, 1)
        forecast_quantiles:  shape (N, K)
        quantile_levels:     shape (K,) values in (0, 1)
        reduce: 'mean' -> scalar; 'none' -> (N, K) per-obs per-quantile loss

    Returns:
        float when reduce='mean', else ndarray shape (N, K).
    """
    a, q, levels = _validate_inputs(actuals, forecast_quantiles, quantile_levels)
    # Broadcast actuals (N,1) against quantile forecasts (N,K)
    diff = a.reshape(-1, 1) - q
    loss = np.maximum(levels * diff, (levels - 1.0) * diff)
    if reduce == "none":
        return loss
    if reduce == "mean":
        return float(np.mean(loss))
    raise ValueError(f"Unknown reduce={reduce!r}; expected 'mean' or 'none'")


def compute_crps(
    actuals: ArrayLike,
    forecast_quantiles: ArrayLike,
    quantile_levels: ArrayLike,
) -> float:
    """Quantile-based CRPS approximation.

    Equivalent to the average pinball loss scaled by 2 over the quantile
    grid — converges to the continuous CRPS as the grid densifies.
    Lower is better.

    Args:
        actuals:             shape (N,)
        forecast_quantiles:  shape (N, K) — each row is the K-quantile
                             forecast for that observation.
        quantile_levels:     shape (K,) quantile levels in (0, 1).

    Returns:
        Mean CRPS across observations.
    """
    per_obs_per_q = compute_pinball_loss(
        actuals, forecast_quantiles, quantile_levels, reduce="none"
    )
    # 2 * mean across quantiles -> CRPS, then mean across observations.
    crps_per_obs = 2.0 * np.mean(per_obs_per_q, axis=1)
    return float(np.mean(crps_per_obs))
