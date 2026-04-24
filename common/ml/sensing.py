"""Near-term demand sensing — horizon-weighted blend of short vs long forecasts.

Gen-4 Roadmap AI-3. The planner carries two independent forecasts:

    * ``long_range``  — tree or foundation-model output, good beyond ~2 weeks
    * ``near_term``   — POS-driven sensing model, good in the 0-14 day window

``blend_forecasts`` interpolates a horizon-dependent weight between the
two. For short horizons the near-term signal dominates; for long
horizons the long-range signal dominates; between the configured
``short_horizon_days`` and ``long_horizon_days`` the near-term weight
decays linearly.

Config lives in ``config/sensing_config.yaml``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _near_term_weight(
    horizon_days_out: int,
    *,
    short_horizon_days: int,
    long_horizon_days: int,
    short_weight: float,
    long_weight: float,
    min_near_term_weight: float,
) -> float:
    """Return the weight to apply to the near-term forecast.

    Inside the [short_horizon_days, long_horizon_days] band the weight
    decays linearly from ``short_weight`` to ``(1 - long_weight)``. Outside
    the band the endpoint values apply verbatim (subject to
    ``min_near_term_weight``).
    """
    if short_horizon_days >= long_horizon_days:
        raise ValueError("short_horizon_days must be < long_horizon_days")

    # Near-term weight at the long-horizon endpoint is the residual
    # after the long-range forecast takes its configured weight.
    long_side_near_term = max(1.0 - long_weight, 0.0)

    if horizon_days_out <= short_horizon_days:
        weight = short_weight
    elif horizon_days_out >= long_horizon_days:
        weight = long_side_near_term
    else:
        span = long_horizon_days - short_horizon_days
        t = (horizon_days_out - short_horizon_days) / span
        weight = short_weight + t * (long_side_near_term - short_weight)

    return max(min_near_term_weight, min(1.0, weight))


def blend_forecasts(
    long_range: float,
    near_term: float,
    horizon_days_out: int,
    weights_config: dict[str, Any],
) -> float:
    """Blend ``long_range`` and ``near_term`` by horizon-dependent weight.

    Args:
        long_range: prediction from the long-range model.
        near_term:  prediction from the sensing model.
        horizon_days_out: integer days between "now" and the target day.
        weights_config: dict loaded from ``config/sensing_config.yaml``.

    Returns:
        The blended point forecast (``w * near_term + (1-w) * long_range``).
    """
    if horizon_days_out < 0:
        raise ValueError("horizon_days_out must be >= 0")

    short_weight = float(weights_config.get("short_weight_horizon_0_7", 0.8))
    long_weight = float(weights_config.get("long_weight_horizon_30_plus", 0.95))
    short_horizon_days = int(weights_config.get("short_horizon_days", 7))
    long_horizon_days = int(weights_config.get("long_horizon_days", 30))
    min_near_term_weight = float(weights_config.get("min_near_term_weight", 0.0))

    w_near = _near_term_weight(
        horizon_days_out,
        short_horizon_days=short_horizon_days,
        long_horizon_days=long_horizon_days,
        short_weight=short_weight,
        long_weight=long_weight,
        min_near_term_weight=min_near_term_weight,
    )
    blended = w_near * float(near_term) + (1.0 - w_near) * float(long_range)
    logger.debug(
        "blend_forecasts horizon=%d w_near=%.3f long=%.3f near=%.3f -> %.3f",
        horizon_days_out, w_near, long_range, near_term, blended,
    )
    return blended


__all__ = ["blend_forecasts"]
