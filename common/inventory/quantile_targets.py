"""Shadow quantile-protection inventory target calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuantileProtectionTarget:
    protection_p50_qty: float
    protection_p90_qty: float
    safety_stock_qty: float
    reorder_point_qty: float
    target_stock_qty: float
    method: str = "quantile_protection_shadow"


def quantile_protection_target(
    *,
    lead_window: tuple[tuple[float, float], ...],
    review_window_p50: tuple[float, ...],
) -> QuantileProtectionTarget:
    """Combine monthly P90 widths under the documented independence approximation."""
    if not lead_window:
        raise ValueError("lead window is required")
    if not review_window_p50:
        raise ValueError("review window is required")
    widths: list[float] = []
    p50_total = 0.0
    for p50, p90 in lead_window:
        if not all(math.isfinite(value) and value >= 0 for value in (p50, p90)):
            raise ValueError("lead-window quantiles must be finite and non-negative")
        if p90 < p50:
            raise ValueError("P90 must be greater than or equal to P50")
        p50_total += p50
        widths.append(p90 - p50)
    if not all(math.isfinite(value) and value >= 0 for value in review_window_p50):
        raise ValueError("review-window P50 values must be finite and non-negative")
    safety_stock = math.sqrt(sum(width**2 for width in widths))
    protection_p90 = p50_total + safety_stock
    return QuantileProtectionTarget(
        protection_p50_qty=p50_total,
        protection_p90_qty=protection_p90,
        safety_stock_qty=safety_stock,
        reorder_point_qty=protection_p90,
        target_stock_qty=protection_p90 + sum(review_window_p50),
    )
