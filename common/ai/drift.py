"""Drift detection helpers: PSI and rolling WAPE.

Gen-4 Stream G / AI-9.

Two numpy-only detectors feed the auto-retrain trigger:

- :func:`compute_psi` — population stability index comparing a baseline
  distribution to a current one. PSI > 0.2 is the standard "material
  drift" cutoff in credit risk / demand planning.

- :func:`rolling_wape` — online WAPE over a sliding window; surfaces
  steady-state degradation that isn't captured by PSI on features.

Both return scalars and do not hit the DB. Callers decide whether to
write a ``fact_drift_signal`` row.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

# PSI < 0.10 = stable, 0.10-0.20 = moderate, > 0.20 = material drift.
_DEFAULT_PSI_BREACH: float = 0.20
# Standard 10-bin PSI — works well for tabular features & model scores.
_DEFAULT_BINS: int = 10
# Tiny epsilon keeps log() finite when a bin has zero mass on one side.
_PSI_EPSILON: float = 1e-6


@dataclass(frozen=True)
class DriftSignal:
    """Wire-compatible with `fact_drift_signal` insert payload."""

    model_id: str
    metric: str
    value: float
    baseline: float | None
    threshold: float | None
    threshold_breached: bool
    window_label: str | None = None
    details: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


def compute_psi(
    baseline: ArrayLike,
    current: ArrayLike,
    *,
    bins: int = _DEFAULT_BINS,
    eps: float = _PSI_EPSILON,
) -> float:
    """Population Stability Index between two 1-D samples.

    Uses baseline-quantile bin edges so the reference distribution always
    sees roughly equal mass per bin. Both samples are clipped to the
    baseline's edges to keep the bin grid finite.

    Args:
        baseline: shape (N,). The reference distribution.
        current:  shape (M,). The observed distribution to compare.
        bins: number of equal-frequency bins (default 10).
        eps: small floor used for empty-bin probabilities.

    Returns:
        PSI as a non-negative float. 0 means identical distributions.
    """
    b = np.asarray(baseline, dtype=float).reshape(-1)
    c = np.asarray(current, dtype=float).reshape(-1)
    if b.size == 0 or c.size == 0:
        raise ValueError("baseline and current must be non-empty arrays")
    if bins < 2:
        raise ValueError("bins must be >= 2")

    # Quantile edges from baseline. Deduplicate in case many ties collapse
    # adjacent quantile boundaries (common for count data).
    edges = np.quantile(b, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        # Distribution is near-constant; PSI is degenerate but we can
        # still return 0 when current matches, else a large value.
        if np.allclose(b.mean(), c.mean()):
            return 0.0
        return float("inf")

    # Extend outer edges so np.histogram doesn't drop samples sitting
    # outside the baseline's range.
    edges[0] = min(edges[0], c.min()) - 1e-9
    edges[-1] = max(edges[-1], c.max()) + 1e-9

    b_counts, _ = np.histogram(b, bins=edges)
    c_counts, _ = np.histogram(c, bins=edges)

    b_prop = np.maximum(b_counts / b.size, eps)
    c_prop = np.maximum(c_counts / c.size, eps)
    psi = float(np.sum((c_prop - b_prop) * np.log(c_prop / b_prop)))
    return psi


def psi_signal(
    model_id: str,
    feature: str,
    baseline: ArrayLike,
    current: ArrayLike,
    *,
    threshold: float = _DEFAULT_PSI_BREACH,
    window_label: str | None = None,
) -> DriftSignal:
    """Compute PSI and wrap it as a DriftSignal ready for `fact_drift_signal`."""
    value = compute_psi(baseline, current)
    return DriftSignal(
        model_id=model_id,
        metric=f"psi_{feature}",
        value=value,
        baseline=0.0,
        threshold=threshold,
        threshold_breached=value > threshold,
        window_label=window_label,
        details={"bins": _DEFAULT_BINS},
    )


# ---------------------------------------------------------------------------
# Rolling WAPE
# ---------------------------------------------------------------------------


def rolling_wape(
    actuals: Sequence[float] | ArrayLike,
    forecasts: Sequence[float] | ArrayLike,
    *,
    window: int,
) -> np.ndarray:
    """Sliding-window WAPE over paired actual/forecast series.

    Returns an array of length ``len(actuals) - window + 1``. Index i
    corresponds to the WAPE of the window ``[i : i + window]``.

    Args:
        actuals, forecasts: 1-D arrays of equal length.
        window: positive integer window size.

    Raises:
        ValueError: when inputs are length-mismatched or window is bad.
    """
    a = np.asarray(actuals, dtype=float).reshape(-1)
    f = np.asarray(forecasts, dtype=float).reshape(-1)
    if a.size != f.size:
        raise ValueError("actuals and forecasts must be the same length")
    if window <= 0:
        raise ValueError("window must be positive")
    if window > a.size:
        raise ValueError(f"window ({window}) larger than series length ({a.size})")

    n_windows = a.size - window + 1
    out = np.empty(n_windows, dtype=float)
    for i in range(n_windows):
        denom = float(np.abs(a[i : i + window]).sum())
        if denom <= 0:
            out[i] = float("nan")
            continue
        num = float(np.abs(f[i : i + window] - a[i : i + window]).sum())
        out[i] = num / denom
    return out


def wape_signal(
    model_id: str,
    actuals: ArrayLike,
    forecasts: ArrayLike,
    *,
    window: int,
    threshold: float,
    window_label: str | None = None,
) -> DriftSignal:
    """Compute the latest rolling WAPE value and wrap as a DriftSignal."""
    values = rolling_wape(actuals, forecasts, window=window)
    latest = float(values[-1]) if values.size else float("nan")
    breached = bool(np.isfinite(latest) and latest > threshold)
    return DriftSignal(
        model_id=model_id,
        metric="rolling_wape",
        value=latest,
        baseline=None,
        threshold=threshold,
        threshold_breached=breached,
        window_label=window_label,
        details={"window": window, "n_windows": int(values.size)},
    )


__all__ = [
    "DriftSignal",
    "compute_psi",
    "psi_signal",
    "rolling_wape",
    "wape_signal",
]
