"""Revised demand-sensing volatility score.

Self-contained, dependency-light (numpy/pandas only) implementation of the
volatility / confidence gate used to decide whether the MTD-vs-LY consumption
comparison is allowed to auto-adjust a DFU's forecast.

Design notes (see the accompanying PDF for full rationale):

1. ADI and ZeroShare are mathematically locked: ADI = 1 / (1 - ZeroShare).
   Keeping both double-counts intermittency, so the score here uses ONE
   intermittency term (ADI) and drops ZeroShare from the weighted sum.
2. SpikeRatio uses a robust p95/median instead of max/mean (max is fragile to
   a single outlier and grows with sample size).
3. The CV2 ramp top is widened from 0.49 -> 1.0 so the subscore discriminates
   across the erratic range instead of saturating immediately.
4. Small-sample guard: < MIN_NONZERO non-zero periods -> not scorable, route
   to manual review (low confidence regardless of computed volatility).
5. Seasonality guard: strongly-seasonal-but-predictable DFUs are flagged so the
   pipeline can deseasonalize / exempt them rather than punishing structure.

Subscore convention: 0 = stable, 1 = highly volatile. Final score in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# --- Tunable constants (calibrate against backtested forecast error) ----------
ADI_LO, ADI_HI = 1.32, 6.0          # Syntetos-Boylan intermittency boundaries
CV2_LO, CV2_HI = 0.25, 1.0          # widened top vs the SBC 0.49 cutoff
SPIKE_LO, SPIKE_HI = 1.0, 2.0       # p95/median ramp
MIN_NONZERO = 3                     # below this, CV2/spike are unstable

# Revised weights (ZeroShare dropped as redundant with ADI)
W_INTERMITTENCY = 0.40
W_CV2 = 0.35
W_SPIKE = 0.25

# Risk-band cut points (UAT default; re-derive from backtest)
LOW_MAX = 0.33
MED_MAX = 0.66

# Seasonality: fraction of variance explained by calendar-month means
SEASONAL_STRENGTH_EXEMPT = 0.60
SEASONAL_MIN_MONTHS = 24


def _clamp_ramp(x: float, lo: float, hi: float) -> float:
    """Linear ramp from 0 at lo to 1 at hi, clamped to [0, 1]."""
    if hi == lo:
        return 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def seasonal_strength(series: np.ndarray, period: int = 12) -> float:
    """Fraction of variance explained by calendar-month means (0..1).

    Lightweight stand-in for an STL seasonal-strength measure: high values mean
    the series is well explained by a repeating month-of-year pattern, i.e.
    predictable structure that should NOT be penalised as volatility.
    """
    n = len(series)
    if n < period * 2:
        return 0.0
    total_var = np.var(series)
    if total_var == 0:
        return 0.0
    idx = np.arange(n) % period
    month_means = np.array([series[idx == m].mean() for m in range(period)])
    seasonal = month_means[idx]
    resid_var = np.var(series - seasonal)
    return float(max(0.0, 1.0 - resid_var / total_var))


@dataclass
class VolatilityResult:
    n_periods: int
    n_nonzero: int
    adi: float
    zero_share: float            # reported for transparency, NOT in the score
    cv2: float
    spike_ratio: float
    seasonal_strength: float
    intermittency_sub: float
    cv2_sub: float
    spike_sub: float
    score: float                 # NaN when not scorable
    risk_band: str               # low | medium | high | manual_review
    scorable: bool
    seasonal_exempt: bool

    def as_dict(self) -> dict:
        return asdict(self)


def volatility_score(monthly_qty) -> VolatilityResult:
    """Compute the revised volatility score for one DFU's monthly demand series.

    Args:
        monthly_qty: 1-D sequence of per-month demand (include zero months).

    Returns:
        VolatilityResult. score is NaN and risk_band='manual_review' when the
        DFU lacks enough non-zero history to score reliably.
    """
    q = np.asarray(monthly_qty, dtype=float)
    n_periods = int(q.size)
    nz = q[q > 0]
    n_nonzero = int(nz.size)

    zero_share = float((n_periods - n_nonzero) / n_periods) if n_periods else 1.0
    adi = float(n_periods / n_nonzero) if n_nonzero else float(n_periods or 1)

    seas = seasonal_strength(q) if n_periods >= SEASONAL_MIN_MONTHS else 0.0
    seasonal_exempt = seas >= SEASONAL_STRENGTH_EXEMPT

    # --- Small-sample guard: not enough non-zero points to trust CV2/spike ----
    if n_nonzero < MIN_NONZERO:
        return VolatilityResult(
            n_periods=n_periods, n_nonzero=n_nonzero, adi=adi,
            zero_share=zero_share, cv2=float("nan"), spike_ratio=float("nan"),
            seasonal_strength=seas, intermittency_sub=float("nan"),
            cv2_sub=float("nan"), spike_sub=float("nan"), score=float("nan"),
            risk_band="manual_review", scorable=False,
            seasonal_exempt=seasonal_exempt,
        )

    mean_nz = nz.mean()
    std_nz = nz.std(ddof=1)
    cv2 = float((std_nz / mean_nz) ** 2) if mean_nz else 0.0

    median_nz = np.median(nz)
    p95_nz = np.percentile(nz, 95)
    spike_ratio = float(p95_nz / median_nz) if median_nz else 1.0

    intermittency_sub = _clamp_ramp(adi, ADI_LO, ADI_HI)
    cv2_sub = _clamp_ramp(cv2, CV2_LO, CV2_HI)
    spike_sub = _clamp_ramp(spike_ratio, SPIKE_LO, SPIKE_HI)

    score = (
        W_INTERMITTENCY * intermittency_sub
        + W_CV2 * cv2_sub
        + W_SPIKE * spike_sub
    )

    if seasonal_exempt:
        band = "low"            # predictable seasonality -> eligible for automation
    elif score <= LOW_MAX:
        band = "low"
    elif score <= MED_MAX:
        band = "medium"
    else:
        band = "high"

    return VolatilityResult(
        n_periods=n_periods, n_nonzero=n_nonzero, adi=adi,
        zero_share=zero_share, cv2=cv2, spike_ratio=spike_ratio,
        seasonal_strength=seas, intermittency_sub=intermittency_sub,
        cv2_sub=cv2_sub, spike_sub=spike_sub, score=float(score),
        risk_band=band, scorable=True, seasonal_exempt=seasonal_exempt,
    )


# --- Original spec (for side-by-side comparison in the report) ----------------
def original_score(monthly_qty) -> float:
    """The as-proposed score: 0.25*(ADI_sub + CV2_sub + ZeroShare + Spike_sub),
    with max/mean spike and the narrow 0.25-0.49 CV2 band. Returns NaN if no
    non-zero demand."""
    q = np.asarray(monthly_qty, dtype=float)
    n = q.size
    nz = q[q > 0]
    if nz.size == 0:
        return float("nan")
    adi = n / nz.size
    zero_share = (n - nz.size) / n
    mean_nz = nz.mean()
    cv2 = (nz.std(ddof=1) / mean_nz) ** 2 if nz.size > 1 and mean_nz else 0.0
    spike = nz.max() / mean_nz if mean_nz else 1.0
    adi_sub = _clamp_ramp(adi, 1.32, 6.0)
    cv2_sub = _clamp_ramp(cv2, 0.25, 0.49)
    spike_sub = _clamp_ramp(spike, 1.0, 2.0)
    return 0.25 * (adi_sub + cv2_sub + zero_share + spike_sub)
