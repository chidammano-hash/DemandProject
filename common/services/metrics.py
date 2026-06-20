"""Shared forecast accuracy metrics (WAPE, bias, accuracy).

Gen-4 Roadmap SC-10 (P2): the three historical call-sites (dashboard KPI,
FVA reporting, backtest framework) diverged on accuracy denominators. This
module is the canonical source; new code MUST call ``compute_accuracy`` /
``compute_bias_pct`` / ``compute_wape`` below, and existing call-sites should
migrate with a TODO marker pointing here.

Canonical formulas:
    WAPE         = SUM(|F - A|) / NULLIF(ABS(SUM(A)), 0)
    accuracy_pct = max(0, 100 * (1 - WAPE))
    bias_pct     = 100 * (SUM(F) / NULLIF(SUM(A), 0) - 1)

For SQL-level usage, use ``ACCURACY_SQL_TEMPLATE``.
"""

import statistics
from collections.abc import Sequence
from typing import Any

import pandas as pd

# SQL template — callers should standardize on this exact expression.
# Usage:
#   sql = ACCURACY_SQL_TEMPLATE.format(forecast="forecast_qty", actual="actual_qty")
ACCURACY_SQL_TEMPLATE = (
    "CASE WHEN ABS(SUM({actual})) > 0 "
    "THEN GREATEST(0.0, 100.0 - 100.0 * SUM(ABS({forecast} - {actual})) "
    "/ ABS(SUM({actual}))) "
    "ELSE NULL END"
)


def compute_accuracy(
    actuals: Sequence[float],
    forecasts: Sequence[float],
) -> float | None:
    """Canonical accuracy metric = 100 * (1 - WAPE), clamped to [0, 100].

    Returns None when the absolute sum of actuals is zero (undefined).

    Raises:
        ValueError: when sequences differ in length.
    """
    if len(actuals) != len(forecasts):
        raise ValueError(
            f"Length mismatch: actuals={len(actuals)}, forecasts={len(forecasts)}"
        )
    if not actuals:
        return None

    abs_sum_actual = abs(sum(actuals))
    if abs_sum_actual == 0:
        return None

    sum_abs_error = sum(abs(f - a) for f, a in zip(forecasts, actuals))
    wape = sum_abs_error / abs_sum_actual
    return max(0.0, 100.0 * (1.0 - wape))


def compute_bias_pct(
    actuals: Sequence[float],
    forecasts: Sequence[float],
) -> float | None:
    """Canonical bias = (SUM(F) / SUM(A) - 1) * 100. Returns None when SUM(A) = 0."""
    if len(actuals) != len(forecasts):
        raise ValueError(
            f"Length mismatch: actuals={len(actuals)}, forecasts={len(forecasts)}"
        )
    sum_a = sum(actuals)
    if sum_a == 0:
        return None
    return (sum(forecasts) / sum_a - 1.0) * 100.0


def compute_wape(
    actuals: Sequence[float],
    forecasts: Sequence[float],
) -> float | None:
    """WAPE as a fraction (not percentage). Returns None when SUM(|A|) = 0."""
    if len(actuals) != len(forecasts):
        raise ValueError(
            f"Length mismatch: actuals={len(actuals)}, forecasts={len(forecasts)}"
        )
    if not actuals:
        return None
    abs_sum = abs(sum(actuals))
    if abs_sum == 0:
        return None
    return sum(abs(f - a) for f, a in zip(forecasts, actuals)) / abs_sum


def compute_unweighted_accuracy(
    per_dfu: Sequence[tuple[float, float]],
) -> dict[str, float | int | None]:
    """Unweighted per-DFU accuracy: WAPE per DFU, then mean/median across DFUs.

    The headline accuracy (``compute_accuracy`` / ``compute_kpis``) is
    *volume-weighted* — it sums error across every DFU before dividing, so a few
    high-volume SKUs dominate. This function instead scores each DFU on its own
    and averages, giving every DFU equal say and exposing the long tail.

    Each element of ``per_dfu`` is ``(sum_actual, sum_abs_error)`` for one DFU,
    already aggregated over that DFU's months. Per-DFU accuracy uses the canonical
    formula ``max(0, 100 * (1 - sum_abs_error / abs(sum_actual)))``.

    A DFU with ``sum_actual == 0`` has an undefined WAPE (zero denominator). Such
    DFUs are EXCLUDED from the mean/median and reported in ``n_undefined`` rather
    than counted as 0% — folding clamped zeros into the average would overstate the
    long-tail problem and conflate "no demand" with "badly forecast".

    Returns a dict with keys ``n_dfus``, ``n_undefined``, ``mean_accuracy_pct``,
    ``median_accuracy_pct``. Mean/median are ``None`` when every DFU is undefined.
    """
    accuracies: list[float] = []
    n_undefined = 0
    for sum_actual, sum_abs_error in per_dfu:
        if abs(sum_actual) == 0:
            n_undefined += 1
            continue
        accuracies.append(max(0.0, 100.0 * (1.0 - sum_abs_error / abs(sum_actual))))

    n_dfus = len(per_dfu)
    if not accuracies:
        return {
            "n_dfus": n_dfus,
            "n_undefined": n_undefined,
            "mean_accuracy_pct": None,
            "median_accuracy_pct": None,
        }

    return {
        "n_dfus": n_dfus,
        "n_undefined": n_undefined,
        "mean_accuracy_pct": round(statistics.fmean(accuracies), 4),
        "median_accuracy_pct": round(statistics.median(accuracies), 4),
    }


def compute_mase(
    actuals: Sequence[float],
    forecasts: Sequence[float],
    insample_actuals: Sequence[float],
    seasonal_period: int = 1,
) -> float | None:
    """MASE — Mean Absolute Scaled Error (Hyndman & Koehler 2006).

    MASE scores a forecast *relative to the in-sample seasonal-naive forecast*,
    so it is scale-free and fair to structurally-hard SKUs whose tiny base makes
    WAPE% brutal. Reported ALONGSIDE the WAPE headline, never replacing it.

        MASE = mean(|forecasts[i] - actuals[i]|)  /  q

    where the scale ``q`` is the in-sample mean absolute seasonal-naive error::

        q = mean(|insample_actuals[t] - insample_actuals[t - m]|)
            for t in range(m, len(insample_actuals)),  m = seasonal_period

    Interpretation:
        MASE < 1  → the forecast beats the in-sample naive baseline.
        MASE = 1  → on par with naive.
        MASE > 1  → worse than naive.

    seasonal_period (m):
        1  → one-step random-walk naive (default; |x_t - x_{t-1}|).
        12 → monthly series with annual seasonality (|x_t - x_{t-12}|).

    LEAKAGE RULE: the scale ``q`` is computed STRICTLY from ``insample_actuals``
    (the training history BEFORE the eval window). It NEVER touches ``actuals`` or
    ``forecasts`` from the eval window — otherwise the denominator would borrow
    information from the period being scored and the metric would be optimistic.

    Returns ``None`` (never raises, never inf/nan) when the scale is undefined or
    unstable:
        - ``actuals`` / ``forecasts`` empty,
        - ``len(insample_actuals) <= seasonal_period`` (no seasonal diff exists),
        - ``q == 0`` (flat / all-zero in-sample series → scaling undefined).

    Raises:
        ValueError: when ``actuals`` and ``forecasts`` differ in length, or when
            ``seasonal_period < 1``.
    """
    if len(actuals) != len(forecasts):
        raise ValueError(
            f"Length mismatch: actuals={len(actuals)}, forecasts={len(forecasts)}"
        )
    if seasonal_period < 1:
        raise ValueError(f"seasonal_period must be >= 1, got {seasonal_period}")
    if not actuals:
        return None
    if len(insample_actuals) <= seasonal_period:
        return None

    m = seasonal_period
    naive_diffs = [
        abs(insample_actuals[t] - insample_actuals[t - m])
        for t in range(m, len(insample_actuals))
    ]
    q = statistics.fmean(naive_diffs)
    if q == 0:
        return None

    mae_eval = statistics.fmean(
        abs(f - a) for f, a in zip(forecasts, actuals, strict=True)
    )
    return mae_eval / q


def compute_unweighted_mase(
    per_dfu: Sequence[tuple[float, float]],
) -> dict[str, float | int | None]:
    """Unweighted per-DFU MASE: MASE per DFU, then mean/median across DFUs.

    MASE is reported ALONGSIDE WAPE (not replacing it) to give the long tail a
    fairer, naive-relative score: a low-volume SKU that is hard for *everyone*
    scores near 1 (on par with naive) rather than being crushed by WAPE%'s tiny
    denominator. Every DFU is weighted equally here (no volume weighting).

    Each element of ``per_dfu`` is ``(mae_eval, scale_q)`` for one DFU, already
    aggregated: ``mae_eval`` is that DFU's mean ``|F - A|`` over its eval window
    and ``scale_q`` is its in-sample seasonal-naive MAE. Per-DFU MASE is
    ``mae_eval / scale_q``.

    A DFU with ``scale_q <= 0`` has an undefined MASE (zero/degenerate scale).
    Such DFUs are EXCLUDED from the mean/median and reported in ``n_undefined``
    rather than folded in — same philosophy as ``compute_unweighted_accuracy``
    excluding zero-denominator DFUs.

    Returns a dict with keys ``n_dfus``, ``n_undefined``, ``mean_mase``,
    ``median_mase``. Mean/median are ``None`` when every DFU is undefined.
    """
    mases: list[float] = []
    n_undefined = 0
    for mae_eval, scale_q in per_dfu:
        if scale_q <= 0:
            n_undefined += 1
            continue
        mases.append(mae_eval / scale_q)

    n_dfus = len(per_dfu)
    if not mases:
        return {
            "n_dfus": n_dfus,
            "n_undefined": n_undefined,
            "mean_mase": None,
            "median_mase": None,
        }

    return {
        "n_dfus": n_dfus,
        "n_undefined": n_undefined,
        "mean_mase": round(statistics.fmean(mases), 4),
        "median_mase": round(statistics.median(mases), 4),
    }


def compute_accuracy_metrics(
    forecast_col: pd.Series,
    actual_col: pd.Series,
) -> dict[str, Any]:
    """Compute WAPE, bias, and accuracy from forecast and actual series.

    Returns dict with keys: n_rows, wape, bias, accuracy_pct (any may be None).
    """
    df = pd.DataFrame({"f": forecast_col, "a": actual_col}).dropna()
    n_rows = len(df)

    if n_rows == 0 or df["a"].abs().sum() == 0:
        return {"n_rows": n_rows, "wape": None, "bias": None, "accuracy_pct": None}

    total_f = df["f"].sum()
    total_a = df["a"].sum()
    abs_error = (df["f"] - df["a"]).abs().sum()

    wape = 100 * abs_error / abs(total_a) if abs(total_a) > 0 else None
    bias = (total_f / total_a) - 1 if abs(total_a) > 0 else None
    accuracy = 100 - wape if wape is not None else None

    return {
        "n_rows": n_rows,
        "wape": round(float(wape), 2) if wape is not None else None,
        "bias": round(float(bias), 4) if bias is not None else None,
        "accuracy_pct": round(float(accuracy), 2) if accuracy is not None else None,
    }
