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
