"""Shared forecast accuracy metrics (WAPE, bias, accuracy)."""

from typing import Any

import pandas as pd


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
