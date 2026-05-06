"""Foundational champion strategies: expanding, rolling, decay, ensemble, ensemble_rolling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.ml.champion.helpers import (
    _blend_forecasts,
    _compute_blend_weights,
    _expanding_stats,
    _get_exec_lag,
    _rolling_stats,
)
from common.ml.champion.registry import (
    _DFU_COLS,
    _DFU_MODEL_COLS,
    _DFU_MONTH_COLS,
    _OUTPUT_COLS,
    register_strategy,
)


# ---------------------------------------------------------------------------
# Strategy: expanding
# ---------------------------------------------------------------------------

@register_strategy("expanding")
def strategy_expanding(
    df: pd.DataFrame, *, min_prior_months: int = 3, **kwargs: Any,
) -> pd.DataFrame:
    """Cumulative WAPE from all causally-available prior months.

    For each (DFU, month T), computes each model's cumulative WAPE over
    months with startdate < T - execution_lag (i.e. months whose actuals
    existed when the forecast was issued at fcstdate = T - exec_lag).

    With execution_lag = 0 this is identical to the previous behaviour.
    With execution_lag = L, the last L months are excluded from the prior
    window to prevent using actuals that weren't available at issuance.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = _expanding_stats(df)

    qualified = df[df["prior_count"] >= min_prior_months].copy()
    qualified["prior_wape"] = qualified["cum_abs_err"] / qualified["cum_actual"].abs()
    qualified = qualified[qualified["prior_wape"].notna()]

    qualified = qualified.sort_values("prior_wape")
    winners = qualified.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")

    return winners[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: rolling
# ---------------------------------------------------------------------------

@register_strategy("rolling")
def strategy_rolling(
    df: pd.DataFrame,
    *,
    window_months: int = 6,
    min_prior_months: int = 3,
    **kwargs: Any,
) -> pd.DataFrame:
    """Rolling window WAPE: only the last N causally-available months.

    Adapts faster to regime changes than expanding. Uses shift(exec_lag + 1)
    to exclude months whose actuals weren't available at issuance time.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = _rolling_stats(df, window_months, min_prior_months)

    qualified = df.dropna(subset=["roll_abs_err", "roll_actual"]).copy()
    qualified["prior_wape"] = qualified["roll_abs_err"] / qualified["roll_actual"].abs()
    qualified = qualified[qualified["prior_wape"].notna()]

    qualified = qualified.sort_values("prior_wape")
    winners = qualified.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")

    return winners[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: decay
# ---------------------------------------------------------------------------

@register_strategy("decay")
def strategy_decay(
    df: pd.DataFrame,
    *,
    decay_factor: float = 0.9,
    min_prior_months: int = 3,
    **kwargs: Any,
) -> pd.DataFrame:
    """Exponential decay: recent causally-available months weighted more.

    Weight for prior month at distance d: w(d) = decay_factor^d
    (d=0 is the most recent available prior month).

    With execution_lag = L, the prior window is months with
    startdate < T - L so the last L months are excluded.
    """
    df = df.sort_values(_DFU_MODEL_COLS + ["startdate"]).copy()

    results = []
    for dfu_key, dfu_df in df.groupby(_DFU_COLS, sort=False):
        item_id, customer_group, loc = dfu_key
        exec_lag = _get_exec_lag(dfu_df)
        months = sorted(dfu_df["startdate"].unique())

        for i, current_month in enumerate(months):
            # Causal prior: exclude last exec_lag months
            # Available prior months: months[0 .. i - exec_lag - 1]
            n_available = i - exec_lag
            if n_available < min_prior_months:
                continue

            prior_months_available = months[:n_available]

            best_wape = float("inf")
            best_model = None

            current_rows = dfu_df[dfu_df["startdate"] == current_month]

            # Pre-build O(1) lookup: month → position (sorted order)
            month_to_pos = {m: idx for idx, m in enumerate(prior_months_available)}

            for model_id, model_df in dfu_df[
                dfu_df["startdate"].isin(prior_months_available)
            ].groupby("model_id", sort=False):
                if len(model_df) < min_prior_months:
                    continue

                # No need to sort — parent dfu_df is already sorted by startdate
                # Distance: 0 = most recent prior month, increasing toward the past
                distances = [
                    len(month_to_pos) - 1 - month_to_pos[m]
                    for m in model_df["startdate"]
                ]
                weights = np.array([decay_factor ** d for d in distances])

                w_abs_err = (weights * model_df["abs_err"].values).sum()
                w_actual = (weights * model_df["tothist_dmd"].values.astype(float)).sum()

                if abs(w_actual) > 0:
                    wape = w_abs_err / abs(w_actual)
                    if wape < best_wape:
                        best_wape = wape
                        best_model = model_id

            if best_model is not None:
                row = current_rows[current_rows["model_id"] == best_model]
                if len(row) > 0:
                    r = row.iloc[0]
                    results.append({
                        "item_id": item_id,
                        "customer_group": customer_group,
                        "loc": loc,
                        "startdate": current_month,
                        "model_id": best_model,
                        "prior_wape": best_wape,
                        "basefcst_pref": r["basefcst_pref"],
                        "tothist_dmd": r["tothist_dmd"],
                    })

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: ensemble
# ---------------------------------------------------------------------------

@register_strategy("ensemble")
def strategy_ensemble(
    df: pd.DataFrame,
    *,
    top_k: int = 3,
    min_prior_months: int = 3,
    weight_method: str = "inverse_wape",
    **kwargs: Any,
) -> pd.DataFrame:
    """Blend top-K models using only causally-available prior months.

    For each DFU-month, rank models by prior cumulative WAPE (expanding,
    exec-lag-aware), take top K, and compute weighted average forecast.

    weight_method: 'inverse_wape' or 'equal'.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = _expanding_stats(df)

    qualified = df[df["prior_count"] >= min_prior_months].copy()
    qualified["prior_wape"] = qualified["cum_abs_err"] / qualified["cum_actual"].abs()
    qualified = qualified[qualified["prior_wape"].notna()]

    results = []
    for key, month_df in qualified.groupby(_DFU_MONTH_COLS, sort=False):
        item_id, customer_group, loc, startdate = key
        top = month_df.nsmallest(top_k, "prior_wape")
        if len(top) == 0:
            continue

        weights = _compute_blend_weights(top["prior_wape"], weight_method)
        blended_fcst, actual, avg_wape = _blend_forecasts(top, weights)

        results.append({
            "item_id": item_id,
            "customer_group": customer_group,
            "loc": loc,
            "startdate": startdate,
            "model_id": "ensemble",
            "prior_wape": avg_wape,
            "basefcst_pref": blended_fcst,
            "tothist_dmd": actual,
        })

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: ensemble_rolling
# ---------------------------------------------------------------------------

@register_strategy("ensemble_rolling")
def strategy_ensemble_rolling(
    df: pd.DataFrame,
    *,
    top_k: int = 3,
    window_months: int = 6,
    min_prior_months: int = 3,
    weight_method: str = "inverse_wape",
    **kwargs: Any,
) -> pd.DataFrame:
    """Blend top-K models using rolling window WAPE instead of expanding.

    Combines the adaptiveness of rolling (reacts to regime changes) with
    the robustness of ensemble blending (hedges against wrong picks).
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = _rolling_stats(df, window_months, min_prior_months)

    qualified = df.dropna(subset=["roll_abs_err", "roll_actual"]).copy()
    qualified["prior_wape"] = qualified["roll_abs_err"] / qualified["roll_actual"].abs()
    qualified = qualified[qualified["prior_wape"].notna()]

    results = []
    for key, month_df in qualified.groupby(_DFU_MONTH_COLS, sort=False):
        item_id, customer_group, loc, startdate = key
        top = month_df.nsmallest(top_k, "prior_wape")
        if len(top) == 0:
            continue

        weights = _compute_blend_weights(top["prior_wape"], weight_method)
        blended_fcst, actual, avg_wape = _blend_forecasts(top, weights)

        results.append({
            "item_id": item_id,
            "customer_group": customer_group,
            "loc": loc,
            "startdate": startdate,
            "model_id": "ensemble",
            "prior_wape": avg_wape,
            "basefcst_pref": blended_fcst,
            "tothist_dmd": actual,
        })

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)
