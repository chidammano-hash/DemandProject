"""Regime detection and DFU-level cross-validation strategies.

Includes:
  - ``dynamic_window``  — Walk-forward optimal lookback window per DFU
  - ``regime_adaptive`` — Variance-ratio regime detection + strategy switching
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.ml.champion.basic import strategy_expanding, strategy_rolling
from common.ml.champion.helpers import (
    _get_exec_lag,
    compute_strategy_accuracy,
)
from common.ml.champion.registry import (
    _DFU_COLS,
    _DFU_MONTH_COLS,
    _OUTPUT_COLS,
    STRATEGY_REGISTRY,
    register_strategy,
)


# ---------------------------------------------------------------------------
# Strategy: dynamic_window
# ---------------------------------------------------------------------------

@register_strategy("dynamic_window")
def strategy_dynamic_window(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    window_candidates: list[int] | None = None,
    cv_months: int = 3,
    **kwargs: Any,
) -> pd.DataFrame:
    """Dynamic window: cross-validate lookback window per DFU.

    Instead of a fixed rolling window or expanding, try multiple window
    sizes and pick the one with lowest recent cross-validation error.

    For each DFU:
      1. Try each candidate window [2, 3, 4, 6, 9, 12]
      2. Compute WAPE of rolling-window champion selection on the last
         cv_months (walk-forward, strictly causal)
      3. Pick the window with lowest CV WAPE
      4. Apply that window's rolling strategy to all months
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    if window_candidates is None:
        window_candidates = [2, 3, 4, 6, 9, 12]

    all_months = sorted(df["startdate"].unique())
    n_months = len(all_months)

    if n_months <= min_prior_months + cv_months:
        # Not enough months — fall back to rolling with default window
        return strategy_rolling(
            df, window_months=6, min_prior_months=min_prior_months,
        )

    # For each DFU, find optimal window via walk-forward CV
    dfu_best_window: dict[tuple, int] = {}

    for dfu_key, dfu_df in df.groupby(_DFU_COLS, sort=False):
        dfu_months = sorted(dfu_df["startdate"].unique())
        n_dfu_months = len(dfu_months)

        if n_dfu_months <= min_prior_months + cv_months:
            dfu_best_window[dfu_key] = 6  # Default
            continue

        best_window = 6
        best_cv_wape = float("inf")

        for window in window_candidates:
            if window > n_dfu_months - cv_months:
                continue

            # Run rolling strategy on this DFU's data
            dfu_winners = strategy_rolling(
                dfu_df, window_months=window,
                min_prior_months=min(min_prior_months, window),
            )
            if dfu_winners.empty:
                continue

            # Evaluate on last cv_months
            cv_set = set(dfu_months[-cv_months:])
            cv_winners = dfu_winners[dfu_winners["startdate"].isin(cv_set)]
            if cv_winners.empty:
                continue

            acc = compute_strategy_accuracy(cv_winners)
            wape = acc.get("wape")
            if wape is not None and wape < best_cv_wape:
                best_cv_wape = wape
                best_window = window

        dfu_best_window[dfu_key] = best_window

    # Group DFUs by optimal window and run rolling for each group
    window_to_dfus: dict[int, list[tuple]] = {}
    for dfu_key, window in dfu_best_window.items():
        window_to_dfus.setdefault(window, []).append(dfu_key)

    all_winners: list[pd.DataFrame] = []
    for window, dfu_keys in window_to_dfus.items():
        key_df = pd.DataFrame(dfu_keys, columns=_DFU_COLS)
        subset = df.merge(key_df, on=_DFU_COLS, how="inner")
        if subset.empty:
            continue
        winners = strategy_rolling(
            subset, window_months=window,
            min_prior_months=min(min_prior_months, window),
        )
        if not winners.empty:
            all_winners.append(winners)

    if not all_winners:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    combined = pd.concat(all_winners, ignore_index=True)
    combined = combined.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")
    return combined[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: regime_adaptive
# ---------------------------------------------------------------------------

@register_strategy("regime_adaptive")
def strategy_regime_adaptive(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    variance_window: int = 4,
    variance_threshold: float = 2.0,
    stable_strategy: str = "expanding",
    shift_strategy: str = "rolling",
    shift_window: int = 3,
    **kwargs: Any,
) -> pd.DataFrame:
    """Regime-adaptive: detect demand regime changes and switch strategies.

    For each DFU-month, compute a regime indicator by comparing recent
    demand variance to historical variance:

      ratio = var(last N months) / var(all prior months)

    If ratio > threshold → regime shift detected → use short-memory strategy.
    If ratio <= threshold → stable regime → use long-memory strategy.

    Both sub-strategies are strictly causal.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = df.sort_values(_DFU_COLS + ["startdate"]).copy()

    # Deduplicate to one demand value per DFU-month
    demand = df.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")

    # Compute regime indicators per DFU-month
    regime_map: dict[tuple, str] = {}  # (item_id, cg, loc, startdate) -> "stable"|"shift"

    for dfu_key, group in demand.groupby(_DFU_COLS, sort=False):
        g = group.sort_values("startdate")
        exec_lag = _get_exec_lag(g)
        vals = g["tothist_dmd"].astype(float).values
        months = g["startdate"].values

        for i in range(len(vals)):
            n_available = i - exec_lag
            if n_available < variance_window + 2:
                # Not enough history for variance ratio
                regime_map[(*dfu_key, months[i])] = "stable"
                continue

            causal_vals = vals[:n_available]
            recent = causal_vals[-variance_window:]
            historical = causal_vals[:-variance_window]

            var_recent = float(np.var(recent)) if len(recent) > 1 else 0.0
            var_hist = float(np.var(historical)) if len(historical) > 1 else 1e-6

            ratio = var_recent / max(var_hist, 1e-6)
            regime_map[(*dfu_key, months[i])] = "shift" if ratio > variance_threshold else "stable"

    # Split data by regime
    stable_keys = [k for k, v in regime_map.items() if v == "stable"]
    shift_keys = [k for k, v in regime_map.items() if v == "shift"]

    parts: list[pd.DataFrame] = []

    for regime_keys, strategy_name, extra_kwargs in [
        (stable_keys, stable_strategy, {}),
        (shift_keys, shift_strategy, {"window_months": shift_window}),
    ]:
        if not regime_keys:
            continue
        key_df = pd.DataFrame(regime_keys, columns=_DFU_MONTH_COLS)
        subset = df.merge(key_df, on=_DFU_MONTH_COLS, how="inner")
        if subset.empty:
            continue

        fn = STRATEGY_REGISTRY.get(strategy_name, strategy_expanding)
        winners = fn(subset, min_prior_months=min_prior_months, **extra_kwargs)
        if not winners.empty:
            parts.append(winners)

    if not parts:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")
    return combined[_OUTPUT_COLS].reset_index(drop=True)
