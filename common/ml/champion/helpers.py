"""Shared helpers, stats builders, and metric utilities for champion strategies."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from common.core.constants import FORECAST_QTY_COL
from common.ml.champion.registry import (
    _DFU_MODEL_COLS,
    _DFU_MONTH_COLS,
    _OUTPUT_COLS,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output row construction
# ---------------------------------------------------------------------------

def make_blend_row(
    item_id: Any,
    customer_group: Any,
    loc: Any,
    startdate: Any,
    model_id: str,
    prior_wape: float,
    forecast: float,
    actual: float,
) -> dict[str, Any]:
    """Build one champion output row keyed off ``_OUTPUT_COLS``.

    Every champion strategy emits the same per-DFU-month result shape:
    the DFU identity (``item_id``/``customer_group``/``loc``/``startdate``),
    the winning/blended ``model_id``, its ``prior_wape``, the forecast
    quantity (under ``FORECAST_QTY_COL``), and the realised ``tothist_dmd``.
    Centralising the dict here keeps the forecast-quantity key in one place
    (never the ``"basefcst_pref"`` literal) and guarantees column parity with
    ``_OUTPUT_COLS`` across all ~20 emit sites.
    """
    return {
        "item_id": item_id,
        "customer_group": customer_group,
        "loc": loc,
        "startdate": startdate,
        "model_id": model_id,
        "prior_wape": prior_wape,
        FORECAST_QTY_COL: forecast,
        "tothist_dmd": actual,
    }


# ---------------------------------------------------------------------------
# Per-DFU helpers
# ---------------------------------------------------------------------------

def _get_exec_lag(group: pd.DataFrame) -> int:
    """Return execution_lag for a DFU group; defaults to 0 if column absent."""
    if "execution_lag" in group.columns and len(group) > 0:
        val = group["execution_lag"].iloc[0]
        return int(val) if pd.notna(val) else 0
    return 0


def _compute_blend_weights(
    prior_wape: pd.Series,
    weight_method: str = "inverse_wape",
) -> pd.Series:
    """Compute blending weights from prior WAPE values.

    Returns a Series of weights summing to 1.0, indexed like ``prior_wape``.

    weight_method:
        'inverse_wape': weights proportional to 1/WAPE (better models get more weight)
        'equal': uniform weights
    """
    n = len(prior_wape)
    if n == 0:
        return pd.Series(dtype=float)
    if weight_method == "inverse_wape":
        inv = 1.0 / prior_wape.clip(lower=1e-6)
        return inv / inv.sum()
    return pd.Series([1.0 / n] * n, index=prior_wape.index)


def _blend_forecasts(
    top: pd.DataFrame,
    weights: pd.Series,
) -> tuple[float, float, float]:
    """Compute blended forecast, actual, and weighted average WAPE from top models.

    Returns (blended_fcst, actual, avg_wape).
    """
    blended_fcst = float((top[FORECAST_QTY_COL].astype(float) * weights).sum())
    actual = float(top["tothist_dmd"].iloc[0])
    avg_wape = float((top["prior_wape"] * weights).sum())
    return blended_fcst, actual, avg_wape


def _resolve_fallback_rows(
    fallback_rows: list[pd.DataFrame],
    results: list[dict[str, Any]],
    min_prior_months: int = 1,
) -> None:
    """Apply expanding fallback for DFU-months not covered by primary results.

    Deduplicates ``fallback_rows``, ensures ``abs_err`` is present, removes
    DFU-months already in ``results``, and appends fallback winners to
    ``results`` in-place.

    Shared by ``learned_blend`` and ``ridge_blend`` strategies.
    """
    if not fallback_rows:
        return

    # Deferred import to avoid a circular dependency with basic.py
    from common.ml.champion.basic import strategy_expanding

    fallback_df = pd.concat(fallback_rows, ignore_index=True)
    fallback_df = fallback_df.drop_duplicates(
        subset=_DFU_MONTH_COLS + ["model_id"],
    )
    if "abs_err" not in fallback_df.columns:
        fallback_df["abs_err"] = (
            fallback_df[FORECAST_QTY_COL] - fallback_df["tothist_dmd"]
        ).abs()
    if results:
        covered = {
            (r["item_id"], r["customer_group"], r["loc"], r["startdate"])
            for r in results
        }
        fallback_df = fallback_df[
            ~fallback_df[_DFU_MONTH_COLS].apply(tuple, axis=1).isin(covered)
        ]
    if not fallback_df.empty:
        fallback_winners = strategy_expanding(
            fallback_df, min_prior_months=min_prior_months,
        )
        if not fallback_winners.empty:
            results.extend(fallback_winners.to_dict("records"))


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------

def compute_strategy_accuracy(winners_df: pd.DataFrame) -> dict[str, Any]:
    """Compute overall WAPE and accuracy from a winners DataFrame.

    Uses the standard formula: WAPE = SUM(|F-A|) / |SUM(A)| * 100.
    """
    if len(winners_df) == 0:
        return {"wape": None, "accuracy_pct": None, "n_dfu_months": 0}

    abs_err = float((winners_df[FORECAST_QTY_COL] - winners_df["tothist_dmd"]).abs().sum())
    total_actual = float(winners_df["tothist_dmd"].sum())

    if abs(total_actual) == 0:
        return {"wape": None, "accuracy_pct": None, "n_dfu_months": len(winners_df)}

    wape = 100.0 * abs_err / abs(total_actual)
    return {
        "wape": round(float(wape), 4),
        "accuracy_pct": round(100.0 - float(wape), 4),
        "n_dfu_months": len(winners_df),
    }


def compute_ceiling(df: pd.DataFrame) -> pd.DataFrame:
    """Compute oracle (ceiling) winners — lowest absolute error per DFU-month.

    This is the theoretical upper bound with perfect foresight.
    """
    ranked = df.copy()
    ranked["_rank"] = ranked.groupby(_DFU_MONTH_COLS)["abs_err"].rank(
        method="first", ascending=True,
    )
    winners = ranked[ranked["_rank"] == 1].drop(columns=["_rank"])
    winners["prior_wape"] = 0.0  # ceiling has no prior WAPE concept
    return winners[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal: exec-lag-aware cumulative stats per DFU-model group
# ---------------------------------------------------------------------------

def _expanding_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add cum_abs_err / cum_actual / prior_count columns using shift(exec_lag+1).

    Iterates over each (DFU, model) group explicitly to avoid pandas
    FutureWarning from groupby.apply operating on grouping columns.
    """
    df = df.sort_values(_DFU_MODEL_COLS + ["startdate"]).copy()
    groups = []
    for _, group in df.groupby(_DFU_MODEL_COLS, sort=False):
        g = group.sort_values("startdate").copy()
        shift_n = _get_exec_lag(g) + 1
        shifted_err = g["abs_err"].shift(shift_n)
        shifted_act = g["tothist_dmd"].shift(shift_n)
        g["cum_abs_err"] = shifted_err.expanding(min_periods=1).sum()
        g["cum_actual"] = shifted_act.expanding(min_periods=1).sum()
        g["prior_count"] = shifted_err.expanding(min_periods=1).count()
        groups.append(g)
    return pd.concat(groups, ignore_index=True)


def _rolling_stats(
    df: pd.DataFrame, window_months: int, min_prior_months: int,
) -> pd.DataFrame:
    """Add roll_abs_err / roll_actual columns using shift(exec_lag+1)."""
    df = df.sort_values(_DFU_MODEL_COLS + ["startdate"]).copy()
    groups = []
    for _, group in df.groupby(_DFU_MODEL_COLS, sort=False):
        g = group.sort_values("startdate").copy()
        shift_n = _get_exec_lag(g) + 1
        shifted_err = g["abs_err"].shift(shift_n)
        shifted_act = g["tothist_dmd"].shift(shift_n)
        g["roll_abs_err"] = shifted_err.rolling(
            window=window_months, min_periods=min_prior_months
        ).sum()
        g["roll_actual"] = shifted_act.rolling(
            window=window_months, min_periods=min_prior_months
        ).sum()
        groups.append(g)
    return pd.concat(groups, ignore_index=True)


def _expanding_uncertainty_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add cum_abs_err, cum_actual, prior_count, cum_std_err, cum_mean_actual.

    Extends ``_expanding_stats`` by also computing the expanding standard
    deviation of absolute errors over causally-available prior months
    (shift(exec_lag + 1)).

    ``cum_std_err`` is the population std-dev of abs_err values in the
    expanding window --- it captures how *consistent* (or erratic) a model's
    errors are.
    """
    df = df.sort_values(_DFU_MODEL_COLS + ["startdate"]).copy()
    groups: list[pd.DataFrame] = []
    for _, group in df.groupby(_DFU_MODEL_COLS, sort=False):
        g = group.sort_values("startdate").copy()
        shift_n = _get_exec_lag(g) + 1
        shifted_err = g["abs_err"].shift(shift_n)
        shifted_act = g["tothist_dmd"].shift(shift_n)
        g["cum_abs_err"] = shifted_err.expanding(min_periods=1).sum()
        g["cum_actual"] = shifted_act.expanding(min_periods=1).sum()
        g["prior_count"] = shifted_err.expanding(min_periods=1).count()
        # Standard deviation of absolute errors across expanding prior window
        # ddof=0 (population std) to avoid NaN when only 1 prior observation
        g["cum_std_err"] = shifted_err.expanding(min_periods=1).std(ddof=0)
        g["cum_mean_actual"] = shifted_act.expanding(min_periods=1).mean()
        groups.append(g)
    return pd.concat(groups, ignore_index=True)


# Re-export the model-family map used by diversity-aware strategies.
_MODEL_FAMILIES: dict[str, str] = {
    "chronos2": "chronos",
    "chronos2_enriched": "chronos",
    "chronos_bolt": "chronos",
    "catboost_cluster": "tree",
    "xgboost_cluster": "tree",
    "lgbm_cluster": "tree",
    "seasonal_naive": "baseline",
    "rolling_mean": "baseline",
    "mstl": "statistical",
    "nhits": "dl",
    "nbeats": "dl",
}
