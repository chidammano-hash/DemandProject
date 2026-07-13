"""Meta-learner-driven champion selection.

Includes ``strategy_meta_learner`` and its exec-lag-aware feature builder.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from common.core.constants import FORECAST_QTY_COL
from common.ml.champion.basic import strategy_expanding
from common.ml.champion.helpers import _get_exec_lag, select_output_cols
from common.ml.champion.registry import (
    _DFU_COLS,
    _DFU_MODEL_COLS,
    _DFU_MONTH_COLS,
    register_strategy,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy: meta_learner
# ---------------------------------------------------------------------------

@register_strategy("meta_learner")
def strategy_meta_learner(
    df: pd.DataFrame,
    *,
    meta_model_path: str | None = None,
    dfu_features: pd.DataFrame | None = None,
    min_prior_months: int = 3,
    performance_window: int = 6,
    **kwargs: Any,
) -> pd.DataFrame:
    """ML-based model selection using a trained meta-learner.

    Predicts which model will perform best for a given DFU-month based on
    DFU features and recent causally-available model performance stats.

    Falls back to expanding strategy if model not found or features missing.
    """
    if meta_model_path is None or dfu_features is None:
        return strategy_expanding(df, min_prior_months=min_prior_months)

    try:
        import joblib
        meta = joblib.load(meta_model_path)
    except (FileNotFoundError, OSError, ValueError, KeyError, ModuleNotFoundError):
        _logger.warning("Meta-learner model not found, falling back to expanding strategy")
        return strategy_expanding(df, min_prior_months=min_prior_months)

    model_obj = meta["model"]
    feature_cols = meta["feature_columns"]
    models = sorted(df["model_id"].unique())

    feature_rows = _build_meta_features(
        df, dfu_features, models, performance_window, min_prior_months,
    )
    if feature_rows.empty:
        return strategy_expanding(df, min_prior_months=min_prior_months)

    for col in feature_cols:
        if col not in feature_rows.columns:
            feature_rows[col] = 0
    X = feature_rows[feature_cols].fillna(0)

    feature_rows["predicted_model"] = model_obj.predict(X)

    predictions = feature_rows[[*_DFU_MONTH_COLS, "predicted_model"]].rename(
        columns={"predicted_model": "model_id"}
    )
    merged = predictions.merge(
        df[[*_DFU_MONTH_COLS, "model_id", FORECAST_QTY_COL, "tothist_dmd"]],
        on=[*_DFU_MONTH_COLS, "model_id"],
        how="inner",
    )
    merged["prior_wape"] = 0.0

    if merged.empty:
        return strategy_expanding(df, min_prior_months=min_prior_months)
    return select_output_cols(merged)


def _build_meta_features(
    df: pd.DataFrame,
    dfu_features: pd.DataFrame,
    models: list[str],
    performance_window: int,
    min_prior_months: int,
) -> pd.DataFrame:
    """Build feature matrix for meta-learner predictions.

    Features are strictly causal — only data from months causally available
    at issuance time are used (respects execution_lag per DFU).
    """
    df = df.sort_values([*_DFU_MODEL_COLS, "startdate"]).copy()

    # Compute per-model rolling stats with exec-lag-aware causal shift (in-place)
    df["_roll_abs_err"] = np.nan
    df["_roll_actual"] = np.nan
    df["_roll_bias_num"] = np.nan
    df["_prior_count"] = np.nan
    df["_bias_raw"] = df[FORECAST_QTY_COL] - df["tothist_dmd"]

    for _group_key, idx in df.groupby(_DFU_MODEL_COLS, sort=False).groups.items():
        sub = df.loc[idx]
        exec_lag = _get_exec_lag(sub)
        shift_n = exec_lag + 1
        shifted_err = sub["abs_err"].shift(shift_n)
        shifted_act = sub["tothist_dmd"].shift(shift_n)
        shifted_bias = sub["_bias_raw"].shift(shift_n)
        df.loc[idx, "_roll_abs_err"] = shifted_err.rolling(
            window=performance_window, min_periods=1
        ).sum().values
        df.loc[idx, "_roll_actual"] = shifted_act.rolling(
            window=performance_window, min_periods=1
        ).sum().values
        df.loc[idx, "_roll_bias_num"] = shifted_bias.rolling(
            window=performance_window, min_periods=1
        ).sum().values
        df.loc[idx, "_prior_count"] = shifted_err.expanding(min_periods=1).count().values

    df = df.drop(columns=["_bias_raw"])

    df["_roll_wape"] = df["_roll_abs_err"] / df["_roll_actual"].abs().clip(lower=1e-6)
    df["_roll_bias"] = df["_roll_bias_num"] / df["_roll_actual"].abs().clip(lower=1e-6)

    dfu_months = df[df["_prior_count"] >= min_prior_months][
        _DFU_MONTH_COLS
    ].drop_duplicates()

    pivoted = dfu_months.copy()
    # Single pivot instead of K sequential merges
    pivot_src = df[[*_DFU_MONTH_COLS, "model_id", "_roll_wape", "_roll_bias"]].copy()
    for col, prefix in [("_roll_wape", "roll_wape"), ("_roll_bias", "roll_bias")]:
        wide = pivot_src.pivot_table(
            index=_DFU_MONTH_COLS, columns="model_id", values=col, aggfunc="first",
        )
        wide.columns = [f"{prefix}_{m}" for m in wide.columns]
        pivoted = pivoted.merge(wide, on=_DFU_MONTH_COLS, how="left")

    demand = df.drop_duplicates(subset=[*_DFU_MONTH_COLS, "model_id"]).copy()
    demand_agg = demand.groupby([*_DFU_COLS, "startdate"], sort=False).agg(
        avg_demand=("tothist_dmd", "first"),
    ).reset_index()
    demand_agg = demand_agg.sort_values([*_DFU_COLS, "startdate"])

    # Precompute DFU → exec_lag map once (O(N)) to avoid per-DFU full-table scans
    exec_lag_map: dict[tuple, int] = {}
    if "execution_lag" in df.columns:
        exec_lag_map = {
            k: int(v) if pd.notna(v) else 0
            for k, v in df.groupby(_DFU_COLS)["execution_lag"].first().items()
        }

    # Compute demand stats per DFU — in-place assignment avoids concat of many small frames
    demand_agg = demand_agg.sort_values([*_DFU_COLS, "startdate"])
    demand_agg["mean_qty"] = np.nan
    demand_agg["cv_demand"] = np.nan

    for dfu_key, idx in demand_agg.groupby(_DFU_COLS, sort=False).groups.items():
        exec_lag = exec_lag_map.get(dfu_key, 0)
        shift_n = exec_lag + 1
        vals = demand_agg.loc[idx, "avg_demand"]
        shifted = vals.shift(shift_n)
        mean_vals = shifted.expanding(min_periods=1).mean()
        std_vals = shifted.expanding(min_periods=1).std()
        demand_agg.loc[idx, "mean_qty"] = mean_vals.values
        demand_agg.loc[idx, "cv_demand"] = std_vals.values / mean_vals.clip(lower=1e-6).values

    pivoted = pivoted.merge(
        demand_agg[[*_DFU_MONTH_COLS, "mean_qty", "cv_demand"]],
        on=_DFU_MONTH_COLS,
        how="left",
    )

    pivoted["month"] = pivoted["startdate"].dt.month
    pivoted["quarter"] = pivoted["startdate"].dt.quarter
    # Fourier terms (period 12 replaces legacy month_sin/month_cos)
    month_vals = pivoted["month"].values.astype(np.float64)
    for period in [12, 6, 4, 3]:
        angle = 2.0 * np.pi * month_vals / period
        pivoted[f"fourier_sin_{period}"] = np.sin(angle).astype(np.float32)
        pivoted[f"fourier_cos_{period}"] = np.cos(angle).astype(np.float32)

    pivoted = pivoted.merge(dfu_features, on=_DFU_COLS, how="left")

    return pivoted
