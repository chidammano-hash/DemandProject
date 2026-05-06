"""Meta-learner-driven champion strategies.

Includes ``strategy_meta_learner`` (single-model classifier prediction) and
``strategy_hybrid_meta_router`` (confidence-gated meta + inverse-WAPE blend),
plus the shared ``_build_meta_features`` exec-lag-aware feature builder.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from common.ml.champion.basic import strategy_expanding
from common.ml.champion.helpers import (
    _blend_forecasts,
    _compute_blend_weights,
    _expanding_stats,
    _get_exec_lag,
)
from common.ml.champion.registry import (
    _DFU_COLS,
    _DFU_MODEL_COLS,
    _DFU_MONTH_COLS,
    _OUTPUT_COLS,
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
    label_encoder = meta.get("label_encoder")
    model_type = meta.get("training_metadata", {}).get("model_type", "random_forest")
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

    raw_preds = model_obj.predict(X)
    if model_type == "xgboost" and label_encoder is not None:
        predicted_models = label_encoder.inverse_transform(raw_preds)
    else:
        predicted_models = raw_preds
    feature_rows["predicted_model"] = predicted_models

    predictions = feature_rows[_DFU_MONTH_COLS + ["predicted_model"]].rename(
        columns={"predicted_model": "model_id"}
    )
    merged = predictions.merge(
        df[_DFU_MONTH_COLS + ["model_id", "basefcst_pref", "tothist_dmd"]],
        on=_DFU_MONTH_COLS + ["model_id"],
        how="inner",
    )
    merged["prior_wape"] = 0.0

    if merged.empty:
        return strategy_expanding(df, min_prior_months=min_prior_months)
    return merged[_OUTPUT_COLS].reset_index(drop=True)


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
    df = df.sort_values(_DFU_MODEL_COLS + ["startdate"]).copy()

    # Compute per-model rolling stats with exec-lag-aware causal shift (in-place)
    df["_roll_abs_err"] = np.nan
    df["_roll_actual"] = np.nan
    df["_roll_bias_num"] = np.nan
    df["_prior_count"] = np.nan
    df["_bias_raw"] = df["basefcst_pref"] - df["tothist_dmd"]

    for group_key, idx in df.groupby(_DFU_MODEL_COLS, sort=False).groups.items():
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
    pivot_src = df[_DFU_MONTH_COLS + ["model_id", "_roll_wape", "_roll_bias"]].copy()
    for col, prefix in [("_roll_wape", "roll_wape"), ("_roll_bias", "roll_bias")]:
        wide = pivot_src.pivot_table(
            index=_DFU_MONTH_COLS, columns="model_id", values=col, aggfunc="first",
        )
        wide.columns = [f"{prefix}_{m}" for m in wide.columns]
        pivoted = pivoted.merge(wide, on=_DFU_MONTH_COLS, how="left")

    demand = df.drop_duplicates(subset=_DFU_MONTH_COLS + ["model_id"]).copy()
    demand_agg = demand.groupby(_DFU_COLS + ["startdate"], sort=False).agg(
        avg_demand=("tothist_dmd", "first"),
    ).reset_index()
    demand_agg = demand_agg.sort_values(_DFU_COLS + ["startdate"])

    # Precompute DFU → exec_lag map once (O(N)) to avoid per-DFU full-table scans
    exec_lag_map: dict[tuple, int] = {}
    if "execution_lag" in df.columns:
        exec_lag_map = {
            k: int(v) if pd.notna(v) else 0
            for k, v in df.groupby(_DFU_COLS)["execution_lag"].first().items()
        }

    # Compute demand stats per DFU — in-place assignment avoids concat of many small frames
    demand_agg = demand_agg.sort_values(_DFU_COLS + ["startdate"])
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
        demand_agg[_DFU_MONTH_COLS + ["mean_qty", "cv_demand"]],
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


# ---------------------------------------------------------------------------
# Strategy: hybrid_meta_router
# ---------------------------------------------------------------------------

@register_strategy("hybrid_meta_router")
def strategy_hybrid_meta_router(
    df: pd.DataFrame,
    *,
    meta_model_path: str | None = None,
    dfu_features: pd.DataFrame | None = None,
    confidence_threshold: float = 0.6,
    blend_top_k: int = 3,
    min_prior_months: int = 3,
    performance_window: int = 6,
    **kwargs: Any,
) -> pd.DataFrame:
    """Confidence-gated hybrid: meta-router prediction OR inverse-WAPE blend.

    Ports the common.ml.expert_panel hybrid ensemble approach into the champion
    framework with full execution-lag causality.

    For each DFU-month:
      1. Build meta features (exec-lag-aware via ``_build_meta_features``).
      2. Get the classifier's predicted best model + confidence (softmax max).
      3. HIGH confidence (>= threshold): use the single predicted best model.
      4. LOW confidence (< threshold): blend top-K models by inverse-WAPE
         weights computed from causally-available expanding WAPE.

    Falls back to ``strategy_expanding`` if the meta-learner model file is
    missing or ``dfu_features`` are not provided.

    Args:
        df: Monthly errors DataFrame (standard input schema).
        meta_model_path: Path to a joblib-serialised meta-learner dict with
            keys ``model``, ``feature_columns``, and optionally
            ``label_encoder`` and ``training_metadata``.
        dfu_features: Static DFU features DataFrame (item_id, customer_group,
            loc, plus cluster/seasonality/volume columns).
        confidence_threshold: Minimum softmax probability to trust the
            classifier's single-model prediction.  Below this, blend.
        blend_top_k: Number of top models to blend for low-confidence months.
        min_prior_months: Minimum causally-available prior months required
            before a model qualifies for selection or blending.
        performance_window: Rolling window size (months) passed to
            ``_build_meta_features`` for computing per-model rolling stats.
    """
    if meta_model_path is None or dfu_features is None:
        return strategy_expanding(df, min_prior_months=min_prior_months)

    try:
        import joblib
        meta = joblib.load(meta_model_path)
    except (FileNotFoundError, OSError, ValueError, KeyError, ModuleNotFoundError):
        _logger.warning("Hybrid meta-router model not found, falling back to expanding strategy")
        return strategy_expanding(df, min_prior_months=min_prior_months)

    model_obj = meta["model"]
    feature_cols = meta["feature_columns"]
    label_encoder = meta.get("label_encoder")
    model_type = meta.get("training_metadata", {}).get("model_type", "random_forest")
    models = sorted(df["model_id"].unique())

    # ── Build meta features (exec-lag-aware) ──────────────────────────────
    feature_rows = _build_meta_features(
        df, dfu_features, models, performance_window, min_prior_months,
    )
    if feature_rows.empty:
        return strategy_expanding(df, min_prior_months=min_prior_months)

    for col in feature_cols:
        if col not in feature_rows.columns:
            feature_rows[col] = 0
    X = feature_rows[feature_cols].fillna(0)

    # ── Predict + confidence via softmax probabilities ────────────────────
    has_proba = hasattr(model_obj, "predict_proba")
    if has_proba:
        proba = model_obj.predict_proba(X)
        raw_preds = proba.argmax(axis=1)
        confidences = proba.max(axis=1)
    else:
        raw_preds = model_obj.predict(X)
        confidences = np.ones(len(raw_preds), dtype=float)

    if model_type == "xgboost" and label_encoder is not None:
        predicted_models = label_encoder.inverse_transform(raw_preds)
    elif has_proba and hasattr(model_obj, "classes_"):
        predicted_models = np.array(model_obj.classes_)[raw_preds]
    else:
        predicted_models = raw_preds

    feature_rows["predicted_model"] = predicted_models
    feature_rows["confidence"] = confidences

    # ── Split into high-confidence and low-confidence DFU-months ──────────
    high_conf = feature_rows[feature_rows["confidence"] >= confidence_threshold].copy()
    low_conf = feature_rows[feature_rows["confidence"] < confidence_threshold].copy()

    parts: list[pd.DataFrame] = []

    # ── HIGH confidence: use the single predicted best model ──────────────
    if not high_conf.empty:
        high_predictions = high_conf[_DFU_MONTH_COLS + ["predicted_model"]].rename(
            columns={"predicted_model": "model_id"},
        )
        high_merged = high_predictions.merge(
            df[_DFU_MONTH_COLS + ["model_id", "basefcst_pref", "tothist_dmd"]],
            on=_DFU_MONTH_COLS + ["model_id"],
            how="inner",
        )
        high_merged["prior_wape"] = 0.0
        if not high_merged.empty:
            parts.append(high_merged[_OUTPUT_COLS])

    # ── LOW confidence: blend top-K models by inverse-WAPE ────────────────
    if not low_conf.empty:
        # Need expanding stats for WAPE-based blending
        df_expanded = _expanding_stats(df)
        qualified = df_expanded[df_expanded["prior_count"] >= min_prior_months].copy()
        qualified["prior_wape"] = (
            qualified["cum_abs_err"] / qualified["cum_actual"].abs()
        )
        qualified = qualified[qualified["prior_wape"].notna()]

        # Restrict to low-confidence DFU-months
        low_keys = low_conf[_DFU_MONTH_COLS]
        blend_pool = qualified.merge(low_keys, on=_DFU_MONTH_COLS, how="inner")

        blend_results: list[dict[str, Any]] = []
        for key, month_df in blend_pool.groupby(_DFU_MONTH_COLS, sort=False):
            item_id, customer_group, loc, startdate = key
            top = month_df.nsmallest(blend_top_k, "prior_wape")
            if len(top) == 0:
                continue

            weights = _compute_blend_weights(top["prior_wape"])
            blended_fcst, actual, avg_wape = _blend_forecasts(top, weights)

            blend_results.append({
                "item_id": item_id,
                "customer_group": customer_group,
                "loc": loc,
                "startdate": startdate,
                "model_id": "hybrid_meta_router",
                "prior_wape": avg_wape,
                "basefcst_pref": blended_fcst,
                "tothist_dmd": actual,
            })

        if blend_results:
            parts.append(pd.DataFrame(blend_results)[_OUTPUT_COLS])

    # ── Combine high-confidence + blended low-confidence ──────────────────
    if not parts:
        return strategy_expanding(df, min_prior_months=min_prior_months)

    combined = pd.concat(parts, ignore_index=True)
    # High-confidence single-model takes priority if duplicates exist
    combined = combined.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")
    return combined[_OUTPUT_COLS].reset_index(drop=True)
