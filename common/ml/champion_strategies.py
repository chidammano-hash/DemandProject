"""Champion model selection strategies.

All strategies take a DataFrame of per-DFU per-month per-model errors
and return per-DFU per-month winner selections.

Input DataFrame schema (monthly_errors):
    item_id, customer_group, loc, startdate, model_id,
    basefcst_pref, tothist_dmd, abs_err
    [optional: execution_lag, fcstdate]

Output DataFrame schema:
    item_id, customer_group, loc, startdate, model_id,
    basefcst_pref, tothist_dmd
    (+ strategy-specific columns like prior_wape)

CRITICAL: Every strategy must be strictly causal — selection for month T
uses ONLY data from months < T - execution_lag, i.e. only months whose
actuals were available at the time the forecast was issued (fcstdate = T - L).

With execution_lag = L:
  - Forecast for month T is issued at fcstdate = T - L months
  - At issuance time, actuals are known for startdate < T - L
  - Prior window = shift(L + 1) on a startdate-sorted series

This prevents two forms of data leakage:
  1. Using future actuals (standard causality guard)
  2. Using actuals that weren't available at forecast issuance time (exec-lag guard)
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {}


def register_strategy(name: str):
    """Decorator to register a strategy function."""
    def decorator(fn: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        STRATEGY_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DFU_COLS = ["item_id", "customer_group", "loc"]
_DFU_MONTH_COLS = ["item_id", "customer_group", "loc", "startdate"]
_DFU_MODEL_COLS = ["item_id", "customer_group", "loc", "model_id"]

_OUTPUT_COLS = [
    "item_id", "customer_group", "loc", "startdate",
    "model_id", "prior_wape", "basefcst_pref", "tothist_dmd",
]


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
    blended_fcst = float((top["basefcst_pref"].astype(float) * weights).sum())
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

    fallback_df = pd.concat(fallback_rows, ignore_index=True)
    fallback_df = fallback_df.drop_duplicates(
        subset=_DFU_MONTH_COLS + ["model_id"],
    )
    if "abs_err" not in fallback_df.columns:
        fallback_df["abs_err"] = (
            fallback_df["basefcst_pref"] - fallback_df["tothist_dmd"]
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


def compute_strategy_accuracy(winners_df: pd.DataFrame) -> dict[str, Any]:
    """Compute overall WAPE and accuracy from a winners DataFrame.

    Uses the standard formula: WAPE = SUM(|F-A|) / |SUM(A)| * 100.
    """
    if len(winners_df) == 0:
        return {"wape": None, "accuracy_pct": None, "n_dfu_months": 0}

    abs_err = float((winners_df["basefcst_pref"] - winners_df["tothist_dmd"]).abs().sum())
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
    except (FileNotFoundError, Exception):
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
# Strategy: hybrid_warmup
# ---------------------------------------------------------------------------

@register_strategy("hybrid_warmup")
def strategy_hybrid_warmup(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    warmup_strategy: str = "rolling",
    warmup_window: int = 2,
    warmup_min_prior: int = 1,
    primary_strategy: str = "ensemble",
    primary_top_k: int = 3,
    primary_weight_method: str = "inverse_wape",
    **kwargs: Any,
) -> pd.DataFrame:
    """Hybrid strategy: use a fast-adapting strategy for warm-up months,
    then switch to a stronger strategy once enough history is available.

    This addresses the 58% coverage gap where the expanding/ensemble strategies
    discard DFU-months with fewer than min_prior_months of history.

    Phase 1 (warm-up): months with prior_count < min_prior_months
        → use rolling with low min_prior (default: 1 month)
    Phase 2 (stable): months with prior_count >= min_prior_months
        → use ensemble_top_k with inverse_wape blending

    Both phases are strictly causal (exec-lag-aware).
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    # ── Phase 2: Primary strategy on qualified DFU-months ─────────────────
    # Resolve primary strategy from the registry (supports any registered
    # strategy, not just "ensemble" or "expanding").
    primary_fn = STRATEGY_REGISTRY.get(primary_strategy)
    if primary_fn is None:
        primary_fn = strategy_expanding

    # Build kwargs for the primary strategy
    primary_kwargs: dict[str, Any] = {"min_prior_months": min_prior_months}
    if primary_strategy in ("ensemble", "ensemble_rolling"):
        primary_kwargs["top_k"] = primary_top_k
        primary_kwargs["weight_method"] = primary_weight_method
    elif primary_strategy == "adaptive_ensemble":
        primary_kwargs["weight_method"] = primary_weight_method
    # Pass through any extra kwargs the caller provided
    for k, v in kwargs.items():
        if k not in primary_kwargs:
            primary_kwargs[k] = v

    primary_df = primary_fn(df, **primary_kwargs)

    # ── Phase 1: Warm-up strategy on unqualified months ───────────────────
    # Find DFU-months covered by primary
    if len(primary_df) > 0:
        primary_keys = set(
            primary_df[_DFU_MONTH_COLS].apply(tuple, axis=1)
        )
    else:
        primary_keys = set()

    # Get all DFU-months in the data
    all_keys = set(
        df[_DFU_MONTH_COLS].drop_duplicates().apply(tuple, axis=1)
    )
    warmup_keys = all_keys - primary_keys

    if warmup_keys:
        warmup_key_df = pd.DataFrame(
            list(warmup_keys), columns=_DFU_MONTH_COLS,
        )
        warmup_data = df.merge(warmup_key_df, on=_DFU_MONTH_COLS, how="inner")

        if not warmup_data.empty:
            warmup_fn = STRATEGY_REGISTRY.get(warmup_strategy)
            if warmup_fn is None:
                warmup_fn = strategy_rolling

            warmup_kwargs: dict[str, Any] = {
                "min_prior_months": warmup_min_prior,
            }
            if warmup_strategy == "rolling":
                warmup_kwargs["window_months"] = warmup_window
            warmup_df = warmup_fn(warmup_data, **warmup_kwargs)
        else:
            warmup_df = pd.DataFrame(columns=_OUTPUT_COLS)
    else:
        warmup_df = pd.DataFrame(columns=_OUTPUT_COLS)

    # ── Combine ─────────────────────────────────────────────────────────────
    combined = pd.concat([primary_df[_OUTPUT_COLS], warmup_df[_OUTPUT_COLS]], ignore_index=True)
    combined = combined.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: adaptive_ensemble
# ---------------------------------------------------------------------------

@register_strategy("adaptive_ensemble")
def strategy_adaptive_ensemble(
    df: pd.DataFrame,
    *,
    min_k: int = 2,
    max_k: int = 5,
    spread_threshold: float = 0.15,
    min_prior_months: int = 3,
    weight_method: str = "inverse_wape",
    **kwargs: Any,
) -> pd.DataFrame:
    """Adaptive ensemble: vary top-K per DFU-month based on model WAPE spread.

    For DFU-months where models have similar WAPE (low spread), use fewer
    models (min_k) to avoid diluting the best. For high-spread months,
    use more models (max_k) for robustness.

    spread = (worst_wape - best_wape) / best_wape among qualified models.
    If spread > threshold → use max_k; else use min_k.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = _expanding_stats(df)

    qualified = df[df["prior_count"] >= min_prior_months].copy()
    qualified["prior_wape"] = qualified["cum_abs_err"] / qualified["cum_actual"].abs()
    qualified = qualified[qualified["prior_wape"].notna()]

    results = []
    weight_sums: dict[str, float] = {}
    weight_count = 0
    k_distribution: dict[int, int] = {}
    for key, month_df in qualified.groupby(_DFU_MONTH_COLS, sort=False):
        item_id, customer_group, loc, startdate = key
        if len(month_df) == 0:
            continue

        sorted_models = month_df.nsmallest(max_k, "prior_wape")
        best_wape = sorted_models["prior_wape"].iloc[0]
        worst_wape = sorted_models["prior_wape"].iloc[-1]

        if best_wape > 0:
            spread = (worst_wape - best_wape) / best_wape
        else:
            spread = 0.0

        k = max_k if spread > spread_threshold else min_k
        top = sorted_models.head(k)

        weights = _compute_blend_weights(top["prior_wape"], weight_method)

        for model_id_w, w in zip(top["model_id"], weights):
            weight_sums[model_id_w] = weight_sums.get(model_id_w, 0.0) + float(w)
        weight_count += 1
        k_distribution[k] = k_distribution.get(k, 0) + 1

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
    result_df = pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)
    if weight_count > 0:
        avg_weights = {m: round(s / weight_count * 100, 2) for m, s in sorted(weight_sums.items(), key=lambda x: -x[1])}
        result_df.attrs["weight_diagnostics"] = {
            "type": "adaptive_ensemble",
            "avg_model_weight_pct": avg_weights,
            "n_dfu_months_blended": weight_count,
            "k_distribution": k_distribution,
        }
    return result_df


# ---------------------------------------------------------------------------
# Strategy: learned_blend
# ---------------------------------------------------------------------------

@register_strategy("learned_blend")
def strategy_learned_blend(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 6,
    train_months: int = 6,
    alpha: float = 100.0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Learn optimal blending weights per DFU using Ridge regression.

    Instead of heuristic inverse-WAPE weighting, fits a Ridge model per DFU
    on causally-available prior months to learn: actual ≈ w1*model1 + w2*model2 + ...

    Weights are clipped to [0, 1] and normalized to sum=1 (constrained blend).
    Falls back to expanding strategy for DFUs with insufficient history.

    Parameters
    ----------
    min_prior_months : int
        Minimum causally-available prior months required to fit Ridge.
    train_months : int
        Number of most-recent causally-available months to use for training.
        If fewer than ``min_prior_months`` are available, the DFU falls back.
    alpha : float
        Ridge regularization strength (higher → more uniform weights).
    """
    from sklearn.linear_model import Ridge

    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = df.sort_values(_DFU_MODEL_COLS + ["startdate"]).copy()
    models = sorted(df["model_id"].unique())

    results: list[dict[str, Any]] = []
    fallback_rows: list[pd.DataFrame] = []

    for dfu_key, dfu_df in df.groupby(_DFU_COLS, sort=False):
        item_id, customer_group, loc = dfu_key
        exec_lag = _get_exec_lag(dfu_df)
        months = sorted(dfu_df["startdate"].unique())

        for i, current_month in enumerate(months):
            # Causal prior: months whose actuals were known at issuance
            n_available = i - exec_lag
            if n_available < 1:
                continue

            prior_months = months[max(0, n_available - train_months):n_available]

            # Gather prior data for this DFU
            prior_data = dfu_df[dfu_df["startdate"].isin(prior_months)]

            # Pivot wide: rows = months, columns = model predictions
            pivot = prior_data.pivot_table(
                index="startdate",
                columns="model_id",
                values="basefcst_pref",
                aggfunc="first",
            )
            # Target = actual demand (same for all models in a given month)
            target_series = prior_data.drop_duplicates(
                subset=["startdate"],
            ).set_index("startdate")["tothist_dmd"]

            # Align pivot and target
            common_months = pivot.index.intersection(target_series.index)
            if len(common_months) < min_prior_months:
                # Not enough history — collect for fallback
                current_rows = dfu_df[dfu_df["startdate"] == current_month]
                if len(current_rows) > 0:
                    fallback_rows.append(current_rows)
                continue

            X_train = pivot.loc[common_months].fillna(0).values
            y_train = target_series.loc[common_months].fillna(0).values.astype(float)
            model_cols = list(pivot.columns)

            # Fit Ridge regression
            ridge = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr")
            ridge.fit(X_train, y_train)

            # Clip weights to [0, 1] and normalize to sum=1
            raw_weights = ridge.coef_.copy()
            raw_weights = np.clip(raw_weights, 0.0, 1.0)
            weight_sum = raw_weights.sum()
            if weight_sum < 1e-9:
                # All weights zero — equal blend as last resort
                raw_weights = np.ones(len(model_cols)) / len(model_cols)
            else:
                raw_weights = raw_weights / weight_sum

            # Apply learned weights to current month predictions
            current_rows = dfu_df[dfu_df["startdate"] == current_month]
            if len(current_rows) == 0:
                continue

            blended_fcst = 0.0
            for j, model_id in enumerate(model_cols):
                model_row = current_rows[current_rows["model_id"] == model_id]
                if len(model_row) > 0:
                    blended_fcst += raw_weights[j] * float(
                        model_row["basefcst_pref"].iloc[0],
                    )

            actual = float(current_rows["tothist_dmd"].iloc[0])

            results.append({
                "item_id": item_id,
                "customer_group": customer_group,
                "loc": loc,
                "startdate": current_month,
                "model_id": "learned_blend",
                "prior_wape": 0.0,
                "basefcst_pref": blended_fcst,
                "tothist_dmd": actual,
            })

    # ── Fallback: expanding strategy for DFUs with insufficient history ──
    _resolve_fallback_rows(fallback_rows, results, min_prior_months=1)

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: seasonal
# ---------------------------------------------------------------------------

@register_strategy("seasonal")
def strategy_seasonal(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 2,
    fallback_strategy: str = "expanding",
    **kwargs: Any,
) -> pd.DataFrame:
    """Same-quarter cumulative WAPE: selects champion per DFU-month using
    only prior months from the SAME calendar quarter.

    Different models dominate different seasons (e.g., Chronos for holiday Q4,
    CatBoost for stable Q1). This strategy exploits that seasonality by
    evaluating each model on its same-quarter track record only.

    For each (DFU, month T, model):
      1. Identify quarter Q = quarter(T)
      2. Gather all prior months with quarter == Q AND startdate < T - exec_lag
      3. Compute cumulative WAPE over those same-quarter prior months
      4. Pick the model with lowest same-quarter WAPE

    If a DFU-month has fewer than min_prior_months same-quarter history rows,
    fall back to the specified fallback_strategy (default: expanding) for
    that DFU-month.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = df.sort_values(_DFU_MODEL_COLS + ["startdate"]).copy()
    df["quarter"] = df["startdate"].dt.quarter

    # ── Phase 1: same-quarter cumulative stats ─────────────────────────────
    # For each (DFU, model, quarter), compute expanding WAPE over same-quarter
    # rows only, respecting exec-lag causality via shift(exec_lag + 1).
    groups: list[pd.DataFrame] = []
    for _, group in df.groupby(_DFU_MODEL_COLS + ["quarter"], sort=False):
        g = group.sort_values("startdate").copy()
        shift_n = _get_exec_lag(g) + 1
        shifted_err = g["abs_err"].shift(shift_n)
        shifted_act = g["tothist_dmd"].shift(shift_n)
        g["sq_cum_abs_err"] = shifted_err.expanding(min_periods=1).sum()
        g["sq_cum_actual"] = shifted_act.expanding(min_periods=1).sum()
        g["sq_prior_count"] = shifted_err.expanding(min_periods=1).count()
        groups.append(g)
    df_sq = pd.concat(groups, ignore_index=True)

    # ── Phase 2: pick best model per DFU-month where enough history ────────
    qualified = df_sq[df_sq["sq_prior_count"] >= min_prior_months].copy()
    qualified["prior_wape"] = (
        qualified["sq_cum_abs_err"] / qualified["sq_cum_actual"].abs()
    )
    qualified = qualified[qualified["prior_wape"].notna()]

    qualified = qualified.sort_values("prior_wape")
    seasonal_winners = qualified.drop_duplicates(
        subset=_DFU_MONTH_COLS, keep="first",
    )[_OUTPUT_COLS].copy()

    # ── Phase 3: fallback for DFU-months with insufficient quarter history ─
    if len(seasonal_winners) > 0:
        covered_keys = set(
            seasonal_winners[_DFU_MONTH_COLS].apply(tuple, axis=1)
        )
    else:
        covered_keys = set()

    all_keys = set(
        df[_DFU_MONTH_COLS].drop_duplicates().apply(tuple, axis=1)
    )
    uncovered_keys = all_keys - covered_keys

    if uncovered_keys:
        uncovered_df = pd.DataFrame(
            list(uncovered_keys), columns=_DFU_MONTH_COLS,
        )
        fallback_data = df.drop(columns=["quarter"]).merge(
            uncovered_df, on=_DFU_MONTH_COLS, how="inner",
        )
        if not fallback_data.empty:
            fallback_fn = STRATEGY_REGISTRY.get(
                fallback_strategy, strategy_expanding,
            )
            fallback_winners = fallback_fn(
                fallback_data, min_prior_months=min_prior_months, **kwargs,
            )
        else:
            fallback_winners = pd.DataFrame(columns=_OUTPUT_COLS)
    else:
        fallback_winners = pd.DataFrame(columns=_OUTPUT_COLS)

    # ── Combine ────────────────────────────────────────────────────────────
    combined = pd.concat(
        [seasonal_winners[_OUTPUT_COLS], fallback_winners[_OUTPUT_COLS]],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: ensemble_rolling  (NOTE: per_cluster is at the very end of file)
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


# ---------------------------------------------------------------------------
# Strategy: optimized_decay
# ---------------------------------------------------------------------------

_logger = logging.getLogger(__name__)


@register_strategy("optimized_decay")
def strategy_optimized_decay(
    df: pd.DataFrame,
    *,
    decay_candidates: list[float] | None = None,
    min_prior_months: int = 3,
    validation_months: int = 3,
    **kwargs: Any,
) -> pd.DataFrame:
    """Walk-forward decay factor optimizer.

    Selects the best exponential decay factor via a train/validation split:

    1. Split the timeline into train (all months except the last
       ``validation_months``) and validation (last ``validation_months``).
    2. For each candidate decay factor, run ``strategy_decay`` on the full
       data (the decay strategy is itself strictly causal), then evaluate
       WAPE only on the validation months.
    3. Pick the decay factor with the lowest validation WAPE.
    4. Run ``strategy_decay`` with the optimal factor on the **full** data.

    This is strictly causal — the decay strategy only uses months
    < T - exec_lag for each month T, and the validation split mirrors
    walk-forward CV.
    """
    if decay_candidates is None:
        decay_candidates = [0.75, 0.80, 0.85, 0.90, 0.95]

    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    # ── Timeline split ─────────────────────────────────────────────────────
    all_months = sorted(df["startdate"].unique())
    n_months = len(all_months)

    # Need enough months for train (min_prior_months) + validation
    if n_months <= validation_months + min_prior_months:
        _logger.info(
            "optimized_decay: not enough months for validation split "
            "(%d total, need >%d); falling back to default decay=0.90",
            n_months,
            validation_months + min_prior_months,
        )
        return strategy_decay(
            df, decay_factor=0.90, min_prior_months=min_prior_months,
        )

    val_months_set = set(all_months[-validation_months:])

    # ── Candidate evaluation ───────────────────────────────────────────────
    best_wape = float("inf")
    best_decay = decay_candidates[0]

    for candidate in decay_candidates:
        # Run strategy_decay on FULL data — it is itself strictly causal
        # (uses only months < T - exec_lag for each month T).  We then
        # score only on the held-out validation months.
        full_winners = strategy_decay(
            df,
            decay_factor=candidate,
            min_prior_months=min_prior_months,
        )

        if full_winners.empty:
            continue

        # Evaluate only on validation months
        val_winners = full_winners[
            full_winners["startdate"].isin(val_months_set)
        ]

        if val_winners.empty:
            continue

        val_acc = compute_strategy_accuracy(val_winners)
        val_wape = val_acc.get("wape")

        if val_wape is not None and val_wape < best_wape:
            best_wape = val_wape
            best_decay = candidate

    _logger.info(
        "optimized_decay: best decay factor = %.2f (val WAPE = %.4f%%)",
        best_decay,
        best_wape if best_wape < float("inf") else float("nan"),
    )

    # ── Final run with optimal decay on full data ──────────────────────────
    return strategy_decay(
        df,
        decay_factor=best_decay,
        min_prior_months=min_prior_months,
    )


# ---------------------------------------------------------------------------
# Strategy: per_segment (Syntetos-Boylan demand classification routing)
# ---------------------------------------------------------------------------

_DEFAULT_SEGMENT_STRATEGY_MAP: dict[str, dict[str, Any]] = {
    "smooth": {"strategy": "expanding"},
    "erratic": {"strategy": "ensemble", "top_k": 5},
    "intermittent": {"strategy": "rolling", "window_months": 6},
    "lumpy": {"strategy": "rolling", "window_months": 3, "min_prior": 2},
}


def _classify_demand_segments(
    df: pd.DataFrame,
    adi_threshold: float,
    cv2_threshold: float,
) -> dict[tuple[str, str, str], str]:
    """Classify each DFU into a Syntetos-Boylan demand segment.

    Uses only causally-available history: months with startdate < T - exec_lag
    where T is the latest month per DFU.  Classification is computed once per
    DFU using all available causal history (not per-month).

    Returns a mapping of (item_id, customer_group, loc) -> segment name.

    Segments:
        smooth:       ADI < threshold AND CV^2 < threshold
        erratic:      ADI < threshold AND CV^2 >= threshold
        intermittent: ADI >= threshold AND CV^2 < threshold
        lumpy:        ADI >= threshold AND CV^2 >= threshold

    DFUs with all-zero demand are assigned to "lumpy" (most conservative).
    """
    df_sorted = df.sort_values(_DFU_COLS + ["startdate"]).copy()

    # Deduplicate to one row per DFU-month (demand is same across models)
    dfu_demand = df_sorted.drop_duplicates(
        subset=_DFU_MONTH_COLS, keep="first",
    )[_DFU_COLS + ["startdate", "tothist_dmd"]].copy()

    if "execution_lag" in df.columns:
        lag_map: dict[tuple, int] = {
            k: int(v) if pd.notna(v) else 0
            for k, v in df.groupby(_DFU_COLS)["execution_lag"].first().items()
        }
    else:
        lag_map = {}

    segment_map: dict[tuple[str, str, str], str] = {}

    for dfu_key, group in dfu_demand.groupby(_DFU_COLS, sort=False):
        g = group.sort_values("startdate")
        exec_lag = lag_map.get(dfu_key, 0)

        # Causal window: exclude the last (exec_lag) months
        if exec_lag > 0 and len(g) > exec_lag:
            g_causal = g.iloc[:-exec_lag]
        else:
            g_causal = g

        demands = g_causal["tothist_dmd"].values.astype(float)
        n_periods = len(demands)
        n_nonzero = int(np.count_nonzero(demands))

        # All-zero demand -> lumpy (most conservative / short-memory routing)
        if n_nonzero == 0 or n_periods == 0:
            segment_map[dfu_key] = "lumpy"
            continue

        # ADI = average demand interval = n_periods / n_nonzero
        adi = n_periods / n_nonzero

        # CV^2 of non-zero demands
        nonzero_vals = demands[demands != 0]
        mean_nz = float(nonzero_vals.mean())
        std_nz = float(nonzero_vals.std(ddof=1)) if len(nonzero_vals) > 1 else 0.0
        cv2 = (std_nz / mean_nz) ** 2 if mean_nz > 0 else 0.0

        # Classify using Syntetos-Boylan quadrant
        if adi < adi_threshold and cv2 < cv2_threshold:
            segment_map[dfu_key] = "smooth"
        elif adi < adi_threshold and cv2 >= cv2_threshold:
            segment_map[dfu_key] = "erratic"
        elif adi >= adi_threshold and cv2 < cv2_threshold:
            segment_map[dfu_key] = "intermittent"
        else:
            segment_map[dfu_key] = "lumpy"

    return segment_map


@register_strategy("per_segment")
def strategy_per_segment(
    df: pd.DataFrame,
    *,
    adi_threshold: float = 1.32,
    cv2_threshold: float = 0.49,
    segment_strategy_map: dict[str, dict[str, Any]] | None = None,
    min_prior_months: int = 3,
    **kwargs: Any,
) -> pd.DataFrame:
    """Per-segment champion selection using Syntetos-Boylan demand classification.

    Classifies each DFU into one of four demand archetypes (smooth, erratic,
    intermittent, lumpy) and routes to different selection strategies:

        smooth:       expanding (stable demand, long history helps)
        erratic:      ensemble top_k=5 (high variance, hedge more)
        intermittent: rolling window=6 (recent patterns matter)
        lumpy:        rolling window=3, min_prior=2 (very short memory)

    Classification is strictly causal -- only months with startdate < T - exec_lag
    are used for computing ADI and CV^2.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with per-DFU per-month per-model errors.
    adi_threshold : float
        ADI threshold separating smooth/erratic from intermittent/lumpy.
        Default 1.32 (Syntetos-Boylan standard).
    cv2_threshold : float
        CV^2 threshold separating smooth/intermittent from erratic/lumpy.
        Default 0.49 (Syntetos-Boylan standard).
    segment_strategy_map : dict[str, dict] | None
        Optional override for segment-to-strategy mapping.  Keys are segment
        names, values are dicts with ``"strategy"`` plus any strategy-specific
        kwargs.  If None, uses the default SBA mapping.
    min_prior_months : int
        Minimum prior months passed to each sub-strategy (can be overridden
        per-segment via segment_strategy_map).
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    # Resolve segment -> strategy mapping
    seg_map = dict(_DEFAULT_SEGMENT_STRATEGY_MAP)
    if segment_strategy_map is not None:
        seg_map.update(segment_strategy_map)

    # Classify each DFU
    segment_assignments = _classify_demand_segments(
        df, adi_threshold, cv2_threshold,
    )

    # Group DFU keys by segment
    segment_to_dfus: dict[str, list[tuple[str, str, str]]] = {}
    for dfu_key, segment in segment_assignments.items():
        segment_to_dfus.setdefault(segment, []).append(dfu_key)

    # Run each segment's sub-strategy on its subset of data
    all_winners: list[pd.DataFrame] = []

    for segment, dfu_keys in segment_to_dfus.items():
        # Build filter for this segment's DFUs
        key_df = pd.DataFrame(dfu_keys, columns=_DFU_COLS)
        segment_data = df.merge(key_df, on=_DFU_COLS, how="inner")

        if segment_data.empty:
            continue

        # Resolve strategy name and kwargs for this segment
        seg_config = seg_map.get(segment, {"strategy": "expanding"})
        strategy_name = seg_config.get("strategy", "expanding")
        seg_kwargs: dict[str, Any] = {
            k: v for k, v in seg_config.items() if k != "strategy"
        }

        # Use segment-specific min_prior if provided, else the global default
        if "min_prior" in seg_kwargs:
            seg_kwargs["min_prior_months"] = seg_kwargs.pop("min_prior")
        elif "min_prior_months" not in seg_kwargs:
            seg_kwargs["min_prior_months"] = min_prior_months

        # Look up the sub-strategy from the registry
        strategy_fn = STRATEGY_REGISTRY.get(strategy_name)
        if strategy_fn is None:
            strategy_fn = STRATEGY_REGISTRY["expanding"]

        # Execute the sub-strategy
        winners = strategy_fn(segment_data, **seg_kwargs)
        if not winners.empty:
            all_winners.append(winners)

    if not all_winners:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    combined = pd.concat(all_winners, ignore_index=True)
    # Deduplicate -- a DFU-month should only appear once
    combined = combined.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")
    return combined[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal: exec-lag-aware expanding stats with std_err for uncertainty
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Strategy: uncertainty_aware
# ---------------------------------------------------------------------------

@register_strategy("uncertainty_aware")
def strategy_uncertainty_aware(
    df: pd.DataFrame,
    *,
    uncertainty_weight: float = 0.3,
    min_prior_months: int = 3,
    use_ensemble: bool = False,
    top_k: int = 3,
    **kwargs: Any,
) -> pd.DataFrame:
    """Uncertainty-aware model selection: penalizes erratic models.

    Instead of picking purely by WAPE, computes a risk-adjusted score that
    factors in prediction consistency (standard deviation of absolute errors).

    For each (DFU, month T, model), using strictly causal prior months
    (shift(exec_lag + 1)):

      1. ``prior_wape`` = cumulative WAPE (sum(abs_err) / |sum(actual)|)
      2. ``prior_std_err`` = std-dev of absolute errors across prior months
      3. ``score = prior_wape + uncertainty_weight * (prior_std_err / mean_actual)``

    The second term penalizes models whose errors are volatile --- even if
    their average WAPE is low, high variance means the model is unreliable.

    Parameters
    ----------
    uncertainty_weight : float
        How much to penalize error variance. 0 = pure WAPE (equivalent to
        expanding strategy). 1 = equal weight to WAPE and normalized std-err.
    min_prior_months : int
        Minimum causally-available prior months to qualify a model.
    use_ensemble : bool
        If True, blend top-K models by inverse risk-adjusted score instead
        of picking the single best model.
    top_k : int
        Number of models to blend when ``use_ensemble`` is True.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = _expanding_uncertainty_stats(df)

    # Filter to rows with enough causal prior history
    qualified = df[df["prior_count"] >= min_prior_months].copy()
    qualified["prior_wape"] = (
        qualified["cum_abs_err"] / qualified["cum_actual"].abs()
    )
    qualified = qualified[qualified["prior_wape"].notna()]

    # Compute risk-adjusted score
    # Normalize std_err by mean actual demand to make it scale-invariant
    mean_actual_safe = qualified["cum_mean_actual"].abs().clip(lower=1e-6)
    normalized_std = qualified["cum_std_err"] / mean_actual_safe
    qualified["_risk_score"] = (
        qualified["prior_wape"] + uncertainty_weight * normalized_std
    )
    qualified = qualified[qualified["_risk_score"].notna()]

    if not use_ensemble:
        # -- Single-model selection: lowest risk-adjusted score --------
        qualified = qualified.sort_values("_risk_score")
        winners = qualified.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")
        return winners[_OUTPUT_COLS].reset_index(drop=True)

    # -- Ensemble mode: blend top-K by inverse risk-adjusted score -----
    results: list[dict[str, Any]] = []
    for key, month_df in qualified.groupby(_DFU_MONTH_COLS, sort=False):
        item_id, customer_group, loc, startdate = key
        top = month_df.nsmallest(top_k, "_risk_score")
        if len(top) == 0:
            continue

        # Inverse risk-score weighting (lower score -> higher weight)
        inv_scores = 1.0 / top["_risk_score"].clip(lower=1e-6)
        weights = inv_scores / inv_scores.sum()

        blended_fcst = (top["basefcst_pref"].astype(float) * weights).sum()
        actual = float(top["tothist_dmd"].iloc[0])
        avg_wape = float((top["prior_wape"] * weights).sum())

        results.append({
            "item_id": item_id,
            "customer_group": customer_group,
            "loc": loc,
            "startdate": startdate,
            "model_id": "uncertainty_ensemble",
            "prior_wape": avg_wape,
            "basefcst_pref": blended_fcst,
            "tothist_dmd": actual,
        })

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: ridge_blend
# ---------------------------------------------------------------------------

@register_strategy("ridge_blend")
def strategy_ridge_blend(
    df: pd.DataFrame,
    *,
    ridge_alpha: float = 100.0,
    min_train_months: int = 6,
    min_prior_months: int = 3,
    normalize_weights: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Ridge regression blending — learn optimal model weights from causal prior data.

    For each DFU-month, instead of heuristic inverse-WAPE weighting:
      1. Collect causally-available prior months (respecting exec_lag)
      2. Build matrix X where each column is a model's forecast, y = actuals
      3. Fit Ridge(alpha, solver='lsqr') to learn weights: y approx X @ w
      4. Clip negative weights to 0 and normalize to sum=1
      5. Apply learned weights to current month's forecasts for blended output

    Requires at least ``min_train_months`` prior months AND at least 2 models
    with prior data. Falls back to expanding strategy for DFU-months with
    insufficient history.

    Parameters
    ----------
    ridge_alpha : float
        Ridge regularization strength (higher = more uniform weights).
    min_train_months : int
        Minimum causally-available prior months required to fit Ridge.
    min_prior_months : int
        Minimum prior months for the expanding fallback strategy.
    normalize_weights : bool
        If True, clip negative weights to 0 and normalize to sum=1.
    """
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        # sklearn not available — fall back entirely to expanding
        return strategy_expanding(df, min_prior_months=min_prior_months)

    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = df.sort_values(_DFU_MODEL_COLS + ["startdate"]).copy()

    results: list[dict[str, Any]] = []
    fallback_rows: list[pd.DataFrame] = []

    for dfu_key, dfu_df in df.groupby(_DFU_COLS, sort=False):
        item_id, customer_group, loc = dfu_key
        exec_lag = _get_exec_lag(dfu_df)
        months = sorted(dfu_df["startdate"].unique())

        for i, current_month in enumerate(months):
            # Causal prior: months whose actuals were known at issuance
            n_available = i - exec_lag
            if n_available < 1:
                continue

            prior_months = months[:n_available]

            # Gather prior data for this DFU
            prior_data = dfu_df[dfu_df["startdate"].isin(prior_months)]

            # Pivot wide: rows = months, columns = model forecasts
            pivot = prior_data.pivot_table(
                index="startdate",
                columns="model_id",
                values="basefcst_pref",
                aggfunc="first",
            )

            # Target = actual demand (same for all models in a given month)
            target_series = prior_data.drop_duplicates(
                subset=["startdate"],
            ).set_index("startdate")["tothist_dmd"]

            # Align pivot and target on common months
            common_months = pivot.index.intersection(target_series.index)

            # Drop models (columns) that are entirely NaN over the prior window
            X_raw = pivot.loc[common_months].copy()
            X_raw = X_raw.dropna(axis=1, how="all")

            # Need enough prior months AND at least 2 models
            if len(common_months) < min_train_months or X_raw.shape[1] < 2:
                current_rows = dfu_df[dfu_df["startdate"] == current_month]
                if len(current_rows) > 0:
                    fallback_rows.append(current_rows)
                continue

            X_train = X_raw.fillna(0).values.astype(float)
            y_train = target_series.loc[common_months].fillna(0).values.astype(float)
            model_cols = list(X_raw.columns)

            # Drop constant columns before fitting to avoid LinAlgWarning
            col_stds = X_train.std(axis=0)
            non_const_mask = col_stds > 1e-12
            if non_const_mask.sum() < 2:
                # Fewer than 2 non-constant model columns — cannot blend
                current_rows = dfu_df[dfu_df["startdate"] == current_month]
                if len(current_rows) > 0:
                    fallback_rows.append(current_rows)
                continue

            X_fit = X_train[:, non_const_mask]

            # Fit Ridge regression
            ridge = Ridge(alpha=ridge_alpha, fit_intercept=False, solver="lsqr")
            ridge.fit(X_fit, y_train)

            # Build full weight vector (zeros for dropped constant columns)
            full_weights = np.zeros(len(model_cols))
            fit_idx = 0
            for col_idx, keep in enumerate(non_const_mask):
                if keep:
                    full_weights[col_idx] = ridge.coef_[fit_idx]
                    fit_idx += 1

            # Normalize: clip negatives to 0, normalize to sum=1
            if normalize_weights:
                full_weights = np.clip(full_weights, 0.0, None)
                weight_sum = full_weights.sum()
                if weight_sum < 1e-9:
                    # All weights zero — equal blend as last resort
                    full_weights = np.ones(len(model_cols)) / len(model_cols)
                else:
                    full_weights = full_weights / weight_sum

            # Apply learned weights to current month predictions
            current_rows = dfu_df[dfu_df["startdate"] == current_month]
            if len(current_rows) == 0:
                continue

            blended_fcst = 0.0
            for j, model_id in enumerate(model_cols):
                model_row = current_rows[current_rows["model_id"] == model_id]
                if len(model_row) > 0:
                    blended_fcst += full_weights[j] * float(
                        model_row["basefcst_pref"].iloc[0],
                    )

            actual = float(current_rows["tothist_dmd"].iloc[0])

            results.append({
                "item_id": item_id,
                "customer_group": customer_group,
                "loc": loc,
                "startdate": current_month,
                "model_id": "ridge_blend",
                "prior_wape": 0.0,
                "basefcst_pref": blended_fcst,
                "tothist_dmd": actual,
            })

    # ── Fallback: expanding strategy for DFU-months with insufficient data ──
    _resolve_fallback_rows(fallback_rows, results, min_prior_months=min_prior_months)

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


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

    Ports the algorithm_testing hybrid ensemble approach into the champion
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
    except (FileNotFoundError, Exception):
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


# ---------------------------------------------------------------------------
# Strategy: diverse_ensemble
# ---------------------------------------------------------------------------

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


@register_strategy("diverse_ensemble")
def strategy_diverse_ensemble(
    df: pd.DataFrame,
    *,
    top_k: int = 3,
    min_prior_months: int = 3,
    correlation_penalty: float = 0.5,
    **kwargs: Any,
) -> pd.DataFrame:
    """Diversity-penalised ensemble: penalise correlated models from the same family.

    For each DFU-month, greedily select top-K models by prior WAPE while
    applying a multiplicative penalty when a candidate belongs to the same
    model family as an already-selected model.  This discourages blending
    structurally similar models (e.g. chronos2 + chronos2_enriched) that
    would otherwise dominate the ensemble.

    Effective WAPE for selection:
        prior_wape * (1 + correlation_penalty * n_same_family_already_selected)

    Blending weights use the *original* (unpenalised) inverse-WAPE so that
    the final forecast is not distorted by the selection heuristic.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = _expanding_stats(df)

    qualified = df[df["prior_count"] >= min_prior_months].copy()
    qualified["prior_wape"] = qualified["cum_abs_err"] / qualified["cum_actual"].abs()
    qualified = qualified[qualified["prior_wape"].notna()]

    results: list[dict[str, Any]] = []
    for key, month_df in qualified.groupby(_DFU_MONTH_COLS, sort=False):
        item_id, customer_group, loc, startdate = key
        candidates = month_df.sort_values("prior_wape", ascending=True)
        if len(candidates) == 0:
            continue

        # Greedy selection with diversity penalty
        selected_indices: list[int] = []
        selected_families: list[str] = []
        remaining = candidates.copy()

        for _ in range(min(top_k, len(candidates))):
            if remaining.empty:
                break
            # Compute effective WAPE for all remaining candidates
            families_arr = remaining["model_id"].map(
                lambda m: _MODEL_FAMILIES.get(m, m),
            )
            penalty_counts = families_arr.map(
                lambda f: selected_families.count(f),
            )
            effective_wapes = remaining["prior_wape"] * (
                1.0 + correlation_penalty * penalty_counts
            )
            best_idx = effective_wapes.idxmin()
            selected_indices.append(best_idx)
            selected_families.append(
                _MODEL_FAMILIES.get(
                    remaining.loc[best_idx, "model_id"],
                    remaining.loc[best_idx, "model_id"],
                ),
            )
            remaining = remaining.drop(index=best_idx)

        if not selected_indices:
            continue

        top = candidates.loc[selected_indices]

        # Blend with original (unpenalised) inverse-WAPE weights
        weights = _compute_blend_weights(top["prior_wape"])
        blended_fcst, actual, avg_wape = _blend_forecasts(top, weights)

        results.append({
            "item_id": item_id,
            "customer_group": customer_group,
            "loc": loc,
            "startdate": startdate,
            "model_id": "diverse_ensemble",
            "prior_wape": avg_wape,
            "basefcst_pref": blended_fcst,
            "tothist_dmd": actual,
        })

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: per_cluster
# ---------------------------------------------------------------------------

@register_strategy("per_cluster")
def strategy_per_cluster(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    dfu_features: pd.DataFrame | None = None,
    cluster_col: str = "ml_cluster",
    **kwargs: Any,
) -> pd.DataFrame:
    """Per-cluster champion: best model per ML cluster applied to all DFUs.

    Different ML clusters represent distinct demand profiles (e.g. smooth
    high-volume vs intermittent low-volume).  This strategy learns which
    model performs best *within each cluster* and assigns that cluster-level
    champion to every DFU in the cluster.

    Steps:
      1. Join ``dfu_features`` (must contain ``ml_cluster``) onto ``df``.
      2. Compute expanding WAPE per (cluster, model) across all DFUs in the
         cluster, strictly causal (exec-lag-aware via ``_expanding_stats``).
      3. For each cluster, rank models by aggregate WAPE and pick the best.
      4. For each DFU-month, look up the DFU's cluster and select that
         cluster's champion model's forecast.

    Falls back to ``strategy_expanding`` when:
      - ``dfu_features`` is None or missing the cluster column.
      - A DFU has no cluster assignment (unmapped DFUs use the global best).

    Parameters
    ----------
    dfu_features : pd.DataFrame | None
        Must contain item_id, customer_group, loc, and ``cluster_col``.
    cluster_col : str
        Column name in ``dfu_features`` holding the cluster label.
    min_prior_months : int
        Minimum causally-available prior months for a model to qualify.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    # No features → pure expanding fallback
    if dfu_features is None or cluster_col not in dfu_features.columns:
        return strategy_expanding(df, min_prior_months=min_prior_months)

    # ── Merge cluster labels onto monthly errors ──────────────────────────
    cluster_map = dfu_features[_DFU_COLS + [cluster_col]].drop_duplicates(
        subset=_DFU_COLS, keep="first",
    )
    df_c = df.merge(cluster_map, on=_DFU_COLS, how="left")

    # DFUs without cluster assignment → will use global best later
    has_cluster = df_c[df_c[cluster_col].notna()]
    no_cluster = df_c[df_c[cluster_col].isna()]

    # ── Compute expanding stats (causal, exec-lag-aware) ──────────────────
    if has_cluster.empty:
        return strategy_expanding(df, min_prior_months=min_prior_months)

    has_cluster = _expanding_stats(has_cluster)

    qualified = has_cluster[has_cluster["prior_count"] >= min_prior_months].copy()
    qualified["prior_wape"] = (
        qualified["cum_abs_err"] / qualified["cum_actual"].abs()
    )
    qualified = qualified[qualified["prior_wape"].notna()]

    if qualified.empty:
        return strategy_expanding(df, min_prior_months=min_prior_months)

    # ── Per-cluster: aggregate WAPE per (cluster, model) ──────────────────
    cluster_model_stats = (
        qualified.groupby([cluster_col, "model_id"], sort=False)
        .agg(
            total_abs_err=("cum_abs_err", "sum"),
            total_actual=("cum_actual", "sum"),
        )
        .reset_index()
    )
    cluster_model_stats["agg_wape"] = (
        cluster_model_stats["total_abs_err"]
        / cluster_model_stats["total_actual"].abs().clip(lower=1e-6)
    )

    # Best model per cluster
    cluster_champions: dict[Any, str] = {}
    for cluster_id, group in cluster_model_stats.groupby(cluster_col, sort=False):
        best = group.loc[group["agg_wape"].idxmin()]
        cluster_champions[cluster_id] = best["model_id"]

    # Global best model (fallback for unmapped DFUs)
    global_stats = cluster_model_stats.groupby("model_id", sort=False).agg(
        total_abs_err=("total_abs_err", "sum"),
        total_actual=("total_actual", "sum"),
    ).reset_index()
    global_stats["agg_wape"] = (
        global_stats["total_abs_err"]
        / global_stats["total_actual"].abs().clip(lower=1e-6)
    )
    global_best = global_stats.loc[global_stats["agg_wape"].idxmin(), "model_id"]

    # ── Assign cluster champion to each DFU-month ─────────────────────────
    # Map each qualified row to its cluster champion
    qualified["_cluster_champion"] = qualified[cluster_col].map(cluster_champions)

    # Filter to rows where model_id matches the cluster champion
    winners = qualified[
        qualified["model_id"] == qualified["_cluster_champion"]
    ].copy()
    winners = winners.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")

    parts: list[pd.DataFrame] = [winners[_OUTPUT_COLS]]

    # ── Handle unmapped DFUs (no cluster) with global best ────────────────
    if not no_cluster.empty:
        no_cluster_subset = no_cluster.drop(columns=[cluster_col])
        global_winners = no_cluster_subset[
            no_cluster_subset["model_id"] == global_best
        ].copy()
        if not global_winners.empty:
            global_winners["prior_wape"] = 0.0
            for col in _OUTPUT_COLS:
                if col not in global_winners.columns:
                    global_winners[col] = 0.0
            parts.append(global_winners[_OUTPUT_COLS])

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")
    combined = combined[_OUTPUT_COLS].reset_index(drop=True)

    # Build cluster diagnostics
    cluster_diag = {}
    for cid, group in cluster_model_stats.groupby(cluster_col, sort=False):
        best_model = cluster_champions.get(cid, "unknown")
        best_wape = float(group.loc[group["agg_wape"].idxmin(), "agg_wape"])
        cluster_diag[str(cid)] = {"champion": best_model, "agg_wape": round(best_wape * 100, 2)}

    # Count DFU-months per champion in the output
    model_counts = combined["model_id"].value_counts().to_dict()
    total = sum(model_counts.values())
    model_pct = {m: round(c / total * 100, 2) for m, c in sorted(model_counts.items(), key=lambda x: -x[1])}

    combined.attrs["weight_diagnostics"] = {
        "type": "per_cluster",
        "cluster_champions": {str(k): v for k, v in cluster_champions.items()},
        "global_fallback": global_best,
        "model_share_pct": model_pct,
        "cluster_details": cluster_diag,
    }
    return combined


# ---------------------------------------------------------------------------
# Strategy: cascade_ensemble
# ---------------------------------------------------------------------------

@register_strategy("cascade_ensemble")
def strategy_cascade_ensemble(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    solo_threshold: float = 0.10,
    mid_threshold: float = 0.25,
    mid_k: int = 2,
    wide_k: int = 5,
    weight_method: str = "inverse_wape",
    **kwargs: Any,
) -> pd.DataFrame:
    """Tiered cascade: adapt ensemble breadth to model confidence level.

    For each DFU-month, compute the best model's prior WAPE:
      - WAPE < solo_threshold (10%): trust best model solo
      - solo_threshold <= WAPE < mid_threshold (25%): blend top mid_k (2)
      - WAPE >= mid_threshold (25%): blend top wide_k (5) for safety

    Intuition: when the best model is already very good, adding more models
    dilutes its accuracy. When it's mediocre, hedging helps.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = _expanding_stats(df)

    qualified = df[df["prior_count"] >= min_prior_months].copy()
    qualified["prior_wape"] = qualified["cum_abs_err"] / qualified["cum_actual"].abs()
    qualified = qualified[qualified["prior_wape"].notna()]

    results: list[dict[str, Any]] = []
    for key, month_df in qualified.groupby(_DFU_MONTH_COLS, sort=False):
        item_id, customer_group, loc, startdate = key
        sorted_models = month_df.nsmallest(wide_k, "prior_wape")
        if len(sorted_models) == 0:
            continue

        best_wape = float(sorted_models["prior_wape"].iloc[0])

        # Determine cascade tier
        if best_wape < solo_threshold:
            top = sorted_models.head(1)
        elif best_wape < mid_threshold:
            top = sorted_models.head(min(mid_k, len(sorted_models)))
        else:
            top = sorted_models.head(min(wide_k, len(sorted_models)))

        if len(top) == 1:
            r = top.iloc[0]
            results.append({
                "item_id": item_id, "customer_group": customer_group,
                "loc": loc, "startdate": startdate,
                "model_id": r["model_id"], "prior_wape": best_wape,
                "basefcst_pref": r["basefcst_pref"],
                "tothist_dmd": r["tothist_dmd"],
            })
        else:
            weights = _compute_blend_weights(top["prior_wape"], weight_method)
            blended, actual, _ = _blend_forecasts(top, weights)
            results.append({
                "item_id": item_id, "customer_group": customer_group,
                "loc": loc, "startdate": startdate,
                "model_id": "cascade_ensemble", "prior_wape": best_wape,
                "basefcst_pref": blended, "tothist_dmd": actual,
            })

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: adversarial_filter
# ---------------------------------------------------------------------------

@register_strategy("adversarial_filter")
def strategy_adversarial_filter(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    outlier_z_threshold: float = 1.5,
    top_k: int = 3,
    weight_method: str = "inverse_wape",
    **kwargs: Any,
) -> pd.DataFrame:
    """Adversarial filter: exclude outlier models before ensembling.

    For each DFU-month, detect models whose current forecast is far from
    the consensus (z-score > threshold) and exclude them before blending.
    This removes the model most likely to blow up the forecast.

    Steps:
      1. Compute forecast z-score per model: (forecast - mean) / std
      2. Exclude models with |z| > outlier_z_threshold
      3. Rank remaining models by prior WAPE, blend top-K
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = _expanding_stats(df)

    qualified = df[df["prior_count"] >= min_prior_months].copy()
    qualified["prior_wape"] = qualified["cum_abs_err"] / qualified["cum_actual"].abs()
    qualified = qualified[qualified["prior_wape"].notna()]

    results: list[dict[str, Any]] = []
    for key, month_df in qualified.groupby(_DFU_MONTH_COLS, sort=False):
        item_id, customer_group, loc, startdate = key
        if len(month_df) < 2:
            # Only 1 model — no filtering possible
            r = month_df.iloc[0]
            results.append({
                "item_id": item_id, "customer_group": customer_group,
                "loc": loc, "startdate": startdate,
                "model_id": r["model_id"], "prior_wape": float(r["prior_wape"]),
                "basefcst_pref": r["basefcst_pref"],
                "tothist_dmd": r["tothist_dmd"],
            })
            continue

        # Compute forecast z-scores
        fcsts = month_df["basefcst_pref"].astype(float)
        mean_fcst = fcsts.mean()
        std_fcst = fcsts.std()
        if std_fcst > 1e-6:
            z_scores = ((fcsts - mean_fcst) / std_fcst).abs()
            filtered = month_df[z_scores <= outlier_z_threshold]
        else:
            filtered = month_df  # All forecasts identical — no outliers

        if len(filtered) == 0:
            filtered = month_df  # Fallback: keep all if all are "outliers"

        top = filtered.nsmallest(top_k, "prior_wape")
        if len(top) == 0:
            continue

        if len(top) == 1:
            r = top.iloc[0]
            results.append({
                "item_id": item_id, "customer_group": customer_group,
                "loc": loc, "startdate": startdate,
                "model_id": r["model_id"], "prior_wape": float(r["prior_wape"]),
                "basefcst_pref": r["basefcst_pref"],
                "tothist_dmd": r["tothist_dmd"],
            })
        else:
            weights = _compute_blend_weights(top["prior_wape"], weight_method)
            blended, actual, avg_wape = _blend_forecasts(top, weights)
            results.append({
                "item_id": item_id, "customer_group": customer_group,
                "loc": loc, "startdate": startdate,
                "model_id": "adversarial_filter",
                "prior_wape": avg_wape,
                "basefcst_pref": blended, "tothist_dmd": actual,
            })

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


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


# ---------------------------------------------------------------------------
# Strategy: bayesian_model_avg
# ---------------------------------------------------------------------------

@register_strategy("bayesian_model_avg")
def strategy_bayesian_model_avg(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    prior_weight: float = 1.0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Bayesian model averaging: update model probabilities with evidence.

    For each DFU, starts with a uniform prior across models and updates
    the posterior after each causally-available month using the likelihood
    of each model's forecast (Gaussian likelihood based on absolute error).

    The posterior-weighted blend is the final forecast.

    posterior(model) ∝ prior(model) × prod(likelihood(month_t | model))
    likelihood ∝ exp(-0.5 * (abs_err / scale)^2)

    The scale parameter is the mean absolute error across all models for
    that DFU, preventing numerical underflow.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = df.sort_values(_DFU_MODEL_COLS + ["startdate"]).copy()
    models = sorted(df["model_id"].unique())
    n_models = len(models)

    results: list[dict[str, Any]] = []
    weight_sums: dict[str, float] = {}
    weight_count = 0

    for dfu_key, dfu_df in df.groupby(_DFU_COLS, sort=False):
        item_id, customer_group, loc = dfu_key
        exec_lag = _get_exec_lag(dfu_df)
        dfu_months = sorted(dfu_df["startdate"].unique())

        # Initialize uniform prior
        log_posterior = {m: 0.0 for m in models}

        for i, current_month in enumerate(dfu_months):
            n_available = i - exec_lag
            if n_available < min_prior_months:
                continue

            # Update posterior with all newly available evidence
            # Available months: dfu_months[:n_available]
            # Use the most recent available month for the update
            update_month = dfu_months[n_available - 1]
            update_rows = dfu_df[dfu_df["startdate"] == update_month]

            if not update_rows.empty:
                errs = update_rows.set_index("model_id")["abs_err"]
                scale = float(errs.mean()) + 1e-6

                for m in models:
                    if m in errs.index:
                        err = float(errs[m])
                        # Log-likelihood: -0.5 * (err/scale)^2
                        log_posterior[m] += -0.5 * (err / scale) ** 2

            # Convert log-posterior to weights (softmax)
            max_lp = max(log_posterior.values())
            raw_weights = {
                m: np.exp(log_posterior[m] - max_lp) for m in models
            }
            total_w = sum(raw_weights.values())
            if total_w < 1e-12:
                weights = {m: 1.0 / n_models for m in models}
            else:
                weights = {m: w / total_w for m, w in raw_weights.items()}

            for m_w, w_val in weights.items():
                weight_sums[m_w] = weight_sums.get(m_w, 0.0) + w_val
            weight_count += 1

            # Apply Bayesian weights to current month's forecasts
            current_rows = dfu_df[dfu_df["startdate"] == current_month]
            blended = 0.0
            actual = None
            for m in models:
                m_row = current_rows[current_rows["model_id"] == m]
                if len(m_row) > 0:
                    blended += weights[m] * float(m_row["basefcst_pref"].iloc[0])
                    if actual is None:
                        actual = float(m_row["tothist_dmd"].iloc[0])

            if actual is not None:
                best_model = max(weights, key=weights.get)
                results.append({
                    "item_id": item_id, "customer_group": customer_group,
                    "loc": loc, "startdate": current_month,
                    "model_id": "bayesian_avg",
                    "prior_wape": weights.get(best_model, 0.0),
                    "basefcst_pref": blended, "tothist_dmd": actual,
                })

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    result_df = pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)
    if weight_count > 0:
        avg_weights = {m: round(s / weight_count * 100, 2) for m, s in sorted(weight_sums.items(), key=lambda x: -x[1])}
        result_df.attrs["weight_diagnostics"] = {
            "type": "bayesian_model_avg",
            "avg_model_weight_pct": avg_weights,
            "n_dfu_months_blended": weight_count,
        }
    return result_df


# ---------------------------------------------------------------------------
# Strategy: error_correcting
# ---------------------------------------------------------------------------

@register_strategy("error_correcting")
def strategy_error_correcting(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    correction_window: int = 3,
    correction_strength: float = 0.5,
    **kwargs: Any,
) -> pd.DataFrame:
    """Error-correcting ensemble: best model + learned bias correction.

    Picks the best model by expanding WAPE, then adds a correction term
    based on the model's recent systematic bias (signed mean error).

    corrected_forecast = best_model_forecast - correction_strength * recent_bias

    This captures cases where the best model consistently over- or under-
    forecasts and corrects for it.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = df.sort_values(_DFU_MODEL_COLS + ["startdate"]).copy()

    # Compute expanding stats for model ranking
    df_exp = _expanding_stats(df)

    # Also compute signed errors for bias correction
    df_exp["signed_err"] = df_exp["basefcst_pref"].astype(float) - df_exp["tothist_dmd"].astype(float)

    qualified = df_exp[df_exp["prior_count"] >= min_prior_months].copy()
    qualified["prior_wape"] = qualified["cum_abs_err"] / qualified["cum_actual"].abs()
    qualified = qualified[qualified["prior_wape"].notna()]

    results: list[dict[str, Any]] = []

    for dfu_key, dfu_df in qualified.groupby(_DFU_COLS, sort=False):
        item_id, customer_group, loc = dfu_key
        exec_lag = _get_exec_lag(dfu_df)

        for _, month_models in dfu_df.groupby("startdate", sort=True):
            if month_models.empty:
                continue

            # Pick best model by prior WAPE
            best_row = month_models.loc[month_models["prior_wape"].idxmin()]
            best_model = best_row["model_id"]
            startdate = best_row["startdate"]

            # Compute recent bias for the best model (last N causal months)
            model_history = dfu_df[
                (dfu_df["model_id"] == best_model)
                & (dfu_df["startdate"] < startdate)
            ].tail(correction_window)

            if len(model_history) >= 1:
                recent_bias = float(model_history["signed_err"].mean())
            else:
                recent_bias = 0.0

            # Correct the forecast
            raw_fcst = float(best_row["basefcst_pref"])
            corrected = raw_fcst - correction_strength * recent_bias

            results.append({
                "item_id": item_id, "customer_group": customer_group,
                "loc": loc, "startdate": startdate,
                "model_id": best_model,
                "prior_wape": float(best_row["prior_wape"]),
                "basefcst_pref": corrected,
                "tothist_dmd": float(best_row["tothist_dmd"]),
            })

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: shrinkage_blend
# ---------------------------------------------------------------------------

@register_strategy("shrinkage_blend")
def strategy_shrinkage_blend(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    shrinkage_intensity: float = 0.5,
    **kwargs: Any,
) -> pd.DataFrame:
    """Bates-Granger shrinkage blend: blend ALL models with shrinkage to equal.

    Forecast combination theory (Bates & Granger 1969) shows that blending
    forecasts with shrinkage toward equal weights often outperforms picking
    the single best or using unconstrained optimal weights.

    weight(model) = shrinkage * (1/N) + (1-shrinkage) * inverse_wape_weight

    shrinkage_intensity = 0: pure inverse-WAPE weights (aggressive)
    shrinkage_intensity = 1: equal weights (conservative)
    shrinkage_intensity = 0.5: balanced (recommended default)
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = _expanding_stats(df)

    qualified = df[df["prior_count"] >= min_prior_months].copy()
    qualified["prior_wape"] = qualified["cum_abs_err"] / qualified["cum_actual"].abs()
    qualified = qualified[qualified["prior_wape"].notna()]

    results: list[dict[str, Any]] = []
    weight_sums: dict[str, float] = {}
    weight_count = 0
    for key, month_df in qualified.groupby(_DFU_MONTH_COLS, sort=False):
        item_id, customer_group, loc, startdate = key
        n_models = len(month_df)
        if n_models == 0:
            continue

        # Inverse-WAPE weights
        inv_wapes = 1.0 / month_df["prior_wape"].clip(lower=1e-6)
        inv_weights = inv_wapes / inv_wapes.sum()

        # Equal weights
        equal_weight = 1.0 / n_models

        # Shrinkage blend
        weights = (
            shrinkage_intensity * equal_weight
            + (1.0 - shrinkage_intensity) * inv_weights
        )
        weights = weights / weights.sum()  # Re-normalize

        for model_id_w, w in zip(month_df["model_id"], weights):
            weight_sums[model_id_w] = weight_sums.get(model_id_w, 0.0) + float(w)
        weight_count += 1

        blended = (month_df["basefcst_pref"].astype(float) * weights).sum()
        actual = float(month_df["tothist_dmd"].iloc[0])
        avg_wape = float((month_df["prior_wape"] * weights).sum())

        results.append({
            "item_id": item_id, "customer_group": customer_group,
            "loc": loc, "startdate": startdate,
            "model_id": "shrinkage_blend",
            "prior_wape": avg_wape,
            "basefcst_pref": blended, "tothist_dmd": actual,
        })

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    result_df = pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)
    if weight_count > 0:
        avg_weights = {m: round(s / weight_count * 100, 2) for m, s in sorted(weight_sums.items(), key=lambda x: -x[1])}
        result_df.attrs["weight_diagnostics"] = {
            "type": "blend",
            "avg_model_weight_pct": avg_weights,
            "n_dfu_months_blended": weight_count,
            "shrinkage_intensity": shrinkage_intensity,
        }
    return result_df


# ---------------------------------------------------------------------------
# Strategy: dfu_strategy_router
# ---------------------------------------------------------------------------

@register_strategy("dfu_strategy_router")
def strategy_dfu_strategy_router(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    candidate_strategies: list[str] | None = None,
    eval_months: int = 3,
    **kwargs: Any,
) -> pd.DataFrame:
    """Per-DFU strategy router: learn which strategy works best per DFU.

    For each DFU, evaluate multiple strategies on recent months (walk-forward)
    and route that DFU to its best-performing strategy.

    Steps:
      1. For each DFU, run each candidate strategy on the DFU's data
      2. Evaluate WAPE on the last eval_months (causal walk-forward)
      3. Pick the strategy with lowest CV WAPE for this DFU
      4. Use that strategy's output for this DFU's predictions
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    if candidate_strategies is None:
        candidate_strategies = ["expanding", "rolling", "decay", "ensemble"]

    all_months = sorted(df["startdate"].unique())
    n_months = len(all_months)

    if n_months <= min_prior_months + eval_months:
        return strategy_expanding(df, min_prior_months=min_prior_months)

    eval_set = set(all_months[-eval_months:])

    # Pre-compute all candidate strategy outputs on the full data
    strategy_outputs: dict[str, pd.DataFrame] = {}
    for strat_name in candidate_strategies:
        fn = STRATEGY_REGISTRY.get(strat_name)
        if fn is None:
            continue
        strategy_outputs[strat_name] = fn(
            df, min_prior_months=min_prior_months,
        )

    if not strategy_outputs:
        return strategy_expanding(df, min_prior_months=min_prior_months)

    # For each DFU, find the best strategy via eval-month WAPE
    dfu_best: dict[tuple, str] = {}

    for dfu_key in df.groupby(_DFU_COLS, sort=False).groups:
        best_strat = candidate_strategies[0]
        best_wape = float("inf")

        for strat_name, output in strategy_outputs.items():
            dfu_output = output[
                (output["item_id"] == dfu_key[0])
                & (output["customer_group"] == dfu_key[1])
                & (output["loc"] == dfu_key[2])
            ]
            eval_output = dfu_output[dfu_output["startdate"].isin(eval_set)]
            if eval_output.empty:
                continue

            acc = compute_strategy_accuracy(eval_output)
            wape = acc.get("wape")
            if wape is not None and wape < best_wape:
                best_wape = wape
                best_strat = strat_name

        dfu_best[dfu_key] = best_strat

    # Collect winners from the best strategy for each DFU
    strat_to_dfus: dict[str, list[tuple]] = {}
    for dfu_key, strat_name in dfu_best.items():
        strat_to_dfus.setdefault(strat_name, []).append(dfu_key)

    parts: list[pd.DataFrame] = []
    for strat_name, dfu_keys in strat_to_dfus.items():
        output = strategy_outputs.get(strat_name)
        if output is None or output.empty:
            continue
        key_df = pd.DataFrame(dfu_keys, columns=_DFU_COLS)
        matched = output.merge(key_df, on=_DFU_COLS, how="inner")
        if not matched.empty:
            parts.append(matched[_OUTPUT_COLS])

    if not parts:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")
    return combined[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: stacked_strategies
# ---------------------------------------------------------------------------

@register_strategy("stacked_strategies")
def strategy_stacked_strategies(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    base_strategies: list[str] | None = None,
    blend_method: str = "inverse_wape",
    eval_months: int = 3,
    **kwargs: Any,
) -> pd.DataFrame:
    """Stacked strategies: run multiple strategies, blend their outputs.

    Instead of picking one strategy, runs several base strategies and
    blends their forecasts weighted by recent walk-forward accuracy.

    This is a meta-ensemble of strategies rather than models.

    Steps:
      1. Run each base strategy on the full data
      2. Evaluate each strategy's accuracy on the last eval_months
      3. Compute inverse-WAPE weights across strategies
      4. For each DFU-month, blend the strategy outputs using these weights
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    if base_strategies is None:
        base_strategies = [
            "expanding", "rolling", "ensemble", "adaptive_ensemble",
        ]

    all_months = sorted(df["startdate"].unique())
    n_months = len(all_months)

    # Run all base strategies
    strat_results: dict[str, pd.DataFrame] = {}
    for strat_name in base_strategies:
        fn = STRATEGY_REGISTRY.get(strat_name)
        if fn is None:
            continue
        output = fn(df, min_prior_months=min_prior_months)
        if not output.empty:
            strat_results[strat_name] = output

    if not strat_results:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    if len(strat_results) == 1:
        return list(strat_results.values())[0]

    # Compute strategy-level weights via eval-month accuracy
    eval_set = set(all_months[-eval_months:]) if n_months > eval_months else set()
    strat_weights: dict[str, float] = {}

    for strat_name, output in strat_results.items():
        if eval_set:
            eval_output = output[output["startdate"].isin(eval_set)]
        else:
            eval_output = output
        acc = compute_strategy_accuracy(eval_output)
        wape = acc.get("wape")
        if wape is not None and wape > 0:
            strat_weights[strat_name] = 1.0 / wape
        else:
            strat_weights[strat_name] = 1.0

    total_w = sum(strat_weights.values())
    strat_weights = {k: v / total_w for k, v in strat_weights.items()}

    # Blend strategy outputs per DFU-month
    # Collect all DFU-month keys across all strategies
    all_dfu_months = set()
    for output in strat_results.values():
        keys = output[_DFU_MONTH_COLS].apply(tuple, axis=1)
        all_dfu_months.update(keys)

    results: list[dict[str, Any]] = []
    # Build lookup dicts for fast access
    strat_lookups: dict[str, dict[tuple, pd.Series]] = {}
    for strat_name, output in strat_results.items():
        lookup = {}
        for _, row in output.iterrows():
            k = (row["item_id"], row["customer_group"], row["loc"], row["startdate"])
            lookup[k] = row
        strat_lookups[strat_name] = lookup

    for dfu_month_key in all_dfu_months:
        blended = 0.0
        actual = None
        total_weight = 0.0

        for strat_name, lookup in strat_lookups.items():
            row = lookup.get(dfu_month_key)
            if row is not None:
                w = strat_weights.get(strat_name, 0.0)
                blended += w * float(row["basefcst_pref"])
                total_weight += w
                if actual is None:
                    actual = float(row["tothist_dmd"])

        if total_weight > 0 and actual is not None:
            blended /= total_weight
            results.append({
                "item_id": dfu_month_key[0],
                "customer_group": dfu_month_key[1],
                "loc": dfu_month_key[2],
                "startdate": dfu_month_key[3],
                "model_id": "stacked_strategies",
                "prior_wape": 0.0,
                "basefcst_pref": blended,
                "tothist_dmd": actual,
            })

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: cluster_regime_hybrid
# ---------------------------------------------------------------------------

@register_strategy("cluster_regime_hybrid")
def strategy_cluster_regime_hybrid(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    dfu_features: pd.DataFrame | None = None,
    cluster_col: str = "ml_cluster",
    variance_window: int = 4,
    variance_threshold: float = 2.0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Cluster + regime hybrid: per-cluster strategy with regime switching.

    Combines per_cluster structural grouping with regime_adaptive temporal
    adaptation. Within each cluster:
      - Stable regime DFU-months: use expanding (long-memory, stable)
      - Shift regime DFU-months: use rolling (short-memory, adaptive)

    Falls back to regime_adaptive when cluster features unavailable.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    if dfu_features is None or cluster_col not in dfu_features.columns:
        return strategy_regime_adaptive(
            df, min_prior_months=min_prior_months,
            variance_window=variance_window,
            variance_threshold=variance_threshold,
        )

    # Merge cluster labels
    cluster_map = dfu_features[_DFU_COLS + [cluster_col]].drop_duplicates(
        subset=_DFU_COLS, keep="first",
    )
    df_c = df.merge(cluster_map, on=_DFU_COLS, how="left")

    # Run regime_adaptive per cluster
    parts: list[pd.DataFrame] = []
    for cluster_id, cluster_df in df_c.groupby(cluster_col, sort=False):
        if cluster_df.empty:
            continue
        # Drop cluster column before passing to regime_adaptive
        cluster_data = cluster_df.drop(columns=[cluster_col])
        winners = strategy_regime_adaptive(
            cluster_data,
            min_prior_months=min_prior_months,
            variance_window=variance_window,
            variance_threshold=variance_threshold,
        )
        if not winners.empty:
            parts.append(winners)

    # Handle DFUs without cluster
    no_cluster = df_c[df_c[cluster_col].isna()].drop(columns=[cluster_col])
    if not no_cluster.empty:
        winners = strategy_regime_adaptive(
            no_cluster, min_prior_months=min_prior_months,
        )
        if not winners.empty:
            parts.append(winners)

    if not parts:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first")
    return combined[_OUTPUT_COLS].reset_index(drop=True)

# ===========================================================================
# REINFORCEMENT LEARNING STRATEGIES
# ===========================================================================


def _update_thompson_posteriors(
    alphas: dict[str, float],
    betas: dict[str, float],
    models: list[str],
    oracle_best: str,
    discount: float,
) -> None:
    """Discount all posteriors and reward the oracle-best model.

    Shared by ``thompson_sampling`` and ``thompson_ensemble``. Mutates
    ``alphas`` and ``betas`` in-place.
    """
    for m in models:
        alphas[m] *= discount
        betas[m] *= discount
        alphas[m] = max(alphas[m], 0.1)
        betas[m] = max(betas[m], 0.1)
    for m in models:
        if m == oracle_best:
            alphas[m] += 1.0
        else:
            betas[m] += 1.0


# ---------------------------------------------------------------------------
# Strategy: thompson_sampling
# ---------------------------------------------------------------------------

@register_strategy("thompson_sampling")
def strategy_thompson_sampling(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 2,
    discount: float = 0.95,
    n_samples: int = 100,
    **kwargs: Any,
) -> pd.DataFrame:
    """Thompson Sampling bandit with discounted Bayesian updating.

    Maintains a Beta(alpha, beta) posterior per (DFU, model) representing
    the probability that the model is "good" for this DFU. At each month:

      1. Sample from each model's posterior: theta ~ Beta(alpha, beta)
      2. Pick the model with highest sampled theta
      3. Observe reward (1 if model had lowest error, 0 otherwise)
      4. Update posterior: alpha += reward, beta += (1 - reward)
      5. Apply discount: alpha *= discount, beta *= discount
         (allows forgetting old observations -> adapts to regime changes)

    The discount factor controls exploration vs exploitation:
      - discount = 1.0: never forget -> converges to greedy (pure exploit)
      - discount = 0.90: forget quickly -> more exploration, adapts faster
      - discount = 0.95: balanced (recommended)

    Parameters
    ----------
    min_prior_months : int
        Months before the bandit starts making selections.
    discount : float
        Discount factor applied to alpha/beta each month (0 < d <= 1).
    n_samples : int
        Number of Thompson samples to average for stability.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = df.sort_values(_DFU_COLS + ["startdate"]).copy()
    models = sorted(df["model_id"].unique())
    rng = np.random.RandomState(42)

    results: list[dict[str, Any]] = []

    for dfu_key, dfu_df in df.groupby(_DFU_COLS, sort=False):
        item_id, customer_group, loc = dfu_key
        exec_lag = _get_exec_lag(dfu_df)
        dfu_months = sorted(dfu_df["startdate"].unique())

        # Initialize Beta(1, 1) = uniform prior for each model
        alphas = {m: 1.0 for m in models}
        betas = {m: 1.0 for m in models}

        for i, current_month in enumerate(dfu_months):
            n_available = i - exec_lag
            current_rows = dfu_df[dfu_df["startdate"] == current_month]

            if n_available < min_prior_months or current_rows.empty:
                continue

            # -- Thompson Sampling: sample from posteriors --
            model_scores: dict[str, float] = {}
            for m in models:
                samples = rng.beta(
                    max(alphas[m], 0.01), max(betas[m], 0.01), size=n_samples,
                )
                model_scores[m] = float(samples.mean())

            best_model = max(model_scores, key=model_scores.get)

            winner_row = current_rows[current_rows["model_id"] == best_model]
            if len(winner_row) == 0:
                available = current_rows["model_id"].unique()
                available_scores = {m: model_scores[m] for m in available if m in model_scores}
                if not available_scores:
                    continue
                best_model = max(available_scores, key=available_scores.get)
                winner_row = current_rows[current_rows["model_id"] == best_model]
                if len(winner_row) == 0:
                    continue

            r = winner_row.iloc[0]
            results.append({
                "item_id": item_id, "customer_group": customer_group,
                "loc": loc, "startdate": current_month,
                "model_id": best_model,
                "prior_wape": model_scores.get(best_model, 0.0),
                "basefcst_pref": r["basefcst_pref"],
                "tothist_dmd": r["tothist_dmd"],
            })

            # -- Update posteriors with PREVIOUS available month --
            update_idx = n_available - 1
            if update_idx >= 0:
                update_month = dfu_months[update_idx]
                update_rows = dfu_df[dfu_df["startdate"] == update_month]

                if not update_rows.empty:
                    errors = update_rows.set_index("model_id")["abs_err"]
                    if len(errors) > 0:
                        _update_thompson_posteriors(
                            alphas, betas, models, errors.idxmin(), discount,
                        )

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: linucb
# ---------------------------------------------------------------------------

@register_strategy("linucb")
def strategy_linucb(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 3,
    alpha_ucb: float = 1.0,
    **kwargs: Any,
) -> pd.DataFrame:
    """LinUCB contextual bandit: model selection conditioned on DFU context.

    For each DFU-month, builds a context vector from recent demand features
    and selects the model with the highest Upper Confidence Bound:

      UCB(model) = theta^T @ context + alpha * sqrt(context^T @ A_inv @ context)

    Context features (computed causally per DFU-month):
      - mean_demand (scale-normalized), cv_demand, trend, n_zeros,
        month_sin, month_cos

    Parameters
    ----------
    alpha_ucb : float
        Exploration parameter. Higher = more exploration.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = df.sort_values(_DFU_COLS + ["startdate"]).copy()
    models = sorted(df["model_id"].unique())
    n_features = 6

    results: list[dict[str, Any]] = []

    for dfu_key, dfu_df in df.groupby(_DFU_COLS, sort=False):
        item_id, customer_group, loc = dfu_key
        exec_lag = _get_exec_lag(dfu_df)
        dfu_months = sorted(dfu_df["startdate"].unique())

        A = {m: np.eye(n_features) for m in models}
        b = {m: np.zeros(n_features) for m in models}

        for i, current_month in enumerate(dfu_months):
            n_available = i - exec_lag
            if n_available < min_prior_months:
                continue

            current_rows = dfu_df[dfu_df["startdate"] == current_month]
            if current_rows.empty:
                continue

            # -- Build context vector from causal demand history --
            prior_months = dfu_months[:n_available]
            demand_vals = []
            for pm in prior_months:
                pm_rows = dfu_df[dfu_df["startdate"] == pm]
                if not pm_rows.empty:
                    demand_vals.append(float(pm_rows["tothist_dmd"].iloc[0]))

            if len(demand_vals) < 2:
                continue

            demand_arr = np.array(demand_vals, dtype=float)
            mean_d = demand_arr.mean()
            std_d = demand_arr.std()
            cv_d = std_d / max(abs(mean_d), 1e-6)
            if len(demand_arr) > 2:
                trend = float(np.polyfit(range(len(demand_arr)), demand_arr, 1)[0])
            else:
                trend = 0.0
            n_zeros = float((demand_arr == 0).mean())
            month_num = pd.Timestamp(current_month).month
            month_sin = np.sin(2.0 * np.pi * month_num / 12.0)
            month_cos = np.cos(2.0 * np.pi * month_num / 12.0)

            context = np.array([
                mean_d / max(abs(mean_d), 1.0),
                cv_d, trend / max(abs(mean_d), 1.0),
                n_zeros, month_sin, month_cos,
            ], dtype=float)

            # -- Compute UCB for each model --
            ucb_scores: dict[str, float] = {}
            for m in models:
                A_inv = np.linalg.inv(A[m])
                theta = A_inv @ b[m]
                exploitation = float(theta @ context)
                exploration = alpha_ucb * float(np.sqrt(context @ A_inv @ context))
                ucb_scores[m] = exploitation + exploration

            best_model = max(ucb_scores, key=ucb_scores.get)

            winner_row = current_rows[current_rows["model_id"] == best_model]
            if len(winner_row) == 0:
                available = current_rows["model_id"].unique()
                available_scores = {m: ucb_scores[m] for m in available if m in ucb_scores}
                if not available_scores:
                    continue
                best_model = max(available_scores, key=available_scores.get)
                winner_row = current_rows[current_rows["model_id"] == best_model]
                if len(winner_row) == 0:
                    continue

            r = winner_row.iloc[0]
            results.append({
                "item_id": item_id, "customer_group": customer_group,
                "loc": loc, "startdate": current_month,
                "model_id": best_model,
                "prior_wape": ucb_scores.get(best_model, 0.0),
                "basefcst_pref": r["basefcst_pref"],
                "tothist_dmd": r["tothist_dmd"],
            })

            # -- Update with previous available month's reward --
            update_idx = n_available - 1
            if update_idx >= 0:
                update_month = dfu_months[update_idx]
                update_rows = dfu_df[dfu_df["startdate"] == update_month]
                if not update_rows.empty:
                    errors = update_rows.set_index("model_id")["abs_err"]
                    if len(errors) > 0:
                        max_err = float(errors.max()) + 1e-6
                        for m in models:
                            if m in errors.index:
                                reward = 1.0 - float(errors[m]) / max_err
                                A[m] += np.outer(context, context)
                                b[m] += reward * context

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: exp3
# ---------------------------------------------------------------------------

@register_strategy("exp3")
def strategy_exp3(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 2,
    gamma: float = 0.1,
    **kwargs: Any,
) -> pd.DataFrame:
    """EXP3 adversarial bandit: robust to non-stationary rewards.

    Makes NO assumptions about how rewards change. Uses multiplicative
    weight updates and is provably optimal against adversarial sequences.

    Parameters
    ----------
    gamma : float
        Exploration rate (0 to 1). Higher = more uniform exploration.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = df.sort_values(_DFU_COLS + ["startdate"]).copy()
    models = sorted(df["model_id"].unique())
    n_models = len(models)

    results: list[dict[str, Any]] = []

    for dfu_key, dfu_df in df.groupby(_DFU_COLS, sort=False):
        item_id, customer_group, loc = dfu_key
        exec_lag = _get_exec_lag(dfu_df)
        dfu_months = sorted(dfu_df["startdate"].unique())

        log_weights = {m: 0.0 for m in models}

        for i, current_month in enumerate(dfu_months):
            n_available = i - exec_lag
            if n_available < min_prior_months:
                continue

            current_rows = dfu_df[dfu_df["startdate"] == current_month]
            if current_rows.empty:
                continue

            # -- Compute mixed probabilities --
            max_lw = max(log_weights.values())
            exp_weights = {m: np.exp(log_weights[m] - max_lw) for m in models}
            total_w = sum(exp_weights.values())
            base_probs = {m: exp_weights[m] / total_w for m in models}

            probs = {
                m: (1.0 - gamma) * base_probs[m] + gamma / n_models
                for m in models
            }

            best_model = max(probs, key=probs.get)

            winner_row = current_rows[current_rows["model_id"] == best_model]
            if len(winner_row) == 0:
                available = current_rows["model_id"].unique()
                available_probs = {m: probs[m] for m in available if m in probs}
                if not available_probs:
                    continue
                best_model = max(available_probs, key=available_probs.get)
                winner_row = current_rows[current_rows["model_id"] == best_model]
                if len(winner_row) == 0:
                    continue

            r = winner_row.iloc[0]
            results.append({
                "item_id": item_id, "customer_group": customer_group,
                "loc": loc, "startdate": current_month,
                "model_id": best_model,
                "prior_wape": probs.get(best_model, 0.0),
                "basefcst_pref": r["basefcst_pref"],
                "tothist_dmd": r["tothist_dmd"],
            })

            # -- Update weights with previous available month --
            update_idx = n_available - 1
            if update_idx >= 0:
                update_month = dfu_months[update_idx]
                update_rows = dfu_df[dfu_df["startdate"] == update_month]
                if not update_rows.empty:
                    errors = update_rows.set_index("model_id")["abs_err"]
                    if len(errors) > 0:
                        max_err = float(errors.max()) + 1e-6
                        for m in models:
                            if m in errors.index:
                                reward = 1.0 - float(errors[m]) / max_err
                                p_m = probs.get(m, gamma / n_models)
                                reward_hat = reward / max(p_m, 1e-6)
                                log_weights[m] += gamma * reward_hat / n_models

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: thompson_ensemble
# ---------------------------------------------------------------------------

@register_strategy("thompson_ensemble")
def strategy_thompson_ensemble(
    df: pd.DataFrame,
    *,
    min_prior_months: int = 2,
    discount: float = 0.95,
    top_k: int = 3,
    n_samples: int = 100,
    **kwargs: Any,
) -> pd.DataFrame:
    """Thompson Sampling + ensemble: sample posteriors, blend top-K.

    Combines Thompson Sampling's exploration with ensemble hedging:
      1. Sample from each model's Beta posterior
      2. Select top-K models by sampled value
      3. Blend their forecasts weighted by sampled probabilities
      4. Update posteriors with causal reward feedback
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = df.sort_values(_DFU_COLS + ["startdate"]).copy()
    models = sorted(df["model_id"].unique())
    rng = np.random.RandomState(42)

    results: list[dict[str, Any]] = []

    for dfu_key, dfu_df in df.groupby(_DFU_COLS, sort=False):
        item_id, customer_group, loc = dfu_key
        exec_lag = _get_exec_lag(dfu_df)
        dfu_months = sorted(dfu_df["startdate"].unique())

        alphas = {m: 1.0 for m in models}
        betas = {m: 1.0 for m in models}

        for i, current_month in enumerate(dfu_months):
            n_available = i - exec_lag
            if n_available < min_prior_months:
                continue

            current_rows = dfu_df[dfu_df["startdate"] == current_month]
            if current_rows.empty:
                continue

            model_scores: dict[str, float] = {}
            for m in models:
                samples = rng.beta(
                    max(alphas[m], 0.01), max(betas[m], 0.01), size=n_samples,
                )
                model_scores[m] = float(samples.mean())

            available_models = [
                m for m in models
                if len(current_rows[current_rows["model_id"] == m]) > 0
            ]
            if not available_models:
                continue

            available_scores = {m: model_scores[m] for m in available_models}
            sorted_m = sorted(available_scores, key=available_scores.get, reverse=True)
            top_models = sorted_m[:min(top_k, len(sorted_m))]

            top_scores = np.array([available_scores[m] for m in top_models])
            weights = top_scores / top_scores.sum()

            blended = 0.0
            actual = None
            for j, m in enumerate(top_models):
                m_row = current_rows[current_rows["model_id"] == m].iloc[0]
                blended += weights[j] * float(m_row["basefcst_pref"])
                if actual is None:
                    actual = float(m_row["tothist_dmd"])

            if actual is not None:
                results.append({
                    "item_id": item_id, "customer_group": customer_group,
                    "loc": loc, "startdate": current_month,
                    "model_id": "thompson_ensemble",
                    "prior_wape": float(top_scores[0]),
                    "basefcst_pref": blended, "tothist_dmd": actual,
                })

            # Update posteriors
            update_idx = n_available - 1
            if update_idx >= 0:
                update_month = dfu_months[update_idx]
                update_rows = dfu_df[dfu_df["startdate"] == update_month]
                if not update_rows.empty:
                    errors = update_rows.set_index("model_id")["abs_err"]
                    if len(errors) > 0:
                        _update_thompson_posteriors(
                            alphas, betas, models, errors.idxmin(), discount,
                        )

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)
