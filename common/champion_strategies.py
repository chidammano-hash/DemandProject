"""Champion model selection strategies.

All strategies take a DataFrame of per-DFU per-month per-model errors
and return per-DFU per-month winner selections.

Input DataFrame schema (monthly_errors):
    dmdunit, dmdgroup, loc, startdate, model_id,
    basefcst_pref, tothist_dmd, abs_err
    [optional: execution_lag, fcstdate]

Output DataFrame schema:
    dmdunit, dmdgroup, loc, startdate, model_id,
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

_DFU_COLS = ["dmdunit", "dmdgroup", "loc"]
_DFU_MONTH_COLS = ["dmdunit", "dmdgroup", "loc", "startdate"]
_DFU_MODEL_COLS = ["dmdunit", "dmdgroup", "loc", "model_id"]

_OUTPUT_COLS = [
    "dmdunit", "dmdgroup", "loc", "startdate",
    "model_id", "prior_wape", "basefcst_pref", "tothist_dmd",
]


def _get_exec_lag(group: pd.DataFrame) -> int:
    """Return execution_lag for a DFU group; defaults to 0 if column absent."""
    if "execution_lag" in group.columns and len(group) > 0:
        val = group["execution_lag"].iloc[0]
        return int(val) if pd.notna(val) else 0
    return 0


def compute_strategy_accuracy(winners_df: pd.DataFrame) -> dict[str, Any]:
    """Compute overall WAPE and accuracy from a winners DataFrame.

    Uses the standard formula: WAPE = SUM(|F-A|) / |SUM(A)| * 100.
    """
    if len(winners_df) == 0:
        return {"wape": None, "accuracy_pct": None, "n_dfu_months": 0}

    abs_err = (winners_df["basefcst_pref"] - winners_df["tothist_dmd"]).abs().sum()
    total_actual = winners_df["tothist_dmd"].sum()

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
        dmdunit, dmdgroup, loc = dfu_key
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

            for model_id, model_df in dfu_df[
                dfu_df["startdate"].isin(prior_months_available)
            ].groupby("model_id", sort=False):
                if len(model_df) < min_prior_months:
                    continue

                model_df = model_df.sort_values("startdate")
                # Distance: 0 = most recent prior month, increasing toward the past
                distances = [
                    n_available - 1 - list(prior_months_available).index(m)
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
                        "dmdunit": dmdunit,
                        "dmdgroup": dmdgroup,
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
        dmdunit, dmdgroup, loc, startdate = key
        top = month_df.nsmallest(top_k, "prior_wape")
        if len(top) == 0:
            continue

        if weight_method == "inverse_wape":
            inv_wapes = 1.0 / top["prior_wape"].clip(lower=1e-6)
            weights = inv_wapes / inv_wapes.sum()
        else:  # equal
            weights = pd.Series(
                [1.0 / len(top)] * len(top), index=top.index,
            )

        blended_fcst = (top["basefcst_pref"].astype(float) * weights).sum()
        actual = float(top["tothist_dmd"].iloc[0])
        avg_wape = float((top["prior_wape"] * weights).sum())

        results.append({
            "dmdunit": dmdunit,
            "dmdgroup": dmdgroup,
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
        print("  Meta-learner model not found, falling back to expanding strategy")
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

    # Compute per-model rolling stats with exec-lag-aware causal shift
    def _apply_meta_rolling(group: pd.DataFrame) -> pd.DataFrame:
        exec_lag = _get_exec_lag(group)
        shift_n = exec_lag + 1
        g = group.sort_values("startdate").copy()
        shifted_err = g["abs_err"].shift(shift_n)
        shifted_act = g["tothist_dmd"].shift(shift_n)
        shifted_bias = (g["basefcst_pref"] - g["tothist_dmd"]).shift(shift_n)
        g["_roll_abs_err"] = shifted_err.rolling(
            window=performance_window, min_periods=1
        ).sum()
        g["_roll_actual"] = shifted_act.rolling(
            window=performance_window, min_periods=1
        ).sum()
        g["_roll_bias_num"] = shifted_bias.rolling(
            window=performance_window, min_periods=1
        ).sum()
        g["_prior_count"] = shifted_err.expanding(min_periods=1).count()
        return g

    # Apply meta rolling stats per DFU-model group (explicit loop avoids FutureWarning)
    meta_groups = []
    for _, group in df.groupby(_DFU_MODEL_COLS, sort=False):
        meta_groups.append(_apply_meta_rolling(group))
    df = pd.concat(meta_groups, ignore_index=True)

    df["_roll_wape"] = df["_roll_abs_err"] / df["_roll_actual"].abs().clip(lower=1e-6)
    df["_roll_bias"] = df["_roll_bias_num"] / df["_roll_actual"].abs().clip(lower=1e-6)

    dfu_months = df[df["_prior_count"] >= min_prior_months][
        _DFU_MONTH_COLS
    ].drop_duplicates()

    pivoted = dfu_months.copy()
    for model_id in models:
        model_df = df[df["model_id"] == model_id][
            _DFU_MONTH_COLS + ["_roll_wape", "_roll_bias"]
        ].rename(columns={
            "_roll_wape": f"roll_wape_{model_id}",
            "_roll_bias": f"roll_bias_{model_id}",
        })
        pivoted = pivoted.merge(model_df, on=_DFU_MONTH_COLS, how="left")

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

    # Apply demand stats per DFU group (explicit loop avoids FutureWarning)
    demand_groups = []
    for _, group in demand_agg.groupby(_DFU_COLS, sort=False):
        g = group.sort_values("startdate").copy()
        dfu_key = (g.iloc[0]["dmdunit"], g.iloc[0]["dmdgroup"], g.iloc[0]["loc"])
        exec_lag = exec_lag_map.get(dfu_key, 0)
        shift_n = exec_lag + 1
        shifted = g["avg_demand"].shift(shift_n)
        g["mean_qty"] = shifted.expanding(min_periods=1).mean()
        g["cv_demand"] = shifted.expanding(min_periods=1).std() / g["mean_qty"].clip(lower=1e-6)
        demand_groups.append(g)
    demand_agg = pd.concat(demand_groups, ignore_index=True)

    pivoted = pivoted.merge(
        demand_agg[_DFU_MONTH_COLS + ["mean_qty", "cv_demand"]],
        on=_DFU_MONTH_COLS,
        how="left",
    )

    pivoted["month"] = pivoted["startdate"].dt.month
    pivoted["quarter"] = pivoted["startdate"].dt.quarter
    pivoted["month_sin"] = np.sin(2 * np.pi * pivoted["month"] / 12)
    pivoted["month_cos"] = np.cos(2 * np.pi * pivoted["month"] / 12)

    pivoted = pivoted.merge(dfu_features, on=_DFU_COLS, how="left")

    return pivoted
