"""Weight-learning, ensemble-shaping, and bias-correction champion strategies.

Includes:
  - ``learned_blend``       — Ridge over recent months
  - ``ridge_blend``         — Ridge with per-DFU expanding history
  - ``shrinkage_blend``     — Bates-Granger shrinkage to equal weights
  - ``bayesian_model_avg``  — Bayesian posterior averaging
  - ``error_correcting``    — Best model + recent signed-bias correction
  - ``adaptive_ensemble``   — Variable-K ensemble based on WAPE spread
  - ``uncertainty_aware``   — Risk-adjusted score using error std-dev
  - ``diverse_ensemble``    — Same-family correlation-penalised greedy ensemble
  - ``cascade_ensemble``    — Tiered top-K selection by best WAPE confidence
  - ``adversarial_filter``  — Outlier-forecast filter then top-K blend
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.core.constants import FORECAST_QTY_COL
from common.ml.champion.basic import strategy_expanding
from common.ml.champion.helpers import (
    _blend_forecasts,
    _compute_blend_weights,
    _expanding_stats,
    _expanding_uncertainty_stats,
    _get_exec_lag,
    _MODEL_FAMILIES,
    _resolve_fallback_rows,
    make_blend_row,
)
from common.ml.champion.registry import (
    _DFU_COLS,
    _DFU_MODEL_COLS,
    _DFU_MONTH_COLS,
    _OUTPUT_COLS,
    register_strategy,
)


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
                values=FORECAST_QTY_COL,
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
                        model_row[FORECAST_QTY_COL].iloc[0],
                    )

            actual = float(current_rows["tothist_dmd"].iloc[0])

            results.append(make_blend_row(
                item_id, customer_group, loc, current_month,
                "learned_blend", 0.0, blended_fcst, actual,
            ))

    # ── Fallback: expanding strategy for DFUs with insufficient history ──
    _resolve_fallback_rows(fallback_rows, results, min_prior_months=1)

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
                values=FORECAST_QTY_COL,
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
                        model_row[FORECAST_QTY_COL].iloc[0],
                    )

            actual = float(current_rows["tothist_dmd"].iloc[0])

            results.append(make_blend_row(
                item_id, customer_group, loc, current_month,
                "ridge_blend", 0.0, blended_fcst, actual,
            ))

    # ── Fallback: expanding strategy for DFU-months with insufficient data ──
    _resolve_fallback_rows(fallback_rows, results, min_prior_months=min_prior_months)

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

        blended = (month_df[FORECAST_QTY_COL].astype(float) * weights).sum()
        actual = float(month_df["tothist_dmd"].iloc[0])
        avg_wape = float((month_df["prior_wape"] * weights).sum())

        results.append(make_blend_row(
            item_id, customer_group, loc, startdate,
            "shrinkage_blend", avg_wape, blended, actual,
        ))

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
                    blended += weights[m] * float(m_row[FORECAST_QTY_COL].iloc[0])
                    if actual is None:
                        actual = float(m_row["tothist_dmd"].iloc[0])

            if actual is not None:
                best_model = max(weights, key=weights.get)
                results.append(make_blend_row(
                    item_id, customer_group, loc, current_month,
                    "bayesian_avg", weights.get(best_model, 0.0),
                    blended, actual,
                ))

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
    df_exp["signed_err"] = df_exp[FORECAST_QTY_COL].astype(float) - df_exp["tothist_dmd"].astype(float)

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
            raw_fcst = float(best_row[FORECAST_QTY_COL])
            corrected = raw_fcst - correction_strength * recent_bias

            results.append(make_blend_row(
                item_id, customer_group, loc, startdate,
                best_model, float(best_row["prior_wape"]),
                corrected, float(best_row["tothist_dmd"]),
            ))

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


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

        results.append(make_blend_row(
            item_id, customer_group, loc, startdate,
            "ensemble", avg_wape, blended_fcst, actual,
        ))

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

        blended_fcst = (top[FORECAST_QTY_COL].astype(float) * weights).sum()
        actual = float(top["tothist_dmd"].iloc[0])
        avg_wape = float((top["prior_wape"] * weights).sum())

        results.append(make_blend_row(
            item_id, customer_group, loc, startdate,
            "uncertainty_ensemble", avg_wape, blended_fcst, actual,
        ))

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Strategy: diverse_ensemble
# ---------------------------------------------------------------------------

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

        results.append(make_blend_row(
            item_id, customer_group, loc, startdate,
            "diverse_ensemble", avg_wape, blended_fcst, actual,
        ))

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)


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
            results.append(make_blend_row(
                item_id, customer_group, loc, startdate,
                r["model_id"], best_wape,
                r[FORECAST_QTY_COL], r["tothist_dmd"],
            ))
        else:
            weights = _compute_blend_weights(top["prior_wape"], weight_method)
            blended, actual, _ = _blend_forecasts(top, weights)
            results.append(make_blend_row(
                item_id, customer_group, loc, startdate,
                "cascade_ensemble", best_wape, blended, actual,
            ))

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
            results.append(make_blend_row(
                item_id, customer_group, loc, startdate,
                r["model_id"], float(r["prior_wape"]),
                r[FORECAST_QTY_COL], r["tothist_dmd"],
            ))
            continue

        # Compute forecast z-scores
        fcsts = month_df[FORECAST_QTY_COL].astype(float)
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
            results.append(make_blend_row(
                item_id, customer_group, loc, startdate,
                r["model_id"], float(r["prior_wape"]),
                r[FORECAST_QTY_COL], r["tothist_dmd"],
            ))
        else:
            weights = _compute_blend_weights(top["prior_wape"], weight_method)
            blended, actual, avg_wape = _blend_forecasts(top, weights)
            results.append(make_blend_row(
                item_id, customer_group, loc, startdate,
                "adversarial_filter", avg_wape, blended, actual,
            ))

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)
