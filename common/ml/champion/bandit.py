"""Reinforcement learning bandit champion strategies.

Includes:
  - ``thompson_sampling``  — Discounted Beta-Bernoulli Thompson sampling
  - ``linucb``             — Contextual UCB bandit conditioned on demand context
  - ``exp3``               — Adversarial multiplicative-weights bandit
  - ``thompson_ensemble``  — Thompson sampling + top-K ensemble blend
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.ml.champion.helpers import _get_exec_lag
from common.ml.champion.registry import (
    _DFU_COLS,
    _OUTPUT_COLS,
    register_strategy,
)


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
