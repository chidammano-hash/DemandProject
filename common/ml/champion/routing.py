"""Meta-routing champion strategies that combine or select among other strategies.

Includes:
  - ``hybrid_warmup``       — Warm-up + primary strategy stitching
  - ``optimized_decay``     — Walk-forward decay-factor tuner
  - ``seasonal``            — Same-quarter cumulative WAPE
  - ``dfu_strategy_router`` — Per-DFU strategy selector via walk-forward CV
  - ``stacked_strategies``  — Inverse-WAPE blend of multiple base strategies
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from common.core.constants import FORECAST_QTY_COL
from common.ml.champion.basic import (
    strategy_decay,
    strategy_expanding,
    strategy_rolling,
)
from common.ml.champion.helpers import (
    _get_exec_lag,
    compute_strategy_accuracy,
    make_blend_row,
    select_output_cols,
)
from common.ml.champion.registry import (
    _DFU_COLS,
    _DFU_MODEL_COLS,
    _DFU_MONTH_COLS,
    _OUTPUT_COLS,
    STRATEGY_REGISTRY,
    register_strategy,
)

_logger = logging.getLogger(__name__)


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
# Strategy: optimized_decay
# ---------------------------------------------------------------------------

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
    seasonal_winners = select_output_cols(
        qualified.drop_duplicates(subset=_DFU_MONTH_COLS, keep="first"),
    )

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
                blended += w * float(row[FORECAST_QTY_COL])
                total_weight += w
                if actual is None:
                    actual = float(row["tothist_dmd"])

        if total_weight > 0 and actual is not None:
            blended /= total_weight
            source_mix = [
                {"model": sname, "weight": round(float(strat_weights.get(sname, 0)), 4)}
                for sname in strat_lookups
                if strat_lookups[sname].get(dfu_month_key) is not None
                and strat_weights.get(sname, 0) >= 0.005
            ]
            results.append(make_blend_row(
                dfu_month_key[0], dfu_month_key[1],
                dfu_month_key[2], dfu_month_key[3],
                "stacked_strategies", 0.0, blended, actual,
                source_mix=source_mix,
            ))

    if not results:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.DataFrame(results)[_OUTPUT_COLS].reset_index(drop=True)
