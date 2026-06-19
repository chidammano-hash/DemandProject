"""Comparison engine: Expert Panel portfolio vs baselines.

Compares portfolio accuracy against:
1. Seasonal Naive (simplest baseline)
2. External Forecast (ERP system forecast)
3. Current Tree Champion (best of LGBM/CatBoost/XGBoost)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from common.core.constants import FORECAST_QTY_COL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _compute_wape(sum_abs_error: float, sum_actual: float) -> float:
    """WAPE = SUM(|forecast - actual|) / max(|SUM(actual)|, 1.0) * 100."""
    return sum_abs_error / max(abs(sum_actual), 1.0) * 100


def _compute_accuracy(wape: float) -> float:
    """Accuracy = 100 - WAPE."""
    return 100.0 - wape


def _compute_bias(sum_forecast: float, sum_actual: float) -> float:
    """Bias = (SUM(forecast) / max(SUM(actual), 1.0)) - 1."""
    return (sum_forecast / max(sum_actual, 1.0)) - 1.0


def _metrics_from_joined(
    joined: pd.DataFrame,
    forecast_col: str = FORECAST_QTY_COL,
    actual_col: str = "qty",
) -> dict[str, float]:
    """Compute WAPE, accuracy, and bias from a joined forecast-vs-actual frame.

    Args:
        joined: DataFrame containing at least ``forecast_col`` and ``actual_col``.
        forecast_col: Column name for forecast values.
        actual_col: Column name for actual values.

    Returns:
        {'wape': float, 'accuracy_pct': float, 'bias': float}
    """
    abs_error = (joined[forecast_col] - joined[actual_col]).abs()
    sum_abs_error = float(abs_error.sum())
    sum_actual = float(joined[actual_col].sum())
    sum_forecast = float(joined[forecast_col].sum())

    wape = _compute_wape(sum_abs_error, sum_actual)
    return {
        "wape": wape,
        "accuracy_pct": _compute_accuracy(wape),
        "bias": _compute_bias(sum_forecast, sum_actual),
    }


# ---------------------------------------------------------------------------
# Per-segment breakdown
# ---------------------------------------------------------------------------


def _per_segment_metrics(
    joined: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Compute WAPE / accuracy / n_dfus for each archetype.

    Args:
        joined: Forecast-vs-actual join (must contain sku_ck, basefcst_pref, qty).
        classification_df: DFU classification with columns [sku_ck, archetype].

    Returns:
        {archetype: {'wape': float, 'accuracy_pct': float, 'n_dfus': int}}
    """
    merged = joined.merge(
        classification_df[["sku_ck", "archetype"]], on="sku_ck", how="left"
    )
    merged["archetype"] = merged["archetype"].fillna("unclassified")

    segments: dict[str, dict[str, Any]] = {}
    for archetype, group in merged.groupby("archetype", sort=True):
        metrics = _metrics_from_joined(group)
        segments[str(archetype)] = {
            "wape": metrics["wape"],
            "accuracy_pct": metrics["accuracy_pct"],
            "n_dfus": int(group["sku_ck"].nunique()),
        }
    return segments


# ---------------------------------------------------------------------------
# Baseline accuracy
# ---------------------------------------------------------------------------


def compute_baseline_accuracy(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    algorithm_id: str,
    classification_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute accuracy metrics for a single baseline algorithm.

    Args:
        predictions_df: Predictions with columns [sku_ck, startdate, basefcst_pref].
        actuals_df: Actuals with columns [sku_ck, startdate, qty].
        algorithm_id: Name of the baseline for labeling.
        classification_df: Optional. If provided, also computes per-segment breakdown.

    Returns:
        {
            'algorithm_id': str,
            'wape': float,
            'accuracy_pct': float,
            'bias': float,
            'n_dfu_months': int,
            'n_dfus': int,
            'per_segment': {archetype: {'wape': float, 'accuracy_pct': float, 'n_dfus': int}} | None
        }
    """
    # Inner join predictions with actuals on (sku_ck, startdate)
    joined = predictions_df.merge(
        actuals_df[["sku_ck", "startdate", "qty"]],
        on=["sku_ck", "startdate"],
        how="inner",
    )

    if joined.empty:
        logger.warning(
            "No matching (sku_ck, startdate) rows for algorithm '%s'", algorithm_id
        )
        return {
            "algorithm_id": algorithm_id,
            "wape": np.nan,
            "accuracy_pct": np.nan,
            "bias": np.nan,
            "n_dfu_months": 0,
            "n_dfus": 0,
            "per_segment": None,
        }

    metrics = _metrics_from_joined(joined)

    per_segment: dict[str, dict[str, Any]] | None = None
    if classification_df is not None:
        per_segment = _per_segment_metrics(joined, classification_df)

    n_dfu_months = len(joined)
    n_dfus = int(joined["sku_ck"].nunique())

    logger.info(
        "Baseline '%s': WAPE=%.2f%%, Accuracy=%.2f%%, Bias=%.4f "
        "(%d DFU-months, %d DFUs)",
        algorithm_id,
        metrics["wape"],
        metrics["accuracy_pct"],
        metrics["bias"],
        n_dfu_months,
        n_dfus,
    )

    return {
        "algorithm_id": algorithm_id,
        "wape": metrics["wape"],
        "accuracy_pct": metrics["accuracy_pct"],
        "bias": metrics["bias"],
        "n_dfu_months": n_dfu_months,
        "n_dfus": n_dfus,
        "per_segment": per_segment,
    }


# ---------------------------------------------------------------------------
# Current tree champion (oracle best-per-month)
# ---------------------------------------------------------------------------


def compute_champion_accuracy(
    existing_predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    classification_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute accuracy of the current tree champion (best model per DFU-month).

    Uses a per-month oracle strategy: for each (sku_ck, startdate), the model
    with the lowest absolute error is selected. This gives the tree models
    their *best possible* outcome, making the comparison conservative -- if
    the Expert Panel still beats this, it is a strong signal.

    Args:
        existing_predictions_df: Predictions from backtest_lag_archive with columns
            [sku_ck, startdate, basefcst_pref, model_id].
            Contains predictions from lgbm_cluster, catboost_cluster, xgboost_cluster.
        actuals_df: Actuals with columns [sku_ck, startdate, qty].
        classification_df: Optional per-segment breakdown.

    Returns:
        Same schema as compute_baseline_accuracy, with algorithm_id='current_champion'.
    """
    if existing_predictions_df.empty:
        logger.warning("No existing tree predictions provided for champion comparison")
        return {
            "algorithm_id": "current_champion",
            "wape": np.nan,
            "accuracy_pct": np.nan,
            "bias": np.nan,
            "n_dfu_months": 0,
            "n_dfus": 0,
            "per_segment": None,
        }

    # Join each model's predictions with actuals
    joined = existing_predictions_df.merge(
        actuals_df[["sku_ck", "startdate", "qty"]],
        on=["sku_ck", "startdate"],
        how="inner",
    )

    if joined.empty:
        logger.warning("No matching actuals for existing tree predictions")
        return {
            "algorithm_id": "current_champion",
            "wape": np.nan,
            "accuracy_pct": np.nan,
            "bias": np.nan,
            "n_dfu_months": 0,
            "n_dfus": 0,
            "per_segment": None,
        }

    # Compute absolute error per row
    joined["abs_error"] = (joined[FORECAST_QTY_COL] - joined["qty"]).abs()

    # Oracle selection: for each (sku_ck, startdate), pick the model_id
    # with the lowest absolute error
    idx_best = joined.groupby(["sku_ck", "startdate"])["abs_error"].idxmin()
    best = joined.loc[idx_best].copy()

    logger.info(
        "Champion oracle: selected best model per DFU-month from %d candidates "
        "-> %d winning rows",
        len(joined),
        len(best),
    )

    # Log model selection distribution
    model_counts = best["model_id"].value_counts().to_dict()
    logger.info("Champion model selection distribution: %s", model_counts)

    # Compute overall metrics from the oracle-selected predictions
    metrics = _metrics_from_joined(best)

    per_segment: dict[str, dict[str, Any]] | None = None
    if classification_df is not None:
        per_segment = _per_segment_metrics(best, classification_df)

    n_dfu_months = len(best)
    n_dfus = int(best["sku_ck"].nunique())

    logger.info(
        "Current champion (oracle): WAPE=%.2f%%, Accuracy=%.2f%%, Bias=%.4f "
        "(%d DFU-months, %d DFUs)",
        metrics["wape"],
        metrics["accuracy_pct"],
        metrics["bias"],
        n_dfu_months,
        n_dfus,
    )

    return {
        "algorithm_id": "current_champion",
        "wape": metrics["wape"],
        "accuracy_pct": metrics["accuracy_pct"],
        "bias": metrics["bias"],
        "n_dfu_months": n_dfu_months,
        "n_dfus": n_dfus,
        "per_segment": per_segment,
    }


# ---------------------------------------------------------------------------
# Golden set oracle (true theoretical ceiling — all algorithms)
# ---------------------------------------------------------------------------


def compute_golden_set_oracle_accuracy(
    all_predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    classification_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute the true theoretical ceiling from the golden set.

    For each (sku_ck, startdate), picks the algorithm with the lowest absolute
    error across ALL algorithms in all_predictions_df.  This is the best
    accuracy achievable with perfect hindsight over the full candidate pool,
    regardless of which algorithms are in the production backtest archive.

    Args:
        all_predictions_df: All expert-panel predictions
            (columns: sku_ck, startdate, basefcst_pref, algorithm_id).
        actuals_df: Actual demand (columns: sku_ck, startdate, qty).
        classification_df: Optional — enables per-segment breakdown.
    """
    if all_predictions_df.empty:
        return {
            "algorithm_id": "golden_oracle",
            "wape": np.nan, "accuracy_pct": np.nan, "bias": np.nan,
            "n_dfu_months": 0, "n_dfus": 0, "per_segment": None,
        }

    # Average duplicate timeframe predictions for the same (DFU, month, algorithm)
    preds = (
        all_predictions_df.groupby(
            ["sku_ck", "startdate", "algorithm_id"], sort=False
        )[FORECAST_QTY_COL]
        .mean()
        .reset_index()
    )

    merged = preds.merge(
        actuals_df[["sku_ck", "startdate", "qty"]],
        on=["sku_ck", "startdate"],
        how="inner",
    )
    if merged.empty:
        logger.warning("No matching actuals for golden set oracle computation")
        return {
            "algorithm_id": "golden_oracle",
            "wape": np.nan, "accuracy_pct": np.nan, "bias": np.nan,
            "n_dfu_months": 0, "n_dfus": 0, "per_segment": None,
        }

    merged["abs_error"] = (merged[FORECAST_QTY_COL] - merged["qty"]).abs()

    # Oracle: lowest absolute error per (DFU, month) across all algorithms
    idx_best = merged.groupby(["sku_ck", "startdate"], sort=False)["abs_error"].idxmin()
    best = merged.loc[idx_best].copy().rename(columns={"abs_error": "_abs_error"})

    # Rename to match _metrics_from_joined expectations
    best = best.rename(columns={"_abs_error": "abs_error"})

    algo_dist = best["algorithm_id"].value_counts().to_dict()
    logger.info(
        "Golden oracle: selected best algorithm per DFU-month from %d candidates "
        "-> %d winning rows | top picks: %s",
        len(merged),
        len(best),
        dict(list(algo_dist.items())[:5]),
    )

    metrics = _metrics_from_joined(best)

    per_segment: dict[str, dict[str, Any]] | None = None
    if classification_df is not None:
        per_segment = _per_segment_metrics(best, classification_df)

    n_dfu_months = len(best)
    n_dfus = int(best["sku_ck"].nunique())

    logger.info(
        "Golden oracle: WAPE=%.2f%%, Accuracy=%.2f%%, Bias=%.4f "
        "(%d DFU-months, %d DFUs)",
        metrics["wape"], metrics["accuracy_pct"], metrics["bias"],
        n_dfu_months, n_dfus,
    )

    return {
        "algorithm_id": "golden_oracle",
        "wape": metrics["wape"],
        "accuracy_pct": metrics["accuracy_pct"],
        "bias": metrics["bias"],
        "n_dfu_months": n_dfu_months,
        "n_dfus": n_dfus,
        "per_segment": per_segment,
    }


# ---------------------------------------------------------------------------
# Current tree champion (causal per-DFU selection)
# ---------------------------------------------------------------------------


def compute_causal_champion_accuracy(
    existing_predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    classification_df: pd.DataFrame | None = None,
    min_prior_months: int = 3,
) -> dict[str, Any]:
    """Compute tree champion accuracy using causal per-DFU model selection.

    Unlike compute_champion_accuracy (oracle), this selects the best tree
    model per DFU using only causally-available prior performance: for each
    (sku_ck, startdate=T), picks the model with the lowest expanding
    cumulative WAPE over months strictly before T (shift(1) causal guard).

    This reflects what production champion selection would actually deliver,
    as opposed to the oracle upper bound.

    DFU-months with fewer than min_prior_months of history fall back to the
    model with the most prior observations; ties broken alphabetically by
    model_id (deterministic, no current-period information used).
    """
    if existing_predictions_df.empty:
        logger.warning("No existing tree predictions for causal champion computation")
        return {
            "algorithm_id": "causal_champion",
            "wape": np.nan, "accuracy_pct": np.nan, "bias": np.nan,
            "n_dfu_months": 0, "n_dfus": 0, "per_segment": None,
        }

    joined = existing_predictions_df.merge(
        actuals_df[["sku_ck", "startdate", "qty"]],
        on=["sku_ck", "startdate"],
        how="inner",
    )
    if joined.empty:
        logger.warning("No matching actuals for causal champion computation")
        return {
            "algorithm_id": "causal_champion",
            "wape": np.nan, "accuracy_pct": np.nan, "bias": np.nan,
            "n_dfu_months": 0, "n_dfus": 0, "per_segment": None,
        }

    joined["abs_err"] = (joined[FORECAST_QTY_COL] - joined["qty"]).abs()
    # Sort ascending by startdate within each (sku_ck, model_id) group — required
    # for shift(1) to produce a strictly-causal lag (prior month's error).
    joined = joined.sort_values(["sku_ck", "model_id", "startdate"]).copy()

    # Causal expanding stats: shift(1) within (sku_ck, model_id) so month T
    # only uses cumulative errors from months 0..T-1.
    # Uses cumsum() (not expanding().sum()) to avoid pandas 2.2 transform-lambda
    # deprecation and to leverage the Cython fast-path.
    gb = joined.groupby(["sku_ck", "model_id"], sort=False)
    joined["_shifted_err"] = gb["abs_err"].shift(1)
    joined["_shifted_act"] = gb["qty"].shift(1)
    joined["_cum_abs_err"] = gb["_shifted_err"].cumsum()
    joined["_cum_actual"] = gb["_shifted_act"].cumsum()
    # prior_count = number of non-NaN shifted errors seen so far per group
    joined["_is_valid"] = joined["_shifted_err"].notna().astype(int)
    joined["_prior_count"] = gb["_is_valid"].cumsum()
    joined["_prior_wape"] = (
        joined["_cum_abs_err"] / joined["_cum_actual"].abs().clip(lower=1e-6)
    )

    # Rows with enough causal history: pick model with lowest prior WAPE
    qualified = joined[joined["_prior_count"] >= min_prior_months].copy()
    qualified = qualified.sort_values("_prior_wape", na_position="last")
    best_qualified = qualified.drop_duplicates(subset=["sku_ck", "startdate"], keep="first")

    # Fallback for DFU-months not yet covered by the qualified window.
    # Use a left-merge to avoid fragile MultiIndex isin on duplicate keys.
    # .values strips the merge's index so the mask aligns positionally with joined.
    if not best_qualified.empty:
        covered_keys = best_qualified[["sku_ck", "startdate"]].drop_duplicates().assign(_covered=True)
        join_flag = joined.merge(covered_keys, on=["sku_ck", "startdate"], how="left")
        fallback_mask = join_flag["_covered"].isna().values
        fallback_df = joined[fallback_mask].copy()
    else:
        fallback_df = joined.copy()

    if not fallback_df.empty:
        # Pick model with most prior observations; tie-break alphabetically by
        # model_id so the selection is deterministic and uses NO current-period info.
        fallback_df = fallback_df.sort_values(
            ["_prior_count", "model_id"], ascending=[False, True]
        )
        fallback_best = fallback_df.drop_duplicates(subset=["sku_ck", "startdate"], keep="first")
        best = pd.concat([best_qualified, fallback_best], ignore_index=True)
    else:
        best = best_qualified

    best = best.drop_duplicates(subset=["sku_ck", "startdate"], keep="first")

    model_counts = best["model_id"].value_counts().to_dict()
    logger.info(
        "Causal champion model selection: %s", model_counts,
    )

    metrics = _metrics_from_joined(best)

    per_segment: dict[str, dict[str, Any]] | None = None
    if classification_df is not None:
        per_segment = _per_segment_metrics(best, classification_df)

    n_dfu_months = len(best)
    n_dfus = int(best["sku_ck"].nunique())

    logger.info(
        "Causal champion: WAPE=%.2f%%, Accuracy=%.2f%%, Bias=%.4f "
        "(%d DFU-months, %d DFUs)",
        metrics["wape"], metrics["accuracy_pct"], metrics["bias"],
        n_dfu_months, n_dfus,
    )

    return {
        "algorithm_id": "causal_champion",
        "wape": metrics["wape"],
        "accuracy_pct": metrics["accuracy_pct"],
        "bias": metrics["bias"],
        "n_dfu_months": n_dfu_months,
        "n_dfus": n_dfus,
        "per_segment": per_segment,
    }


# ---------------------------------------------------------------------------
# Portfolio prediction routing
# ---------------------------------------------------------------------------


def compute_portfolio_predictions(
    all_predictions_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    classification_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build portfolio predictions by routing each DFU to its assigned algorithm.

    Args:
        all_predictions_df: All predictions from all algorithms.
            Columns: sku_ck, startdate, basefcst_pref, algorithm_id
        assignments_df: Portfolio assignments.
            Columns: archetype, best_algorithm
        classification_df: DFU classification.
            Columns: sku_ck, archetype

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref
        One row per (DFU, month) using the assigned algorithm's prediction.
    """
    # Step 1: Map sku_ck -> best_algorithm via classification + assignments
    dfu_routing = classification_df[["sku_ck", "archetype"]].merge(
        assignments_df[["archetype", "best_algorithm"]],
        on="archetype",
        how="left",
    )

    # DFUs whose archetype has no assignment fall back to seasonal_naive
    dfu_routing["best_algorithm"] = dfu_routing["best_algorithm"].fillna(
        "seasonal_naive"
    )

    logger.info(
        "Portfolio routing: %d DFUs mapped to %d algorithms",
        len(dfu_routing),
        dfu_routing["best_algorithm"].nunique(),
    )

    # Step 2: Use latest timeframe per (sku_ck, startdate, algorithm_id) so the
    # portfolio gets the lowest-lag (most accurate) prediction for each DFU-month.
    # Without this, predictions arrive in tf_idx=0,1,...,N order and the merge
    # keeps tf_idx=0 (the oldest, worst-lag forecast) via drop_duplicates.
    preds_for_routing = all_predictions_df
    if "timeframe_idx" in all_predictions_df.columns:
        preds_for_routing = (
            all_predictions_df
            .sort_values("timeframe_idx", ascending=False)
            .drop_duplicates(subset=["sku_ck", "startdate", "algorithm_id"], keep="first")
        )

    routed = dfu_routing.merge(
        preds_for_routing,
        left_on=["sku_ck", "best_algorithm"],
        right_on=["sku_ck", "algorithm_id"],
        how="left",
    )

    # Step 3: Identify DFU-months missing from the assigned algorithm
    assigned = routed.dropna(subset=[FORECAST_QTY_COL])
    missing_mask = routed[FORECAST_QTY_COL].isna()
    n_missing = int(missing_mask.sum())

    if n_missing > 0:
        logger.info(
            "Portfolio: %d DFU-months missing from assigned algorithm, "
            "applying demand-aware fallback",
            n_missing,
        )

        missing_skus = routed.loc[missing_mask, "sku_ck"].unique()

        # Demand-aware fallback: intermittent/erratic archetypes perform
        # catastrophically with seasonal_naive — prefer croston_sba → tsb →
        # rolling_mean → seasonal_naive in that order.
        # Smooth/insufficient archetypes: rolling_mean → seasonal_naive.
        archetype_map: dict[str, str] = {}
        if classification_df is not None and "archetype" in classification_df.columns:
            archetype_map = (
                classification_df.set_index("sku_ck")["archetype"]
                .reindex(missing_skus)
                .fillna("smooth_low")
                .to_dict()
            )

        _INTERMITTENT_ARCHETYPES = frozenset(
            a for a in archetype_map.values()
            if any(a.startswith(p) for p in ("intermittent", "lumpy", "erratic"))
        )
        _INTERMITTENT_SKUS = {
            s for s, a in archetype_map.items() if a in _INTERMITTENT_ARCHETYPES
        }

        # Priority lists: first algorithm with predictions wins
        _INTERMITTENT_PRIORITY = ["croston_sba", "tsb", "rolling_mean", "seasonal_naive"]
        _SMOOTH_PRIORITY = ["rolling_mean", "seasonal_naive"]

        fallback_parts: list[pd.DataFrame] = []
        covered_skus: set[str] = set()

        for algo in _INTERMITTENT_PRIORITY:
            remaining_intermittent = _INTERMITTENT_SKUS - covered_skus
            if not remaining_intermittent:
                break
            algo_preds = preds_for_routing[
                (preds_for_routing["algorithm_id"] == algo)
                & (preds_for_routing["sku_ck"].isin(remaining_intermittent))
            ]
            if not algo_preds.empty:
                part = algo_preds.merge(
                    pd.DataFrame({"sku_ck": list(remaining_intermittent)}),
                    on="sku_ck", how="inner",
                )[["sku_ck", "startdate", FORECAST_QTY_COL]]
                fallback_parts.append(part)
                covered_skus.update(part["sku_ck"].unique())

        smooth_skus = set(missing_skus) - _INTERMITTENT_SKUS
        for algo in _SMOOTH_PRIORITY:
            remaining_smooth = smooth_skus - covered_skus
            if not remaining_smooth:
                break
            algo_preds = preds_for_routing[
                (preds_for_routing["algorithm_id"] == algo)
                & (preds_for_routing["sku_ck"].isin(remaining_smooth))
            ]
            if not algo_preds.empty:
                part = algo_preds.merge(
                    pd.DataFrame({"sku_ck": list(remaining_smooth)}),
                    on="sku_ck", how="inner",
                )[["sku_ck", "startdate", FORECAST_QTY_COL]]
                fallback_parts.append(part)
                covered_skus.update(part["sku_ck"].unique())

        if fallback_parts:
            fallback = pd.concat(fallback_parts, ignore_index=True)
            fallback = fallback.drop_duplicates(subset=["sku_ck", "startdate"], keep="first")
            logger.info(
                "Portfolio fallback covered %d DFUs via demand-aware cascade",
                fallback["sku_ck"].nunique(),
            )
            assigned = pd.concat(
                [
                    assigned[["sku_ck", "startdate", FORECAST_QTY_COL]],
                    fallback,
                ],
                ignore_index=True,
            )
        else:
            assigned = assigned[["sku_ck", "startdate", FORECAST_QTY_COL]]
    else:
        assigned = assigned[["sku_ck", "startdate", FORECAST_QTY_COL]]

    # Deduplicate: if a DFU-month appears in both assigned and fallback
    # (shouldn't happen, but guard), keep the first (assigned takes priority)
    result = (
        assigned.drop_duplicates(subset=["sku_ck", "startdate"], keep="first")
        .reset_index(drop=True)
    )

    logger.info(
        "Portfolio predictions: %d rows covering %d DFUs",
        len(result),
        result["sku_ck"].nunique(),
    )

    return result


# ---------------------------------------------------------------------------
# Full comparison
# ---------------------------------------------------------------------------


def compare_all(
    portfolio_predictions: pd.DataFrame,
    naive_predictions: pd.DataFrame,
    external_predictions: pd.DataFrame | None,
    existing_predictions: pd.DataFrame | None,
    actuals_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    all_predictions_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Full comparison of portfolio vs all baselines.

    Args:
        portfolio_predictions: Routed portfolio predictions
            (output of compute_portfolio_predictions).
        naive_predictions: Seasonal naive predictions
            with columns [sku_ck, startdate, basefcst_pref].
        external_predictions: External forecast predictions, or None if unavailable.
        existing_predictions: Tree champion predictions with model_id column,
            or None if unavailable.
        actuals_df: Actuals with columns [sku_ck, startdate, qty].
        classification_df: DFU classification with columns [sku_ck, archetype].
        all_predictions_df: All expert-panel predictions across every algorithm
            (columns: sku_ck, startdate, basefcst_pref, algorithm_id).
            When provided, computes the true golden-set oracle (theoretical ceiling).

    Returns:
        {
            'portfolio': {accuracy metrics},
            'baselines': {
                'seasonal_naive': {accuracy metrics},
                'external_forecast': {accuracy metrics} | None,
                'current_champion': {accuracy metrics} | None,
                'causal_champion': {accuracy metrics} | None,
                'golden_oracle': {accuracy metrics} | None,
            },
            'lift': {
                'vs_naive_bps': int,
                'vs_external_bps': int | None,
                'vs_champion_bps': int | None,
                'vs_causal_champion_bps': int | None,
                'vs_golden_oracle_bps': int | None,
            },
            'per_segment': DataFrame with columns:
                archetype, portfolio_acc, naive_acc, external_acc, champion_acc,
                causal_champion_acc, golden_oracle_acc,
                lift_vs_naive, lift_vs_external, lift_vs_champion,
                lift_vs_causal_champion, lift_vs_golden_oracle, n_dfus
        }
    """
    logger.info("Running full portfolio vs baselines comparison")

    # ── Common universe: actuals define the ground truth ──────────────
    # All comparisons run on the SAME (sku_ck, startdate) pairs from
    # actuals_df -- i.e. the golden set DFUs x predict months.
    # Each method is evaluated only on that fixed universe.
    universe_keys = actuals_df[["sku_ck", "startdate"]].drop_duplicates()
    logger.info(
        "Comparison universe (from actuals): %d DFU-months across %d DFUs",
        len(universe_keys),
        universe_keys["sku_ck"].nunique(),
    )

    def _restrict(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or df.empty:
            return df
        return df.merge(universe_keys, on=["sku_ck", "startdate"], how="inner")

    # Restrict all predictions to the common universe
    portfolio_restricted = _restrict(portfolio_predictions)
    naive_restricted = _restrict(naive_predictions)
    external_restricted = _restrict(external_predictions)
    existing_restricted = _restrict(existing_predictions)

    # Portfolio accuracy (on the golden set universe)
    portfolio_metrics = compute_baseline_accuracy(
        portfolio_restricted, actuals_df, "portfolio", classification_df
    )

    # Baseline: Seasonal Naive (on the same golden set universe)
    naive_metrics = compute_baseline_accuracy(
        naive_restricted, actuals_df, "seasonal_naive", classification_df
    )

    # Baseline: External Forecast (on the same golden set universe)
    external_metrics: dict[str, Any] | None = None
    if external_restricted is not None and not external_restricted.empty:
        external_metrics = compute_baseline_accuracy(
            external_restricted, actuals_df, "external_forecast", classification_df
        )

    # Baseline: Current Tree Champion — oracle (on the same golden set universe)
    champion_metrics: dict[str, Any] | None = None
    if existing_restricted is not None and not existing_restricted.empty:
        champion_metrics = compute_champion_accuracy(
            existing_restricted, actuals_df, classification_df
        )

    # Baseline: Causal Champion — per-DFU expanding-WAPE selection (realistic)
    causal_champion_metrics: dict[str, Any] | None = None
    if existing_restricted is not None and not existing_restricted.empty:
        causal_champion_metrics = compute_causal_champion_accuracy(
            existing_restricted, actuals_df, classification_df
        )

    # Baseline: Golden Oracle — true theoretical ceiling over ALL algorithms
    golden_oracle_metrics: dict[str, Any] | None = None
    all_restricted = _restrict(all_predictions_df)
    if all_restricted is not None and not all_restricted.empty:
        golden_oracle_metrics = compute_golden_set_oracle_accuracy(
            all_restricted, actuals_df, classification_df
        )

    # Lift calculation (in basis points: 1 percentage point = 100 bps)
    portfolio_acc = portfolio_metrics["accuracy_pct"]

    vs_naive_bps: int = 0
    if not np.isnan(portfolio_acc) and not np.isnan(naive_metrics["accuracy_pct"]):
        vs_naive_bps = round((portfolio_acc - naive_metrics["accuracy_pct"]) * 100)

    vs_external_bps: int | None = None
    if external_metrics is not None and not np.isnan(
        external_metrics["accuracy_pct"]
    ):
        vs_external_bps = round((portfolio_acc - external_metrics["accuracy_pct"]) * 100)

    vs_champion_bps: int | None = None
    if champion_metrics is not None and not np.isnan(
        champion_metrics["accuracy_pct"]
    ):
        vs_champion_bps = round((portfolio_acc - champion_metrics["accuracy_pct"]) * 100)

    vs_causal_champion_bps: int | None = None
    if causal_champion_metrics is not None and not np.isnan(
        causal_champion_metrics["accuracy_pct"]
    ):
        vs_causal_champion_bps = round(
            (portfolio_acc - causal_champion_metrics["accuracy_pct"]) * 100
        )

    vs_golden_oracle_bps: int | None = None
    if golden_oracle_metrics is not None and not np.isnan(
        golden_oracle_metrics["accuracy_pct"]
    ):
        vs_golden_oracle_bps = round(
            (portfolio_acc - golden_oracle_metrics["accuracy_pct"]) * 100
        )

    # Per-segment comparison table
    per_segment_df = _build_per_segment_table(
        portfolio_metrics,
        naive_metrics,
        external_metrics,
        champion_metrics,
        classification_df,
        causal_champion_metrics=causal_champion_metrics,
        golden_oracle_metrics=golden_oracle_metrics,
    )

    logger.info(
        "Comparison complete: Portfolio=%.2f%%, Naive=%.2f%%, "
        "Lift vs Naive=%+d bps",
        portfolio_acc,
        naive_metrics["accuracy_pct"],
        vs_naive_bps,
    )

    return {
        "portfolio": portfolio_metrics,
        "baselines": {
            "seasonal_naive": naive_metrics,
            "external_forecast": external_metrics,
            "current_champion": champion_metrics,
            "causal_champion": causal_champion_metrics,
            "golden_oracle": golden_oracle_metrics,
        },
        "lift": {
            "vs_naive_bps": vs_naive_bps,
            "vs_external_bps": vs_external_bps,
            "vs_champion_bps": vs_champion_bps,
            "vs_causal_champion_bps": vs_causal_champion_bps,
            "vs_golden_oracle_bps": vs_golden_oracle_bps,
        },
        "per_segment": per_segment_df,
    }


# ---------------------------------------------------------------------------
# Per-segment table builder
# ---------------------------------------------------------------------------


def _build_per_segment_table(
    portfolio_metrics: dict[str, Any],
    naive_metrics: dict[str, Any],
    external_metrics: dict[str, Any] | None,
    champion_metrics: dict[str, Any] | None,
    classification_df: pd.DataFrame,
    causal_champion_metrics: dict[str, Any] | None = None,
    golden_oracle_metrics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a per-archetype comparison table.

    Returns:
        DataFrame with columns:
            archetype, portfolio_acc, naive_acc, external_acc, champion_acc,
            causal_champion_acc, golden_oracle_acc, lift_vs_naive, lift_vs_external,
            lift_vs_champion, lift_vs_causal_champion, lift_vs_golden_oracle, n_dfus
    """
    archetypes = sorted(classification_df["archetype"].unique())

    rows: list[dict[str, Any]] = []
    for arch in archetypes:
        port_seg = _get_segment_acc(portfolio_metrics, arch)
        naive_seg = _get_segment_acc(naive_metrics, arch)
        ext_seg = _get_segment_acc(external_metrics, arch) if external_metrics else np.nan
        champ_seg = _get_segment_acc(champion_metrics, arch) if champion_metrics else np.nan
        causal_seg = (
            _get_segment_acc(causal_champion_metrics, arch)
            if causal_champion_metrics else np.nan
        )
        golden_seg = (
            _get_segment_acc(golden_oracle_metrics, arch)
            if golden_oracle_metrics else np.nan
        )

        n_dfus = int(
            classification_df[classification_df["archetype"] == arch]["sku_ck"].nunique()
        )

        row: dict[str, Any] = {
            "archetype": arch,
            "portfolio_acc": port_seg,
            "naive_acc": naive_seg,
            "external_acc": ext_seg,
            "champion_acc": champ_seg,
            "causal_champion_acc": causal_seg,
            "golden_oracle_acc": golden_seg,
            "lift_vs_naive": _safe_diff(port_seg, naive_seg),
            "lift_vs_external": _safe_diff(port_seg, ext_seg),
            "lift_vs_champion": _safe_diff(port_seg, champ_seg),
            "lift_vs_causal_champion": _safe_diff(port_seg, causal_seg),
            "lift_vs_golden_oracle": _safe_diff(port_seg, golden_seg),
            "n_dfus": n_dfus,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _get_segment_acc(
    metrics: dict[str, Any] | None, archetype: str
) -> float:
    """Extract per-segment accuracy for a given archetype, or NaN if missing."""
    if metrics is None:
        return np.nan
    per_segment = metrics.get("per_segment")
    if per_segment is None:
        return np.nan
    segment_data = per_segment.get(archetype)
    if segment_data is None:
        return np.nan
    return float(segment_data["accuracy_pct"])


def _safe_diff(a: float, b: float) -> float:
    """Compute a - b, returning NaN if either is NaN."""
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return a - b


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------


def format_comparison_summary(comparison: dict[str, Any]) -> str:
    """Format the comparison as a human-readable summary.

    Shows:
    - Portfolio accuracy
    - Each baseline accuracy
    - Lift vs each baseline (in basis points)
    - Per-segment detail table

    Args:
        comparison: Output of compare_all().

    Returns:
        Multi-line formatted string.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("EXPERT PANEL PORTFOLIO vs BASELINES")
    lines.append("=" * 70)

    # Portfolio
    port = comparison["portfolio"]
    lines.append("")
    lines.append(
        f"  Portfolio:       Accuracy {port['accuracy_pct']:6.2f}%  "
        f"WAPE {port['wape']:6.2f}%  Bias {port['bias']:+.4f}  "
        f"({port['n_dfus']} DFUs, {port['n_dfu_months']} DFU-months)"
    )

    # Baselines
    lines.append("")
    lines.append("  Baselines:")
    baselines = comparison["baselines"]

    naive = baselines["seasonal_naive"]
    lines.append(
        f"    Seasonal Naive:    Accuracy {naive['accuracy_pct']:6.2f}%  "
        f"WAPE {naive['wape']:6.2f}%  Bias {naive['bias']:+.4f}"
    )

    ext = baselines.get("external_forecast")
    if ext is not None:
        lines.append(
            f"    External Forecast: Accuracy {ext['accuracy_pct']:6.2f}%  "
            f"WAPE {ext['wape']:6.2f}%  Bias {ext['bias']:+.4f}"
        )
    else:
        lines.append("    External Forecast: (not available)")

    champ = baselines.get("current_champion")
    if champ is not None:
        lines.append(
            f"    Tree Backtest Oracle: Accuracy {champ['accuracy_pct']:6.2f}%  "
            f"WAPE {champ['wape']:6.2f}%  Bias {champ['bias']:+.4f}"
        )
    else:
        lines.append("    Tree Backtest Oracle: (not available)")

    causal = baselines.get("causal_champion")
    if causal is not None:
        lines.append(
            f"    Causal Champion:      Accuracy {causal['accuracy_pct']:6.2f}%  "
            f"WAPE {causal['wape']:6.2f}%  Bias {causal['bias']:+.4f}"
        )
    else:
        lines.append("    Causal Champion:      (not available)")

    golden = baselines.get("golden_oracle")
    if golden is not None:
        lines.append(
            f"    Golden Oracle:        Accuracy {golden['accuracy_pct']:6.2f}%  "
            f"WAPE {golden['wape']:6.2f}%  Bias {golden['bias']:+.4f}  "
            f"({golden['n_dfus']} DFUs, {golden['n_dfu_months']} DFU-months)"
        )
    else:
        lines.append("    Golden Oracle:        (not available)")

    # Lift
    lines.append("")
    lines.append("  Lift (basis points):")
    lift = comparison["lift"]
    lines.append(f"    vs Seasonal Naive:    {lift['vs_naive_bps']:+d} bps")

    if lift["vs_external_bps"] is not None:
        lines.append(f"    vs External Forecast: {lift['vs_external_bps']:+d} bps")
    else:
        lines.append("    vs External Forecast: N/A")

    if lift["vs_champion_bps"] is not None:
        lines.append(f"    vs Tree Backtest Oracle: {lift['vs_champion_bps']:+d} bps")
    else:
        lines.append("    vs Tree Backtest Oracle: N/A")

    if lift.get("vs_causal_champion_bps") is not None:
        lines.append(f"    vs Causal Champion:      {lift['vs_causal_champion_bps']:+d} bps")
    else:
        lines.append("    vs Causal Champion:      N/A")

    if lift.get("vs_golden_oracle_bps") is not None:
        lines.append(f"    vs Golden Oracle:        {lift['vs_golden_oracle_bps']:+d} bps")
    else:
        lines.append("    vs Golden Oracle:        N/A")

    # Per-segment table
    per_segment: pd.DataFrame = comparison.get("per_segment", pd.DataFrame())
    if not per_segment.empty:
        lines.append("")
        lines.append("-" * 70)
        lines.append("  PER-SEGMENT BREAKDOWN")
        lines.append("-" * 70)

        # Header
        lines.append(
            f"  {'Archetype':<20s} {'Portfolio':>9s} {'Naive':>9s} "
            f"{'External':>9s} {'TBOracle':>9s} {'Causal':>9s} {'GOracle':>9s} "
            f"{'Lift/N':>7s} {'DFUs':>6s}"
        )
        lines.append("  " + "-" * 87)

        for _, row in per_segment.iterrows():
            port_str = f"{row['portfolio_acc']:.1f}%" if not np.isnan(row["portfolio_acc"]) else "  N/A"
            naive_str = f"{row['naive_acc']:.1f}%" if not np.isnan(row["naive_acc"]) else "  N/A"
            ext_str = f"{row['external_acc']:.1f}%" if not np.isnan(row["external_acc"]) else "  N/A"
            champ_str = f"{row['champion_acc']:.1f}%" if not np.isnan(row["champion_acc"]) else "  N/A"
            causal_str = (
                f"{row['causal_champion_acc']:.1f}%"
                if "causal_champion_acc" in row and not np.isnan(row["causal_champion_acc"])
                else "  N/A"
            )
            golden_str = (
                f"{row['golden_oracle_acc']:.1f}%"
                if "golden_oracle_acc" in row and not np.isnan(row["golden_oracle_acc"])
                else "  N/A"
            )
            lift_str = f"{row['lift_vs_naive']:+.1f}" if not np.isnan(row["lift_vs_naive"]) else " N/A"

            lines.append(
                f"  {row['archetype']:<20s} {port_str:>9s} {naive_str:>9s} "
                f"{ext_str:>9s} {champ_str:>9s} {causal_str:>9s} {golden_str:>9s} "
                f"{lift_str:>7s} {row['n_dfus']:>6d}"
            )

    lines.append("")
    lines.append("=" * 70)

    summary = "\n".join(lines)
    logger.info("Comparison summary:\n%s", summary)
    return summary
