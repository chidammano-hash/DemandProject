"""Per-DFU algorithm accuracy matrix and inverse-WAPE blending.

Builds a DFU × algorithm accuracy table from backtest predictions and actuals.
This is the data foundation for the hybrid per-DFU ensemble: each cell stores
the WAPE a given algorithm achieves on a specific DFU across all backtest
timeframes, enabling per-DFU routing and inverse-WAPE weighted blending.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_dfu_accuracy_matrix(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    min_n_months: int = 2,
) -> pd.DataFrame:
    """Build per-DFU per-algorithm WAPE matrix.

    For each (sku_ck, algorithm_id) pair, computes WAPE aggregated over all
    matched actuals.  Predictions spanning multiple timeframes are averaged
    per (sku_ck, startdate, algorithm_id) first — matching the deduplication
    logic in ``build_affinity_matrix`` — so overlapping timeframe windows do
    not inflate the sample count.

    WAPE formula (same scale as affinity_matrix.py):
        WAPE = SUM(|forecast - actual|) / max(|SUM(actual)|, 1.0) × 100

    Args:
        predictions_df: All algorithm predictions.
            Required columns: sku_ck, startdate, basefcst_pref, algorithm_id.
        actuals_df: Ground-truth sales.
            Required columns: sku_ck, startdate, qty.
        min_n_months: Minimum matched prediction-actual pairs required per
            (sku_ck, algorithm_id) to include the row.  Rows below this
            threshold have unreliable WAPE estimates and are excluded from
            meta-router training but may still be used in blending if no
            better data exists.  Default 2.

    Returns:
        DataFrame with columns:
            sku_ck        — DFU identifier
            algorithm_id  — algorithm name
            wape          — WAPE in percentage points
            accuracy_pct  — 100 - wape, floored at 0
            n_months      — number of matched (sku_ck, startdate) pairs used
    """
    if predictions_df.empty or actuals_df.empty:
        logger.warning("build_dfu_accuracy_matrix: empty input, returning empty matrix")
        return pd.DataFrame(
            columns=["sku_ck", "algorithm_id", "wape", "accuracy_pct", "n_months"]
        )

    # Step 1: Average across timeframes for overlapping (sku_ck, startdate, algorithm_id)
    deduped = (
        predictions_df.groupby(
            ["sku_ck", "startdate", "algorithm_id"], sort=False
        )["basefcst_pref"]
        .mean()
        .reset_index()
    )

    # Step 2: Inner-join with actuals on (sku_ck, startdate)
    joined = deduped.merge(
        actuals_df[["sku_ck", "startdate", "qty"]],
        on=["sku_ck", "startdate"],
        how="inner",
    )

    if joined.empty:
        logger.warning(
            "build_dfu_accuracy_matrix: no (sku_ck, startdate) overlap "
            "between predictions and actuals"
        )
        return pd.DataFrame(
            columns=["sku_ck", "algorithm_id", "wape", "accuracy_pct", "n_months"]
        )

    joined["abs_error"] = (joined["basefcst_pref"] - joined["qty"]).abs()

    # Step 3: Aggregate per (sku_ck, algorithm_id)
    agg = (
        joined.groupby(["sku_ck", "algorithm_id"], sort=False)
        .agg(
            sum_abs_error=("abs_error", "sum"),
            sum_actual=("qty", "sum"),
            n_months=("qty", "count"),
        )
        .reset_index()
    )

    # Step 4: Apply min_n_months filter
    agg = agg[agg["n_months"] >= min_n_months].copy()

    if agg.empty:
        logger.warning(
            "build_dfu_accuracy_matrix: no rows survived min_n_months=%d filter",
            min_n_months,
        )
        return pd.DataFrame(
            columns=["sku_ck", "algorithm_id", "wape", "accuracy_pct", "n_months"]
        )

    agg["wape"] = (
        agg["sum_abs_error"] / agg["sum_actual"].abs().clip(lower=1.0) * 100.0
    )
    agg["accuracy_pct"] = (100.0 - agg["wape"]).clip(lower=0.0)

    result = agg[
        ["sku_ck", "algorithm_id", "wape", "accuracy_pct", "n_months"]
    ].reset_index(drop=True)

    logger.info(
        "DFU accuracy matrix: %d rows, %d DFUs, %d algorithms",
        len(result),
        result["sku_ck"].nunique(),
        result["algorithm_id"].nunique(),
    )
    return result


def compute_inverse_wape_blend(
    predictions_df: pd.DataFrame,
    dfu_accuracy_matrix: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """Blend top-K algorithm predictions per DFU using inverse-WAPE weights.

    For each DFU, selects the ``top_k`` algorithms with the lowest WAPE from
    ``dfu_accuracy_matrix`` and produces a weighted-average forecast.  Weights
    are ``1 / max(wape, 1e-6)`` normalised so they sum to 1 per DFU.

    If fewer than ``top_k`` algorithms are available for a DFU, all available
    algorithms are blended.  DFUs with no entry in ``dfu_accuracy_matrix`` are
    silently excluded (handled upstream by the hybrid ensemble).

    Args:
        predictions_df: Predictions with columns
            [sku_ck, startdate, basefcst_pref, algorithm_id].
            A ``timeframe_idx`` column is ignored if present.
        dfu_accuracy_matrix: Output of ``build_dfu_accuracy_matrix``.
            Required columns: sku_ck, algorithm_id, wape.
        top_k: Number of best algorithms to blend per DFU.

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref, algorithm_id.
        ``algorithm_id`` is fixed to ``"hybrid_blend"`` for all rows.
        ``basefcst_pref`` is non-negative.
    """
    if dfu_accuracy_matrix.empty or predictions_df.empty:
        logger.warning("compute_inverse_wape_blend: empty input, returning empty frame")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    # Select top-K algorithms per DFU by lowest WAPE
    ranked = (
        dfu_accuracy_matrix.sort_values("wape")
        .groupby("sku_ck", sort=False)
        .head(top_k)[["sku_ck", "algorithm_id", "wape"]]
        .copy()
    )

    # Inverse-WAPE weights normalised per DFU
    ranked["inv_wape"] = 1.0 / ranked["wape"].clip(lower=1e-6)
    weight_sums = ranked.groupby("sku_ck")["inv_wape"].transform("sum")
    ranked["weight"] = ranked["inv_wape"] / weight_sums

    # Average predictions across timeframes (dedup overlapping windows)
    deduped = (
        predictions_df.groupby(
            ["sku_ck", "startdate", "algorithm_id"], sort=False
        )["basefcst_pref"]
        .mean()
        .reset_index()
    )

    # Join predictions with per-DFU algorithm weights
    blended_input = deduped.merge(
        ranked[["sku_ck", "algorithm_id", "weight"]],
        on=["sku_ck", "algorithm_id"],
        how="inner",
    )

    if blended_input.empty:
        logger.warning(
            "compute_inverse_wape_blend: no predictions matched after joining "
            "with accuracy matrix"
        )
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    blended_input["weighted_pred"] = (
        blended_input["basefcst_pref"] * blended_input["weight"]
    )

    result = (
        blended_input.groupby(["sku_ck", "startdate"], sort=False)["weighted_pred"]
        .sum()
        .reset_index()
        .rename(columns={"weighted_pred": "basefcst_pref"})
    )
    result["basefcst_pref"] = result["basefcst_pref"].clip(lower=0.0)
    result["algorithm_id"] = "hybrid_blend"

    logger.info(
        "Inverse-WAPE blend (top_%d): %d predictions, %d DFUs",
        top_k,
        len(result),
        result["sku_ck"].nunique(),
    )
    return result
