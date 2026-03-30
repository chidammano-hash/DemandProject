"""Hybrid per-DFU ensemble: meta-router + inverse-WAPE blend.

Routes each DFU to either:
- its meta-router-predicted best algorithm (when confidence >= threshold), or
- an inverse-WAPE weighted blend of the top-K algorithms (low confidence
  or DFUs absent from the meta-router's feature set).

Targets a ~10 percentage-point improvement over segment-level portfolio
routing by replacing one-algorithm-per-archetype assignment with per-DFU
algorithm selection grounded in each DFU's own accuracy history.

Integration point:
    Called in run_expert_panel.py / run_adv_expert_panel.py after the
    affinity-matrix portfolio step and before compare_all().
"""

import logging

import pandas as pd

from algorithm_testing.dfu_accuracy_matrix import compute_inverse_wape_blend
from algorithm_testing.meta_router import MetaRouterModel, predict_meta_router

logger = logging.getLogger(__name__)


def compute_hybrid_predictions(
    all_predictions_df: pd.DataFrame,
    dfu_accuracy_matrix: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
    classification_df: pd.DataFrame,
    meta_model: MetaRouterModel,
    blend_top_k: int = 3,
    confidence_threshold: float = 0.6,
) -> pd.DataFrame:
    """Build per-DFU hybrid predictions.

    Decision logic per DFU
    ----------------------
    1. **High-confidence** (meta-router probability >= ``confidence_threshold``):
       Use the single predicted-best algorithm's forecast for this DFU.
    2. **Low-confidence** (probability < ``confidence_threshold``):
       Use an inverse-WAPE weighted blend of the top ``blend_top_k``
       algorithms from ``dfu_accuracy_matrix``.
    3. **Unrouted** (DFU missing from ``dfu_attrs`` or ``classification_df``):
       Falls into the blend path automatically.
    4. **No accuracy history** (DFU missing from ``dfu_accuracy_matrix``):
       Falls back to ``seasonal_naive``.
    5. **Predicted algorithm has no predictions for a DFU-month**:
       Falls back to ``seasonal_naive`` for that DFU-month.

    Args:
        all_predictions_df: All algorithm predictions with columns
            [sku_ck, startdate, basefcst_pref, algorithm_id].
            May include a ``timeframe_idx`` column (averaged away).
        dfu_accuracy_matrix: Output of ``build_dfu_accuracy_matrix``.
            Columns: sku_ck, algorithm_id, wape.
        dfu_attrs: DFU attribute table.
        classification_df: Demand classification output.
        meta_model: Fitted ``MetaRouterModel`` from ``train_meta_router``.
        blend_top_k: Number of top algorithms to blend for low-confidence DFUs.
        confidence_threshold: Minimum meta-router confidence to use single-
            algorithm routing instead of the blend.

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref, algorithm_id.
        ``algorithm_id`` is fixed to ``"hybrid"`` for all rows.
        ``basefcst_pref`` is non-negative.
    """
    if all_predictions_df.empty:
        logger.warning("compute_hybrid_predictions: all_predictions_df is empty")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    # ── Step 1: Get meta-router predictions ────────────────────────────────
    routing_df = predict_meta_router(meta_model, dfu_attrs, classification_df)

    high_conf = routing_df[routing_df["confidence"] >= confidence_threshold]
    low_conf = routing_df[routing_df["confidence"] < confidence_threshold]

    # DFUs with predictions but no meta-router output (missing attrs/classification)
    all_skus = set(all_predictions_df["sku_ck"].unique())
    routed_skus = set(routing_df["sku_ck"].unique())
    unrouted_skus = all_skus - routed_skus
    if unrouted_skus:
        logger.warning(
            "%d DFUs not reachable by meta-router (missing attrs/classification); "
            "routing to blend",
            len(unrouted_skus),
        )

    # ── Step 2: Deduplicate predictions across timeframes ──────────────────
    deduped = (
        all_predictions_df.groupby(
            ["sku_ck", "startdate", "algorithm_id"], sort=False
        )["basefcst_pref"]
        .mean()
        .reset_index()
    )

    parts: list[pd.DataFrame] = []

    # ── Step 3a: High-confidence DFUs → single predicted algorithm ─────────
    if not high_conf.empty:
        high_conf_preds = _route_single_algorithm(
            deduped, high_conf, all_predictions_df
        )
        if not high_conf_preds.empty:
            parts.append(high_conf_preds)
        logger.info(
            "  Hybrid high-conf: %d DFUs → single algorithm (confidence >= %.2f)",
            high_conf["sku_ck"].nunique(),
            confidence_threshold,
        )

    # ── Step 3b: Low-confidence + unrouted DFUs → inverse-WAPE blend ───────
    blend_skus = set(low_conf["sku_ck"].unique()) | unrouted_skus
    if blend_skus:
        blend_matrix = dfu_accuracy_matrix[
            dfu_accuracy_matrix["sku_ck"].isin(blend_skus)
        ]
        blend_preds_input = all_predictions_df[
            all_predictions_df["sku_ck"].isin(blend_skus)
        ]

        if not blend_matrix.empty:
            blend_result = compute_inverse_wape_blend(
                blend_preds_input, blend_matrix, top_k=blend_top_k
            )
            blend_result = blend_result.copy()
            blend_result["algorithm_id"] = "hybrid"
            parts.append(blend_result)
            logger.info(
                "  Hybrid blend: %d DFUs → top-%d inverse-WAPE blend "
                "(confidence < %.2f or unrouted)",
                blend_result["sku_ck"].nunique(),
                blend_top_k,
                confidence_threshold,
            )
        else:
            # No accuracy history at all → fall back to seasonal_naive
            naive_fallback = all_predictions_df[
                (all_predictions_df["algorithm_id"] == "seasonal_naive")
                & (all_predictions_df["sku_ck"].isin(blend_skus))
            ][["sku_ck", "startdate", "basefcst_pref"]].copy()
            if not naive_fallback.empty:
                naive_fallback["algorithm_id"] = "hybrid"
                parts.append(naive_fallback)
            logger.warning(
                "  %d DFUs have no accuracy history — falling back to seasonal_naive",
                len(blend_skus),
            )

    if not parts:
        logger.warning("compute_hybrid_predictions: no predictions produced")
        return pd.DataFrame(
            columns=["sku_ck", "startdate", "basefcst_pref", "algorithm_id"]
        )

    result = pd.concat(parts, ignore_index=True)
    # High-confidence assignment takes priority over blend for the same DFU-month
    result = result.drop_duplicates(subset=["sku_ck", "startdate"], keep="first")
    result["algorithm_id"] = "hybrid"
    result["basefcst_pref"] = result["basefcst_pref"].clip(lower=0.0)

    n_total = result["sku_ck"].nunique()
    n_high = high_conf["sku_ck"].nunique()
    pct_high = 100.0 * n_high / max(n_total, 1)
    logger.info(
        "Hybrid predictions: %d rows, %d DFUs "
        "(%.0f%% single-algo, %.0f%% blend)",
        len(result),
        n_total,
        pct_high,
        100.0 - pct_high,
    )
    return result


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _route_single_algorithm(
    deduped_preds: pd.DataFrame,
    routing_df: pd.DataFrame,
    all_predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract single-algorithm predictions for high-confidence DFUs.

    Args:
        deduped_preds: Predictions averaged across timeframes
            (sku_ck, startdate, algorithm_id, basefcst_pref).
        routing_df: Per-DFU routing with columns
            [sku_ck, predicted_algorithm, confidence].
        all_predictions_df: Original undeduped predictions (for naive fallback).

    Returns:
        DataFrame [sku_ck, startdate, basefcst_pref, algorithm_id="hybrid"].
    """
    routed = routing_df[["sku_ck", "predicted_algorithm"]].merge(
        deduped_preds,
        left_on=["sku_ck", "predicted_algorithm"],
        right_on=["sku_ck", "algorithm_id"],
        how="left",
    )

    assigned = routed.dropna(subset=["basefcst_pref"])
    missing_mask = routed["basefcst_pref"].isna()
    n_missing = int(missing_mask.sum())

    if n_missing > 0:
        missing_skus = routed.loc[missing_mask, "sku_ck"].unique()
        naive = all_predictions_df[
            (all_predictions_df["algorithm_id"] == "seasonal_naive")
            & (all_predictions_df["sku_ck"].isin(missing_skus))
        ]
        assigned_base = assigned[["sku_ck", "startdate", "basefcst_pref"]].assign(
            algorithm_id="hybrid"
        )
        if not naive.empty:
            naive_deduped = (
                naive.groupby(["sku_ck", "startdate"])["basefcst_pref"]
                .mean()
                .reset_index()
                .assign(algorithm_id="hybrid")
            )
            result = pd.concat([assigned_base, naive_deduped], ignore_index=True)
        else:
            result = assigned_base
        logger.debug(
            "%d DFU-months missing predicted algorithm, fell back to seasonal_naive",
            n_missing,
        )
    else:
        result = assigned[["sku_ck", "startdate", "basefcst_pref"]].assign(
            algorithm_id="hybrid"
        )

    return result.drop_duplicates(subset=["sku_ck", "startdate"], keep="first")
