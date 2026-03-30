"""Syntetos-Boylan demand archetype classification.

Classifies DFUs into 8 archetypes: {smooth,erratic,intermittent,lumpy} x {high,low} volume.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def classify_demand(
    sales_df: pd.DataFrame,
    adi_threshold: float = 1.32,
    cv2_threshold: float = 0.49,
    high_volume_percentile: int = 90,
    min_history_months: int = 6,
) -> pd.DataFrame:
    """Classify each DFU into a Syntetos-Boylan demand archetype.

    Args:
        sales_df: Sales data with columns [sku_ck, startdate, qty].
        adi_threshold: ADI boundary (1.32 per Syntetos & Boylan 2005).
        cv2_threshold: CV² boundary (0.49 per Syntetos & Boylan 2005).
        high_volume_percentile: Percentile threshold for high/low volume split.
        min_history_months: Minimum months of history to classify (else 'insufficient').

    Returns:
        DataFrame with columns:
            sku_ck, n_periods, n_nonzero, adi, cv2, mean_demand, std_demand,
            segment (smooth|erratic|intermittent|lumpy|insufficient),
            volume_tier (high|low),
            archetype (e.g. "smooth_high", "lumpy_low", "insufficient")
    """
    if sales_df.empty:
        logger.warning("Empty sales_df passed to classify_demand; returning empty result")
        return pd.DataFrame(
            columns=[
                "sku_ck", "n_periods", "n_nonzero", "adi", "cv2",
                "mean_demand", "std_demand", "segment", "volume_tier", "archetype",
            ]
        )

    required_cols = {"sku_ck", "startdate", "qty"}
    missing = required_cols - set(sales_df.columns)
    if missing:
        raise ValueError(f"sales_df missing required columns: {missing}")

    logger.info("Classifying demand for %d rows across DFUs", len(sales_df))

    # --- Per-DFU aggregation ---------------------------------------------------
    grouped = sales_df.groupby("sku_ck", sort=False)

    n_periods = grouped["startdate"].nunique().rename("n_periods")
    n_nonzero = grouped["qty"].apply(lambda s: (s > 0).sum()).rename("n_nonzero")

    # ADI: average demand interval = n_periods / n_nonzero
    adi = (n_periods / n_nonzero.replace(0, np.nan)).fillna(n_periods).rename("adi")

    # Non-zero demand stats for CV² computation
    nz_stats = (
        sales_df.loc[sales_df["qty"] > 0]
        .groupby("sku_ck", sort=False)["qty"]
        .agg(mean_nz="mean", std_nz="std")
    )
    # std returns NaN for single-element groups; fill with 0
    nz_stats["std_nz"] = nz_stats["std_nz"].fillna(0.0)

    # Overall demand stats (including zero months)
    overall_stats = grouped["qty"].agg(mean_demand="mean", std_demand="std")
    overall_stats["std_demand"] = overall_stats["std_demand"].fillna(0.0)

    # Assemble intermediate DataFrame
    result = pd.DataFrame({"n_periods": n_periods, "n_nonzero": n_nonzero, "adi": adi})
    result = result.join(nz_stats).join(overall_stats)

    # CV² = (std_nz / mean_nz)² -- guard against zero mean_nz
    safe_mean_nz = result["mean_nz"].replace(0, np.nan)
    result["cv2"] = ((result["std_nz"] / safe_mean_nz) ** 2).fillna(0.0)

    # Drop intermediate non-zero stats
    result = result.drop(columns=["mean_nz", "std_nz"])

    # --- Segment assignment ----------------------------------------------------
    insufficient_mask = result["n_periods"] < min_history_months

    smooth_mask = (~insufficient_mask) & (result["adi"] < adi_threshold) & (result["cv2"] < cv2_threshold)
    erratic_mask = (~insufficient_mask) & (result["adi"] < adi_threshold) & (result["cv2"] >= cv2_threshold)
    intermittent_mask = (~insufficient_mask) & (result["adi"] >= adi_threshold) & (result["cv2"] < cv2_threshold)
    lumpy_mask = (~insufficient_mask) & (result["adi"] >= adi_threshold) & (result["cv2"] >= cv2_threshold)

    result["segment"] = "insufficient"
    result.loc[smooth_mask, "segment"] = "smooth"
    result.loc[erratic_mask, "segment"] = "erratic"
    result.loc[intermittent_mask, "segment"] = "intermittent"
    result.loc[lumpy_mask, "segment"] = "lumpy"

    # --- Volume tier -----------------------------------------------------------
    classified_mask = ~insufficient_mask
    if classified_mask.any():
        volume_threshold = np.percentile(
            result.loc[classified_mask, "mean_demand"].values,
            high_volume_percentile,
        )
    else:
        volume_threshold = 0.0

    result["volume_tier"] = np.where(
        result["mean_demand"] >= volume_threshold, "high", "low"
    )

    # --- Archetype label -------------------------------------------------------
    result["archetype"] = np.where(
        insufficient_mask,
        "insufficient",
        result["segment"] + "_" + result["volume_tier"],
    )

    # Reset index so sku_ck becomes a column
    result = result.reset_index()

    col_order = [
        "sku_ck", "n_periods", "n_nonzero", "adi", "cv2",
        "mean_demand", "std_demand", "segment", "volume_tier", "archetype",
    ]
    result = result[col_order]

    segment_counts = result["archetype"].value_counts().to_dict()
    logger.info(
        "Classified %d DFUs into %d archetypes: %s",
        len(result),
        len(segment_counts),
        segment_counts,
    )

    return result


def get_segment_summary(classification_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize DFU counts and demand stats per archetype.

    Args:
        classification_df: Output of classify_demand().

    Returns:
        DataFrame grouped by archetype with columns:
            archetype, n_dfus, mean_adi, mean_cv2, mean_demand, total_demand.
    """
    if classification_df.empty:
        logger.warning("Empty classification_df passed to get_segment_summary")
        return pd.DataFrame(
            columns=["archetype", "n_dfus", "mean_adi", "mean_cv2", "mean_demand", "total_demand"]
        )

    summary = (
        classification_df.groupby("archetype", sort=False)
        .agg(
            n_dfus=("sku_ck", "count"),
            mean_adi=("adi", "mean"),
            mean_cv2=("cv2", "mean"),
            mean_demand=("mean_demand", "mean"),
            total_demand=("mean_demand", "sum"),
        )
        .reset_index()
        .sort_values("n_dfus", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("Segment summary: %d archetypes covering %d DFUs", len(summary), summary["n_dfus"].sum())

    return summary
