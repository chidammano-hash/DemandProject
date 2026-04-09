"""
Assign meaningful business labels to clusters based on feature centroids.

This module analyzes cluster centroids and assigns rich, hierarchical labels using
a priority-ordered taxonomy:
  1. Intermittency check (ADI / zero-demand-pct) -- FIRST
  2. Periodicity check (non-12-month cycles)
  3. Seasonality check (amplitude + seasonal_r2)
  4. Trend check (trend_r2 + CAGR together)
  5. Volatility check (cv_demand)
  6. Volume tier (very_high / high / medium / low / very_low)

Example labels produced:
  high_volume_seasonal_growing
  medium_volume_steady_seasonal
  low_volume_intermittent
  high_volume_periodic
  very_high_volume_growing
  medium_volume_volatile
  low_volume_seasonal
  high_volume_declining
  medium_volume_steady
  very_low_volume_intermittent
"""

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def assign_cluster_labels(
    centroids_df: pd.DataFrame,
    volume_thresholds: dict[str, float],
    cv_thresholds: dict[str, float],
    labeling_config: dict[str, Any],
) -> dict[int, str]:
    """Assign unique labels to all clusters based on centroid features.

    Priority order (highest to lowest):
      1. Intermittency (zero_demand_pct or ADI)
      2. Periodicity (non-12-month cycles)
      3. Seasonality (amplitude + seasonal_r2)
      4. Trend (trend_r2 + CAGR)
      5. Volatility (cv_demand)
      6. Volume tier (very_high/high/medium/low/very_low)

    Uses a two-pass approach: first assigns base labels from volume + pattern,
    then disambiguates any remaining duplicates using secondary features.
    """
    # Pull thresholds from labeling_config
    seasonality_threshold = labeling_config.get("seasonality_threshold", 0.3)
    seasonality_r2_threshold = labeling_config.get("seasonality_r2_threshold", 0.25)
    periodicity_threshold = labeling_config.get("periodicity_threshold", 0.25)
    zero_demand_threshold = labeling_config.get("zero_demand_threshold", 0.15)
    adi_threshold = labeling_config.get("adi_threshold", 1.5)
    trend_r2_threshold = labeling_config.get("trend_r2_threshold", 0.25)
    cagr_growing = labeling_config.get("cagr_growing", 5.0)
    cagr_declining = labeling_config.get("cagr_declining", -5.0)
    recency_ratio_high = labeling_config.get("recency_ratio_high", 1.2)
    recency_ratio_low = labeling_config.get("recency_ratio_low", 0.8)

    # Validate required volume threshold keys up-front so missing keys cause a
    # clear error rather than silently misclassifying all clusters.
    _validate_volume_thresholds(volume_thresholds)

    def _volume_tier(mean_demand: float) -> str:
        """Map centroid mean_demand to a volume tier using percentile thresholds."""
        if mean_demand >= volume_thresholds["very_high"]:
            return "very_high_volume"
        if mean_demand >= volume_thresholds["high"]:
            return "high_volume"
        if mean_demand <= volume_thresholds["very_low"]:
            return "very_low_volume"
        if mean_demand <= volume_thresholds["low"]:
            return "low_volume"
        return "medium_volume"

    def _demand_pattern(row: pd.Series) -> str:
        """Determine the primary demand pattern using the priority taxonomy."""
        zero_pct = float(row.get("zero_demand_pct", 0))
        adi = float(row.get("adi", 0))
        periodicity = float(row.get("periodicity_strength", 0))
        seas_amp = float(row.get("seasonal_amplitude", 0))
        seas_r2 = float(row.get("seasonal_r2", 0))
        trend_r2 = float(row.get("trend_r2", 0))
        cagr = float(row.get("cagr", 0))
        cv = float(row.get("cv_demand", 0))
        recency = float(row.get("recency_ratio", 1.0))

        # -- Priority 1: Intermittency --
        if zero_pct > zero_demand_threshold or (adi > adi_threshold and adi > 0):
            return "intermittent"

        # -- Priority 2: Periodicity (strong non-12-month cycle) --
        if periodicity > periodicity_threshold:
            return "periodic"

        # -- Priority 3: Seasonality --
        is_seasonal = seas_amp > seasonality_threshold or seas_r2 > seasonality_r2_threshold

        # -- Priority 4: Trend --
        has_trend = abs(trend_r2) > trend_r2_threshold
        is_growing = has_trend and cagr > cagr_growing
        is_declining = has_trend and cagr < cagr_declining

        # Combine seasonal + trend into compound label
        if is_seasonal and is_growing:
            return "seasonal_growing"
        if is_seasonal and is_declining:
            return "seasonal_declining"
        if is_seasonal:
            # Check volatility within seasonal bucket
            if cv > cv_thresholds.get("volatile", 0.8):
                return "seasonal_volatile"
            return "seasonal"

        if is_growing:
            # Accelerating recency bonus
            if recency > recency_ratio_high:
                return "accelerating"
            return "growing"
        if is_declining:
            if recency < recency_ratio_low:
                return "decelerating"
            return "declining"

        # -- Priority 5: Volatility --
        if cv > cv_thresholds.get("very_volatile", 1.2):
            return "very_volatile"
        if cv > cv_thresholds.get("volatile", 0.8):
            return "volatile"

        # -- Steady tiers --
        if cv < cv_thresholds.get("very_steady", 0.2):
            return "very_steady"
        if cv < cv_thresholds.get("steady", 0.4):
            return "steady"

        return "moderate"

    def _disambiguator(row: pd.Series, base_label: str) -> str:
        """Return a secondary descriptor to break ties on duplicate base labels."""
        cv = float(row.get("cv_demand", 0))
        seas_amp = float(row.get("seasonal_amplitude", 0))
        cagr = float(row.get("cagr", 0))
        adi = float(row.get("adi", 0))
        mean_d = float(row.get("mean_demand", 0))
        recency = float(row.get("recency_ratio", 1.0))

        if seas_amp > seasonality_threshold:
            return "seasonal"
        if adi > adi_threshold:
            return "sparse"
        if cv > cv_thresholds.get("volatile", 0.8):
            return "volatile"
        if cv < cv_thresholds.get("steady", 0.4):
            return "stable"
        if cagr > cagr_growing:
            return "growing"
        if cagr < cagr_declining:
            return "declining"
        if recency > recency_ratio_high:
            return "accelerating"
        if recency < recency_ratio_low:
            return "decelerating"
        # Final fallback: split by volume sub-range
        if mean_d > np.percentile([mean_d], 75):
            return "higher_avg"
        return "lower_avg"

    # Pass 1: assign base labels
    base_labels: dict[int, str] = {}
    for _, row in centroids_df.iterrows():
        cid = int(row["cluster_id"])
        centroid = row.drop("cluster_id")
        vol = _volume_tier(float(centroid.get("mean_demand", 0)))
        pat = _demand_pattern(centroid)
        base_labels[cid] = f"{vol}_{pat}"

    # Pass 2: disambiguate duplicates
    label_counts = Counter(base_labels.values())
    duplicated = {lbl for lbl, cnt in label_counts.items() if cnt > 1}

    final_labels: dict[int, str] = {}
    for cid, base in base_labels.items():
        if base not in duplicated:
            final_labels[cid] = base
        else:
            row = centroids_df[centroids_df["cluster_id"] == cid].iloc[0]
            suffix = _disambiguator(row.drop("cluster_id"), base)
            candidate = f"{base}_{suffix}"
            # If still duplicate, append cluster_id as last resort
            if candidate in final_labels.values():
                candidate = f"{base}_c{cid}"
            final_labels[cid] = candidate

    return final_labels


def _validate_volume_thresholds(volume_thresholds: dict[str, float]) -> None:
    """Validate that all required keys are present in volume_thresholds.

    Raises ValueError with a clear message listing any missing keys.
    """
    _required_vol_keys = ("very_high", "high", "low", "very_low")
    _missing = [k for k in _required_vol_keys if k not in volume_thresholds]
    if _missing:
        raise ValueError(
            f"volume_thresholds is missing required keys: {_missing}. "
            "Check labeling.volume_thresholds in cluster experiment params."
        )
