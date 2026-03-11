"""
Assign meaningful business labels to clusters based on feature centroids.

This script analyzes cluster centroids and assigns rich, hierarchical labels using
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

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    _required_vol_keys = ("very_high", "high", "low", "very_low")
    _missing = [k for k in _required_vol_keys if k not in volume_thresholds]
    if _missing:
        raise ValueError(
            f"volume_thresholds is missing required keys: {_missing}. "
            "Check clustering_config.yaml → labeling.volume_thresholds."
        )

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

        # ── Priority 1: Intermittency ───────────────────────────────────────
        if zero_pct > zero_demand_threshold or (adi > adi_threshold and adi > 0):
            return "intermittent"

        # ── Priority 2: Periodicity (strong non-12-month cycle) ────────────
        if periodicity > periodicity_threshold:
            return "periodic"

        # ── Priority 3: Seasonality ─────────────────────────────────────────
        is_seasonal = seas_amp > seasonality_threshold or seas_r2 > seasonality_r2_threshold

        # ── Priority 4: Trend ───────────────────────────────────────────────
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

        # ── Priority 5: Volatility ──────────────────────────────────────────
        if cv > cv_thresholds.get("very_volatile", 1.2):
            return "very_volatile"
        if cv > cv_thresholds.get("volatile", 0.8):
            return "volatile"

        # ── Steady tiers ────────────────────────────────────────────────────
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Label clusters based on feature centroids")
    parser.add_argument("--centroids", type=str, default="data/clustering/cluster_centroids.csv", help="Cluster centroids file")
    parser.add_argument("--assignments", type=str, default="data/clustering/cluster_assignments.csv", help="Cluster assignments file")
    parser.add_argument("--metadata", type=str, default="data/clustering/cluster_metadata.json", help="Cluster metadata file")
    parser.add_argument("--output", type=str, default="data/clustering/cluster_labels.csv", help="Output labeled assignments file")
    parser.add_argument("--config", type=str, default="config/clustering_config.yaml", help="Configuration file")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    # Load configuration (with defaults if file doesn't exist)
    config_path = root / args.config
    if config_path.exists():
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        labeling_config = config.get("clustering", {}).get("labeling", {})
    else:
        labeling_config = {
            "volume_thresholds": {
                "very_high": 0.90, "high": 0.75, "low": 0.25, "very_low": 0.10,
            },
            "cv_thresholds": {
                "very_steady": 0.2, "steady": 0.4, "volatile": 0.8, "very_volatile": 1.2,
            },
            "seasonality_threshold": 0.3,
            "seasonality_r2_threshold": 0.25,
            "periodicity_threshold": 0.25,
            "zero_demand_threshold": 0.15,
            "adi_threshold": 1.5,
            "trend_r2_threshold": 0.25,
            "cagr_growing": 5.0,
            "cagr_declining": -5.0,
            "recency_ratio_high": 1.2,
            "recency_ratio_low": 0.8,
        }

    volume_thresholds_pct = labeling_config.get(
        "volume_thresholds",
        {"very_high": 0.90, "high": 0.75, "low": 0.25, "very_low": 0.10},
    )
    cv_thresholds = labeling_config.get(
        "cv_thresholds",
        {"very_steady": 0.2, "steady": 0.4, "volatile": 0.8, "very_volatile": 1.2},
    )

    # Load centroids
    centroids_path = root / args.centroids
    if not centroids_path.exists():
        print(f"Error: Centroids file not found: {centroids_path}")
        sys.exit(1)

    centroids_df = pd.read_csv(centroids_path)
    print(f"Loaded {len(centroids_df)} cluster centroids")

    # Load assignments
    assignments_path = root / args.assignments
    if not assignments_path.exists():
        print(f"Error: Assignments file not found: {assignments_path}")
        sys.exit(1)

    assignments_df = pd.read_csv(assignments_path)
    print(f"Loaded {len(assignments_df)} DFU assignments")

    # Load metadata to get feature names
    metadata_path = root / args.metadata
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        feature_names = metadata.get("feature_names", [])
    else:
        feature_names = [c for c in centroids_df.columns if c != "cluster_id"]

    # Compute percentile-based volume thresholds from actual centroid mean_demand values
    mean_demands = centroids_df["mean_demand"].values
    if len(mean_demands) > 0:
        volume_thresholds_abs = {
            "very_high": float(np.percentile(mean_demands, volume_thresholds_pct.get("very_high", 0.90) * 100)),
            "high": float(np.percentile(mean_demands, volume_thresholds_pct.get("high", 0.75) * 100)),
            "low": float(np.percentile(mean_demands, volume_thresholds_pct.get("low", 0.25) * 100)),
            "very_low": float(np.percentile(mean_demands, volume_thresholds_pct.get("very_low", 0.10) * 100)),
        }
    else:
        volume_thresholds_abs = {
            "very_high": 100000.0, "high": 10000.0, "low": 500.0, "very_low": 100.0,
        }

    print(
        f"Volume thresholds (from centroid percentiles): "
        f"very_high={volume_thresholds_abs['very_high']:.0f}, "
        f"high={volume_thresholds_abs['high']:.0f}, "
        f"low={volume_thresholds_abs['low']:.0f}, "
        f"very_low={volume_thresholds_abs['very_low']:.0f}"
    )

    # Assign unique labels to all clusters
    cluster_labels = assign_cluster_labels(
        centroids_df,
        volume_thresholds_abs,
        cv_thresholds,
        labeling_config,
    )

    # Build cluster profiles with the full centroid feature set
    cluster_profiles = []
    for _, row in centroids_df.iterrows():
        cluster_id = int(row["cluster_id"])
        centroid = row.drop("cluster_id")
        label = cluster_labels[cluster_id]

        profile = {
            "cluster_id": cluster_id,
            "label": label,
            # Core volume
            "mean_demand": float(centroid.get("mean_demand", 0)),
            "cv_demand": float(centroid.get("cv_demand", 0)),
            "iqr_demand": float(centroid.get("iqr_demand", 0)),
            # Trend
            "trend_slope_norm": float(centroid.get("trend_slope_norm", 0)),
            "trend_r2": float(centroid.get("trend_r2", 0)),
            "cagr": float(centroid.get("cagr", 0)),
            # Seasonality
            "seasonal_amplitude": float(centroid.get("seasonal_amplitude", 0)),
            "seasonal_r2": float(centroid.get("seasonal_r2", 0)),
            "yoy_correlation": float(centroid.get("yoy_correlation", 0)),
            # Periodicity
            "periodicity_strength": float(centroid.get("periodicity_strength", 0)),
            # Intermittency
            "zero_demand_pct": float(centroid.get("zero_demand_pct", 0)),
            "adi": float(centroid.get("adi", 0)),
            # Lifecycle
            "months_available": float(centroid.get("months_available", 0)),
            "recency_ratio": float(centroid.get("recency_ratio", 1.0)),
        }
        cluster_profiles.append(profile)
        print(f"Cluster {cluster_id}: {label}")

    # Validate label uniqueness
    labels_list = list(cluster_labels.values())
    if len(labels_list) != len(set(labels_list)):
        dupes = [lbl for lbl, cnt in Counter(labels_list).items() if cnt > 1]
        print(f"Warning: {len(dupes)} duplicate labels remain: {dupes}")
    else:
        print(f"All {len(labels_list)} cluster labels are unique.")

    # Add labels to assignments
    assignments_df["cluster_label"] = assignments_df["cluster_id"].map(cluster_labels)

    # Count DFUs per cluster
    cluster_counts = assignments_df.groupby("cluster_id").size()
    total_dfus = len(assignments_df)
    for cluster_id, count in cluster_counts.items():
        label = cluster_labels.get(cluster_id, "unknown")
        pct = count / total_dfus * 100
        print(f"  Cluster {cluster_id} ({label}): {count:,} DFUs ({pct:.1f}%)")

    # Save labeled assignments
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments_df.to_csv(output_path, index=False)
    print(f"\nSaved labeled assignments to {output_path}")

    # Save cluster profiles
    profiles_path = output_path.parent / "cluster_profiles.json"
    with open(profiles_path, "w") as f:
        json.dump(cluster_profiles, f, indent=2)
    print(f"Saved cluster profiles to {profiles_path}")


if __name__ == "__main__":
    main()
