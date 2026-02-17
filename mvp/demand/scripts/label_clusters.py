"""
Assign meaningful business labels to clusters based on feature centroids.

This script analyzes cluster centroids and assigns labels like "high_volume_steady",
"seasonal_high_volume", "intermittent_low_volume", etc.
"""

import argparse
import json
import sys
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
    seasonality_threshold: float,
    zero_demand_threshold: float,
) -> dict[int, str]:
    """Assign unique labels to all clusters based on centroid features.

    Uses a two-pass approach: first assigns base labels from volume + pattern,
    then disambiguates any duplicates using secondary features.
    """

    def _volume_tier(mean_demand: float) -> str:
        if mean_demand > volume_thresholds.get("very_high", volume_thresholds.get("high", 1000) * 10):
            return "very_high_volume"
        if mean_demand > volume_thresholds.get("high", 1000):
            return "high_volume"
        if mean_demand < volume_thresholds.get("low", 100):
            return "low_volume"
        return "medium_volume"

    def _pattern(row: pd.Series) -> str:
        cv = row.get("cv_demand", 0)
        seas = row.get("seasonality_strength", 0)
        slope = row.get("trend_slope", 0)
        growth = row.get("growth_rate", 0)
        zero_pct = row.get("zero_demand_pct", 0)

        if zero_pct > zero_demand_threshold:
            return "intermittent"
        if seas > seasonality_threshold and cv > cv_thresholds.get("volatile", 0.8):
            return "seasonal_volatile"
        if seas > seasonality_threshold:
            return "seasonal"
        if slope > 0.05 or growth > 5:
            return "growing"
        if slope < -0.05 or growth < -5:
            return "declining"
        if slope > 0.01 or growth > 1:
            return "slight_growth"
        if slope < -0.01 or growth < -1:
            return "slight_decline"
        if cv > cv_thresholds.get("volatile", 0.8):
            return "volatile"
        if cv < cv_thresholds.get("steady", 0.3):
            return "steady"
        return "moderate"

    def _disambiguator(row: pd.Series) -> str:
        """Return a secondary descriptor to break ties."""
        cv = row.get("cv_demand", 0)
        seas = row.get("seasonality_strength", 0)
        mean_d = row.get("mean_demand", 0)
        growth = row.get("growth_rate", 0)

        # Pick the most distinguishing secondary trait
        if seas > seasonality_threshold:
            return "seasonal"
        if cv > cv_thresholds.get("volatile", 0.8):
            return "volatile"
        if cv < cv_thresholds.get("steady", 0.3):
            return "stable"
        if abs(growth) < 0.5:
            return "flat"
        if growth > 3:
            return "accelerating"
        if growth < -3:
            return "contracting"
        if mean_d > 200:
            return "higher_avg"
        if mean_d > 90:
            return "mid_range"
        return "lower_avg"

    # Pass 1: assign base labels
    base_labels: dict[int, str] = {}
    for _, row in centroids_df.iterrows():
        cid = int(row["cluster_id"])
        centroid = row.drop("cluster_id")
        vol = _volume_tier(centroid.get("mean_demand", 0))
        pat = _pattern(centroid)
        base_labels[cid] = f"{vol}_{pat}"

    # Pass 2: disambiguate duplicates
    from collections import Counter
    label_counts = Counter(base_labels.values())
    duplicated = {lbl for lbl, cnt in label_counts.items() if cnt > 1}

    final_labels: dict[int, str] = {}
    for cid, base in base_labels.items():
        if base not in duplicated:
            final_labels[cid] = base
        else:
            row = centroids_df[centroids_df["cluster_id"] == cid].iloc[0]
            suffix = _disambiguator(row.drop("cluster_id"))
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
        # Use defaults
        labeling_config = {
            "volume_thresholds": {"high": 0.75, "low": 0.25},
            "cv_thresholds": {"steady": 0.3, "volatile": 0.8},
            "seasonality_threshold": 0.5,
            "zero_demand_threshold": 0.2,
        }
    
    volume_thresholds = labeling_config.get("volume_thresholds", {"high": 0.75, "low": 0.25})
    cv_thresholds = labeling_config.get("cv_thresholds", {"steady": 0.3, "volatile": 0.8})
    seasonality_threshold = labeling_config.get("seasonality_threshold", 0.5)
    zero_demand_threshold = labeling_config.get("zero_demand_threshold", 0.2)
    
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
        # Infer from centroids
        feature_names = [c for c in centroids_df.columns if c != "cluster_id"]
    
    # Compute percentile-based volume thresholds from actual data
    mean_demands = centroids_df["mean_demand"].values
    if len(mean_demands) > 0:
        volume_high_threshold = np.percentile(mean_demands, volume_thresholds.get("high", 75) * 100)
        volume_low_threshold = np.percentile(mean_demands, volume_thresholds.get("low", 25) * 100)
        very_high_threshold = volume_high_threshold * 10
        volume_thresholds_abs = {
            "very_high": very_high_threshold,
            "high": volume_high_threshold,
            "low": volume_low_threshold,
        }
    else:
        volume_thresholds_abs = {"very_high": 100000, "high": 1000, "low": 100}

    print(f"Volume thresholds: very_high={volume_thresholds_abs['very_high']:.0f}, high={volume_thresholds_abs['high']:.0f}, low={volume_thresholds_abs['low']:.0f}")

    # Assign unique labels to all clusters
    cluster_labels = assign_cluster_labels(
        centroids_df,
        volume_thresholds_abs,
        cv_thresholds,
        seasonality_threshold,
        zero_demand_threshold,
    )

    cluster_profiles = []
    for _, row in centroids_df.iterrows():
        cluster_id = int(row["cluster_id"])
        centroid = row.drop("cluster_id")
        label = cluster_labels[cluster_id]

        profile = {
            "cluster_id": cluster_id,
            "label": label,
            "mean_demand": float(centroid.get("mean_demand", 0)),
            "cv_demand": float(centroid.get("cv_demand", 0)),
            "seasonality_strength": float(centroid.get("seasonality_strength", 0)),
            "trend_slope": float(centroid.get("trend_slope", 0)),
            "growth_rate": float(centroid.get("growth_rate", 0)),
            "zero_demand_pct": float(centroid.get("zero_demand_pct", 0)),
        }
        cluster_profiles.append(profile)
        print(f"Cluster {cluster_id}: {label}")

    # Validate label uniqueness
    labels = list(cluster_labels.values())
    if len(labels) != len(set(labels)):
        print("Warning: Some clusters still have duplicate labels!")
    else:
        print(f"All {len(labels)} cluster labels are unique.")
    
    # Add labels to assignments
    assignments_df["cluster_label"] = assignments_df["cluster_id"].map(cluster_labels)
    
    # Count DFUs per cluster
    cluster_counts = assignments_df.groupby("cluster_id").size()
    for cluster_id, count in cluster_counts.items():
        label = cluster_labels.get(cluster_id, "unknown")
        print(f"  Cluster {cluster_id} ({label}): {count} DFUs")
    
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
