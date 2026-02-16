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


def assign_cluster_label(
    centroid: pd.Series,
    volume_thresholds: dict[str, float],
    cv_thresholds: dict[str, float],
    seasonality_threshold: float,
    zero_demand_threshold: float
) -> str:
    """Assign a label to a cluster based on its centroid features."""
    
    mean_demand = centroid.get("mean_demand", 0)
    cv_demand = centroid.get("cv_demand", 0)
    seasonality_strength = centroid.get("seasonality_strength", 0)
    trend_slope = centroid.get("trend_slope", 0)
    growth_rate = centroid.get("growth_rate", 0)
    zero_demand_pct = centroid.get("zero_demand_pct", 0)
    
    # Determine volume tier
    # Note: thresholds are percentiles, so we need to compare relative to other clusters
    # For now, use absolute thresholds based on typical values
    if mean_demand > volume_thresholds.get("high", 1000):
        volume_tier = "high_volume"
    elif mean_demand < volume_thresholds.get("low", 100):
        volume_tier = "low_volume"
    else:
        volume_tier = "medium_volume"
    
    # Determine pattern type
    pattern_type = None
    
    # Check for intermittent pattern
    if zero_demand_pct > zero_demand_threshold:
        pattern_type = "intermittent"
    # Check for seasonal pattern
    elif seasonality_strength > seasonality_threshold:
        pattern_type = "seasonal"
    # Check for trending patterns
    elif trend_slope > 0.01 or growth_rate > 5:
        pattern_type = "trending_up"
    elif trend_slope < -0.01 or growth_rate < -5:
        pattern_type = "trending_down"
    # Check for volatile pattern
    elif cv_demand > cv_thresholds.get("volatile", 0.8):
        pattern_type = "volatile"
    # Check for steady pattern
    elif cv_demand < cv_thresholds.get("steady", 0.3) and abs(trend_slope) < 0.01:
        pattern_type = "steady"
    else:
        pattern_type = "mixed"
    
    # Combine into composite label if both dimensions are strong
    if volume_tier == "high_volume" and pattern_type == "steady":
        return "high_volume_steady"
    elif volume_tier == "high_volume" and pattern_type == "seasonal":
        return "seasonal_high_volume"
    elif volume_tier == "low_volume" and pattern_type == "intermittent":
        return "intermittent_low_volume"
    elif volume_tier == "high_volume" and pattern_type == "trending_up":
        return "high_volume_growing"
    elif volume_tier == "low_volume" and pattern_type == "trending_down":
        return "low_volume_declining"
    elif volume_tier == "medium_volume" and pattern_type == "seasonal":
        return "seasonal_medium_volume"
    elif volume_tier == "medium_volume" and pattern_type == "steady":
        return "medium_volume_steady"
    else:
        # Return volume-based label if pattern is weak
        return f"{volume_tier}_{pattern_type}" if pattern_type != "mixed" else volume_tier


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
    # We'll use the mean_demand values from centroids
    mean_demands = centroids_df["mean_demand"].values
    if len(mean_demands) > 0:
        volume_high_threshold = np.percentile(mean_demands, volume_thresholds.get("high", 75) * 100)
        volume_low_threshold = np.percentile(mean_demands, volume_thresholds.get("low", 25) * 100)
        volume_thresholds_abs = {"high": volume_high_threshold, "low": volume_low_threshold}
    else:
        volume_thresholds_abs = {"high": 1000, "low": 100}
    
    print(f"Volume thresholds: high={volume_thresholds_abs['high']:.2f}, low={volume_thresholds_abs['low']:.2f}")
    
    # Assign labels to each cluster
    cluster_labels = {}
    cluster_profiles = []
    
    for _, row in centroids_df.iterrows():
        cluster_id = int(row["cluster_id"])
        centroid = row.drop("cluster_id")
        
        label = assign_cluster_label(
            centroid,
            volume_thresholds_abs,
            cv_thresholds,
            seasonality_threshold,
            zero_demand_threshold
        )
        
        cluster_labels[cluster_id] = label
        
        # Create profile
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
        print("Warning: Some clusters have duplicate labels!")
        from collections import Counter
        duplicates = [label for label, count in Counter(labels).items() if count > 1]
        print(f"Duplicate labels: {duplicates}")
    
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
