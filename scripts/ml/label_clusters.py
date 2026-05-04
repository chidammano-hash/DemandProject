"""
Assign meaningful business labels to clusters based on feature centroids.

Thin CLI shim -- the core labeling logic lives in
``common.ml.clustering.labeling``.  This script re-exports
``assign_cluster_labels`` for backward compatibility and provides the
``main()`` CLI entry-point.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-export for backward compatibility
from common.ml.clustering.labeling import assign_cluster_labels  # noqa: E402, F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Label clusters based on feature centroids")
    parser.add_argument("--centroids", type=str, default="data/clustering/cluster_centroids.csv", help="Cluster centroids file")
    parser.add_argument("--assignments", type=str, default="data/clustering/cluster_assignments.csv", help="Cluster assignments file")
    parser.add_argument("--metadata", type=str, default="data/clustering/cluster_metadata.json", help="Cluster metadata file")
    parser.add_argument("--output", type=str, default="data/clustering/cluster_labels.csv", help="Output labeled assignments file")
    parser.add_argument("--config", type=str, default=None, help="Optional config YAML (defaults resolved from DB)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")

    # Load configuration (with defaults if file doesn't exist)
    config_path = root / args.config
    if config_path.exists():
        import yaml
        with open(config_path) as f:
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

    # Load metadata (feature_names available for future centroid analysis)
    metadata_path = root / args.metadata
    if metadata_path.exists():
        with open(metadata_path) as f:
            json.load(f)  # metadata loaded for validation; feature_names reserved for future use

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
