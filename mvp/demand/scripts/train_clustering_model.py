"""
Train clustering model on DFU features with optimal K selection.

This script performs KMeans clustering with multiple K selection methods
(elbow, silhouette, gap statistic) and logs results to MLflow.

Key design choices for balanced clusters (tree-model friendly):
  1. Log-transform skewed volume features before scaling
  2. Select only volume/trend/seasonality features (drop item attributes)
  3. Merge small clusters into nearest large neighbor post-hoc
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Feature groups ──────────────────────────────────────────────────────────
# Core features that drive business-meaningful clusters for tree models.
# Volume + trend + seasonality + history length.
CORE_FEATURES = [
    # Volume (will be log-transformed)
    "mean_demand",
    "cv_demand",
    # Trend
    "trend_slope",
    "growth_rate",
    # Seasonality
    "seasonality_strength",
    # History
    "months_available",
    # Volatility / pattern
    "zero_demand_pct",
    "demand_stability",
]

# Features that get log1p-transformed (highly skewed, spans orders of magnitude)
LOG_TRANSFORM_FEATURES = [
    "mean_demand",
    "median_demand",
    "std_demand",
    "total_demand",
    "max_demand",
    "min_demand",
    "seasonal_index_std",
]


def gap_statistic(X: np.ndarray, k: int, n_refs: int = 10, random_state: int = 42) -> float:
    """Compute gap statistic for K selection."""
    np.random.seed(random_state)

    # Fit KMeans to actual data
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(X)
    actual_inertia = kmeans.inertia_

    # Generate reference datasets
    ref_inertias = []
    for _ in range(n_refs):
        random_data = np.random.uniform(
            low=X.min(axis=0),
            high=X.max(axis=0),
            size=X.shape
        )
        ref_kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        ref_kmeans.fit(random_data)
        ref_inertias.append(ref_kmeans.inertia_)

    gap = np.log(np.mean(ref_inertias)) - np.log(actual_inertia)
    return gap


def merge_small_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    min_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge clusters smaller than min_size into nearest large neighbor.

    Returns (new_labels, new_centroids) with contiguous cluster IDs.
    """
    labels = labels.copy()
    unique, counts = np.unique(labels, return_counts=True)
    size_map = dict(zip(unique, counts))

    large = {c for c, n in size_map.items() if n >= min_size}
    small = {c for c, n in size_map.items() if n < min_size}

    if not small:
        return labels, centroids

    print(f"\nMerging {len(small)} small clusters (min_size={min_size}):")
    for sc in sorted(small):
        print(f"  Cluster {sc} ({size_map[sc]} DFUs) → ", end="")
        # Find nearest large cluster by centroid distance
        sc_centroid = centroids[sc]
        best_dist = np.inf
        best_target = None
        for lc in sorted(large):
            d = np.linalg.norm(sc_centroid - centroids[lc])
            if d < best_dist:
                best_dist = d
                best_target = lc
        print(f"merged into cluster {best_target} (dist={best_dist:.4f})")
        labels[labels == sc] = best_target

    # Re-number labels to be contiguous 0..K-1
    remaining = sorted(set(labels))
    remap = {old: new for new, old in enumerate(remaining)}
    labels = np.array([remap[l] for l in labels])

    # Recompute centroids from actual data
    new_k = len(remaining)
    new_centroids = np.zeros((new_k, X_scaled.shape[1]))
    for cid in range(new_k):
        mask = labels == cid
        new_centroids[cid] = X_scaled[mask].mean(axis=0)

    return labels, new_centroids


def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: tuple[int, int],
    min_cluster_size_pct: float = 0.01,
    skip_gap: bool = False,
) -> dict[str, Any]:
    """Find optimal K using multiple methods."""
    k_min, k_max = k_range
    k_values = list(range(k_min, k_max + 1))

    inertias = []
    silhouette_scores = []
    gap_stats = []
    cluster_sizes_list = []

    n_samples = len(X_scaled)
    min_cluster_size = int(n_samples * min_cluster_size_pct)

    print(f"Testing K values from {k_min} to {k_max} ({n_samples} samples, skip_gap={skip_gap})...")

    for k in k_values:
        print(f"  K={k}...", end=" ", flush=True)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouette_scores.append(sil)

        # Gap statistic (expensive — skip for large datasets)
        if not skip_gap and k >= 2:
            gap = gap_statistic(X_scaled, k, n_refs=5)
            gap_stats.append(gap)
        else:
            gap_stats.append(0.0)

        # Check cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes_list.append(dict(zip(unique, counts)))

        smallest = min(counts)
        print(f"silhouette={sil:.4f}, smallest_cluster={smallest}")
        if smallest < min_cluster_size:
            print(f"    Warning: smallest cluster {smallest} < {min_cluster_size}")

    # Find optimal K by silhouette score
    optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]

    # Find optimal K by gap statistic
    optimal_k_gap = optimal_k_silhouette
    if not skip_gap and any(g > 0 for g in gap_stats):
        gaps = np.array(gap_stats)
        s = np.zeros(len(k_values))
        for i, k in enumerate(k_values):
            if gaps[i] > 0:
                ref_gaps = [gap_statistic(X_scaled, k, n_refs=5, random_state=seed) for seed in [42, 123, 456]]
                s[i] = np.std(ref_gaps)
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i + 1] - s[i + 1]:
                optimal_k_gap = k_values[i]
                break

    # Elbow method: find point of maximum curvature
    if len(inertias) >= 3:
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        optimal_k_elbow = k_values[np.argmin(second_diffs) + 1] if len(second_diffs) > 0 else optimal_k_silhouette
    else:
        optimal_k_elbow = optimal_k_silhouette

    # Choose best K (prefer silhouette, fallback to elbow)
    optimal_k = optimal_k_silhouette

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "gap_stats": gap_stats,
        "optimal_k": optimal_k,
        "optimal_k_silhouette": optimal_k_silhouette,
        "optimal_k_elbow": optimal_k_elbow,
        "optimal_k_gap": optimal_k_gap,
        "cluster_sizes": cluster_sizes_list,
    }


def plot_k_selection(
    k_values: list[int],
    inertias: list[float],
    silhouette_scores: list[float],
    gap_stats: list[float],
    output_dir: Path
) -> None:
    """Generate visualization plots for K selection."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Elbow plot
    axes[0].plot(k_values, inertias, "bo-")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Within-Cluster Sum of Squares (WCSS)")
    axes[0].set_title("Elbow Method")
    axes[0].grid(True)

    # Silhouette plot
    axes[1].plot(k_values, silhouette_scores, "ro-")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score")
    axes[1].grid(True)

    # Gap statistic plot
    if len(gap_stats) > 0:
        axes[2].plot(k_values, gap_stats, "go-")
        axes[2].set_xlabel("Number of Clusters (K)")
        axes[2].set_ylabel("Gap Statistic")
        axes[2].set_title("Gap Statistic")
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "k_selection_plots.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_cluster_visualization(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    output_dir: Path
) -> None:
    """Generate 2D PCA visualization of clusters."""
    if X_scaled.shape[1] < 2:
        return

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.title("Cluster Visualization (2D PCA)")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "cluster_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train clustering model with optimal K selection")
    parser.add_argument("--input", type=str, default="data/clustering_features.csv", help="Input feature matrix")
    parser.add_argument("--k-range", type=int, nargs=2, default=[3, 12], metavar=("MIN", "MAX"), help="K range to test")
    parser.add_argument("--min-cluster-size-pct", type=float, default=0.02, help="Minimum cluster size as pct of total (clusters below this get merged)")
    parser.add_argument("--use-pca", action="store_true", help="Use PCA for dimensionality reduction")
    parser.add_argument("--pca-components", type=int, default=None, help="Number of PCA components (auto if not specified)")
    parser.add_argument("--output-dir", type=str, default="data/clustering", help="Output directory for artifacts")
    parser.add_argument("--skip-gap", action="store_true", help="Skip gap statistic (much faster for large datasets)")
    parser.add_argument("--all-features", action="store_true", help="Use all numeric features instead of core volume/trend/seasonality subset")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

    # Load feature matrix
    input_path = root / args.input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading feature matrix from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")

    # ── Feature selection ───────────────────────────────────────────────────
    metadata_cols = ["dfu_ck", "dmdunit", "dmdgroup", "loc"]

    if args.all_features:
        feature_cols = [c for c in df.columns if c not in metadata_cols]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    else:
        # Use only core volume/trend/seasonality features
        numeric_cols = [c for c in CORE_FEATURES if c in df.columns]
        missing = [c for c in CORE_FEATURES if c not in df.columns]
        if missing:
            print(f"Warning: missing core features (skipped): {missing}")

    print(f"Selected {len(numeric_cols)} features: {numeric_cols}")

    # ── Log-transform skewed volume features ────────────────────────────────
    # log1p handles zeros gracefully: log1p(0) = 0
    X_df = df[numeric_cols].copy()
    log_applied = []
    for col in numeric_cols:
        if col in LOG_TRANSFORM_FEATURES and col in X_df.columns:
            # Only log-transform non-negative columns
            col_min = X_df[col].min()
            if col_min >= 0:
                X_df[col] = np.log1p(X_df[col])
                log_applied.append(col)
            else:
                # Shift to non-negative, then log
                X_df[col] = np.log1p(X_df[col] - col_min)
                log_applied.append(f"{col}(shifted)")

    if log_applied:
        print(f"Log-transformed: {log_applied}")

    # Also log-transform trend_slope (can be negative, use sign-preserving log)
    if "trend_slope" in X_df.columns:
        X_df["trend_slope"] = np.sign(X_df["trend_slope"]) * np.log1p(np.abs(X_df["trend_slope"]))
        print("Sign-preserving log applied to trend_slope")

    X = X_df.values
    feature_names = list(numeric_cols)

    # Remove low-variance features
    variances = np.var(X, axis=0)
    high_var_mask = variances > 0.001
    X = X[:, high_var_mask]
    feature_names = [feature_names[i] for i in range(len(feature_names)) if high_var_mask[i]]
    print(f"Using {len(feature_names)} features after variance filtering: {feature_names}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional PCA
    pca = None
    if args.use_pca:
        n_components = args.pca_components or min(50, X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_scaled = pca.fit_transform(X_scaled)
        print(f"Applied PCA: {X_scaled.shape[1]} components ({pca.explained_variance_ratio_.sum():.2%} variance explained)")

    # Find optimal K
    k_range = tuple(args.k_range)
    k_results = find_optimal_k(X_scaled, k_range, args.min_cluster_size_pct, skip_gap=args.skip_gap)
    optimal_k = k_results["optimal_k"]

    print(f"\nOptimal K selection:")
    print(f"  Silhouette method: K={k_results['optimal_k_silhouette']}")
    print(f"  Elbow method: K={k_results['optimal_k_elbow']}")
    print(f"  Gap statistic: K={k_results['optimal_k_gap']}")
    print(f"  Selected: K={optimal_k}")

    # Train final model with more inits for stability
    print(f"\nTraining final KMeans with K={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    # ── Merge small clusters ────────────────────────────────────────────────
    n_samples = len(X_scaled)
    min_cluster_size = int(n_samples * args.min_cluster_size_pct)
    unique_pre, counts_pre = np.unique(labels, return_counts=True)
    small_count = sum(1 for c in counts_pre if c < min_cluster_size)

    if small_count > 0:
        labels, centroids_scaled = merge_small_clusters(
            X_scaled, labels, kmeans.cluster_centers_, min_cluster_size
        )
        final_k = len(set(labels))
        print(f"After merging: {final_k} clusters (was {optimal_k})")
        optimal_k = final_k
    else:
        centroids_scaled = kmeans.cluster_centers_
        print("All clusters meet minimum size — no merging needed.")

    # Compute metrics
    silhouette = silhouette_score(X_scaled, labels)
    inertia = kmeans.inertia_

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.astype(int), counts.astype(int)))

    print(f"\nFinal results:")
    print(f"  Clusters: {optimal_k}")
    print(f"  Silhouette score: {silhouette:.4f}")
    print(f"  Inertia: {inertia:.2f}")
    for cid, sz in sorted(cluster_sizes.items()):
        pct = sz / n_samples * 100
        print(f"  Cluster {cid}: {sz:,} DFUs ({pct:.1f}%)")

    # Create output directory
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("Generating visualizations...")
    plot_k_selection(
        k_results["k_values"],
        k_results["inertias"],
        k_results["silhouette_scores"],
        k_results["gap_stats"],
        output_dir
    )

    if X_scaled.shape[1] >= 2:
        plot_cluster_visualization(X_scaled, labels, feature_names, output_dir)

    # Save cluster assignments
    assignments_df = pd.DataFrame({
        "dfu_ck": df["dfu_ck"].values,
        "cluster_id": labels
    })
    assignments_path = output_dir / "cluster_assignments.csv"
    assignments_df.to_csv(assignments_path, index=False)
    print(f"Saved cluster assignments to {assignments_path}")

    # Save cluster metadata
    cluster_metadata = {
        "optimal_k": int(optimal_k),
        "silhouette_score": float(silhouette),
        "inertia": float(inertia),
        "n_clusters": int(optimal_k),
        "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
        "feature_names": feature_names,
        "log_transformed": log_applied,
        "feature_selection": "core" if not args.all_features else "all",
        "min_cluster_size_pct": args.min_cluster_size_pct,
        "k_selection_results": {
            "k_values": [int(k) for k in k_results["k_values"]],
            "inertias": [float(x) for x in k_results["inertias"]],
            "silhouette_scores": [float(x) for x in k_results["silhouette_scores"]],
            "gap_stats": [float(x) for x in k_results["gap_stats"]],
        }
    }

    metadata_path = output_dir / "cluster_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(cluster_metadata, f, indent=2)
    print(f"Saved cluster metadata to {metadata_path}")

    # Save centroids — inverse-transform back to original feature space
    if pca is None:
        centroids_original = scaler.inverse_transform(centroids_scaled)
    else:
        centroids_pca_inv = pca.inverse_transform(centroids_scaled)
        centroids_original = scaler.inverse_transform(centroids_pca_inv)

    # Note: centroids are in log-space for log-transformed features.
    # For interpretability in label_clusters.py, expm1 to get back to original scale.
    centroids_df = pd.DataFrame(centroids_original, columns=feature_names)
    for col in feature_names:
        if col in LOG_TRANSFORM_FEATURES:
            centroids_df[col] = np.expm1(centroids_df[col])
    # Undo sign-preserving log for trend_slope
    if "trend_slope" in centroids_df.columns:
        centroids_df["trend_slope"] = np.sign(centroids_df["trend_slope"]) * np.expm1(np.abs(centroids_df["trend_slope"]))

    centroids_df["cluster_id"] = range(optimal_k)
    centroids_path = output_dir / "cluster_centroids.csv"
    centroids_df.to_csv(centroids_path, index=False)
    print(f"Saved cluster centroids to {centroids_path}")

    # MLflow logging (optional: skip if server unavailable)
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5003")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("dfu_clustering")

        with mlflow.start_run():
            mlflow.set_tag("model_type", "clustering")
            mlflow.set_tag("feature_set", "core" if not args.all_features else "all")
            mlflow.set_tag("version", "v2.0")

            mlflow.log_params({
                "k": int(optimal_k),
                "k_min": int(k_range[0]),
                "k_max": int(k_range[1]),
                "min_cluster_size_pct": args.min_cluster_size_pct,
                "use_pca": args.use_pca,
                "n_features": len(feature_names),
                "log_transform": True,
                "feature_selection": "core" if not args.all_features else "all",
            })

            mlflow.log_metrics({
                "silhouette_score": float(silhouette),
                "inertia": float(inertia),
                "n_clusters": int(optimal_k),
            })

            for cluster_id, size in cluster_sizes.items():
                mlflow.log_metric(f"cluster_{cluster_id}_size", int(size))

            mlflow.log_artifact(str(assignments_path), "cluster_assignments.csv")
            mlflow.log_artifact(str(metadata_path), "cluster_metadata.json")
            mlflow.log_artifact(str(centroids_path), "cluster_centroids.csv")
            mlflow.log_artifact(str(output_dir / "k_selection_plots.png"), "k_selection_plots.png")
            if (output_dir / "cluster_visualization.png").exists():
                mlflow.log_artifact(str(output_dir / "cluster_visualization.png"), "cluster_visualization.png")

            mlflow.sklearn.log_model(kmeans, "model")

            print(f"\nLogged to MLflow: {mlflow.get_artifact_uri()}")
    except Exception as e:  # ConnectionError, MlflowException, etc.
        print(f"\nMLflow logging skipped (server unavailable or error): {e}")
        print("Clustering outputs were saved to disk. Start MLflow (e.g. make up) to enable experiment tracking.")


if __name__ == "__main__":
    main()
