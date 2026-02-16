"""
Train clustering model on DFU features with optimal K selection.

This script performs KMeans clustering with multiple K selection methods
(elbow, silhouette, gap statistic) and logs results to MLflow.
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
        # Generate random data within bounds of actual data
        random_data = np.random.uniform(
            low=X.min(axis=0),
            high=X.max(axis=0),
            size=X.shape
        )
        ref_kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        ref_kmeans.fit(random_data)
        ref_inertias.append(ref_kmeans.inertia_)
    
    # Gap statistic
    gap = np.log(np.mean(ref_inertias)) - np.log(actual_inertia)
    return gap


def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: tuple[int, int],
    min_cluster_size_pct: float = 0.01
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
    
    print(f"Testing K values from {k_min} to {k_max}...")
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        
        # Gap statistic (only for k >= 2)
        if k >= 2:
            gap = gap_statistic(X_scaled, k, n_refs=5)
            gap_stats.append(gap)
        else:
            gap_stats.append(0.0)
        
        # Check cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes_list.append(dict(zip(unique, counts)))
        
        # Validate minimum cluster size
        if min(counts) < min_cluster_size:
            print(f"  K={k}: Warning - smallest cluster has {min(counts)} samples (< {min_cluster_size})")
    
    # Find optimal K by silhouette score
    optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
    
    # Find optimal K by gap statistic (first k where gap(k) >= gap(k+1) - s(k+1))
    if len(gap_stats) > 1:
        gaps = np.array(gap_stats)
        s = np.array([np.std([gap_statistic(X_scaled, k, n_refs=5) for _ in range(3)]) for k in k_values])
        gap_diff = gaps[:-1] - (gaps[1:] - s[1:])
        optimal_k_gap = k_values[np.argmax(gap_diff)] if len(gap_diff) > 0 else optimal_k_silhouette
    else:
        optimal_k_gap = optimal_k_silhouette
    
    # Elbow method: find point of maximum curvature
    if len(inertias) >= 3:
        # Calculate second derivative approximation
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        # Find maximum negative second derivative (sharpest drop)
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
    
    # Apply PCA to 2D
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
    parser.add_argument("--min-cluster-size-pct", type=float, default=0.01, help="Minimum cluster size as percentage")
    parser.add_argument("--use-pca", action="store_true", help="Use PCA for dimensionality reduction")
    parser.add_argument("--pca-components", type=int, default=None, help="Number of PCA components (auto if not specified)")
    parser.add_argument("--output-dir", type=str, default="data/clustering", help="Output directory for artifacts")
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
    
    # Separate features from metadata
    metadata_cols = ["dfu_ck", "dmdunit", "dmdgroup", "loc"]
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    # Select numeric features only
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].values
    
    # Remove low-variance features
    variances = np.var(X, axis=0)
    high_var_mask = variances > 0.01
    X = X[:, high_var_mask]
    feature_names = [numeric_cols[i] for i in range(len(numeric_cols)) if high_var_mask[i]]
    print(f"Using {len(feature_names)} features after variance filtering")
    
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
    k_results = find_optimal_k(X_scaled, k_range, args.min_cluster_size_pct)
    optimal_k = k_results["optimal_k"]
    
    print(f"\nOptimal K selection:")
    print(f"  Silhouette method: K={k_results['optimal_k_silhouette']}")
    print(f"  Elbow method: K={k_results['optimal_k_elbow']}")
    print(f"  Gap statistic: K={k_results['optimal_k_gap']}")
    print(f"  Selected: K={optimal_k}")
    
    # Train final model
    print(f"\nTraining KMeans with K={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    
    # Compute metrics
    silhouette = silhouette_score(X_scaled, labels)
    inertia = kmeans.inertia_
    
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.astype(int), counts.astype(int)))
    
    print(f"Silhouette score: {silhouette:.4f}")
    print(f"Inertia: {inertia:.2f}")
    print(f"Cluster sizes: {cluster_sizes}")
    
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
    
    # PCA visualization (if not using PCA, apply temporary PCA for visualization)
    if not args.use_pca and X_scaled.shape[1] >= 2:
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
    
    # Save centroids (if not using PCA, transform back to original space)
    if pca is None:
        centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    else:
        # Transform PCA centroids back to scaled space, then to original
        centroids_scaled = pca.inverse_transform(kmeans.cluster_centers_)
        centroids_original = scaler.inverse_transform(centroids_scaled)
    
    centroids_df = pd.DataFrame(centroids_original, columns=feature_names)
    centroids_df["cluster_id"] = range(optimal_k)
    centroids_path = output_dir / "cluster_centroids.csv"
    centroids_df.to_csv(centroids_path, index=False)
    print(f"Saved cluster centroids to {centroids_path}")
    
    # MLflow logging
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5003")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("dfu_clustering")
    
    with mlflow.start_run():
        mlflow.set_tag("model_type", "clustering")
        mlflow.set_tag("feature_set", "time_series_item_dfu")
        mlflow.set_tag("version", "v1.0")
        
        mlflow.log_params({
            "k": int(optimal_k),
            "k_min": int(k_range[0]),
            "k_max": int(k_range[1]),
            "min_cluster_size_pct": args.min_cluster_size_pct,
            "use_pca": args.use_pca,
            "n_features": len(feature_names),
        })
        
        mlflow.log_metrics({
            "silhouette_score": float(silhouette),
            "inertia": float(inertia),
            "n_clusters": int(optimal_k),
        })
        
        # Log cluster sizes as metrics
        for cluster_id, size in cluster_sizes.items():
            mlflow.log_metric(f"cluster_{cluster_id}_size", int(size))
        
        # Log artifacts
        mlflow.log_artifact(str(assignments_path), "cluster_assignments.csv")
        mlflow.log_artifact(str(metadata_path), "cluster_metadata.json")
        mlflow.log_artifact(str(centroids_path), "cluster_centroids.csv")
        mlflow.log_artifact(str(output_dir / "k_selection_plots.png"), "k_selection_plots.png")
        if (output_dir / "cluster_visualization.png").exists():
            mlflow.log_artifact(str(output_dir / "cluster_visualization.png"), "cluster_visualization.png")
        
        # Log model
        mlflow.sklearn.log_model(kmeans, "model")
        
        print(f"\nLogged to MLflow: {mlflow.get_artifact_uri()}")


if __name__ == "__main__":
    main()
