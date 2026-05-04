"""
Train clustering model on DFU features with optimal K selection.

This script performs KMeans clustering with combined Silhouette + Calinski-Harabasz
K selection and logs results to MLflow.

Library functions (CORE_FEATURES, LOG_TRANSFORM_FEATURES, find_optimal_k,
merge_small_clusters, plot_k_selection, plot_cluster_visualization) live in
common.ml.clustering.training and are re-exported here for backward compatibility.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.services.perf_profiler import profiled_section

# Re-export library functions for backward compatibility
from common.ml.clustering.training import (  # noqa: F401
    CORE_FEATURES,
    LOG_TRANSFORM_FEATURES,
    find_optimal_k,
    merge_small_clusters,
    plot_k_selection,
    plot_cluster_visualization,
    _evaluate_single_k,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train clustering model with optimal K selection")
    parser.add_argument("--input", type=str, default="data/clustering_features.csv", help="Input feature matrix")
    parser.add_argument("--k-range", type=int, nargs=2, default=None, metavar=("MIN", "MAX"), help="K range to test (default: from config)")
    parser.add_argument("--min-cluster-size-pct", type=float, default=None, help="Minimum cluster size as pct of total (default: from config)")
    parser.add_argument("--use-pca", action="store_true", help="Use PCA for dimensionality reduction")
    parser.add_argument("--pca-components", type=int, default=None, help="Number of PCA components (auto if not specified)")
    parser.add_argument("--output-dir", type=str, default="data/clustering", help="Output directory for artifacts")
    parser.add_argument("--all-features", action="store_true", help="Use all numeric features instead of core volume/trend/seasonality subset")
    parser.add_argument("--config", type=str, default=None, help="Optional clustering config YAML (defaults resolved from DB)")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers for K-search (default: min(cpu_count, 8); 1 for serial)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")

    # Load config — CLI args override config values
    config_path = root / args.config
    cfg = {}
    if config_path.exists():
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f).get("clustering", {})

    k_range = tuple(args.k_range) if args.k_range is not None else tuple(cfg.get("k_range", [5, 18]))
    min_cluster_size_pct = args.min_cluster_size_pct if args.min_cluster_size_pct is not None else cfg.get("min_cluster_size_pct", 5.0)
    use_pca = args.use_pca or cfg.get("use_pca", False)
    pca_components = args.pca_components or cfg.get("pca_components", None)

    # Load feature matrix
    input_path = root / args.input
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    logger.info("Loading feature matrix from %s...", input_path)
    with profiled_section("load_feature_matrix"):
        df = pd.read_csv(input_path)
    logger.info("Loaded %d samples with %d features", len(df), len(df.columns))

    # ── Feature selection ───────────────────────────────────────────────────
    metadata_cols = ["sku_ck", "item_id", "customer_group", "loc"]

    if args.all_features:
        feature_cols = [c for c in df.columns if c not in metadata_cols]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    else:
        # Use only core demand-pattern features
        numeric_cols = [c for c in CORE_FEATURES if c in df.columns]
        missing = [c for c in CORE_FEATURES if c not in df.columns]
        if missing:
            logger.warning("Missing core features (skipped): %s", missing)

    logger.info("Selected %d features: %s", len(numeric_cols), numeric_cols)

    # ── Log-transform skewed volume features ────────────────────────────────
    # log1p handles zeros gracefully: log1p(0) = 0
    X_df = df[numeric_cols].copy()
    log_applied = []
    for col in numeric_cols:
        if col in LOG_TRANSFORM_FEATURES and col in X_df.columns:
            col_min = X_df[col].min()
            if col_min >= 0:
                X_df[col] = np.log1p(X_df[col])
                log_applied.append(col)
            else:
                # Shift to non-negative, then log
                X_df[col] = np.log1p(X_df[col] - col_min)
                log_applied.append(f"{col}(shifted)")

    if log_applied:
        logger.info("Log-transformed: %s", log_applied)

    # Sign-preserving log for trend_slope_norm (can be negative)
    if "trend_slope_norm" in X_df.columns:
        X_df["trend_slope_norm"] = (
            np.sign(X_df["trend_slope_norm"]) * np.log1p(np.abs(X_df["trend_slope_norm"]))
        )
        logger.info("Sign-preserving log applied to trend_slope_norm")

    X = X_df.values
    feature_names = list(numeric_cols)

    # Remove low-variance features
    variances = np.var(X, axis=0)
    high_var_mask = variances > 0.001
    X = X[:, high_var_mask]
    feature_names = [feature_names[i] for i in range(len(feature_names)) if high_var_mask[i]]
    logger.info("Using %d features after variance filtering: %s", len(feature_names), feature_names)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional PCA
    pca_model = None
    if use_pca:
        n_components = pca_components or min(50, X_scaled.shape[1])
        pca_model = PCA(n_components=n_components, random_state=42)
        X_scaled = pca_model.fit_transform(X_scaled)
        logger.info("Applied PCA: %d components (%.2f%% variance explained)",
                    X_scaled.shape[1], pca_model.explained_variance_ratio_.sum() * 100)

    # Determine worker count for parallel K-search
    n_workers = args.workers if args.workers is not None else min(os.cpu_count() or 1, 8)
    n_workers = max(1, n_workers)

    # Find optimal K using combined silhouette + CH score with min-size constraint
    with profiled_section("find_optimal_k"):
        k_results = find_optimal_k(X_scaled, k_range, min_cluster_size_pct, n_workers=n_workers)
    optimal_k = k_results["optimal_k"]

    logger.info("Optimal K selection:")
    logger.info("  Combined score method: K=%d", optimal_k)
    logger.info("  Silhouette-only method: K=%d", k_results['optimal_k_silhouette'])
    logger.info("  Elbow method: K=%d", k_results['optimal_k_elbow'])
    logger.info("  Selected: K=%d", optimal_k)

    # Train final model with n_init=30 for robustness with 14 features
    # algorithm="elkan" is faster than "lloyd" for dense low-dimensional data
    logger.info("Training final KMeans with K=%d, n_init=30, algorithm=elkan, max_iter=300...", optimal_k)
    with profiled_section("train_final_kmeans"):
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=30, algorithm="elkan", max_iter=300)
        labels = kmeans.fit_predict(X_scaled)

    # ── Merge small clusters ────────────────────────────────────────────────
    n_samples = len(X_scaled)
    min_cluster_size = int(n_samples * min_cluster_size_pct / 100.0)
    unique_pre, counts_pre = np.unique(labels, return_counts=True)
    small_count = sum(1 for c in counts_pre if c < min_cluster_size)

    with profiled_section("merge_small_clusters"):
        if small_count > 0:
            labels, centroids_scaled = merge_small_clusters(
                X_scaled, labels, kmeans.cluster_centers_, min_cluster_size
            )
            final_k = len(set(labels))
            logger.info("After merging: %d clusters (was %d)", final_k, optimal_k)
            optimal_k = final_k
        else:
            centroids_scaled = kmeans.cluster_centers_
            logger.info("All clusters meet minimum size -- no merging needed.")

    # Compute final metrics
    silhouette = silhouette_score(X_scaled, labels)
    ch_score_final = calinski_harabasz_score(X_scaled, labels)
    inertia = kmeans.inertia_

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.astype(int), counts.astype(int)))

    logger.info("Final results:")
    logger.info("  Clusters: %d", optimal_k)
    logger.info("  Silhouette score: %.4f", silhouette)
    logger.info("  Calinski-Harabasz score: %.2f", ch_score_final)
    logger.info("  Inertia: %.2f", inertia)
    for cid, sz in sorted(cluster_sizes.items()):
        pct = sz / n_samples * 100
        logger.info("  Cluster %s: %s DFUs (%.1f%%)", cid, f"{sz:,}", pct)

    # Create output directory
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    logger.info("Generating visualizations...")
    with profiled_section("generate_visualizations"):
        plot_k_selection(
            k_results["k_values"],
            k_results["inertias"],
            k_results["silhouette_scores"],
            k_results["ch_scores"],
            k_results["combined_scores"],
            k_results["feasible_mask"],
            optimal_k,
            output_dir,
        )

        if X_scaled.shape[1] >= 2:
            plot_cluster_visualization(X_scaled, labels, feature_names, output_dir)

    # Save cluster assignments
    with profiled_section("save_artifacts"):
        assignments_df = pd.DataFrame({
            "sku_ck": df["sku_ck"].values,
            "cluster_id": labels
        })
        assignments_path = output_dir / "cluster_assignments.csv"
        assignments_df.to_csv(assignments_path, index=False)
        logger.info("Saved cluster assignments to %s", assignments_path)

    # Save cluster metadata
    cluster_metadata = {
        "optimal_k": int(optimal_k),
        "silhouette_score": float(silhouette),
        "calinski_harabasz_score": float(ch_score_final),
        "inertia": float(inertia),
        "n_clusters": int(optimal_k),
        "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
        "feature_names": feature_names,
        "log_transformed": log_applied,
        "feature_selection": "core" if not args.all_features else "all",
        "min_cluster_size_pct": min_cluster_size_pct,
        "k_selection_results": {
            "k_values": [int(k) for k in k_results["k_values"]],
            "inertias": [float(x) for x in k_results["inertias"]],
            "silhouette_scores": [float(x) for x in k_results["silhouette_scores"]],
            "ch_scores": [float(x) for x in k_results["ch_scores"]],
            "combined_scores": [float(x) for x in k_results["combined_scores"]],
            "feasible_mask": k_results["feasible_mask"],
        },
    }

    metadata_path = output_dir / "cluster_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(cluster_metadata, f, indent=2)
    logger.info("Saved cluster metadata to %s", metadata_path)

    # Save centroids -- inverse-transform back to original feature space
    if pca_model is None:
        centroids_original = scaler.inverse_transform(centroids_scaled)
    else:
        centroids_pca_inv = pca_model.inverse_transform(centroids_scaled)
        centroids_original = scaler.inverse_transform(centroids_pca_inv)

    # Note: centroids are in log-space for log-transformed features.
    # expm1 to get back to original scale for interpretability in label_clusters.py.
    centroids_df = pd.DataFrame(centroids_original, columns=feature_names)
    for col in feature_names:
        if col in LOG_TRANSFORM_FEATURES:
            centroids_df[col] = np.expm1(centroids_df[col])
    # Undo sign-preserving log for trend_slope_norm
    if "trend_slope_norm" in centroids_df.columns:
        centroids_df["trend_slope_norm"] = (
            np.sign(centroids_df["trend_slope_norm"])
            * np.expm1(np.abs(centroids_df["trend_slope_norm"]))
        )

    centroids_df["cluster_id"] = range(optimal_k)
    centroids_path = output_dir / "cluster_centroids.csv"
    centroids_df.to_csv(centroids_path, index=False)
    logger.info("Saved cluster centroids to %s", centroids_path)

    # MLflow logging (optional: skip if server unavailable)
    with profiled_section("mlflow_logging"):
        try:
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5003")
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("dfu_clustering")

            with mlflow.start_run():
                mlflow.set_tag("model_type", "clustering")
                mlflow.set_tag("feature_set", "core" if not args.all_features else "all")
                mlflow.set_tag("version", "v3.0")
                mlflow.set_tag("k_selection_method", "combined_silhouette_ch")

                mlflow.log_params({
                    "k": int(optimal_k),
                    "k_min": int(k_range[0]),
                    "k_max": int(k_range[1]),
                    "min_cluster_size_pct": min_cluster_size_pct,
                    "use_pca": use_pca,
                    "n_features": len(feature_names),
                    "log_transform": True,
                    "feature_selection": "core" if not args.all_features else "all",
                    "n_init_final": 30,
                })

                mlflow.log_metrics({
                    "silhouette_score": float(silhouette),
                    "calinski_harabasz_score": float(ch_score_final),
                    "inertia": float(inertia),
                    "n_clusters": int(optimal_k),
                })

                for cluster_id, size in cluster_sizes.items():
                    mlflow.log_metric(f"cluster_{cluster_id}_size", int(size))
                    mlflow.log_metric(f"cluster_{cluster_id}_pct", float(size / n_samples * 100))

                mlflow.log_artifact(str(assignments_path), "cluster_assignments.csv")
                mlflow.log_artifact(str(metadata_path), "cluster_metadata.json")
                mlflow.log_artifact(str(centroids_path), "cluster_centroids.csv")
                mlflow.log_artifact(str(output_dir / "k_selection_plots.png"), "k_selection_plots.png")
                if (output_dir / "combined_score_plot.png").exists():
                    mlflow.log_artifact(str(output_dir / "combined_score_plot.png"), "combined_score_plot.png")
                if (output_dir / "cluster_visualization.png").exists():
                    mlflow.log_artifact(str(output_dir / "cluster_visualization.png"), "cluster_visualization.png")

                mlflow.sklearn.log_model(kmeans, name="model")

                logger.info("Logged to MLflow: %s", mlflow.get_artifact_uri())
        except (ConnectionError, OSError, mlflow.exceptions.MlflowException) as e:
            logger.warning("MLflow logging skipped (server unavailable or error): %s", e)
            logger.info("Clustering outputs were saved to disk. Start MLflow (e.g. make up) to enable experiment tracking.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
