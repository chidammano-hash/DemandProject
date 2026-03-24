"""
Train clustering model on DFU features with optimal K selection.

This script performs KMeans clustering with combined Silhouette + Calinski-Harabasz
K selection and logs results to MLflow.

Key design choices for balanced clusters (tree-model friendly):
  1. Log-transform skewed volume features before scaling
  2. Select only core demand-pattern features (volume/trend/seasonality/intermittency)
  3. Enforce 5% minimum cluster size during K selection (hard constraint)
  4. Merge any remaining small clusters into nearest large neighbor post-hoc
  5. Combined score = 0.5 * silhouette_norm + 0.5 * calinski_norm for robust K selection
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)

# ── Feature groups ──────────────────────────────────────────────────────────
# Core features that drive business-meaningful clusters for tree models.
# Covers volume, trend, seasonality, periodicity, intermittency, and lifecycle.
CORE_FEATURES = [
    # Volume (log-transformed)
    "mean_demand",
    "cv_demand",
    "iqr_demand",
    # Trend (scale-invariant)
    "trend_slope_norm",
    "trend_r2",
    "cagr",
    # Seasonality
    "seasonal_amplitude",
    "seasonal_r2",
    "yoy_correlation",
    # Periodicity
    "periodicity_strength",
    # Intermittency
    "zero_demand_pct",
    "adi",
    # Lifecycle
    "months_available",
    "recency_ratio",
]

# Features that get log1p-transformed (highly skewed, spans orders of magnitude)
LOG_TRANSFORM_FEATURES = [
    "mean_demand",
    "median_demand",
    "std_demand",
    "total_demand",
    "max_demand",
    "iqr_demand",
    "adi",
]


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

    logger.info("Merging %d small clusters (min_size=%d):", len(small), min_size)
    for sc in sorted(small):
        # Find nearest large cluster by centroid distance
        sc_centroid = centroids[sc]
        best_dist = np.inf
        best_target = None
        for lc in sorted(large):
            d = np.linalg.norm(sc_centroid - centroids[lc])
            if d < best_dist:
                best_dist = d
                best_target = lc
        logger.info("  Cluster %d (%d DFUs) -> merged into cluster %s (dist=%.4f)",
                    sc, size_map[sc], best_target, best_dist)
        labels[labels == sc] = best_target

    # Re-number labels to be contiguous 0..K-1
    remaining = sorted(set(labels))
    remap = {old: new for new, old in enumerate(remaining)}
    labels = np.array([remap[lbl] for lbl in labels])

    # Recompute centroids from actual data
    new_k = len(remaining)
    new_centroids = np.zeros((new_k, X_scaled.shape[1]))
    for cid in range(new_k):
        mask = labels == cid
        new_centroids[cid] = X_scaled[mask].mean(axis=0)

    return labels, new_centroids


def _evaluate_single_k(
    k: int,
    X_scaled: np.ndarray,
    n_samples: int,
    min_cluster_size: int,
    silhouette_sample_size: int | None,
) -> dict[str, Any]:
    """Evaluate a single K value.  Designed for parallel execution.

    Returns a dict with k, inertia, sil, ch, cluster_sizes, smallest, feasible.
    """
    kmeans = MiniBatchKMeans(
        n_clusters=k, random_state=42, n_init=5, batch_size=10000, max_iter=200
    )
    labels = kmeans.fit_predict(X_scaled)

    # Use sampling for silhouette on large datasets (>50K) for speed
    sil_kwargs: dict[str, Any] = {}
    if silhouette_sample_size is not None:
        sil_kwargs["sample_size"] = silhouette_sample_size
        sil_kwargs["random_state"] = 42
    sil = silhouette_score(X_scaled, labels, **sil_kwargs)
    ch = calinski_harabasz_score(X_scaled, labels)

    unique, counts = np.unique(labels, return_counts=True)
    smallest = int(min(counts))
    feasible = smallest >= min_cluster_size

    return {
        "k": k,
        "inertia": float(kmeans.inertia_),
        "sil": float(sil),
        "ch": float(ch),
        "cluster_sizes": dict(zip(unique.tolist(), counts.tolist())),
        "smallest": smallest,
        "feasible": feasible,
    }


def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: tuple[int, int],
    min_cluster_size_pct: float = 5.0,
    n_workers: int = 1,
) -> dict[str, Any]:
    """Find optimal K using combined Silhouette + Calinski-Harabasz score.

    Enforces a hard minimum cluster size constraint: any K where any cluster
    falls below the threshold is penalized (score set to -1) so the optimizer
    steers toward well-balanced solutions.

    Parameters
    ----------
    n_workers : int
        Number of parallel workers for K evaluation.  1 = serial (default).

    Returns a dict with k_values, inertias, silhouette_scores, ch_scores,
    combined_scores, feasible_mask, and optimal_k variants.
    """
    k_min, k_max = k_range
    k_values = list(range(k_min, k_max + 1))

    n_samples = len(X_scaled)
    min_cluster_size = int(n_samples * min_cluster_size_pct / 100.0)

    # Use silhouette sampling for large datasets (>50K) — O(n^2) otherwise
    silhouette_sample_size = 10_000 if n_samples > 50_000 else None

    logger.info(
        "Testing K values from %d to %d (%d samples, min_cluster_size=%d [%.1f%%]%s, workers=%d)...",
        k_min, k_max, n_samples, min_cluster_size, min_cluster_size_pct,
        f", silhouette_sample={silhouette_sample_size}" if silhouette_sample_size else "",
        n_workers,
    )

    # Evaluate K values — parallel if workers > 1
    results_by_k: dict[int, dict] = {}

    if n_workers > 1 and len(k_values) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_single_k, k, X_scaled, n_samples,
                    min_cluster_size, silhouette_sample_size,
                ): k
                for k in k_values
            }
            for future in as_completed(futures):
                result = future.result()
                k = result["k"]
                results_by_k[k] = result
                smallest = result["smallest"]
                smallest_pct = smallest / n_samples * 100
                status = "OK" if result["feasible"] else f"PENALIZED (smallest={smallest} = {smallest_pct:.1f}%)"
                logger.info("  K=%d... sil=%.4f, CH=%.1f, smallest=%d (%.1f%%) [%s]",
                            k, result['sil'], result['ch'], smallest, smallest_pct, status)
    else:
        for k in k_values:
            result = _evaluate_single_k(
                k, X_scaled, n_samples, min_cluster_size, silhouette_sample_size,
            )
            results_by_k[k] = result
            smallest = result["smallest"]
            smallest_pct = smallest / n_samples * 100
            status = "OK" if result["feasible"] else f"PENALIZED (smallest={smallest} = {smallest_pct:.1f}%)"
            logger.info("  K=%d... sil=%.4f, CH=%.1f, smallest=%d (%.1f%%) [%s]",
                        k, result['sil'], result['ch'], smallest, smallest_pct, status)

    # Collect results in k_values order
    inertias = [results_by_k[k]["inertia"] for k in k_values]
    silhouette_scores_list = [results_by_k[k]["sil"] for k in k_values]
    ch_scores = [results_by_k[k]["ch"] for k in k_values]
    cluster_sizes_list = [results_by_k[k]["cluster_sizes"] for k in k_values]
    feasible_mask = [results_by_k[k]["feasible"] for k in k_values]

    # Normalize metrics to [0, 1] across all K values
    sil_arr = np.array(silhouette_scores_list)
    ch_arr = np.array(ch_scores)

    sil_norm = (sil_arr - sil_arr.min()) / (sil_arr.max() - sil_arr.min() + 1e-8)
    ch_norm = (ch_arr - ch_arr.min()) / (ch_arr.max() - ch_arr.min() + 1e-8)
    combined = 0.5 * sil_norm + 0.5 * ch_norm

    # Penalize K values that violate the min-cluster-size constraint
    for i, feasible in enumerate(feasible_mask):
        if not feasible:
            combined[i] = -1.0

    combined_scores = combined.tolist()

    # Best K: highest combined score among feasible K values
    if any(feasible_mask):
        optimal_k = k_values[int(np.argmax(combined))]
    else:
        # All K values violated the constraint — fall back to best silhouette
        logger.warning(
            "No K passed the min-cluster-size constraint. "
            "Falling back to best silhouette score."
        )
        optimal_k = k_values[int(np.argmax(sil_arr))]

    # Elbow method: point of maximum curvature (secondary reference)
    if len(inertias) >= 3:
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        optimal_k_elbow = k_values[int(np.argmin(second_diffs)) + 1] if len(second_diffs) > 0 else optimal_k
    else:
        optimal_k_elbow = optimal_k

    optimal_k_silhouette = k_values[int(np.argmax(sil_arr))]

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores_list,
        "ch_scores": ch_scores,
        "combined_scores": combined_scores,
        "feasible_mask": feasible_mask,
        "optimal_k": optimal_k,
        "optimal_k_silhouette": optimal_k_silhouette,
        "optimal_k_elbow": optimal_k_elbow,
        "cluster_sizes": cluster_sizes_list,
    }


def plot_k_selection(
    k_values: list[int],
    inertias: list[float],
    silhouette_scores: list[float],
    ch_scores: list[float],
    combined_scores: list[float],
    feasible_mask: list[bool],
    optimal_k: int,
    output_dir: Path,
) -> None:
    """Generate visualization plots for K selection (elbow, silhouette, CH score)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Elbow plot
    axes[0].plot(k_values, inertias, "bo-")
    axes[0].axvline(x=optimal_k, color="red", linestyle="--", alpha=0.7, label=f"K={optimal_k}")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Within-Cluster Sum of Squares (WCSS)")
    axes[0].set_title("Elbow Method")
    axes[0].legend()
    axes[0].grid(True)

    # Silhouette plot — color infeasible K values differently
    colors_sil = ["green" if f else "lightcoral" for f in feasible_mask]
    axes[1].bar(k_values, silhouette_scores, color=colors_sil, alpha=0.7)
    axes[1].plot(k_values, silhouette_scores, "o-", color="darkblue", alpha=0.8)
    axes[1].axvline(x=optimal_k, color="red", linestyle="--", alpha=0.7, label=f"K={optimal_k}")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score\n(red=infeasible <5%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    # Calinski-Harabasz score plot
    colors_ch = ["green" if f else "lightcoral" for f in feasible_mask]
    axes[2].bar(k_values, ch_scores, color=colors_ch, alpha=0.7)
    axes[2].plot(k_values, ch_scores, "o-", color="darkorange", alpha=0.8)
    axes[2].axvline(x=optimal_k, color="red", linestyle="--", alpha=0.7, label=f"K={optimal_k}")
    axes[2].set_xlabel("Number of Clusters (K)")
    axes[2].set_ylabel("Calinski-Harabasz Score")
    axes[2].set_title("Calinski-Harabasz Score\n(higher = better separation)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_dir / "k_selection_plots.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Additional combined-score plot
    fig2, ax = plt.subplots(figsize=(8, 5))
    colors_comb = ["green" if f else "lightcoral" for f in feasible_mask]
    combined_display = [max(s, 0) for s in combined_scores]  # clip -1 to 0 for display
    ax.bar(k_values, combined_display, color=colors_comb, alpha=0.7)
    ax.axvline(x=optimal_k, color="red", linestyle="--", alpha=0.9, label=f"Selected K={optimal_k}")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Combined Score (0.5*sil + 0.5*CH, normalized)")
    ax.set_title("Combined K Selection Score\n(green=feasible >=5%, red=penalized)")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / "combined_score_plot.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_cluster_visualization(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    """Generate 2D PCA visualization of clusters."""
    if X_scaled.shape[1] < 2:
        return

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab20", alpha=0.6)
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
    parser.add_argument("--k-range", type=int, nargs=2, default=None, metavar=("MIN", "MAX"), help="K range to test (default: from config)")
    parser.add_argument("--min-cluster-size-pct", type=float, default=None, help="Minimum cluster size as pct of total (default: from config)")
    parser.add_argument("--use-pca", action="store_true", help="Use PCA for dimensionality reduction")
    parser.add_argument("--pca-components", type=int, default=None, help="Number of PCA components (auto if not specified)")
    parser.add_argument("--output-dir", type=str, default="data/clustering", help="Output directory for artifacts")
    parser.add_argument("--all-features", action="store_true", help="Use all numeric features instead of core volume/trend/seasonality subset")
    parser.add_argument("--config", type=str, default="config/clustering_config.yaml", help="Clustering config YAML")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers for K-search (default: min(cpu_count, 8); 1 for serial)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
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

                mlflow.sklearn.log_model(kmeans, "model")

                logger.info("Logged to MLflow: %s", mlflow.get_artifact_uri())
        except (ConnectionError, OSError, mlflow.exceptions.MlflowException) as e:
            logger.warning("MLflow logging skipped (server unavailable or error): %s", e)
            logger.info("Clustering outputs were saved to disk. Start MLflow (e.g. make up) to enable experiment tracking.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
