"""
Clustering training library: feature constants, optimal-K search, and KMeans training.

Extracted from scripts/train_clustering_model.py for reuse across scenarios,
experiments, and the main training pipeline.

Key design choices for balanced clusters (tree-model friendly):
  1. Log-transform skewed volume features before scaling
  2. Select only core demand-pattern features (volume/trend/seasonality/intermittency)
  3. Enforce 5% minimum cluster size during K selection (hard constraint)
  4. Merge any remaining small clusters into nearest large neighbor post-hoc
  5. Combined score = 0.5 * silhouette_norm + 0.5 * calinski_norm for robust K selection
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score

logger = logging.getLogger(__name__)

# Feature constants live in `constants.py` (no heavy imports). Import-only
# consumers (e.g. API routers) should import directly from
# `common.ml.clustering.constants` to avoid loading matplotlib/sklearn.


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
    X_scaled : np.ndarray
        Scaled feature matrix (n_samples, n_features).
    k_range : tuple[int, int]
        (k_min, k_max) inclusive range of K values to test.
    min_cluster_size_pct : float
        Minimum cluster size as percentage of total samples.
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
