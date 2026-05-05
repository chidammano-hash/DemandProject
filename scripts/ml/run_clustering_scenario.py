"""
Run a clustering scenario with custom parameters in a temp directory.

Orchestrates the clustering pipeline (feature gen, training, labeling)
without modifying production data. Returns structured results for the
What-If Scenarios UI.
"""

import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params
from common.core.planning_date import get_planning_date
from common.services.perf_profiler import profiled_section
from common.ml.clustering.features import compute_time_series_features
from common.ml.clustering.training import (
    CORE_FEATURES,
    LOG_TRANSFORM_FEATURES,
    find_optimal_k,
    merge_small_clusters,
)
from common.ml.clustering.labeling import assign_cluster_labels

# Import reusable infrastructure from common.ml.clustering.scenario
# and re-export at module level for backward compatibility — other modules
# (job_state, clusters router, cluster_experiments router, run_cluster_pipeline)
# import these names from scripts.ml.run_clustering_scenario.
from common.ml.clustering.scenario import (  # noqa: F401
    SCENARIO_BASE,
    _SCENARIO_ID_RE,
    _safe_scenario_dir,
    generate_scenario_id,
    _load_config_defaults,
    get_scenario_result,
    promote_scenario,
)


def _update_experiment_completed(
    experiment_id: int,
    result: dict[str, Any],
    runtime_seconds: float,
    artifacts_path: str,
) -> None:
    """Write completed results to the cluster_experiment table."""
    import psycopg

    db = get_db_params()
    try:
        with psycopg.connect(**db) as conn:
            conn.execute(
                """
                UPDATE cluster_experiment
                SET status = 'completed',
                    completed_at = NOW(),
                    runtime_seconds = %s,
                    optimal_k = %s,
                    silhouette_score = %s,
                    inertia = %s,
                    total_dfus = %s,
                    n_clusters = %s,
                    cluster_sizes = %s::jsonb,
                    profiles = %s::jsonb,
                    k_selection_results = %s::jsonb,
                    artifacts_path = %s
                WHERE experiment_id = %s
                """,
                (
                    runtime_seconds,
                    result.get("optimal_k"),
                    result.get("silhouette_score"),
                    result.get("inertia"),
                    result.get("total_dfus"),
                    result.get("n_clusters"),
                    json.dumps(result.get("cluster_sizes", {})),
                    json.dumps(result.get("profiles", [])),
                    json.dumps({
                        **result.get("k_selection_results", {}),
                        **({"pca_scatter": result["pca_scatter"]} if result.get("pca_scatter") else {}),
                    }),
                    artifacts_path,
                    experiment_id,
                ),
            )
    except psycopg.Error:
        import logging
        logging.getLogger(__name__).exception(
            "Failed to update cluster_experiment %d to completed", experiment_id
        )


def _update_experiment_failed(experiment_id: int) -> None:
    """Mark a cluster_experiment as failed."""
    import psycopg

    db = get_db_params()
    try:
        with psycopg.connect(**db) as conn:
            conn.execute(
                "UPDATE cluster_experiment SET status = 'failed', completed_at = NOW() "
                "WHERE experiment_id = %s",
                (experiment_id,),
            )
    except psycopg.Error:
        import logging
        logging.getLogger(__name__).exception(
            "Failed to update cluster_experiment %d to failed", experiment_id
        )


def run_scenario(
    feature_params: dict[str, Any] | None = None,
    model_params: dict[str, Any] | None = None,
    label_params: dict[str, Any] | None = None,
    scenario_id: str | None = None,
    relabel_only: bool = False,
    previous_scenario_id: str | None = None,
    experiment_id: int | None = None,
) -> dict[str, Any]:
    """Run a complete clustering scenario.

    Parameters
    ----------
    feature_params : dict with time_window_months, min_months_history
    model_params : dict with k_range, min_cluster_size_pct, use_pca, etc.
    label_params : dict with volume_high/low, cv_steady/volatile, etc.
    scenario_id : optional ID (auto-generated if not provided)
    relabel_only : skip feature gen and training, just re-label
    previous_scenario_id : scenario to relabel from (for relabel_only)
    experiment_id : optional cluster_experiment PK — when provided, writes
        results to the cluster_experiment table on completion or failure

    Returns
    -------
    dict with scenario_id, status, runtime_seconds, params, result
    """
    load_dotenv(ROOT / ".env")

    if scenario_id is None:
        scenario_id = generate_scenario_id()

    # Load defaults from promoted experiment or hardcoded fallbacks
    cfg = _load_config_defaults()
    labeling_cfg = cfg.get("labeling", {})
    vol_cfg = labeling_cfg.get("volume_thresholds", {})
    cv_cfg = labeling_cfg.get("cv_thresholds", {})

    # Merge with config-driven defaults (user params override)
    fp = {
        "time_window_months": cfg.get("time_window_months", 36),
        "min_months_history": cfg.get("min_months_history", 12),
    }
    if feature_params:
        fp.update(feature_params)

    mp = {
        "k_range": cfg.get("k_range", [5, 18]),
        "min_cluster_size_pct": cfg.get("min_cluster_size_pct", 5.0),
        "use_pca": cfg.get("use_pca", False),
        "pca_components": cfg.get("pca_components", None),
        "all_features": False,
    }
    if model_params:
        mp.update(model_params)

    lp = {
        "volume_high": vol_cfg.get("high", 0.75),
        "volume_low": vol_cfg.get("low", 0.25),
        "cv_steady": cv_cfg.get("steady", 0.4),
        "cv_volatile": cv_cfg.get("volatile", 0.8),
        "seasonality_threshold": labeling_cfg.get("seasonality_threshold", 0.3),
        "zero_demand_threshold": labeling_cfg.get("zero_demand_threshold", 0.15),
    }
    if label_params:
        lp.update(label_params)

    merged_params = {
        "feature_params": fp,
        "model_params": mp,
        "label_params": lp,
    }

    # Create scenario directory (validates scenario_id format + path containment)
    scenario_dir = _safe_scenario_dir(scenario_id)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        if relabel_only and previous_scenario_id:
            result = _run_relabel_only(previous_scenario_id, lp, scenario_dir)
        else:
            result = _run_full_pipeline(fp, mp, lp, scenario_dir)

        runtime = time.time() - start_time

        output = {
            "scenario_id": scenario_id,
            "status": "completed",
            "runtime_seconds": round(runtime, 1),
            "params": merged_params,
            "result": result,
        }

        # Save result to disk for later retrieval
        with open(scenario_dir / "scenario_result.json", "w") as f:
            json.dump(output, f, indent=2, default=str)

        # Write results to cluster_experiment table when experiment_id is provided
        if experiment_id is not None:
            _update_experiment_completed(
                experiment_id=experiment_id,
                result=result,
                runtime_seconds=round(runtime, 1),
                artifacts_path=str(scenario_dir),
            )

        return output

    except Exception as e:
        runtime = time.time() - start_time

        # Mark cluster_experiment as failed when experiment_id is provided
        if experiment_id is not None:
            _update_experiment_failed(experiment_id)

        output = {
            "scenario_id": scenario_id,
            "status": "failed",
            "runtime_seconds": round(runtime, 1),
            "params": merged_params,
            "result": None,
            "error": str(e),
        }

        # Save failed result to disk so status endpoint can return the error
        try:
            with open(scenario_dir / "scenario_result.json", "w") as f:
                json.dump(output, f, indent=2, default=str)
        except Exception:
            pass  # best-effort

        return output


MAX_DFUS_FOR_TRAINING = 20_000  # Sample if DFU count exceeds this

# Default time window used by the unified SKU features pipeline.  When the
# scenario's ``time_window_months`` matches this value we can safely reuse
# pre-computed features stored in ``dim_sku``.
_DEFAULT_FEATURE_TIME_WINDOW = 36

# Column mapping: dim_sku column name → feature name produced by
# ``compute_time_series_features()``.  Columns whose dim_sku name matches
# the feature name exactly are *not* listed here.
_DIM_SKU_COL_TO_FEATURE: dict[str, str] = {
    "demand_mean": "mean_demand",
    "demand_std": "std_demand",
    "demand_cv": "cv_demand",
    "demand_mad": "demand_mad",
    "demand_p50": "median_demand",
    "demand_p90": "demand_p90",
    "demand_skewness": "demand_skewness",
    "demand_kurtosis": "demand_kurtosis",
    "intermittency_ratio": "zero_demand_pct",
    "total_demand_months": "months_available",
}

# dim_sku columns that already use the exact same name as the feature.
# These come from sql/015 (seasonality) and sql/120 (unified features).
_DIM_SKU_SAME_NAME_COLS: list[str] = [
    "seasonality_strength",
    "peak_month",
    "trough_month",
    "peak_trough_ratio",
    "iqr_demand",
    "min_demand",
    "max_demand",
    "total_demand",
    "trend_slope",
    "trend_slope_norm",
    "trend_r2",
    "trend_pct_change",
    "trend_direction",
    "seasonal_amplitude",
    "seasonal_r2",
    "yoy_correlation",
    "seasonal_index_std",
    "periodicity_strength",
    "adi",
    "cagr",
    "recency_ratio",
    "acceleration",
    "outlier_count",
    "acf_lag12",
]


def _build_thresholds(
    lp: dict, mean_demands: np.ndarray
) -> tuple[dict, dict, dict]:
    """Build volume_thresholds, cv_thresholds, and labeling_config from label params.

    Args:
        lp: Merged label params dict (from config + user overrides).
        mean_demands: 1-D array of centroid mean_demand values used to derive
                      percentile-based absolute volume thresholds.

    Returns:
        (volume_thresholds, cv_thresholds, labeling_config)
    """
    if len(mean_demands) == 0:
        raise ValueError("Cannot build volume thresholds: mean_demands array is empty")
    if "mean_demand" not in lp and len(mean_demands) == 0:
        pass  # validated above

    vol_very_high_pctl = lp.get("volume_very_high", 0.90)
    vol_high_pctl = lp.get("volume_high", 0.75)
    vol_low_pctl = lp.get("volume_low", 0.25)
    vol_very_low_pctl = lp.get("volume_very_low", 0.10)

    volume_thresholds = {
        "very_high": float(np.percentile(mean_demands, vol_very_high_pctl * 100)),
        "high":      float(np.percentile(mean_demands, vol_high_pctl * 100)),
        "low":       float(np.percentile(mean_demands, vol_low_pctl * 100)),
        "very_low":  float(np.percentile(mean_demands, vol_very_low_pctl * 100)),
    }
    cv_thresholds = {
        "very_steady":  lp.get("cv_very_steady", 0.2),
        "steady":       lp.get("cv_steady", 0.4),
        "volatile":     lp.get("cv_volatile", 0.8),
        "very_volatile": lp.get("cv_very_volatile", 1.2),
    }
    labeling_config = {
        "seasonality_threshold":   lp.get("seasonality_threshold", 0.3),
        "seasonality_r2_threshold": lp.get("seasonality_r2_threshold", 0.25),
        "periodicity_threshold":   lp.get("periodicity_threshold", 0.25),
        "zero_demand_threshold":   lp.get("zero_demand_threshold", 0.15),
        "adi_threshold":           lp.get("adi_threshold", 1.5),
        "trend_r2_threshold":      lp.get("trend_r2_threshold", 0.25),
        "cagr_growing":            lp.get("cagr_growing", 5.0),
        "cagr_declining":          lp.get("cagr_declining", -5.0),
        "recency_ratio_high":      lp.get("recency_ratio_high", 1.2),
        "recency_ratio_low":       lp.get("recency_ratio_low", 0.8),
    }
    return volume_thresholds, cv_thresholds, labeling_config


def _try_load_precomputed_features(
    db: dict,
    fp: dict,
    scenario_dir: Path,
) -> pd.DataFrame | None:
    """Attempt to load pre-computed features from ``dim_sku``.

    Returns a DataFrame identical in shape to what ``compute_time_series_features``
    would produce (with ``sku_ck`` column), or ``None`` if the fast path cannot
    be used.

    The fast path is skipped when:
    - ``features_computed_ts`` is NULL for every SKU (pipeline hasn't run yet)
    - The user passed a custom ``time_window_months`` that differs from the
      default used by the unified SKU features pipeline (what-if experiments)
    - The user passed a custom ``min_months_history`` that differs from the
      default (12), since the stored features may include SKUs we should skip
    """
    import psycopg

    _log = logging.getLogger(__name__)

    # Guard: if the user customized feature_params, fall back to raw computation
    time_window = fp.get("time_window_months", _DEFAULT_FEATURE_TIME_WINDOW)
    if isinstance(time_window, str) and time_window.lower() == "all":
        _log.info("Custom time_window_months='all' — skipping pre-computed feature fast path")
        return None
    if int(time_window) != _DEFAULT_FEATURE_TIME_WINDOW:
        _log.info(
            "Custom time_window_months=%s (default=%s) — skipping pre-computed feature fast path",
            time_window,
            _DEFAULT_FEATURE_TIME_WINDOW,
        )
        return None

    min_months = fp.get("min_months_history", 1)
    if int(min_months) != 1:
        _log.info(
            "Custom min_months_history=%s (default=1) — skipping pre-computed feature fast path",
            min_months,
        )
        return None

    # Build the SELECT: aliased columns that need renaming + same-name columns
    alias_parts = [f"{col} AS {feat}" for col, feat in _DIM_SKU_COL_TO_FEATURE.items()]
    same_parts = list(_DIM_SKU_SAME_NAME_COLS)
    select_cols = ", ".join(["sku_ck", *alias_parts, *same_parts])

    query = (
        f"SELECT {select_cols} "
        "FROM dim_sku "
        "WHERE features_computed_ts IS NOT NULL"
    )

    try:
        with psycopg.connect(**db) as conn:
            df = pd.read_sql(query, conn)
    except psycopg.Error:
        _log.warning("Failed to query dim_sku for pre-computed features — falling back", exc_info=True)
        return None

    if df.empty:
        _log.info("No SKUs with features_computed_ts set — falling back to raw computation")
        return None

    # Filter by min_months_history (months_available comes from total_demand_months)
    if "months_available" in df.columns:
        before = len(df)
        df = df[df["months_available"] >= int(min_months)].reset_index(drop=True)
        skipped = before - len(df)
        if skipped > 0:
            _log.info("Filtered out %d SKUs with < %s months history", skipped, min_months)

    if df.empty:
        _log.info("All pre-computed SKUs filtered out by min_months_history — falling back")
        return None

    # Add derived features that compute_time_series_features produces but
    # dim_sku does not store (backward-compat aliases and derived columns)
    if "zero_demand_pct" in df.columns:
        df["sparsity_score"] = df["zero_demand_pct"]
    if "cv_demand" in df.columns:
        df["demand_stability"] = 1.0 / (1.0 + df["cv_demand"])
    if "cagr" in df.columns:
        df["growth_rate"] = df["cagr"]
    if "recency_ratio" in df.columns:
        df["recent_vs_historical"] = df["recency_ratio"]
    if "yoy_correlation" in df.columns:
        df["year_over_year_correlation"] = df["yoy_correlation"]

    # Fill NaN with 0 (matches the fillna(0) in the slow path's feature prep)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    _log.info("Reading pre-computed features from dim_sku (%d SKUs)", len(df))

    # Persist the CSV artifact so downstream relabel-only scenarios work
    df.to_csv(scenario_dir / "clustering_features.csv", index=False)

    return df


def _run_full_pipeline(
    fp: dict, mp: dict, lp: dict, scenario_dir: Path
) -> dict[str, Any]:
    """Run the full clustering pipeline: features → training → labeling."""
    import psycopg
    from datetime import timedelta
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score as sk_silhouette

    db = get_db_params()

    # ── Fast path: try pre-computed features from dim_sku ────────────────────
    feature_df = _try_load_precomputed_features(db, fp, scenario_dir)

    if feature_df is not None:
        # Fast path succeeded — skip Steps 1 & 2
        pass
    else:
        # ── Slow path: compute features from raw fact_sales_monthly ──────────
        # Step 1: Load sales data
        with profiled_section("load_sales_data"):
            time_window = fp["time_window_months"]
            if isinstance(time_window, str) and time_window.lower() == "all":
                cutoff_date = None
            else:
                cutoff_date = get_planning_date() - timedelta(days=int(time_window) * 30)

            with psycopg.connect(**db) as conn:
                sales_query = """
                    SELECT d.sku_ck, s.startdate, s.qty
                    FROM fact_sales_monthly s
                    INNER JOIN dim_sku d
                        ON d.item_id = s.item_id AND d.customer_group = s.customer_group AND d.loc = s.loc
                    WHERE s.qty IS NOT NULL
                """
                params: dict[str, object] = {}
                if cutoff_date:
                    sales_query += " AND s.startdate >= %(cutoff)s"
                    params["cutoff"] = cutoff_date
                # Cap at planning date — exclude any data beyond current planning horizon
                planning_upper = get_planning_date().replace(day=1)
                sales_query += " AND s.startdate <= %(planning_upper)s"
                params["planning_upper"] = planning_upper
                sales_query += " ORDER BY d.sku_ck, s.startdate"

                sales_df = pd.read_sql(sales_query, conn, params=params if params else None)
            sales_df["startdate"] = pd.to_datetime(sales_df["startdate"])

        # Step 2: Compute time series features
        with profiled_section("compute_ts_features"):
            logging.getLogger(__name__).info(
                "Computing features from raw sales (fallback)"
            )
            grouped = sales_df.groupby("sku_ck", sort=False)
            ts_features_list = []
            for sku_ck, dfu_sales in grouped:
                if len(dfu_sales) < fp["min_months_history"]:
                    continue
                ts = compute_time_series_features(dfu_sales)
                ts["sku_ck"] = sku_ck
                ts_features_list.append(ts)

            if not ts_features_list:
                raise ValueError("No DFUs met minimum history requirement")

            feature_df = pd.DataFrame(ts_features_list)
            feature_df.to_csv(scenario_dir / "clustering_features.csv", index=False)

    # Step 3: Prepare features for clustering
    with profiled_section("prepare_features"):
        if mp["all_features"]:
            metadata_cols = ["sku_ck"]
            numeric_cols = [c for c in feature_df.select_dtypes(include=[np.number]).columns
                            if c not in metadata_cols]
        else:
            numeric_cols = [c for c in CORE_FEATURES if c in feature_df.columns]

        X_df = feature_df[numeric_cols].copy().fillna(0)

        # Log-transform skewed features
        for col in numeric_cols:
            if col in LOG_TRANSFORM_FEATURES and col in X_df.columns:
                col_min = X_df[col].min()
                if col_min >= 0:
                    X_df[col] = np.log1p(X_df[col])
                else:
                    X_df[col] = np.log1p(X_df[col] - col_min)
        if "trend_slope_norm" in X_df.columns:
            X_df["trend_slope_norm"] = np.sign(X_df["trend_slope_norm"]) * np.log1p(np.abs(X_df["trend_slope_norm"]))

        X = X_df.values
        feature_names = list(numeric_cols)

        # Remove low-variance features
        variances = np.var(X, axis=0)
        high_var_mask = variances > 0.001
        X = X[:, high_var_mask]
        feature_names = [feature_names[i] for i in range(len(feature_names)) if high_var_mask[i]]

        # Sanitize inf/NaN before scaling (prevents KMeans overflow warnings)
        X = np.where(np.isinf(X), np.nan, X)
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        X = np.where(np.isnan(X), col_medians, X)

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Optional PCA
        pca = None
        if mp["use_pca"]:
            n_comp = mp["pca_components"] or min(50, X_scaled.shape[1])
            pca = PCA(n_components=n_comp, random_state=42)
            X_scaled = pca.fit_transform(X_scaled)

        n_total = len(X_scaled)
        sampled = n_total > MAX_DFUS_FOR_TRAINING

        # Sample for training if dataset is large
        if sampled:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(n_total, MAX_DFUS_FOR_TRAINING, replace=False)
            X_train = X_scaled[sample_idx]
        else:
            X_train = X_scaled

    # Step 4: Find optimal K (on sample if large)
    with profiled_section("find_optimal_k_and_train"):
        k_range = tuple(mp["k_range"])
        k_results = find_optimal_k(
            X_train, k_range, mp["min_cluster_size_pct"]
        )
        optimal_k = k_results["optimal_k"]

        # Train final model on sample, then predict all
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans.fit(X_train)
        labels = kmeans.predict(X_scaled)  # Assign ALL DFUs

        # Merge small clusters
        n_samples = n_total
        min_cluster_size = int(n_samples * mp["min_cluster_size_pct"] / 100)
        unique_pre, counts_pre = np.unique(labels, return_counts=True)
        if any(c < min_cluster_size for c in counts_pre):
            labels, centroids_scaled = merge_small_clusters(
                X_scaled, labels, kmeans.cluster_centers_, min_cluster_size
            )
            optimal_k = len(set(labels))
        else:
            centroids_scaled = kmeans.cluster_centers_

        # Metrics (use sample for silhouette to avoid O(n^2) on full dataset)
        if sampled:
            sil_sample_idx = np.random.RandomState(42).choice(n_total, min(10000, n_total), replace=False)
            silhouette = float(sk_silhouette(X_scaled[sil_sample_idx], labels[sil_sample_idx]))
        else:
            silhouette = float(sk_silhouette(X_scaled, labels))
        inertia = float(kmeans.inertia_)

    # Inverse transform centroids
    if pca is None:
        centroids_original = scaler.inverse_transform(centroids_scaled)
    else:
        centroids_original = scaler.inverse_transform(pca.inverse_transform(centroids_scaled))

    centroids_df = pd.DataFrame(centroids_original, columns=feature_names)
    for col in feature_names:
        if col in LOG_TRANSFORM_FEATURES:
            centroids_df[col] = np.expm1(centroids_df[col])
    if "trend_slope_norm" in centroids_df.columns:
        centroids_df["trend_slope_norm"] = np.sign(centroids_df["trend_slope_norm"]) * np.expm1(np.abs(centroids_df["trend_slope_norm"]))

    centroids_df["cluster_id"] = range(optimal_k)

    # Step 5: Label clusters
    with profiled_section("label_clusters_and_save"):
        if "mean_demand" not in centroids_df.columns:
            raise ValueError("Centroids CSV missing required column 'mean_demand'")
        volume_thresholds, cv_thresholds, labeling_config = _build_thresholds(
            lp, centroids_df["mean_demand"].values
        )

        cluster_labels = assign_cluster_labels(
            centroids_df, volume_thresholds, cv_thresholds, labeling_config
        )

        # Save artifacts
        assignments_df = pd.DataFrame({"sku_ck": feature_df["sku_ck"].values[:n_samples], "cluster_id": labels})
        assignments_df["cluster_label"] = assignments_df["cluster_id"].map(cluster_labels)
        assignments_df.to_csv(scenario_dir / "cluster_labels.csv", index=False)
        centroids_df.to_csv(scenario_dir / "cluster_centroids.csv", index=False)

    # Build profiles
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {str(int(k)): int(v) for k, v in zip(unique_labels, counts)}
    total_dfus = n_samples

    profiles = []
    for _, row in centroids_df.iterrows():
        cid = int(row["cluster_id"])
        count = int(counts[list(unique_labels).index(cid)]) if cid in unique_labels else 0
        profiles.append({
            "cluster_id": cid,
            "label": cluster_labels.get(cid, f"cluster_{cid}"),
            "count": count,
            "pct_of_total": round(count / total_dfus * 100, 2) if total_dfus > 0 else 0,
            "mean_demand": round(float(row.get("mean_demand", 0)), 2),
            "cv_demand": round(float(row.get("cv_demand", 0)), 4),
            "seasonality_strength": round(float(row.get("seasonality_strength", 0)), 4),
            "trend_slope": round(float(row.get("trend_slope_norm", 0)), 4),
            "growth_rate": round(float(row.get("cagr", 0)), 4),
            "zero_demand_pct": round(float(row.get("zero_demand_pct", 0)), 4),
        })

    # Feature importance (variance ratio)
    feature_importance = []
    if len(feature_names) > 0:
        total_var = np.var(X_scaled, axis=0).sum()
        for i, fname in enumerate(feature_names[:X_scaled.shape[1]]):
            var_ratio = float(np.var(X_scaled[:, i]) / total_var) if total_var > 0 else 0
            feature_importance.append({"feature": fname, "variance_ratio": round(var_ratio, 4)})
        feature_importance.sort(key=lambda x: x["variance_ratio"], reverse=True)

    # PCA 2D projection for visualization (always computed, independent of model PCA)
    pca_scatter: dict[str, Any] | None = None
    with profiled_section("pca_2d_scatter"):
        pca_2d = PCA(n_components=2, random_state=42)
        X_2d = pca_2d.fit_transform(X_scaled)
        pc1_var = round(float(pca_2d.explained_variance_ratio_[0]) * 100, 2)
        pc2_var = round(float(pca_2d.explained_variance_ratio_[1]) * 100, 2)

        # Stratified downsample to max 2000 points for payload size
        max_scatter_points = 2000
        if n_total > max_scatter_points:
            rng = np.random.RandomState(42)
            sample_indices: list[int] = []
            for cid in np.unique(labels):
                cid_indices = np.where(labels == cid)[0]
                n_take = max(1, int(max_scatter_points * len(cid_indices) / n_total))
                chosen = rng.choice(cid_indices, min(n_take, len(cid_indices)), replace=False)
                sample_indices.extend(chosen.tolist())
            scatter_idx = np.array(sample_indices[:max_scatter_points])
        else:
            scatter_idx = np.arange(n_total)

        pca_scatter = {
            "pc1_variance": pc1_var,
            "pc2_variance": pc2_var,
            "points": [
                {
                    "pc1": round(float(X_2d[i, 0]), 3),
                    "pc2": round(float(X_2d[i, 1]), 3),
                    "cluster": int(labels[i]),
                }
                for i in scatter_idx
            ],
        }

    return {
        "optimal_k": optimal_k,
        "silhouette_score": silhouette,
        "inertia": inertia,
        "n_clusters": optimal_k,
        "total_dfus": total_dfus,
        "training_sample_size": MAX_DFUS_FOR_TRAINING if sampled else total_dfus,
        "sampled": sampled,
        "cluster_sizes": cluster_sizes,
        "k_selection_results": {
            "k_values": [int(k) for k in k_results["k_values"]],
            "inertias": [float(x) for x in k_results["inertias"]],
            "silhouette_scores": [float(x) for x in k_results["silhouette_scores"]],
            "ch_scores": [float(x) for x in k_results["ch_scores"]],
            "combined_scores": [float(x) for x in k_results["combined_scores"]],
            "feasible_mask": k_results["feasible_mask"],
        },
        "profiles": profiles,
        "feature_importance": feature_importance,
        "pca_scatter": pca_scatter,
    }


def _run_relabel_only(
    previous_scenario_id: str, lp: dict, scenario_dir: Path
) -> dict[str, Any]:
    """Re-label clusters from a previous scenario using new thresholds."""
    prev_dir = _safe_scenario_dir(previous_scenario_id)

    centroids_path = prev_dir / "cluster_centroids.csv"
    result_path = prev_dir / "scenario_result.json"

    if not centroids_path.exists():
        raise FileNotFoundError(f"Previous scenario centroids not found: {centroids_path}")
    if not result_path.exists():
        raise FileNotFoundError(f"Previous scenario result not found: {result_path}")

    centroids_df = pd.read_csv(centroids_path)
    with open(result_path) as f:
        prev_result = json.load(f)

    # Re-label with new thresholds
    if "mean_demand" not in centroids_df.columns:
        raise ValueError("Centroids CSV missing required column 'mean_demand'")
    volume_thresholds, cv_thresholds, labeling_config = _build_thresholds(
        lp, centroids_df["mean_demand"].values
    )

    cluster_labels = assign_cluster_labels(
        centroids_df, volume_thresholds, cv_thresholds, labeling_config
    )

    # Copy and update from previous result
    prev_data = prev_result.get("result", {})
    if prev_data is None:
        raise ValueError("Previous scenario has no result data")

    profiles = []
    for p in prev_data.get("profiles", []):
        cid = p["cluster_id"]
        updated = dict(p)
        updated["label"] = cluster_labels.get(cid, p["label"])
        profiles.append(updated)

    # Update labels in cluster_labels.csv if previous assignments exist
    prev_labels_path = prev_dir / "cluster_labels.csv"
    if prev_labels_path.exists():
        labels_df = pd.read_csv(prev_labels_path)
        labels_df["cluster_label"] = labels_df["cluster_id"].map(cluster_labels)
        labels_df.to_csv(scenario_dir / "cluster_labels.csv", index=False)

    centroids_df.to_csv(scenario_dir / "cluster_centroids.csv", index=False)

    return {
        "optimal_k": prev_data.get("optimal_k", 0),
        "silhouette_score": prev_data.get("silhouette_score", 0),
        "inertia": prev_data.get("inertia", 0),
        "n_clusters": prev_data.get("n_clusters", 0),
        "total_dfus": prev_data.get("total_dfus", 0),
        "cluster_sizes": prev_data.get("cluster_sizes", {}),
        "k_selection_results": prev_data.get("k_selection_results", {}),
        "profiles": profiles,
        "feature_importance": prev_data.get("feature_importance", []),
    }


def get_scenario_result(scenario_id: str) -> dict[str, Any] | None:
    """Retrieve a previously run scenario result."""
    try:
        scenario_dir = _safe_scenario_dir(scenario_id)
    except ValueError:
        return None
    result_path = scenario_dir / "scenario_result.json"
    if not result_path.exists():
        return None
    with open(result_path) as f:
        return json.load(f)


def promote_scenario(scenario_id: str) -> dict[str, Any]:
    """Promote a scenario to production by updating dim_sku.ml_cluster."""
    import shutil
    import psycopg

    scenario_dir = _safe_scenario_dir(scenario_id)
    labels_path = scenario_dir / "cluster_labels.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"Scenario labels not found: {labels_path}")

    # Copy labels to production location
    with profiled_section("copy_artifacts_to_production"):
        prod_dir = ROOT / "data" / "clustering"
        prod_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(labels_path, prod_dir / "cluster_labels.csv")

        # Also copy centroids and profiles
        for fname in ["cluster_centroids.csv", "scenario_result.json"]:
            src = scenario_dir / fname
            if src.exists():
                shutil.copy2(src, prod_dir / fname)

        # Update cluster_metadata.json so the overview panel shows correct metrics
        result_path = scenario_dir / "scenario_result.json"
        if result_path.exists():
            with open(result_path) as f:
                scenario_data = json.load(f)
            scenario_result = scenario_data.get("result", {})
            metadata = {
                "optimal_k": scenario_result.get("optimal_k"),
                "silhouette_score": scenario_result.get("silhouette_score"),
                "inertia": scenario_result.get("inertia"),
                "total_dfus": scenario_result.get("total_dfus"),
                "k_selection_results": scenario_result.get("k_selection_results"),
            }
            with open(prod_dir / "cluster_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

    # Update database
    with profiled_section("update_database"):
        df = pd.read_csv(labels_path)
        db = get_db_params()

        with psycopg.connect(**db) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TEMP TABLE _cluster_updates (
                        sku_ck TEXT PRIMARY KEY,
                        cluster_label TEXT NOT NULL
                    ) ON COMMIT DROP
                """)

                valid = df.dropna(subset=["cluster_label"])
                with cur.copy("COPY _cluster_updates (sku_ck, cluster_label) FROM STDIN") as copy:
                    for _, r in valid.iterrows():
                        copy.write_row((str(r["sku_ck"]), str(r["cluster_label"])))

                cur.execute("""
                    UPDATE dim_sku d
                    SET ml_cluster = u.cluster_label,
                        modified_ts = NOW()
                    FROM _cluster_updates u
                    WHERE d.sku_ck = u.sku_ck
                """)
                updated_count = cur.rowcount
                conn.commit()

                # Refresh accuracy MVs so Accuracy Comparison reflects new clusters
                _log = logging.getLogger(__name__)
                _log.info("Refreshing accuracy materialized views after cluster promotion ...")
                for mv in (
                    "agg_accuracy_by_dim",
                    "agg_accuracy_lag_archive",
                    "agg_dfu_coverage",
                    "agg_dfu_coverage_lag_archive",
                ):
                    try:
                        cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv}")
                    except Exception:
                        cur.execute("ROLLBACK")
                        cur.execute(f"REFRESH MATERIALIZED VIEW {mv}")
                conn.commit()
                _log.info("Accuracy MVs refreshed.")

    # Build distribution
    distribution = {}
    if "cluster_label" in df.columns:
        for label, count in df["cluster_label"].value_counts().items():
            distribution[str(label)] = int(count)

    return {
        "status": "promoted",
        "scenario_id": scenario_id,
        "dfus_updated": updated_count,
        "cluster_distribution": distribution,
    }
