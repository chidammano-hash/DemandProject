"""
Run a clustering scenario with custom parameters in a temp directory.

Orchestrates the clustering pipeline (feature gen, training, labeling)
without modifying production data. Returns structured results for the
What-If Scenarios UI.
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_clustering_features import compute_time_series_features, get_db_conn
from scripts.train_clustering_model import (
    CORE_FEATURES,
    LOG_TRANSFORM_FEATURES,
    find_optimal_k,
    merge_small_clusters,
)
from scripts.label_clusters import assign_cluster_labels

# Directory to store scenario temp data
SCENARIO_BASE = Path("/tmp/clustering_scenarios")


def generate_scenario_id() -> str:
    """Generate a unique scenario ID."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:4]
    return f"sc_{ts}_{short_uuid}"


def run_scenario(
    feature_params: dict[str, Any] | None = None,
    model_params: dict[str, Any] | None = None,
    label_params: dict[str, Any] | None = None,
    scenario_id: str | None = None,
    relabel_only: bool = False,
    previous_scenario_id: str | None = None,
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

    Returns
    -------
    dict with scenario_id, status, runtime_seconds, params, result
    """
    load_dotenv(ROOT / ".env")

    if scenario_id is None:
        scenario_id = generate_scenario_id()

    # Merge with defaults
    fp = {"time_window_months": 24, "min_months_history": 1}
    if feature_params:
        fp.update(feature_params)

    mp = {
        "k_range": [3, 12],
        "min_cluster_size_pct": 2.0,
        "use_pca": False,
        "pca_components": None,
        "skip_gap": True,
        "all_features": False,
    }
    if model_params:
        mp.update(model_params)

    lp = {
        "volume_high": 0.75,
        "volume_low": 0.25,
        "cv_steady": 0.3,
        "cv_volatile": 0.8,
        "seasonality_threshold": 0.5,
        "zero_demand_threshold": 0.2,
    }
    if label_params:
        lp.update(label_params)

    merged_params = {
        "feature_params": fp,
        "model_params": mp,
        "label_params": lp,
    }

    # Create scenario directory
    scenario_dir = SCENARIO_BASE / scenario_id
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

        return output

    except Exception as e:
        runtime = time.time() - start_time
        return {
            "scenario_id": scenario_id,
            "status": "failed",
            "runtime_seconds": round(runtime, 1),
            "params": merged_params,
            "result": None,
            "error": str(e),
        }


def _run_full_pipeline(
    fp: dict, mp: dict, lp: dict, scenario_dir: Path
) -> dict[str, Any]:
    """Run the full clustering pipeline: features → training → labeling."""
    import psycopg
    from datetime import date, timedelta
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score as sk_silhouette

    db = get_db_conn()

    # Step 1: Load sales data
    time_window = fp["time_window_months"]
    if isinstance(time_window, str) and time_window.lower() == "all":
        cutoff_date = None
    else:
        cutoff_date = date.today() - timedelta(days=int(time_window) * 30)

    with psycopg.connect(**db) as conn:
        sales_query = """
            SELECT d.dfu_ck, s.startdate, s.qty
            FROM fact_sales_monthly s
            INNER JOIN dim_dfu d
                ON d.dmdunit = s.dmdunit AND d.dmdgroup = s.dmdgroup AND d.loc = s.loc
            WHERE s.qty IS NOT NULL
        """
        params: dict[str, object] = {}
        if cutoff_date:
            sales_query += " AND s.startdate >= %(cutoff)s"
            params["cutoff"] = cutoff_date
        sales_query += " ORDER BY d.dfu_ck, s.startdate"

        sales_df = pd.read_sql(sales_query, conn, params=params if params else None)
    sales_df["startdate"] = pd.to_datetime(sales_df["startdate"])

    # Step 2: Compute time series features
    grouped = sales_df.groupby("dfu_ck", sort=False)
    ts_features_list = []
    for dfu_ck, dfu_sales in grouped:
        if len(dfu_sales) < fp["min_months_history"]:
            continue
        ts = compute_time_series_features(dfu_sales)
        ts["dfu_ck"] = dfu_ck
        ts_features_list.append(ts)

    if not ts_features_list:
        raise ValueError("No DFUs met minimum history requirement")

    feature_df = pd.DataFrame(ts_features_list)
    feature_df.to_csv(scenario_dir / "clustering_features.csv", index=False)

    # Step 3: Prepare features for clustering
    if mp["all_features"]:
        metadata_cols = ["dfu_ck"]
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
    if "trend_slope" in X_df.columns:
        X_df["trend_slope"] = np.sign(X_df["trend_slope"]) * np.log1p(np.abs(X_df["trend_slope"]))

    X = X_df.values
    feature_names = list(numeric_cols)

    # Remove low-variance features
    variances = np.var(X, axis=0)
    high_var_mask = variances > 0.001
    X = X[:, high_var_mask]
    feature_names = [feature_names[i] for i in range(len(feature_names)) if high_var_mask[i]]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional PCA
    pca = None
    if mp["use_pca"]:
        n_comp = mp["pca_components"] or min(50, X_scaled.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        X_scaled = pca.fit_transform(X_scaled)

    # Step 4: Find optimal K
    k_range = tuple(mp["k_range"])
    k_results = find_optimal_k(
        X_scaled, k_range, mp["min_cluster_size_pct"], skip_gap=mp["skip_gap"]
    )
    optimal_k = k_results["optimal_k"]

    # Train final model
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    # Merge small clusters
    n_samples = len(X_scaled)
    min_cluster_size = int(n_samples * mp["min_cluster_size_pct"] / 100)
    unique_pre, counts_pre = np.unique(labels, return_counts=True)
    if any(c < min_cluster_size for c in counts_pre):
        labels, centroids_scaled = merge_small_clusters(
            X_scaled, labels, kmeans.cluster_centers_, min_cluster_size
        )
        optimal_k = len(set(labels))
    else:
        centroids_scaled = kmeans.cluster_centers_

    # Metrics
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
    if "trend_slope" in centroids_df.columns:
        centroids_df["trend_slope"] = np.sign(centroids_df["trend_slope"]) * np.expm1(np.abs(centroids_df["trend_slope"]))

    centroids_df["cluster_id"] = range(optimal_k)

    # Step 5: Label clusters
    mean_demands = centroids_df["mean_demand"].values
    vol_high_abs = np.percentile(mean_demands, lp["volume_high"] * 100)
    vol_low_abs = np.percentile(mean_demands, lp["volume_low"] * 100)
    volume_thresholds = {"very_high": vol_high_abs * 10, "high": vol_high_abs, "low": vol_low_abs}
    cv_thresholds = {"steady": lp["cv_steady"], "volatile": lp["cv_volatile"]}

    cluster_labels = assign_cluster_labels(
        centroids_df, volume_thresholds, cv_thresholds,
        lp["seasonality_threshold"], lp["zero_demand_threshold"]
    )

    # Save artifacts
    assignments_df = pd.DataFrame({"dfu_ck": feature_df["dfu_ck"].values[:n_samples], "cluster_id": labels})
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
            "trend_slope": round(float(row.get("trend_slope", 0)), 4),
            "growth_rate": round(float(row.get("growth_rate", 0)), 4),
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

    return {
        "optimal_k": optimal_k,
        "silhouette_score": silhouette,
        "inertia": inertia,
        "n_clusters": optimal_k,
        "total_dfus": total_dfus,
        "cluster_sizes": cluster_sizes,
        "k_selection_results": {
            "k_values": [int(k) for k in k_results["k_values"]],
            "inertias": [float(x) for x in k_results["inertias"]],
            "silhouette_scores": [float(x) for x in k_results["silhouette_scores"]],
            "gap_stats": [float(x) for x in k_results["gap_stats"]] if k_results["gap_stats"] else None,
        },
        "profiles": profiles,
        "feature_importance": feature_importance,
    }


def _run_relabel_only(
    previous_scenario_id: str, lp: dict, scenario_dir: Path
) -> dict[str, Any]:
    """Re-label clusters from a previous scenario using new thresholds."""
    prev_dir = SCENARIO_BASE / previous_scenario_id

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
    mean_demands = centroids_df["mean_demand"].values
    vol_high_abs = np.percentile(mean_demands, lp["volume_high"] * 100)
    vol_low_abs = np.percentile(mean_demands, lp["volume_low"] * 100)
    volume_thresholds = {"very_high": vol_high_abs * 10, "high": vol_high_abs, "low": vol_low_abs}
    cv_thresholds = {"steady": lp["cv_steady"], "volatile": lp["cv_volatile"]}

    cluster_labels = assign_cluster_labels(
        centroids_df, volume_thresholds, cv_thresholds,
        lp["seasonality_threshold"], lp["zero_demand_threshold"]
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
    result_path = SCENARIO_BASE / scenario_id / "scenario_result.json"
    if not result_path.exists():
        return None
    with open(result_path) as f:
        return json.load(f)


def promote_scenario(scenario_id: str) -> dict[str, Any]:
    """Promote a scenario to production by updating dim_dfu.ml_cluster."""
    import shutil
    import psycopg

    scenario_dir = SCENARIO_BASE / scenario_id
    labels_path = scenario_dir / "cluster_labels.csv"

    if not labels_path.exists():
        raise FileNotFoundError(f"Scenario labels not found: {labels_path}")

    # Copy labels to production location
    prod_dir = ROOT / "data" / "clustering"
    prod_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(labels_path, prod_dir / "cluster_labels.csv")

    # Also copy centroids and profiles
    for fname in ["cluster_centroids.csv", "scenario_result.json"]:
        src = scenario_dir / fname
        if src.exists():
            shutil.copy2(src, prod_dir / fname)

    # Update database
    df = pd.read_csv(labels_path)
    db = get_db_conn()

    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TEMP TABLE _cluster_updates (
                    dfu_ck TEXT PRIMARY KEY,
                    cluster_label TEXT NOT NULL
                ) ON COMMIT DROP
            """)

            valid = df.dropna(subset=["cluster_label"])
            with cur.copy("COPY _cluster_updates (dfu_ck, cluster_label) FROM STDIN") as copy:
                for _, r in valid.iterrows():
                    copy.write_row((str(r["dfu_ck"]), str(r["cluster_label"])))

            cur.execute("""
                UPDATE dim_dfu d
                SET ml_cluster = u.cluster_label,
                    modified_ts = NOW()
                FROM _cluster_updates u
                WHERE d.dfu_ck = u.dfu_ck
            """)
            updated_count = cur.rowcount
            conn.commit()

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
