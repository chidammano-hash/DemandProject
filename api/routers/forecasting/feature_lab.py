"""Feature Lab — interactive feature importance exploration for LGBM tuning UI.

Reads SHAP output files from ``data/backtest/{model_id}/shap/`` and serves
feature importance, stability, correlation, per-cluster breakdown, and
categorization endpoints.  All endpoints are read-only (file-based, no DB).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import set_cache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feature-lab"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path("data/backtest")

# Feature category definitions — maps prefix/name patterns to category metadata
CATEGORY_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "lag",
        "pattern_prefixes": ["qty_lag_"],
        "exact_names": [],
        "description": "Historical demand lags",
    },
    {
        "name": "rolling",
        "pattern_prefixes": ["rolling_mean_", "rolling_std_"],
        "exact_names": [],
        "description": "Rolling statistics",
    },
    {
        "name": "calendar",
        "pattern_prefixes": [],
        "exact_names": [
            "month", "quarter", "month_sin", "month_cos",
            "is_quarter_end", "is_year_end", "days_in_month",
        ],
        "description": "Calendar/seasonal signals",
    },
    {
        "name": "derived",
        "pattern_prefixes": ["lag_ratio_"],
        "exact_names": [
            "mom_growth", "demand_accel", "volatility_ratio",
            "n_zero_last_6m",
        ],
        "description": "Computed demand dynamics",
    },
    {
        "name": "profile",
        "pattern_prefixes": [],
        "exact_names": [
            "cv_demand", "zero_demand_pct", "trend_slope_norm",
            "recency_ratio", "seasonal_amplitude", "adi",
            "mean_demand", "yoy_correlation",
        ],
        "description": "Static DFU characteristics",
    },
    {
        "name": "categorical",
        "pattern_prefixes": [],
        "exact_names": [
            "ml_cluster", "region", "brand", "abc_vol",
            "execution_lag", "total_lt", "case_weight",
            "item_proof", "bpc",
        ],
        "description": "Categorical attributes",
    },
    {
        "name": "fourier",
        "pattern_prefixes": ["fourier_sin_", "fourier_cos_"],
        "exact_names": [],
        "description": "Fourier seasonal harmonics",
    },
    {
        "name": "croston",
        "pattern_prefixes": ["croston_"],
        "exact_names": [],
        "description": "Intermittent demand signals",
    },
    {
        "name": "cross_dfu",
        "pattern_prefixes": ["cluster_"],
        "exact_names": [],
        "description": "Cross-DFU cluster aggregates",
    },
    {
        "name": "external",
        "pattern_prefixes": ["ext_fcst_"],
        "exact_names": [],
        "description": "External forecast signals",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_feature(name: str) -> str:
    """Return the category name for a given feature, or 'other'."""
    for cat in CATEGORY_DEFINITIONS:
        if name in cat["exact_names"]:
            return cat["name"]
        for prefix in cat["pattern_prefixes"]:
            if name.startswith(prefix):
                return cat["name"]
    return "other"


_MODEL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")


def _shap_dir(model_id: str) -> Path:
    if not _MODEL_ID_PATTERN.match(model_id):
        raise ValueError(f"Invalid model_id: {model_id!r}")
    resolved = (DATA_ROOT / model_id / "shap").resolve()
    if not str(resolved).startswith(str(DATA_ROOT.resolve())):
        raise ValueError(f"Invalid model_id: {model_id!r}")
    return resolved


def _read_summary(model_id: str) -> pd.DataFrame | None:
    """Read the shap_summary.csv for a model.  Returns None if missing."""
    path = _shap_dir(model_id) / "shap_summary.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _read_timeframe_files(model_id: str) -> list[pd.DataFrame]:
    """Read all shap_timeframe_*.csv files for a model."""
    shap_dir = _shap_dir(model_id)
    if not shap_dir.exists():
        return []
    frames: list[pd.DataFrame] = []
    for path in sorted(shap_dir.glob("shap_timeframe_*.csv")):
        try:
            df = pd.read_csv(path)
            if not df.empty:
                frames.append(df)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
    return frames


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/feature-lab/importance")
def get_feature_importance(
    response: FastAPIResponse,
    model_id: str = Query(default="lgbm_cluster", max_length=120),
):
    """Current feature importance from latest backtest SHAP summary."""
    set_cache(response, max_age=120)

    summary = _read_summary(model_id)
    if summary is None:
        return {
            "available": False,
            "model_id": model_id,
            "features": [],
            "total_features": 0,
            "selected_features": 0,
        }

    features: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        name = str(row["feature"])
        features.append({
            "name": name,
            "mean_abs_shap": float(row["mean_abs_shap_across_timeframes"]),
            "rank": int(row["mean_rank"]),
            "selected_count": int(row["selected_count"]),
            "n_timeframes": int(row["n_timeframes"]),
            "category": _classify_feature(name),
        })

    # Sort by mean_abs_shap descending (highest importance first)
    features.sort(key=lambda f: f["mean_abs_shap"], reverse=True)
    # Assign rank based on sorted position
    for i, feat in enumerate(features, start=1):
        feat["rank"] = i

    n_timeframes = int(summary["n_timeframes"].iloc[0]) if len(summary) > 0 else 0
    selected = sum(1 for f in features if f["selected_count"] == n_timeframes)

    return {
        "available": True,
        "model_id": model_id,
        "features": features,
        "total_features": len(features),
        "selected_features": selected,
    }


@router.get("/feature-lab/stability")
def get_feature_stability(
    response: FastAPIResponse,
    model_id: str = Query(default="lgbm_cluster", max_length=120),
):
    """Feature rank stability across backtest timeframes."""
    set_cache(response, max_age=120)

    frames = _read_timeframe_files(model_id)
    if not frames:
        return {"available": False, "features": []}

    # Only use pooled (cluster=="all") rows for rank stability
    pooled_frames: list[pd.DataFrame] = []
    for df in frames:
        if "cluster" in df.columns:
            pooled = df[df["cluster"] == "all"]
        else:
            pooled = df
        pooled_frames.append(pooled)

    # Build feature → list of ranks across timeframes
    rank_map: dict[str, list[int]] = {}
    for df in pooled_frames:
        for _, row in df.iterrows():
            name = str(row["feature"])
            rank = int(row["rank"])
            rank_map.setdefault(name, []).append(rank)

    features: list[dict[str, Any]] = []
    for name, ranks in sorted(rank_map.items()):
        arr = np.array(ranks, dtype=np.float64)
        mean_rank = float(np.mean(arr))
        rank_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        if rank_std < 2:
            stability = "high"
        elif rank_std < 5:
            stability = "medium"
        else:
            stability = "unstable"

        features.append({
            "name": name,
            "ranks_by_timeframe": ranks,
            "mean_rank": round(mean_rank, 1),
            "rank_std": round(rank_std, 1),
            "stability": stability,
            "min_rank": int(np.min(arr)),
            "max_rank": int(np.max(arr)),
        })

    # Sort by mean_rank ascending (most important first)
    features.sort(key=lambda f: f["mean_rank"])

    return {"available": True, "features": features}


@router.get("/feature-lab/correlation")
def get_feature_correlation(
    response: FastAPIResponse,
    model_id: str = Query(default="lgbm_cluster", max_length=120),
    top_n: int = Query(default=20, ge=2, le=100),
):
    """Feature importance correlation matrix across timeframes.

    Computes pairwise Pearson correlation of feature SHAP importance profiles
    across backtest timeframes for the top-N most important features.
    """
    set_cache(response, max_age=300)

    frames = _read_timeframe_files(model_id)
    if not frames:
        return {
            "available": False,
            "features": [],
            "matrix": [],
            "high_correlation_pairs": [],
        }

    # Build a matrix: rows=features, cols=timeframes, values=mean_abs_shap
    pooled_frames: list[pd.DataFrame] = []
    for df in frames:
        if "cluster" in df.columns:
            pooled = df[df["cluster"] == "all"]
        else:
            pooled = df
        pooled_frames.append(pooled)

    # Identify top-N features by average SHAP across timeframes
    shap_sums: dict[str, float] = {}
    for df in pooled_frames:
        for _, row in df.iterrows():
            name = str(row["feature"])
            shap_sums[name] = shap_sums.get(name, 0.0) + float(row["mean_abs_shap"])

    top_features = sorted(shap_sums, key=shap_sums.get, reverse=True)[:top_n]  # type: ignore[arg-type]

    if len(top_features) < 2:
        return {
            "available": True,
            "features": top_features,
            "matrix": [[1.0]] if top_features else [],
            "high_correlation_pairs": [],
        }

    # Build importance profile matrix: feature × timeframe
    n_tf = len(pooled_frames)
    profile_data: dict[str, list[float]] = {f: [0.0] * n_tf for f in top_features}
    for tf_idx, df in enumerate(pooled_frames):
        for _, row in df.iterrows():
            name = str(row["feature"])
            if name in profile_data:
                profile_data[name][tf_idx] = float(row["mean_abs_shap"])

    profile_df = pd.DataFrame(profile_data, index=range(n_tf))
    corr_matrix = profile_df.corr(method="pearson")

    # Replace NaN with 0 (features with constant SHAP across all timeframes)
    corr_matrix = corr_matrix.fillna(0.0)

    feature_list = list(corr_matrix.columns)
    matrix = corr_matrix.values.tolist()

    # Identify high-correlation pairs (|r| > 0.9, exclude diagonal)
    high_pairs: list[dict[str, Any]] = []
    for i in range(len(feature_list)):
        for j in range(i + 1, len(feature_list)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.9:
                recommendation = (
                    "consider dropping one"
                    if abs(r) > 0.95
                    else "keep both - different signals"
                )
                high_pairs.append({
                    "feature_a": feature_list[i],
                    "feature_b": feature_list[j],
                    "correlation": round(float(r), 4),
                    "recommendation": recommendation,
                })

    high_pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)

    return {
        "available": True,
        "features": feature_list,
        "matrix": matrix,
        "high_correlation_pairs": high_pairs,
    }


@router.get("/feature-lab/per-cluster-importance")
def get_per_cluster_importance(
    response: FastAPIResponse,
    model_id: str = Query(default="lgbm_cluster", max_length=120),
):
    """Feature importance breakdown by cluster from per-timeframe SHAP files."""
    set_cache(response, max_age=120)

    frames = _read_timeframe_files(model_id)
    if not frames:
        return {
            "available": False,
            "clusters": [],
            "features": [],
            "importance_matrix": [],
            "cluster_specific_features": [],
        }

    # Collect per-cluster rows (exclude "all" pooled rows)
    cluster_rows: list[pd.DataFrame] = []
    for df in frames:
        if "cluster" not in df.columns:
            continue
        per_cluster = df[df["cluster"] != "all"]
        if not per_cluster.empty:
            cluster_rows.append(per_cluster)

    if not cluster_rows:
        return {
            "available": False,
            "clusters": [],
            "features": [],
            "importance_matrix": [],
            "cluster_specific_features": [],
            "note": "No per-cluster SHAP data found (model may use global strategy)",
        }

    combined = pd.concat(cluster_rows, ignore_index=True)

    # Average mean_abs_shap per (cluster, feature) across timeframes
    agg = combined.groupby(["cluster", "feature"])["mean_abs_shap"].mean().reset_index()
    clusters = sorted(agg["cluster"].unique().tolist())
    all_features = sorted(agg["feature"].unique().tolist())

    # Build importance matrix: rows=clusters, cols=features
    importance_matrix: list[list[float]] = []
    for cluster in clusters:
        cluster_data = agg[agg["cluster"] == cluster]
        feat_map = dict(zip(cluster_data["feature"], cluster_data["mean_abs_shap"]))
        row = [round(float(feat_map.get(f, 0.0)), 4) for f in all_features]
        importance_matrix.append(row)

    # Identify cluster-specific top features
    # For each cluster, find the feature with the highest relative importance
    # compared to its average importance across clusters
    cluster_specific: list[dict[str, str]] = []
    if len(clusters) > 1:
        for ci, cluster in enumerate(clusters):
            cluster_importances = np.array(importance_matrix[ci])
            # Mean importance across all clusters for each feature
            all_importances = np.array(importance_matrix)
            mean_across_clusters = all_importances.mean(axis=0)
            # Relative lift: how much more important in this cluster vs average
            with np.errstate(divide="ignore", invalid="ignore"):
                lift = np.where(
                    mean_across_clusters > 0,
                    cluster_importances / mean_across_clusters,
                    0.0,
                )
            best_idx = int(np.argmax(lift))
            if lift[best_idx] > 1.2:  # At least 20% above average
                cluster_specific.append({
                    "cluster": str(cluster),
                    "top_unique_feature": all_features[best_idx],
                    "note": (
                        f"{all_features[best_idx]} is {lift[best_idx]:.1f}x "
                        f"more important in cluster {cluster} than average"
                    ),
                })

    return {
        "available": True,
        "clusters": clusters,
        "features": all_features,
        "importance_matrix": importance_matrix,
        "cluster_specific_features": cluster_specific,
    }


@router.get("/feature-lab/categories")
def get_feature_categories(
    response: FastAPIResponse,
    model_id: str = Query(default="lgbm_cluster", max_length=120),
):
    """Feature categorization — groups features by type with descriptions."""
    set_cache(response, max_age=600)

    summary = _read_summary(model_id)

    # If summary available, use its feature list; otherwise return static defs
    if summary is not None:
        feature_names = summary["feature"].tolist()
    else:
        feature_names = []

    # Build categorized output
    categorized: dict[str, list[str]] = {}
    for name in feature_names:
        cat = _classify_feature(name)
        categorized.setdefault(cat, []).append(name)

    categories: list[dict[str, Any]] = []
    for cat_def in CATEGORY_DEFINITIONS:
        cat_name = cat_def["name"]
        cat_features = categorized.pop(cat_name, [])
        categories.append({
            "name": cat_name,
            "features": sorted(cat_features),
            "description": cat_def["description"],
            "count": len(cat_features),
        })

    # Add 'other' for unclassified features
    other_features = categorized.pop("other", [])
    if other_features:
        categories.append({
            "name": "other",
            "features": sorted(other_features),
            "description": "Uncategorized features",
            "count": len(other_features),
        })

    return {
        "available": summary is not None,
        "model_id": model_id,
        "categories": categories,
        "total_features": len(feature_names),
    }
