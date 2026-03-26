"""SHAP-based per-timeframe feature selection for tree-based backtests (Feature 42).

Provides model-agnostic SHAP computation, cumulative-importance-based feature
selection, and CSV output helpers.  Model-specific SHAP extractors (LGBM / XGBoost
vs CatBoost native) are passed as callables so this module stays framework-agnostic.

Typical flow per backtest timeframe:
  1. Train initial model on all features.
  2. Call compute_timeframe_shap() → (selected_features, shap_df).
  3. If selected_features ⊊ all_features, retrain on selected_features.
  4. Call save_shap_outputs() once after all timeframes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import logging

from common.constants import PROTECTED_FEATURES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# Signature: (model_or_dict, X_sample, feature_cols, cat_cols) → np.ndarray
# Returned shape: (n_samples, n_features), values are absolute SHAP values.
ShapExtractorFn = Callable[[Any, pd.DataFrame, list[str], list[str]], np.ndarray]

SHAP_REPORT_COLS = ["feature", "mean_abs_shap", "rank", "selected", "timeframe", "cutoff_date", "cluster"]
SHAP_SUMMARY_COLS = [
    "feature",
    "mean_abs_shap_across_timeframes",
    "mean_rank",
    "selected_count",
    "n_timeframes",
]


# ---------------------------------------------------------------------------
# Model-specific SHAP extractors
# ---------------------------------------------------------------------------


def compute_shap_global(
    model: Any,
    X_sample: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
) -> np.ndarray:
    """Extract |SHAP| values via native pred_contribs (XGBoost, LGBM) or shap.TreeExplainer.

    Uses native XGBoost pred_contribs when available (avoids shap library issues
    with categorical dtypes). Falls back to shap.TreeExplainer for LGBM.

    Returns absolute SHAP values, shape (n_samples, n_features).
    """
    # XGBoost: use native pred_contribs (more reliable than shap library with categoricals)
    module_name = type(model).__module__.split(".")[0].lower()
    if module_name == "xgboost" or hasattr(model, "get_booster"):
        try:
            import xgboost as xgb
            booster = model.get_booster() if hasattr(model, "get_booster") else model
            dmatrix = xgb.DMatrix(X_sample, enable_categorical=True)
            full = booster.predict(dmatrix, pred_contribs=True)
            return np.abs(full[:, :-1])  # strip baseline column
        except Exception:
            pass  # fall through to shap.TreeExplainer

    # LGBM: use native pred_contrib (avoids shap library dtype issues)
    if module_name == "lightgbm" or hasattr(model, "booster_"):
        try:
            full = model.predict(X_sample, pred_contrib=True)
            return np.abs(full[:, :-1])  # strip baseline column
        except Exception:
            pass  # fall through to shap.TreeExplainer

    # Fallback: shap.TreeExplainer
    import shap  # lazy import — not required for module-level usage

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        # Multi-output: average across outputs
        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    return np.abs(shap_values)


def compute_shap_catboost(
    model: Any,
    X_sample: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
) -> np.ndarray:
    """Extract |SHAP| values via CatBoost native ShapValues (no shap library needed).

    CatBoost's get_feature_importance(type="ShapValues") returns shape
    (n_samples, n_features + 1) — the last column is the baseline expected
    value and must be stripped.

    Returns absolute SHAP values, shape (n_samples, n_features).
    """
    import catboost as cb

    cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
    pool = cb.Pool(X_sample, cat_features=cat_indices)
    shap_matrix = model.get_feature_importance(data=pool, type="ShapValues")
    return np.abs(shap_matrix[:, :-1])  # strip baseline column


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _weighted_pool_cluster_shap(
    models: dict[str, Any],
    train_data: pd.DataFrame,
    effective_feature_cols: list[str],
    effective_cat_cols: list[str],
    shap_extractor_fn: ShapExtractorFn,
    sample_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute SHAP for each cluster model and pool by cluster size.

    Skips the "__base__" key present in transfer-learning model dicts.

    Returns:
        pooled: mean absolute SHAP values per feature, shape (n_features,).
        per_cluster: dict mapping cluster_label → mean |SHAP| array, shape (n_features,).
    """
    weighted_shap = np.zeros(len(effective_feature_cols))
    total_rows = 0
    per_cluster: dict[str, np.ndarray] = {}

    for cluster_label, model in models.items():
        if cluster_label == "__base__":
            continue
        cluster_data = train_data[train_data["ml_cluster"] == cluster_label]
        if len(cluster_data) == 0:
            continue
        n = min(sample_size, len(cluster_data))
        X_sample = cluster_data[effective_feature_cols].sample(n=n, random_state=42)
        try:
            abs_shap = shap_extractor_fn(model, X_sample, effective_feature_cols, effective_cat_cols)
            cluster_mean = abs_shap.mean(axis=0)
            per_cluster[str(cluster_label)] = cluster_mean
            weighted_shap += cluster_mean * n
            total_rows += n
        except Exception as exc:
            logger.warning("[shap] SHAP extraction failed for cluster '%s': %s", cluster_label, exc)

    if total_rows > 0:
        weighted_shap /= total_rows
    return weighted_shap, per_cluster


def _select_features_from_shap(
    mean_abs_shap: np.ndarray,
    feature_cols: list[str],
    timeframe_idx: int,
    cutoff_date: pd.Timestamp,
    cumulative_threshold: float = 0.95,
    min_features: int = 5,
    top_n: int | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Select features by cumulative SHAP importance and build the report DataFrame.

    Two modes:
    - top_n is set: keep exactly max(top_n, min_features) features.
    - top_n is None: keep features covering cumulative_threshold of total SHAP mass,
      with min_features as a lower bound.

    Returns (selected_feature_names, shap_report_df).
    shap_report_df columns: feature, mean_abs_shap, rank, selected, timeframe, cutoff_date.
    """
    total_shap = float(mean_abs_shap.sum())
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    sorted_shap = mean_abs_shap[sorted_indices]
    sorted_features = [feature_cols[i] for i in sorted_indices]

    if top_n is not None:
        n_select = max(min_features, min(int(top_n), len(feature_cols)))
    elif total_shap == 0:
        n_select = len(feature_cols)
    else:
        cumsum = np.cumsum(sorted_shap) / total_shap
        n_select = int(np.searchsorted(cumsum, cumulative_threshold)) + 1
        n_select = max(min_features, min(n_select, len(feature_cols)))

    selected_set = set(sorted_features[:n_select])

    # Always keep protected features (month, month_sin, month_cos, quarter, ml_cluster)
    for feat in feature_cols:
        if feat in PROTECTED_FEATURES:
            selected_set.add(feat)

    shap_df = pd.DataFrame({
        "feature": sorted_features,
        "mean_abs_shap": sorted_shap.tolist(),
        "rank": list(range(1, len(sorted_features) + 1)),
        "selected": [f in selected_set for f in sorted_features],
        "timeframe": [timeframe_idx] * len(sorted_features),
        "cutoff_date": [str(cutoff_date.date())] * len(sorted_features),
    })

    # Return selected features in SHAP-rank order, with protected features appended
    result_features = [f for f in sorted_features if f in selected_set]
    logger.info(
        "[shap] Selected %d/%d features (threshold=%.2f, protected=%d, top: %s)",
        len(result_features),
        len(feature_cols),
        cumulative_threshold,
        len(PROTECTED_FEATURES & set(feature_cols)),
        sorted_features[0],
    )
    return result_features, shap_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_timeframe_shap(
    model_or_dict: Any,
    train_data: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    timeframe_idx: int,
    cutoff_date: pd.Timestamp,
    shap_extractor_fn: ShapExtractorFn,
    cluster_strategy: str,
    sample_size: int = 500,
    cumulative_threshold: float = 0.95,
    top_n: int | None = None,
    min_features: int = 5,
) -> tuple[list[str], pd.DataFrame]:
    """Compute SHAP values for one backtest timeframe and select top features.

    Handles both single-model (global strategy) and dict-of-models
    (per_cluster / transfer strategies).  ml_cluster is kept in the feature
    list for all strategies — per-cluster models are trained WITH ml_cluster
    as a hard feature (constant within each cluster partition).

    Args:
        model_or_dict: Trained model (global) or dict[cluster_label → model].
        train_data: Causally masked training DataFrame for this timeframe.
        feature_cols: Full feature column list from the feature matrix.
        cat_cols: Categorical feature names.
        timeframe_idx: 0-based timeframe index (0=A, 1=B, …).
        cutoff_date: Training cutoff date for this timeframe.
        shap_extractor_fn: Model-specific SHAP extractor callable.
        cluster_strategy: "global", "per_cluster", or "transfer".
        sample_size: Max rows to sample per cluster for SHAP computation.
        cumulative_threshold: Cumulative SHAP mass threshold for feature selection.
        top_n: If set, select exactly this many features (overrides threshold).
        min_features: Minimum number of features to keep regardless of threshold.

    Returns:
        (selected_features, shap_df) where shap_df has SHAP_REPORT_COLS columns.
        When per-cluster SHAP data is available, shap_df also includes a
        'cluster' column (value "all" for pooled rows, cluster label for per-cluster rows).
    """
    t0 = time.time()

    # All strategies keep ml_cluster — models are trained with it as a hard feature
    effective_feature_cols = feature_cols
    effective_cat_cols = cat_cols

    # Compute mean absolute SHAP across the training sample
    per_cluster_shap: dict[str, np.ndarray] = {}
    try:
        if isinstance(model_or_dict, dict):
            mean_abs_shap, per_cluster_shap = _weighted_pool_cluster_shap(
                model_or_dict,
                train_data,
                effective_feature_cols,
                effective_cat_cols,
                shap_extractor_fn,
                sample_size,
            )
        else:
            n = min(sample_size, len(train_data))
            X_sample = train_data[effective_feature_cols].sample(n=n, random_state=42)
            abs_shap = shap_extractor_fn(
                model_or_dict, X_sample, effective_feature_cols, effective_cat_cols
            )
            mean_abs_shap = abs_shap.mean(axis=0)
    except Exception as exc:
        logger.warning("[shap] SHAP computation failed: %s. Keeping all features.", exc)
        shap_df = pd.DataFrame({
            "feature": effective_feature_cols,
            "mean_abs_shap": [0.0] * len(effective_feature_cols),
            "rank": list(range(1, len(effective_feature_cols) + 1)),
            "selected": [True] * len(effective_feature_cols),
            "timeframe": [timeframe_idx] * len(effective_feature_cols),
            "cutoff_date": [str(cutoff_date.date())] * len(effective_feature_cols),
            "cluster": ["all"] * len(effective_feature_cols),
        })
        return effective_feature_cols, shap_df

    logger.info("[shap] SHAP computed (%.1fs)", time.time() - t0)

    selected, pooled_df = _select_features_from_shap(
        mean_abs_shap,
        effective_feature_cols,
        timeframe_idx,
        cutoff_date,
        cumulative_threshold=cumulative_threshold,
        min_features=min_features,
        top_n=top_n,
    )
    pooled_df["cluster"] = "all"

    # Append per-cluster breakdowns
    if per_cluster_shap:
        cluster_dfs = []
        for cluster_label, cluster_shap in per_cluster_shap.items():
            _, cluster_df = _select_features_from_shap(
                cluster_shap,
                effective_feature_cols,
                timeframe_idx,
                cutoff_date,
                cumulative_threshold=cumulative_threshold,
                min_features=min_features,
                top_n=top_n,
            )
            cluster_df["cluster"] = cluster_label
            cluster_dfs.append(cluster_df)
        combined_df = pd.concat([pooled_df] + cluster_dfs, ignore_index=True)
    else:
        combined_df = pooled_df

    return selected, combined_df


def build_shap_summary(
    timeframe_reports: list[pd.DataFrame],
    n_timeframes: int,
) -> pd.DataFrame:
    """Aggregate per-timeframe SHAP reports into a cross-timeframe summary.

    Returns a DataFrame with SHAP_SUMMARY_COLS columns sorted by descending
    mean importance.  Returns an empty DataFrame if timeframe_reports is empty.
    """
    if not timeframe_reports:
        return pd.DataFrame(columns=SHAP_SUMMARY_COLS)

    combined = pd.concat(timeframe_reports, ignore_index=True)
    summary = (
        combined.groupby("feature")
        .agg(
            mean_abs_shap_across_timeframes=("mean_abs_shap", "mean"),
            mean_rank=("rank", "mean"),
            selected_count=("selected", "sum"),
        )
        .reset_index()
    )
    summary["n_timeframes"] = n_timeframes
    summary = summary.sort_values("mean_abs_shap_across_timeframes", ascending=False).reset_index(drop=True)
    return summary[SHAP_SUMMARY_COLS]


def save_shap_outputs(
    timeframe_reports: list[pd.DataFrame],
    output_dir: Path,
    n_timeframes: int,
) -> tuple[list[Path], Path | None]:
    """Save per-timeframe SHAP CSVs and a cross-timeframe summary.

    Files written to output_dir/shap/:
      shap_timeframe_00.csv … shap_timeframe_09.csv   (one per timeframe, includes per-cluster rows)
      shap_summary.csv                                  (aggregated, pooled only)

    Per-cluster rows in the timeframe CSVs have cluster != "all".
    The summary CSV aggregates only the pooled ("all") rows.

    Returns (list_of_timeframe_paths, summary_path) — summary_path is None
    if timeframe_reports is empty.
    """
    if not timeframe_reports:
        return [], None

    shap_dir = output_dir / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    # Ensure 'cluster' column exists for backward compat with old reports
    for df in timeframe_reports:
        if "cluster" not in df.columns:
            df["cluster"] = "all"

    timeframe_paths: list[Path] = []
    for df in timeframe_reports:
        if df.empty:
            continue
        idx = int(df["timeframe"].iloc[0])
        path = shap_dir / f"shap_timeframe_{idx:02d}.csv"
        cols = [c for c in SHAP_REPORT_COLS if c in df.columns]
        df[cols].to_csv(path, index=False)
        timeframe_paths.append(path)

    # Summary uses only pooled rows
    pooled_reports = []
    for df in timeframe_reports:
        pooled = df[df["cluster"] == "all"] if "cluster" in df.columns else df
        if not pooled.empty:
            pooled_reports.append(pooled)
    summary_df = build_shap_summary(pooled_reports, n_timeframes)
    summary_path = shap_dir / "shap_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    logger.info("[shap] Saved %d timeframe reports + summary to %s", len(timeframe_paths), shap_dir)
    return timeframe_paths, summary_path
