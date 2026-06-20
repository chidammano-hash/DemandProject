"""Multi-stage per-timeframe feature selection for tree-based backtests.

Pipeline stages (all per-timeframe, causal — computed on training data only):
  Stage 0: Remove exact-duplicate aliases (static mapping from constants.py)
  Stage 1: Near-zero variance filter (relative variance < threshold)
  Stage 2: Correlation-based pre-filter (drop lower-variance member of pairs > threshold)
  Stage 3: SHAP cumulative-importance selection (existing, on reduced set)

Model-specific SHAP extractors (LGBM / XGBoost vs CatBoost native) are passed
as callables so this module stays framework-agnostic.

Typical flow per backtest timeframe:
  1. Train initial model on all features.
  2. Call compute_timeframe_shap() → (selected_features, shap_df).
  3. If selected_features ⊊ all_features, retrain on selected_features.
  4. Call save_shap_outputs() once after all timeframes.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.core.constants import DUPLICATE_FEATURE_ALIASES, PROTECTED_FEATURES

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
# Sparse cluster thresholds for SHAP sampling
# ---------------------------------------------------------------------------

# Clusters with zero_demand_pct above this use stratified SHAP sampling
# (50% zero + 50% non-zero) instead of random sampling.
SPARSE_ZERO_PCT_THRESHOLD = 0.5

# Clusters with fewer non-zero training rows than this skip SHAP selection
# entirely and keep all features (SHAP is unreliable with too few non-zero samples).
SPARSE_MIN_NONZERO_ROWS = 100


# ---------------------------------------------------------------------------
# Sparse cluster stratified sampling
# ---------------------------------------------------------------------------


def _stratified_sample_for_shap(
    cluster_data: pd.DataFrame,
    feature_cols: list[str],
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample rows for SHAP, stratifying by zero/non-zero demand for sparse clusters.

    For clusters where >50% of rows have zero demand (``qty == 0``), random
    sampling produces a SHAP sample dominated by zeros.  SHAP then attributes
    high importance to features that separate zero from non-zero (e.g.
    ``zero_demand_pct``, ``brand``) rather than features that predict demand
    *levels*.

    This function detects sparse clusters and applies stratified sampling:
    50% zero-demand rows + 50% non-zero-demand rows (up to ``sample_size``).
    For non-sparse clusters, standard random sampling is used.

    If the cluster has fewer than ``SPARSE_MIN_NONZERO_ROWS`` non-zero rows,
    returns ``None`` — the caller should skip SHAP and keep all features.

    Args:
        cluster_data: Training data for a single cluster (must contain ``qty``
            column and ``feature_cols``).
        feature_cols: Feature columns to select from the sample.
        sample_size: Maximum total sample size.
        random_state: Random seed for reproducibility.

    Returns:
        Sampled DataFrame with ``feature_cols`` columns, or ``None`` if the
        cluster has too few non-zero rows for reliable SHAP analysis.
    """
    # If qty column is missing, fall back to standard random sampling
    if "qty" not in cluster_data.columns:
        n = min(sample_size, len(cluster_data))
        return cluster_data[feature_cols].sample(n=n, random_state=random_state)

    qty = cluster_data["qty"]
    nonzero_mask = qty > 0
    n_nonzero = int(nonzero_mask.sum())
    n_total = len(cluster_data)
    zero_pct = 1.0 - (n_nonzero / n_total) if n_total > 0 else 1.0

    # Too few non-zero rows → SHAP is unreliable, skip selection
    if n_nonzero < SPARSE_MIN_NONZERO_ROWS:
        logger.info(
            "[shap] Cluster has only %d non-zero rows (< %d threshold); "
            "skipping SHAP selection — keeping all features.",
            n_nonzero, SPARSE_MIN_NONZERO_ROWS,
        )
        return None

    # Sparse cluster: stratified sampling (50% zero + 50% non-zero)
    if zero_pct > SPARSE_ZERO_PCT_THRESHOLD:
        half = sample_size // 2
        nonzero_data = cluster_data[nonzero_mask]
        zero_data = cluster_data[~nonzero_mask]

        n_nonzero_sample = min(half, len(nonzero_data))
        n_zero_sample = min(half, len(zero_data))

        sampled = pd.concat([
            nonzero_data.sample(n=n_nonzero_sample, random_state=random_state),
            zero_data.sample(n=n_zero_sample, random_state=random_state),
        ], ignore_index=True)

        logger.info(
            "[shap] Sparse cluster (%.0f%% zeros): stratified SHAP sample "
            "%d non-zero + %d zero = %d total (was %d random)",
            zero_pct * 100, n_nonzero_sample, n_zero_sample,
            len(sampled), min(sample_size, n_total),
        )
        return sampled[feature_cols]

    # Non-sparse cluster: standard random sampling
    n = min(sample_size, n_total)
    return cluster_data[feature_cols].sample(n=n, random_state=random_state)


# ---------------------------------------------------------------------------
# Stage 0: Static duplicate removal
# ---------------------------------------------------------------------------


def _remove_duplicate_features(feature_cols: list[str]) -> tuple[list[str], set[str]]:
    """Remove known exact-duplicate features (alias → canonical mapping).

    Returns (filtered_cols, excluded_set).
    """
    aliases = set(DUPLICATE_FEATURE_ALIASES.keys())
    excluded = {f for f in feature_cols if f in aliases}
    filtered = [f for f in feature_cols if f not in aliases]
    if excluded:
        logger.info("[feat-select] Stage 0: removed %d duplicate aliases: %s",
                    len(excluded), sorted(excluded))
    return filtered, excluded


# ---------------------------------------------------------------------------
# Stage 1: Near-zero variance filter
# ---------------------------------------------------------------------------


def _remove_low_variance_features(
    train_data: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    threshold: float = 0.01,
) -> tuple[list[str], set[str]]:
    """Remove numeric features with near-zero variance relative to range.

    Categorical features and PROTECTED_FEATURES are never removed.
    threshold = fraction of range squared: var / (max - min + eps)^2.

    Returns (filtered_cols, excluded_set).
    """
    cat_set = set(cat_cols)
    keep: list[str] = []
    excluded: set[str] = set()
    for col in feature_cols:
        if col in cat_set or col in PROTECTED_FEATURES:
            keep.append(col)
            continue
        if col not in train_data.columns:
            keep.append(col)
            continue
        series = train_data[col]
        val_range = float(series.max() - series.min())
        if val_range < 1e-12:
            excluded.add(col)
            continue
        relative_var = float(series.var()) / (val_range ** 2)
        if relative_var < threshold:
            excluded.add(col)
            continue
        keep.append(col)
    if excluded:
        logger.info("[feat-select] Stage 1: removed %d near-zero-variance features: %s",
                    len(excluded), sorted(excluded))
    return keep, excluded


# ---------------------------------------------------------------------------
# Stage 2: Correlation-based pre-filter
# ---------------------------------------------------------------------------


def _remove_correlated_features(
    train_data: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    threshold: float = 0.95,
    sample_size: int = 5000,
) -> tuple[list[str], set[str]]:
    """Remove one feature from each highly-correlated pair (>threshold).

    For each pair, keeps the feature with higher variance.
    PROTECTED_FEATURES are never dropped. Categorical features are skipped.
    Uses a sample of train_data for efficiency.

    Returns (filtered_cols, excluded_set).
    """
    cat_set = set(cat_cols)
    numeric_cols = [c for c in feature_cols if c not in cat_set and c in train_data.columns]

    if len(numeric_cols) < 2:
        return feature_cols, set()

    # Sample for speed
    if len(train_data) > sample_size:
        sample = train_data[numeric_cols].sample(n=sample_size, random_state=42)
    else:
        sample = train_data[numeric_cols]

    corr_matrix = sample.corr().abs()
    variances = sample.var()
    to_drop: set[str] = set()

    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
            if corr_matrix.iloc[i, j] >= threshold:
                fi, fj = cols[i], cols[j]
                fi_protected = fi in PROTECTED_FEATURES
                fj_protected = fj in PROTECTED_FEATURES
                if fi_protected and not fj_protected:
                    to_drop.add(fj)
                elif fj_protected and not fi_protected:
                    to_drop.add(fi)
                elif fi_protected and fj_protected:
                    continue
                else:
                    # Drop the one with lower variance
                    if variances[fi] >= variances[fj]:
                        to_drop.add(fj)
                    else:
                        to_drop.add(fi)

    filtered = [f for f in feature_cols if f not in to_drop]
    if to_drop:
        logger.info("[feat-select] Stage 2: removed %d correlated features (threshold=%.2f): %s",
                    len(to_drop), threshold, sorted(to_drop))
    return filtered, to_drop


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
            # Ensure categorical columns have category dtype to match model training
            for col in cat_cols:
                if col in X_sample.columns and X_sample[col].dtype.name != "category":
                    X_sample = X_sample.copy()
                    X_sample[col] = X_sample[col].astype("category")
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
    all_feature_cols: list[str] | None = None,
    all_cat_cols: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute SHAP for each cluster model and pool by cluster size.

    Skips the "__base__" key present in transfer-learning model dicts.

    When ``all_feature_cols`` is provided, the model is called with the full
    feature set (matching its training shape) and the result is sliced to
    ``effective_feature_cols`` indices.  This avoids the LightGBM error
    "train and valid dataset categorical_feature do not match" that occurs
    when ``ml_cluster`` is stripped before calling a model that was trained
    with it.

    Returns:
        pooled: mean absolute SHAP values per feature, shape (n_features,).
        per_cluster: dict mapping cluster_label → mean |SHAP| array, shape (n_features,).
    """
    # If all_feature_cols provided, call the model with full features and slice output
    model_feature_cols = all_feature_cols or effective_feature_cols
    model_cat_cols = all_cat_cols or effective_cat_cols
    # Build index mask to select effective columns from the full SHAP output
    if all_feature_cols and all_feature_cols != effective_feature_cols:
        effective_indices = [model_feature_cols.index(c) for c in effective_feature_cols]
    else:
        effective_indices = None  # no slicing needed

    weighted_shap = np.zeros(len(effective_feature_cols))
    total_rows = 0
    per_cluster: dict[str, np.ndarray] = {}

    for cluster_label, model in models.items():
        if cluster_label == "__base__":
            continue
        cluster_data = train_data[train_data["ml_cluster"] == cluster_label]
        if len(cluster_data) == 0:
            continue

        # Use stratified sampling for sparse clusters
        X_sample = _stratified_sample_for_shap(
            cluster_data, model_feature_cols, sample_size, random_state=42,
        )
        if X_sample is None:
            # Too few non-zero rows — skip SHAP for this cluster
            logger.info(
                "[shap] Skipping SHAP for cluster '%s' (too few non-zero rows).",
                cluster_label,
            )
            continue

        n = len(X_sample)
        # Ensure categorical columns have category dtype to match the model's
        # training data (required by LightGBM's _data_from_pandas validation).
        for col in model_cat_cols:
            if col in X_sample.columns:
                X_sample[col] = X_sample[col].astype("category")
        try:
            abs_shap = shap_extractor_fn(model, X_sample, model_feature_cols, model_cat_cols)
            # Slice to effective features if we passed extra columns to the model
            if effective_indices is not None:
                abs_shap = abs_shap[:, effective_indices]
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
    min_features: int = 20,
    top_n: int | None = None,
    excluded_features: set[str] | None = None,
    label: str = "pooled",
) -> tuple[list[str], pd.DataFrame]:
    """Select features by cumulative SHAP importance and build the report DataFrame.

    Two modes:
    - top_n is set: keep exactly max(top_n, min_features) features.
    - top_n is None: keep features covering cumulative_threshold of total SHAP mass,
      with min_features as a lower bound.

    Features in ``excluded_features`` (removed by pre-SHAP stages) are never
    selected regardless of SHAP rank. They appear in the report with
    ``selected=False`` and their ``exclusion_stage`` set.

    Returns (selected_feature_names, shap_report_df).
    """
    excluded = excluded_features or set()
    total_shap = float(mean_abs_shap.sum())
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    sorted_shap = mean_abs_shap[sorted_indices]
    sorted_features = [feature_cols[i] for i in sorted_indices]

    # Only eligible (non-excluded) features participate in selection
    eligible = [f for f in sorted_features if f not in excluded]

    if top_n is not None:
        n_select = max(min_features, min(int(top_n), len(eligible)))
    elif total_shap == 0:
        n_select = len(eligible)
    else:
        # Cumulative threshold over eligible features only.
        # Map name -> SHAP column index once. feature_cols MUST have unique names
        # or list.index() / this lookup silently picks the wrong SHAP mass and
        # selects the wrong features — fail loud if a duplicate is ever introduced.
        if len(set(feature_cols)) != len(feature_cols):
            raise ValueError(
                "feature_cols contains duplicate names; SHAP selection requires "
                "unique feature names."
            )
        col_idx = {name: i for i, name in enumerate(feature_cols)}
        eligible_shap = [mean_abs_shap[col_idx[f]] for f in eligible]
        eligible_total = sum(eligible_shap)
        if eligible_total > 0:
            cumsum = np.cumsum(eligible_shap) / eligible_total
            n_select = int(np.searchsorted(cumsum, cumulative_threshold)) + 1
            n_select = max(min_features, min(n_select, len(eligible)))
        else:
            n_select = len(eligible)

    selected_set = set(eligible[:n_select])

    # Always keep protected features (month, quarter, fourier terms)
    for feat in feature_cols:
        if feat in PROTECTED_FEATURES and feat not in excluded:
            selected_set.add(feat)

    shap_df = pd.DataFrame({
        "feature": sorted_features,
        "mean_abs_shap": sorted_shap.tolist(),
        "rank": list(range(1, len(sorted_features) + 1)),
        "selected": [f in selected_set for f in sorted_features],
        "timeframe": [timeframe_idx] * len(sorted_features),
        "cutoff_date": [str(cutoff_date.date())] * len(sorted_features),
    })

    # Return selected features in SHAP-rank order
    result_features = [f for f in sorted_features if f in selected_set]
    logger.info(
        "[shap] Selected %d/%d features [%s] (threshold=%.2f, excluded_pre_shap=%d, top: %s)",
        len(result_features),
        len(feature_cols),
        label,
        cumulative_threshold,
        len(excluded),
        sorted_features[0] if sorted_features else "none",
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
    min_features: int = 20,
    # Pre-SHAP pipeline stages
    correlation_filter: bool = False,
    correlation_threshold: float = 0.95,
    variance_filter: bool = False,
    variance_threshold: float = 0.01,
) -> tuple[list[str], pd.DataFrame]:
    """Multi-stage feature selection for one backtest timeframe.

    Pipeline:
      Stage 0: Remove exact-duplicate aliases (static)
      Stage 1: Near-zero variance filter (optional, per-timeframe)
      Stage 2: Correlation pre-filter (optional, per-timeframe)
      Stage 3: SHAP cumulative-importance selection (on reduced set)

    Handles both single-model (global strategy) and dict-of-models
    (per_cluster / transfer strategies).

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
        correlation_filter: Enable Stage 2 correlation pre-filter.
        correlation_threshold: Correlation threshold for Stage 2 (default 0.95).
        variance_filter: Enable Stage 1 near-zero variance filter.
        variance_threshold: Relative variance threshold for Stage 1 (default 0.01).

    Returns:
        (selected_features, shap_df) where shap_df has SHAP_REPORT_COLS columns.
        When per-cluster SHAP data is available, shap_df also includes a
        'cluster' column (value "all" for pooled rows, cluster label for per-cluster rows).
    """
    t0 = time.time()

    # ── Pre-SHAP feature reduction pipeline ────────────────────────────
    # These stages determine which features are excluded from selection.
    # SHAP is still computed on the full feature set (matching the trained
    # model), but excluded features are masked out of the selection pool.
    pre_shap_excluded: set[str] = set()

    # Stage 0: Static duplicate removal
    _, s0_excluded = _remove_duplicate_features(feature_cols)
    pre_shap_excluded |= s0_excluded

    # Stage 1: Near-zero variance filter (per-timeframe, causal)
    if variance_filter:
        eligible_for_s1 = [f for f in feature_cols if f not in pre_shap_excluded]
        _, s1_excluded = _remove_low_variance_features(
            train_data, eligible_for_s1, cat_cols, variance_threshold,
        )
        pre_shap_excluded |= s1_excluded

    # Stage 2: Correlation-based pre-filter (per-timeframe, causal)
    if correlation_filter:
        eligible_for_s2 = [f for f in feature_cols if f not in pre_shap_excluded]
        _, s2_excluded = _remove_correlated_features(
            train_data, eligible_for_s2, cat_cols, correlation_threshold,
        )
        pre_shap_excluded |= s2_excluded

    if pre_shap_excluded:
        logger.info(
            "[feat-select] Pre-SHAP reduction: %d → %d features (excluded %d)",
            len(feature_cols),
            len(feature_cols) - len(pre_shap_excluded),
            len(pre_shap_excluded),
        )

    # ── SHAP computation ───────────────────────────────────────────────
    # SHAP must use the full feature set matching the trained model.
    # The pre-SHAP excluded set is passed to _select_features_from_shap
    # to mask them out of the cumulative selection pool.
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
                all_feature_cols=feature_cols,
                all_cat_cols=cat_cols,
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
        excluded_features=pre_shap_excluded,
        label="pooled",
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
                excluded_features=pre_shap_excluded,
                label=f"cluster={cluster_label}",
            )
            cluster_df["cluster"] = cluster_label
            cluster_dfs.append(cluster_df)
        combined_df = pd.concat([pooled_df, *cluster_dfs], ignore_index=True)
    else:
        combined_df = pooled_df

    return selected, combined_df


def compute_timeframe_shap_per_cluster(
    model_dict: dict[str, Any],
    train_data: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    timeframe_idx: int,
    cutoff_date: pd.Timestamp,
    shap_extractor_fn: ShapExtractorFn,
    sample_size: int = 500,
    cumulative_threshold: float = 0.95,
    top_n: int | None = None,
    min_features: int = 20,
    correlation_filter: bool = False,
    correlation_threshold: float = 0.95,
    variance_filter: bool = False,
    variance_threshold: float = 0.01,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Per-cluster independent feature selection via SHAP importance.

    Unlike ``compute_timeframe_shap`` which pools SHAP across clusters into a
    single weighted average and returns ONE shared feature list, this function
    runs SHAP selection *independently* for each cluster and returns a dict of
    per-cluster selected feature lists.  Each cluster gets its own optimal
    feature subset, allowing cluster-specific models to focus on the features
    most relevant to their data distribution.

    Pre-SHAP stages (0-2) are shared across all clusters — the same duplicate,
    variance, and correlation exclusions apply globally.

    Args:
        model_dict: Dict mapping cluster_label → trained model.
            The ``"__base__"`` key (used by transfer-learning) is skipped.
        train_data: Causally masked training DataFrame for this timeframe.
            Must contain an ``ml_cluster`` column for cluster partitioning.
        feature_cols: Full feature column list from the feature matrix.
        cat_cols: Categorical feature names.
        timeframe_idx: 0-based timeframe index (0=A, 1=B, …).
        cutoff_date: Training cutoff date for this timeframe.
        shap_extractor_fn: Model-specific SHAP extractor callable.
        sample_size: Max rows to sample per cluster for SHAP computation.
        cumulative_threshold: Cumulative SHAP mass threshold for feature
            selection (default 0.95).
        top_n: If set, select exactly this many features (overrides threshold).
        min_features: Minimum number of features to keep per cluster.
        correlation_filter: Enable Stage 2 correlation pre-filter.
        correlation_threshold: Correlation threshold for Stage 2 (default 0.95).
        variance_filter: Enable Stage 1 near-zero variance filter.
        variance_threshold: Relative variance threshold for Stage 1.

    Returns:
        (per_cluster_features, combined_shap_df) where:
        - per_cluster_features: ``{cluster_label: [selected_features]}`` — each
          cluster has its own independently selected feature list.
        - combined_shap_df: DataFrame with SHAP_REPORT_COLS + ``cluster``
          column, containing per-cluster SHAP reports concatenated.
    """
    t0 = time.time()

    # ── Pre-SHAP feature reduction pipeline (shared across clusters) ──
    pre_shap_excluded: set[str] = set()

    # Stage 0: Static duplicate removal
    _, s0_excluded = _remove_duplicate_features(feature_cols)
    pre_shap_excluded |= s0_excluded

    # Stage 1: Near-zero variance filter (per-timeframe, causal)
    if variance_filter:
        eligible_for_s1 = [f for f in feature_cols if f not in pre_shap_excluded]
        _, s1_excluded = _remove_low_variance_features(
            train_data, eligible_for_s1, cat_cols, variance_threshold,
        )
        pre_shap_excluded |= s1_excluded

    # Stage 2: Correlation-based pre-filter (per-timeframe, causal)
    if correlation_filter:
        eligible_for_s2 = [f for f in feature_cols if f not in pre_shap_excluded]
        _, s2_excluded = _remove_correlated_features(
            train_data, eligible_for_s2, cat_cols, correlation_threshold,
        )
        pre_shap_excluded |= s2_excluded

    if pre_shap_excluded:
        logger.info(
            "[feat-select] Pre-SHAP reduction: %d → %d features (excluded %d)",
            len(feature_cols),
            len(feature_cols) - len(pre_shap_excluded),
            len(pre_shap_excluded),
        )

    # ── Per-cluster SHAP computation and selection ────────────────────
    # Use full feature_cols for SHAP extraction to match the trained model.
    # The model may have been trained with features (e.g. ml_cluster) that
    # are not in the effective selection pool — we must still call SHAP with
    # the full set to avoid LightGBM shape mismatches.
    model_feature_cols = feature_cols
    model_cat_cols = cat_cols

    per_cluster_features: dict[str, list[str]] = {}
    cluster_dfs: list[pd.DataFrame] = []

    for cluster_label, model in model_dict.items():
        if cluster_label == "__base__":
            continue

        cluster_data = train_data[train_data["ml_cluster"] == cluster_label]
        if len(cluster_data) == 0:
            logger.warning(
                "[shap] No training data for cluster '%s'; keeping all features.",
                cluster_label,
            )
            per_cluster_features[str(cluster_label)] = [
                f for f in feature_cols if f not in pre_shap_excluded
            ]
            continue

        # Use stratified sampling for sparse clusters; returns None if
        # too few non-zero rows for reliable SHAP analysis.
        X_sample = _stratified_sample_for_shap(
            cluster_data, model_feature_cols, sample_size, random_state=42,
        )
        if X_sample is None:
            # Too few non-zero rows — SHAP is unreliable, keep all features
            logger.info(
                "[shap] Cluster '%s': too few non-zero rows for SHAP; keeping all features.",
                cluster_label,
            )
            per_cluster_features[str(cluster_label)] = [
                f for f in feature_cols if f not in pre_shap_excluded
            ]
            continue

        # Ensure categorical columns have category dtype to match the model's
        # training data (required by LightGBM's _data_from_pandas validation).
        for col in model_cat_cols:
            if col in X_sample.columns:
                X_sample[col] = X_sample[col].astype("category")

        try:
            abs_shap = shap_extractor_fn(model, X_sample, model_feature_cols, model_cat_cols)
            mean_abs_shap = abs_shap.mean(axis=0)

            selected, cluster_df = _select_features_from_shap(
                mean_abs_shap,
                feature_cols,
                timeframe_idx,
                cutoff_date,
                cumulative_threshold=cumulative_threshold,
                min_features=min_features,
                top_n=top_n,
                excluded_features=pre_shap_excluded,
                label=f"cluster={cluster_label}",
            )
            per_cluster_features[str(cluster_label)] = selected
            cluster_df["cluster"] = str(cluster_label)
            cluster_dfs.append(cluster_df)

        except Exception as exc:
            logger.warning(
                "[shap] SHAP extraction failed for cluster '%s': %s. Keeping all features.",
                cluster_label,
                exc,
            )
            per_cluster_features[str(cluster_label)] = [
                f for f in feature_cols if f not in pre_shap_excluded
            ]

    # Concatenate all per-cluster reports
    if cluster_dfs:
        combined_shap_df = pd.concat(cluster_dfs, ignore_index=True)
    else:
        combined_shap_df = pd.DataFrame(columns=[*SHAP_REPORT_COLS])

    elapsed = time.time() - t0
    logger.info(
        "[shap] Per-cluster SHAP computed (%.1fs), %d clusters",
        elapsed,
        len(per_cluster_features),
    )
    return per_cluster_features, combined_shap_df


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
