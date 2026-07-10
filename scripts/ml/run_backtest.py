"""
Run tree-model backtesting with per-cluster strategy and expanding-window timeframes.

Supports the LightGBM member of the lite forecasting roster.
All run options (recursive, SHAP, tuning) are controlled via
config/forecasting/forecast_pipeline_config.yaml rather than CLI flags.

Produces two CSVs under data/backtest/<model_id>/:
  - backtest_predictions.csv          (execution-lag row for DB load)
  - backtest_predictions_all_lags.csv (lag 0-4 archive)
"""

import importlib
import json
import logging
import os
import pickle
import platform
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.constants import FORECAST_QTY_COL, METADATA_COLS, compute_min_cluster_rows
from common.core.utils import get_algorithm_roster, load_forecast_pipeline_config
from common.ml.backtest_framework import (
    compute_cluster_demand_stats,
    resolve_cluster_params,
    run_tree_backtest,
)
from common.ml.model_registry import (
    build_tree_model,
    fit_final_model,
    fit_model,
    get_best_iteration,
    get_tree_default_params,
    probe_tree_gpu_available,
)
from common.ml.tuning import TRAIN_FOLD_FNS, load_best_params, tune_for_timeframe
from common.scripts_base import load_project_env, setup_logging
from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)


def _model_feature_cols(feature_cols: list[str]) -> list[str]:
    """Return model input columns with metadata/target columns stripped."""
    return [col for col in feature_cols if col not in METADATA_COLS]


def _model_cat_cols(cat_cols: list[str], feature_cols: list[str]) -> list[str]:
    """Return categorical columns that are valid model inputs."""
    feature_set = set(feature_cols)
    return [col for col in cat_cols if col in feature_set and col not in METADATA_COLS]


# ── Demand pattern classification for Tweedie routing ─────────────────────────

# Objective override maps per model for Tweedie loss
_TWEEDIE_OBJECTIVE: dict[str, dict[str, object]] = {
    "lgbm": {"objective": "tweedie"},
}


def _classify_cluster_demand(
    train_c: pd.DataFrame,
    *,
    intermittent_threshold: float = 0.5,
    lumpy_threshold: float = 0.3,
) -> str:
    """Classify a cluster's demand pattern based on zero-demand percentage.

    Args:
        train_c: Training data for one cluster.
        intermittent_threshold: Zero-demand fraction above which the cluster
            is classified as intermittent.  Default 0.5 (from YAML config).
        lumpy_threshold: Zero-demand fraction above which (but below
            intermittent_threshold) the cluster is classified as lumpy.

    Returns:
        One of ``"intermittent"``, ``"lumpy"``, or ``"continuous"``.
    """
    if "qty" not in train_c.columns or len(train_c) == 0:
        return "continuous"
    zero_pct = (train_c["qty"] == 0).mean()
    if zero_pct >= intermittent_threshold:
        return "intermittent"
    if zero_pct > lumpy_threshold:
        return "lumpy"
    return "continuous"


def _apply_tweedie_objective(
    params: dict[str, object],
    model_name: str,
    demand_pattern: str,
    tweedie_variance_power: float = 1.5,
) -> dict[str, object]:
    """Return a copy of *params* with objective adjusted for non-continuous demand.

    For continuous patterns the original params are returned unchanged.
    For intermittent patterns (>= intermittent_threshold zeros, typically 70%+),
    Tweedie is NOT applied — MAE (regression_l1) is used instead.  Tweedie's log
    link function produces reasonable predictions at iteration 0 for highly sparse
    data, causing WAPE-based early stopping to fire at iter 1 before the model
    learns any signal.  MAE is robust to zero-inflation and lets the model train
    for many rounds, dramatically improving accuracy on sparse clusters.

    For lumpy patterns (moderate intermittency, 30-70% zeros), the original
    params are returned unchanged (uses default MAE/RMSE from config).
    """
    if demand_pattern == "continuous":
        return params

    if demand_pattern == "lumpy":
        # Lumpy demand (30-70% zeros): keep default objective (MAE/RMSE).
        # Tweedie doesn't help here either — default loss works well enough.
        return params

    # Intermittent demand (>70% zeros): force MAE objective.
    # Tweedie is catastrophic for very sparse data (best_iter=1, negative accuracy).
    # MAE (regression_l1) is robust for zero-inflated time series.
    overrides: dict[str, object] = {}
    if model_name == "lgbm":
        overrides["objective"] = "regression_l1"
        # Remove any residual Tweedie params
        merged = {**params, **overrides}
        merged.pop("tweedie_variance_power", None)
    else:
        merged = {**params, **overrides}

    return merged


# ── Model Registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "lgbm": {
        "class": "lightgbm.LGBMRegressor",
        "config_key": "lgbm",
        "config_section": "lgbm_cluster",
        "iter_param": "n_estimators",
        "gpu_params": lambda: {"device": "gpu"},
        "gpu_test_platform_check": True,  # only auto-detect on Darwin
        "fit_extras_per_cluster": lambda params, iter_param: {},
        "fit_extras_global": lambda params, iter_param: {},
        "default_params": lambda algo, seed=42: get_tree_default_params("lgbm", algo, seed=seed),
        "cat_dtype": "category",
        "model_params_key": "lgbm_params",
        "model_type_tag": "lgbm_backtest",
        "shap_extractor": "compute_shap_global",
        "best_iteration_attr": "best_iteration_",
        "feature_importance_fn": lambda model: model.feature_importances_,
        "constant_target_guard": True,
        "needs_cat_indices": False,
        "needs_cat_dtype_cast": False,
    },
}




def _import_model_class(dotted_path: str) -> type:
    """Dynamically import a model class from a dotted module.class path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ── Single-cluster training worker (used by both sequential and parallel) ─────


def _train_single_cluster(
    cluster_label: str,
    ci: int,
    n_clusters: int,
    train_c: pd.DataFrame,
    pred_c: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict,
    *,
    model_name: str,
    model_class: type | str,
    lib_module: Any | str,
    needs_cat_dtype_cast: bool,
    constant_target_guard: bool,
    iter_param: str,
    fit_extras: dict | None = None,
) -> tuple[str, pd.DataFrame | None, Any | None, dict | str | None]:
    """Train a single cluster model. Returns (cluster_label, result_df, model, meta).

    meta is a dict with training metadata, "fallback_needed" for small clusters,
    or None if the cluster was skipped entirely.

    This function is self-contained with no shared mutable state, making it safe
    for use in ProcessPoolExecutor.  Accepts only picklable arguments (no lambdas
    or registry dicts).  Tree estimator construction is delegated to
    model_registry.build_tree_model; lib_module can be passed as a dotted module
    string for fit-time callback support.
    """
    if isinstance(lib_module, str):
        lib_module = importlib.import_module(lib_module)

    feature_cols = _model_feature_cols(feature_cols)
    cat_cols = _model_cat_cols(cat_cols, feature_cols)
    cat_cols_in_features = (
        [c for c in cat_cols if c in feature_cols] if needs_cat_dtype_cast else []
    )

    min_rows = compute_min_cluster_rows(len(feature_cols))
    if len(train_c) < min_rows or len(pred_c) == 0:
        if len(pred_c) > 0:
            logger.info(
                "Cluster %d/%d '%s': skipped (train=%d < %d), marking %d predictions for fallback",
                ci,
                n_clusters,
                cluster_label,
                len(train_c),
                min_rows,
                len(pred_c),
            )
            result = pred_c[["sku_ck", "item_id", "customer_group", "loc", "startdate"]].copy()
            result[FORECAST_QTY_COL] = 0.0  # placeholder — overwritten by fallback
            return cluster_label, result, None, "fallback_needed"
        return cluster_label, None, None, None

    # Sort by (startdate, sku_ck) — date-primary for time ordering, sku_ck
    # as tiebreaker for reproducible row ordering when two DFUs share a date.
    train_c = train_c.sort_values(["startdate", "sku_ck"])

    X_train = train_c[feature_cols].copy() if needs_cat_dtype_cast else train_c[feature_cols]
    y_train = train_c["qty"]
    X_pred = pred_c[feature_cols].copy() if needs_cat_dtype_cast else pred_c[feature_cols]

    if needs_cat_dtype_cast:
        for col in cat_cols_in_features:
            X_train[col] = X_train[col].astype("category")
            X_pred[col] = X_pred[col].astype("category")

    t0 = time.time()
    # Calendar-month-based train/val split — last 20% of unique months as validation.
    # More robust than row-count split for sparse/cold-start DFUs: a DFU with only
    # 6 months of history would otherwise have its last few rows (not latest calendar
    # months) used for validation, which can contaminate the validation window.
    _unique_months = sorted(train_c["startdate"].unique())
    _n_val_months = max(1, int(len(_unique_months) * 0.20))
    _val_months = set(_unique_months[-_n_val_months:])
    _val_mask = train_c["startdate"].isin(_val_months)
    X_tr, X_val = X_train.loc[~_val_mask], X_train.loc[_val_mask]
    y_tr, y_val = y_train.loc[~_val_mask], y_train.loc[_val_mask]

    # Guard: some models crash on constant targets
    if constant_target_guard and y_tr.nunique() <= 1:
        const_val = float(y_tr.iloc[0]) if len(y_tr) > 0 else 0.0
        logger.info(
            "Cluster %d/%d '%s': skipped (constant target=%.0f), using constant for %d predictions",
            ci,
            n_clusters,
            cluster_label,
            const_val,
            len(pred_c),
        )
        result = pred_c[["sku_ck", "item_id", "customer_group", "loc", "startdate"]].copy()
        result[FORECAST_QTY_COL] = const_val
        return cluster_label, result, None, None

    # Per-cluster adaptive hyperparameter profiles: resolve cluster-specific
    # overrides based on demand characteristics (sparse, volatile, stable, etc.)
    cluster_stats = compute_cluster_demand_stats(train_c, cluster_label)
    resolved_params, profile_name = resolve_cluster_params(
        cluster_label,
        cluster_stats,
        params,
    )
    if profile_name not in ("none", "default"):
        logger.info(
            "Cluster %d/%d '%s': matched profile '%s' (mean_demand=%.1f, cv=%.2f, "
            "zero_pct=%.2f, seasonal_amp=%.2f)",
            ci,
            n_clusters,
            cluster_label,
            profile_name,
            cluster_stats["mean_demand"],
            cluster_stats["cv_demand"],
            cluster_stats["zero_demand_pct"],
            cluster_stats["seasonal_amplitude"],
        )

    # Filter resolved params to only include keys valid for this model
    # (cluster profiles may inject LGBM-specific keys like reg_alpha, num_leaves)
    valid_keys = set(params.keys())
    filtered_params = {k: v for k, v in resolved_params.items() if k in valid_keys}
    fit_params = {**filtered_params, **(fit_extras or {})}

    # Classify demand pattern and apply Tweedie objective for intermittent clusters
    pcfg = load_forecast_pipeline_config()
    backtest_cfg = pcfg.get("backtest", {})
    # Fallback matches config default (forecast_pipeline_config.yaml: 0.7) and the
    # documented ">70% zero-demand → intermittent baseline" routing.
    intermittent_threshold = backtest_cfg.get("intermittent_threshold", 0.7)
    lumpy_threshold = backtest_cfg.get("lumpy_threshold", 0.3)
    tweedie_vp = backtest_cfg.get("tweedie_variance_power", 1.5)

    demand_pattern = _classify_cluster_demand(
        train_c,
        intermittent_threshold=intermittent_threshold,
        lumpy_threshold=lumpy_threshold,
    )
    fit_params = _apply_tweedie_objective(fit_params, model_name, demand_pattern, tweedie_vp)
    if demand_pattern != "continuous":
        logger.info(
            "Cluster %d/%d '%s': demand_pattern=%s (zero_pct=%.2f), objective=%s",
            ci,
            n_clusters,
            cluster_label,
            demand_pattern,
            cluster_stats["zero_demand_pct"],
            fit_params.get("objective", fit_params.get("loss_function", "default")),
        )


    # In parallel mode, cap per-model thread count to avoid oversubscription.
    # With 8 workers each requesting all cores, thread contention degrades perf.
    thread_key = "thread_count" if "thread_count" in fit_params else "n_jobs"
    if fit_params.get(thread_key) == -1 and n_clusters > 4:
        n_cpus = os.cpu_count() or 8
        fit_params[thread_key] = max(2, n_cpus // 4)

    max_iters = fit_params[iter_param]
    model = build_tree_model(model_name, fit_params)

    # Unified fit call — all model-specific logic in model_registry.fit_model()
    fit_model(
        model,
        model_name,
        X_tr,
        y_tr,
        X_val,
        y_val,
        cat_cols,
        feature_cols,
        lib_module,
        max_iters,
        demand_pattern=demand_pattern,
    )

    # Per-cluster validation WAPE — use scaled floor for sparse clusters to avoid
    # division instability when val actuals sum near zero
    val_preds = model.predict(X_val)
    val_abs_sum = float(abs(y_val.sum()))
    val_floor = max(len(y_val) * 0.01, 1.0)
    val_denom = max(val_abs_sum, val_floor)
    val_wape = round(float((abs(val_preds - y_val.values)).sum() / val_denom * 100), 2)

    n_est_used = get_best_iteration(model, model_name)
    if n_est_used is None:
        n_est_used = fit_params[iter_param]

    # Refit on every row available at this backtest cutoff before predicting.
    # The split model is only for early-stopping and validation telemetry; using
    # it for forecasts discards the newest validation months from training.
    final_params = dict(fit_params)
    final_params[iter_param] = n_est_used
    final_model = build_tree_model(model_name, final_params)
    fit_final_model(final_model, model_name, X_train, y_train, cat_cols, feature_cols)
    preds = final_model.predict(X_pred)

    result = pred_c[["sku_ck", "item_id", "customer_group", "loc", "startdate"]].copy()
    # Always clip predictions to non-negative (MAE/RMSE can produce negatives
    # and negative forecasts are nonsensical)
    result[FORECAST_QTY_COL] = np.maximum(preds, 0)

    # val_wape and train_rows are consumed by persist_cluster_models().
    # cluster_profile, demand_pattern, and cluster_stats are diagnostic — they
    # ride along in model_meta for inspection/logging but are not read back
    # by any downstream function.
    meta = {
        "val_wape": val_wape,
        "train_rows": len(X_train),
        "early_stop_train_rows": len(X_tr),
        "val_rows": len(X_val),
        "n_estimators_used": n_est_used,
        "cluster_profile": profile_name,
        "demand_pattern": demand_pattern,
        "cluster_stats": cluster_stats,
    }
    val_accuracy = round(100.0 - val_wape, 2)
    logger.info(
        "Cluster %d/%d '%s': train=%s, pred=%s, best_iter=%s, "
        "val_accuracy=%.1f%%, val_wape=%.1f%%, profile=%s, pattern=%s (%.1fs)",
        ci,
        n_clusters,
        cluster_label,
        f"{len(train_c):,}",
        f"{len(pred_c):,}",
        n_est_used,
        val_accuracy,
        val_wape,
        profile_name,
        demand_pattern,
        time.time() - t0,
    )

    return cluster_label, result, final_model, meta


# ── Naive fallback for small clusters ─────────────────────────────────────────


def _compute_naive_fallback(
    train_c: pd.DataFrame,
    pred_c: pd.DataFrame,
) -> pd.DataFrame:
    """Compute seasonal naive baseline for small clusters that cannot train a model.

    For each prediction row, uses the historical mean demand for the same
    calendar month from the training data.  If no matching month exists in
    training history, falls back to the overall mean demand across all months.
    The result is always >= 0 (legitimate zero demand is preserved).

    Returns a DataFrame with columns:
        sku_ck, item_id, customer_group, loc, startdate, basefcst_pref
    """
    result = pred_c[["sku_ck", "item_id", "customer_group", "loc", "startdate"]].copy()

    if len(train_c) == 0 or "qty" not in train_c.columns:
        result[FORECAST_QTY_COL] = 0.0
        return result

    # Extract calendar month from startdate for both train and predict
    train_months = pd.to_datetime(train_c["startdate"]).dt.month
    pred_months = pd.to_datetime(pred_c["startdate"]).dt.month

    # Compute per-month historical mean demand
    month_means = train_c.assign(_month=train_months).groupby("_month")["qty"].mean()

    # Overall mean as fallback for months with no history
    overall_mean = float(train_c["qty"].mean())

    # Map each prediction row to its monthly mean (or overall mean)
    fallback_values = pred_months.map(month_means).fillna(overall_mean).values
    result[FORECAST_QTY_COL] = np.maximum(fallback_values, 0.0)

    return result


# ── Per-cluster training function ─────────────────────────────────────────────


def train_and_predict_per_cluster(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict,
    *,
    model_name: str,
    registry: dict[str, Any],
    model_class: type,
    lib_module: Any,
    parallel: bool = False,
    max_workers: int = 4,
    per_cluster_feature_cols: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict, dict[str, dict]]:
    """Train separate tree models per ml_cluster.

    ml_cluster is a metadata column (in METADATA_COLS) — used to partition DFUs
    into per-cluster models, but excluded from feature_cols (not a model feature).

    When ``parallel=True`` and there are >4 clusters, uses ProcessPoolExecutor
    to train clusters concurrently.  Each cluster's training is independent
    (no shared mutable state).

    When ``per_cluster_feature_cols`` is provided, each cluster uses its own
    SHAP-selected feature list instead of the shared ``feature_cols``.

    Returns (predictions, models, model_meta) where model_meta stores per-cluster
    training metadata (val_wape, train_rows).
    """
    all_results: list[pd.DataFrame] = []
    models: dict = {}
    model_meta: dict[str, dict] = {}
    fallback_clusters: list[str] = []  # clusters needing naive fallback

    clusters = sorted(train_df["ml_cluster"].dropna().unique())
    label = model_name.upper()
    n_clusters = len(clusters)
    logger.info("Training %d per-cluster %s models...", n_clusters, label)

    # For parallel mode, pass model_class and lib_module as importable strings
    # so they can be pickled by ProcessPoolExecutor.  Sequential mode passes
    # the objects directly (avoids re-import overhead).
    _model_class_ref: type | str = model_class
    _lib_module_ref: Any | str = lib_module
    if parallel:
        _model_class_ref = registry["class"]  # e.g. "lightgbm.LGBMRegressor"
        _lib_module_ref = registry["class"].rsplit(".", 1)[0]  # e.g. "lightgbm"

    _worker_kwargs = {
        "model_name": model_name,
        "model_class": _model_class_ref,
        "lib_module": _lib_module_ref,
        "needs_cat_dtype_cast": registry["needs_cat_dtype_cast"],
        "constant_target_guard": registry["constant_target_guard"],
        "iter_param": registry["iter_param"],
        "fit_extras": registry["fit_extras_per_cluster"](params, registry["iter_param"]),
    }

    def _collect_result(
        cl: str, result: pd.DataFrame | None, model: Any, meta: dict | str | None
    ) -> None:
        """Collect training result, separating fallback-needed clusters."""
        if meta == "fallback_needed":
            fallback_clusters.append(cl)
            # Don't append the placeholder result — we'll recompute via naive fallback
        elif result is not None:
            all_results.append(result)
        if model is not None:
            models[cl] = model
        if isinstance(meta, dict):
            model_meta[cl] = meta

    use_parallel = parallel and n_clusters > 4
    if use_parallel:
        logger.info(
            "Parallel cluster training enabled: %d workers for %d clusters", max_workers, n_clusters
        )
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for ci, cluster_label in enumerate(clusters, 1):
                train_c = train_df[train_df["ml_cluster"] == cluster_label]
                pred_c = predict_df[predict_df["ml_cluster"] == cluster_label]
                cluster_feature_cols = (
                    per_cluster_feature_cols.get(cluster_label, feature_cols)
                    if per_cluster_feature_cols
                    else feature_cols
                )
                cluster_cat_cols = (
                    [c for c in cat_cols if c in cluster_feature_cols]
                    if per_cluster_feature_cols
                    else cat_cols
                )
                future = executor.submit(
                    _train_single_cluster,
                    cluster_label,
                    ci,
                    n_clusters,
                    train_c,
                    pred_c,
                    cluster_feature_cols,
                    cluster_cat_cols,
                    params,
                    **_worker_kwargs,
                )
                futures[future] = cluster_label

            for future in as_completed(futures):
                cl, result, model, meta = future.result()
                _collect_result(cl, result, model, meta)
    else:
        if parallel:
            logger.info(
                "Parallel mode requested but only %d clusters (<= 4), using sequential", n_clusters
            )
        for ci, cluster_label in enumerate(clusters, 1):
            train_c = train_df[train_df["ml_cluster"] == cluster_label]
            pred_c = predict_df[predict_df["ml_cluster"] == cluster_label]
            cluster_feature_cols = (
                per_cluster_feature_cols.get(cluster_label, feature_cols)
                if per_cluster_feature_cols
                else feature_cols
            )
            cluster_cat_cols = (
                [c for c in cat_cols if c in cluster_feature_cols]
                if per_cluster_feature_cols
                else cat_cols
            )
            cl, result, model, meta = _train_single_cluster(
                cluster_label,
                ci,
                n_clusters,
                train_c,
                pred_c,
                cluster_feature_cols,
                cluster_cat_cols,
                params,
                **_worker_kwargs,
            )
            _collect_result(cl, result, model, meta)

    # Apply naive fallback for small clusters
    if fallback_clusters:
        logger.info(
            "Computing naive fallback for %d small cluster(s): %s",
            len(fallback_clusters),
            fallback_clusters,
        )
        for cl in fallback_clusters:
            train_c = train_df[train_df["ml_cluster"] == cl]
            pred_c = predict_df[predict_df["ml_cluster"] == cl]
            if len(pred_c) > 0:
                fb_result = _compute_naive_fallback(train_c, pred_c)
                all_results.append(fb_result)
                logger.info(
                    "Cluster '%s': naive fallback applied to %d predictions "
                    "(mean basefcst_pref=%.2f)",
                    cl,
                    len(fb_result),
                    float(fb_result[FORECAST_QTY_COL].mean()),
                )

    no_cluster = predict_df[
        predict_df["ml_cluster"].isna()
        | ((predict_df["ml_cluster"] == "__unknown__") & ("__unknown__" not in models))
    ]
    if len(no_cluster) > 0:
        logger.info("%d predict rows with no cluster -> zeroing", len(no_cluster))
        result = no_cluster[["sku_ck", "item_id", "customer_group", "loc", "startdate"]].copy()
        result[FORECAST_QTY_COL] = 0.0
        all_results.append(result)

    return pd.concat(all_results, ignore_index=True), models, model_meta


# ── Global training function ──────────────────────────────────────────────────


def train_and_predict_global(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict,
    *,
    model_name: str,
    registry: dict[str, Any],
    model_class: type,
    lib_module: Any,
) -> tuple[pd.DataFrame, dict, dict[str, dict]]:
    """Train a single global tree model on all data without metadata features."""
    feature_cols = _model_feature_cols(feature_cols)
    cat_cols = _model_cat_cols(cat_cols, feature_cols)
    iter_param = registry["iter_param"]
    label = model_name.upper()

    logger.info(
        "Training global %s on %s rows, %d model features...",
        label,
        f"{len(train_df):,}",
        len(feature_cols),
    )

    # Sort by startdate so last 15% = most recent months (not last DFUs alphabetically)
    sorted_idx = train_df["startdate"].argsort(kind="mergesort")
    train_sorted = train_df.iloc[sorted_idx]
    X_train = train_sorted[feature_cols].copy()
    y_train = train_sorted["qty"]
    X_pred = predict_df[feature_cols].copy()

    # Ensure all categorical columns have category dtype (LGBM requires this)
    for col in cat_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
            X_pred[col] = X_pred[col].astype("category")

    # Time-aware train/val split — last 15% of rows for validation (most recent months)
    n_val = max(1, int(len(X_train) * 0.15))
    X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

    fit_params = {**params, **registry["fit_extras_global"](params, iter_param)}
    max_iters = fit_params[iter_param]
    model = build_tree_model(model_name, fit_params)

    # Unified fit call — all model-specific logic in model_registry.fit_model()
    fit_model(
        model, model_name, X_tr, y_tr, X_val, y_val, cat_cols, feature_cols, lib_module, max_iters
    )

    val_preds = model.predict(X_val)
    val_denom = float(abs(y_val.sum()))
    val_wape = (
        round(float((abs(val_preds - y_val.values)).sum() / val_denom * 100), 2)
        if val_denom > 0
        else 0.0
    )
    n_est_used = get_best_iteration(model, model_name)
    if not n_est_used:
        n_est_used = fit_params[iter_param]

    final_params = dict(fit_params)
    final_params[iter_param] = n_est_used
    final_model = build_tree_model(model_name, final_params)
    fit_final_model(final_model, model_name, X_train, y_train, cat_cols, feature_cols)
    preds = final_model.predict(X_pred)

    logger.info(
        "Global %s: val_WAPE=%.1f%%, best_iter=%s, train=%s, pred=%s",
        label,
        val_wape,
        n_est_used,
        f"{len(train_df):,}",
        f"{len(predict_df):,}",
    )

    result = predict_df[["sku_ck", "item_id", "customer_group", "loc", "startdate"]].copy()
    result[FORECAST_QTY_COL] = np.clip(preds, 0, None)
    global_meta = {
        "global": {
            "val_wape": val_wape,
            "train_rows": len(X_train),
            "early_stop_train_rows": len(X_tr),
            "val_rows": len(X_val),
            "n_estimators_used": n_est_used,
        }
    }
    return result, {"global": final_model}, global_meta


# ── Model persistence ─────────────────────────────────────────────────────────


def persist_cluster_models(
    models: dict,
    feature_cols: list[str] | dict[str, list[str]],
    model_id: str,
    timeframe_label: str,
    prod_config: dict | None = None,
    model_meta: dict[str, dict] | None = None,
    *,
    feature_importance_fn: Callable | None = None,
    model_name: str = "lgbm",
) -> None:
    """Persist trained cluster models to disk for production inference (F1.1).

    Saves one .pkl file per cluster under data/models/<model_id>/cluster_<N>.pkl.
    Only called for the last (most recent) backtest timeframe so the most
    up-to-date models are always on disk.

    Args:
        models: {cluster_label: model} from train_and_predict_per_cluster.
        feature_cols: Ordered feature column names used during training.
            Can be a flat list (shared across clusters) or a dict mapping
            cluster labels to per-cluster feature lists (from per-cluster SHAP).
        model_id: e.g. 'lgbm_cluster'.
        timeframe_label: Backtest timeframe label (e.g. 'J' = most recent).
        prod_config: Loaded production_forecast section from forecast_pipeline_config.yaml (optional).
        model_meta: {cluster_label: {val_wape, train_rows}} from training functions.
        feature_importance_fn: Callable that extracts importance array from a model.
        model_name: LightGBM backend name used by ``get_best_iteration``.
    """
    if not models:
        return

    # Default to sklearn-style feature_importances_
    _get_importance = feature_importance_fn or (lambda m: m.feature_importances_)

    base_path = "data/models"
    if prod_config:
        base_path = prod_config.get("model_registry", {}).get("base_path", "data/models")

    out_dir = ROOT / base_path / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    _meta = model_meta or {}
    saved = 0
    for cluster_label, model in models.items():
        # Resolve per-cluster feature list when feature_cols is a dict
        if isinstance(feature_cols, dict):
            cluster_features = feature_cols.get(
                str(cluster_label), next(iter(feature_cols.values()))
            )
        else:
            cluster_features = feature_cols

        cluster_meta = _meta.get(cluster_label, {})
        n_est_used = (
            get_best_iteration(model, model_name) or cluster_meta.get("n_estimators_used") or 0
        )
        try:
            importance_raw = _get_importance(model)
        except AttributeError:
            importance_raw = []
        importance_dict = (
            dict(zip(cluster_features, [float(v) for v in importance_raw]))
            if len(importance_raw) == len(cluster_features)
            else {}
        )
        artifact = {
            "model": model,
            "feature_cols": cluster_features,
            "model_id": model_id,
            "cluster_label": str(cluster_label),
            "n_estimators_used": n_est_used,
            "train_rows": cluster_meta.get("train_rows"),
            "val_wape": cluster_meta.get("val_wape"),
            "trained_at": datetime.now(UTC).isoformat(),
            "timeframe": timeframe_label,
            "feature_importance": importance_dict,
        }
        file_path = out_dir / f"cluster_{cluster_label}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(artifact, f)
        saved += 1

    # Write feature importance summary per cluster
    fi_dir = out_dir / "feature_importance"
    fi_dir.mkdir(parents=True, exist_ok=True)
    for cluster_label, model in models.items():
        if isinstance(feature_cols, dict):
            cluster_features = feature_cols.get(
                str(cluster_label), next(iter(feature_cols.values()))
            )
        else:
            cluster_features = feature_cols

        try:
            imp = _get_importance(model)
        except AttributeError:
            imp = []
        if len(imp) == len(cluster_features):
            fi_dict = dict(zip(cluster_features, [float(v) for v in imp]))
            fi_sorted = dict(sorted(fi_dict.items(), key=lambda x: x[1], reverse=True))
            with open(fi_dir / f"cluster_{cluster_label}.json", "w") as f:
                json.dump(fi_sorted, f, indent=2)

    logger.info(
        "Persisted %d %s cluster models to %s/ (timeframe=%s)",
        saved,
        model_id,
        out_dir,
        timeframe_label,
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    # TODO(gen-4 cross-cutting): bias correction + quantile heads.
    # If `algorithms.<model_id>.params.quantile_heads` is non-empty in
    # config/forecasting/forecast_pipeline_config.yaml, run one fit per quantile
    # using LightGBM's `objective: quantile` with each `alpha`. That
    # will triple fit time per cluster and produce three parallel
    # prediction sets (p10/p50/p90) to write alongside the existing
    # regression column. Scaffold-only today — the config key is live
    # but not yet consumed by the fit loop.
    import argparse

    parser = argparse.ArgumentParser(
        description="Run tree-model per-cluster backtest (settings from forecast_pipeline_config.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model type to run (default: lgbm)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: config/forecasting/forecast_pipeline_config.yaml)",
    )
    parser.add_argument("--model-id", type=str, default=None, help="Override model_id from config")
    parser.add_argument(
        "--n-timeframes", type=int, default=None, help="Override n_timeframes from config"
    )
    parser.add_argument(
        "--cluster-override",
        type=str,
        default=None,
        help="CSV path with sku_ck,cluster_label columns to override promoted ml_cluster assignments",
    )
    parser.add_argument(
        "--clusters",
        type=str,
        default=None,
        help="Comma-separated cluster labels to restrict training to (e.g. '0' or '0,1,2'). "
        "Filters both DFU attrs and sales to matching clusters only.",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Train clusters in parallel (only when >4 clusters)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Max parallel workers (default: 4, requires --parallel)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Number of random seeds for variance estimation (default from config, fallback 1)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoints if a previous run crashed"
    )
    args = parser.parse_args()

    load_project_env()

    model_name = args.model
    registry = MODEL_REGISTRY[model_name]
    label = model_name.upper()

    with profiled_section("load_config"):
        model_class = _import_model_class(registry["class"])
        lib_module = importlib.import_module(registry["class"].rsplit(".", 1)[0])

        # Load config: prefer CLI --config path, else forecast_pipeline_config.yaml.
        # When a temp config is passed via --config (from tuning), it uses the
        # pipeline config format (algorithms.<model_id>.params).
        if args.config:
            config_path = Path(args.config)
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = load_forecast_pipeline_config()

        # Resolve algorithm params from pipeline config structure.
        # In pipeline config, model_id IS the key (e.g. "lgbm_cluster")
        # and params live under algorithms.<model_id>.params.
        # Also support legacy format where config_key maps model_name -> section.
        _config_key = registry["config_key"]
        _default_config_section = registry.get("config_section", f"{_config_key}_cluster")
        _algorithm_cfg = cfg.get("algorithms", {})
        _candidate_sections = [
            args.model_id,
            _default_config_section,
            _config_key,
        ]
        _selected_config_section = next(
            (section for section in _candidate_sections if section and section in _algorithm_cfg),
            _default_config_section,
        )
        algo_entry = _algorithm_cfg.get(_selected_config_section, {})
        # Extract params sub-dict if present (pipeline config format)
        if "params" in algo_entry:
            algo = dict(algo_entry["params"])
            # Merge lifecycle/meta keys the script needs
            for mk in (
                "enabled",
                "model_id",
                "cluster_strategy",
                "recursive",
                "shap_select",
                "shap_threshold",
                "shap_top_n",
                "shap_sample_size",
                "tune_inline",
                "params_file",
                "customer_features",
                "cluster_override_path",
            ):
                if mk in algo_entry:
                    algo.setdefault(mk, algo_entry[mk])
        else:
            # Legacy format: all keys flat in the section
            algo = dict(algo_entry)

        # Pipeline-level backtest settings from pipeline config
        pipeline_cfg = cfg if not args.config else None
        if pipeline_cfg is None:
            try:
                pipeline_cfg = load_forecast_pipeline_config()
            except FileNotFoundError:
                pipeline_cfg = None
        backtest_cfg = (pipeline_cfg or cfg).get("backtest", {})

        # Check algorithm enabled flag from the roster (skip if disabled).
        # Match by config_section (pipeline model_id, e.g. "lgbm_cluster")
        # or by config_key (base name, e.g. "lgbm") as prefix of roster keys.
        try:
            roster = get_algorithm_roster(stage="backtest")
            roster_entry = roster.get(_selected_config_section)
            if roster_entry is None:
                # Not in roster with backtest=true — check if explicitly disabled
                all_roster = get_algorithm_roster()
                disabled = _selected_config_section not in all_roster
                if disabled:
                    logger.warning(
                        "Model '%s' is disabled in forecast_pipeline_config.yaml — skipping",
                        args.model_id or model_name,
                    )
                    return
        except FileNotFoundError:
            pass  # No pipeline config — run unconditionally

        # Propagate backtest-level noise/smoothing settings into algo dict so run_tree_backtest
        # can access them via algo_config (algo only holds the per-algorithm sub-dict).
        algo.setdefault(
            "recursive_noise_enabled",
            backtest_cfg.get("recursive_noise_enabled", cfg.get("recursive_noise_enabled", False)),
        )
        algo.setdefault(
            "recursive_noise_pct",
            backtest_cfg.get("recursive_noise_pct", cfg.get("recursive_noise_pct", 0.05)),
        )
        algo.setdefault(
            "recursive_lag_smooth",
            backtest_cfg.get("recursive_lag_smooth", cfg.get("recursive_lag_smooth", 0.0)),
        )

    # Resolve cluster override: CLI flag takes priority, then algo_config key
    cluster_override = args.cluster_override or algo.get("cluster_override_path")
    if cluster_override:
        algo["cluster_override_path"] = cluster_override
        logger.info("Cluster override enabled: %s", cluster_override)

    if args.clusters:
        algo["cluster_filter"] = [c.strip() for c in args.clusters.split(",")]
        logger.info("Cluster filter: restricting to clusters %s", algo["cluster_filter"])

    # Resolve cluster_strategy: pipeline config entry > algo dict > default
    _roster_cs = algo.get("cluster_strategy")
    if not _roster_cs and pipeline_cfg:
        # Look up by the selected pipeline model_id first, then fall back to base model.
        _algo_entry = pipeline_cfg.get("algorithms", {}).get(_selected_config_section)
        if _algo_entry is None and _selected_config_section != model_name:
            _algo_entry = pipeline_cfg.get("algorithms", {}).get(model_name)
        if _algo_entry:
            _roster_cs = _algo_entry.get("cluster_strategy")
    cluster_strategy = _roster_cs or "per_cluster"
    # Guard: if clustering is disabled but strategy is per_cluster, fall back to global
    if (
        pipeline_cfg
        and not pipeline_cfg.get("clustering", {}).get("enabled", True)
        and cluster_strategy == "per_cluster"
    ):
        logger.warning("Clustering disabled in pipeline config — falling back to global strategy")
        cluster_strategy = "global"

    # Registry captures model metadata across timeframes for use in _persistence_fn.
    _model_meta_registry: dict[str, dict] = {}

    iter_param = registry["iter_param"]
    _model_kwargs: dict[str, Any] = {
        "model_name": model_name,
        "registry": registry,
        "model_class": model_class,
        "lib_module": lib_module,
    }
    if cluster_strategy == "global":
        _inner_train_fn = train_and_predict_global
        default_model_id = f"{model_name}_global"
    else:
        _inner_train_fn = train_and_predict_per_cluster
        default_model_id = f"{model_name}_cluster"
        _model_kwargs["parallel"] = args.parallel
        _model_kwargs["max_workers"] = args.workers
        if args.parallel:
            logger.info("Parallel cluster training enabled (max_workers=%d)", args.workers)

    def train_fn(train_df, predict_df, feature_cols, cat_cols, params):
        result, models, meta = _inner_train_fn(
            train_df,
            predict_df,
            feature_cols,
            cat_cols,
            params,
            **_model_kwargs,
        )
        _model_meta_registry.update(meta)
        return result, models

    model_id = args.model_id or algo.get("model_id", default_model_id)
    n_timeframes = args.n_timeframes or backtest_cfg.get("n_timeframes", 10)
    output_dir = ROOT / backtest_cfg.get("output_dir", "data/backtest")
    embargo_months = backtest_cfg.get("embargo_months", 0)

    # Build model-specific default params from config
    model_params = registry["default_params"](algo)

    recursive = algo.get("recursive", False)
    shap_select = algo.get("shap_select", False)
    shap_threshold = algo.get("shap_threshold", 0.95)
    shap_top_n = algo.get("shap_top_n", None)
    shap_sample_size = algo.get("shap_sample_size", 500)
    _corr_filter = algo.get("correlation_filter", False)
    _corr_threshold = algo.get("correlation_threshold", 0.95)
    _var_filter = algo.get("variance_filter", False)
    _var_threshold = algo.get("variance_threshold", 0.01)
    _shap_min_features = backtest_cfg.get("shap_min_features", 20)
    tune_inline = algo.get("tune_inline", False)
    params_file = algo.get("params_file", None)
    iter_param = registry["iter_param"]

    logger.info(
        "%s config: model_id=%s, cluster_strategy=%s, recursive=%s, shap_select=%s, "
        "tune_inline=%s, n_timeframes=%d",
        label,
        model_id,
        cluster_strategy,
        recursive,
        shap_select,
        tune_inline,
        n_timeframes,
    )

    # GPU detection with env-var override: DEMAND_GPU=on|off|auto (default: auto)
    with profiled_section("detect_gpu"):
        _gpu_pref = os.getenv("DEMAND_GPU", "auto").lower()
        _use_gpu = False
        if _gpu_pref == "on":
            _use_gpu = True
            logger.info("GPU forced ON via DEMAND_GPU env var")
        elif _gpu_pref == "off":
            _use_gpu = False
            logger.info("GPU disabled via DEMAND_GPU env var")
        else:  # auto
            should_test = True
            if registry["gpu_test_platform_check"] and platform.system() != "Darwin":
                should_test = False
            if should_test:
                try:
                    _use_gpu = probe_tree_gpu_available(model_name, registry["gpu_params"]())
                    if _use_gpu:
                        logger.info("Using GPU for %s", label)
                except Exception:
                    logger.info("GPU not available, falling back to CPU")

    # Resolve n_seeds: CLI > config > default (1)
    n_seeds = args.n_seeds or backtest_cfg.get("n_seeds", 1)
    _seed_param_key = "random_state"

    params_source = "config_defaults"
    if params_file:
        tuning_data = load_best_params(Path(params_file))
        tuned = tuning_data.get("best_params", {})
        n_est_tuned = tuning_data.get("best_n_estimators", None)
        model_params.update(tuned)
        if n_est_tuned:
            model_params[iter_param] = n_est_tuned
        params_source = f"tuning_file:{params_file}"
        logger.info(
            "Loaded tuned params from %s (best_wape=%s%%, n_est=%s)",
            params_file,
            tuning_data.get("best_wape"),
            model_params[iter_param],
        )

    if _use_gpu:
        model_params.update(registry["gpu_params"]())

    # Build causal per-timeframe tuner when tune_inline is set (PL-002)
    inline_tuner_fn = None
    if tune_inline:
        _tune_config_path = ROOT / "config" / "forecasting" / "hyperparameter_tuning.yaml"
        with open(_tune_config_path) as _f:
            _tune_config = yaml.safe_load(_f)
        _fold_fn = TRAIN_FOLD_FNS[model_name]
        _base_params = model_params.copy()

        def inline_tuner_fn(full_grid, feature_cols, cat_cols, train_end):
            tuned, n_est = tune_for_timeframe(
                model_name=model_name,
                train_fold_fn=_fold_fn,
                full_grid=full_grid,
                feature_cols=feature_cols,
                cat_cols=cat_cols,
                cutoff_date=train_end,
                config=_tune_config,
                n_trials=None,
            )
            if not tuned:
                return _base_params.copy()
            result = {**_base_params, **tuned, iter_param: n_est}
            logger.info(
                "Inline tuned: %s=%s, lr=%.4f", iter_param, n_est, tuned.get("learning_rate", 0)
            )
            return result

        params_source = "inline_tuning"
        logger.info(
            "Inline tuning enabled (inline_n_trials=%s, inline_n_splits=%s)",
            _tune_config["tuning"].get("inline_n_trials", 20),
            _tune_config["tuning"].get("inline_n_splits", 3),
        )

    logger.info("Params source: %s", params_source)

    # Build SHAP feature selector closure (Feature 42)
    feature_selector_fn = None
    if shap_select:
        from common.ml.shap_selector import (
            compute_timeframe_shap,
            compute_timeframe_shap_per_cluster,
        )

        shap_extractor_name = registry["shap_extractor"]
        shap_extractor_fn = getattr(
            importlib.import_module("common.ml.shap_selector"),
            shap_extractor_name,
        )

        def feature_selector_fn(model_or_dict, train_data, feature_cols, cat_cols, tf_idx, cutoff):
            if isinstance(model_or_dict, dict):
                return compute_timeframe_shap_per_cluster(
                    model_or_dict,
                    train_data,
                    feature_cols,
                    cat_cols,
                    tf_idx,
                    cutoff,
                    shap_extractor_fn=shap_extractor_fn,
                    sample_size=shap_sample_size,
                    cumulative_threshold=shap_threshold,
                    top_n=shap_top_n,
                    min_features=_shap_min_features,
                    correlation_filter=_corr_filter,
                    correlation_threshold=_corr_threshold,
                    variance_filter=_var_filter,
                    variance_threshold=_var_threshold,
                )
            return compute_timeframe_shap(
                model_or_dict,
                train_data,
                feature_cols,
                cat_cols,
                tf_idx,
                cutoff,
                shap_extractor_fn=shap_extractor_fn,
                cluster_strategy="per_cluster",
                sample_size=shap_sample_size,
                cumulative_threshold=shap_threshold,
                top_n=shap_top_n,
                min_features=_shap_min_features,
                correlation_filter=_corr_filter,
                correlation_threshold=_corr_threshold,
                variance_filter=_var_filter,
                variance_threshold=_var_threshold,
            )

        logger.info(
            "Feature selection enabled (shap=%.2f, corr_filter=%s/%.2f, var_filter=%s/%.3f)",
            shap_threshold,
            _corr_filter,
            _corr_threshold,
            _var_filter,
            _var_threshold,
        )

    # Load production forecast config for model persistence (F1.1)
    pipeline_config_path = ROOT / "config" / "forecasting" / "forecast_pipeline_config.yaml"
    prod_config = None
    if pipeline_config_path.exists():
        with open(pipeline_config_path) as f:
            raw = yaml.safe_load(f)
        prod_config = raw.get("production_forecast", {})

    _fi_fn = registry["feature_importance_fn"]

    def _persistence_fn(
        models: dict, feature_cols: list[str] | dict[str, list[str]], timeframe_label: str
    ) -> None:
        persist_cluster_models(
            models,
            feature_cols,
            model_id,
            timeframe_label,
            prod_config,
            _model_meta_registry,
            feature_importance_fn=_fi_fn,
            model_name=model_name,
        )

    if n_seeds > 1:
        logger.info("Multi-seed evaluation: running %d seeds for variance estimation", n_seeds)

    seed_accuracies: list[float] = []
    for seed_idx in range(n_seeds):
        seed_value = seed_idx
        seed_params = {**model_params, _seed_param_key: seed_value}

        seed_extra_meta: dict[str, Any] = {"params_source": params_source}
        if n_seeds > 1:
            seed_extra_meta["seed"] = seed_value
            seed_extra_meta["n_seeds"] = n_seeds
            logger.info(
                "Seed %d/%d (random_%s=%d)",
                seed_idx + 1,
                n_seeds,
                _seed_param_key.split("_")[-1],
                seed_value,
            )

        with profiled_section("run_backtest"):
            run_tree_backtest(
                model_id=model_id,
                n_timeframes=n_timeframes,
                output_dir=output_dir,
                model_params=seed_params,
                model_params_key=registry["model_params_key"],
                model_type_tag=registry["model_type_tag"],
                train_fn_per_cluster=train_fn,
                extra_metadata=seed_extra_meta,
                cat_dtype=registry["cat_dtype"],
                inline_tuner_fn=inline_tuner_fn,
                feature_selector_fn=feature_selector_fn,
                recursive=recursive,
                model_persistence_fn=_persistence_fn,
                algo_config=algo,
                embargo_months=embargo_months,
                resume=args.resume,
            )

        # Read accuracy from metadata written by run_tree_backtest
        if n_seeds > 1:
            meta_path = output_dir / model_id / "backtest_metadata.json"
            if meta_path.exists():
                with open(meta_path) as mf:
                    seed_meta = json.load(mf)
                acc_block = seed_meta.get("accuracy_at_execution_lag", {})
                accuracy_pct = acc_block.get("accuracy_pct")
                if accuracy_pct is not None:
                    seed_accuracies.append(float(accuracy_pct))
                    logger.info("Seed %d/%d accuracy: %.2f%%", seed_idx + 1, n_seeds, accuracy_pct)

    # Log and persist multi-seed summary
    if n_seeds > 1 and seed_accuracies:
        mean_acc = float(np.mean(seed_accuracies))
        std_acc = float(np.std(seed_accuracies))
        logger.info(
            "Multi-seed accuracy: %.2f%% +/- %.2f%% (n=%d seeds)",
            mean_acc,
            std_acc,
            n_seeds,
        )
        # Write seed summary to metadata
        meta_path = output_dir / model_id / "backtest_metadata.json"
        if meta_path.exists():
            with open(meta_path) as mf:
                final_meta = json.load(mf)
            final_meta["multi_seed_summary"] = {
                "n_seeds": n_seeds,
                "seed_accuracies": seed_accuracies,
                "mean_accuracy_pct": round(mean_acc, 4),
                "std_accuracy_pct": round(std_acc, 4),
            }
            with open(meta_path, "w") as mf:
                json.dump(final_meta, mf, indent=2, default=str)
            logger.info("Saved multi-seed summary to %s", meta_path)


if __name__ == "__main__":
    setup_logging()
    main()
