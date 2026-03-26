"""
Run tree-model backtesting with per-cluster strategy and expanding-window timeframes.

Supports LGBM (default), CatBoost, and XGBoost via the --model flag.
All run options (recursive, SHAP, tuning) are controlled via
config/algorithm_config.yaml rather than CLI flags.

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.backtest_framework import (
    compute_cluster_demand_stats,
    resolve_cluster_params,
    run_tree_backtest,
)
from common.constants import MIN_CLUSTER_ROWS
from common.ml.model_registry import (
    compute_early_stop_patience,
    fit_model,
    get_best_iteration,
)
from common.services.perf_profiler import profiled_section
from common.tuning import TRAIN_FOLD_FNS, load_best_params, tune_for_timeframe

logger = logging.getLogger(__name__)


# ── Model Registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "lgbm": {
        "class": "lightgbm.LGBMRegressor",
        "config_key": "lgbm",
        "config_section": "lgbm_cluster",
        "iter_param": "n_estimators",
        "gpu_params": lambda: {"device": "gpu"},
        "gpu_test": lambda cls: cls(device="gpu", n_estimators=1, verbosity=-1),
        "gpu_test_platform_check": True,  # only auto-detect on Darwin
        "fit_extras_per_cluster": lambda params, iter_param: {},
        "fit_extras_global": lambda params, iter_param: {},
        "default_params": lambda algo: {
            "n_estimators": algo.get("n_estimators", 300),
            "learning_rate": algo.get("learning_rate", 0.08),
            "num_leaves": algo.get("num_leaves", 31),
            "min_child_samples": algo.get("min_child_samples", 20),
            "max_depth": algo.get("max_depth", -1),
            "min_gain_to_split": algo.get("min_gain_to_split", 0.01),
            "subsample": algo.get("subsample", 0.80),
            "bagging_freq": algo.get("bagging_freq", 1),
            "colsample_bytree": algo.get("colsample_bytree", 0.80),
            "feature_fraction_bynode": algo.get("feature_fraction_bynode", 0.7),
            "reg_lambda": algo.get("reg_lambda", 1.0),
            "reg_alpha": algo.get("reg_alpha", 0.1),
            "path_smooth": algo.get("path_smooth", 2.0),
            "max_bin": algo.get("max_bin", 127),  # halve histogram memory (default 255)
            "feature_pre_filter": True,  # skip features not used in splits
            "verbosity": -1,
            "random_state": 42,
            "n_jobs": -1,
        },
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
    "catboost": {
        "class": "catboost.CatBoostRegressor",
        "config_key": "catboost",
        "config_section": "catboost_cluster",
        "iter_param": "iterations",
        "gpu_params": lambda: {"task_type": "GPU"},
        "gpu_test": lambda cls: cls(task_type="GPU", iterations=1, verbose=0),
        "gpu_test_platform_check": False,
        "fit_extras_per_cluster": lambda params, iter_param: {},
        "fit_extras_global": lambda params, iter_param: {},
        "default_params": lambda algo: {
            k: v
            for k, v in {
                "iterations": algo.get("iterations", 300),
                "learning_rate": algo.get("learning_rate", 0.08),
                "depth": algo.get("depth", 5),
                "l2_leaf_reg": algo.get("l2_leaf_reg", 3.0),
                "border_count": algo.get("border_count", 64),
                "max_ctr_complexity": algo.get("max_ctr_complexity", 1),
                "grow_policy": algo.get("grow_policy"),
                "max_leaves": algo.get("max_leaves"),
                "subsample": algo.get("subsample"),
                "reg_lambda": algo.get("reg_lambda"),
                "random_strength": algo.get("random_strength"),
                "min_data_in_leaf": algo.get("min_data_in_leaf"),
                "colsample_bylevel": algo.get("colsample_bylevel"),
                "bagging_temperature": algo.get("bagging_temperature"),
                "bootstrap_type": algo.get("bootstrap_type"),
                "model_size_reg": algo.get("model_size_reg"),
                "score_function": algo.get("score_function"),
                "boost_from_average": algo.get("boost_from_average"),
                "leaf_estimation_method": algo.get("leaf_estimation_method"),
                "leaf_estimation_iterations": algo.get("leaf_estimation_iterations"),
                "langevin": algo.get("langevin"),
                "diffusion_temperature": algo.get("diffusion_temperature"),
                "random_seed": 42,
                "loss_function": "RMSE",
                "verbose": 0,
                "thread_count": -1,
            }.items()
            if v is not None
        },
        "cat_dtype": "str",
        "model_params_key": "catboost_params",
        "model_type_tag": "catboost_backtest",
        "shap_extractor": "compute_shap_catboost",
        "best_iteration_attr": "best_iteration_",
        "feature_importance_fn": lambda model: model.get_feature_importance(),
        "constant_target_guard": True,
        "needs_cat_indices": True,
        "needs_cat_dtype_cast": False,
    },
    "xgboost": {
        "class": "xgboost.XGBRegressor",
        "config_key": "xgboost",
        "config_section": "xgboost_cluster",
        "iter_param": "n_estimators",
        "gpu_params": lambda: {"device": "cuda"},
        "gpu_test": lambda cls: cls(device="cuda", n_estimators=1, verbosity=0),
        "gpu_test_platform_check": False,
        "fit_extras_per_cluster": lambda params, iter_param: {},
        "fit_extras_global": lambda params, iter_param: {},
        "default_params": lambda algo: {
            k: v
            for k, v in {
                "n_estimators": algo.get("n_estimators", 500),
                "learning_rate": algo.get("learning_rate", 0.05),
                "max_depth": algo.get("max_depth", 6),
                "min_child_weight": algo.get("min_child_weight", 5),
                "subsample": algo.get("subsample", 0.8),
                "colsample_bytree": algo.get("colsample_bytree", 0.8),
                "grow_policy": algo.get("grow_policy"),
                "max_leaves": algo.get("max_leaves"),
                "max_bin": algo.get("max_bin"),
                "reg_lambda": algo.get("reg_lambda"),
                "reg_alpha": algo.get("reg_alpha"),
                "gamma": algo.get("gamma"),
                "colsample_bylevel": algo.get("colsample_bylevel"),
                "booster": algo.get("booster"),
                **({"rate_drop": algo["rate_drop"]} if algo.get("booster") == "dart" and "rate_drop" in algo else {}),
                **({"skip_drop": algo["skip_drop"]} if algo.get("booster") == "dart" and "skip_drop" in algo else {}),
                "verbosity": 0,
                "random_state": 42,
                "n_jobs": -1,
                "enable_categorical": True,
                "tree_method": "hist",
            }.items()
            if v is not None
        },
        "cat_dtype": "category",
        "model_params_key": "xgboost_params",
        "model_type_tag": "xgboost_backtest",
        "shap_extractor": "compute_shap_global",
        "best_iteration_attr": "best_iteration",  # no trailing underscore for XGBoost
        "feature_importance_fn": lambda model: model.feature_importances_,
        "constant_target_guard": True,
        "needs_cat_indices": False,
        "needs_cat_dtype_cast": True,  # XGBoost needs explicit .astype("category")
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
    registry: dict[str, Any],
    model_class: type,
    lib_module: Any,
) -> tuple[str, pd.DataFrame | None, Any | None, dict | None]:
    """Train a single cluster model. Returns (cluster_label, result_df, model, meta_dict).

    This function is self-contained with no shared mutable state, making it safe
    for use in ProcessPoolExecutor.
    """
    needs_cat_dtype_cast = registry["needs_cat_dtype_cast"]
    constant_target_guard = registry["constant_target_guard"]
    iter_param = registry["iter_param"]

    cat_cols_in_features = [c for c in cat_cols if c in feature_cols] if needs_cat_dtype_cast else []

    if len(train_c) < MIN_CLUSTER_ROWS or len(pred_c) == 0:
        if len(pred_c) > 0:
            logger.info("Cluster %d/%d '%s': skipped (train=%d), zeroing %d predictions",
                        ci, n_clusters, cluster_label, len(train_c), len(pred_c))
            result = pred_c[["sku_ck", "item_id", "customer_group", "loc", "startdate"]].copy()
            result["basefcst_pref"] = 0.0
            return cluster_label, result, None, None
        return cluster_label, None, None, None

    X_train = train_c[feature_cols].copy() if needs_cat_dtype_cast else train_c[feature_cols]
    y_train = train_c["qty"]
    X_pred = pred_c[feature_cols].copy() if needs_cat_dtype_cast else pred_c[feature_cols]

    if needs_cat_dtype_cast:
        for col in cat_cols_in_features:
            X_train[col] = X_train[col].astype("category")
            X_pred[col] = X_pred[col].astype("category")

    t0 = time.time()
    # Time-aware train/val split — last 20% of cluster rows for validation
    n_val = max(1, int(len(X_train) * 0.20))
    X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

    # Guard: some models crash on constant targets
    if constant_target_guard and y_tr.nunique() <= 1:
        const_val = float(y_tr.iloc[0]) if len(y_tr) > 0 else 0.0
        logger.info("Cluster %d/%d '%s': skipped (constant target=%.0f), using constant for %d predictions",
                    ci, n_clusters, cluster_label, const_val, len(pred_c))
        result = pred_c[["sku_ck", "item_id", "customer_group", "loc", "startdate"]].copy()
        result["basefcst_pref"] = const_val
        return cluster_label, result, None, None

    # Per-cluster adaptive hyperparameter profiles: resolve cluster-specific
    # overrides based on demand characteristics (sparse, volatile, stable, etc.)
    cluster_stats = compute_cluster_demand_stats(train_c, cluster_label)
    resolved_params, profile_name = resolve_cluster_params(
        cluster_label, cluster_stats, params,
    )
    if profile_name not in ("none", "default"):
        logger.info(
            "Cluster %d/%d '%s': matched profile '%s' (mean_demand=%.1f, cv=%.2f, "
            "zero_pct=%.2f, seasonal_amp=%.2f)",
            ci, n_clusters, cluster_label, profile_name,
            cluster_stats["mean_demand"], cluster_stats["cv_demand"],
            cluster_stats["zero_demand_pct"], cluster_stats["seasonal_amplitude"],
        )

    # Filter resolved params to only include keys valid for this model
    # (cluster profiles may inject LGBM-specific keys like reg_alpha, num_leaves)
    valid_keys = set(params.keys())
    filtered_params = {k: v for k, v in resolved_params.items() if k in valid_keys}
    fit_params = {**filtered_params, **registry["fit_extras_per_cluster"](filtered_params, iter_param)}
    max_iters = fit_params.get(iter_param, 1000)
    model = model_class(**fit_params)

    # Unified fit call — all model-specific logic in model_registry.fit_model()
    fit_model(model, model_name, X_tr, y_tr, X_val, y_val, cat_cols, feature_cols, lib_module, max_iters)

    preds = model.predict(X_pred)

    # Per-cluster validation WAPE
    val_preds = model.predict(X_val)
    val_denom = float(abs(y_val.sum()))
    val_wape = round(float((abs(val_preds - y_val.values)).sum() / val_denom * 100), 2) if val_denom > 0 else 0.0

    result = pred_c[["sku_ck", "item_id", "customer_group", "loc", "startdate"]].copy()
    result["basefcst_pref"] = np.maximum(preds, 0)
    n_est_used = get_best_iteration(model, model_name)
    if n_est_used is None:
        n_est_used = fit_params[iter_param]
    meta = {
        "val_wape": val_wape,
        "train_rows": len(X_tr),
        "cluster_profile": profile_name,
        "cluster_stats": cluster_stats,
    }
    logger.info("Cluster %d/%d '%s': train=%s, pred=%s, best_iter=%s, val_wape=%.1f%%, profile=%s (%.1fs)",
                ci, n_clusters, cluster_label, f"{len(train_c):,}", f"{len(pred_c):,}",
                n_est_used, val_wape, profile_name, time.time() - t0)

    return cluster_label, result, model, meta


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
) -> tuple[pd.DataFrame, dict, dict[str, dict]]:
    """Train separate tree models per ml_cluster.

    ml_cluster is kept as a hard feature (constant within each cluster partition,
    but required for consistent feature alignment with global models).

    When ``parallel=True`` and there are >4 clusters, uses ProcessPoolExecutor
    to train clusters concurrently.  Each cluster's training is independent
    (no shared mutable state).

    Returns (predictions, models, model_meta) where model_meta stores per-cluster
    training metadata (val_wape, train_rows).
    """
    all_results: list[pd.DataFrame] = []
    models: dict = {}
    model_meta: dict[str, dict] = {}

    clusters = sorted(train_df["ml_cluster"].dropna().unique())
    label = model_name.upper()
    n_clusters = len(clusters)
    logger.info("Training %d per-cluster %s models...", n_clusters, label)

    _worker_kwargs = {
        "model_name": model_name,
        "registry": registry,
        "model_class": model_class,
        "lib_module": lib_module,
    }

    use_parallel = parallel and n_clusters > 4
    if use_parallel:
        logger.info("Parallel cluster training enabled: %d workers for %d clusters",
                    max_workers, n_clusters)
        futures = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for ci, cluster_label in enumerate(clusters, 1):
                train_c = train_df[train_df["ml_cluster"] == cluster_label]
                pred_c = predict_df[predict_df["ml_cluster"] == cluster_label]
                future = executor.submit(
                    _train_single_cluster,
                    cluster_label, ci, n_clusters,
                    train_c, pred_c,
                    feature_cols, cat_cols, params,
                    **_worker_kwargs,
                )
                futures[future] = cluster_label

            for future in as_completed(futures):
                cl, result, model, meta = future.result()
                if result is not None:
                    all_results.append(result)
                if model is not None:
                    models[cl] = model
                if meta is not None:
                    model_meta[cl] = meta
    else:
        if parallel:
            logger.info("Parallel mode requested but only %d clusters (<= 4), using sequential",
                        n_clusters)
        for ci, cluster_label in enumerate(clusters, 1):
            train_c = train_df[train_df["ml_cluster"] == cluster_label]
            pred_c = predict_df[predict_df["ml_cluster"] == cluster_label]
            cl, result, model, meta = _train_single_cluster(
                cluster_label, ci, n_clusters,
                train_c, pred_c,
                feature_cols, cat_cols, params,
                **_worker_kwargs,
            )
            if result is not None:
                all_results.append(result)
            if model is not None:
                models[cl] = model
            if meta is not None:
                model_meta[cl] = meta

    no_cluster = predict_df[
        predict_df["ml_cluster"].isna() | (
            (predict_df["ml_cluster"] == "__unknown__") & ("__unknown__" not in models)
        )
    ]
    if len(no_cluster) > 0:
        logger.info("%d predict rows with no cluster -> zeroing", len(no_cluster))
        result = no_cluster[["sku_ck", "item_id", "customer_group", "loc", "startdate"]].copy()
        result["basefcst_pref"] = 0.0
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
    """Train a single global tree model on ALL data with ml_cluster as categorical feature."""
    needs_cat_dtype_cast = registry["needs_cat_dtype_cast"]
    iter_param = registry["iter_param"]
    label = model_name.upper()

    cat_cols_in_features = [c for c in cat_cols if c in feature_cols] if needs_cat_dtype_cast else []

    logger.info("Training global %s on %s rows, %d features (includes ml_cluster)...",
                label, f"{len(train_df):,}", len(feature_cols))

    # Sort by startdate so last 15% = most recent months (not last DFUs alphabetically)
    sorted_idx = train_df["startdate"].argsort(kind="mergesort")
    train_sorted = train_df.iloc[sorted_idx]
    X_train = train_sorted[feature_cols].copy() if needs_cat_dtype_cast else train_sorted[feature_cols]
    y_train = train_sorted["qty"]
    X_pred = predict_df[feature_cols].copy() if needs_cat_dtype_cast else predict_df[feature_cols]

    if needs_cat_dtype_cast:
        for col in cat_cols_in_features:
            X_train[col] = X_train[col].astype("category")
            X_pred[col] = X_pred[col].astype("category")

    # Time-aware train/val split — last 15% of rows for validation (most recent months)
    n_val = max(1, int(len(X_train) * 0.15))
    X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

    fit_params = {**params, **registry["fit_extras_global"](params, iter_param)}
    max_iters = fit_params.get(iter_param, 1000)
    model = model_class(**fit_params)

    # Unified fit call — all model-specific logic in model_registry.fit_model()
    fit_model(model, model_name, X_tr, y_tr, X_val, y_val, cat_cols, feature_cols, lib_module, max_iters)

    preds = model.predict(X_pred)

    val_preds = model.predict(X_val)
    val_denom = float(abs(y_val.sum()))
    val_wape = round(float((abs(val_preds - y_val.values)).sum() / val_denom * 100), 2) if val_denom > 0 else 0.0
    n_est_used = get_best_iteration(model, model_name)
    if not n_est_used:
        n_est_used = fit_params[iter_param]
    logger.info("Global %s: val_WAPE=%.1f%%, best_iter=%s, train=%s, pred=%s",
                label, val_wape, n_est_used, f"{len(train_df):,}", f"{len(predict_df):,}")

    result = predict_df[["sku_ck", "item_id", "customer_group", "loc", "startdate"]].copy()
    result["basefcst_pref"] = np.clip(preds, 0, None)
    global_meta = {"global": {"val_wape": val_wape, "train_rows": len(X_tr)}}
    return result, {"global": model}, global_meta


# ── Model persistence ─────────────────────────────────────────────────────────


def persist_cluster_models(
    models: dict,
    feature_cols: list[str],
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
        model_id: e.g. 'lgbm_cluster'.
        timeframe_label: Backtest timeframe label (e.g. 'J' = most recent).
        prod_config: Loaded production_forecast_config.yaml dict (optional).
        model_meta: {cluster_label: {val_wape, train_rows}} from training functions.
        feature_importance_fn: Callable that extracts importance array from a model.
        model_name: One of 'lgbm', 'catboost', 'xgboost' (for get_best_iteration).
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
        n_est_used = get_best_iteration(model, model_name) or 0
        importance_raw = _get_importance(model)
        importance_dict = dict(zip(feature_cols, [float(v) for v in importance_raw])) if len(importance_raw) == len(feature_cols) else {}
        cluster_meta = _meta.get(cluster_label, {})
        artifact = {
            "model": model,
            "feature_cols": feature_cols,
            "model_id": model_id,
            "cluster_label": str(cluster_label),
            "n_estimators_used": n_est_used,
            "train_rows": cluster_meta.get("train_rows"),
            "val_wape": cluster_meta.get("val_wape"),
            "trained_at": datetime.now(timezone.utc).isoformat(),
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
        imp = _get_importance(model)
        if len(imp) == len(feature_cols):
            fi_dict = dict(zip(feature_cols, [float(v) for v in imp]))
            fi_sorted = dict(sorted(fi_dict.items(), key=lambda x: x[1], reverse=True))
            with open(fi_dir / f"cluster_{cluster_label}.json", "w") as f:
                json.dump(fi_sorted, f, indent=2)

    logger.info("Persisted %d %s cluster models to %s/ (timeframe=%s)",
                saved, model_id, out_dir, timeframe_label)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Run tree-model per-cluster backtest (settings from algorithm_config.yaml)",
    )
    parser.add_argument("--model", type=str, default="lgbm",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model type to run (default: lgbm)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to algorithm_config.yaml (default: config/algorithm_config.yaml)")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model_id from config")
    parser.add_argument("--n-timeframes", type=int, default=None,
                        help="Override n_timeframes from config")
    parser.add_argument("--cluster-override", type=str, default=None,
                        help="CSV path with sku_ck,cluster_label columns to override dim_sku.ml_cluster")
    parser.add_argument("--parallel", action="store_true",
                        help="Train clusters in parallel (only when >4 clusters)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Max parallel workers (default: 4, requires --parallel)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    model_name = args.model
    registry = MODEL_REGISTRY[model_name]
    label = model_name.upper()

    with profiled_section("load_config"):
        # Dynamically import model class and its parent library module
        model_class = _import_model_class(registry["class"])
        lib_module = importlib.import_module(registry["class"].rsplit(".", 1)[0])

        # Load algorithm config
        config_path = Path(args.config) if args.config else ROOT / "config" / "algorithm_config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        algo = cfg["algorithms"][registry["config_key"]]
        backtest_cfg = cfg.get("backtest", {})

    # Resolve cluster override: CLI flag takes priority, then algo_config key
    cluster_override = args.cluster_override or algo.get("cluster_override_path")
    if cluster_override:
        algo["cluster_override_path"] = cluster_override
        logger.info("Cluster override enabled: %s", cluster_override)

    cluster_strategy = algo.get("cluster_strategy", "per_cluster")
    iter_param = registry["iter_param"]

    # Registry captures model metadata across timeframes for use in _persistence_fn.
    _model_meta_registry: dict[str, dict] = {}

    # Build partial train functions with model-specific kwargs bound
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
            train_df, predict_df, feature_cols, cat_cols, params,
            **_model_kwargs,
        )
        _model_meta_registry.update(meta)
        return result, models

    model_id = args.model_id or algo.get("model_id", default_model_id)
    n_timeframes = args.n_timeframes or backtest_cfg.get("n_timeframes", 10)
    output_dir = ROOT / backtest_cfg.get("output_dir", "data/backtest")
    recursive = algo.get("recursive", False)
    shap_select = algo.get("shap_select", False)
    shap_threshold = algo.get("shap_threshold", 0.95)
    shap_top_n = algo.get("shap_top_n", None)
    shap_sample_size = algo.get("shap_sample_size", 500)
    tune_inline = algo.get("tune_inline", False)
    params_file = algo.get("params_file", None)

    logger.info("%s config: model_id=%s, cluster_strategy=%s, recursive=%s, shap_select=%s, "
                "tune_inline=%s, n_timeframes=%d",
                label, model_id, cluster_strategy, recursive, shap_select, tune_inline, n_timeframes)

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
                    _test_model = registry["gpu_test"](model_class)
                    _test_model.fit([[0]], [0])
                    _use_gpu = True
                    logger.info("Using GPU for %s", label)
                except Exception:
                    logger.info("GPU not available, falling back to CPU")

    # Build model-specific default params from config
    model_params = registry["default_params"](algo)

    params_source = "config_defaults"
    if params_file:
        tuning_data = load_best_params(Path(params_file))
        tuned = tuning_data.get("best_params", {})
        n_est_tuned = tuning_data.get("best_n_estimators", None)
        model_params.update(tuned)
        if n_est_tuned:
            model_params[iter_param] = n_est_tuned
        params_source = f"tuning_file:{params_file}"
        logger.info("Loaded tuned params from %s (best_wape=%s%%, n_est=%s)",
                    params_file, tuning_data.get('best_wape'), model_params[iter_param])

    if _use_gpu:
        model_params.update(registry["gpu_params"]())

    # Build causal per-timeframe tuner when tune_inline is set (PL-002)
    inline_tuner_fn = None
    if tune_inline:
        _tune_config_path = ROOT / "config" / "hyperparameter_tuning.yaml"
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
            logger.info("Inline tuned: %s=%s, lr=%.4f",
                        iter_param, n_est, tuned.get('learning_rate', 0))
            return result

        params_source = "inline_tuning"
        logger.info("Inline tuning enabled (inline_n_trials=%s, inline_n_splits=%s)",
                    _tune_config['tuning'].get('inline_n_trials', 20),
                    _tune_config['tuning'].get('inline_n_splits', 3))

    logger.info("Params source: %s", params_source)

    # Build SHAP feature selector closure (Feature 42)
    feature_selector_fn = None
    if shap_select:
        from common.shap_selector import compute_timeframe_shap
        shap_extractor_name = registry["shap_extractor"]
        shap_extractor_fn = getattr(
            importlib.import_module("common.shap_selector"),
            shap_extractor_name,
        )

        def feature_selector_fn(model_or_dict, train_data, feature_cols, cat_cols, tf_idx, cutoff):
            return compute_timeframe_shap(
                model_or_dict, train_data, feature_cols, cat_cols,
                tf_idx, cutoff,
                shap_extractor_fn=shap_extractor_fn,
                cluster_strategy="per_cluster",
                sample_size=shap_sample_size,
                cumulative_threshold=shap_threshold,
                top_n=shap_top_n,
            )

        logger.info("SHAP feature selection enabled (threshold=%s, top_n=%s, sample=%s)",
                    shap_threshold, shap_top_n, shap_sample_size)

    # Load production forecast config for model persistence (F1.1)
    prod_config_path = ROOT / "config" / "production_forecast_config.yaml"
    prod_config = None
    if prod_config_path.exists():
        with open(prod_config_path) as f:
            prod_config = yaml.safe_load(f)

    _fi_fn = registry["feature_importance_fn"]

    def _persistence_fn(models: dict, feature_cols: list[str], timeframe_label: str) -> None:
        persist_cluster_models(
            models, feature_cols, model_id, timeframe_label, prod_config,
            _model_meta_registry,
            feature_importance_fn=_fi_fn,
            model_name=model_name,
        )

    with profiled_section("run_backtest"):
        run_tree_backtest(
            model_id=model_id,
            n_timeframes=n_timeframes,
            output_dir=output_dir,
            model_params=model_params,
            model_params_key=registry["model_params_key"],
            model_type_tag=registry["model_type_tag"],
            train_fn_per_cluster=train_fn,
            extra_metadata={"params_source": params_source},
            cat_dtype=registry["cat_dtype"],
            inline_tuner_fn=inline_tuner_fn,
            feature_selector_fn=feature_selector_fn,
            recursive=recursive,
            model_persistence_fn=_persistence_fn,
            algo_config=algo,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
