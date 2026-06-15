"""Tree-based model wrapper for the Expert Panel test.

Wraps existing LGBM/CatBoost/XGBoost infrastructure from common.ml.model_registry.
Trains one model per demand archetype when classification_df is provided (preferred),
falling back to per-ml_cluster training when it is not.
"""

import importlib
import logging
from typing import Any

import numpy as np
import pandas as pd

from common.core.constants import CAT_FEATURES, FORECAST_QTY_COL
from common.core.utils import get_algorithm_params, load_config, load_forecast_pipeline_config
from common.ml.feature_engineering import get_feature_columns
from common.ml.model_registry import (
    UnknownAlgorithm,
    build_tree_model,
    fit_model,
    get_best_iteration,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model library registry — maps model_name to import path and max-iteration param.
# Estimator construction is delegated to model_registry.build_tree_model; the
# module import here is only needed so fit_model can wire early-stopping callbacks.
# ---------------------------------------------------------------------------

_MODEL_LIB: dict[str, dict[str, str]] = {
    "lgbm": {"module": "lightgbm", "iter_param": "n_estimators"},
    "catboost": {"module": "catboost", "iter_param": "iterations"},
    "xgboost": {"module": "xgboost", "iter_param": "n_estimators"},
}

# Minimum training rows per partition group before we skip it
_MIN_GROUP_ROWS = 50


def _extract_model_params(model_name: str, algo_section: dict[str, Any]) -> dict[str, Any]:
    """Extract model constructor params from the algorithm_config section.

    Strips meta keys (enabled, model_id, cluster_strategy, recursive,
    shap_select, etc.) and returns only hyperparameter keys suitable
    for passing to the model constructor.
    """
    meta_keys = {
        "enabled", "model_id", "cluster_strategy", "recursive",
        "shap_select", "shap_threshold", "shap_top_n", "shap_sample_size",
        "tune_inline", "params_file",
    }
    params = {k: v for k, v in algo_section.items() if k not in meta_keys and v is not None}

    # Add model-specific defaults not in config
    if model_name == "lgbm":
        params.setdefault("verbosity", -1)
        params.setdefault("random_state", 42)
        params.setdefault("n_jobs", -1)
        params.setdefault("feature_pre_filter", True)
    elif model_name == "catboost":
        params.setdefault("verbose", 0)
        params.setdefault("random_seed", 42)
        params.setdefault("thread_count", -1)
        params.setdefault("loss_function", "RMSE")
    elif model_name == "xgboost":
        params.setdefault("verbosity", 0)
        params.setdefault("random_state", 42)
        params.setdefault("n_jobs", -1)
        params.setdefault("enable_categorical", True)
        params.setdefault("tree_method", "hist")

    return params


def _prepare_cat_features(
    df: pd.DataFrame,
    cat_cols: list[str],
    model_name: str,
) -> pd.DataFrame:
    """Cast categorical columns to the dtype expected by each model library.

    Handles both the known CAT_FEATURES *and* any extra object/string columns
    that come from item_attrs (category, brand_name, class, etc.).

    Returns a copy of df with columns converted.
    """
    out = df.copy()

    # Collect all columns that need categorical treatment:
    # explicitly listed cat_cols + any remaining object/string dtype columns
    obj_cols = set(out.select_dtypes(include=["object", "string"]).columns)
    all_cat = set(cat_cols) | obj_cols

    for col in all_cat:
        if col not in out.columns:
            continue
        if model_name == "catboost":
            # .astype(object) first to escape Categorical restrictions on fillna
            out[col] = out[col].astype(object).fillna("__NA__").astype(str)
        else:
            # LGBM and XGBoost both use pandas category dtype
            out[col] = out[col].astype("category")
    return out


def run_tree_models(
    grid: pd.DataFrame,
    train_end: pd.Timestamp,
    predict_months: list[pd.Timestamp],
    enabled_models: dict[str, dict],
    algo_config: dict[str, Any] | None = None,
    classification_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run enabled tree models (LGBM/CatBoost/XGBoost) for one timeframe.

    Partition strategy (controls which DFUs train together):
    - When classification_df is provided: one model per demand archetype.
      The archetype column is merged into the grid and used as both the
      partitioning key and a categorical feature.
    - When classification_df is None: falls back to per-ml_cluster training
      (original behaviour, matching the production backtest pattern).

    Args:
        grid: Full feature matrix from build_feature_matrix(). Contains all months.
        train_end: Last month of training data (inclusive).
        predict_months: Months to predict.
        enabled_models: {model_name: config_dict} for enabled tree models.
            model_name is 'lgbm', 'catboost', or 'xgboost'.
        algo_config: Optional override for algorithm_config (legacy format).
            If None, builds a compat dict from forecast_pipeline_config.yaml.
        classification_df: Optional DFU demand classification with columns
            [sku_ck, archetype]. When provided, partitions by archetype instead
            of ml_cluster and adds archetype as a categorical feature.

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref, algorithm_id
        algorithm_id matches the model_id from algo_config (e.g. 'lgbm_cluster').
    """
    if algo_config is None:
        # Build a legacy-compatible dict from pipeline config
        pcfg = load_forecast_pipeline_config()
        algo_config = {"algorithms": {}}
        for mid, entry in pcfg.get("algorithms", {}).items():
            config_key = entry.get("config_key", mid)
            params = entry.get("params", {})
            algo_config["algorithms"][config_key] = {
                **params,
                "model_id": mid,
                "enabled": entry.get("enabled", True),
                "cluster_strategy": entry.get("cluster_strategy", "per_cluster"),
            }

    # Merge archetype into grid when classification_df is provided
    working_grid = grid.copy()
    if classification_df is not None and "archetype" in classification_df.columns:
        archetype_map = classification_df[["sku_ck", "archetype"]].drop_duplicates("sku_ck")
        working_grid = working_grid.merge(archetype_map, on="sku_ck", how="left")
        working_grid["archetype"] = working_grid["archetype"].fillna("unclassified")
        partition_col = "archetype"
        partition_label = "archetype"
        logger.info("Tree models: partitioning by demand archetype")
    else:
        partition_col = "ml_cluster"
        partition_label = "cluster"
        logger.info("Tree models: partitioning by ml_cluster (no classification_df provided)")

    feature_cols = get_feature_columns(working_grid)
    # Cat cols = known CAT_FEATURES + archetype (if present) + any object/string columns
    obj_cols = set(working_grid[feature_cols].select_dtypes(include=["object", "string"]).columns)
    cat_col_set = {c for c in feature_cols if c in CAT_FEATURES} | obj_cols
    if partition_col == "archetype" and "archetype" in working_grid.columns:
        cat_col_set.add("archetype")
    cat_cols = list(cat_col_set)

    # Temporal split
    train_mask = working_grid["startdate"] <= train_end
    predict_mask = working_grid["startdate"].isin(predict_months)
    train_df = working_grid[train_mask]
    predict_df = working_grid[predict_mask]

    if len(predict_df) == 0:
        logger.warning("No prediction rows for predict_months=%s", predict_months)
        return pd.DataFrame(columns=["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"])

    all_model_results: list[pd.DataFrame] = []

    for model_name, _model_cfg in enabled_models.items():
        if model_name not in _MODEL_LIB:
            logger.warning("Unknown tree model '%s'; skipping", model_name)
            continue

        lib_info = _MODEL_LIB[model_name]
        algo_section = algo_config.get("algorithms", {}).get(model_name, {})
        model_id = algo_section.get("model_id", f"{model_name}_cluster")

        # Import library module (passed to fit_model for early-stopping callbacks).
        # Estimator construction itself is delegated to build_tree_model below.
        try:
            lib_module = importlib.import_module(lib_info["module"])
        except ImportError:
            logger.warning(
                "Library '%s' not installed; skipping %s",
                lib_info["module"], model_name,
            )
            continue

        # Build params from algo_config
        params = _extract_model_params(model_name, algo_section)
        iter_param = lib_info["iter_param"]
        max_iterations = params.get(iter_param, 1000)

        groups = sorted(train_df[partition_col].dropna().unique())
        logger.info(
            "Running %s (model_id=%s, max_iter=%d) across %d %ss",
            model_name.upper(), model_id, max_iterations, len(groups), partition_label,
        )

        group_results: list[pd.DataFrame] = []

        for gi, group_label in enumerate(groups, 1):
            train_g = train_df[train_df[partition_col] == group_label]
            pred_g = predict_df[predict_df[partition_col] == group_label]

            if len(pred_g) == 0:
                continue

            if len(train_g) < _MIN_GROUP_ROWS:
                logger.warning(
                    "%s %s %d/%d '%s': skipped (train=%d < %d)",
                    model_name.upper(), partition_label, gi, len(groups),
                    group_label, len(train_g), _MIN_GROUP_ROWS,
                )
                continue

            # Sort training data by date for temporal val split
            train_g = train_g.sort_values("startdate")

            # Time-aware train/val split — last 20% of months as validation
            n_val = max(1, int(len(train_g) * 0.20))
            train_part = train_g.iloc[:-n_val]
            val_part = train_g.iloc[-n_val:]

            X_tr = train_part[feature_cols]
            y_tr = train_part["qty"]
            X_val = val_part[feature_cols]
            y_val = val_part["qty"]
            X_pred = pred_g[feature_cols]

            # Handle categorical features per model type
            X_tr = _prepare_cat_features(X_tr, cat_cols, model_name)
            X_val = _prepare_cat_features(X_val, cat_cols, model_name)
            X_pred = _prepare_cat_features(X_pred, cat_cols, model_name)

            # Constant target guard — models crash on constant targets
            if y_tr.nunique() <= 1:
                const_val = float(y_tr.iloc[0]) if len(y_tr) > 0 else 0.0
                logger.info(
                    "%s %s %d/%d '%s': constant target=%.0f, "
                    "using constant for %d predictions",
                    model_name.upper(), partition_label, gi, len(groups),
                    group_label, const_val, len(pred_g),
                )
                result = pred_g[["sku_ck", "startdate"]].copy()
                result[FORECAST_QTY_COL] = max(const_val, 0.0)
                result["algorithm_id"] = model_id
                group_results.append(result)
                continue

            # Instantiate through the model registry — the single source of
            # truth for constructing LGBM/CatBoost/XGBoost estimators. The
            # registry translates canonical param names to each library's
            # native names; native keys pass through unchanged.
            try:
                model = build_tree_model(model_name, params)
            except (TypeError, UnknownAlgorithm) as te:
                logger.warning(
                    "%s %s %d/%d '%s': instantiation failed: %s",
                    model_name.upper(), partition_label, gi, len(groups),
                    group_label, te,
                )
                continue

            # Unified fit via model_registry
            try:
                fit_model(
                    model, model_name,
                    X_tr, y_tr, X_val, y_val,
                    cat_cols, feature_cols,
                    lib_module, max_iterations,
                )
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "%s %s %d/%d '%s': fit failed: %s",
                    model_name.upper(), partition_label, gi, len(groups),
                    group_label, exc,
                )
                continue

            # Predict and clip to non-negative
            preds = model.predict(X_pred)
            preds = np.maximum(preds, 0.0)

            best_iter = get_best_iteration(model, model_name)
            if best_iter is None:
                best_iter = max_iterations

            logger.info(
                "%s %s %d/%d '%s': train=%d, pred=%d, best_iter=%s",
                model_name.upper(), partition_label, gi, len(groups),
                group_label, len(train_g), len(pred_g), best_iter,
            )

            result = pred_g[["sku_ck", "startdate"]].copy()
            result[FORECAST_QTY_COL] = preds
            result["algorithm_id"] = model_id
            group_results.append(result)

        if group_results:
            model_result = pd.concat(group_results, ignore_index=True)
            all_model_results.append(model_result)
            logger.info(
                "%s complete: %d predictions across %d %ss",
                model_name.upper(), len(model_result), len(group_results), partition_label,
            )
        else:
            logger.warning("%s: no predictions produced", model_name.upper())

    if not all_model_results:
        logger.warning("No tree model predictions produced")
        return pd.DataFrame(columns=["sku_ck", "startdate", FORECAST_QTY_COL, "algorithm_id"])

    combined = pd.concat(all_model_results, ignore_index=True)
    logger.info(
        "Tree models complete: %d total predictions from %d algorithms",
        len(combined), len(all_model_results),
    )
    return combined
