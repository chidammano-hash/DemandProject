"""
Run XGBoost backtesting with per-cluster strategy and expanding-window timeframes.

All run options (recursive, SHAP, tuning) are controlled via
config/algorithm_config.yaml rather than CLI flags.

Produces two CSVs under data/backtest/xgboost_cluster/:
  - backtest_predictions.csv          (execution-lag row for DB load)
  - backtest_predictions_all_lags.csv (lag 0-4 archive)
"""

import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.backtest_framework import run_tree_backtest
from common.constants import MIN_CLUSTER_ROWS
from common.tuning import TRAIN_FOLD_FNS, load_best_params, tune_for_timeframe

logger = logging.getLogger(__name__)


# ── XGBoost per-cluster training function ────────────────────────────────────


def train_and_predict_per_cluster(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict,
) -> tuple[pd.DataFrame, dict]:
    """Train separate XGBoost per ml_cluster.

    ml_cluster is kept as a hard feature (constant within each cluster partition,
    but required for consistent feature alignment with global models).
    """
    all_results = []
    models = {}

    clusters = sorted(train_df["ml_cluster"].dropna().unique())
    logger.info("Training %d per-cluster XGBoost models...", len(clusters))
    for ci, cluster_label in enumerate(clusters, 1):
        train_c = train_df[train_df["ml_cluster"] == cluster_label]
        pred_c = predict_df[predict_df["ml_cluster"] == cluster_label]

        if len(train_c) < MIN_CLUSTER_ROWS or len(pred_c) == 0:
            if len(pred_c) > 0:
                logger.info("Cluster %d/%d '%s': skipped (train=%d), zeroing %d predictions",
                            ci, len(clusters), cluster_label, len(train_c), len(pred_c))
                result = pred_c[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
                result["basefcst_pref"] = 0.0
                all_results.append(result)
            continue

        # Identify categorical columns present in features
        cat_cols_in_features = [c for c in cat_cols if c in feature_cols]

        X_train = train_c[feature_cols].copy()
        y_train = train_c["qty"]
        X_pred = pred_c[feature_cols].copy()

        # Ensure pandas category dtype so XGBoost native categorical support activates
        for col in cat_cols_in_features:
            X_train[col] = X_train[col].astype("category")
            X_pred[col] = X_pred[col].astype("category")

        t0 = time.time()
        # Time-aware train/val split — last 20% of cluster rows for validation
        n_val = max(1, int(len(X_train) * 0.20))
        X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
        y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

        # Guard: constant targets cannot be trained (XGBoost may not converge)
        if y_tr.nunique() <= 1:
            logger.info("Cluster %d/%d '%s': skipped (constant target), zeroing %d predictions",
                        ci, len(clusters), cluster_label, len(pred_c))
            result = pred_c[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
            result["basefcst_pref"] = float(y_tr.iloc[0]) if len(y_tr) > 0 else 0.0
            all_results.append(result)
            continue

        fit_params = {
            **params,
            "n_estimators": max(params.get("n_estimators", 1000), 1000),
            "early_stopping_rounds": 50,
        }
        model = xgb.XGBRegressor(**fit_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_pred)

        # Per-cluster validation WAPE
        val_preds = model.predict(X_val)
        val_denom = float(abs(y_val.sum()))
        val_wape = round(float((abs(val_preds - y_val.values)).sum() / val_denom * 100), 2) if val_denom > 0 else 0.0
        model._val_wape = val_wape
        model._train_rows = len(X_tr)

        result = pred_c[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
        result["basefcst_pref"] = np.maximum(preds, 0)
        all_results.append(result)
        models[cluster_label] = model
        n_est_used = model.best_iteration if model.best_iteration is not None else fit_params["n_estimators"]
        logger.info("Cluster %d/%d '%s': train=%s, pred=%s, best_iter=%s, val_wape=%.1f%% (%.1fs)",
                    ci, len(clusters), cluster_label, f"{len(train_c):,}", f"{len(pred_c):,}",
                    n_est_used, val_wape, time.time() - t0)

    no_cluster = predict_df[
        predict_df["ml_cluster"].isna() | (
            (predict_df["ml_cluster"] == "__unknown__") & ("__unknown__" not in models)
        )
    ]
    if len(no_cluster) > 0:
        logger.info("%d predict rows with no cluster -> zeroing", len(no_cluster))
        result = no_cluster[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
        result["basefcst_pref"] = 0.0
        all_results.append(result)

    return pd.concat(all_results, ignore_index=True), models


def train_and_predict_global(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict,
) -> tuple[pd.DataFrame, dict]:
    """Train a single global XGBoost on ALL data with ml_cluster as categorical feature."""
    # XGBoost uses pandas category dtype for native categorical support
    # ml_cluster is INCLUDED as a feature
    cat_cols_in_features = [c for c in cat_cols if c in feature_cols]
    logger.info("Training global XGBoost on %s rows, %d features (includes ml_cluster)...",
                f"{len(train_df):,}", len(feature_cols))

    X_train = train_df[feature_cols].copy()
    y_train = train_df["qty"]
    X_pred = predict_df[feature_cols].copy()

    # Ensure pandas category dtype so XGBoost native categorical support activates
    for col in cat_cols_in_features:
        X_train[col] = X_train[col].astype("category")
        X_pred[col] = X_pred[col].astype("category")

    # Time-aware train/val split — last 15% of rows for validation
    n_val = max(1, int(len(X_train) * 0.15))
    X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

    fit_params = {
        **params,
        "n_estimators": max(params.get("n_estimators", 1200), 1200),
        "early_stopping_rounds": 50,
    }
    model = xgb.XGBRegressor(**fit_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    preds = model.predict(X_pred)

    val_preds = model.predict(X_val)
    val_denom = float(abs(y_val.sum()))
    val_wape = round(float((abs(val_preds - y_val.values)).sum() / val_denom * 100), 2) if val_denom > 0 else 0.0
    model._val_wape = val_wape
    n_est_used = model.best_iteration if model.best_iteration else fit_params["n_estimators"]
    logger.info("Global XGBoost: val_WAPE=%.1f%%, best_iter=%s, train=%s, pred=%s",
                val_wape, n_est_used, f"{len(train_df):,}", f"{len(predict_df):,}")

    result = predict_df[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
    result["basefcst_pref"] = np.clip(preds, 0, None)
    # Use "global" as the single key so persist logic saves one file
    return result, {"global": model}


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run XGBoost per-cluster backtest (settings from algorithm_config.yaml)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to algorithm_config.yaml (default: config/algorithm_config.yaml)")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model_id from config")
    parser.add_argument("--n-timeframes", type=int, default=None,
                        help="Override n_timeframes from config")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    config_path = Path(args.config) if args.config else ROOT / "config" / "algorithm_config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    algo = cfg["algorithms"]["xgboost"]
    backtest_cfg = cfg.get("backtest", {})

    cluster_strategy = algo.get("cluster_strategy", "per_cluster")
    if cluster_strategy == "global":
        train_fn = train_and_predict_global
        default_model_id = "xgboost_global"
    else:
        train_fn = train_and_predict_per_cluster
        default_model_id = "xgboost_cluster"

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

    logger.info("XGBoost config: model_id=%s, cluster_strategy=%s, recursive=%s, shap_select=%s, "
                "tune_inline=%s, n_timeframes=%d",
                model_id, cluster_strategy, recursive, shap_select, tune_inline, n_timeframes)

    # GPU detection with env-var override: DEMAND_GPU=on|off|auto (default: auto)
    _gpu_pref = os.getenv("DEMAND_GPU", "auto").lower()
    _use_gpu = False
    if _gpu_pref == "on":
        _use_gpu = True
        logger.info("GPU forced ON via DEMAND_GPU env var")
    elif _gpu_pref == "off":
        _use_gpu = False
        logger.info("GPU disabled via DEMAND_GPU env var")
    else:  # auto
        try:
            _test = xgb.XGBRegressor(device="cuda", n_estimators=1, verbosity=0)
            _test.fit([[0]], [0])
            _use_gpu = True
            logger.info("Using GPU (CUDA) for XGBoost")
        except Exception:
            logger.info("GPU not available, falling back to CPU")

    xgb_params = {
        "n_estimators": algo.get("n_estimators", 500),
        "learning_rate": algo.get("learning_rate", 0.05),
        "max_depth": algo.get("max_depth", 6),
        "min_child_weight": algo.get("min_child_weight", 5),
        "subsample": algo.get("subsample", 0.8),
        "colsample_bytree": algo.get("colsample_bytree", 0.8),
        "verbosity": 0,
        "random_state": 42,
        "n_jobs": -1,
        "enable_categorical": True,
        "tree_method": "hist",
    }

    params_source = "config_defaults"
    if params_file:
        tuning_data = load_best_params(Path(params_file))
        tuned = tuning_data.get("best_params", {})
        n_est_tuned = tuning_data.get("best_n_estimators", None)
        xgb_params.update(tuned)
        if n_est_tuned:
            xgb_params["n_estimators"] = n_est_tuned
        params_source = f"tuning_file:{params_file}"
        logger.info("Loaded tuned params from %s (best_wape=%s%%, n_est=%s)",
                    params_file, tuning_data.get('best_wape'), xgb_params['n_estimators'])

    if _use_gpu:
        xgb_params["device"] = "cuda"

    # Build causal per-timeframe tuner when tune_inline is set (PL-002)
    inline_tuner_fn = None
    if tune_inline:
        _tune_config_path = ROOT / "config" / "hyperparameter_tuning.yaml"
        with open(_tune_config_path) as _f:
            _tune_config = yaml.safe_load(_f)
        _fold_fn = TRAIN_FOLD_FNS["xgboost"]
        _base_params = xgb_params.copy()

        def inline_tuner_fn(full_grid, feature_cols, cat_cols, train_end):
            tuned, n_est = tune_for_timeframe(
                model_name="xgboost",
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
            result = {**_base_params, **tuned, "n_estimators": n_est}
            logger.info("Inline tuned: n_estimators=%s, lr=%.4f",
                        n_est, tuned.get('learning_rate', 0))
            return result

        params_source = "inline_tuning"
        logger.info("Inline tuning enabled (inline_n_trials=%s, inline_n_splits=%s)",
                    _tune_config['tuning'].get('inline_n_trials', 20),
                    _tune_config['tuning'].get('inline_n_splits', 3))

    logger.info("Params source: %s", params_source)

    # Build SHAP feature selector closure (Feature 42)
    feature_selector_fn = None
    if shap_select:
        from common.shap_selector import compute_shap_global, compute_timeframe_shap

        def feature_selector_fn(model_or_dict, train_data, feature_cols, cat_cols, tf_idx, cutoff):
            return compute_timeframe_shap(
                model_or_dict, train_data, feature_cols, cat_cols,
                tf_idx, cutoff,
                shap_extractor_fn=compute_shap_global,
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

    def _persist_models(models: dict, feature_cols: list[str], timeframe_label: str) -> None:
        base_path = "data/models"
        if prod_config:
            base_path = prod_config.get("model_registry", {}).get("base_path", "data/models")
        out_dir = ROOT / base_path / model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for cluster_label, model in models.items():
            n_est_used = getattr(model, "best_iteration", None) or 0
            importance_raw = model.feature_importances_
            importance_dict = dict(zip(feature_cols, [float(v) for v in importance_raw])) if len(importance_raw) == len(feature_cols) else {}
            artifact = {
                "model": model,
                "feature_cols": feature_cols,
                "model_id": model_id,
                "cluster_label": str(cluster_label),
                "n_estimators_used": n_est_used,
                "train_rows": getattr(model, "_train_rows", None),
                "val_wape": getattr(model, "_val_wape", None),
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "timeframe": timeframe_label,
                "feature_importance": importance_dict,
            }
            with open(out_dir / f"cluster_{cluster_label}.pkl", "wb") as f:
                pickle.dump(artifact, f)

        # Write feature importance JSON per cluster
        fi_dir = out_dir / "feature_importance"
        fi_dir.mkdir(parents=True, exist_ok=True)
        for cluster_label, model in models.items():
            imp = model.feature_importances_
            if len(imp) == len(feature_cols):
                fi_sorted = dict(sorted(zip(feature_cols, [float(v) for v in imp]), key=lambda x: x[1], reverse=True))
                with open(fi_dir / f"cluster_{cluster_label}.json", "w") as f:
                    json.dump(fi_sorted, f, indent=2)

        logger.info("Persisted %d %s cluster models (timeframe=%s)",
                    len(models), model_id, timeframe_label)

    run_tree_backtest(
        model_id=model_id,
        n_timeframes=n_timeframes,
        output_dir=output_dir,
        model_params=xgb_params,
        model_params_key="xgboost_params",
        model_type_tag="xgboost_backtest",
        train_fn_per_cluster=train_fn,
        extra_metadata={"params_source": params_source},
        cat_dtype="category",
        inline_tuner_fn=inline_tuner_fn,
        feature_selector_fn=feature_selector_fn,
        recursive=recursive,
        model_persistence_fn=_persist_models,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
