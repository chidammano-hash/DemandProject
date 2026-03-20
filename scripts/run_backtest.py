"""
Run LGBM backtesting with per-cluster strategy and expanding-window timeframes.

All run options (recursive, SHAP, tuning) are controlled via
config/algorithm_config.yaml rather than CLI flags.

Produces two CSVs under data/backtest/lgbm_cluster/:
  - backtest_predictions.csv          (execution-lag row for DB load)
  - backtest_predictions_all_lags.csv (lag 0-4 archive)
"""

import json
import logging
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.backtest_framework import run_tree_backtest
from common.constants import MIN_CLUSTER_ROWS
from common.tuning import TRAIN_FOLD_FNS, load_best_params, tune_for_timeframe

logger = logging.getLogger(__name__)


# ── LGBM per-cluster training function ───────────────────────────────────────


def train_and_predict_per_cluster(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict,
) -> tuple[pd.DataFrame, dict, dict]:
    """Train separate LGBM per ml_cluster.

    ml_cluster is kept as a hard feature (constant within each cluster partition,
    but required for consistent feature alignment with global models).

    Returns (predictions, models, model_meta) where model_meta stores per-cluster
    training metadata (val_wape, train_rows) without monkey-patching sklearn objects.
    """
    all_results = []
    models = {}
    model_meta: dict[str, dict] = {}  # metadata stored separately, not monkey-patched onto model

    clusters = sorted(train_df["ml_cluster"].dropna().unique())
    logger.info("Training %d per-cluster LGBM models...", len(clusters))
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

        X_train = train_c[feature_cols]
        y_train = train_c["qty"]
        X_pred = pred_c[feature_cols]

        t0 = time.time()
        # Time-aware train/val split — last 20% of cluster rows for validation
        n_val = max(1, int(len(X_train) * 0.20))
        X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
        y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        fit_params = {**params, "n_estimators": max(params.get("n_estimators", 1000), 1000)}
        model = lgb.LGBMRegressor(**fit_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            categorical_feature=cat_cols,
            callbacks=callbacks,
        )
        preds = model.predict(X_pred)

        # Per-cluster validation WAPE
        val_preds = model.predict(X_val)
        val_denom = float(abs(y_val.sum()))
        val_wape = round(float((abs(val_preds - y_val.values)).sum() / val_denom * 100), 2) if val_denom > 0 else 0.0

        result = pred_c[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
        result["basefcst_pref"] = np.maximum(preds, 0)
        all_results.append(result)
        models[cluster_label] = model
        n_est_used = model.best_iteration_ if model.best_iteration_ is not None else fit_params["n_estimators"]
        model_meta[cluster_label] = {"val_wape": val_wape, "train_rows": len(X_tr)}
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

    return pd.concat(all_results, ignore_index=True), models, model_meta


def train_and_predict_global(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict,
) -> tuple[pd.DataFrame, dict, dict]:
    """Train a single global LGBM on ALL data with ml_cluster as categorical feature."""
    # ml_cluster is INCLUDED as a feature — do NOT strip it
    logger.info("Training global LGBM on %s rows, %d features (includes ml_cluster)...",
                f"{len(train_df):,}", len(feature_cols))

    X_train = train_df[feature_cols]
    y_train = train_df["qty"]
    X_pred = predict_df[feature_cols]

    # Time-aware train/val split — last 15% of rows for validation
    n_val = max(1, int(len(X_train) * 0.15))
    X_tr, X_val = X_train.iloc[:-n_val], X_train.iloc[-n_val:]
    y_tr, y_val = y_train.iloc[:-n_val], y_train.iloc[-n_val:]

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),
    ]
    fit_params = {**params, "n_estimators": max(params.get("n_estimators", 1200), 1200)}
    model = lgb.LGBMRegressor(**fit_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        categorical_feature=cat_cols,
        callbacks=callbacks,
    )
    preds = model.predict(X_pred)

    val_preds = model.predict(X_val)
    val_denom = float(abs(y_val.sum()))
    val_wape = round(float((abs(val_preds - y_val.values)).sum() / val_denom * 100), 2) if val_denom > 0 else 0.0
    n_est_used = model.best_iteration_ if model.best_iteration_ is not None else fit_params["n_estimators"]
    logger.info("Global LGBM: val_WAPE=%.1f%%, best_iter=%s, train=%s, pred=%s",
                val_wape, n_est_used, f"{len(train_df):,}", f"{len(predict_df):,}")

    result = predict_df[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
    result["basefcst_pref"] = np.clip(preds, 0, None)
    # Use "global" as the single key so persist_cluster_models saves one file
    global_meta = {"global": {"val_wape": val_wape, "train_rows": len(X_tr)}}
    return result, {"global": model}, global_meta


def persist_cluster_models(
    models: dict,
    feature_cols: list[str],
    model_id: str,
    timeframe_label: str,
    prod_config: dict | None = None,
    model_meta: dict[str, dict] | None = None,
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
    """
    if not models:
        return

    base_path = "data/models"
    if prod_config:
        base_path = prod_config.get("model_registry", {}).get("base_path", "data/models")

    out_dir = ROOT / base_path / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    _meta = model_meta or {}
    saved = 0
    for cluster_label, model in models.items():
        n_est_used = getattr(model, "best_iteration_", None) or 0
        importance_raw = model.feature_importances_
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
        imp = getattr(model, "feature_importances_", None)
        if imp is not None and len(imp) == len(feature_cols):
            fi_dict = dict(zip(feature_cols, [float(v) for v in imp]))
            fi_sorted = dict(sorted(fi_dict.items(), key=lambda x: x[1], reverse=True))
            with open(fi_dir / f"cluster_{cluster_label}.json", "w") as f:
                json.dump(fi_sorted, f, indent=2)

    logger.info("Persisted %d %s cluster models to %s/ (timeframe=%s)",
                saved, model_id, out_dir, timeframe_label)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run LGBM per-cluster backtest (settings from algorithm_config.yaml)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to algorithm_config.yaml (default: config/algorithm_config.yaml)")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model_id from config")
    parser.add_argument("--n-timeframes", type=int, default=None,
                        help="Override n_timeframes from config")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    # Load algorithm config
    config_path = Path(args.config) if args.config else ROOT / "config" / "algorithm_config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    algo = cfg["algorithms"]["lgbm"]
    backtest_cfg = cfg.get("backtest", {})

    cluster_strategy = algo.get("cluster_strategy", "per_cluster")
    # Registry captures model metadata across timeframes for use in _persistence_fn.
    # Training functions return (result, models, meta); the backtest framework expects
    # only (result, models), so we strip meta here and stash it in the registry.
    _model_meta_registry: dict[str, dict] = {}

    if cluster_strategy == "global":
        _inner_train_fn = train_and_predict_global
        default_model_id = "lgbm_global"
    else:
        _inner_train_fn = train_and_predict_per_cluster
        default_model_id = "lgbm_cluster"

    def train_fn(train_df, predict_df, feature_cols, cat_cols, params):
        result, models, meta = _inner_train_fn(train_df, predict_df, feature_cols, cat_cols, params)
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

    logger.info("LGBM config: model_id=%s, cluster_strategy=%s, recursive=%s, shap_select=%s, "
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
        if platform.system() == "Darwin":
            try:
                _test = lgb.LGBMRegressor(device="gpu", n_estimators=1, verbosity=-1)
                _test.fit([[0]], [0])
                _use_gpu = True
                logger.info("Using Apple GPU (OpenCL) for LightGBM")
            except Exception:
                logger.info("GPU not available, falling back to CPU")

    lgbm_params = {
        "n_estimators": algo.get("n_estimators", 500),
        "learning_rate": algo.get("learning_rate", 0.05),
        "num_leaves": algo.get("num_leaves", 31),
        "min_child_samples": algo.get("min_child_samples", 20),
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    }

    params_source = "config_defaults"
    if params_file:
        tuning_data = load_best_params(Path(params_file))
        tuned = tuning_data.get("best_params", {})
        n_est_tuned = tuning_data.get("best_n_estimators", None)
        lgbm_params.update(tuned)
        if n_est_tuned:
            lgbm_params["n_estimators"] = n_est_tuned
        params_source = f"tuning_file:{params_file}"
        logger.info("Loaded tuned params from %s (best_wape=%s%%, n_est=%s)",
                    params_file, tuning_data.get('best_wape'), lgbm_params['n_estimators'])

    if _use_gpu:
        lgbm_params["device"] = "gpu"

    # Build causal per-timeframe tuner when tune_inline is set (PL-002)
    inline_tuner_fn = None
    if tune_inline:
        _tune_config_path = ROOT / "config" / "hyperparameter_tuning.yaml"
        with open(_tune_config_path) as _f:
            _tune_config = yaml.safe_load(_f)
        _fold_fn = TRAIN_FOLD_FNS["lgbm"]
        _base_params = lgbm_params.copy()

        def inline_tuner_fn(full_grid, feature_cols, cat_cols, train_end):
            tuned, n_est = tune_for_timeframe(
                model_name="lgbm",
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

    def _persistence_fn(models: dict, feature_cols: list[str], timeframe_label: str) -> None:
        persist_cluster_models(models, feature_cols, model_id, timeframe_label, prod_config, _model_meta_registry)

    run_tree_backtest(
        model_id=model_id,
        n_timeframes=n_timeframes,
        output_dir=output_dir,
        model_params=lgbm_params,
        model_params_key="lgbm_params",
        model_type_tag="lgbm_backtest",
        train_fn_per_cluster=train_fn,
        extra_metadata={"params_source": params_source},
        cat_dtype="category",
        inline_tuner_fn=inline_tuner_fn,
        feature_selector_fn=feature_selector_fn,
        recursive=recursive,
        model_persistence_fn=_persistence_fn,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
