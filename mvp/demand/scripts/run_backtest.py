"""
Run LGBM backtesting with expanding-window timeframes.

Supports three strategies:
  - global:      One LGBM for all DFUs, ml_cluster as categorical feature (model_id=lgbm_global)
  - per_cluster:  Separate LGBM per ml_cluster (model_id=lgbm_cluster)
  - transfer:    Global base model (no ml_cluster) → per-cluster fine-tune via init_model (model_id=lgbm_transfer)

Produces two CSVs:
  - backtest_predictions.csv: execution-lag only (for fact_external_forecast_monthly)
  - backtest_predictions_all_lags.csv: lag 0-4 archive (for backtest_lag_archive)
"""

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.backtest_framework import run_tree_backtest
from common.constants import MIN_CLUSTER_ROWS


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ── LGBM training functions (model-specific) ─────────────────────────────────


def train_and_predict_global(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict,
) -> tuple[pd.DataFrame, Any]:
    """Train one global LGBM and predict."""
    t0 = time.time()
    X_train = train_df[feature_cols]
    y_train = train_df["qty"]
    X_pred = predict_df[feature_cols]

    print(f"    [{_ts()}] Training LGBM global ({len(X_train):,} rows, {len(feature_cols)} features)...")
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, categorical_feature=cat_cols)
    print(f"    [{_ts()}] Training done ({time.time() - t0:.1f}s)")

    print(f"    [{_ts()}] Predicting {len(X_pred):,} rows...")
    preds = model.predict(X_pred)
    result = predict_df[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
    result["basefcst_pref"] = np.maximum(preds, 0)
    print(f"    [{_ts()}] Prediction done ({time.time() - t0:.1f}s total)")
    return result, model


def train_and_predict_per_cluster(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict,
) -> tuple[pd.DataFrame, dict]:
    """Train separate LGBM per ml_cluster."""
    feat_cols_no_cluster = [c for c in feature_cols if c != "ml_cluster"]
    cat_cols_no_cluster = [c for c in cat_cols if c != "ml_cluster"]

    all_results = []
    models = {}

    clusters = sorted(train_df["ml_cluster"].dropna().unique())
    print(f"    [{_ts()}] Training {len(clusters)} per-cluster models...")
    for ci, cluster_label in enumerate(clusters, 1):
        train_c = train_df[train_df["ml_cluster"] == cluster_label]
        pred_c = predict_df[predict_df["ml_cluster"] == cluster_label]

        if len(train_c) < MIN_CLUSTER_ROWS or len(pred_c) == 0:
            if len(pred_c) > 0:
                print(f"    [{_ts()}] Cluster {ci}/{len(clusters)} '{cluster_label}': skipped (train={len(train_c)}), "
                      f"zeroing {len(pred_c)} predictions")
                result = pred_c[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
                result["basefcst_pref"] = 0.0
                all_results.append(result)
            continue

        X_train = train_c[feat_cols_no_cluster]
        y_train = train_c["qty"]
        X_pred = pred_c[feat_cols_no_cluster]

        t0 = time.time()
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, categorical_feature=cat_cols_no_cluster)
        preds = model.predict(X_pred)

        result = pred_c[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
        result["basefcst_pref"] = np.maximum(preds, 0)
        all_results.append(result)
        models[cluster_label] = model
        print(f"    [{_ts()}] Cluster {ci}/{len(clusters)} '{cluster_label}': "
              f"train={len(train_c):,}, pred={len(pred_c):,} ({time.time() - t0:.1f}s)")

    # Handle DFUs with no cluster assignment
    no_cluster = predict_df[predict_df["ml_cluster"].isna() | (predict_df["ml_cluster"] == "__unknown__")]
    if len(no_cluster) > 0:
        print(f"    [{_ts()}] {len(no_cluster)} predict rows with no cluster → using 0")
        result = no_cluster[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
        result["basefcst_pref"] = 0.0
        all_results.append(result)

    return pd.concat(all_results, ignore_index=True), models


def train_and_predict_transfer(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict,
    transfer_n_estimators: int = 100,
    transfer_min_rows: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Transfer learning: global base → per-cluster fine-tune via init_model."""
    feat_cols_no_cluster = [c for c in feature_cols if c != "ml_cluster"]
    cat_cols_no_cluster = [c for c in cat_cols if c != "ml_cluster"]

    # Phase 1: Train base model on ALL data (no ml_cluster)
    t0 = time.time()
    X_train_all = train_df[feat_cols_no_cluster]
    y_train_all = train_df["qty"]

    print(f"    [{_ts()}] Phase 1: Training base LGBM ({len(X_train_all):,} rows, "
          f"{len(feat_cols_no_cluster)} features, no ml_cluster)...")
    base_model = lgb.LGBMRegressor(**params)
    base_model.fit(X_train_all, y_train_all, categorical_feature=cat_cols_no_cluster)
    print(f"    [{_ts()}] Base model trained ({time.time() - t0:.1f}s)")

    # Phase 2: Fine-tune per cluster
    all_results = []
    models = {"__base__": base_model}

    clusters = sorted(train_df["ml_cluster"].dropna().unique())
    clusters = [c for c in clusters if c != "__unknown__"]
    print(f"    [{_ts()}] Phase 2: Fine-tuning {len(clusters)} clusters "
          f"(min_rows={transfer_min_rows}, extra_trees={transfer_n_estimators})...")

    for ci, cluster_label in enumerate(clusters, 1):
        train_c = train_df[train_df["ml_cluster"] == cluster_label]
        pred_c = predict_df[predict_df["ml_cluster"] == cluster_label]

        if len(pred_c) == 0:
            continue

        if len(train_c) < transfer_min_rows:
            print(f"    [{_ts()}] Cluster {ci}/{len(clusters)} '{cluster_label}': "
                  f"train={len(train_c)} < {transfer_min_rows} → base model fallback")
            X_pred = pred_c[feat_cols_no_cluster]
            preds = base_model.predict(X_pred)
            result = pred_c[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
            result["basefcst_pref"] = np.maximum(preds, 0)
            all_results.append(result)
            continue

        X_train_c = train_c[feat_cols_no_cluster]
        y_train_c = train_c["qty"]
        X_pred = pred_c[feat_cols_no_cluster]

        t1 = time.time()
        ft_params = {**params, "n_estimators": transfer_n_estimators}
        ft_model = lgb.LGBMRegressor(**ft_params)
        ft_model.fit(
            X_train_c, y_train_c,
            categorical_feature=cat_cols_no_cluster,
            init_model=base_model.booster_,
        )
        preds = ft_model.predict(X_pred)

        result = pred_c[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
        result["basefcst_pref"] = np.maximum(preds, 0)
        all_results.append(result)
        models[cluster_label] = ft_model
        print(f"    [{_ts()}] Cluster {ci}/{len(clusters)} '{cluster_label}': "
              f"train={len(train_c):,}, pred={len(pred_c):,}, fine-tuned ({time.time() - t1:.1f}s)")

    # Handle DFUs with no cluster assignment → base model fallback
    no_cluster = predict_df[predict_df["ml_cluster"].isna() | (predict_df["ml_cluster"] == "__unknown__")]
    if len(no_cluster) > 0:
        print(f"    [{_ts()}] {len(no_cluster)} predict rows with no cluster → base model fallback")
        X_pred = no_cluster[feat_cols_no_cluster]
        preds = base_model.predict(X_pred)
        result = no_cluster[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
        result["basefcst_pref"] = np.maximum(preds, 0)
        all_results.append(result)

    return pd.concat(all_results, ignore_index=True), models


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LGBM backtest with expanding-window timeframes")
    parser.add_argument("--cluster-strategy", choices=["global", "per_cluster", "transfer"], default="global",
                        help="global: one model, per_cluster: model per ml_cluster, transfer: global base → per-cluster fine-tune")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model_id (default: lgbm_global, lgbm_cluster, or lgbm_transfer)")
    parser.add_argument("--n-timeframes", type=int, default=10, help="Number of expanding windows")
    parser.add_argument("--output-dir", type=str, default="data/backtest", help="Output directory")
    parser.add_argument("--transfer-n-estimators", type=int, default=100,
                        help="Number of additional trees for per-cluster fine-tuning (transfer strategy)")
    parser.add_argument("--transfer-min-rows", type=int, default=20,
                        help="Minimum cluster rows for fine-tuning; smaller clusters use base model (transfer strategy)")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--verbosity", type=int, default=-1, help="LightGBM verbosity (-1=silent)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    _default_model_ids = {
        "global": "lgbm_global",
        "per_cluster": "lgbm_cluster",
        "transfer": "lgbm_transfer",
    }
    model_id = args.model_id or _default_model_ids[args.cluster_strategy]

    # Detect Apple GPU support for LightGBM
    _use_gpu = False
    if platform.system() == "Darwin":
        try:
            _test = lgb.LGBMRegressor(device="gpu", n_estimators=1, verbosity=-1)
            _test.fit([[0]], [0])
            _use_gpu = True
            print(f"[{_ts()}] Using Apple GPU (OpenCL) for LightGBM")
        except Exception:
            print(f"[{_ts()}] GPU not available, falling back to CPU")

    lgbm_params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "verbosity": args.verbosity,
        "random_state": 42,
        "n_jobs": -1,
    }
    if _use_gpu:
        lgbm_params["device"] = "gpu"

    run_tree_backtest(
        model_id=model_id,
        cluster_strategy=args.cluster_strategy,
        n_timeframes=args.n_timeframes,
        output_dir=ROOT / args.output_dir,
        model_params=lgbm_params,
        model_params_key="lgbm_params",
        model_type_tag="lgbm_backtest",
        train_fn_global=train_and_predict_global,
        train_fn_per_cluster=train_and_predict_per_cluster,
        train_fn_transfer=train_and_predict_transfer,
        transfer_kwargs={
            "transfer_n_estimators": args.transfer_n_estimators,
            "transfer_min_rows": args.transfer_min_rows,
        } if args.cluster_strategy == "transfer" else None,
        extra_metadata={
            "transfer_n_estimators": args.transfer_n_estimators,
            "transfer_min_rows": args.transfer_min_rows,
        } if args.cluster_strategy == "transfer" else None,
        cat_dtype="category",
    )


if __name__ == "__main__":
    main()
