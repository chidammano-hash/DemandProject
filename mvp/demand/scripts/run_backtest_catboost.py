"""
Run CatBoost backtesting with expanding-window timeframes.

Supports three strategies:
  - global:      One CatBoost for all DFUs, ml_cluster as categorical feature (model_id=catboost_global)
  - per_cluster:  Separate CatBoost per ml_cluster (model_id=catboost_cluster)
  - transfer:    Global base model (no ml_cluster) → per-cluster fine-tune via init_model (model_id=catboost_transfer)

Produces two CSVs:
  - backtest_predictions.csv: execution-lag only (for fact_external_forecast_monthly)
  - backtest_predictions_all_lags.csv: lag 0-4 archive (for backtest_lag_archive)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import catboost as cb
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


# ── CatBoost training functions (model-specific) ─────────────────────────────


def train_and_predict_global(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict,
) -> tuple[pd.DataFrame, Any]:
    """Train one global CatBoost and predict."""
    t0 = time.time()
    X_train = train_df[feature_cols]
    y_train = train_df["qty"]
    X_pred = predict_df[feature_cols]

    cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

    print(f"    [{_ts()}] Training CatBoost global ({len(X_train):,} rows, {len(feature_cols)} features)...")
    model = cb.CatBoostRegressor(**params)
    model.fit(X_train, y_train, cat_features=cat_indices, verbose=False)
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
    """Train separate CatBoost per ml_cluster."""
    feat_cols_no_cluster = [c for c in feature_cols if c != "ml_cluster"]
    cat_cols_no_cluster = [c for c in cat_cols if c != "ml_cluster"]
    cat_indices_no_cluster = [feat_cols_no_cluster.index(c) for c in cat_cols_no_cluster if c in feat_cols_no_cluster]

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
        model = cb.CatBoostRegressor(**params)
        model.fit(X_train, y_train, cat_features=cat_indices_no_cluster, verbose=False)
        preds = model.predict(X_pred)

        result = pred_c[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
        result["basefcst_pref"] = np.maximum(preds, 0)
        all_results.append(result)
        models[cluster_label] = model
        print(f"    [{_ts()}] Cluster {ci}/{len(clusters)} '{cluster_label}': "
              f"train={len(train_c):,}, pred={len(pred_c):,} ({time.time() - t0:.1f}s)")

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
    transfer_iterations: int = 100,
    transfer_min_rows: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Transfer learning: global base → per-cluster fine-tune via init_model."""
    feat_cols_no_cluster = [c for c in feature_cols if c != "ml_cluster"]
    cat_cols_no_cluster = [c for c in cat_cols if c != "ml_cluster"]
    cat_indices_no_cluster = [feat_cols_no_cluster.index(c) for c in cat_cols_no_cluster if c in feat_cols_no_cluster]

    t0 = time.time()
    X_train_all = train_df[feat_cols_no_cluster]
    y_train_all = train_df["qty"]

    print(f"    [{_ts()}] Phase 1: Training base CatBoost ({len(X_train_all):,} rows, "
          f"{len(feat_cols_no_cluster)} features, no ml_cluster)...")
    base_model = cb.CatBoostRegressor(**params)
    base_model.fit(X_train_all, y_train_all, cat_features=cat_indices_no_cluster, verbose=False)
    print(f"    [{_ts()}] Base model trained ({time.time() - t0:.1f}s)")

    all_results = []
    models = {"__base__": base_model}

    clusters = sorted(train_df["ml_cluster"].dropna().unique())
    clusters = [c for c in clusters if c != "__unknown__"]
    print(f"    [{_ts()}] Phase 2: Fine-tuning {len(clusters)} clusters "
          f"(min_rows={transfer_min_rows}, extra_iters={transfer_iterations})...")

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
        ft_params = {**params, "iterations": transfer_iterations}
        ft_model = cb.CatBoostRegressor(**ft_params)
        ft_model.fit(
            X_train_c, y_train_c,
            cat_features=cat_indices_no_cluster,
            init_model=base_model,
            verbose=False,
        )
        preds = ft_model.predict(X_pred)

        result = pred_c[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
        result["basefcst_pref"] = np.maximum(preds, 0)
        all_results.append(result)
        models[cluster_label] = ft_model
        print(f"    [{_ts()}] Cluster {ci}/{len(clusters)} '{cluster_label}': "
              f"train={len(train_c):,}, pred={len(pred_c):,}, fine-tuned ({time.time() - t1:.1f}s)")

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
    parser = argparse.ArgumentParser(description="Run CatBoost backtest with expanding-window timeframes")
    parser.add_argument("--cluster-strategy", choices=["global", "per_cluster", "transfer"], default="global",
                        help="global: one model, per_cluster: model per ml_cluster, transfer: global base → per-cluster fine-tune")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model_id (default: catboost_global, catboost_cluster, or catboost_transfer)")
    parser.add_argument("--n-timeframes", type=int, default=10, help="Number of expanding windows")
    parser.add_argument("--output-dir", type=str, default="data/backtest", help="Output directory")
    parser.add_argument("--transfer-iterations", type=int, default=100,
                        help="Number of additional iterations for per-cluster fine-tuning (transfer strategy)")
    parser.add_argument("--transfer-min-rows", type=int, default=20,
                        help="Minimum cluster rows for fine-tuning; smaller clusters use base model (transfer strategy)")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    _default_model_ids = {
        "global": "catboost_global",
        "per_cluster": "catboost_cluster",
        "transfer": "catboost_transfer",
    }
    model_id = args.model_id or _default_model_ids[args.cluster_strategy]

    # Detect GPU support for CatBoost
    _use_gpu = False
    try:
        _test = cb.CatBoostRegressor(task_type="GPU", iterations=1, verbose=0)
        _test.fit([[0]], [0])
        _use_gpu = True
        print(f"[{_ts()}] Using GPU for CatBoost")
    except Exception:
        print(f"[{_ts()}] GPU not available, falling back to CPU")

    cb_params = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_seed": args.random_seed,
        "loss_function": "RMSE",
        "verbose": 0,
    }
    if _use_gpu:
        cb_params["task_type"] = "GPU"

    run_tree_backtest(
        model_id=model_id,
        cluster_strategy=args.cluster_strategy,
        n_timeframes=args.n_timeframes,
        output_dir=ROOT / args.output_dir,
        model_params=cb_params,
        model_params_key="catboost_params",
        model_type_tag="catboost_backtest",
        train_fn_global=train_and_predict_global,
        train_fn_per_cluster=train_and_predict_per_cluster,
        train_fn_transfer=train_and_predict_transfer,
        transfer_kwargs={
            "transfer_iterations": args.transfer_iterations,
            "transfer_min_rows": args.transfer_min_rows,
        } if args.cluster_strategy == "transfer" else None,
        extra_metadata={
            "transfer_iterations": args.transfer_iterations,
            "transfer_min_rows": args.transfer_min_rows,
        } if args.cluster_strategy == "transfer" else None,
        cat_dtype="str",
    )


if __name__ == "__main__":
    main()
