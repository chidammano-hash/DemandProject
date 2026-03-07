"""
Run CatBoost backtesting with per-cluster strategy and expanding-window timeframes.

All run options (recursive, SHAP, tuning) are controlled via
config/algorithm_config.yaml rather than CLI flags.

Produces two CSVs under data/backtest/catboost_cluster/:
  - backtest_predictions.csv          (execution-lag row for DB load)
  - backtest_predictions_all_lags.csv (lag 0-4 archive)
"""

import pickle
import sys
import time
from pathlib import Path
from typing import Any

import catboost as cb
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


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ── CatBoost per-cluster training function ────────────────────────────────────


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
    print(f"    [{_ts()}] Training {len(clusters)} per-cluster CatBoost models...")
    for ci, cluster_label in enumerate(clusters, 1):
        train_c = train_df[train_df["ml_cluster"] == cluster_label]
        pred_c = predict_df[predict_df["ml_cluster"] == cluster_label]

        if len(train_c) < MIN_CLUSTER_ROWS or len(pred_c) == 0:
            if len(pred_c) > 0:
                print(f"    [{_ts()}] Cluster {ci}/{len(clusters)} '{cluster_label}': "
                      f"skipped (train={len(train_c)}), zeroing {len(pred_c)} predictions")
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

    no_cluster = predict_df[
        predict_df["ml_cluster"].isna() | (
            (predict_df["ml_cluster"] == "__unknown__") & ("__unknown__" not in models)
        )
    ]
    if len(no_cluster) > 0:
        print(f"    [{_ts()}] {len(no_cluster)} predict rows with no cluster → zeroing")
        result = no_cluster[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
        result["basefcst_pref"] = 0.0
        all_results.append(result)

    return pd.concat(all_results, ignore_index=True), models


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run CatBoost per-cluster backtest (settings from algorithm_config.yaml)")
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

    algo = cfg["algorithms"]["catboost"]
    backtest_cfg = cfg.get("backtest", {})

    model_id = args.model_id or algo.get("model_id", "catboost_cluster")
    n_timeframes = args.n_timeframes or backtest_cfg.get("n_timeframes", 10)
    output_dir = ROOT / backtest_cfg.get("output_dir", "data/backtest")
    recursive = algo.get("recursive", False)
    shap_select = algo.get("shap_select", False)
    shap_threshold = algo.get("shap_threshold", 0.95)
    shap_top_n = algo.get("shap_top_n", None)
    shap_sample_size = algo.get("shap_sample_size", 500)
    tune_inline = algo.get("tune_inline", False)
    params_file = algo.get("params_file", None)

    print(f"[{_ts()}] CatBoost config: model_id={model_id}, recursive={recursive}, "
          f"shap_select={shap_select}, tune_inline={tune_inline}, n_timeframes={n_timeframes}")

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
        "iterations": algo.get("iterations", 500),
        "learning_rate": algo.get("learning_rate", 0.05),
        "depth": algo.get("depth", 6),
        "l2_leaf_reg": algo.get("l2_leaf_reg", 3.0),
        "random_seed": 42,
        "loss_function": "RMSE",
        "verbose": 0,
    }

    params_source = "config_defaults"
    if params_file:
        tuning_data = load_best_params(Path(params_file))
        tuned = tuning_data.get("best_params", {})
        n_est_tuned = tuning_data.get("best_n_estimators", None)
        cb_params.update(tuned)
        if n_est_tuned:
            cb_params["iterations"] = n_est_tuned
        params_source = f"tuning_file:{params_file}"
        print(f"[{_ts()}] Loaded tuned params from {params_file} "
              f"(best_wape={tuning_data.get('best_wape')}%, n_est={cb_params['iterations']})")

    if _use_gpu:
        cb_params["task_type"] = "GPU"

    # Build causal per-timeframe tuner when tune_inline is set (PL-002)
    inline_tuner_fn = None
    if tune_inline:
        _tune_config_path = ROOT / "config" / "hyperparameter_tuning.yaml"
        with open(_tune_config_path) as _f:
            _tune_config = yaml.safe_load(_f)
        _fold_fn = TRAIN_FOLD_FNS["catboost"]
        _base_params = cb_params.copy()

        def inline_tuner_fn(full_grid, feature_cols, cat_cols, train_end):
            tuned, n_est = tune_for_timeframe(
                model_name="catboost",
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
            result = {**_base_params, **tuned, "iterations": n_est}
            print(f"    [{_ts()}] Inline tuned: iterations={n_est}, "
                  f"lr={tuned.get('learning_rate', 'n/a'):.4f}")
            return result

        params_source = "inline_tuning"
        print(f"[{_ts()}] Inline tuning enabled "
              f"(inline_n_trials={_tune_config['tuning'].get('inline_n_trials', 20)}, "
              f"inline_n_splits={_tune_config['tuning'].get('inline_n_splits', 3)})")

    print(f"[{_ts()}] Params source: {params_source}")

    # Build SHAP feature selector closure (Feature 42)
    feature_selector_fn = None
    if shap_select:
        from common.shap_selector import compute_shap_catboost, compute_timeframe_shap

        def feature_selector_fn(model_or_dict, train_data, feature_cols, cat_cols, tf_idx, cutoff):
            return compute_timeframe_shap(
                model_or_dict, train_data, feature_cols, cat_cols,
                tf_idx, cutoff,
                shap_extractor_fn=compute_shap_catboost,
                cluster_strategy="per_cluster",
                sample_size=shap_sample_size,
                cumulative_threshold=shap_threshold,
                top_n=shap_top_n,
            )

        print(f"[{_ts()}] SHAP feature selection enabled "
              f"(threshold={shap_threshold}, top_n={shap_top_n}, sample={shap_sample_size})")

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
            artifact = {"model": model, "feature_cols": feature_cols, "model_id": model_id}
            with open(out_dir / f"cluster_{cluster_label}.pkl", "wb") as f:
                pickle.dump(artifact, f)
        print(f"  [{_ts()}] Persisted {len(models)} {model_id} cluster models (timeframe={timeframe_label})")

    run_tree_backtest(
        model_id=model_id,
        n_timeframes=n_timeframes,
        output_dir=output_dir,
        model_params=cb_params,
        model_params_key="catboost_params",
        model_type_tag="catboost_backtest",
        train_fn_per_cluster=train_and_predict_per_cluster,
        extra_metadata={"params_source": params_source},
        cat_dtype="str",
        inline_tuner_fn=inline_tuner_fn,
        feature_selector_fn=feature_selector_fn,
        recursive=recursive,
        model_persistence_fn=_persist_models,
    )


if __name__ == "__main__":
    main()
