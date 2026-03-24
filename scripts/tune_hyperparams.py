"""
Hyperparameter tuning for tree-based cluster models (Feature 41).

Tunes LGBM, CatBoost, and XGBoost using Optuna Bayesian optimisation with
walk-forward cross-validation that respects demand-forecasting causality.

Usage:
    uv run python scripts/tune_hyperparams.py --model lgbm
    uv run python scripts/tune_hyperparams.py --model catboost --n-trials 20
    uv run python scripts/tune_hyperparams.py --model xgboost --output-dir data/tuning

Outputs:
    data/tuning/best_params_<model>.json   — best params for backtest scripts
    data/tuning/optuna_<model>.db          — Optuna study (SQLite, resumable)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from common.backtest_framework import load_backtest_data
from common.constants import CAT_FEATURES, LAG_RANGE
from common.db import get_db_params
from common.feature_engineering import build_feature_matrix, get_feature_columns, mask_future_sales
from common.mlflow_utils import log_backtest_run
from common.tuning import (
    TRAIN_FOLD_FNS,
    best_rounds_to_n_estimators,
    compute_wape_stabilised,
    generate_cv_month_splits,
    save_best_params,
    suggest_params,
)
from common.utils import _ts


# ── Objective factory ─────────────────────────────────────────────────────────


def make_objective(
    model_name: str,
    full_grid: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    month_splits: list[tuple[list, list]],
    config: dict,
    cat_dtype: str,
) -> Any:
    """Return an Optuna objective function closure for the given model."""
    t_cfg = config["tuning"]
    early_stopping_rounds = t_cfg["early_stopping_rounds"]
    n_estimators_max = t_cfg["n_estimators_max"]
    train_fn = TRAIN_FOLD_FNS[model_name]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name, config)

        fold_wapes: list[float] = []
        fold_best_rounds: list[int] = []

        for fold_idx, (train_months, val_months) in enumerate(month_splits):
            train_end = max(train_months)

            # Causal masking: recompute lag/rolling features with cutoff at train_end
            masked_grid = mask_future_sales(full_grid, train_end)

            train_mask = masked_grid["startdate"].isin(set(train_months))
            val_mask = masked_grid["startdate"].isin(set(val_months))

            # Drop rows with NaN in lag features (first few months of each DFU)
            train_data = masked_grid[train_mask].dropna(
                subset=[f"qty_lag_{lag}" for lag in LAG_RANGE]
            )
            val_data = masked_grid[val_mask].copy()

            # Fill NaN lag/rolling features in val with 0 (same as backtest framework)
            for col in feature_cols:
                if col in val_data.columns and col not in cat_cols:
                    val_data[col] = val_data[col].fillna(0)

            if len(train_data) == 0 or len(val_data) == 0:
                continue

            X_train = train_data[feature_cols]
            y_train = train_data["qty"]
            X_val = val_data[feature_cols]
            y_val = val_data["qty"].values

            try:
                preds, best_rounds = train_fn(
                    X_train, y_train, X_val, y_val,
                    cat_cols, params, n_estimators_max, early_stopping_rounds,
                )
            except Exception as exc:
                print(f"    [{_ts()}] Fold {fold_idx + 1} training failed: {exc}")
                return float("inf")

            wape = compute_wape_stabilised(preds, y_val)
            if not np.isinf(wape) and not np.isnan(wape):
                fold_wapes.append(wape)
                fold_best_rounds.append(best_rounds)

            # Report intermediate value for Optuna pruner
            trial.report(float(np.mean(fold_wapes)) if fold_wapes else float("inf"), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not fold_wapes:
            return float("inf")

        trial.set_user_attr("best_n_estimators", int(np.mean(fold_best_rounds)))
        trial.set_user_attr("fold_wapes", [round(w, 6) for w in fold_wapes])
        return float(np.mean(fold_wapes))

    return objective


# ── Per-cluster WAPE evaluation ───────────────────────────────────────────────


def evaluate_per_cluster_wape(
    model_name: str,
    full_grid: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    month_splits: list[tuple[list, list]],
    best_params: dict,
    best_n_estimators: int,
    cat_dtype: str,
    early_stopping_rounds: int,
    n_estimators_max: int,
) -> dict[str, float]:
    """Compute WAPE per cluster using the best hyperparameters.

    Uses the last CV fold (largest training set) for efficiency.
    Returns dict of {cluster_label: wape_fraction}.
    """
    if not month_splits:
        return {}

    train_months, val_months = month_splits[-1]
    train_end = max(train_months)
    masked_grid = mask_future_sales(full_grid, train_end)

    train_mask = masked_grid["startdate"].isin(set(train_months))
    val_mask = masked_grid["startdate"].isin(set(val_months))

    train_data = masked_grid[train_mask].dropna(
        subset=[f"qty_lag_{lag}" for lag in LAG_RANGE]
    )
    val_data = masked_grid[val_mask].copy()
    for col in feature_cols:
        if col in val_data.columns and col not in cat_cols:
            val_data[col] = val_data[col].fillna(0)

    if len(train_data) == 0 or len(val_data) == 0:
        return {}

    train_fn = TRAIN_FOLD_FNS[model_name]
    try:
        preds, _ = train_fn(
            train_data[feature_cols], train_data["qty"],
            val_data[feature_cols], val_data["qty"].values,
            cat_cols, best_params, n_estimators_max, early_stopping_rounds,
        )
    except Exception:
        return {}

    val_data = val_data.copy()
    val_data["_pred"] = preds

    per_cluster: dict[str, float] = {}
    if "ml_cluster" not in val_data.columns:
        return per_cluster

    for cluster, grp in val_data.groupby("ml_cluster"):
        if pd.isna(cluster) or cluster == "__unknown__":
            continue
        wape = compute_wape_stabilised(
            grp["_pred"].values, grp["qty"].values
        )
        if not np.isinf(wape):
            per_cluster[str(cluster)] = round(wape, 6)

    return per_cluster


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for tree-based backtest models (Feature 41)"
    )
    parser.add_argument(
        "--model",
        choices=["lgbm", "catboost", "xgboost"],
        required=True,
        help="Model to tune",
    )
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override n_trials from config")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory from config")
    parser.add_argument("--config", type=str,
                        default=str(ROOT / "config" / "hyperparameter_tuning.yaml"),
                        help="Path to hyperparameter tuning YAML config")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing Optuna study from SQLite DB")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    t_cfg = config["tuning"]
    n_trials = args.n_trials or t_cfg["n_trials"]
    output_dir = Path(args.output_dir or (ROOT / t_cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model
    # CatBoost uses str dtype for categoricals; LGBM and XGBoost use "category"
    cat_dtype = "str" if model_name == "catboost" else "category"

    print(f"[{_ts()}] Tuning {model_name.upper()} — {n_trials} trials")
    print(f"[{_ts()}] Output dir: {output_dir}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[{_ts()}] Loading data from Postgres...")
    db = get_db_params()
    sales_df, dfu_attrs, item_attrs = load_backtest_data(db)

    # ── Build feature matrix ──────────────────────────────────────────────────
    all_months = sorted(sales_df["startdate"].unique())
    print(f"[{_ts()}] Building feature matrix ({len(all_months)} months)...")
    full_grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, all_months, cat_dtype=cat_dtype)
    feature_cols = get_feature_columns(full_grid)
    cat_cols = [c for c in CAT_FEATURES if c in feature_cols and c in full_grid.columns]
    print(f"[{_ts()}] Features: {len(feature_cols)} total, {len(cat_cols)} categorical")

    # ── Generate CV splits ────────────────────────────────────────────────────
    month_splits = generate_cv_month_splits(
        all_months,
        n_splits=t_cfg["n_splits"],
        gap_months=t_cfg["gap_months"],
        min_train_months=t_cfg["min_train_months"],
        val_months_per_fold=t_cfg["val_months_per_fold"],
    )
    print(f"[{_ts()}] CV splits: {len(month_splits)} folds")
    for i, (tm, vm) in enumerate(month_splits):
        print(f"  Fold {i + 1}: train [{min(tm).date()} → {max(tm).date()}] "
              f"({len(tm)} months), gap {t_cfg['gap_months']}m, "
              f"val [{min(vm).date()} → {max(vm).date()}] ({len(vm)} months)")

    if not month_splits:
        print(f"[{_ts()}] No valid CV splits — need more data or reduce n_splits")
        sys.exit(1)

    # ── Create Optuna study ───────────────────────────────────────────────────
    study_path = output_dir / f"optuna_{model_name}.db"
    storage = f"sqlite:///{study_path}"
    study_name = f"{model_name}_tuning"

    sampler = optuna.samplers.TPESampler(seed=t_cfg["random_seed"])
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=t_cfg["pruner_n_startup_trials"],
        n_warmup_steps=t_cfg["pruner_n_warmup_steps"],
    )

    if args.resume:
        print(f"[{_ts()}] Resuming study from {study_path}")
        study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler)
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

    completed_before = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed_before)
    print(f"[{_ts()}] Study: {completed_before} existing complete trials, running {remaining} more")

    if remaining == 0:
        print(f"[{_ts()}] Study already has {completed_before} completed trials — use --resume or increase --n-trials")

    # ── Build objective ───────────────────────────────────────────────────────
    objective_fn = make_objective(
        model_name=model_name,
        full_grid=full_grid,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        month_splits=month_splits,
        config=config,
        cat_dtype=cat_dtype,
    )

    # ── Optimise ──────────────────────────────────────────────────────────────
    t_opt_start = time.time()

    def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            best_wape_pct = study.best_value * 100 if study.best_value < float("inf") else float("inf")
            print(f"  [{_ts()}] Trial {trial.number:3d} | WAPE={trial.value * 100:.2f}% | "
                  f"best={best_wape_pct:.2f}% | {completed}/{n_trials}")

    if remaining > 0:
        study.optimize(
            objective_fn,
            n_trials=remaining,
            callbacks=[_trial_callback],
            show_progress_bar=False,
        )

    elapsed = time.time() - t_opt_start
    print(f"\n[{_ts()}] Optimisation done in {elapsed:.0f}s ({elapsed / 60:.1f}m)")

    # ── Collect best trial ────────────────────────────────────────────────────
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print(f"[{_ts()}] No completed trials — check data availability and config")
        sys.exit(1)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_n_est_raw = best_trial.user_attrs.get("best_n_estimators", 500)
    best_n_estimators = best_rounds_to_n_estimators(
        [best_n_est_raw], buffer=t_cfg.get("n_estimators_buffer", 1.1)
    )
    fold_wapes = best_trial.user_attrs.get("fold_wapes", [])

    print(f"\n[{_ts()}] Best trial #{best_trial.number}:")
    print(f"  WAPE: {best_trial.value * 100:.4f}%")
    print(f"  n_estimators: {best_n_estimators}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # ── Per-cluster WAPE ──────────────────────────────────────────────────────
    print(f"\n[{_ts()}] Computing per-cluster WAPE for best params...")
    per_cluster_wape = evaluate_per_cluster_wape(
        model_name=model_name,
        full_grid=full_grid,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        month_splits=month_splits,
        best_params=best_params,
        best_n_estimators=best_n_estimators,
        cat_dtype=cat_dtype,
        early_stopping_rounds=t_cfg["early_stopping_rounds"],
        n_estimators_max=t_cfg["n_estimators_max"],
    )
    if per_cluster_wape:
        print("  Per-cluster WAPE:")
        for cluster, wape in sorted(per_cluster_wape.items(), key=lambda x: x[1]):
            print(f"    {cluster}: {wape * 100:.2f}%")

    # ── Save output ───────────────────────────────────────────────────────────
    output_path = output_dir / f"best_params_{model_name}.json"
    config_snapshot = {
        "n_splits": t_cfg["n_splits"],
        "gap_months": t_cfg["gap_months"],
        "val_months_per_fold": t_cfg["val_months_per_fold"],
        "min_train_months": t_cfg["min_train_months"],
        "early_stopping_rounds": t_cfg["early_stopping_rounds"],
        "n_estimators_max": t_cfg["n_estimators_max"],
    }
    save_best_params(
        output_path=output_path,
        model_name=model_name,
        best_wape=best_trial.value,
        best_n_estimators=best_n_estimators,
        best_params=best_params,
        per_cluster_wape=per_cluster_wape,
        n_trials_completed=len(completed_trials),
        cv_fold_wapes=fold_wapes,
        config_snapshot=config_snapshot,
    )

    # ── MLflow logging ────────────────────────────────────────────────────────
    mlflow_metrics: dict[str, Any] = {
        "best_wape_pct": round(best_trial.value * 100, 4),
        "best_n_estimators": best_n_estimators,
        "n_completed_trials": len(completed_trials),
        "n_cv_folds": len(month_splits),
    }
    mlflow_metrics.update({f"cluster_wape_pct_{k}": round(v * 100, 4) for k, v in per_cluster_wape.items()})

    log_backtest_run(
        model_type=f"{model_name}_tuning",
        model_id=f"{model_name}_tuned",
        cluster_strategy="global",
        hyperparams={
            "model": model_name,
            "n_trials": n_trials,
            **config_snapshot,
            **best_params,
        },
        metrics=mlflow_metrics,
        metadata={
            "best_params": best_params,
            "best_n_estimators": best_n_estimators,
        },
        artifact_paths=[str(output_path)],
    )

    print(f"\n[{_ts()}] Tuning complete. Best params saved to {output_path}")
    print(f"  Use: make backtest-{model_name}-cluster ARGS=\"--params-file {output_path}\"")


if __name__ == "__main__":
    main()
