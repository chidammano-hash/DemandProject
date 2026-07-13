"""
Hyperparameter tuning for tree-based cluster models (Feature 41).

Tunes LightGBM using Optuna Bayesian optimisation with
walk-forward cross-validation that respects demand-forecasting causality.

Usage:
    uv run python scripts/tune_hyperparams.py --model lgbm
    uv run python scripts/tune_hyperparams.py --model lgbm --n-trials 20

Outputs:
    data/tuning/best_params_<model>.json   — best params for backtest scripts
    data/tuning/optuna_<model>.db          — Optuna study (SQLite, resumable)
"""

# ruff: noqa: E402

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from common.core.constants import CAT_FEATURES, LAG_RANGE
from common.core.db import get_db_params
from common.core.utils import _ts, get_algorithm_roster, load_forecast_pipeline_config
from common.ml.backtest_framework import load_backtest_data
from common.ml.feature_engineering import (
    build_feature_matrix,
    get_feature_columns,
    mask_future_sales,
)
from common.ml.mlflow_utils import log_backtest_run
from common.ml.tuning import (
    TRAIN_FOLD_FNS,
    best_rounds_to_n_estimators,
    compute_wape_stabilised,
    generate_cv_month_splits,
    merge_fixed_params,
    save_best_params,
    suggest_model_params,
    trial_best_rounds_or_max,
)

TREE_MODEL_PREFIXES = ("lgbm",)
DFU_KEY_COLS = ["item_id", "customer_group", "loc"]


def sample_tuning_dfus(
    sales_df: pd.DataFrame,
    *,
    max_dfus: int,
    random_seed: int,
) -> pd.DataFrame:
    """Deterministically sample DFUs while retaining every row in their histories."""
    if max_dfus < 1:
        raise ValueError("max_dfus must be positive")
    missing = [column for column in DFU_KEY_COLS if column not in sales_df.columns]
    if missing:
        raise ValueError(f"sales data is missing DFU keys: {missing}")
    dfus = sales_df[DFU_KEY_COLS].drop_duplicates()
    if len(dfus) <= max_dfus:
        return sales_df
    selected = dfus.sample(n=max_dfus, random_state=random_seed)
    row_keys = pd.MultiIndex.from_frame(sales_df[DFU_KEY_COLS])
    selected_keys = pd.MultiIndex.from_frame(selected)
    return sales_df.loc[row_keys.isin(selected_keys)].copy()


def _base_model_name(model_id: str) -> str:
    """Return the tree backend name for a pipeline model id."""
    for prefix in TREE_MODEL_PREFIXES:
        if model_id == prefix or model_id.startswith(f"{prefix}_"):
            return prefix
    raise ValueError(
        f"Cannot resolve tree backend for model_id={model_id!r}. "
        f"Expected one of: {', '.join(TREE_MODEL_PREFIXES)}."
    )


def _default_model_id(model_name: str, pipeline_cfg: dict) -> str:
    """Return the canonical base model id for a backend."""
    candidate = f"{model_name}_cluster"
    algorithms = pipeline_cfg.get("algorithms", {}) or {}
    if candidate in algorithms:
        return candidate
    return model_name


def _resolve_tuning_target(
    model_name: str,
    model_id: str | None,
    pipeline_cfg: dict | None,
) -> tuple[str, str, dict]:
    """Resolve backend/model-id config for a tuning run."""
    if pipeline_cfg is None:
        return model_name, model_id or model_name, {}

    resolved_model_id = model_id or _default_model_id(model_name, pipeline_cfg)
    algorithms = pipeline_cfg.get("algorithms", {}) or {}
    entry = algorithms.get(resolved_model_id)
    if entry is None:
        raise ValueError(
            f"Model id {resolved_model_id!r} not found in forecast_pipeline_config.yaml"
        )
    if entry.get("type") != "tree":
        raise ValueError(f"Model id {resolved_model_id!r} is not a tree algorithm")
    resolved_model_name = _base_model_name(resolved_model_id)
    if resolved_model_name != model_name:
        raise ValueError(f"--model {model_name!r} does not match --model-id {resolved_model_id!r}")
    return resolved_model_name, resolved_model_id, entry


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
        params = suggest_model_params(trial, model_name, config)

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
            y_val = full_grid.loc[val_data.index, "qty"].values

            try:
                preds, best_rounds = train_fn(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    cat_cols,
                    params,
                    n_estimators_max,
                    early_stopping_rounds,
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

    train_data = masked_grid[train_mask].dropna(subset=[f"qty_lag_{lag}" for lag in LAG_RANGE])
    val_data = masked_grid[val_mask].copy()
    for col in feature_cols:
        if col in val_data.columns and col not in cat_cols:
            val_data[col] = val_data[col].fillna(0)

    if len(train_data) == 0 or len(val_data) == 0:
        return {}

    train_fn = TRAIN_FOLD_FNS[model_name]
    try:
        preds, _ = train_fn(
            train_data[feature_cols],
            train_data["qty"],
            val_data[feature_cols],
            full_grid.loc[val_data.index, "qty"].values,
            cat_cols,
            best_params,
            n_estimators_max,
            early_stopping_rounds,
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
        wape = compute_wape_stabilised(grp["_pred"].values, grp["qty"].values)
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
        choices=["lgbm"],
        default="lgbm",
        help="Model to tune (the canonical tree model)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Canonical pipeline model id to tune (lgbm_cluster)",
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Override n_trials from config")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override output directory from config"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "forecasting" / "hyperparameter_tuning.yaml"),
        help="Path to hyperparameter tuning YAML config",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume an existing Optuna study from SQLite DB"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the bounded exploratory profile designed to finish within five minutes",
    )
    args = parser.parse_args()
    run_started_at = time.monotonic()

    load_dotenv(ROOT / ".env")

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Overlay pipeline-level tuning params from forecast_pipeline_config.yaml
    # (n_trials, n_splits, gap_months, etc.). Search spaces remain in
    # hyperparameter_tuning.yaml.
    pipeline_cfg: dict | None = None
    try:
        pipeline_cfg = load_forecast_pipeline_config()
        pipeline_tuning = pipeline_cfg.get("tuning", {})
        # Merge: pipeline values override old config for shared keys
        for key in (
            "n_trials",
            "n_splits",
            "gap_months",
            "val_months_per_fold",
            "min_train_months",
            "early_stopping_rounds",
            "n_estimators_max",
            "n_estimators_buffer",
            "random_seed",
            "output_dir",
        ):
            if key in pipeline_tuning:
                config["tuning"][key] = pipeline_tuning[key]
    except FileNotFoundError:
        pass  # No pipeline config — use hyperparameter_tuning.yaml as-is

    if args.fast:
        fast_profile = config["fast_profile"]
        for key in (
            "n_trials",
            "n_splits",
            "val_months_per_fold",
            "early_stopping_rounds",
            "n_estimators_max",
            "pruner_n_startup_trials",
            "pruner_n_warmup_steps",
        ):
            config["tuning"][key] = fast_profile[key]

    try:
        model_name, resolved_model_id, algo_entry = _resolve_tuning_target(
            args.model,
            args.model_id,
            pipeline_cfg,
        )
    except ValueError as exc:
        print(f"[{_ts()}] {exc}")
        sys.exit(2)

    # Check algorithm tune flag from the roster
    try:
        roster = get_algorithm_roster(stage="tune")
        if args.model_id:
            if resolved_model_id not in roster:
                print(
                    f"[{_ts()}] Model '{resolved_model_id}' has tune=false in "
                    "forecast_pipeline_config.yaml — skipping"
                )
                sys.exit(0)
        else:
            model_in_roster = any(
                rid == args.model or rid.startswith(f"{args.model}_") for rid in roster
            )
            if not model_in_roster:
                # Check if the model exists but has tune=false
                all_roster = get_algorithm_roster()
                exists = any(
                    rid == args.model or rid.startswith(f"{args.model}_") for rid in all_roster
                )
                if exists:
                    print(
                        f"[{_ts()}] Model '{args.model}' has tune=false in "
                        f"forecast_pipeline_config.yaml — skipping"
                    )
                    sys.exit(0)
    except FileNotFoundError:
        pass  # No pipeline config — run unconditionally

    t_cfg = config["tuning"]
    n_trials = args.n_trials or t_cfg["n_trials"]
    output_dir = Path(args.output_dir or (ROOT / t_cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    cat_dtype = "category"
    output_stem = resolved_model_id if args.model_id else model_name
    include_customer_features = bool(
        (algo_entry.get("params", {}) if algo_entry else {}).get("customer_features", False)
    )

    print(f"[{_ts()}] Tuning {resolved_model_id} ({model_name.upper()}) — {n_trials} trials")
    print(f"[{_ts()}] Output dir: {output_dir}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[{_ts()}] Loading data from Postgres...")
    db = get_db_params()
    data_result = load_backtest_data(
        db,
        include_customer_features=include_customer_features,
    )
    if include_customer_features:
        sales_df, dfu_attrs, item_attrs, customer_features = data_result
    else:
        sales_df, dfu_attrs, item_attrs = data_result
        customer_features = None

    if args.fast:
        original_dfus = sales_df[DFU_KEY_COLS].drop_duplicates().shape[0]
        sales_df = sample_tuning_dfus(
            sales_df,
            max_dfus=int(config["fast_profile"]["max_dfus"]),
            random_seed=int(t_cfg["random_seed"]),
        )
        sampled_dfus = sales_df[DFU_KEY_COLS].drop_duplicates().shape[0]
        print(f"[{_ts()}] Fast profile DFU sample: {sampled_dfus}/{original_dfus}")

    # ── Build feature matrix ──────────────────────────────────────────────────
    all_months = sorted(sales_df["startdate"].unique())
    print(f"[{_ts()}] Building feature matrix ({len(all_months)} months)...")
    full_grid = build_feature_matrix(
        sales_df,
        dfu_attrs,
        item_attrs,
        all_months,
        cat_dtype=cat_dtype,
        customer_features=customer_features,
    )
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
        print(
            f"  Fold {i + 1}: train [{min(tm).date()} → {max(tm).date()}] "
            f"({len(tm)} months), gap {t_cfg['gap_months']}m, "
            f"val [{min(vm).date()} → {max(vm).date()}] ({len(vm)} months)"
        )

    if not month_splits:
        print(f"[{_ts()}] No valid CV splits — need more data or reduce n_splits")
        sys.exit(1)

    # ── Create Optuna study ───────────────────────────────────────────────────
    study_suffix = "_fast" if args.fast else ""
    study_path = output_dir / f"optuna_{output_stem}{study_suffix}.db"
    storage = f"sqlite:///{study_path}"
    study_name = f"{output_stem}_tuning{study_suffix}"

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

    completed_before = len(
        [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.value is not None
            and np.isfinite(trial.value)
        ]
    )
    remaining = max(0, n_trials - completed_before)
    print(f"[{_ts()}] Study: {completed_before} existing complete trials, running {remaining} more")

    if remaining == 0:
        print(
            f"[{_ts()}] Study already has {completed_before} completed trials — use --resume or increase --n-trials"
        )

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
            completed = len(
                [
                    row
                    for row in study.trials
                    if row.state == optuna.trial.TrialState.COMPLETE
                    and row.value is not None
                    and np.isfinite(row.value)
                ]
            )
            best_wape_pct = (
                study.best_value * 100 if study.best_value < float("inf") else float("inf")
            )
            print(
                f"  [{_ts()}] Trial {trial.number:3d} | WAPE={trial.value * 100:.2f}% | "
                f"best={best_wape_pct:.2f}% | {completed}/{n_trials}"
            )

    if remaining > 0:
        timeout_seconds = t_cfg.get("timeout_seconds")
        if args.fast:
            elapsed_total = time.monotonic() - run_started_at
            timeout_seconds = max(
                1.0,
                float(config["fast_profile"]["time_budget_seconds"]) - elapsed_total,
            )
            print(f"[{_ts()}] Fast profile optimization budget: {timeout_seconds:.0f}s")
        study.optimize(
            objective_fn,
            n_trials=remaining,
            timeout=timeout_seconds,
            callbacks=[_trial_callback],
            show_progress_bar=False,
        )

    elapsed = time.time() - t_opt_start
    print(f"\n[{_ts()}] Optimisation done in {elapsed:.0f}s ({elapsed / 60:.1f}m)")

    # ── Collect best trial ────────────────────────────────────────────────────
    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and np.isfinite(trial.value)
    ]
    if not completed_trials:
        print(f"[{_ts()}] No completed trials — check data availability and config")
        sys.exit(1)

    best_trial = study.best_trial
    best_params = merge_fixed_params(model_name, config, best_trial.params)
    best_n_est_raw = trial_best_rounds_or_max(best_trial, t_cfg["n_estimators_max"])
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
    output_path = output_dir / f"best_params_{output_stem}.json"
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
        model_name=resolved_model_id,
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
    mlflow_metrics.update(
        {f"cluster_wape_pct_{k}": round(v * 100, 4) for k, v in per_cluster_wape.items()}
    )

    log_backtest_run(
        model_type=f"{model_name}_tuning",
        model_id=f"{output_stem}_tuned",
        cluster_strategy="global",
        hyperparams={
            "model": model_name,
            "model_id": resolved_model_id,
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
    print(f'  Use: make backtest-{model_name} ARGS="--params-file {output_path}"')


if __name__ == "__main__":
    main()
