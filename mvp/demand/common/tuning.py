"""Shared hyperparameter tuning utilities (Feature 41).

Provides:
- Month-based expanding walk-forward CV splits with causality gap
- Stabilised WAPE objective (denominator floor prevents division by near-zero)
- Optuna param suggestion from YAML search space config
- Best-params JSON load/save helpers
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ── CV splits ─────────────────────────────────────────────────────────────────


def generate_cv_month_splits(
    all_months: list[pd.Timestamp],
    n_splits: int = 5,
    gap_months: int = 1,
    min_train_months: int = 13,
    val_months_per_fold: int = 3,
) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    """Generate expanding walk-forward CV splits on unique months.

    Splits are computed on the calendar (month-level), not on DataFrame rows,
    so every DFU participates in every fold (same as backtest timeframes).

    Gap of `gap_months` between train_end and val_start prevents lag-feature
    leakage at the fold boundary (month-lag features use data up to train_end;
    the gap ensures the lag window is clean for the validation months).

    Returns:
        List of (train_months, val_months) tuples in chronological order.
        Folds with fewer than min_train_months training months are skipped.
    """
    months = sorted(all_months)
    n = len(months)

    # The last fold's train window ends at n - gap_months - val_months_per_fold - 1
    last_train_end_idx = n - gap_months - val_months_per_fold - 1
    first_train_end_idx = min_train_months - 1  # 0-based

    if last_train_end_idx < first_train_end_idx:
        return []

    # Spread folds evenly from first to last valid train-end index
    total_span = last_train_end_idx - first_train_end_idx
    step = total_span / max(1, n_splits - 1) if n_splits > 1 else 0

    splits: list[tuple[list, list]] = []
    seen_train_ends: set[int] = set()

    for i in range(n_splits):
        train_end_idx = int(round(first_train_end_idx + i * step))
        train_end_idx = min(train_end_idx, last_train_end_idx)

        if train_end_idx in seen_train_ends:
            continue
        seen_train_ends.add(train_end_idx)

        val_start_idx = train_end_idx + 1 + gap_months
        val_end_idx = min(val_start_idx + val_months_per_fold - 1, n - 1)

        if val_start_idx >= n:
            break

        train_months = months[: train_end_idx + 1]
        val_months = months[val_start_idx : val_end_idx + 1]

        if len(train_months) >= min_train_months and len(val_months) > 0:
            splits.append((list(train_months), list(val_months)))

    return splits


# ── WAPE ──────────────────────────────────────────────────────────────────────


def compute_wape_stabilised(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    denominator_floor: float = 1.0,
) -> float:
    """Compute WAPE = Σ|F-A| / max(|ΣA|, denominator_floor).

    The denominator floor prevents division by near-zero actuals in short
    validation windows.  Returns float('inf') if all actuals are NaN/missing.

    Note: returns raw fraction (not multiplied by 100) so Optuna minimises
    a value in [0, ∞) rather than [0, 10000].
    """
    # Drop NaN pairs
    mask = ~(np.isnan(y_pred) | np.isnan(y_true))
    if mask.sum() == 0:
        return float("inf")

    y_pred_clean = y_pred[mask]
    y_true_clean = y_true[mask]

    actual_sum = abs(float(np.sum(y_true_clean)))
    denom = max(actual_sum, denominator_floor)
    return float(np.sum(np.abs(y_pred_clean - y_true_clean))) / denom


# ── Optuna param suggestion ────────────────────────────────────────────────────


def suggest_params(
    trial: Any,  # optuna.Trial
    model_name: str,
    config: dict,
) -> dict[str, Any]:
    """Suggest hyperparameters for a trial from the YAML search space config.

    Handles three param types:
    - float  (log or linear scale)
    - int    (uniform integer)

    Returns a dict of suggested values (excludes fixed_params — those are
    merged by the caller).
    """
    space = config[model_name]["search_space"]
    params: dict[str, Any] = {}

    for name, spec in space.items():
        ptype = spec["type"]
        low = spec["low"]
        high = spec["high"]
        log = spec.get("log", False)

        if ptype == "float":
            params[name] = trial.suggest_float(name, low, high, log=log)
        elif ptype == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        else:
            raise ValueError(f"Unknown param type '{ptype}' for '{name}'")

    return params


# ── Best-params JSON I/O ──────────────────────────────────────────────────────


def save_best_params(
    output_path: Path,
    model_name: str,
    best_wape: float,
    best_n_estimators: int,
    best_params: dict[str, Any],
    per_cluster_wape: dict[str, float],
    n_trials_completed: int,
    cv_fold_wapes: list[float],
    config_snapshot: dict[str, Any],
) -> None:
    """Write tuning results to JSON for consumption by backtest scripts."""
    result = {
        "model": model_name,
        "best_wape": round(best_wape * 100, 4),  # store as % for readability
        "best_n_estimators": best_n_estimators,
        "best_params": best_params,
        "per_cluster_wape": {k: round(v * 100, 4) for k, v in per_cluster_wape.items()},
        "n_trials_completed": n_trials_completed,
        "cv_fold_wapes": [round(w * 100, 4) for w in cv_fold_wapes],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config_snapshot,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [{_ts()}] Saved tuning results to {output_path}")


def load_best_params(params_file: Path) -> dict[str, Any]:
    """Load best hyperparameters from a tuning result JSON file.

    Returns dict with keys: best_params, best_n_estimators.
    Raises FileNotFoundError if path does not exist.
    """
    if not params_file.exists():
        raise FileNotFoundError(f"Params file not found: {params_file}")
    with open(params_file) as f:
        data = json.load(f)
    return data


# ── Best-rounds → n_estimators ────────────────────────────────────────────────


def best_rounds_to_n_estimators(
    fold_best_rounds: list[int],
    buffer: float = 1.1,
) -> int:
    """Convert per-fold early-stopping rounds to a final n_estimators value.

    Applies a buffer (default 10%) to account for the final model training on
    more data than any single CV fold, which generally benefits from more trees.
    Returns at least 50 estimators.
    """
    if not fold_best_rounds:
        return 500
    mean_rounds = float(np.mean(fold_best_rounds))
    return max(50, math.ceil(mean_rounds * buffer))


# ── Per-fold model training functions ─────────────────────────────────────────
# These are used by both tune_hyperparams.py (global tuning) and
# tune_for_timeframe() (causal per-timeframe tuning). Lazy model imports keep
# unused framework libraries out of the import chain at module load time.


def train_lgbm_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
    params: dict,
    n_estimators_max: int,
    early_stopping_rounds: int,
) -> tuple[np.ndarray, int]:
    """Train LightGBM with early stopping on one CV fold. Returns (preds, best_rounds)."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators_max,
        **params,
        verbosity=-1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    preds = np.maximum(model.predict(X_val), 0)
    best_rounds = int(model.best_iteration_) if model.best_iteration_ else n_estimators_max
    return preds, best_rounds


def train_catboost_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
    params: dict,
    n_estimators_max: int,
    early_stopping_rounds: int,
) -> tuple[np.ndarray, int]:
    """Train CatBoost with early stopping on one CV fold. Returns (preds, best_rounds)."""
    import catboost as cb

    feature_cols = list(X_train.columns)
    cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

    model = cb.CatBoostRegressor(
        iterations=n_estimators_max,
        random_seed=42,
        verbose=0,
        loss_function="RMSE",
        **params,
    )
    model.fit(
        X_train, y_train,
        cat_features=cat_indices,
        eval_set=(X_val, y_val),
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )
    preds = np.maximum(model.predict(X_val), 0)
    best_rounds = int(model.best_iteration_) if model.best_iteration_ else n_estimators_max
    return preds, best_rounds


def train_xgboost_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: list[str],
    params: dict,
    n_estimators_max: int,
    early_stopping_rounds: int,
) -> tuple[np.ndarray, int]:
    """Train XGBoost with early stopping on one CV fold. Returns (preds, best_rounds)."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=n_estimators_max,
        verbosity=0,
        random_state=42,
        n_jobs=-1,
        enable_categorical=True,
        tree_method="hist",
        **params,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )
    preds = np.maximum(model.predict(X_val), 0)
    best_rounds = int(model.best_iteration) if model.best_iteration else n_estimators_max
    return preds, best_rounds


# Registry used by tune_hyperparams.py and tune_for_timeframe()
TRAIN_FOLD_FNS: dict[str, Callable] = {
    "lgbm": train_lgbm_fold,
    "catboost": train_catboost_fold,
    "xgboost": train_xgboost_fold,
}


# ── Per-timeframe causal tuning (PL-002 fix) ──────────────────────────────────


def tune_for_timeframe(
    model_name: str,
    train_fold_fn: Callable,
    full_grid: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    cutoff_date: pd.Timestamp,
    config: dict,
    n_trials: int | None = None,
) -> tuple[dict[str, Any], int]:
    """Run Optuna hyperparameter search using only data up to cutoff_date.

    This is the causal alternative to loading a globally-tuned params file.
    Each backtest timeframe calls this with its own train_end as cutoff_date,
    so the tuner never sees future months. This eliminates the temporal data
    leakage described in PL-002.

    Parameters
    ----------
    model_name:     "lgbm" | "catboost" | "xgboost"
    train_fold_fn:  Fold training function from TRAIN_FOLD_FNS
    full_grid:      Full feature matrix (all months). Internally filtered to <= cutoff_date.
    feature_cols:   Feature column names
    cat_cols:       Categorical feature column names
    cutoff_date:    Upper bound on training data (= timeframe's train_end)
    config:         Loaded hyperparameter_tuning.yaml dict
    n_trials:       Optuna trials to run (default: config["tuning"]["inline_n_trials"])

    Returns
    -------
    (best_params_dict, best_n_estimators)
    Returns ({}, 500) when insufficient data for even one CV split.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print(f"  [tune_for_timeframe] optuna not installed — returning empty params")
        return {}, 500

    from common.feature_engineering import mask_future_sales
    from common.constants import LAG_RANGE

    t_cfg = config["tuning"]
    _n_trials = n_trials or t_cfg.get("inline_n_trials", 20)
    n_splits = t_cfg.get("inline_n_splits", 3)
    early_stopping_rounds = t_cfg["early_stopping_rounds"]
    n_estimators_max = t_cfg["n_estimators_max"]

    # Only consider months that were available at forecast time
    causal_months = sorted(m for m in full_grid["startdate"].unique() if m <= cutoff_date)

    month_splits = generate_cv_month_splits(
        causal_months,
        n_splits=n_splits,
        gap_months=t_cfg["gap_months"],
        min_train_months=t_cfg["min_train_months"],
        val_months_per_fold=t_cfg["val_months_per_fold"],
    )

    if not month_splits:
        return {}, 500

    def _objective(trial: "optuna.Trial") -> float:
        params = suggest_params(trial, model_name, config)
        fold_wapes: list[float] = []
        fold_best_rounds: list[int] = []

        for fold_idx, (train_months, val_months) in enumerate(month_splits):
            train_end_fold = max(train_months)
            masked = mask_future_sales(full_grid, train_end_fold)

            train_data = masked[masked["startdate"].isin(set(train_months))].dropna(
                subset=[f"qty_lag_{lag}" for lag in LAG_RANGE]
            )
            val_data = masked[masked["startdate"].isin(set(val_months))].copy()
            for col in feature_cols:
                if col in val_data.columns and col not in cat_cols:
                    val_data[col] = val_data[col].fillna(0)

            if len(train_data) == 0 or len(val_data) == 0:
                continue

            try:
                preds, best_rounds = train_fold_fn(
                    train_data[feature_cols], train_data["qty"],
                    val_data[feature_cols], val_data["qty"].values,
                    cat_cols, params, n_estimators_max, early_stopping_rounds,
                )
            except Exception:
                return float("inf")

            wape = compute_wape_stabilised(preds, val_data["qty"].values)
            if not np.isinf(wape) and not np.isnan(wape):
                fold_wapes.append(wape)
                fold_best_rounds.append(best_rounds)

            trial.report(float(np.mean(fold_wapes)) if fold_wapes else float("inf"), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not fold_wapes:
            return float("inf")

        trial.set_user_attr("best_n_estimators", int(np.mean(fold_best_rounds)))
        return float(np.mean(fold_wapes))

    sampler = optuna.samplers.TPESampler(seed=t_cfg.get("random_seed", 42))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(5, t_cfg.get("pruner_n_startup_trials", 15)),
        n_warmup_steps=t_cfg.get("pruner_n_warmup_steps", 3),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(_objective, n_trials=_n_trials, show_progress_bar=False)

    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        return {}, 500

    best = study.best_trial
    best_n_est_raw = best.user_attrs.get("best_n_estimators", n_estimators_max)
    best_n_estimators = best_rounds_to_n_estimators(
        [best_n_est_raw], buffer=t_cfg.get("n_estimators_buffer", 1.1)
    )
    return best.params, best_n_estimators
