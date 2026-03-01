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
from typing import Any

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
