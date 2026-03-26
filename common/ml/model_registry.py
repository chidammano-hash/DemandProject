"""Centralized model registry and parameter abstraction for tree-based backtests.

Provides:
- Canonical ↔ native parameter name mapping per model library
- Unified ``fit_model()`` that replaces duplicate if/elif/else fit blocks
- ``get_best_iteration()`` abstracting attribute name differences
- ``compute_early_stop_patience()`` for standardized 3% patience
- WAPE eval callbacks for early stopping alignment (LGBM, CatBoost, XGBoost)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical ↔ native parameter name mapping
# ---------------------------------------------------------------------------

CANONICAL_TO_NATIVE: dict[str, dict[str, str | None]] = {
    "lgbm": {
        "estimators": "n_estimators",
        "max_depth": "max_depth",
        "l2_reg": "reg_lambda",
        "l1_reg": "reg_alpha",
        "min_leaf_samples": "min_child_samples",
        "col_sample": "colsample_bytree",
    },
    "catboost": {
        "estimators": "iterations",
        "max_depth": "depth",
        "l2_reg": "l2_leaf_reg",
        "l1_reg": None,
        "min_leaf_samples": "min_data_in_leaf",
        "col_sample": "colsample_bylevel",
    },
    "xgboost": {
        "estimators": "n_estimators",
        "max_depth": "max_depth",
        "l2_reg": "reg_lambda",
        "l1_reg": "reg_alpha",
        "min_leaf_samples": "min_child_weight",
        "col_sample": "colsample_bytree",
    },
}

# Reverse mapping: native → canonical (built dynamically)
NATIVE_TO_CANONICAL: dict[str, dict[str, str]] = {}
for _model, _mapping in CANONICAL_TO_NATIVE.items():
    NATIVE_TO_CANONICAL[_model] = {
        v: k for k, v in _mapping.items() if v is not None
    }


def to_native_params(model_name: str, canonical_params: dict) -> dict:
    """Map canonical parameter names to native library names.

    Keys not found in the canonical map are passed through unchanged
    (supports model-specific params like ``num_leaves``, ``path_smooth``).
    Keys whose native mapping is ``None`` (e.g. CatBoost ``l1_reg``) are skipped.
    """
    mapping = CANONICAL_TO_NATIVE.get(model_name, {})
    result = {}
    for k, v in canonical_params.items():
        if k in mapping:
            native_key = mapping[k]
            if native_key is not None:
                result[native_key] = v
        else:
            result[k] = v
    return result


def from_native_params(model_name: str, native_params: dict) -> dict:
    """Map native library parameter names back to canonical names.

    Keys not found in the reverse map are passed through unchanged.
    """
    mapping = NATIVE_TO_CANONICAL.get(model_name, {})
    result = {}
    for k, v in native_params.items():
        result[mapping.get(k, k)] = v
    return result


# ---------------------------------------------------------------------------
# Best iteration abstraction
# ---------------------------------------------------------------------------

def get_best_iteration(model: Any, model_name: str) -> int | None:
    """Get the best iteration from a trained model, abstracting attribute differences.

    LGBM/CatBoost use ``best_iteration_`` (trailing underscore, sklearn convention).
    XGBoost uses ``best_iteration`` (no trailing underscore).

    Returns None if the attribute is missing or falsy.
    """
    if model_name == "xgboost":
        val = getattr(model, "best_iteration", None)
    else:
        val = getattr(model, "best_iteration_", None)
    return int(val) if val else None


# ---------------------------------------------------------------------------
# Early stopping patience
# ---------------------------------------------------------------------------

EARLY_STOP_PCT = 0.03  # 3% of max iterations
EARLY_STOP_FLOOR = 10  # minimum patience rounds


def compute_early_stop_patience(
    max_iterations: int,
    pct: float = EARLY_STOP_PCT,
) -> int:
    """Compute standardized early stopping patience as a percentage of max iterations.

    Returns max(floor, int(max_iterations * pct)) to ensure a minimum of 10 rounds.
    """
    return max(EARLY_STOP_FLOOR, int(max_iterations * pct))


# ---------------------------------------------------------------------------
# WAPE eval callbacks for early stopping alignment
# ---------------------------------------------------------------------------
# Models train on L2/RMSE but final evaluation uses WAPE.  These custom eval
# functions ensure early stopping also optimises for WAPE so there is no
# metric mismatch.
#
# WAPE = sum(|F - A|) / |sum(A)|
# ---------------------------------------------------------------------------


def _wape_lgbm(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
    """LGBM custom eval function that computes WAPE.

    Signature follows the lightgbm **sklearn** custom-eval contract:
    ``(y_true, y_pred) -> (name, value, is_higher_better)``
    """
    denom = max(abs(y_true.sum()), 1.0)
    wape = float(np.sum(np.abs(y_pred - y_true)) / denom)
    return "wape", wape, False


class WapeMetric:
    """CatBoost custom metric that computes WAPE for early stopping alignment."""

    def get_final_error(self, error: float, weight: float) -> float:  # noqa: ARG002
        return error

    def is_max_optimal(self) -> bool:
        return False

    def evaluate(
        self,
        approxes: list[list[float]],
        target: list[float],
        weight: list[float] | None,  # noqa: ARG002
    ) -> tuple[float, float]:
        y_pred = np.asarray(approxes[0])
        y_true = np.asarray(target)
        denom = max(abs(y_true.sum()), 1.0)
        wape = float(np.sum(np.abs(y_pred - y_true)) / denom)
        return wape, 1.0


def _wape_xgb(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """XGBoost custom eval function that computes WAPE.

    Signature follows the xgboost 3.x **sklearn** eval_metric contract:
    ``(y_true, y_pred) -> float``  (function ``__name__`` is used as metric name).
    """
    denom = max(abs(y_true.sum()), 1.0)
    return float(np.sum(np.abs(y_pred - y_true)) / denom)


# ---------------------------------------------------------------------------
# Unified fit function
# ---------------------------------------------------------------------------

def fit_model(
    model: Any,
    model_name: str,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    cat_cols: list[str],
    feature_cols: list[str],
    lib_module: Any,
    max_iterations: int,
    early_stop_pct: float = EARLY_STOP_PCT,
) -> None:
    """Fit a tree model with early stopping — single source of truth for all models.

    Replaces the duplicate if/elif/else fit blocks that were in
    ``_train_single_cluster`` and ``train_and_predict_global``.

    Args:
        model: Instantiated model (LGBMRegressor, CatBoostRegressor, XGBRegressor).
        model_name: One of "lgbm", "catboost", "xgboost".
        X_tr: Training features.
        y_tr: Training target.
        X_val: Validation features.
        y_val: Validation target.
        cat_cols: Categorical column names.
        feature_cols: Full feature column list (for computing cat_indices).
        lib_module: The model's parent library module (lightgbm, catboost, xgboost).
        max_iterations: Max boosting iterations (for computing early stop patience).
    """
    patience = compute_early_stop_patience(max_iterations, pct=early_stop_pct)

    if model_name == "lgbm":
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric=_wape_lgbm,
            categorical_feature=cat_cols,
            callbacks=[
                lib_module.early_stopping(stopping_rounds=patience, verbose=False),
                lib_module.log_evaluation(period=-1),
            ],
        )
    elif model_name == "catboost":
        cat_indices = [feature_cols.index(c) for c in cat_cols if c in feature_cols]
        eval_pool = lib_module.Pool(X_val, y_val, cat_features=cat_indices)
        model.set_params(eval_metric=WapeMetric())
        model.fit(
            X_tr, y_tr,
            cat_features=cat_indices,
            eval_set=eval_pool,
            early_stopping_rounds=patience,
            verbose=False,
        )
    elif model_name == "xgboost":
        model.set_params(eval_metric=_wape_xgb, early_stopping_rounds=patience)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown model: {model_name!r}. Expected 'lgbm', 'catboost', or 'xgboost'.")
