"""Centralized model registry and parameter abstraction for tree-based backtests.

Provides:
- Canonical ↔ native parameter name mapping per model library
- Unified ``fit_model()`` that replaces duplicate if/elif/else fit blocks
- ``get_best_iteration()`` abstracting attribute name differences
- ``compute_early_stop_patience()`` for standardized 5% patience
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

EARLY_STOP_PCT = 0.05  # 5% of max iterations
EARLY_STOP_FLOOR = 10  # minimum patience rounds
SPARSE_EARLY_STOP_PCT = 0.10  # 10% of max iterations for sparse/intermittent clusters
SPARSE_EARLY_STOP_FLOOR = 50  # minimum patience for sparse clusters


def compute_early_stop_patience(
    max_iterations: int,
    pct: float = EARLY_STOP_PCT,
    *,
    sparse: bool = False,
) -> int:
    """Compute standardized early stopping patience as a percentage of max iterations.

    Returns max(floor, int(max_iterations * pct)) to ensure a minimum of 10 rounds
    (or 50 rounds for sparse clusters).

    For sparse/intermittent clusters, WAPE is noisy due to small denominators in
    the validation set, so we need more patience to avoid premature stopping.
    When ``sparse=True``, uses 10% of max iterations with a floor of 50 rounds.
    """
    if sparse:
        effective_pct = max(pct, SPARSE_EARLY_STOP_PCT)
        return max(SPARSE_EARLY_STOP_FLOOR, int(max_iterations * effective_pct))
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

    The denominator uses a scaled floor: max(|sum(A)|, n_samples * 0.01) with a
    hard minimum of 1.0.  For sparse validation sets where most actuals are zero,
    the raw |sum(A)| can be very small, making WAPE extremely noisy and causing
    premature early stopping.  The scaled floor stabilises the metric while
    preserving correct behavior for normal-volume data.
    """
    abs_sum = abs(y_true.sum())
    # Scaled floor: at least 1% of sample count, never less than 1.0
    floor = max(len(y_true) * 0.01, 1.0)
    denom = max(abs_sum, floor)
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
        abs_sum = abs(y_true.sum())
        floor = max(len(y_true) * 0.01, 1.0)
        denom = max(abs_sum, floor)
        wape = float(np.sum(np.abs(y_pred - y_true)) / denom)
        return wape, 1.0


def _wape_xgb(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """XGBoost custom eval function that computes WAPE.

    Signature follows the xgboost 3.x **sklearn** eval_metric contract:
    ``(y_true, y_pred) -> float``  (function ``__name__`` is used as metric name).

    Uses the same scaled denominator floor as ``_wape_lgbm`` for stability
    on sparse validation sets.
    """
    abs_sum = abs(y_true.sum())
    floor = max(len(y_true) * 0.01, 1.0)
    denom = max(abs_sum, floor)
    return float(np.sum(np.abs(y_pred - y_true)) / denom)


# ---------------------------------------------------------------------------
# Unified fit function
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Algorithm factory (build_model)
# ---------------------------------------------------------------------------

class UnknownAlgorithm(ValueError):
    """Raised when ``build_model`` is called with an unknown algorithm id.

    Subclasses ``ValueError`` so existing ``except ValueError`` blocks
    continue to catch it, but also lets callers discriminate if they want.
    """


def _base_model_name(model_id: str, algo_type: str) -> str:
    """Map a config-level model_id to the canonical backend name used by fit/params helpers.

    Examples:
        ``lgbm_cluster``        -> ``lgbm``
        ``catboost_cust_enriched`` -> ``catboost``
        ``xgboost_cluster``     -> ``xgboost``
        ``chronos2``            -> ``chronos2`` (foundation; returned as-is)

    For foundation / deep_learning / statistical models we just return the
    model_id — they do not use the tree param mapping.
    """
    if algo_type == "tree":
        for prefix in ("lgbm", "catboost", "xgboost"):
            if model_id.startswith(prefix):
                return prefix
    return model_id


def build_model(algorithm_id: str, params: dict | None = None) -> Any:
    """Construct a configured estimator for ``algorithm_id``.

    Looks the algorithm up in ``forecast_pipeline_config.yaml`` ``algorithms:``
    section and returns an instantiated estimator:

    - Tree models (``type: tree``) return real ``LGBMRegressor`` /
      ``CatBoostRegressor`` / ``XGBRegressor`` instances with hyperparameters
      taken from the config (overridable via ``params``).  Canonical keys are
      translated to each library's native names via :func:`to_native_params`.
    - Foundation / deep_learning / statistical models return a small stub
      object that records the algorithm id and params but does not load the
      underlying model weights.  The stub's docstring contains a ``TODO``
      explaining where the real loader should live; call sites for those
      families currently dispatch through their own script-level loaders
      (e.g. ``run_backtest_chronos.py``).

    Raises:
        UnknownAlgorithm: if ``algorithm_id`` is not present in the config.
    """
    # Import inside the function to avoid a hard dependency at import time
    # (model_registry is imported from scripts that don't all load the YAML).
    from common.core.utils import get_algorithm_params, load_forecast_pipeline_config

    cfg = load_forecast_pipeline_config()
    algorithms = cfg.get("algorithms", {}) or {}
    entry = algorithms.get(algorithm_id)
    if entry is None:
        raise UnknownAlgorithm(
            f"Unknown algorithm id: {algorithm_id!r}. "
            f"Expected one of: {sorted(algorithms.keys())}"
        )

    algo_type = entry.get("type", "tree")
    base_name = _base_model_name(algorithm_id, algo_type)

    # Resolve params: start with config defaults, override with caller kwargs.
    resolved = dict(get_algorithm_params(algorithm_id))
    if params:
        resolved.update(params)

    if algo_type == "tree":
        native = to_native_params(base_name, resolved)
        if base_name == "lgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**native)
        if base_name == "catboost":
            from catboost import CatBoostRegressor
            # CatBoost ignores unknown kwargs via allow_writing_files style
            # semantics but common unsupported keys should be dropped first.
            return CatBoostRegressor(**native)
        if base_name == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(**native)
        raise UnknownAlgorithm(
            f"Tree algorithm {algorithm_id!r} mapped to unknown backend {base_name!r}"
        )

    # Non-tree families: return a lightweight stub. Real loaders live in
    # scripts/run_backtest_<family>.py for now.
    return _FoundationStub(algorithm_id=algorithm_id, algo_type=algo_type, params=resolved)


class _FoundationStub:
    """Lightweight stand-in for foundation / DL / statistical algorithms.

    TODO: replace with real loaders once we have a common foundation-model
    wrapper.  For now, scripts like ``run_backtest_chronos.py`` still call
    their own loaders directly; ``build_model`` just surfaces the declared
    algorithm so tests and generic orchestration code can introspect it.
    """

    __slots__ = ("algorithm_id", "algo_type", "params")

    def __init__(self, algorithm_id: str, algo_type: str, params: dict):
        self.algorithm_id = algorithm_id
        self.algo_type = algo_type
        self.params = dict(params)

    def __repr__(self) -> str:
        return f"_FoundationStub({self.algorithm_id!r}, type={self.algo_type!r})"


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
    *,
    demand_pattern: str = "continuous",
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
        demand_pattern: Cluster demand pattern ("continuous", "lumpy", or "intermittent").
            Sparse patterns get increased early stopping patience to compensate for
            noisy WAPE on validation sets with many zeros.
    """
    is_sparse = demand_pattern in ("intermittent", "lumpy")
    patience = compute_early_stop_patience(max_iterations, pct=early_stop_pct, sparse=is_sparse)

    if model_name == "lgbm":
        # Use WAPE (not MAE) so early stopping optimises the same metric we report.
        # Training objective is MAE/Tweedie; stopping on WAPE drives stopping
        # toward the actual accuracy KPI used in champion selection.
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
        # Do NOT override eval_metric — CatBoost must monitor its own loss function.
        # Setting eval_metric="MAE" conflicts with Tweedie loss on intermittent clusters
        # and causes early stopping at suboptimal points for RMSE clusters.
        # Note: custom_metric only tracks additional metrics during training — it does
        # NOT change which metric drives early stopping, so it provides no benefit here.
        model.fit(
            X_tr, y_tr,
            cat_features=cat_indices,
            eval_set=eval_pool,
            early_stopping_rounds=patience,
            verbose=False,
        )
    elif model_name == "xgboost":
        # Use WAPE (not MAE) so early stopping optimises the same metric we report,
        # consistent with the LGBM branch above.
        _wape_xgb.__name__ = "wape"  # XGBoost uses __name__ as metric label
        model.set_params(eval_metric=_wape_xgb, early_stopping_rounds=patience)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown model: {model_name!r}. Expected 'lgbm', 'catboost', or 'xgboost'.")
