"""
Production Model Training — final-refits persisted models on full history.

Unlike backtesting (which trains on partial history via expanding windows for evaluation),
this script trains models on ALL available sales history up to the planning date, creating
production-ready artifacts for the forecast generation pipeline.

The key difference:
  - Backtest: trains on partial history (expanding window, last window ~34 months)
  - Production: trains on ALL available history (potentially 60+ months)

This ensures production forecasts leverage the full signal in the data.

Output:
  - data/models/lgbm_cluster/production_tree/versions/{id}/ — immutable set
  - data/models/lgbm_cluster/production_tree/active.json — atomic active pointer

Usage:
    uv run python scripts/ml/train_production_models.py --model lgbm_cluster
    uv run python scripts/ml/train_production_models.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import logging
import subprocess
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.constants import (  # noqa: E402
    CAT_FEATURES,
    compute_min_cluster_rows,
)
from common.core.db import get_db_params  # noqa: E402
from common.core.planning_date import get_planning_date  # noqa: E402
from common.core.utils import (  # noqa: E402
    get_algorithm_roster,
    load_forecast_pipeline_config,
)
from common.ml.backtest_config import build_backtest_config_snapshot  # noqa: E402
from common.ml.backtest_framework import (  # noqa: E402
    _inject_recursive_noise,
    compute_cluster_demand_stats,
    load_backtest_data,
    resolve_cluster_params,
)
from common.ml.model_registry import (  # noqa: E402
    build_tree_model,
    fit_final_model,
    fit_model,
    get_best_iteration,
    get_tree_default_params,
)
from common.ml.neural_artifacts import (  # noqa: E402
    LoadedNeuralArtifact,
    load_neural_training_cohort_identity,
    publish_neural_artifact,
)
from common.ml.neural_forecast import (  # noqa: E402
    SUPPORTED_NEURAL_MODELS,
    NeuralCohortIdentity,
    fit_neural_model,
)
from common.ml.shap_selector import (  # noqa: E402
    compute_shap_global,
    compute_timeframe_shap_per_cluster,
)
from common.ml.tree_artifact_lineage import (  # noqa: E402
    TREE_ARTIFACT_LINEAGE_KEY,
    ProductionTreeArtifactLineage,
)
from common.ml.tree_artifacts import (  # noqa: E402
    TreeArtifactSpec,
    build_production_tree_model_config_payload,
    build_tree_artifact_spec,
    get_production_validation_fraction,
    publish_tree_artifact_set,
)
from common.ml.tuning import load_best_params  # noqa: E402
from common.scripts_base import load_project_env, setup_logging  # noqa: E402
from common.services.cluster_lineage import load_promoted_cluster_population  # noqa: E402
from common.services.forecast_population import resolve_forecast_sales_table  # noqa: E402
from common.services.perf_profiler import profiled_section  # noqa: E402
from common.services.sales_lineage import load_completed_sales_lineage  # noqa: E402

logger = logging.getLogger(__name__)


def _train_model_in_subprocess(model_id: str) -> int:
    """Train one model in a clean runtime to isolate native ML libraries.

    LightGBM and PyTorch each load an OpenMP runtime. Running them sequentially
    in one macOS process can deadlock PyTorch tensor construction, so the
    ``--all`` coordinator gives every persisted model its own process.
    """
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "ml" / "train_production_models.py"),
        "--model",
        model_id,
    ]
    logger.info("Starting isolated production training process for %s", model_id)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode

# ── Model class/library registry ─────────────────────────────────────────────
# Maps the supported LGBM model to its import path and metadata.
# Mirrors scripts/run_backtest.py MODEL_REGISTRY but only the fields needed
# for production training (no backtest-specific options).

_MODEL_LIBRARY: dict[str, dict[str, Any]] = {
    "lgbm": {
        "class": "lightgbm.LGBMRegressor",
        "iter_param": "n_estimators",
        "cat_dtype": "category",
        "needs_cat_dtype_cast": False,
        "constant_target_guard": True,
        "feature_importance_fn": lambda model: model.feature_importances_,
        "default_params_fn": lambda algo, seed=42: get_tree_default_params("lgbm", algo, seed=seed),
    },
}


def _production_validation_fraction(config: dict[str, Any]) -> float:
    """Read the final-fit validation fraction through the shared artifact contract."""
    return get_production_validation_fraction(config)

# ── Demand pattern classification (mirrors run_backtest.py) ──────────────────


def _classify_cluster_demand(
    train_c: pd.DataFrame,
    *,
    intermittent_threshold: float = 0.5,
    lumpy_threshold: float = 0.3,
) -> str:
    """Classify a cluster's demand pattern based on zero-demand percentage."""
    if "qty" not in train_c.columns or len(train_c) == 0:
        return "continuous"
    zero_pct = (train_c["qty"] == 0).mean()
    if zero_pct >= intermittent_threshold:
        return "intermittent"
    if zero_pct > lumpy_threshold:
        return "lumpy"
    return "continuous"


def _apply_tweedie_objective(
    params: dict[str, object],
    model_name: str,
    demand_pattern: str,
) -> dict[str, object]:
    """Return params with objective adjusted for non-continuous demand.

    For intermittent patterns (>70% zeros), force MAE objective.
    For lumpy/continuous patterns, keep default objective.
    """
    if demand_pattern in ("continuous", "lumpy"):
        return params

    # Intermittent: force MAE objective
    if model_name == "lgbm":
        merged = {**params, "objective": "regression_l1"}
        merged.pop("tweedie_variance_power", None)
    else:
        merged = dict(params)

    return merged


def _import_model_class(dotted_path: str) -> type:
    """Dynamically import a model class from a dotted module.class path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _resolve_model_name(model_id: str) -> str:
    """Resolve the sole supported tree-model library from a model id."""
    if model_id == "lgbm_cluster":
        return "lgbm"
    raise ValueError(
        f"Cannot resolve model library from model_id={model_id!r}. Expected lgbm_cluster."
    )


def _resolve_params_file_path(params_file: str | Path) -> Path:
    """Resolve a tuned-params file path relative to the project root."""
    path = Path(params_file)
    if path.is_absolute():
        return path
    return ROOT / path


def _apply_tuned_params_file(
    params: dict[str, Any],
    *,
    params_file: str | Path | None,
    iter_param: str,
    model_id: str,
    model_name: str,
) -> tuple[dict[str, Any], str]:
    """Overlay a tuning artifact on production training params."""
    if not params_file:
        return dict(params), "config_defaults"

    path = _resolve_params_file_path(params_file)
    tuning_data = load_best_params(path)
    artifact_model = tuning_data.get("model")
    allowed_models = {model_id, model_name}
    if artifact_model and artifact_model not in allowed_models:
        raise ValueError(
            f"Tuning artifact {path} is for model {artifact_model!r}, not {model_id!r}."
        )

    resolved = dict(params)
    resolved.update(tuning_data.get("best_params", {}) or {})
    n_est_tuned = tuning_data.get("best_n_estimators")
    if n_est_tuned:
        resolved[iter_param] = n_est_tuned
    return resolved, f"tuning_file:{path}"


# ── Single cluster training ─────────────────────────────────────────────────


def _categorical_encoders_from_frame(
    frame: pd.DataFrame,
    categorical_columns: list[str],
) -> dict[str, dict[str, int]]:
    """Capture the exact pandas category codes passed to the final fit."""
    encoders: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        if column not in frame.columns:
            raise ValueError(f"Categorical training column is missing: {column}")
        series = frame[column]
        if not isinstance(series.dtype, pd.CategoricalDtype):
            raise ValueError(f"Categorical training column is not category dtype: {column}")
        levels = [str(level) for level in series.cat.categories]
        if len(levels) != len(set(levels)):
            raise ValueError(
                f"Categorical training levels collide after string conversion: {column}"
            )
        encoders[column] = {level: code for code, level in enumerate(levels)}
    return encoders


def _recursive_noise_seed(
    *,
    random_state: int,
    cluster_label: str,
    feature_name: str,
) -> int:
    """Derive a stable feature-local seed independent of cluster iteration order."""
    identity = f"{random_state}\0{cluster_label}\0{feature_name}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")


def _prepare_recursive_training_features(
    train_data: pd.DataFrame,
    *,
    feature_cols: list[str],
    cluster_label: str,
    backtest_cfg: dict[str, Any],
    random_state: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the configured recursive lag-noise contract without mutating input."""
    configured_enabled = backtest_cfg.get("recursive_noise_enabled", False)
    if not isinstance(configured_enabled, bool):
        raise ValueError("backtest.recursive_noise_enabled must be true or false")
    configured_pct = backtest_cfg.get("recursive_noise_pct")
    if configured_pct is None:
        if configured_enabled:
            raise ValueError(
                "backtest.recursive_noise_pct is required when recursive noise is enabled"
            )
        noise_pct = 0.0
    elif isinstance(configured_pct, bool) or not isinstance(configured_pct, (int, float)):
        raise ValueError("backtest.recursive_noise_pct must be numeric")
    else:
        noise_pct = float(configured_pct)
    if noise_pct < 0:
        raise ValueError("backtest.recursive_noise_pct must be non-negative")

    lag_features = [column for column in feature_cols if column.startswith("qty_lag_")]
    enabled = configured_enabled and noise_pct > 0 and bool(lag_features)
    if enabled and (isinstance(random_state, bool) or not isinstance(random_state, int)):
        raise ValueError(
            "LightGBM random_state must be an integer when recursive noise is enabled"
        )

    prepared = train_data[feature_cols].copy(deep=True)
    feature_seeds: dict[str, int] = {}
    if enabled:
        assert random_state is not None  # narrowed by validation above
        for column in lag_features:
            seed = _recursive_noise_seed(
                random_state=random_state,
                cluster_label=cluster_label,
                feature_name=column,
            )
            feature_seeds[column] = seed
            values = prepared[column].to_numpy(dtype=float, copy=True)
            prepared[column] = _inject_recursive_noise(
                values,
                noise_pct,
                rng=np.random.default_rng(seed),
            )

    return prepared, {
        "enabled": enabled,
        "pct": noise_pct,
        "random_state": random_state,
        "lag_features": lag_features,
        "feature_seeds": feature_seeds,
    }


def _select_production_cluster_features(
    *,
    models: dict[str, Any],
    train_data: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    algo_params: dict[str, Any],
    backtest_cfg: dict[str, Any],
    cutoff_date: pd.Timestamp,
) -> tuple[dict[str, list[str]], dict[str, list[str]], pd.DataFrame]:
    """Run the same configured per-cluster SHAP selector as tree backtesting."""
    cluster_labels = set(models)
    if not cluster_labels:
        raise RuntimeError("Production SHAP selection requires trained cluster models")
    if not algo_params.get("shap_select", False):
        full = {label: list(feature_cols) for label in cluster_labels}
        return full, full, pd.DataFrame()

    required_algo_keys = (
        "shap_threshold",
        "shap_top_n",
        "shap_sample_size",
        "correlation_filter",
        "correlation_threshold",
        "variance_filter",
        "variance_threshold",
    )
    missing_algo = [key for key in required_algo_keys if key not in algo_params]
    required_backtest_keys = ("shap_retrain_threshold", "shap_min_features")
    missing_backtest = [key for key in required_backtest_keys if key not in backtest_cfg]
    if missing_algo or missing_backtest:
        missing = [
            *(f"algorithms.lgbm_cluster.params.{key}" for key in missing_algo),
            *(f"backtest.{key}" for key in missing_backtest),
        ]
        raise ValueError(f"Production SHAP configuration is incomplete: {', '.join(missing)}")

    raw_selected, shap_report = compute_timeframe_shap_per_cluster(
        models,
        train_data,
        feature_cols,
        cat_cols,
        0,
        cutoff_date,
        shap_extractor_fn=compute_shap_global,
        sample_size=int(algo_params["shap_sample_size"]),
        cumulative_threshold=float(algo_params["shap_threshold"]),
        top_n=algo_params["shap_top_n"],
        min_features=int(backtest_cfg["shap_min_features"]),
        correlation_filter=bool(algo_params["correlation_filter"]),
        correlation_threshold=float(algo_params["correlation_threshold"]),
        variance_filter=bool(algo_params["variance_filter"]),
        variance_threshold=float(algo_params["variance_threshold"]),
    )
    normalized = {str(label): list(columns) for label, columns in raw_selected.items()}
    if set(normalized) != cluster_labels:
        raise RuntimeError(
            "Production SHAP selection did not return the exact trained cluster population"
        )

    full_feature_set = set(feature_cols)
    retrain_threshold = float(backtest_cfg["shap_retrain_threshold"])
    if not 0 <= retrain_threshold <= 1:
        raise ValueError("backtest.shap_retrain_threshold must be between 0 and 1")
    effective: dict[str, list[str]] = {}
    for cluster_label, selected in normalized.items():
        if not selected or len(selected) != len(set(selected)):
            raise RuntimeError(
                f"Production SHAP returned invalid features for cluster {cluster_label!r}"
            )
        unknown = set(selected) - full_feature_set
        if unknown:
            raise RuntimeError(
                f"Production SHAP returned unknown features for cluster {cluster_label!r}: "
                f"{sorted(unknown)}"
            )
        drop_pct = (len(feature_cols) - len(selected)) / len(feature_cols)
        effective[cluster_label] = (
            selected
            if drop_pct >= retrain_threshold and set(selected) != full_feature_set
            else list(feature_cols)
        )
    return effective, normalized, shap_report


def _retrain_selected_cluster_models(
    *,
    train_data: pd.DataFrame,
    clusters: list[str],
    initial_models: dict[str, Any],
    initial_metadata: dict[str, dict[str, Any]],
    selected_feature_cols: dict[str, list[str]],
    full_feature_cols: list[str],
    cat_cols: list[str],
    params: dict[str, Any],
    model_name: str,
    model_class: type,
    lib_module: Any,
    iter_param: str,
    needs_cat_dtype_cast: bool,
    constant_target_guard: bool,
    backtest_cfg: dict[str, Any],
    validation_fraction: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Retrain selected subsets and retain the original when validation worsens."""
    expected = set(clusters)
    if set(initial_models) != expected or set(initial_metadata) != expected:
        raise RuntimeError("Initial LightGBM fit does not cover the exact cluster population")
    if set(selected_feature_cols) != expected:
        raise RuntimeError("Production SHAP feature map does not cover every cluster")

    models = dict(initial_models)
    metadata = {label: dict(value) for label, value in initial_metadata.items()}
    effective = {label: list(full_feature_cols) for label in clusters}
    full_set = set(full_feature_cols)
    for ci, cluster_label in enumerate(clusters, 1):
        selected = list(selected_feature_cols[cluster_label])
        selection_meta = {
            "selected_features": selected,
            "retrained": False,
            "reverted": False,
        }
        if set(selected) == full_set and len(selected) == len(full_feature_cols):
            metadata[cluster_label]["feature_selection"] = selection_meta
            continue

        label, retrained_model, retrained_meta = _train_cluster(
            cluster_label=cluster_label,
            ci=ci,
            n_clusters=len(clusters),
            train_c=train_data[train_data["ml_cluster"] == cluster_label],
            feature_cols=selected,
            cat_cols=[column for column in cat_cols if column in selected],
            params=params,
            model_name=model_name,
            model_class=model_class,
            lib_module=lib_module,
            iter_param=iter_param,
            needs_cat_dtype_cast=needs_cat_dtype_cast,
            constant_target_guard=constant_target_guard,
            backtest_cfg=backtest_cfg,
            validation_fraction=validation_fraction,
        )
        if label != cluster_label:
            raise RuntimeError("SHAP retrain returned the wrong cluster label")
        selection_meta["retrained"] = True
        original_wape = metadata[cluster_label].get("val_wape")
        retrained_wape = retrained_meta.get("val_wape")
        should_revert = (
            retrained_model is None
            or not isinstance(original_wape, (int, float))
            or not isinstance(retrained_wape, (int, float))
            or float(retrained_wape) > float(original_wape)
        )
        if should_revert:
            selection_meta["reverted"] = True
            metadata[cluster_label]["feature_selection"] = selection_meta
            continue

        retrained_meta = dict(retrained_meta)
        retrained_meta["feature_selection"] = selection_meta
        models[cluster_label] = retrained_model
        metadata[cluster_label] = retrained_meta
        effective[cluster_label] = selected
    return models, metadata, effective


def _train_cluster(
    cluster_label: str,
    ci: int,
    n_clusters: int,
    train_c: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict[str, Any],
    *,
    model_name: str,
    model_class: type,
    lib_module: Any,
    iter_param: str,
    needs_cat_dtype_cast: bool,
    constant_target_guard: bool,
    backtest_cfg: dict[str, Any],
    validation_fraction: float,
) -> tuple[str, Any | None, dict[str, Any]]:
    """Train a single cluster model on full history. Returns (label, model, meta)."""
    t0 = time.time()

    min_rows = compute_min_cluster_rows(len(feature_cols))
    if len(train_c) < min_rows:
        logger.warning(
            "Cluster %d/%d '%s': skipped — insufficient rows (%d < %d)",
            ci,
            n_clusters,
            cluster_label,
            len(train_c),
            min_rows,
        )
        return (
            cluster_label,
            None,
            {"skipped": True, "reason": "insufficient_rows", "n_rows": len(train_c)},
        )

    cat_cols_in_features = [c for c in cat_cols if c in feature_cols]

    # Sort by (startdate, sku_ck) for temporal ordering
    train_c = train_c.sort_values(["startdate", "sku_ck"])

    y_all = train_c["qty"]

    # Guard: models crash on constant targets
    if constant_target_guard and y_all.nunique() <= 1:
        const_val = float(y_all.iloc[0]) if len(y_all) > 0 else 0.0
        logger.info(
            "Cluster %d/%d '%s': constant target (%.0f) — skipping model training",
            ci,
            n_clusters,
            cluster_label,
            const_val,
        )
        return (
            cluster_label,
            None,
            {"skipped": True, "reason": "constant_target", "constant_value": const_val},
        )

    # Per-cluster adaptive hyperparameters
    cluster_stats = compute_cluster_demand_stats(train_c, cluster_label)
    resolved_params, profile_name = resolve_cluster_params(
        cluster_label,
        cluster_stats,
        params,
    )

    # Filter resolved params to only include keys valid for this model
    valid_keys = set(params.keys())
    fit_params = {k: v for k, v in resolved_params.items() if k in valid_keys}

    X_all, recursive_noise = _prepare_recursive_training_features(
        train_c,
        feature_cols=feature_cols,
        cluster_label=cluster_label,
        backtest_cfg=backtest_cfg,
        random_state=fit_params.get("random_state"),
    )
    if needs_cat_dtype_cast:
        for col in cat_cols_in_features:
            X_all[col] = X_all[col].astype("category")
    categorical_encoders = _categorical_encoders_from_frame(
        X_all,
        cat_cols_in_features,
    )

    # Calendar-month-based train/val split — the configured tail of unique months
    # is validation for early stopping. Full data is still used for the final model,
    # but we need validation to determine the best iteration. Both fits use the same
    # deterministic noisy feature frame so the selected boosting round matches the
    # persisted refit.
    unique_months = sorted(train_c["startdate"].unique())
    n_val_months = max(1, int(len(unique_months) * validation_fraction))
    val_months = set(unique_months[-n_val_months:])
    val_mask = train_c["startdate"].isin(val_months)

    X_tr, X_val = X_all.loc[~val_mask], X_all.loc[val_mask]
    y_tr, y_val = y_all.loc[~val_mask], y_all.loc[val_mask]

    # Classify demand pattern and adjust objective if needed.
    # Fallback matches config default (forecast_pipeline_config.yaml: 0.7) and the
    # documented ">70% zero-demand → intermittent baseline" routing.
    intermittent_threshold = backtest_cfg.get("intermittent_threshold", 0.7)
    lumpy_threshold = backtest_cfg.get("lumpy_threshold", 0.3)
    demand_pattern = _classify_cluster_demand(
        train_c,
        intermittent_threshold=intermittent_threshold,
        lumpy_threshold=lumpy_threshold,
    )
    fit_params = _apply_tweedie_objective(fit_params, model_name, demand_pattern)

    max_iters = fit_params[iter_param]
    model = build_tree_model(model_name, fit_params)

    # Fit with early stopping using validation set
    fit_model(
        model,
        model_name,
        X_tr,
        y_tr,
        X_val,
        y_val,
        cat_cols_in_features,
        feature_cols,
        lib_module,
        max_iters,
        demand_pattern=demand_pattern,
    )

    # Compute validation WAPE for metadata
    val_preds = model.predict(X_val)
    val_abs_sum = float(abs(y_val.sum()))
    val_floor = max(len(y_val) * 0.01, 1.0)
    val_denom = max(val_abs_sum, val_floor)
    val_wape = round(float((abs(val_preds - y_val.values)).sum() / val_denom * 100), 2)

    n_est_used = get_best_iteration(model, model_name)
    if n_est_used is None:
        n_est_used = fit_params[iter_param]

    # Refit the persisted artifact on ALL available history with the selected
    # boosting round. The split model above is only for early-stopping/validation
    # measurement; shipping it would discard the most recent validation months.
    final_params = dict(fit_params)
    final_params[iter_param] = n_est_used
    final_model = build_tree_model(model_name, final_params)
    fit_final_model(
        final_model,
        model_name,
        X_all,
        y_all,
        cat_cols_in_features,
        feature_cols,
    )

    meta = {
        "val_wape": val_wape,
        "train_rows": len(X_all),
        "early_stop_train_rows": len(X_tr),
        "total_rows": len(X_all),
        "val_rows": len(X_val),
        "n_estimators_used": n_est_used,
        "cluster_profile": profile_name,
        "demand_pattern": demand_pattern,
        "cluster_stats": cluster_stats,
        "n_val_months": n_val_months,
        "n_train_months": len(unique_months) - n_val_months,
        "validation_fraction": validation_fraction,
        "categorical_encoders": categorical_encoders,
        "recursive_noise": recursive_noise,
    }

    val_accuracy = round(100.0 - val_wape, 2)
    logger.info(
        "Cluster %d/%d '%s': total=%s, train=%s, val=%s, best_iter=%s, "
        "val_accuracy=%.1f%%, profile=%s, pattern=%s (%.1fs)",
        ci,
        n_clusters,
        cluster_label,
        f"{len(X_all):,}",
        f"{len(X_tr):,}",
        f"{len(X_val):,}",
        n_est_used,
        val_accuracy,
        profile_name,
        demand_pattern,
        time.time() - t0,
    )

    return cluster_label, final_model, meta


# ── Artifact persistence ────────────────────────────────────────────────────


def _build_cluster_artifact(
    cluster_label: str,
    model: Any,
    feature_cols: list[str],
    model_id: str,
    model_name: str,
    meta: dict[str, Any],
    tree_spec: TreeArtifactSpec,
) -> dict[str, Any]:
    """Build one cluster payload for an atomic all-cluster publication."""
    if tree_spec.model_id != model_id:
        raise ValueError("Tree artifact spec model does not match cluster artifact")
    if str(cluster_label) not in tree_spec.cluster_labels:
        raise ValueError("Cluster artifact label is absent from the expected cluster set")
    recursive_training = tree_spec.model_config.get("recursive_training")
    if recursive_training is not None and not isinstance(recursive_training, dict):
        raise ValueError("Tree artifact recursive_training contract must be a mapping")
    _get_importance = _MODEL_LIBRARY[model_name]["feature_importance_fn"]
    n_est_used = meta.get("n_estimators_used") or get_best_iteration(model, model_name) or 0

    try:
        importance_raw = _get_importance(model)
    except AttributeError:
        importance_raw = []
    importance_dict = (
        dict(zip(feature_cols, [float(v) for v in importance_raw], strict=True))
        if len(importance_raw) == len(feature_cols)
        else {}
    )
    raw_encoders = meta.get("categorical_encoders", {})
    if not isinstance(raw_encoders, dict):
        raise ValueError("categorical_encoders must be a mapping")
    categorical_features = [column for column in feature_cols if column in CAT_FEATURES]
    missing_encoders = [column for column in categorical_features if column not in raw_encoders]
    if missing_encoders:
        raise ValueError(
            f"Production artifact is missing categorical encoder(s): {', '.join(missing_encoders)}"
        )
    categorical_encoders: dict[str, dict[str, int]] = {}
    for column in categorical_features:
        mapping = raw_encoders[column]
        if not isinstance(mapping, dict):
            raise ValueError(f"Categorical encoder must be a mapping: {column}")
        normalized = {str(level): code for level, code in mapping.items()}
        codes = list(normalized.values())
        if any(not isinstance(code, int) or isinstance(code, bool) for code in codes):
            raise ValueError(f"Categorical encoder codes must be integers: {column}")
        if sorted(codes) != list(range(len(codes))):
            raise ValueError(f"Categorical encoder codes must be contiguous: {column}")
        categorical_encoders[column] = normalized

    return {
        "model": model,
        "feature_cols": feature_cols,
        "model_id": model_id,
        "cluster_label": str(cluster_label),
        "n_estimators_used": n_est_used,
        "train_rows": meta.get("train_rows"),
        "val_wape": meta.get("val_wape"),
        "trained_at": datetime.now(UTC).isoformat(),
        "timeframe": "production",
        "training_mode": "production",
        "cluster_strategy": tree_spec.cluster_strategy,
        "n_rows": meta.get("total_rows"),
        "feature_importance": importance_dict,
        "categorical_encoders": categorical_encoders,
        "recursive_noise": meta.get("recursive_noise"),
        "recursive_training": dict(recursive_training) if recursive_training is not None else None,
        "feature_selection": meta.get("feature_selection"),
        "config_checksum": tree_spec.config_checksum,
        TREE_ARTIFACT_LINEAGE_KEY: tree_spec.lineage.to_metadata(),
    }


def _build_training_metadata(
    *,
    model_id: str,
    planning_date: str,
    params_source: str,
    cluster_results: dict[str, dict[str, Any]],
    feature_cols_per_cluster: dict[str, list[str]],
    total_rows: int,
    total_dfus: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build the auditable metadata for one complete production final fit."""
    n_clusters_trained = sum(1 for m in cluster_results.values() if not m.get("skipped", False))
    n_clusters_skipped = sum(1 for m in cluster_results.values() if m.get("skipped", False))

    return {
        "model_id": model_id,
        "trained_at": datetime.now(UTC).isoformat(),
        "training_mode": "production",
        "planning_date": planning_date,
        "params_source": params_source,
        "n_clusters": n_clusters_trained,
        "n_clusters_skipped": n_clusters_skipped,
        "n_dfus": total_dfus,
        "n_rows": total_rows,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "feature_cols_per_cluster": {str(k): v for k, v in feature_cols_per_cluster.items()},
        "cluster_details": {str(k): v for k, v in cluster_results.items()},
    }


def _load_completed_sales_lineage(db: dict[str, Any]) -> tuple[int, str]:
    """Return the latest completed sales batch and its immutable source hash."""
    with psycopg.connect(**db) as conn:
        lineage = load_completed_sales_lineage(conn)
    return lineage.batch_id, lineage.source_hash


def _load_current_neural_training_cohort(
    db: dict[str, Any],
    *,
    history_end: date | pd.Timestamp,
    min_history: int,
) -> NeuralCohortIdentity:
    """Return the current normalized neural roster without loading sales values."""
    with psycopg.connect(**db) as conn:
        with conn.cursor() as cursor:
            sales_table = resolve_forecast_sales_table(cursor)
        return load_neural_training_cohort_identity(
            conn,
            sales_table=sales_table,
            history_end=history_end,
            min_history=min_history,
        )


def _load_production_tree_lineage(
    db: dict[str, Any],
    *,
    history_end: date,
    clustering_enabled: bool,
) -> tuple[ProductionTreeArtifactLineage, frozenset[str]]:
    """Load one sales/cluster lineage snapshot for a LightGBM final fit."""
    with psycopg.connect(**db) as conn:
        sales = load_completed_sales_lineage(conn)
        clusters = load_promoted_cluster_population(conn) if clustering_enabled else None
    return (
        ProductionTreeArtifactLineage(
            source_sales_batch_id=sales.batch_id,
            data_checksum=sales.source_hash,
            history_end=history_end,
            cluster_experiment_id=(clusters.experiment_id if clusters else None),
            cluster_assignment_count=(clusters.assignment_count if clusters else None),
            cluster_assignment_checksum=(clusters.assignment_checksum if clusters else None),
        ),
        clusters.cluster_labels if clusters else frozenset({"global"}),
    )


def _select_closed_training_history(
    sales_df: pd.DataFrame,
    *,
    planning_month: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return history capped at, and complete through, the latest closed month."""
    normalized_planning_month = pd.Timestamp(planning_month).normalize().replace(day=1)
    history_end = normalized_planning_month - pd.DateOffset(months=1)
    history = sales_df.copy()
    history["startdate"] = (
        pd.to_datetime(history["startdate"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    )
    if history["startdate"].isna().any():
        raise RuntimeError("Production training history contains an invalid month")
    history = history[history["startdate"] <= history_end].copy()
    latest_month = history["startdate"].max() if not history.empty else None
    if latest_month != history_end:
        available = latest_month.strftime("%Y-%m") if latest_month is not None else "none"
        raise RuntimeError(
            f"Production training requires latest closed month {history_end:%Y-%m}; "
            f"sales history is current through {available}"
        )
    return history, history_end


def _apply_production_cluster_strategy(
    training: pd.DataFrame,
    *,
    clustering_enabled: bool,
) -> pd.DataFrame:
    """Apply the configured per-cluster or single-global final-fit strategy."""
    result = training.copy()
    if not clustering_enabled:
        result["ml_cluster"] = "global"
    return result


def train_production_neural_model(model_id: str) -> LoadedNeuralArtifact:
    """Refit one global neural model on all closed history and publish it."""
    if model_id not in SUPPORTED_NEURAL_MODELS:
        raise ValueError(f"Unsupported production neural model: {model_id}")

    config = load_forecast_pipeline_config()
    build_backtest_config_snapshot(config, model_id)
    algo_entry = config.get("algorithms", {}).get(model_id)
    if not isinstance(algo_entry, dict):
        raise ValueError(f"Forecast configuration is missing algorithm {model_id}")
    if algo_entry.get("type") != "deep_learning" or not algo_entry.get("forecast", False):
        raise ValueError(f"Algorithm {model_id} is not enabled for production forecasting")
    params = algo_entry.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"Forecast configuration is missing parameters for {model_id}")

    db = get_db_params()
    lineage_before = _load_completed_sales_lineage(db)
    sales_df, _dfu_attrs, _item_attrs = load_backtest_data(
        db,
        include_item_attrs=False,
        algo_config=algo_entry,
    )
    lineage_after = _load_completed_sales_lineage(db)
    if lineage_after != lineage_before:
        raise RuntimeError(
            "The completed sales batch changed while neural training data was loaded; retry "
            "the final refit against one stable source batch"
        )
    if sales_df.empty:
        raise RuntimeError("Production neural training requires non-empty sales history")

    sales_df, history_end = _select_closed_training_history(
        sales_df,
        planning_month=pd.Timestamp(get_planning_date()),
    )
    cohort_before = _load_current_neural_training_cohort(
        db,
        history_end=history_end,
        min_history=int(params["min_history"]),
    )

    logger.info(
        "Final-refitting %s on %s rows / %s DFUs through %s (sales batch %s)",
        model_id,
        f"{len(sales_df):,}",
        f"{sales_df['sku_ck'].nunique():,}",
        history_end.strftime("%Y-%m"),
        lineage_before[0],
    )
    fitted = fit_neural_model(sales_df, model_id=model_id, params=dict(params))
    lineage_final = _load_completed_sales_lineage(db)
    if lineage_final != lineage_before:
        raise RuntimeError(
            "The completed sales batch changed while neural model was training; "
            "existing artifacts were left unchanged"
        )
    cohort_final = _load_current_neural_training_cohort(
        db,
        history_end=history_end,
        min_history=int(params["min_history"]),
    )
    if cohort_final != cohort_before:
        raise RuntimeError(
            "The eligible training cohort changed while neural model was training; "
            "existing artifacts were left unchanged"
        )
    if (
        fitted.training_cohort_checksum != cohort_before.checksum
        or fitted.training_dfu_count != cohort_before.dfu_count
    ):
        raise RuntimeError(
            "The normalized neural training frame does not match the current eligible "
            "DFU cohort; retry the final refit from one stable sales/dimension snapshot"
        )
    production_cfg = config.get("production_forecast", {})
    base_path = production_cfg.get("model_registry", {}).get("base_path", "data/models")
    base_dir = Path(base_path)
    if not base_dir.is_absolute():
        base_dir = ROOT / base_dir
    published = publish_neural_artifact(
        fitted=fitted,
        params=params,
        source_sales_batch_id=lineage_before[0],
        data_checksum=lineage_before[1],
        history_end=history_end,
        base_dir=base_dir,
    )
    logger.info(
        "Published immutable %s production artifact %s",
        model_id,
        published.ref.artifact_id,
    )
    return published


# ── Main training pipeline ──────────────────────────────────────────────────


def train_production_model(model_id: str) -> None:
    """Train a single model_id on full history and save production artifacts."""
    from common.ml.feature_engineering import (
        build_feature_matrix,
        get_feature_columns,
    )

    t_start = time.time()
    model_name = _resolve_model_name(model_id)
    lib_info = _MODEL_LIBRARY[model_name]

    logger.info("=" * 70)
    logger.info("Production training: %s (library: %s)", model_id, model_name)
    logger.info("=" * 70)

    # ── Step 1: Load config ─────────────────────────────────────────────────
    with profiled_section("load_config"):
        pcfg = load_forecast_pipeline_config()
        algo_entry = pcfg.get("algorithms", {}).get(model_id, {})
        if not algo_entry:
            logger.error(
                "Model '%s' not found in forecast_pipeline_config.yaml algorithms", model_id
            )
            return

        algo_params = algo_entry.get("params", {})
        backtest_cfg = pcfg.get("backtest", {})
        prod_config = pcfg.get("production_forecast", {})
        validation_fraction = _production_validation_fraction(pcfg)
        clustering_enabled = pcfg.get("clustering", {}).get("enabled")
        if not isinstance(clustering_enabled, bool):
            raise ValueError("clustering.enabled must be explicitly true or false")
        cluster_strategy = "per_cluster" if clustering_enabled else "global"
        tree_model_config = build_production_tree_model_config_payload(
            pcfg,
            model_id=model_id,
            project_root=ROOT,
        )

        # Build model hyperparameters from config
        params = lib_info["default_params_fn"](algo_params)
        iter_param = lib_info["iter_param"]
        params, params_source = _apply_tuned_params_file(
            params,
            params_file=algo_params.get("params_file"),
            iter_param=iter_param,
            model_id=model_id,
            model_name=model_name,
        )
        logger.info("Params source: %s", params_source)
        cat_dtype = lib_info["cat_dtype"]

        # Import model class and library
        model_class = _import_model_class(lib_info["class"])
        lib_module = importlib.import_module(lib_info["class"].rsplit(".", 1)[0])

        # Check if customer features are needed
        include_customer_features = bool(algo_params.get("customer_features", False))

    # ── Step 2: Load data ───────────────────────────────────────────────────
    with profiled_section("load_data"):
        db = get_db_params()
        planning_month = pd.Timestamp(get_planning_date()).normalize().replace(day=1)
        expected_history_end = (planning_month - pd.DateOffset(months=1)).date()
        tree_lineage_before, promoted_cluster_labels = _load_production_tree_lineage(
            db,
            history_end=expected_history_end,
            clustering_enabled=clustering_enabled,
        )
        tree_spec = build_tree_artifact_spec(
            model_id=model_id,
            model_config=tree_model_config,
            lineage=tree_lineage_before,
            cluster_strategy=cluster_strategy,
            cluster_labels=promoted_cluster_labels,
        )
        logger.info("Step 1: Loading data from Postgres...")

        data_result = load_backtest_data(
            db,
            include_item_attrs=True,
            algo_config=algo_entry,
            include_customer_features=include_customer_features,
        )

        if include_customer_features:
            sales_df, dfu_attrs, item_attrs, customer_features = data_result
        else:
            sales_df, dfu_attrs, item_attrs = data_result
            customer_features = None

        # Train and serve from the same latest-closed-month boundary.
        sales_df, latest_month = _select_closed_training_history(
            sales_df,
            planning_month=planning_month,
        )
        tree_lineage_after, current_cluster_labels = _load_production_tree_lineage(
            db,
            history_end=latest_month.date(),
            clustering_enabled=clustering_enabled,
        )
        if (
            tree_lineage_after != tree_lineage_before
            or current_cluster_labels != promoted_cluster_labels
        ):
            raise RuntimeError(
                "Sales or promoted clustering changed while LightGBM training data was loaded; "
                "retry the final refit against one stable lineage"
            )
        earliest_month = sales_df["startdate"].min()

        n_dfus = dfu_attrs["sku_ck"].nunique()
        logger.info(
            "Data loaded: %s sales rows, %s DFUs, date range [%s -> %s]",
            f"{len(sales_df):,}",
            f"{n_dfus:,}",
            earliest_month.date(),
            latest_month.date(),
        )

    # ── Step 3: Build feature matrix ────────────────────────────────────────
    with profiled_section("build_features"):
        logger.info("Step 2: Building feature matrix on FULL history...")
        all_months = sorted(sales_df["startdate"].unique())

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
        logger.info(
            "Feature matrix: %s, features: %d, cat: %s",
            full_grid.shape,
            len(feature_cols),
            cat_cols,
        )

    # ── Step 4: Prepare training data ───────────────────────────────────────
    with profiled_section("prepare_training"):
        # Drop rows without the most recent lag (qty_lag_1) — same as backtest
        train_data = full_grid.dropna(subset=["qty_lag_1"]).copy()
        train_data = _apply_production_cluster_strategy(
            train_data,
            clustering_enabled=clustering_enabled,
        )
        if train_data["ml_cluster"].isna().any():
            raise RuntimeError("LightGBM training data contains missing cluster labels")
        train_data["ml_cluster"] = train_data["ml_cluster"].astype(str)
        training_cluster_labels = {
            str(label) for label in train_data["ml_cluster"].dropna().unique()
        }
        if training_cluster_labels != set(promoted_cluster_labels):
            raise RuntimeError(
                "LightGBM training clusters do not exactly match the promoted cluster "
                "population; refresh features and clustering before final refit"
            )

        # Fill NaN in feature columns (skip categoricals)
        for col in feature_cols:
            if col in train_data.columns and col not in cat_cols:
                train_data[col] = train_data[col].fillna(0)

        logger.info("Training data: %s rows (after dropping NaN lag rows)", f"{len(train_data):,}")

    # ── Step 5: Train per cluster ───────────────────────────────────────────
    with profiled_section("train_clusters"):
        clusters = sorted(train_data["ml_cluster"].dropna().unique())
        n_clusters = len(clusters)
        logger.info(
            "Step 3: Training %d per-cluster %s models on FULL history...",
            n_clusters,
            model_name.upper(),
        )

        # Resolve the versioned artifact registry root.  No active files are
        # touched until the complete set has passed publication validation.
        base_path = prod_config.get("model_registry", {}).get("base_path", "data/models")
        base_dir = Path(base_path)
        if not base_dir.is_absolute():
            base_dir = ROOT / base_dir

        initial_models: dict[str, Any] = {}
        cluster_results: dict[str, dict[str, Any]] = {}

        for ci, cluster_label in enumerate(clusters, 1):
            train_c = train_data[train_data["ml_cluster"] == cluster_label]

            label, trained_model, meta = _train_cluster(
                cluster_label=str(cluster_label),
                ci=ci,
                n_clusters=n_clusters,
                train_c=train_c,
                feature_cols=feature_cols,
                cat_cols=cat_cols,
                params=params,
                model_name=model_name,
                model_class=model_class,
                lib_module=lib_module,
                iter_param=iter_param,
                needs_cat_dtype_cast=lib_info["needs_cat_dtype_cast"],
                constant_target_guard=lib_info["constant_target_guard"],
                backtest_cfg=backtest_cfg,
                validation_fraction=validation_fraction,
            )

            cluster_results[label] = meta
            if trained_model is not None:
                initial_models[label] = trained_model

        if set(initial_models) != set(promoted_cluster_labels):
            missing = sorted(set(promoted_cluster_labels) - set(initial_models))
            raise RuntimeError(
                "Production LightGBM final refit did not produce every promoted cluster "
                f"artifact (missing={missing}); existing artifacts were left unchanged"
            )

        selected_feature_cols, raw_shap_features, shap_report = (
            _select_production_cluster_features(
                models=initial_models,
                train_data=train_data,
                feature_cols=feature_cols,
                cat_cols=cat_cols,
                algo_params=algo_params,
                backtest_cfg=backtest_cfg,
                cutoff_date=latest_month,
            )
        )
        final_models, cluster_results, feature_cols_per_cluster = (
            _retrain_selected_cluster_models(
                train_data=train_data,
                clusters=[str(label) for label in clusters],
                initial_models=initial_models,
                initial_metadata=cluster_results,
                selected_feature_cols=selected_feature_cols,
                full_feature_cols=feature_cols,
                cat_cols=cat_cols,
                params=params,
                model_name=model_name,
                model_class=model_class,
                lib_module=lib_module,
                iter_param=iter_param,
                needs_cat_dtype_cast=lib_info["needs_cat_dtype_cast"],
                constant_target_guard=lib_info["constant_target_guard"],
                backtest_cfg=backtest_cfg,
                validation_fraction=validation_fraction,
            )
        )
        for label, meta in cluster_results.items():
            selection = meta.setdefault("feature_selection", {})
            selection["enabled"] = bool(algo_params.get("shap_select", False))
            selection["shap_selected_features"] = raw_shap_features[label]
            selection["effective_features"] = feature_cols_per_cluster[label]
            selection["shap_retrain_threshold"] = backtest_cfg.get(
                "shap_retrain_threshold"
            )
        logger.info(
            "Training complete: %d/%d clusters final-refit; SHAP report rows=%d",
            len(final_models),
            n_clusters,
            len(shap_report),
        )

        artifacts = {
            label: _build_cluster_artifact(
                cluster_label=label,
                model=final_models[label],
                feature_cols=feature_cols_per_cluster[label],
                model_id=model_id,
                model_name=model_name,
                meta=cluster_results[label],
                tree_spec=tree_spec,
            )
            for label in sorted(final_models)
        }
        n_trained = len(artifacts)

        # A promoted cluster or sales batch change during a long fit makes the
        # whole result stale.  Fail before publication and retain the old active
        # pointer unchanged.
        tree_lineage_final, final_cluster_labels = _load_production_tree_lineage(
            db,
            history_end=latest_month.date(),
            clustering_enabled=clustering_enabled,
        )
        if (
            tree_lineage_final != tree_lineage_before
            or final_cluster_labels != promoted_cluster_labels
        ):
            raise RuntimeError(
                "Sales or promoted clustering changed while LightGBM models were training; "
                "existing artifacts were left unchanged"
            )

    # ── Step 6: Publish complete immutable set ──────────────────────────────
    elapsed = time.time() - t_start
    training_metadata = _build_training_metadata(
        model_id=model_id,
        planning_date=str(get_planning_date()),
        params_source=params_source,
        cluster_results=cluster_results,
        feature_cols_per_cluster=feature_cols_per_cluster,
        total_rows=len(train_data),
        total_dfus=n_dfus,
        elapsed_seconds=elapsed,
    )
    published = publish_tree_artifact_set(
        artifacts=artifacts,
        training_metadata=training_metadata,
        spec=tree_spec,
        base_dir=base_dir,
    )

    logger.info(
        "Production training for %s complete: %d models published as set %s to %s (%.0fs / %.1fm)",
        model_id,
        n_trained,
        published.ref.artifact_set_id,
        published.ref.version_dir,
        elapsed,
        elapsed / 60,
    )


# ── CLI entry point ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train persisted tree and neural models on FULL history for production forecasting. "
            "Unlike backtesting (partial history), this creates artifacts "
            "trained on all available data up to the planning date."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID to train (lgbm_cluster, nbeats, or nhits)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        dest="train_all",
        help="Train all forecastable models that require persisted artifacts",
    )
    args = parser.parse_args()

    load_project_env()

    if args.train_all:
        # Statistical MSTL and pinned Chronos infer directly; LightGBM and the
        # global neural models require persisted final-refit artifacts.
        roster = get_algorithm_roster(stage="forecast")
        persisted_models = [
            mid for mid, entry in roster.items() if entry.get("type") in {"tree", "deep_learning"}
        ]
        if not persisted_models:
            logger.error("No forecastable persisted models found in forecast configuration")
            sys.exit(1)

        logger.info(
            "Training %d forecastable persisted models: %s",
            len(persisted_models),
            persisted_models,
        )
        failed_models: list[str] = []
        for model_id in sorted(persisted_models):
            return_code = _train_model_in_subprocess(model_id)
            if return_code != 0:
                logger.error(
                    "Isolated production training for %s exited with code %d",
                    model_id,
                    return_code,
                )
                failed_models.append(model_id)
                # Continue with remaining models so one bad family does not
                # mask additional missing artifacts in the same production run.
        if failed_models:
            logger.error(
                "Production training failed for %d/%d persisted model(s): %s",
                len(failed_models),
                len(persisted_models),
                ", ".join(failed_models),
            )
            sys.exit(1)
    else:
        model_id = args.model
        if model_id in SUPPORTED_NEURAL_MODELS:
            train_production_neural_model(model_id)
        elif model_id == "lgbm_cluster":
            train_production_model(model_id)
        else:
            logger.error(
                "Unknown model_id '%s'. Expected lgbm_cluster, nbeats, or nhits.",
                model_id,
            )
            sys.exit(1)


if __name__ == "__main__":
    setup_logging()
    main()
