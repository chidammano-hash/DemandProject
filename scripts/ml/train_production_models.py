"""
Production Model Training — trains tree models on full history for production forecasting.

Unlike backtesting (which trains on partial history via expanding windows for evaluation),
this script trains models on ALL available sales history up to the planning date, creating
production-ready artifacts for the forecast generation pipeline.

The key difference:
  - Backtest: trains on partial history (expanding window, last window ~34 months)
  - Production: trains on ALL available history (potentially 60+ months)

This ensures production forecasts leverage the full signal in the data.

Output:
  - data/models/{model_id}/cluster_{label}.pkl — per-cluster model artifacts
  - data/models/{model_id}/training_metadata.json — training run metadata
  - data/models/{model_id}/feature_importance/cluster_{label}.json — per-cluster FI

Usage:
    uv run python scripts/ml/train_production_models.py --model lgbm_cluster
    uv run python scripts/ml/train_production_models.py --model catboost_cluster
    uv run python scripts/ml/train_production_models.py --model xgboost_cluster
    uv run python scripts/ml/train_production_models.py --all
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import pickle
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

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
from common.ml.backtest_framework import (  # noqa: E402
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
from common.ml.seasonal_naive import (  # noqa: E402
    build_seasonal_naive_model,
    make_dfu_key,
)
from common.ml.tuning import load_best_params  # noqa: E402
from common.scripts_base import load_project_env, setup_logging  # noqa: E402
from common.services.perf_profiler import profiled_section  # noqa: E402

logger = logging.getLogger(__name__)

# Validation split: last 20% of calendar months used for early stopping
VAL_SPLIT_PCT = 0.20

# ── Model class/library registry ─────────────────────────────────────────────
# Maps model_name (lgbm, catboost, xgboost) to import paths and metadata.
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
    "catboost": {
        "class": "catboost.CatBoostRegressor",
        "iter_param": "iterations",
        "cat_dtype": "str",
        "needs_cat_dtype_cast": False,
        "constant_target_guard": True,
        "feature_importance_fn": lambda model: model.get_feature_importance(),
        "default_params_fn": lambda algo, seed=42: get_tree_default_params(
            "catboost", algo, seed=seed
        ),
    },
    "xgboost": {
        "class": "xgboost.XGBRegressor",
        "iter_param": "n_estimators",
        "cat_dtype": "category",
        "needs_cat_dtype_cast": True,
        "constant_target_guard": True,
        "feature_importance_fn": lambda model: model.feature_importances_,
        "default_params_fn": lambda algo, seed=42: get_tree_default_params(
            "xgboost", algo, seed=seed
        ),
    },
}

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
    elif model_name == "catboost":
        merged = {**params, "loss_function": "MAE"}
        merged.pop("boost_from_average", None)
        # CatBoost forbids the Newton leaf-estimation method under MAE (no
        # Hessian). Drop the RMSE/Newton-oriented leaf-estimation settings so
        # CatBoost falls back to its valid MAE default (Exact). Without this,
        # intermittent clusters crash with "Newton leaves estimation method is
        # not supported for MAE loss function".
        merged.pop("leaf_estimation_method", None)
        merged.pop("leaf_estimation_iterations", None)
    elif model_name == "xgboost":
        merged = {**params, "objective": "reg:absoluteerror"}
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
    """Extract model library name (lgbm, catboost, xgboost) from model_id.

    model_id examples: lgbm_cluster, catboost_cluster, xgboost_cust_enriched
    """
    for prefix in ("lgbm", "catboost", "xgboost"):
        if model_id.startswith(prefix):
            return prefix
    raise ValueError(
        f"Cannot resolve model library from model_id={model_id!r}. "
        f"Expected prefix: lgbm, catboost, or xgboost."
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

    cat_cols_in_features = (
        [c for c in cat_cols if c in feature_cols] if needs_cat_dtype_cast else []
    )

    # Sort by (startdate, sku_ck) for temporal ordering
    train_c = train_c.sort_values(["startdate", "sku_ck"])

    X_all = train_c[feature_cols].copy() if needs_cat_dtype_cast else train_c[feature_cols]
    y_all = train_c["qty"]

    if needs_cat_dtype_cast:
        for col in cat_cols_in_features:
            X_all[col] = X_all[col].astype("category")

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

    # Calendar-month-based train/val split — last 20% of unique months as validation
    # for early stopping. Full data is still used for the final model, but we need
    # validation to determine the best iteration.
    unique_months = sorted(train_c["startdate"].unique())
    n_val_months = max(1, int(len(unique_months) * VAL_SPLIT_PCT))
    val_months = set(unique_months[-n_val_months:])
    val_mask = train_c["startdate"].isin(val_months)

    X_tr, X_val = X_all.loc[~val_mask], X_all.loc[val_mask]
    y_tr, y_val = y_all.loc[~val_mask], y_all.loc[val_mask]

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

    baseline_intermittent = backtest_cfg.get("baseline_intermittent", True)
    if baseline_intermittent and demand_pattern == "intermittent":
        window = int(backtest_cfg.get("baseline_intermittent_window", 12))
        eval_model = build_seasonal_naive_model(train_c.loc[~val_mask], window=window)
        val_rows = train_c.loc[val_mask]
        val_keys = [
            make_dfu_key(row.item_id, row.customer_group, row.loc)
            for row in val_rows.itertuples(index=False)
        ]
        eval_model._sku_cks = val_keys
        eval_model._months = pd.to_datetime(val_rows["startdate"]).dt.month.tolist()
        val_preds = eval_model.predict(val_rows)
        val_abs_sum = float(abs(y_val.sum()))
        val_floor = max(len(y_val) * 0.01, 1.0)
        val_denom = max(val_abs_sum, val_floor)
        val_wape = round(float((abs(val_preds - y_val.values)).sum() / val_denom * 100), 2)
        model = build_seasonal_naive_model(train_c, window=window)
        meta = {
            "val_wape": val_wape,
            "train_rows": len(X_all),
            "early_stop_train_rows": len(X_tr),
            "total_rows": len(X_all),
            "val_rows": len(X_val),
            "n_estimators_used": 0,
            "cluster_profile": "seasonal_naive_baseline",
            "demand_pattern": demand_pattern,
            "cluster_stats": cluster_stats,
            "n_val_months": n_val_months,
            "n_train_months": len(unique_months) - n_val_months,
        }
        logger.info(
            "Cluster %d/%d '%s': routed to seasonal_naive production artifact "
            "(zero_pct=%.2f, val_wape=%.1f%%)",
            ci,
            n_clusters,
            cluster_label,
            cluster_stats["zero_demand_pct"],
            val_wape,
        )
        return cluster_label, model, meta

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
        cat_cols,
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
        cat_cols,
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


def _save_cluster_artifact(
    out_dir: Path,
    cluster_label: str,
    model: Any,
    feature_cols: list[str],
    model_id: str,
    model_name: str,
    meta: dict[str, Any],
) -> None:
    """Save a single cluster's model artifact as a .pkl file.

    Format matches persist_cluster_models() in run_backtest.py so the
    production forecast pipeline (generate_production_forecasts.py) can
    load artifacts interchangeably regardless of training source.
    """
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

    artifact = {
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
        "n_rows": meta.get("total_rows"),
        "feature_importance": importance_dict,
    }

    file_path = out_dir / f"cluster_{cluster_label}.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(artifact, f)

    # Save feature importance as JSON
    fi_dir = out_dir / "feature_importance"
    fi_dir.mkdir(parents=True, exist_ok=True)
    if importance_dict:
        fi_sorted = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        with open(fi_dir / f"cluster_{cluster_label}.json", "w") as f:
            json.dump(fi_sorted, f, indent=2)


def _save_training_metadata(
    out_dir: Path,
    model_id: str,
    planning_date: str,
    params_source: str,
    cluster_results: dict[str, dict[str, Any]],
    feature_cols_per_cluster: dict[str, list[str]],
    total_rows: int,
    total_dfus: int,
    elapsed_seconds: float,
) -> None:
    """Write training_metadata.json summarizing the production training run."""
    n_clusters_trained = sum(1 for m in cluster_results.values() if not m.get("skipped", False))
    n_clusters_skipped = sum(1 for m in cluster_results.values() if m.get("skipped", False))

    metadata = {
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

    meta_path = out_dir / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Training metadata saved to %s", meta_path)


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

        # Cap sales to planning date
        planning_dt = pd.Timestamp(get_planning_date())
        planning_cutoff = planning_dt.normalize().replace(day=1)
        latest_month = min(sales_df["startdate"].max(), planning_cutoff)
        earliest_month = sales_df["startdate"].min()
        sales_df = sales_df[sales_df["startdate"] <= latest_month].copy()

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

        # Resolve output directory
        base_path = prod_config.get("model_registry", {}).get("base_path", "data/models")
        out_dir = ROOT / base_path / model_id
        out_dir.mkdir(parents=True, exist_ok=True)

        models: dict[str, Any] = {}
        cluster_results: dict[str, dict[str, Any]] = {}
        feature_cols_per_cluster: dict[str, list[str]] = {}
        n_trained = 0

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
            )

            cluster_results[label] = meta
            feature_cols_per_cluster[label] = feature_cols

            if trained_model is not None:
                models[label] = trained_model
                n_trained += 1

                # Save artifact immediately (fail-safe: partial results preserved)
                _save_cluster_artifact(
                    out_dir,
                    label,
                    trained_model,
                    feature_cols,
                    model_id,
                    model_name,
                    meta,
                )

        logger.info(
            "Training complete: %d/%d clusters trained, %d skipped",
            n_trained,
            n_clusters,
            n_clusters - n_trained,
        )

    # ── Step 6: Save metadata ───────────────────────────────────────────────
    elapsed = time.time() - t_start
    _save_training_metadata(
        out_dir=out_dir,
        model_id=model_id,
        planning_date=str(get_planning_date()),
        params_source=params_source,
        cluster_results=cluster_results,
        feature_cols_per_cluster=feature_cols_per_cluster,
        total_rows=len(train_data),
        total_dfus=n_dfus,
        elapsed_seconds=elapsed,
    )

    logger.info(
        "Production training for %s complete: %d models saved to %s/ (%.0fs / %.1fm)",
        model_id,
        n_trained,
        out_dir,
        elapsed,
        elapsed / 60,
    )


# ── CLI entry point ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train tree models on FULL history for production forecasting. "
            "Unlike backtesting (partial history), this creates artifacts "
            "trained on all available data up to the planning date."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID to train (e.g. lgbm_cluster, catboost_cluster, xgboost_cust_enriched)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        dest="train_all",
        help="Train all forecastable tree models from forecast_pipeline_config.yaml",
    )
    args = parser.parse_args()

    load_project_env()

    if args.train_all:
        # Get all enabled tree models with forecast=true
        roster = get_algorithm_roster(stage="forecast")
        tree_models = [mid for mid, entry in roster.items() if entry.get("type") == "tree"]
        if not tree_models:
            logger.error("No forecastable tree models found in forecast_pipeline_config.yaml")
            sys.exit(1)

        logger.info("Training %d forecastable tree models: %s", len(tree_models), tree_models)
        failed_models: list[str] = []
        for model_id in sorted(tree_models):
            try:
                train_production_model(model_id)
            except Exception:
                logger.exception("Failed to train %s", model_id)
                failed_models.append(model_id)
                # Continue with remaining models so one bad family does not
                # mask additional missing artifacts in the same production run.
        if failed_models:
            logger.error(
                "Production training failed for %d/%d tree model(s): %s",
                len(failed_models),
                len(tree_models),
                ", ".join(failed_models),
            )
            sys.exit(1)
    else:
        model_id = args.model
        # Validate model_id has a known model library prefix
        try:
            _resolve_model_name(model_id)
        except ValueError:
            logger.error(
                "Unknown model_id '%s'. Must start with lgbm, catboost, or xgboost.",
                model_id,
            )
            sys.exit(1)

        train_production_model(model_id)


if __name__ == "__main__":
    setup_logging()
    main()
