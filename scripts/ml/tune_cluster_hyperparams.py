"""
Per-cluster hyperparameter tuning using Optuna.

Tunes LGBM/CatBoost/XGBoost independently per ml_cluster, then writes
best params into config/forecasting/cluster_tuning_profiles.yaml with cluster_name-based
matching.

Usage:
    uv run python scripts/tune_cluster_hyperparams.py --model lgbm
    uv run python scripts/tune_cluster_hyperparams.py --model lgbm --trials 30
    uv run python scripts/tune_cluster_hyperparams.py --model lgbm --clusters L2_1 L2_3
"""

# ruff: noqa: E402

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from common.core.constants import CAT_FEATURES, LAG_RANGE
from common.core.db import get_db_params
from common.core.utils import load_forecast_pipeline_config
from common.ml.backtest_framework import load_backtest_data
from common.ml.feature_engineering import (
    build_feature_matrix,
    get_feature_columns,
    mask_future_sales,
)
from common.ml.tuning import (
    TRAIN_FOLD_FNS,
    best_rounds_to_n_estimators,
    compute_wape_stabilised,
    generate_cv_month_splits,
    iteration_param_for_model,
    merge_fixed_params,
    suggest_model_params,
    trial_best_rounds_or_max,
)
from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)


# ── Objective factory ────────────────────────────────────────────────────────


def make_cluster_objective(
    model_name: str,
    cluster_grid: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    month_splits: list[tuple[list, list]],
    config: dict,
) -> Any:
    """Return an Optuna objective function closure for a single cluster's data."""
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
            masked_grid = mask_future_sales(cluster_grid, train_end)

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
                logger.warning("Fold %d training failed: %s", fold_idx + 1, exc)
                return float("inf")

            wape = compute_wape_stabilised(preds, y_val)
            if not np.isinf(wape) and not np.isnan(wape):
                fold_wapes.append(wape)
                fold_best_rounds.append(best_rounds)

            # Report intermediate value for Optuna pruner
            trial.report(
                float(np.mean(fold_wapes)) if fold_wapes else float("inf"),
                step=fold_idx,
            )
            if trial.should_prune():
                raise optuna.TrialPruned()

        if not fold_wapes:
            return float("inf")

        trial.set_user_attr("best_n_estimators", int(np.mean(fold_best_rounds)))
        trial.set_user_attr("fold_wapes", [round(w, 6) for w in fold_wapes])
        return float(np.mean(fold_wapes))

    return objective


# ── Staleness state (cluster_tuning_profile_state, sql/148) ─────────────────


def fetch_stale_clusters(db: dict[str, Any]) -> list[str]:
    """Cluster names flagged stale by a cluster promotion, [] if table absent."""
    try:
        with psycopg.connect(**db) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT cluster_name FROM cluster_tuning_profile_state "
                "WHERE stale = TRUE ORDER BY cluster_name"
            )
            return [row[0] for row in cur.fetchall()]
    except psycopg.Error as exc:
        logger.warning(
            "cluster_tuning_profile_state unavailable (%s) — treating as no stale rows", exc
        )
        return []


def clear_stale_flags(db: dict[str, Any], cluster_names: list[str]) -> None:
    """Mark the given clusters' tuning profiles fresh (after a successful tune)."""
    if not cluster_names:
        return
    try:
        with psycopg.connect(**db) as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE cluster_tuning_profile_state
                   SET stale = FALSE, cleared_at = NOW(), modified_ts = NOW()
                   WHERE cluster_name = ANY(%s)""",
                (cluster_names,),
            )
            conn.commit()
            logger.info("Cleared stale flag on %d tuning profile(s)", cur.rowcount)
    except psycopg.Error as exc:
        logger.warning("Could not clear stale flags (%s)", exc)


def fetch_promoted_cluster_experiment(db: dict[str, Any]) -> tuple[int | None, str | None]:
    """(experiment_id, scenario_id) of the promoted cluster experiment, if any."""
    try:
        with psycopg.connect(**db) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT experiment_id, scenario_id FROM cluster_experiment "
                "WHERE is_promoted ORDER BY promoted_at DESC LIMIT 1"
            )
            row = cur.fetchone()
    except psycopg.Error as exc:
        logger.warning("cluster_experiment lookup failed (%s)", exc)
        return None, None
    if row is None:
        return None, None
    return int(row[0]), str(row[1])


# ── Output writer ────────────────────────────────────────────────────────────


def write_cluster_profiles(
    output_path: Path,
    model_name: str,
    results: dict[str, dict[str, Any]],
    min_cluster_size: int,
    metadata: dict[str, Any] | None = None,
    existing_profiles: dict[str, Any] | None = None,
) -> None:
    """Write per-cluster tuning results to a YAML profile file.

    Backs up existing file to .yaml.bak before overwriting. When
    ``existing_profiles`` is given (subset runs: --clusters / --stale-only),
    untouched cluster entries are preserved instead of being wiped —
    a partial re-tune must never destroy the other clusters' profiles.
    ``metadata`` stamps the generation (cluster_experiment_id) so backtests
    can detect profiles tuned against superseded cluster memberships.
    """
    if output_path.exists():
        bak_path = output_path.with_suffix(".yaml.bak")
        shutil.copy2(output_path, bak_path)
        logger.info("Backed up existing profile to %s", bak_path)

    profiles: dict[str, Any] = {
        name: profile
        for name, profile in (existing_profiles or {}).items()
        if name != "default"
    }
    skipped_nonfinite = 0
    for cluster_name, result in sorted(results.items()):
        # Skip clusters whose tuning produced a non-finite WAPE (e.g. every CV
        # fold failed or was empty). best_params there were chosen by a degenerate
        # objective; enshrining them would apply untrusted hyperparameters the
        # moment the profile file is enabled.
        if not np.isfinite(result["best_wape"]):
            logger.warning(
                "Cluster '%s' tuning produced non-finite WAPE — skipping profile",
                cluster_name,
            )
            skipped_nonfinite += 1
            continue
        n_rows = result["n_rows"]
        best_wape_pct = round(result["best_wape"] * 100, 2)
        profiles[cluster_name] = {
            "description": f"Auto-tuned ({n_rows} rows, WAPE={best_wape_pct}%)",
            "match_criteria": {"cluster_name": cluster_name},
            "overrides": result["best_params"],
        }
    if skipped_nonfinite:
        logger.warning(
            "%d cluster(s) skipped due to non-finite tuning WAPE — they fall back "
            "to global params. Investigate empty CV folds before relying on "
            "per-cluster profiles.",
            skipped_nonfinite,
        )

    # Default fallback entry
    profiles["default"] = {
        "description": "Fallback -- uses global params from forecast_pipeline_config.yaml",
        "match_criteria": {},
        "overrides": {},
    }

    doc: dict[str, Any] = {
        "metadata": {
            "model": model_name,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **(metadata or {}),
        },
        "cluster_profiles": profiles,
        "enabled": True,
        "min_cluster_size": min_cluster_size,
    }

    header = (
        f"# Auto-generated by scripts/ml/tune_cluster_hyperparams.py\n"
        f"# Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        f"# Model: {model_name}\n\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(header)
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)

    logger.info("Wrote cluster profiles to %s", output_path)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-cluster hyperparameter tuning using Optuna")
    parser.add_argument(
        "--model",
        choices=["lgbm", "catboost", "xgboost"],
        required=True,
        help="Model to tune",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Optuna trials per cluster (default: 30)",
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        default=None,
        help="Optional: only tune these clusters (e.g. --clusters L2_1 L2_3)",
    )
    parser.add_argument(
        "--stale-only",
        action="store_true",
        help="Only tune clusters flagged stale in cluster_tuning_profile_state "
             "(set by cluster promotion); exits 0 when nothing is stale",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=500,
        help="Minimum rows per cluster to attempt tuning (default: 500)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "config" / "forecasting" / "hyperparameter_tuning.yaml"),
        help="Path to hyperparameter tuning YAML config",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    load_dotenv(ROOT / ".env")

    # ── Load config ──────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Overlay pipeline-level tuning params from forecast_pipeline_config.yaml
    try:
        pipeline_cfg = load_forecast_pipeline_config()
        pipeline_tuning = pipeline_cfg.get("tuning", {})
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
        pass  # No pipeline config -- use hyperparameter_tuning.yaml as-is

    t_cfg = config["tuning"]
    model_name = args.model
    n_trials = args.trials
    min_rows = args.min_rows

    # CatBoost uses str dtype for categoricals; LGBM and XGBoost use "category"
    cat_dtype = "str" if model_name == "catboost" else "category"

    logger.info(
        "Per-cluster tuning: %s | %d trials/cluster | min_rows=%d",
        model_name.upper(),
        n_trials,
        min_rows,
    )

    # ── Resolve stale-only scope BEFORE the expensive data load ─────────────
    db = get_db_params()
    stale_requested: list[str] | None = None
    if args.stale_only:
        stale_requested = fetch_stale_clusters(db)
        if not stale_requested:
            logger.info("No tuning profiles are flagged stale — nothing to do.")
            return
        logger.info("Stale-only mode: %s", stale_requested)
        if args.clusters:
            stale_requested = [c for c in stale_requested if c in set(args.clusters)]

    # ── Load data ────────────────────────────────────────────────────────────
    with profiled_section("load_data"):
        logger.info("Loading data from Postgres...")
        sales_df, dfu_attrs, item_attrs = load_backtest_data(db)

    # ── Build feature matrix ─────────────────────────────────────────────────
    with profiled_section("build_feature_matrix"):
        all_months = sorted(sales_df["startdate"].unique())
        logger.info("Building feature matrix (%d months)...", len(all_months))
        full_grid = build_feature_matrix(
            sales_df,
            dfu_attrs,
            item_attrs,
            all_months,
            cat_dtype=cat_dtype,
        )
        feature_cols = get_feature_columns(full_grid)
        cat_cols = [c for c in CAT_FEATURES if c in feature_cols and c in full_grid.columns]
        logger.info(
            "Features: %d total, %d categorical",
            len(feature_cols),
            len(cat_cols),
        )

    # ── Discover clusters ────────────────────────────────────────────────────
    if "ml_cluster" not in full_grid.columns:
        logger.error(
            "ml_cluster column not found in feature matrix. "
            "Run clustering pipeline first (make cluster-all)."
        )
        sys.exit(1)

    all_clusters = sorted(
        full_grid["ml_cluster"].dropna().loc[lambda s: s != "__unknown__"].unique().tolist()
    )

    requested_subset = stale_requested if stale_requested is not None else args.clusters
    if requested_subset:
        requested = set(requested_subset)
        missing = requested - set(all_clusters)
        if missing:
            logger.warning("Clusters not found in data: %s", missing)
        all_clusters = [c for c in all_clusters if c in requested]

    if not all_clusters:
        logger.error(
            "No clusters to tune. Available clusters: %s",
            full_grid["ml_cluster"].dropna().unique().tolist(),
        )
        sys.exit(1)

    logger.info("Clusters to tune: %s", all_clusters)

    # ── Per-cluster tuning ───────────────────────────────────────────────────
    cluster_results: dict[str, dict[str, Any]] = {}
    total = len(all_clusters)

    for idx, cluster_name in enumerate(all_clusters, 1):
        logger.info("=" * 60)

        with profiled_section(f"tune_cluster_{cluster_name}"):
            cluster_mask = full_grid["ml_cluster"] == cluster_name
            cluster_grid = full_grid[cluster_mask].copy()
            n_rows = len(cluster_grid)

            if n_rows < min_rows:
                logger.warning(
                    "Skipping cluster %s (%d/%d): %d rows < min_rows=%d",
                    cluster_name,
                    idx,
                    total,
                    n_rows,
                    min_rows,
                )
                continue

            cluster_months = sorted(cluster_grid["startdate"].unique())
            n_months = len(cluster_months)

            logger.info(
                "Tuning cluster %s (%d/%d): %d rows, %d months...",
                cluster_name,
                idx,
                total,
                n_rows,
                n_months,
            )

            # Generate CV splits for this cluster's months
            month_splits = generate_cv_month_splits(
                cluster_months,
                n_splits=t_cfg["n_splits"],
                gap_months=t_cfg["gap_months"],
                min_train_months=t_cfg["min_train_months"],
                val_months_per_fold=t_cfg["val_months_per_fold"],
            )

            if not month_splits:
                logger.warning(
                    "Skipping cluster %s: insufficient months for CV splits "
                    "(%d months, need >= %d + gap + val)",
                    cluster_name,
                    n_months,
                    t_cfg["min_train_months"],
                )
                continue

            logger.info(
                "  CV splits: %d folds",
                len(month_splits),
            )
            for i, (tm, vm) in enumerate(month_splits):
                logger.info(
                    "    Fold %d: train [%s -> %s] (%d months), "
                    "gap %dm, val [%s -> %s] (%d months)",
                    i + 1,
                    min(tm).date(),
                    max(tm).date(),
                    len(tm),
                    t_cfg["gap_months"],
                    min(vm).date(),
                    max(vm).date(),
                    len(vm),
                )

            # Create Optuna study with deterministic but cluster-unique seed
            seed = 42 + abs(hash(cluster_name)) % (2**31)
            sampler = optuna.samplers.TPESampler(seed=seed)
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=t_cfg["pruner_n_startup_trials"],
                n_warmup_steps=t_cfg["pruner_n_warmup_steps"],
            )
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
            )

            # Build objective for this cluster
            objective_fn = make_cluster_objective(
                model_name=model_name,
                cluster_grid=cluster_grid,
                feature_cols=feature_cols,
                cat_cols=cat_cols,
                month_splits=month_splits,
                config=config,
            )

            # Trial progress callback
            def _trial_callback(
                study: optuna.Study,
                trial: optuna.trial.FrozenTrial,
                _cname: str = cluster_name,
            ) -> None:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    completed = len(
                        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    )
                    best_wape_pct = (
                        study.best_value * 100 if study.best_value < float("inf") else float("inf")
                    )
                    logger.info(
                        "  [%s] Trial %3d | WAPE=%.2f%% | best=%.2f%% | %d/%d",
                        _cname,
                        trial.number,
                        trial.value * 100,
                        best_wape_pct,
                        completed,
                        n_trials,
                    )

            # Optimise
            try:
                study.optimize(
                    objective_fn,
                    n_trials=n_trials,
                    callbacks=[_trial_callback],
                    show_progress_bar=False,
                )
            except Exception:
                logger.exception(
                    "Tuning failed for cluster %s -- skipping",
                    cluster_name,
                )
                continue

            # Collect results
            completed_trials = [
                t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
            ]
            if not completed_trials:
                logger.warning(
                    "No completed trials for cluster %s -- skipping",
                    cluster_name,
                )
                continue

            best_trial = study.best_trial
            best_params = merge_fixed_params(model_name, config, best_trial.params)
            best_n_est_raw = trial_best_rounds_or_max(best_trial, t_cfg["n_estimators_max"])
            best_n_estimators = best_rounds_to_n_estimators(
                [best_n_est_raw],
                buffer=t_cfg.get("n_estimators_buffer", 1.1),
            )

            iter_param = iteration_param_for_model(model_name)
            best_params[iter_param] = best_n_estimators

            cluster_results[cluster_name] = {
                "best_wape": best_trial.value,
                "best_params": best_params,
                "n_rows": n_rows,
                "n_months": n_months,
                "n_completed_trials": len(completed_trials),
                "fold_wapes": best_trial.user_attrs.get("fold_wapes", []),
            }

            logger.info(
                "  Cluster %s best: WAPE=%.4f%%, n_estimators=%d",
                cluster_name,
                best_trial.value * 100,
                best_n_estimators,
            )

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TUNING SUMMARY")
    logger.info("=" * 60)

    if not cluster_results:
        logger.error("No clusters were successfully tuned.")
        sys.exit(1)

    for cname, result in sorted(cluster_results.items(), key=lambda x: x[1]["best_wape"]):
        logger.info(
            "  %-20s WAPE=%6.2f%%  (%d rows, %d trials)",
            cname,
            result["best_wape"] * 100,
            result["n_rows"],
            result["n_completed_trials"],
        )

    # ── Write cluster profiles YAML ──────────────────────────────────────────
    output_path = ROOT / "config" / "forecasting" / "cluster_tuning_profiles.yaml"

    # Subset runs (--clusters / --stale-only) merge into the existing file —
    # a partial re-tune must not wipe the untouched clusters' profiles.
    existing_profiles: dict[str, Any] | None = None
    if requested_subset and output_path.exists():
        with open(output_path) as f:
            existing_doc = yaml.safe_load(f) or {}
        existing_profiles = existing_doc.get("cluster_profiles") or {}

    # Stamp the cluster generation so backtests can detect profiles tuned
    # against a superseded cluster membership (warn_if_profiles_stale).
    experiment_id, scenario_id = fetch_promoted_cluster_experiment(db)
    metadata = {
        "cluster_experiment_id": experiment_id,
        "cluster_scenario_id": scenario_id,
    }

    with profiled_section("write_profiles"):
        write_cluster_profiles(
            output_path=output_path,
            model_name=model_name,
            results=cluster_results,
            min_cluster_size=min_rows,
            metadata=metadata,
            existing_profiles=existing_profiles,
        )

    # Close the staleness loop: the tuned clusters' flags are cleared so the
    # promotion → stale → re-tune cycle converges instead of accumulating.
    clear_stale_flags(db, sorted(cluster_results))

    logger.info("Per-cluster tuning complete. Profiles saved to %s", output_path)


if __name__ == "__main__":
    main()
