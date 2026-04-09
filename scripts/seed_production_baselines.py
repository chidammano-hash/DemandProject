#!/usr/bin/env python3
"""Seed production pipeline results as promoted baseline experiments.

Reads production artifacts (backtest metadata, champion summary, clustering
metadata) and inserts them as promoted experiments in the experimentation
tables so the UI always has a baseline to compare against.

Usage:
    python scripts/seed_production_baselines.py                    # All scopes
    python scripts/seed_production_baselines.py --scope tuning     # Model tuning only
    python scripts/seed_production_baselines.py --scope champion   # Champion only
    python scripts/seed_production_baselines.py --scope clustering # Clustering only
    python scripts/seed_production_baselines.py --model lgbm       # Single model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params
from common.utils import load_config, load_forecast_pipeline_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_LABEL = "Production Baseline"
TEMPLATE_ID = "production_baseline"

MODEL_IDS = ["lgbm_cluster", "catboost_cluster", "xgboost_cluster", "chronos"]

# Keys in pipeline config algorithm entries that are training config, not hyperparameters
_TRAINING_KEYS = frozenset({
    "enabled", "model_id", "cluster_strategy", "recursive",
    "shap_select", "shap_threshold", "shap_top_n",
    "shap_sample_size", "tune_inline", "params_file",
})


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def _compute_accuracy(df: pd.DataFrame) -> dict[str, Any]:
    """Compute WAPE, bias, accuracy_pct from forecast/actual columns."""
    forecast_sum = df["basefcst_pref"].sum()
    actual_sum = df["tothist_dmd"].sum()
    abs_error = (df["basefcst_pref"] - df["tothist_dmd"]).abs().sum()
    n_preds = len(df)
    n_skus = df[["item_id", "loc"]].drop_duplicates().shape[0]
    if abs(actual_sum) < 1e-9:
        return {"n_predictions": n_preds, "n_skus": n_skus,
                "accuracy_pct": None, "wape": None, "bias": None}
    wape = 100.0 * abs_error / abs(actual_sum)
    bias = (forecast_sum / actual_sum) - 1.0
    return {
        "n_predictions": n_preds,
        "n_skus": n_skus,
        "accuracy_pct": round(100.0 - wape, 2),
        "wape": round(wape, 2),
        "bias": round(bias, 4),
    }


# ---------------------------------------------------------------------------
# Feature collection
# ---------------------------------------------------------------------------

def _collect_model_features(model_id: str) -> list[str]:
    """Collect union of features from model feature_importance JSONs."""
    fi_dir = ROOT / f"data/models/{model_id}/feature_importance"
    if not fi_dir.exists():
        logger.warning("Feature importance dir not found: %s", fi_dir)
        return []
    all_features: set[str] = set()
    for f in fi_dir.glob("cluster_*.json"):
        try:
            data = json.loads(f.read_text())
            all_features.update(data.keys())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s", f, exc)
    return sorted(all_features)


# ---------------------------------------------------------------------------
# Model Tuning seeding
# ---------------------------------------------------------------------------

def _extract_params(algo_section: dict[str, Any]) -> dict[str, Any]:
    """Extract hyperparameters (excluding training config keys)."""
    return {k: v for k, v in algo_section.items() if k not in _TRAINING_KEYS}


def _extract_training_config(algo_section: dict[str, Any]) -> dict[str, Any]:
    """Extract training config keys."""
    return {k: v for k, v in algo_section.items() if k in _TRAINING_KEYS}


def _insert_timeframe_rows(
    cur: psycopg.Cursor,  # type: ignore[type-arg]
    run_id: int,
    metadata: dict[str, Any],
    all_lags_path: Path,
) -> int:
    """Compute and insert per-timeframe accuracy rows from all-lags CSV."""
    timeframes = metadata.get("timeframes", [])
    if not timeframes:
        return 0

    # Build timeframe date map from metadata
    tf_map = {tf["label"]: tf for tf in timeframes}

    # Try to compute per-timeframe metrics from all-lags CSV
    tf_metrics: dict[str, dict[str, Any]] = {}
    if all_lags_path.exists():
        try:
            df = pd.read_csv(all_lags_path, usecols=[
                "item_id", "loc", "startdate", "lag", "execution_lag",
                "basefcst_pref", "tothist_dmd", "timeframe",
            ])
            df = df.dropna(subset=["basefcst_pref", "tothist_dmd"])
            # Filter to execution-lag rows only
            df["lag"] = pd.to_numeric(df["lag"], errors="coerce")
            df["execution_lag"] = pd.to_numeric(df["execution_lag"], errors="coerce")
            exec_df = df[df["lag"] == df["execution_lag"]].copy()
            for tf_label, grp in exec_df.groupby("timeframe"):
                tf_metrics[str(tf_label)] = _compute_accuracy(grp)
        except (ValueError, KeyError) as exc:
            logger.warning("Could not compute timeframe metrics: %s", exc)

    count = 0
    for tf in timeframes:
        label = tf["label"]
        metrics = tf_metrics.get(label, {})
        cur.execute(
            """
            INSERT INTO lgbm_tuning_timeframe
                (run_id, timeframe, train_end, predict_start, predict_end,
                 n_predictions, accuracy_pct, wape, bias)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, timeframe) DO UPDATE SET
                train_end     = EXCLUDED.train_end,
                predict_start = EXCLUDED.predict_start,
                predict_end   = EXCLUDED.predict_end,
                n_predictions = EXCLUDED.n_predictions,
                accuracy_pct  = EXCLUDED.accuracy_pct,
                wape          = EXCLUDED.wape,
                bias          = EXCLUDED.bias
            """,
            (
                run_id, label,
                tf.get("train_end"), tf.get("predict_start"), tf.get("predict_end"),
                metrics.get("n_predictions"), metrics.get("accuracy_pct"),
                metrics.get("wape"), metrics.get("bias"),
            ),
        )
        count += 1
    return count


def _insert_lag_rows(
    cur: psycopg.Cursor,  # type: ignore[type-arg]
    run_id: int,
    all_lags_path: Path,
) -> int:
    """Compute and insert per-lag accuracy rows from all-lags CSV."""
    if not all_lags_path.exists():
        logger.warning("All-lags CSV not found: %s — skipping lag breakdowns", all_lags_path)
        return 0

    try:
        df = pd.read_csv(all_lags_path, usecols=[
            "item_id", "loc", "lag", "basefcst_pref", "tothist_dmd",
        ])
        df = df.dropna(subset=["basefcst_pref", "tothist_dmd"])
        df["lag"] = pd.to_numeric(df["lag"], errors="coerce")
    except (ValueError, KeyError) as exc:
        logger.warning("Could not read all-lags CSV: %s", exc)
        return 0

    count = 0
    for lag_val, grp in df.groupby("lag"):
        m = _compute_accuracy(grp)
        cur.execute(
            """
            INSERT INTO lgbm_tuning_lag
                (run_id, exec_lag, n_predictions, n_dfus, accuracy_pct, wape, bias)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, exec_lag) DO UPDATE SET
                n_predictions = EXCLUDED.n_predictions,
                n_dfus        = EXCLUDED.n_dfus,
                accuracy_pct  = EXCLUDED.accuracy_pct,
                wape          = EXCLUDED.wape,
                bias          = EXCLUDED.bias
            """,
            (run_id, int(lag_val), m["n_predictions"], m["n_skus"],
             m["accuracy_pct"], m["wape"], m["bias"]),
        )
        count += 1
    return count


def _insert_cluster_month_rows(
    cur: psycopg.Cursor,  # type: ignore[type-arg]
    run_id: int,
    pred_path: Path,
    db_params: dict[str, Any],
) -> tuple[int, int]:
    """Compute and insert per-cluster and per-month breakdowns."""
    if not pred_path.exists():
        logger.warning("Predictions CSV not found: %s — skipping breakdowns", pred_path)
        return 0, 0

    df = pd.read_csv(pred_path, usecols=[
        "item_id", "loc", "startdate", "basefcst_pref", "tothist_dmd",
    ])
    df = df.dropna(subset=["basefcst_pref", "tothist_dmd"])
    if df.empty:
        return 0, 0

    # Fetch cluster mappings from dim_sku
    with psycopg.connect(**db_params) as sku_conn:
        with sku_conn.cursor() as sku_cur:
            sku_cur.execute("SELECT item_id, loc, ml_cluster, cluster_assignment FROM dim_sku")
            sku_rows = sku_cur.fetchall()

    sku_df = pd.DataFrame(sku_rows, columns=["item_id", "loc", "ml_cluster", "cluster_assignment"])
    for col in ("item_id", "loc"):
        df[col] = df[col].astype(str)
        sku_df[col] = sku_df[col].astype(str)
    df = df.merge(sku_df, on=["item_id", "loc"], how="left")
    df["ml_cluster"] = df["ml_cluster"].fillna("unknown")
    df["cluster_assignment"] = df["cluster_assignment"].fillna("unknown")

    # Per-cluster
    cluster_count = 0
    for cluster_type, col in [("ml_cluster", "ml_cluster"), ("business_cluster", "cluster_assignment")]:
        for cluster_val, grp in df.groupby(col):
            m = _compute_accuracy(grp)
            cur.execute(
                """
                INSERT INTO lgbm_tuning_cluster
                    (run_id, cluster_type, cluster_value, n_predictions, n_dfus,
                     accuracy_pct, wape, bias)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, cluster_type, cluster_value) DO UPDATE SET
                    n_predictions = EXCLUDED.n_predictions,
                    n_dfus        = EXCLUDED.n_dfus,
                    accuracy_pct  = EXCLUDED.accuracy_pct,
                    wape          = EXCLUDED.wape,
                    bias          = EXCLUDED.bias
                """,
                (run_id, cluster_type, str(cluster_val),
                 m["n_predictions"], m["n_skus"],
                 m["accuracy_pct"], m["wape"], m["bias"]),
            )
            cluster_count += 1

    # Per-month
    month_count = 0
    for month_val, grp in df.groupby("startdate"):
        m = _compute_accuracy(grp)
        cur.execute(
            """
            INSERT INTO lgbm_tuning_month
                (run_id, month_start, n_predictions, n_dfus,
                 accuracy_pct, wape, bias)
            VALUES (%s, %s::date, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, month_start) DO UPDATE SET
                n_predictions = EXCLUDED.n_predictions,
                n_dfus        = EXCLUDED.n_dfus,
                accuracy_pct  = EXCLUDED.accuracy_pct,
                wape          = EXCLUDED.wape,
                bias          = EXCLUDED.bias
            """,
            (run_id, str(month_val),
             m["n_predictions"], m["n_skus"],
             m["accuracy_pct"], m["wape"], m["bias"]),
        )
        month_count += 1

    return cluster_count, month_count


def seed_tuning_baseline(model_id: str, conn: psycopg.Connection) -> int | None:  # type: ignore[type-arg]
    """Seed a production backtest as a promoted tuning experiment.

    Returns the inserted run_id, or None if artifacts are missing.
    """
    meta_path = ROOT / f"data/backtest/{model_id}/backtest_metadata.json"
    pred_path = ROOT / f"data/backtest/{model_id}/backtest_predictions.csv"
    all_lags_path = ROOT / f"data/backtest/{model_id}/backtest_predictions_all_lags.csv"

    if not meta_path.exists():
        logger.warning("Skipping %s: metadata not found at %s", model_id, meta_path)
        return None

    meta = json.loads(meta_path.read_text())
    acc = meta.get("accuracy_at_execution_lag", {})

    # Read production params from forecast_pipeline_config.yaml
    from common.utils import get_algorithm_params
    pcfg = load_forecast_pipeline_config()
    algo_entry = pcfg.get("algorithms", {}).get(model_id, {})
    algo_section = algo_entry.get("params", algo_entry)
    params = _extract_params(algo_section)
    training = _extract_training_config(algo_section)

    features = _collect_model_features(model_id)
    db_params = get_db_params()

    with conn.cursor() as cur:
        # Remove any existing production baseline for this model (CASCADE deletes children)
        cur.execute(
            "DELETE FROM lgbm_tuning_run WHERE run_label = %s AND model_id = %s",
            (BASELINE_LABEL, model_id),
        )

        # Clear promoted flag on any other run for this model
        cur.execute(
            "UPDATE lgbm_tuning_run SET is_promoted = FALSE, promoted_at = NULL "
            "WHERE model_id = %s AND is_promoted = TRUE",
            (model_id,),
        )

        # Build metadata config section for storage
        meta_with_config = dict(meta)
        meta_with_config["training_config"] = training

        # Insert the baseline run
        cur.execute(
            """
            INSERT INTO lgbm_tuning_run
                (run_label, model_id, status, started_at, completed_at,
                 params, feature_count, features,
                 accuracy_pct, wape, bias, n_predictions, n_dfus,
                 metadata, template_id, notes,
                 is_promoted, promoted_at,
                 is_results_promoted, results_promoted_at,
                 cluster_source)
            VALUES
                (%s, %s, 'completed', NOW(), NOW(),
                 %s, %s, %s,
                 %s, %s, %s, %s, %s,
                 %s, %s, %s,
                 TRUE, NOW(),
                 TRUE, NOW(),
                 'production')
            RETURNING run_id
            """,
            (
                BASELINE_LABEL, model_id,
                json.dumps(params), len(features) if features else 0, json.dumps(features),
                acc.get("accuracy_pct"), acc.get("wape"), acc.get("bias"),
                meta.get("n_predictions"), meta.get("n_dfus"),
                json.dumps(meta_with_config), TEMPLATE_ID,
                f"Auto-seeded from production backtest artifacts at {datetime.now(timezone.utc).isoformat()}",
            ),
        )
        row = cur.fetchone()
        if row is None:
            raise psycopg.Error("INSERT did not return a run_id")
        run_id: int = row[0]
        logger.info("Inserted tuning baseline run_id=%d for %s (accuracy=%.2f%%)",
                     run_id, model_id, acc.get("accuracy_pct", 0))

        # Insert breakdowns
        tf_count = _insert_timeframe_rows(cur, run_id, meta, all_lags_path)
        logger.info("  Timeframes: %d rows", tf_count)

        lag_count = _insert_lag_rows(cur, run_id, all_lags_path)
        logger.info("  Lags: %d rows", lag_count)

        cluster_count, month_count = _insert_cluster_month_rows(
            cur, run_id, pred_path, db_params,
        )
        logger.info("  Clusters: %d rows, Months: %d rows", cluster_count, month_count)

    conn.commit()
    return run_id


# ---------------------------------------------------------------------------
# Champion seeding
# ---------------------------------------------------------------------------

def seed_champion_baseline(conn: psycopg.Connection) -> int | None:  # type: ignore[type-arg]
    """Seed the production champion selection as a promoted champion experiment.

    Returns the inserted experiment_id, or None if artifacts are missing.
    """
    summary_path = ROOT / "data/champion/champion_summary.json"
    if not summary_path.exists():
        logger.warning("Skipping champion: summary not found at %s", summary_path)
        return None

    summary = json.loads(summary_path.read_text())
    pipeline_cfg = load_forecast_pipeline_config()
    comp_config = pipeline_cfg.get("champion", {})

    champion_acc = summary.get("overall_champion_accuracy_pct")
    ceiling_acc = summary.get("overall_ceiling_accuracy_pct")
    gap_bps = None
    if champion_acc is not None and ceiling_acc is not None:
        gap_bps = round((ceiling_acc - champion_acc) * 100, 2)

    model_distribution = summary.get("model_wins", {})
    total_skus = summary.get("total_dfus")
    total_sku_months = summary.get("total_dfu_months")

    strategy = comp_config.get("strategy", "expanding")
    strategy_params = comp_config.get("strategy_params", {})
    meta_learner_params = comp_config.get("meta_learner")
    models = comp_config.get("models", MODEL_IDS)
    metric = comp_config.get("metric", "accuracy_pct")
    lag_mode = comp_config.get("lag", "execution")
    min_sku_rows = comp_config.get("min_sku_rows", 3)

    with conn.cursor() as cur:
        # Remove existing baseline
        cur.execute(
            "DELETE FROM champion_experiment WHERE label = %s AND template_id = %s",
            (BASELINE_LABEL, TEMPLATE_ID),
        )

        # Clear promoted flag
        cur.execute(
            "UPDATE champion_experiment SET is_promoted = FALSE, promoted_at = NULL "
            "WHERE is_promoted = TRUE",
        )

        cur.execute(
            """
            INSERT INTO champion_experiment
                (label, notes, template_id, status, started_at, completed_at,
                 strategy, strategy_params, meta_learner_params,
                 models, metric, lag_mode, min_sku_rows,
                 champion_accuracy, ceiling_accuracy, gap_bps,
                 n_champions, n_dfu_months, model_distribution,
                 is_promoted, promoted_at,
                 is_results_promoted, results_promoted_at)
            VALUES
                (%s, %s, %s, 'completed', NOW(), NOW(),
                 %s, %s, %s,
                 %s, %s, %s, %s,
                 %s, %s, %s,
                 %s, %s, %s,
                 TRUE, NOW(),
                 TRUE, NOW())
            RETURNING experiment_id
            """,
            (
                BASELINE_LABEL,
                f"Auto-seeded from production champion selection at {datetime.now(timezone.utc).isoformat()}",
                TEMPLATE_ID,
                strategy, json.dumps(strategy_params),
                json.dumps(meta_learner_params) if meta_learner_params else None,
                json.dumps(models), metric, lag_mode, min_sku_rows,
                champion_acc, ceiling_acc, gap_bps,
                total_skus, total_sku_months, json.dumps(model_distribution),
            ),
        )
        row = cur.fetchone()
        if row is None:
            raise psycopg.Error("INSERT did not return an experiment_id")
        experiment_id: int = row[0]
        logger.info(
            "Inserted champion baseline experiment_id=%d (accuracy=%.2f%%, gap=%.0f bps)",
            experiment_id, champion_acc or 0, gap_bps or 0,
        )

        # Insert per-lag rows from DB (champion predictions in fact table)
        lag_count = _insert_champion_lag_rows(cur, experiment_id)
        logger.info("  Champion lags: %d rows", lag_count)

        # Insert per-month rows from DB
        month_count = _insert_champion_month_rows(cur, experiment_id)
        logger.info("  Champion months: %d rows", month_count)

    conn.commit()
    return experiment_id


def _insert_champion_lag_rows(
    cur: psycopg.Cursor,  # type: ignore[type-arg]
    experiment_id: int,
) -> int:
    """Compute per-lag champion accuracy from fact_external_forecast_monthly."""
    cur.execute(
        """
        SELECT f.lag,
               100.0 - (100.0 * SUM(ABS(f.basefcst_pref - f.tothist_dmd)) / NULLIF(ABS(SUM(f.tothist_dmd)), 0))
                   AS champion_accuracy,
               COUNT(*) AS n_dfu_months
        FROM fact_external_forecast_monthly f
        WHERE f.model_id = 'champion'
          AND f.tothist_dmd IS NOT NULL
          AND f.basefcst_pref IS NOT NULL
        GROUP BY f.lag
        ORDER BY f.lag
        """,
    )
    rows = cur.fetchall()

    # Also get ceiling accuracy per lag
    cur.execute(
        """
        SELECT f.lag,
               100.0 - (100.0 * SUM(ABS(f.basefcst_pref - f.tothist_dmd)) / NULLIF(ABS(SUM(f.tothist_dmd)), 0))
                   AS ceiling_accuracy
        FROM fact_external_forecast_monthly f
        WHERE f.model_id = 'ceiling'
          AND f.tothist_dmd IS NOT NULL
          AND f.basefcst_pref IS NOT NULL
        GROUP BY f.lag
        ORDER BY f.lag
        """,
    )
    ceiling_map = {r[0]: r[1] for r in cur.fetchall()}

    count = 0
    for lag_val, champ_acc, n_months in rows:
        ceil_acc = ceiling_map.get(lag_val)
        gap = round((float(ceil_acc) - float(champ_acc)) * 100, 2) if ceil_acc and champ_acc else None
        cur.execute(
            """
            INSERT INTO champion_experiment_lag
                (experiment_id, exec_lag, champion_accuracy, ceiling_accuracy,
                 gap_bps, n_dfu_months, model_distribution)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (experiment_id, exec_lag) DO UPDATE SET
                champion_accuracy = EXCLUDED.champion_accuracy,
                ceiling_accuracy  = EXCLUDED.ceiling_accuracy,
                gap_bps           = EXCLUDED.gap_bps,
                n_dfu_months      = EXCLUDED.n_dfu_months
            """,
            (experiment_id, lag_val, champ_acc, ceil_acc, gap, n_months, None),
        )
        count += 1
    return count


def _insert_champion_month_rows(
    cur: psycopg.Cursor,  # type: ignore[type-arg]
    experiment_id: int,
) -> int:
    """Compute per-month champion accuracy from fact_external_forecast_monthly."""
    cur.execute(
        """
        SELECT f.startdate,
               100.0 - (100.0 * SUM(ABS(f.basefcst_pref - f.tothist_dmd)) / NULLIF(ABS(SUM(f.tothist_dmd)), 0))
                   AS champion_accuracy,
               COUNT(*) AS n_champions,
               COUNT(DISTINCT f.forecast_ck) AS n_dfu_months
        FROM fact_external_forecast_monthly f
        WHERE f.model_id = 'champion'
          AND f.tothist_dmd IS NOT NULL
          AND f.basefcst_pref IS NOT NULL
          AND f.lag::text = COALESCE(
              (SELECT execution_lag::text FROM dim_sku s
               WHERE s.item_id = f.item_id AND s.loc = f.loc LIMIT 1),
              f.lag::text)
        GROUP BY f.startdate
        ORDER BY f.startdate
        """,
    )
    champ_rows = cur.fetchall()

    # Ceiling per month
    cur.execute(
        """
        SELECT f.startdate,
               100.0 - (100.0 * SUM(ABS(f.basefcst_pref - f.tothist_dmd)) / NULLIF(ABS(SUM(f.tothist_dmd)), 0))
                   AS ceiling_accuracy
        FROM fact_external_forecast_monthly f
        WHERE f.model_id = 'ceiling'
          AND f.tothist_dmd IS NOT NULL
          AND f.basefcst_pref IS NOT NULL
        GROUP BY f.startdate
        ORDER BY f.startdate
        """,
    )
    ceiling_map = {r[0]: r[1] for r in cur.fetchall()}

    count = 0
    for month, champ_acc, n_champs, n_months in champ_rows:
        ceil_acc = ceiling_map.get(month)
        gap = round((float(ceil_acc) - float(champ_acc)) * 100, 2) if ceil_acc and champ_acc else None
        cur.execute(
            """
            INSERT INTO champion_experiment_month
                (experiment_id, month_start, champion_accuracy, ceiling_accuracy,
                 gap_bps, n_champions, model_distribution)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (experiment_id, month_start) DO UPDATE SET
                champion_accuracy = EXCLUDED.champion_accuracy,
                ceiling_accuracy  = EXCLUDED.ceiling_accuracy,
                gap_bps           = EXCLUDED.gap_bps,
                n_champions       = EXCLUDED.n_champions
            """,
            (experiment_id, month, champ_acc, ceil_acc, gap, n_champs, None),
        )
        count += 1
    return count


# ---------------------------------------------------------------------------
# Clustering seeding
# ---------------------------------------------------------------------------

def seed_clustering_baseline(conn: psycopg.Connection) -> int | None:  # type: ignore[type-arg]
    """Seed the production clustering run as a promoted cluster experiment.

    Returns the inserted experiment_id, or None if artifacts are missing.
    """
    meta_path = ROOT / "data/clustering/cluster_metadata.json"
    profiles_path = ROOT / "data/clustering/cluster_profiles.json"

    if not meta_path.exists():
        logger.warning("Skipping clustering: metadata not found at %s", meta_path)
        return None

    meta = json.loads(meta_path.read_text())
    profiles = None
    if profiles_path.exists():
        profiles = json.loads(profiles_path.read_text())

    # Build config sub-objects matching cluster_experiment schema
    feature_params = {
        "time_window_months": meta.get("time_window_months", 36),
        "min_months_history": meta.get("min_months_history", 12),
    }
    model_params = {
        "k_range": meta.get("k_selection_results", {}).get("k_values", [9, 18]),
        "min_cluster_size_pct": meta.get("min_cluster_size_pct", 2.0),
        "use_pca": meta.get("use_pca", False),
        "pca_components": meta.get("pca_components"),
        "all_features": meta.get("feature_names", []),
    }
    label_params = {
        "volume_high": 0.75, "volume_low": 0.25,
        "cv_steady": 0.4, "cv_volatile": 0.8,
        "seasonality_threshold": 0.3, "zero_demand_threshold": 0.15,
    }

    # Generate a unique scenario_id
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    scenario_id = f"sc_production_{ts}"

    total_skus = sum(meta.get("cluster_sizes", {}).values())

    with conn.cursor() as cur:
        # Remove existing baseline
        cur.execute(
            "DELETE FROM cluster_experiment WHERE label = %s AND template_id = %s",
            (BASELINE_LABEL, TEMPLATE_ID),
        )

        # Clear promoted flag
        cur.execute(
            "UPDATE cluster_experiment SET is_promoted = FALSE, promoted_at = NULL "
            "WHERE is_promoted = TRUE",
        )

        cur.execute(
            """
            INSERT INTO cluster_experiment
                (scenario_id, label, notes, template_id, status,
                 started_at, completed_at,
                 feature_params, model_params, label_params,
                 optimal_k, silhouette_score, inertia,
                 total_dfus, n_clusters, cluster_sizes, profiles,
                 k_selection_results,
                 is_promoted, promoted_at,
                 artifacts_path)
            VALUES
                (%s, %s, %s, %s, 'completed',
                 NOW(), NOW(),
                 %s, %s, %s,
                 %s, %s, %s,
                 %s, %s, %s, %s,
                 %s,
                 TRUE, NOW(),
                 %s)
            RETURNING experiment_id
            """,
            (
                scenario_id, BASELINE_LABEL,
                f"Auto-seeded from production clustering at {datetime.now(timezone.utc).isoformat()}",
                TEMPLATE_ID,
                json.dumps(feature_params), json.dumps(model_params), json.dumps(label_params),
                meta.get("optimal_k"), meta.get("silhouette_score"), meta.get("inertia"),
                total_skus, meta.get("n_clusters"),
                json.dumps(meta.get("cluster_sizes")),
                json.dumps(profiles),
                json.dumps(meta.get("k_selection_results")),
                str(ROOT / "data/clustering"),
            ),
        )
        row = cur.fetchone()
        if row is None:
            raise psycopg.Error("INSERT did not return an experiment_id")
        experiment_id: int = row[0]
        logger.info(
            "Inserted clustering baseline experiment_id=%d (k=%d, silhouette=%.4f)",
            experiment_id, meta.get("optimal_k", 0), meta.get("silhouette_score", 0),
        )

    conn.commit()
    return experiment_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed production pipeline results as promoted baseline experiments.",
    )
    parser.add_argument(
        "--scope",
        choices=["tuning", "champion", "clustering"],
        default=None,
        help="Seed only this scope (default: all)",
    )
    parser.add_argument(
        "--model",
        choices=["lgbm", "catboost", "xgboost", "chronos"],
        default=None,
        help="Seed only this model (tuning scope only)",
    )
    args = parser.parse_args()

    scopes = [args.scope] if args.scope else ["tuning", "champion", "clustering"]
    db_params = get_db_params()

    results: dict[str, Any] = {}

    with psycopg.connect(**db_params) as conn:
        if "tuning" in scopes:
            models = MODEL_IDS
            if args.model:
                # Chronos uses bare model_id; tree models use _cluster suffix
                models = [args.model if args.model == "chronos" else f"{args.model}_cluster"]
            for model_id in models:
                run_id = seed_tuning_baseline(model_id, conn)
                results[model_id] = run_id

        if "champion" in scopes:
            exp_id = seed_champion_baseline(conn)
            results["champion"] = exp_id

        if "clustering" in scopes:
            exp_id = seed_clustering_baseline(conn)
            results["clustering"] = exp_id

    logger.info("Seeding complete: %s", results)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )
    main()
