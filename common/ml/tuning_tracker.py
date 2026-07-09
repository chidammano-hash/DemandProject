"""Tuning run tracker for LightGBM backtest experiments.

Provides helpers to register, complete, fail, compare, and list tuning runs
stored in the ``lgbm_tuning_run``, ``lgbm_tuning_timeframe``, and
``lgbm_tuning_comparison`` Postgres tables (see ``sql/095_create_lgbm_tuning.sql``).

All DB access uses psycopg3 with ``%s`` placeholders and explicit connection
management via ``common.core.db.get_db_params``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import psycopg

from common.core.constants import FORECAST_QTY_COL
from common.core.db import get_db_params
from common.core.utils import load_config, load_forecast_pipeline_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Read and parse a ``backtest_metadata.json`` file."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with open(path) as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _load_tuning_config() -> dict[str, Any]:
    """Load tracking settings from ``forecast_pipeline_config.yaml`` ``tracking`` section."""
    pipeline_cfg = load_forecast_pipeline_config()
    return pipeline_cfg.get("tracking", {})


def _fetch_run(cur: psycopg.Cursor, run_id: int) -> dict[str, Any] | None:  # type: ignore[type-arg]
    """Fetch a single run row as a dict, or ``None`` if not found."""
    cur.execute(
        """
        SELECT run_id, run_label, model_id, started_at, completed_at,
               status, params, feature_count, features,
               accuracy_pct, wape, bias,
               n_predictions, n_dfus, metadata, notes, backup_path
        FROM lgbm_tuning_run
        WHERE run_id = %s
        """,
        (run_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    cols = [
        "run_id", "run_label", "model_id", "started_at", "completed_at",
        "status", "params", "feature_count", "features",
        "accuracy_pct", "wape", "bias",
        "n_predictions", "n_dfus", "metadata", "notes", "backup_path",
    ]
    return dict(zip(cols, row))


def _fetch_timeframes(cur: psycopg.Cursor, run_id: int) -> list[dict[str, Any]]:  # type: ignore[type-arg]
    """Fetch per-timeframe rows for a run."""
    cur.execute(
        """
        SELECT timeframe, train_end, predict_start, predict_end,
               n_predictions, accuracy_pct, wape, bias
        FROM lgbm_tuning_timeframe
        WHERE run_id = %s
        ORDER BY timeframe
        """,
        (run_id,),
    )
    cols = [
        "timeframe", "train_end", "predict_start", "predict_end",
        "n_predictions", "accuracy_pct", "wape", "bias",
    ]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _determine_verdict(delta_accuracy: float, cfg: dict[str, Any]) -> str:
    """Return ``'improved'``, ``'degraded'``, or ``'neutral'`` based on config thresholds."""
    verdicts = cfg.get("verdicts", {})
    improved_min = float(verdicts.get("improved_min_delta_accuracy", 0.05))
    degraded_max = float(verdicts.get("degraded_max_delta_accuracy", -0.05))

    if delta_accuracy >= improved_min:
        return "improved"
    if delta_accuracy <= degraded_max:
        return "degraded"
    return "neutral"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_run(
    run_label: str,
    model_id: str = "lgbm_cluster",
    params: dict[str, Any] | None = None,
    features: list[str] | None = None,
    notes: str | None = None,
) -> int:
    """Insert a new tuning run record and return its ``run_id``.

    The run is created with ``status = 'running'`` and ``started_at = now()``.
    """
    feature_count = len(features) if features else None

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lgbm_tuning_run
                    (run_label, model_id, params, feature_count, features, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING run_id
                """,
                (run_label, model_id, json.dumps(params), feature_count,
                 json.dumps(features), notes),
            )
            row = cur.fetchone()
            if row is None:
                raise psycopg.Error("INSERT did not return a run_id")
            run_id: int = row[0]
        conn.commit()

    logger.info("Registered tuning run %d — label=%s model=%s", run_id, run_label, model_id)
    return run_id


def complete_run(
    run_id: int,
    metadata_path: str | Path,
    backup_path: str | None = None,
) -> None:
    """Mark a run as completed, filling metrics from ``backtest_metadata.json``.

    Reads the metadata file to extract accuracy, WAPE, bias, prediction/DFU
    counts, and the full metadata blob.
    """
    meta = _read_metadata(metadata_path)

    acc_block = meta.get("accuracy_at_execution_lag", {})
    accuracy_pct = acc_block.get("accuracy_pct")
    wape = acc_block.get("wape")
    bias = acc_block.get("bias")
    n_predictions = meta.get("n_predictions")
    n_dfus = meta.get("n_dfus")

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE lgbm_tuning_run
                SET status       = 'completed',
                    completed_at = now(),
                    accuracy_pct = %s,
                    wape         = %s,
                    bias         = %s,
                    n_predictions = %s,
                    n_dfus       = %s,
                    metadata     = %s,
                    backup_path  = %s
                WHERE run_id = %s
                """,
                (accuracy_pct, wape, bias, n_predictions, n_dfus,
                 json.dumps(meta), backup_path, run_id),
            )
        conn.commit()

    logger.info(
        "Completed tuning run %d — accuracy=%.2f%% wape=%.2f bias=%.4f",
        run_id,
        accuracy_pct if accuracy_pct is not None else 0.0,
        wape if wape is not None else 0.0,
        bias if bias is not None else 0.0,
    )


def fail_run(run_id: int, error: str | None = None) -> None:
    """Mark a run as failed with an optional error message stored in ``notes``."""
    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE lgbm_tuning_run
                SET status       = 'failed',
                    completed_at = now(),
                    notes        = COALESCE(notes || E'\\n', '') || %s
                WHERE run_id = %s
                """,
                (error or "Run failed", run_id),
            )
        conn.commit()

    logger.warning("Marked tuning run %d as failed — %s", run_id, error or "(no details)")


def register_timeframes(run_id: int, metadata_path: str | Path) -> None:
    """Insert per-timeframe breakdown rows from the metadata JSON.

    Reads the ``timeframes`` list in ``backtest_metadata.json`` and inserts
    one row per entry into ``lgbm_tuning_timeframe``.
    """
    meta = _read_metadata(metadata_path)
    timeframes = meta.get("timeframes", [])

    if not timeframes:
        logger.warning("No timeframes found in %s — skipping", metadata_path)
        return

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            for tf in timeframes:
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
                        run_id,
                        tf.get("label"),
                        tf.get("train_end"),
                        tf.get("predict_start"),
                        tf.get("predict_end"),
                        tf.get("n_predictions"),
                        tf.get("accuracy_pct"),
                        tf.get("wape"),
                        tf.get("bias"),
                    ),
                )
        conn.commit()

    logger.info("Registered %d timeframes for run %d", len(timeframes), run_id)


def register_cluster_month_breakdowns(
    run_id: int,
    predictions_path: str | Path,
) -> None:
    """Compute and store per-cluster and per-month accuracy breakdowns.

    Reads the backtest predictions CSV, joins to ``dim_sku`` for cluster
    columns, and inserts per-cluster (ml_cluster + business cluster) and
    per-month accuracy rows into ``lgbm_tuning_cluster`` and
    ``lgbm_tuning_month``.
    """
    import pandas as pd

    csv_path = Path(predictions_path)
    if not csv_path.exists():
        logger.warning("Predictions CSV not found at %s — skipping breakdowns", csv_path)
        return

    logger.info("Computing cluster/month breakdowns from %s", csv_path)
    df = pd.read_csv(csv_path, usecols=["item_id", "loc", "startdate", FORECAST_QTY_COL, "tothist_dmd"])
    df = df.dropna(subset=[FORECAST_QTY_COL, "tothist_dmd"])

    if df.empty:
        logger.warning("No valid predictions — skipping breakdowns")
        return

    # Fetch cluster mappings from the promoted assignment view.
    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT d.item_id, d.loc, ca.ml_cluster, d.cluster_assignment
                FROM dim_sku d
                LEFT JOIN current_sku_cluster_assignment ca
                       ON ca.sku_ck = d.sku_ck
            """)
            sku_rows = cur.fetchall()

    sku_df = pd.DataFrame(sku_rows, columns=["item_id", "loc", "ml_cluster", "cluster_assignment"])
    # Ensure consistent types for merge keys
    df["item_id"] = df["item_id"].astype(str)
    df["loc"] = df["loc"].astype(str)
    sku_df["item_id"] = sku_df["item_id"].astype(str)
    sku_df["loc"] = sku_df["loc"].astype(str)
    df = df.merge(sku_df, on=["item_id", "loc"], how="left")
    df["ml_cluster"] = df["ml_cluster"].fillna("unknown")
    df["cluster_assignment"] = df["cluster_assignment"].fillna("unknown")

    def _agg_accuracy(group: pd.DataFrame) -> dict[str, Any]:
        forecast_sum = group[FORECAST_QTY_COL].sum()
        actual_sum = group["tothist_dmd"].sum()
        abs_error = (group[FORECAST_QTY_COL] - group["tothist_dmd"]).abs().sum()
        n_preds = len(group)
        n_dfus = group[["item_id", "loc"]].drop_duplicates().shape[0]
        if abs(actual_sum) < 1e-9:
            return {"n_predictions": n_preds, "n_dfus": n_dfus,
                    "accuracy_pct": None, "wape": None, "bias": None}
        wape = 100.0 * abs_error / abs(actual_sum)
        bias = (forecast_sum / actual_sum) - 1.0
        accuracy_pct = 100.0 - wape
        return {
            "n_predictions": n_preds, "n_dfus": n_dfus,
            "accuracy_pct": round(accuracy_pct, 2),
            "wape": round(wape, 2),
            "bias": round(bias, 4),
        }

    # -- Per-cluster breakdowns ------------------------------------------------
    cluster_rows: list[tuple[Any, ...]] = []
    for cluster_type, col in [("ml_cluster", "ml_cluster"), ("business_cluster", "cluster_assignment")]:
        for cluster_val, grp in df.groupby(col):
            m = _agg_accuracy(grp)
            cluster_rows.append((
                run_id, cluster_type, str(cluster_val),
                m["n_predictions"], m["n_dfus"],
                m["accuracy_pct"], m["wape"], m["bias"],
            ))

    # -- Per-month breakdowns --------------------------------------------------
    month_rows: list[tuple[Any, ...]] = []
    for month_val, grp in df.groupby("startdate"):
        m = _agg_accuracy(grp)
        month_rows.append((
            run_id, str(month_val),
            m["n_predictions"], m["n_dfus"],
            m["accuracy_pct"], m["wape"], m["bias"],
        ))

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            for row in cluster_rows:
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
                    row,
                )
            for row in month_rows:
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
                    row,
                )
        conn.commit()

    logger.info(
        "Registered %d cluster rows + %d month rows for run %d",
        len(cluster_rows), len(month_rows), run_id,
    )


def get_latest_completed_run(model_id: str = "lgbm_cluster") -> dict[str, Any] | None:
    """Return the most recent completed run as a dict, or ``None`` if none exist."""
    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, run_label, model_id, started_at, completed_at,
                       status, params, feature_count, features,
                       accuracy_pct, wape, bias,
                       n_predictions, n_dfus, metadata, notes, backup_path
                FROM lgbm_tuning_run
                WHERE status = 'completed' AND model_id = %s
                ORDER BY completed_at DESC
                LIMIT 1
                """,
                (model_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [
                "run_id", "run_label", "model_id", "started_at", "completed_at",
                "status", "params", "feature_count", "features",
                "accuracy_pct", "wape", "bias",
                "n_predictions", "n_dfus", "metadata", "notes", "backup_path",
            ]
            return dict(zip(cols, row))


def compare_runs(baseline_id: int, candidate_id: int) -> dict[str, Any]:
    """Compare two runs and return delta metrics.

    Steps:
    1. Fetch both runs from the DB
    2. Compute deltas (candidate - baseline) for accuracy, WAPE, and bias
    3. Determine verdict using thresholds from tuning config
    4. Compute per-timeframe deltas where both runs have matching timeframes
    5. Insert a row into ``lgbm_tuning_comparison``
    6. Return the full comparison dict

    Raises:
        ValueError: If either run is not found or not completed.
    """
    cfg = _load_tuning_config()

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            baseline = _fetch_run(cur, baseline_id)
            candidate = _fetch_run(cur, candidate_id)

            if baseline is None:
                raise ValueError(f"Baseline run {baseline_id} not found")
            if candidate is None:
                raise ValueError(f"Candidate run {candidate_id} not found")
            if baseline["status"] != "completed":
                raise ValueError(f"Baseline run {baseline_id} is not completed (status={baseline['status']})")
            if candidate["status"] != "completed":
                raise ValueError(f"Candidate run {candidate_id} is not completed (status={candidate['status']})")

            # -- Compute deltas ------------------------------------------------
            delta_accuracy = float(candidate.get("accuracy_pct") or 0) - float(baseline.get("accuracy_pct") or 0)
            delta_wape = float(candidate.get("wape") or 0) - float(baseline.get("wape") or 0)
            delta_bias = float(candidate.get("bias") or 0) - float(baseline.get("bias") or 0)

            verdict = _determine_verdict(delta_accuracy, cfg)

            # -- Per-timeframe deltas ------------------------------------------
            base_tfs = {tf["timeframe"]: tf for tf in _fetch_timeframes(cur, baseline_id)}
            cand_tfs = {tf["timeframe"]: tf for tf in _fetch_timeframes(cur, candidate_id)}

            per_timeframe: list[dict[str, Any]] = []
            for label in sorted(set(base_tfs.keys()) & set(cand_tfs.keys())):
                bt = base_tfs[label]
                ct = cand_tfs[label]
                per_timeframe.append({
                    "timeframe": label,
                    "baseline_accuracy": float(bt.get("accuracy_pct") or 0),
                    "candidate_accuracy": float(ct.get("accuracy_pct") or 0),
                    "delta_accuracy": float(ct.get("accuracy_pct") or 0) - float(bt.get("accuracy_pct") or 0),
                    "baseline_wape": float(bt.get("wape") or 0),
                    "candidate_wape": float(ct.get("wape") or 0),
                    "delta_wape": float(ct.get("wape") or 0) - float(bt.get("wape") or 0),
                })

            # -- Insert comparison record --------------------------------------
            cur.execute(
                """
                INSERT INTO lgbm_tuning_comparison
                    (baseline_run_id, candidate_run_id,
                     delta_accuracy, delta_wape, delta_bias,
                     per_timeframe_detail, verdict)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (baseline_run_id, candidate_run_id) DO UPDATE SET
                    delta_accuracy       = EXCLUDED.delta_accuracy,
                    delta_wape           = EXCLUDED.delta_wape,
                    delta_bias           = EXCLUDED.delta_bias,
                    per_timeframe_detail = EXCLUDED.per_timeframe_detail,
                    verdict              = EXCLUDED.verdict,
                    created_at           = now()
                """,
                (
                    baseline_id, candidate_id,
                    delta_accuracy, delta_wape, delta_bias,
                    json.dumps(per_timeframe), verdict,
                ),
            )
        conn.commit()

    logger.info(
        "Compared run %d vs %d — delta_accuracy=%.2f verdict=%s",
        baseline_id, candidate_id, delta_accuracy, verdict,
    )

    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta_accuracy": delta_accuracy,
        "delta_wape": delta_wape,
        "delta_bias": delta_bias,
        "verdict": verdict,
        "per_timeframe": per_timeframe,
    }


def list_runs(model_id: str = "lgbm_cluster", limit: int = 20) -> list[dict[str, Any]]:
    """List recent tuning runs ordered by ``started_at DESC``.

    Returns at most *limit* rows. Each row is a dict with all run-level columns.
    """
    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_id, run_label, model_id, started_at, completed_at,
                       status, accuracy_pct, wape, bias,
                       n_predictions, n_dfus, feature_count, notes
                FROM lgbm_tuning_run
                WHERE model_id = %s
                ORDER BY started_at DESC
                LIMIT %s
                """,
                (model_id, limit),
            )
            cols = [
                "run_id", "run_label", "model_id", "started_at", "completed_at",
                "status", "accuracy_pct", "wape", "bias",
                "n_predictions", "n_dfus", "feature_count", "notes",
            ]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
