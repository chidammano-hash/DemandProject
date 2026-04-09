"""Backtest Management API router.

Provides endpoints to list, inspect, submit, and load backtest runs
for all models in the forecast pipeline. Tracks run history in the
``backtest_run`` table and reads current metadata from disk.

All endpoints live under the /backtest-management prefix.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_api_key
from api.core import get_conn
from common.utils import get_algorithm_roster

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest-management", tags=["backtest-management"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BACKTEST_DIR = _PROJECT_ROOT / "data" / "backtest"

# Map pipeline config model_id → job registry type_id.
# The key is the model_id from forecast_pipeline_config.yaml algorithms,
# the value is the job_type key in JOB_TYPE_REGISTRY.
MODEL_TO_JOB_TYPE: dict[str, str] = {
    "lgbm_cluster": "backtest_lgbm",
    "catboost_cluster": "backtest_catboost",
    "xgboost_cluster": "backtest_xgboost",
    "lgbm_cust_enriched": "backtest_lgbm",
    "catboost_cust_enriched": "backtest_catboost",
    "xgboost_cust_enriched": "backtest_xgboost",
    "chronos": "backtest_chronos",
    "chronos_bolt": "backtest_chronos_bolt",
    "chronos2": "backtest_chronos2",
    "chronos2_enriched": "backtest_chronos2_enriched",
    "bolt_hierarchical": "backtest_bolt_hierarchical",
    "mstl": "backtest_mstl",
    "seasonal_naive": "backtest_seasonal_naive",
    "rolling_mean": "backtest_rolling_mean",
    "nhits": "backtest_nhits",
    "nbeats": "backtest_nbeats",
}

# Map model_id → the subdirectory name under data/backtest/ where outputs live.
# Most models use the model_id itself; tree models use the _cluster suffix dir.
MODEL_TO_DIR: dict[str, str] = {
    "lgbm_cluster": "lgbm_cluster",
    "catboost_cluster": "catboost_cluster",
    "xgboost_cluster": "xgboost_cluster",
    "lgbm_cust_enriched": "lgbm_cust_enriched",
    "catboost_cust_enriched": "catboost_cust_enriched",
    "xgboost_cust_enriched": "xgboost_cust_enriched",
    "chronos": "chronos",
    "chronos_bolt": "chronos_bolt",
    "chronos2": "chronos2",
    "chronos2_enriched": "chronos2_enriched",
    "bolt_hierarchical": "bolt_hierarchical",
    "mstl": "mstl",
    "seasonal_naive": "seasonal_naive",
    "rolling_mean": "rolling_mean",
    "nhits": "nhits",
    "nbeats": "nbeats",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_metadata_from_disk(model_id: str) -> dict[str, Any] | None:
    """Read backtest_metadata.json from disk for a model. Returns None if missing."""
    dir_name = MODEL_TO_DIR.get(model_id, model_id)
    meta_path = _BACKTEST_DIR / dir_name / "backtest_metadata.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read metadata for %s: %s", model_id, exc)
        return None


def _row_to_dict(row: tuple) -> dict[str, Any]:
    """Convert a backtest_run row tuple to a response dict.

    Column order must match SELECT in get_all_backtest_summary() and get_model_runs():
    0=id, 1=model_id, 2=job_id, 3=status, 4=accuracy_pct, 5=wape, 6=bias,
    7=n_predictions, 8=n_dfus, 9=n_timeframes, 10=metadata, 11=is_loaded_to_db,
    12=loaded_at, 13=load_job_id, 14=started_at, 15=completed_at, 16=created_at
    """
    return {
        "id": row[0],
        "model_id": row[1],
        "job_id": row[2],
        "status": row[3],
        "accuracy_pct": float(row[4]) if row[4] is not None else None,
        "wape": float(row[5]) if row[5] is not None else None,
        "bias": float(row[6]) if row[6] is not None else None,
        "n_predictions": row[7],
        "n_dfus": row[8],
        "n_timeframes": row[9],
        "metadata": row[10] if isinstance(row[10], dict) else (
            json.loads(row[10]) if row[10] else None
        ),
        "is_loaded_to_db": row[11],
        "loaded_at": row[12].isoformat() if row[12] else None,
        "load_job_id": row[13],
        "started_at": row[14].isoformat() if row[14] else None,
        "completed_at": row[15].isoformat() if row[15] else None,
        "created_at": row[16].isoformat() if row[16] else None,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/training-status")
def get_training_status():
    """Get production training status for all forecastable models."""
    roster = get_algorithm_roster()
    result: dict[str, Any] = {}
    for model_id, algo_info in roster.items():
        if not algo_info.get("forecast"):
            continue
        entry: dict[str, Any] = {
            "model_id": model_id,
            "type": algo_info.get("type", "unknown"),
        }
        meta_path = _PROJECT_ROOT / "data" / "models" / model_id / "training_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                entry["trained"] = True
                entry["trained_at"] = meta.get("trained_at")
                entry["training_mode"] = meta.get("training_mode")
                entry["n_dfus"] = meta.get("n_dfus")
                entry["planning_date"] = meta.get("planning_date")
            except (json.JSONDecodeError, OSError):
                entry["trained"] = False
                entry["trained_at"] = None
                entry["training_mode"] = None
        else:
            # Check if .pkl files exist (backtest artifacts without metadata)
            model_dir = _PROJECT_ROOT / "data" / "models" / model_id
            has_pkl = model_dir.exists() and any(model_dir.glob("cluster_*.pkl"))
            entry["trained"] = has_pkl
            entry["trained_at"] = None
            entry["training_mode"] = "backtest" if has_pkl else None
        result[model_id] = entry
    return result


@router.post("/{model_id}/train", status_code=201, dependencies=[Depends(require_api_key)])
def submit_production_training(model_id: str):
    """Submit a production model training job.

    Use model_id='all' to train all forecastable tree models.
    Only tree models (lgbm, catboost, xgboost variants) support training.
    """
    # Validate model type — only tree models can be trained
    _TRAINABLE_TYPES = {"tree"}
    is_all = model_id == "all"
    if not is_all:
        roster = get_algorithm_roster()
        algo_info = roster.get(model_id)
        if algo_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown model_id '{model_id}'. Valid models: {sorted(roster.keys())}",
            )
        if algo_info.get("type") not in _TRAINABLE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' (type={algo_info.get('type')}) does not require training. "
                       f"Only tree models need explicit training.",
            )

    try:
        from common.services.job_registry import JobManager
        jm = JobManager()
        job_id = jm.submit_job(
            job_type="train_production_model",
            params={"model_id": "" if is_all else model_id, "all_models": is_all},
            label=f"Train Production: {'All Models' if is_all else model_id}",
        )
    except ValueError as exc:
        logger.exception("Failed to submit training job for %s", model_id)
        raise HTTPException(status_code=400, detail=str(exc)) from None

    return {"job_id": job_id, "model_id": model_id}


@router.get("/summary")
def get_all_backtest_summary():
    """Return backtest status for all algorithms in pipeline config.

    For each algorithm in forecast_pipeline_config.yaml:
      - Latest run from backtest_run table (accuracy, status, is_loaded)
      - Current metadata from data/backtest/{model_id}/backtest_metadata.json
    Returns a dict keyed by model_id.
    """
    roster = get_algorithm_roster()

    # Auto-fix stale rows: if a job finished/cancelled but backtest_run wasn't updated
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE backtest_run br SET
                    status = CASE
                        WHEN jh.status = 'completed' THEN 'completed'
                        ELSE 'failed'
                    END,
                    completed_at = NOW()
                FROM job_history jh
                WHERE jh.job_id = br.job_id
                  AND br.status IN ('queued', 'running')
                  AND jh.status IN ('completed', 'failed', 'cancelled')
            """)
            conn.commit()
    except Exception as exc:
        logger.warning("Stale backtest_run cleanup failed: %s", exc)

    # Fetch latest run per model_id from DB
    # Try to include is_loaded_to_candidate if column exists (graceful fallback)
    latest_runs: dict[str, dict[str, Any]] = {}
    _has_candidate_cols = False
    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Probe for the new columns; fall back to the original query if absent
            try:
                cur.execute(
                    """
                    SELECT DISTINCT ON (model_id)
                        id, model_id, job_id, status,
                        accuracy_pct, wape, bias,
                        n_predictions, n_dfus, n_timeframes,
                        metadata, is_loaded_to_db, loaded_at, load_job_id,
                        started_at, completed_at, created_at,
                        is_loaded_to_candidate, candidate_loaded_at
                    FROM backtest_run
                    ORDER BY model_id, created_at DESC
                    """
                )
                _has_candidate_cols = True
            except Exception:
                conn.rollback()
                cur.execute(
                    """
                    SELECT DISTINCT ON (model_id)
                        id, model_id, job_id, status,
                        accuracy_pct, wape, bias,
                        n_predictions, n_dfus, n_timeframes,
                        metadata, is_loaded_to_db, loaded_at, load_job_id,
                        started_at, completed_at, created_at
                    FROM backtest_run
                    ORDER BY model_id, created_at DESC
                    """
                )
            for row in cur.fetchall():
                d = _row_to_dict(row)
                if _has_candidate_cols and len(row) > 17:
                    d["is_loaded_to_candidate"] = row[17]
                    d["candidate_loaded_at"] = row[18].isoformat() if row[18] else None
                latest_runs[row[1]] = d
    except Exception:
        logger.exception("Failed to query backtest_run table")

    result: dict[str, dict[str, Any]] = {}
    for model_id, algo_info in roster.items():
        entry: dict[str, Any] = {
            "model_id": model_id,
            "type": algo_info.get("type", "unknown"),
            "enabled": algo_info.get("enabled", False),
            "has_job_type": model_id in MODEL_TO_JOB_TYPE,
        }

        # Latest DB run
        if model_id in latest_runs:
            run = latest_runs[model_id]
            latest_run_entry: dict[str, Any] = {
                "id": run["id"],
                "status": run["status"],
                "accuracy_pct": run["accuracy_pct"],
                "wape": run["wape"],
                "bias": run["bias"],
                "is_loaded_to_db": run["is_loaded_to_db"],
                "created_at": run["created_at"],
                "completed_at": run["completed_at"],
            }
            # Include candidate load status if available
            if "is_loaded_to_candidate" in run:
                latest_run_entry["is_loaded_to_candidate"] = run["is_loaded_to_candidate"]
                latest_run_entry["candidate_loaded_at"] = run["candidate_loaded_at"]
            entry["latest_run"] = latest_run_entry
        else:
            entry["latest_run"] = None

        # Disk metadata
        disk_meta = _read_metadata_from_disk(model_id)
        entry["disk_metadata"] = disk_meta

        # Check if predictions CSV exists
        dir_name = MODEL_TO_DIR.get(model_id, model_id)
        pred_path = _BACKTEST_DIR / dir_name / "backtest_predictions.csv"
        entry["has_predictions_csv"] = pred_path.exists()
        entry["has_predictions"] = entry["has_predictions_csv"]  # alias for frontend

        # Convenience accuracy/wape from disk metadata or latest run
        entry["current_accuracy"] = (
            disk_meta.get("accuracy_pct") if disk_meta else None
        ) or (latest_runs[model_id]["accuracy_pct"] if model_id in latest_runs else None)
        entry["current_wape"] = (
            disk_meta.get("wape") if disk_meta else None
        ) or (latest_runs[model_id]["wape"] if model_id in latest_runs else None)

        result[model_id] = entry

    return result


@router.get("/{model_id}/runs")
def get_model_runs(model_id: str):
    """List all backtest runs for a model, newest first."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Auto-fix stale rows before returning
            cur.execute("""
                UPDATE backtest_run br SET
                    status = CASE
                        WHEN jh.status = 'completed' THEN 'completed'
                        ELSE 'failed'
                    END,
                    completed_at = NOW()
                FROM job_history jh
                WHERE jh.job_id = br.job_id
                  AND br.model_id = %s
                  AND br.status IN ('queued', 'running')
                  AND jh.status IN ('completed', 'failed', 'cancelled')
            """, (model_id,))
            conn.commit()

            cur.execute(
                """
                SELECT id, model_id, job_id, status,
                       accuracy_pct, wape, bias,
                       n_predictions, n_dfus, n_timeframes,
                       metadata, is_loaded_to_db, loaded_at, load_job_id,
                       started_at, completed_at, created_at
                FROM backtest_run
                WHERE model_id = %s
                ORDER BY created_at DESC
                """,
                (model_id,),
            )
            rows = cur.fetchall()
    except Exception:
        logger.exception("Failed to query backtest_run for model %s", model_id)
        raise HTTPException(status_code=500, detail="Database error") from None

    return [_row_to_dict(row) for row in rows]


@router.get("/{model_id}/current")
def get_current_metadata(model_id: str):
    """Read current backtest_metadata.json from disk for the given model."""
    meta = _read_metadata_from_disk(model_id)
    if meta is None:
        raise HTTPException(
            status_code=404,
            detail=f"No backtest metadata found on disk for model '{model_id}'",
        )
    return meta


@router.post("/{model_id}/run", status_code=201, dependencies=[Depends(require_api_key)])
def submit_backtest_run(model_id: str):
    """Submit a backtest job for the given model.

    Validates model_id exists in pipeline config, maps to the correct job type,
    inserts a tracking row into backtest_run, and submits the job.
    """
    # Validate model_id is in pipeline config
    roster = get_algorithm_roster()
    if model_id not in roster:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model_id '{model_id}'. Valid models: {sorted(roster.keys())}",
        )

    # Validate we have a job type mapping
    job_type = MODEL_TO_JOB_TYPE.get(model_id)
    if not job_type:
        raise HTTPException(
            status_code=400,
            detail=f"No backtest job type configured for model '{model_id}'",
        )

    # Insert tracking row
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO backtest_run (model_id, status)
                VALUES (%s, 'queued')
                RETURNING id
                """,
                (model_id,),
            )
            run_id = cur.fetchone()[0]
            conn.commit()
    except Exception:
        logger.exception("Failed to insert backtest_run for %s", model_id)
        raise HTTPException(status_code=500, detail="Failed to create backtest run") from None

    # Submit job via JobManager
    try:
        from common.services.job_registry import JobManager
        jm = JobManager()
        job_id = jm.submit_job(
            job_type=job_type,
            params={"backtest_run_id": run_id},
            label=f"Backtest: {model_id}",
        )
    except ValueError as exc:
        logger.exception("Failed to submit backtest job for %s", model_id)
        raise HTTPException(status_code=400, detail=str(exc)) from None

    # Store job_id on the tracking row
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE backtest_run SET job_id = %s WHERE id = %s",
                (job_id, run_id),
            )
            conn.commit()
    except Exception:
        logger.warning("Failed to store job_id %s on backtest_run %d", job_id, run_id)

    return {
        "run_id": run_id,
        "job_id": job_id,
        "model_id": model_id,
        "status": "queued",
    }


@router.post("/{model_id}/load", status_code=201, dependencies=[Depends(require_api_key)])
def submit_backtest_load(model_id: str, run_id: int | None = None):
    """Load backtest predictions into DB for the given model.

    Checks that predictions CSV exists on disk, then submits a
    backtest_load_model job. Optionally links to a specific backtest_run
    via run_id.
    """
    # Resolve output directory name
    dir_name = MODEL_TO_DIR.get(model_id, model_id)
    pred_path = _BACKTEST_DIR / dir_name / "backtest_predictions.csv"

    if not pred_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No predictions CSV found at {pred_path.relative_to(_PROJECT_ROOT)}",
        )

    # Submit load job
    try:
        from common.services.job_registry import JobManager
        jm = JobManager()
        job_id = jm.submit_job(
            job_type="backtest_load_model",
            params={"model_id": dir_name, "run_id": run_id},
            label=f"Load Backtest: {model_id}",
        )
    except ValueError as exc:
        logger.exception("Failed to submit load job for %s", model_id)
        raise HTTPException(status_code=400, detail=str(exc)) from None

    # Update backtest_run tracking columns if run_id provided
    if run_id is not None:
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE backtest_run SET load_job_id = %s WHERE id = %s",
                    (job_id, run_id),
                )
                # Also track candidate load status (graceful if columns don't exist yet)
                try:
                    cur.execute(
                        "UPDATE backtest_run SET is_loaded_to_candidate = TRUE, candidate_loaded_at = NOW() WHERE id = %s",
                        (run_id,),
                    )
                except Exception:
                    conn.rollback()
                    logger.debug("is_loaded_to_candidate column not yet available on backtest_run")
                conn.commit()
        except Exception:
            logger.warning("Failed to store load_job_id on backtest_run %d", run_id)

    return {
        "job_id": job_id,
        "model_id": model_id,
        "run_id": run_id,
        "target_table": "fact_candidate_forecast",
    }


# ---------------------------------------------------------------------------
# Promotion & Candidate Endpoints
# ---------------------------------------------------------------------------


def _load_dfu_assignments() -> list[tuple[str, str, str]]:
    """Load DFU-level champion model assignments.

    Tries data/champion/dfu_assignments.csv first.
    Falls back to picking best-accuracy model per DFU from candidates.
    Returns list of (item_id, loc, model_id) tuples.
    """
    csv_path = _PROJECT_ROOT / "data" / "champion" / "dfu_assignments.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            return [(row["item_id"], row["loc"], row["model_id"]) for row in reader]

    # Fallback: pick best accuracy per DFU from candidates
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (item_id, loc)
                item_id, loc, model_id
            FROM fact_candidate_forecast
            WHERE accuracy_pct IS NOT NULL
            ORDER BY item_id, loc, accuracy_pct DESC
        """)
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]


@router.get("/promotion-status")
def get_promotion_status():
    """Get the currently active model promotion.

    Returns {"promoted": {...}} when a model is promoted, or {"promoted": null} when none active.
    The promoted dict contains: id, model_id, promotion_type, champion_experiment_id,
    plan_version, promoted_at, dfu_count, total_rows, promoted_by, notes.
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT id, model_id, promotion_type, champion_experiment_id,
                       plan_version, promoted_at, dfu_count, total_rows, promoted_by, notes
                FROM model_promotion_log
                WHERE is_active = TRUE
                ORDER BY promoted_at DESC
                LIMIT 1
            """)
            row = cur.fetchone()
    except Exception:
        logger.debug("model_promotion_log table may not exist yet")
        return {"promoted": None}
    if not row:
        return {"promoted": None}
    return {
        "promoted": {
            "id": row[0],
            "model_id": row[1],
            "promotion_type": row[2],
            "champion_experiment_id": row[3],
            "plan_version": row[4],
            "promoted_at": row[5].isoformat() if row[5] else None,
            "dfu_count": row[6],
            "total_rows": row[7],
            "promoted_by": row[8],
            "notes": row[9],
        }
    }


@router.get("/candidate-summary")
def get_candidate_summary():
    """Get summary of loaded candidate forecasts per model."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT model_id,
                       COUNT(*) AS row_count,
                       COUNT(DISTINCT item_id || '|' || loc) AS dfu_count,
                       MAX(loaded_at) AS last_loaded_at,
                       AVG(accuracy_pct) FILTER (WHERE accuracy_pct IS NOT NULL) AS avg_accuracy
                FROM fact_candidate_forecast
                GROUP BY model_id
                ORDER BY model_id
            """)
            rows = cur.fetchall()
    except Exception:
        logger.debug("fact_candidate_forecast table may not exist yet")
        return {}
    result: dict[str, Any] = {}
    for r in rows:
        result[r[0]] = {
            "model_id": r[0],
            "row_count": r[1],
            "dfu_count": r[2],
            "last_loaded_at": r[3].isoformat() if r[3] else None,
            "avg_accuracy": float(r[4]) if r[4] is not None else None,
        }
    return result


@router.get("/staging-summary")
def get_staging_summary():
    """Summary of production forecasts in staging per model."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT model_id,
                       COUNT(*) AS row_count,
                       COUNT(DISTINCT item_id || '|' || loc) AS dfu_count,
                       forecast_month_generated,
                       MAX(generated_at) AS last_generated_at,
                       MIN(forecast_month) AS min_forecast_month,
                       MAX(forecast_month) AS max_forecast_month
                FROM fact_production_forecast_staging
                GROUP BY model_id, forecast_month_generated
                ORDER BY model_id
            """)
            rows = cur.fetchall()
    except Exception:
        logger.debug("fact_production_forecast_staging table may not exist yet")
        return {}
    result: dict[str, Any] = {}
    for r in rows:
        result[r[0]] = {
            "model_id": r[0],
            "row_count": r[1],
            "dfu_count": r[2],
            "forecast_month_generated": r[3].isoformat() if r[3] else None,
            "last_generated_at": r[4].isoformat() if r[4] else None,
            "min_forecast_month": r[5].isoformat() if r[5] else None,
            "max_forecast_month": r[6].isoformat() if r[6] else None,
        }
    return result


@router.post("/{model_id}/generate", status_code=201, dependencies=[Depends(require_api_key)])
def submit_generate_forecast(model_id: str):
    """Submit production forecast generation for a model, writing to staging."""
    from common.services.job_registry import JobManager
    jm = JobManager()
    job_id = jm.submit_job(
        job_type="generate_production_forecast",
        params={"model_id": model_id},
        label=f"Generate Forecast: {model_id}",
    )
    return {"job_id": job_id, "model_id": model_id}


@router.post("/{model_id}/promote", status_code=201, dependencies=[Depends(require_api_key)])
def promote_model(model_id: str, notes: str | None = None, promoted_by: str | None = None):
    """Promote a model's staged forecasts to production.

    For model_id='champion', promotes per-DFU best-model selection.
    For any other model_id, promotes all staged rows for that single model.
    """
    is_champion = model_id == "champion"
    plan_version = datetime.now(UTC).strftime("%Y-%m")
    run_id_str = str(__import__("uuid").uuid4())
    champion_experiment_id: int | None = None

    with get_conn() as conn, conn.cursor() as cur:
        # 1. Validate staging has rows
        if is_champion:
            cur.execute("SELECT COUNT(*) FROM fact_production_forecast_staging")
        else:
            cur.execute(
                "SELECT COUNT(*) FROM fact_production_forecast_staging WHERE model_id = %s",
                (model_id,),
            )
        count = cur.fetchone()[0]
        if count == 0:
            raise HTTPException(
                status_code=400,
                detail=f"No staged forecasts found for {'any model' if is_champion else model_id}. Run Generate first.",
            )

        # 2. Demote current active promotion
        cur.execute(
            "UPDATE model_promotion_log SET is_active = FALSE, demoted_at = NOW() WHERE is_active = TRUE"
        )

        # 3. Delete old production rows
        cur.execute("DELETE FROM fact_production_forecast")

        # 4. Copy from staging to production
        if is_champion:
            # Get promoted champion experiment for DFU assignments
            try:
                cur.execute("""
                    SELECT experiment_id, models FROM champion_experiment
                    WHERE is_promoted = TRUE ORDER BY promoted_at DESC LIMIT 1
                """)
                champ_row = cur.fetchone()
                if champ_row:
                    champion_experiment_id = champ_row[0]
            except Exception:
                logger.debug("champion_experiment table not available")

            # For champion: copy ALL staged rows (the generate script already
            # applied champion routing per-DFU during inference)
            cur.execute("""
                INSERT INTO fact_production_forecast
                    (plan_version, item_id, loc, forecast_month, forecast_qty,
                     forecast_qty_lower, forecast_qty_upper, model_id, cluster_id,
                     horizon_months, is_recursive, lag_source, generated_at, run_id)
                SELECT
                    %s, item_id, loc, forecast_month, forecast_qty,
                    forecast_qty_lower, forecast_qty_upper, model_id, cluster_id,
                    horizon_months, is_recursive, lag_source, generated_at, %s::uuid
                FROM fact_production_forecast_staging
            """, (plan_version, run_id_str))
        else:
            # Single model: copy only that model's staged rows
            cur.execute("""
                INSERT INTO fact_production_forecast
                    (plan_version, item_id, loc, forecast_month, forecast_qty,
                     forecast_qty_lower, forecast_qty_upper, model_id, cluster_id,
                     horizon_months, is_recursive, lag_source, generated_at, run_id)
                SELECT
                    %s, item_id, loc, forecast_month, forecast_qty,
                    forecast_qty_lower, forecast_qty_upper, model_id, cluster_id,
                    horizon_months, is_recursive, lag_source, generated_at, %s::uuid
                FROM fact_production_forecast_staging
                WHERE model_id = %s
            """, (plan_version, run_id_str, model_id))

        rows_promoted = cur.rowcount

        # 5. Count DFUs
        cur.execute(
            "SELECT COUNT(DISTINCT item_id || '|' || loc) FROM fact_production_forecast"
        )
        dfu_count = cur.fetchone()[0]

        # 6. Log promotion
        cur.execute("""
            INSERT INTO model_promotion_log
                (model_id, promotion_type, champion_experiment_id, plan_version,
                 is_active, dfu_count, total_rows, promoted_by, notes)
            VALUES (%s, %s, %s, %s, TRUE, %s, %s, %s, %s)
        """, (
            model_id,
            "champion" if is_champion else "single",
            champion_experiment_id,
            plan_version,
            dfu_count,
            rows_promoted,
            promoted_by or "api",
            notes,
        ))

        conn.commit()

    return {
        "model_id": model_id,
        "promotion_type": "champion" if is_champion else "single",
        "plan_version": plan_version,
        "rows_promoted": rows_promoted,
        "dfu_count": dfu_count,
    }
