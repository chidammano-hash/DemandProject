"""Backtest Management API router.

Provides endpoints to list, inspect, submit, and load backtest runs
for all models in the forecast pipeline. Tracks run history in the
``backtest_run`` table and reads current metadata from disk.

All endpoints live under the /backtest-management prefix.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Mapping
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Response

from api.auth import require_api_key
from api.core import get_conn
from common.core.paths import PROJECT_ROOT as _PROJECT_ROOT
from common.core.planning_date import get_planning_date
from common.core.utils import get_algorithm_roster, load_forecast_pipeline_config
from common.ml.backtest_config import build_backtest_config_snapshot
from common.ml.neural_artifacts import (
    NeuralArtifactLineageMismatchError,
    load_neural_training_cohort_identity,
    read_active_neural_artifact_ref,
)
from common.ml.neural_forecast import SUPPORTED_NEURAL_MODELS
from common.ml.production_non_tree import direct_model_runtime_contract
from common.ml.tree_artifact_lineage import ProductionTreeArtifactLineage
from common.ml.tree_artifacts import (
    TreeArtifactLineageMismatchError,
    build_production_tree_model_config_payload,
    build_tree_artifact_spec,
    read_active_tree_artifact_ref,
)
from common.services.cluster_lineage import load_promoted_cluster_population
from common.services.forecast_generation import GENERATOR_CONTRACT_VERSION
from common.services.forecast_population import resolve_forecast_sales_table
from common.services.forecast_snapshot_validation import validate_ready_snapshot_contender
from common.services.sales_lineage import load_completed_sales_lineage

from ._training_readiness import (
    DIRECT_INFERENCE_MODEL_IDS,
    CurrentTrainingLineage,
    evaluate_snapshot_roster_readiness,
    invalid_artifact_reason,
    load_current_training_lineage,
    mark_direct_model_ready,
    mark_not_trained,
    missing_artifact_reason,
    production_model_base_dir,
    retrain_reason,
    validate_active_champion_readiness,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest-management", tags=["backtest-management"])

_BACKTEST_DIR = _PROJECT_ROOT / "data" / "backtest"

# Map pipeline config model_id → job registry type_id.
# The key is the model_id from forecast_pipeline_config.yaml algorithms,
# the value is the job_type key in JOB_TYPE_REGISTRY.
MODEL_TO_JOB_TYPE: dict[str, str] = {
    "lgbm_cluster": "backtest_lgbm",
    "chronos2_enriched": "backtest_chronos2_enriched",
    "mstl": "backtest_mstl",
    "nhits": "backtest_nhits",
    "nbeats": "backtest_nbeats",
}

# Map model_id → the subdirectory name under data/backtest/ where outputs live.
# Every retained model uses its canonical pipeline ID as the output directory.
MODEL_TO_DIR: dict[str, str] = {
    "lgbm_cluster": "lgbm_cluster",
    "chronos2_enriched": "chronos2_enriched",
    "mstl": "mstl",
    "nhits": "nhits",
    "nbeats": "nbeats",
}

PRODUCTION_TRAINABLE_MODEL_IDS = frozenset({"lgbm_cluster", *SUPPORTED_NEURAL_MODELS})

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


def _load_current_training_lineage(
    config: Mapping[str, Any],
    *,
    neural_min_history_values: Collection[int] = (),
    require_direct_history: bool = False,
) -> CurrentTrainingLineage:
    """Load current readiness inputs through this router's patchable dependencies."""
    return load_current_training_lineage(
        config,
        neural_min_history_values=neural_min_history_values,
        get_conn=get_conn,
        get_planning_date=get_planning_date,
        load_completed_sales_lineage=load_completed_sales_lineage,
        load_promoted_cluster_population=load_promoted_cluster_population,
        resolve_forecast_sales_table=resolve_forecast_sales_table,
        load_neural_training_cohort_identity=load_neural_training_cohort_identity,
        require_direct_history=require_direct_history,
    )


def _serialize_backtest_run(row: tuple) -> dict[str, Any]:
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
        "metadata": row[10]
        if isinstance(row[10], dict)
        else (json.loads(row[10]) if row[10] else None),
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
    try:
        roster = get_algorithm_roster()
        config = load_forecast_pipeline_config()
        model_base_dir = production_model_base_dir(config)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        logger.exception("Loading production training configuration failed")
        raise HTTPException(
            status_code=500,
            detail="production training readiness check failed",
        ) from exc
    result: dict[str, Any] = {}
    requires_lineage = any(
        model_id in PRODUCTION_TRAINABLE_MODEL_IDS | DIRECT_INFERENCE_MODEL_IDS
        and algo_info.get("forecast")
        for model_id, algo_info in roster.items()
    )
    current_lineage: CurrentTrainingLineage | None = None
    lineage_unavailable_reason: str | None = None
    if requires_lineage:
        try:
            require_direct_history = any(
                model_id in DIRECT_INFERENCE_MODEL_IDS and algo_info.get("forecast")
                for model_id, algo_info in roster.items()
            )
            neural_min_history_values: set[int] = set()
            for model_id, algo_info in roster.items():
                if model_id not in SUPPORTED_NEURAL_MODELS or not algo_info.get("forecast"):
                    continue
                params = algo_info.get("params")
                min_history = params.get("min_history") if isinstance(params, Mapping) else None
                if (
                    not isinstance(min_history, int)
                    or isinstance(min_history, bool)
                    or min_history <= 0
                ):
                    raise ValueError(f"{model_id} requires a positive integer min_history")
                neural_min_history_values.add(min_history)
            current_lineage = _load_current_training_lineage(
                config,
                neural_min_history_values=neural_min_history_values,
                require_direct_history=require_direct_history,
            )
        except psycopg.Error as exc:
            logger.exception("Loading current production training lineage failed")
            raise HTTPException(
                status_code=500,
                detail="production training readiness check failed",
            ) from exc
        except RuntimeError:
            logger.warning("Current production training lineage is not ready", exc_info=True)
            lineage_unavailable_reason = (
                "Current forecast sales lineage is not ready. Complete a canonical sales load, "
                "then run Forecast Publish to rebuild production artifacts."
            )
        except ValueError as exc:
            logger.exception("Validating production training configuration failed")
            raise HTTPException(
                status_code=500,
                detail="production training readiness check failed",
            ) from exc
    for model_id, algo_info in roster.items():
        if not algo_info.get("forecast"):
            continue
        entry: dict[str, Any] = {
            "model_id": model_id,
            "type": algo_info.get("type", "unknown"),
        }
        if model_id in DIRECT_INFERENCE_MODEL_IDS:
            if current_lineage is None:
                mark_not_trained(entry, stale_reason=lineage_unavailable_reason)
                result[model_id] = entry
                continue
            try:
                config_snapshot = build_backtest_config_snapshot(config, model_id)
                runtime_contract = direct_model_runtime_contract(model_id)
            except (OSError, RuntimeError, TypeError, ValueError):
                logger.warning(
                    "Direct production preflight failed for %s",
                    model_id,
                    exc_info=True,
                )
                mark_not_trained(
                    entry,
                    stale_reason=(
                        f"The current {model_id} direct-inference configuration or runtime "
                        "cannot be proven. Fix the forecast runtime/configuration, then run "
                        "the named Forecast Publish pipeline."
                    ),
                )
            else:
                mark_direct_model_ready(
                    entry,
                    history_end=current_lineage.history_end,
                    source_sales_batch_id=current_lineage.sales.batch_id,
                    config_checksum=config_snapshot.checksum,
                    runtime_contract=runtime_contract,
                )
            result[model_id] = entry
            continue
        if model_id == "lgbm_cluster":
            if current_lineage is None:
                mark_not_trained(entry, stale_reason=lineage_unavailable_reason)
                result[model_id] = entry
                continue
            if current_lineage.clustering_enabled and current_lineage.clusters is None:
                mark_not_trained(
                    entry,
                    stale_reason=current_lineage.cluster_stale_reason,
                )
                result[model_id] = entry
                continue
            clusters = current_lineage.clusters
            cluster_strategy = "per_cluster" if current_lineage.clustering_enabled else "global"
            cluster_labels = clusters.cluster_labels if clusters else frozenset({"global"})
            try:
                lineage = ProductionTreeArtifactLineage(
                    source_sales_batch_id=current_lineage.sales.batch_id,
                    data_checksum=current_lineage.sales.source_hash,
                    history_end=current_lineage.history_end,
                    cluster_experiment_id=(clusters.experiment_id if clusters else None),
                    cluster_assignment_count=(clusters.assignment_count if clusters else None),
                    cluster_assignment_checksum=(
                        clusters.assignment_checksum if clusters else None
                    ),
                    generator_contract_version=GENERATOR_CONTRACT_VERSION,
                )
                expected_spec = build_tree_artifact_spec(
                    model_id=model_id,
                    model_config=build_production_tree_model_config_payload(
                        config,
                        model_id=model_id,
                        project_root=_PROJECT_ROOT,
                    ),
                    lineage=lineage,
                    cluster_strategy=cluster_strategy,
                    cluster_labels=cluster_labels,
                )
                artifact = read_active_tree_artifact_ref(
                    model_id=model_id,
                    base_dir=model_base_dir,
                    expected_spec=expected_spec,
                )
            except FileNotFoundError:
                mark_not_trained(entry, stale_reason=missing_artifact_reason(model_id))
            except TreeArtifactLineageMismatchError:
                mark_not_trained(entry, stale_reason=retrain_reason(model_id))
            except (OSError, RuntimeError, ValueError):
                logger.warning(
                    "Ignoring invalid active LightGBM artifact set",
                    exc_info=True,
                )
                mark_not_trained(entry, stale_reason=invalid_artifact_reason(model_id))
            else:
                metadata = artifact.metadata
                lineage = metadata["lineage"]
                training_metadata = metadata["training_metadata"]
                entry.update(
                    trained=True,
                    ready=True,
                    trained_at=metadata["trained_at"],
                    training_mode="production",
                    n_dfus=training_metadata.get("n_dfus"),
                    planning_date=lineage["history_end"],
                    artifact_id=artifact.artifact_set_id,
                )
            result[model_id] = entry
            continue
        if model_id in SUPPORTED_NEURAL_MODELS:
            if current_lineage is None:
                mark_not_trained(entry, stale_reason=lineage_unavailable_reason)
                result[model_id] = entry
                continue
            params = algo_info.get("params")
            min_history = params.get("min_history") if isinstance(params, Mapping) else None
            cohort = (
                current_lineage.neural_cohorts.get(min_history)
                if isinstance(min_history, int) and not isinstance(min_history, bool)
                else None
            )
            if cohort is None:
                mark_not_trained(
                    entry,
                    stale_reason=(
                        current_lineage.neural_cohort_stale_reason
                        or "The current neural training cohort cannot be proven. Run Forecast "
                        "Publish to rebuild neural production artifacts."
                    ),
                )
                result[model_id] = entry
                continue
            try:
                artifact = read_active_neural_artifact_ref(
                    model_id=model_id,
                    base_dir=model_base_dir,
                    expected_params=algo_info.get("params"),
                    expected_source_sales_batch_id=current_lineage.sales.batch_id,
                    expected_data_checksum=current_lineage.sales.source_hash,
                    expected_history_end=current_lineage.history_end,
                    expected_training_cohort_checksum=cohort.checksum,
                    expected_training_dfu_count=cohort.dfu_count,
                    generator_contract_version=GENERATOR_CONTRACT_VERSION,
                )
            except FileNotFoundError:
                mark_not_trained(entry, stale_reason=missing_artifact_reason(model_id))
            except NeuralArtifactLineageMismatchError:
                mark_not_trained(entry, stale_reason=retrain_reason(model_id))
            except (OSError, RuntimeError, ValueError):
                logger.warning(
                    "Ignoring invalid active neural artifact for %s",
                    model_id,
                    exc_info=True,
                )
                mark_not_trained(entry, stale_reason=invalid_artifact_reason(model_id))
            else:
                metadata = artifact.metadata
                entry.update(
                    trained=True,
                    ready=True,
                    trained_at=metadata["trained_at"],
                    training_mode="production",
                    n_dfus=metadata["training_dfu_count"],
                    planning_date=metadata["history_end"],
                    artifact_id=artifact.artifact_id,
                )
            result[model_id] = entry
            continue
        mark_not_trained(
            entry,
            stale_reason=(
                f"Unsupported production readiness contract for {model_id}. "
                "Keep only the canonical five forecast models."
            ),
        )
        result[model_id] = entry
    return result


@router.get("/snapshot-roster-readiness")
def get_snapshot_roster_readiness() -> dict[str, Any]:
    """Validate the current champion-plus-top-three publish prerequisite."""
    return evaluate_snapshot_roster_readiness(
        get_conn=get_conn,
        get_planning_date=get_planning_date,
        validate_ready_snapshot_contender=validate_ready_snapshot_contender,
        validate_active_champion=validate_active_champion_readiness,
    )


@router.post("/{model_id}/train", status_code=201, dependencies=[Depends(require_api_key)])
def submit_production_training(model_id: str):
    """Submit a production model training job.

    ``model_id='all'`` trains every retained model with a persisted production
    artifact: LightGBM, N-HiTS, and N-BEATS. MSTL and Chronos 2E infer directly.
    """
    is_all = model_id == "all"
    if not is_all:
        roster = get_algorithm_roster()
        algo_info = roster.get(model_id)
        if algo_info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown model_id '{model_id}'. Valid models: {sorted(roster.keys())}",
            )
        if model_id not in PRODUCTION_TRAINABLE_MODEL_IDS:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_id}' (type={algo_info.get('type')}) does not require training. "
                "Trainable production models are LightGBM, N-HiTS, and N-BEATS.",
            )

    try:
        from common.services.job_registry import JobManager

        jm = JobManager()
        job_id = jm.submit_job(
            job_type="train_production_model",
            params={"model_id": "" if is_all else model_id, "all_models": is_all},
            label=f"Train Production: {'Required Models' if is_all else model_id}",
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
                d = _serialize_backtest_run(row)
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
        entry["current_accuracy"] = (disk_meta.get("accuracy_pct") if disk_meta else None) or (
            latest_runs[model_id]["accuracy_pct"] if model_id in latest_runs else None
        )
        entry["current_wape"] = (disk_meta.get("wape") if disk_meta else None) or (
            latest_runs[model_id]["wape"] if model_id in latest_runs else None
        )

        result[model_id] = entry

    return result


@router.get("/{model_id}/runs")
def get_model_runs(model_id: str):
    """List all backtest runs for a model, newest first."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Auto-fix stale rows before returning
            cur.execute(
                """
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
            """,
                (model_id,),
            )
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

    return [_serialize_backtest_run(row) for row in rows]


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


def _release_queued_run(run_id: int) -> None:
    """Mark a never-dispatched backtest_run failed so it can't block future runs.

    Best-effort: called when ``submit_job`` fails after the tracking row was
    already committed. Only flips rows still at 'queued' (never clobbers a row a
    racing job may have moved to 'running').
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE backtest_run SET status = 'failed', completed_at = NOW() "
                "WHERE id = %s AND status = 'queued'",
                (run_id,),
            )
            conn.commit()
    except psycopg.Error:
        logger.warning("Failed to release queued backtest_run %d after submit failure", run_id)


@router.post("/{model_id}/run", status_code=201, dependencies=[Depends(require_api_key)])
def submit_backtest_run(model_id: str, response: Response, parallel: bool = False):
    """Submit a backtest job for the given model.

    Validates model_id exists in pipeline config, maps to the correct job type,
    inserts a tracking row into backtest_run, and submits the job.

    Concurrency is non-blocking by design — a submission is never rejected:
      - If THIS model already has a run queued or running, no duplicate is
        started; the response is ``status="already_running"`` (HTTP 200) so the
        UI can show a calm "already in progress" note instead of an error.
      - Otherwise the job is submitted. When another backtest is active it simply
        queues — the JobManager serialises per group — and the response is
        ``status="queued"`` (HTTP 201).
      - ``parallel=False`` (default): the job uses the shared ``backtest`` group, so
        backtests run one at a time (extra submissions queue and run sequentially).
      - ``parallel=True``: the job uses a per-job-type group, so DIFFERENT model
        families run concurrently (bounded by the scheduler's worker pool). Each
        model writes its own output dir, so sequential same-family runs never
        clobber each other.
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

    # Don't pile up duplicates: if THIS model already has a run in flight, return
    # the existing job instead of starting another. This is informational, not an
    # error — the caller gets status="already_running" and a friendly message.
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT job_id FROM backtest_run "
                "WHERE model_id = %s AND status IN ('queued', 'running') "
                "ORDER BY created_at DESC LIMIT 1",
                (model_id,),
            )
            existing = cur.fetchone()
    except psycopg.Error:
        logger.exception("In-flight check failed for %s", model_id)
        raise HTTPException(status_code=500, detail="Failed to check running jobs") from None
    if existing:
        response.status_code = 200
        return {
            "run_id": None,
            "job_id": existing[0],
            "model_id": model_id,
            "status": "already_running",
            "message": f"A backtest for {model_id} is already in progress.",
        }

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
    except psycopg.Error:
        logger.exception("Failed to insert backtest_run for %s", model_id)
        raise HTTPException(status_code=500, detail="Failed to create backtest run") from None

    # Submit job via JobManager. If another backtest is active, submit_job queues
    # this one (FIFO per group) rather than failing — so concurrency just delays,
    # never blocks. If the submit fails for ANY reason, the `finally` releases the
    # queued tracking row: the in-flight check above keys on status IN
    # ('queued','running'), so a stranded 'queued' row would otherwise lock this
    # model's run endpoint out permanently.
    from common.services.job_registry import JobManager

    jm = JobManager()
    job_id: str | None = None
    try:
        job_id = jm.submit_job(
            job_type=job_type,
            params={"backtest_run_id": run_id, "model_id": model_id},
            label=f"Backtest: {model_id}",
            # Per-job-type group → different families run in parallel; the shared
            # default "backtest" group → strictly sequential.
            group_override=(job_type if parallel else None),
        )
    except ValueError:
        logger.exception("Failed to submit backtest job for %s", model_id)
        raise HTTPException(status_code=400, detail="Invalid backtest job configuration") from None
    except psycopg.Error:
        logger.exception("Failed to submit backtest job for %s", model_id)
        raise HTTPException(status_code=500, detail="Failed to submit backtest run") from None
    finally:
        if job_id is None:
            _release_queued_run(run_id)

    # Store job_id on the tracking row
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE backtest_run SET job_id = %s WHERE id = %s",
                (job_id, run_id),
            )
            conn.commit()
    except psycopg.Error:
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
