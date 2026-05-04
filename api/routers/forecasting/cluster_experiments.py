"""Cluster Experimentation Studio API router.

Provides full CRUD lifecycle for cluster experiments: create, list, compare,
promote, and track downstream usage by algorithm tuning experiments.

All endpoints live under the /cluster-experiments prefix.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse, Response as FastAPIResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from api.core import get_conn, set_cache
from api.auth import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cluster-experiments", tags=["cluster-experiments"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TEMPLATES_PATH = _PROJECT_ROOT / "config" / "cluster_experiment_templates.yaml"
_SCENARIOS_DIR = Path("/tmp/clustering_scenarios")

# Cache TTLs (seconds)
_LIST_CACHE_TTL = 30
_COMPARE_CACHE_TTL = 60
_TEMPLATE_CACHE_TTL = 300
_DETAIL_CACHE_TTL = 30


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class FeatureParams(BaseModel):
    """Feature generation parameters for cluster experiments."""
    time_window_months: int | str = 24
    min_months_history: int = 1

    @field_validator("time_window_months")
    @classmethod
    def _validate_time_window(cls, v: int | str) -> int | str:
        if isinstance(v, str) and v != "all":
            raise ValueError("time_window_months must be a positive integer or 'all'")
        if isinstance(v, int) and not (1 <= v <= 120):
            raise ValueError("time_window_months must be between 1 and 120")
        return v


class ModelParams(BaseModel):
    """KMeans model parameters for cluster experiments."""
    k_range: list[int] = [3, 12]
    min_cluster_size_pct: float = Field(default=2.0, ge=0.0, lt=50.0)
    use_pca: bool = False
    pca_components: int | None = None
    all_features: bool = False

    @model_validator(mode="after")
    def _validate_k_range(self) -> "ModelParams":
        kr = self.k_range
        if len(kr) != 2:
            raise ValueError("k_range must have exactly 2 elements: [min_k, max_k]")
        if kr[0] < 2:
            raise ValueError("k_range min must be >= 2")
        if kr[0] >= kr[1]:
            raise ValueError("k_range[0] must be less than k_range[1]")
        return self


class LabelParams(BaseModel):
    """Cluster labeling threshold parameters."""
    volume_high: float = Field(default=0.75, ge=0.0, le=1.0)
    volume_low: float = Field(default=0.25, ge=0.0, le=1.0)
    cv_steady: float = Field(default=0.3, ge=0.0)
    cv_volatile: float = Field(default=0.8, ge=0.0)
    seasonality_threshold: float = Field(default=0.5, ge=0.0)
    zero_demand_threshold: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_volume_ordering(self) -> "LabelParams":
        if self.volume_low >= self.volume_high:
            raise ValueError("volume_low must be less than volume_high")
        return self


class CreateExperimentBody(BaseModel):
    """Request body for POST /cluster-experiments."""
    label: str = Field(min_length=1, max_length=200)
    notes: str | None = None
    template: str | None = None
    feature_params: FeatureParams | None = None
    model_params: ModelParams | None = None
    label_params: LabelParams | None = None


class UpdateExperimentBody(BaseModel):
    """Request body for PATCH /cluster-experiments/{experiment_id}."""
    label: str | None = Field(default=None, min_length=1, max_length=200)
    notes: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(val: Any) -> Any:
    """Parse JSON from a DB value that may be a string, dict, list, or None."""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return val


def _experiment_row_to_dict(row: tuple) -> dict[str, Any]:
    """Convert a cluster_experiment row (25 columns) to a response dict.

    Column order matches the SELECT used in list and detail queries.
    """
    return {
        "experiment_id": row[0],
        "scenario_id": row[1],
        "label": row[2],
        "notes": row[3],
        "template_id": row[4],
        "status": row[5],
        "created_at": str(row[6]) if row[6] else None,
        "started_at": str(row[7]) if row[7] else None,
        "completed_at": str(row[8]) if row[8] else None,
        "runtime_seconds": float(row[9]) if row[9] is not None else None,
        "job_id": row[10],
        "feature_params": _parse_json(row[11]),
        "model_params": _parse_json(row[12]),
        "label_params": _parse_json(row[13]),
        "optimal_k": int(row[14]) if row[14] is not None else None,
        "silhouette_score": float(row[15]) if row[15] is not None else None,
        "inertia": float(row[16]) if row[16] is not None else None,
        "total_dfus": int(row[17]) if row[17] is not None else None,
        "n_clusters": int(row[18]) if row[18] is not None else None,
        "cluster_sizes": _parse_json(row[19]),
        "profiles": _parse_json(row[20]),
        "k_selection_results": _parse_json(row[21]),
        "is_promoted": bool(row[22]),
        "promoted_at": str(row[23]) if row[23] else None,
        "artifacts_path": row[24],
    }


_SELECT_COLS = """
    experiment_id, scenario_id, label, notes, template_id,
    status, created_at, started_at, completed_at, runtime_seconds,
    job_id, feature_params, model_params, label_params,
    optimal_k, silhouette_score, inertia, total_dfus, n_clusters,
    cluster_sizes, profiles, k_selection_results,
    is_promoted, promoted_at, artifacts_path
"""


# ---------------------------------------------------------------------------
# 1. GET /cluster-experiments — List experiments
# ---------------------------------------------------------------------------

@router.get("")
def list_experiments(
    response: FastAPIResponse,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    status: str = Query(default="", max_length=20),
):
    """List cluster experiments with pagination, newest first."""
    set_cache(response, max_age=_LIST_CACHE_TTL)

    parts: list[str] = []
    params: list[Any] = []
    if status.strip():
        parts.append("status = %s")
        params.append(status.strip())

    where_sql = f"WHERE {' AND '.join(parts)}" if parts else ""

    count_sql = f"SELECT count(*) FROM cluster_experiment {where_sql}"
    data_sql = f"""
        SELECT {_SELECT_COLS}
        FROM cluster_experiment
        {where_sql}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(count_sql, list(params))
            total = cur.fetchone()[0]

            cur.execute(data_sql, [*params, limit, offset])
            rows = cur.fetchall()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to list cluster experiments")
        raise HTTPException(status_code=500, detail="Failed to list cluster experiments")

    experiments = [_experiment_row_to_dict(r) for r in rows]

    return {
        "experiments": experiments,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


# ---------------------------------------------------------------------------
# 8. GET /cluster-experiments/templates — must be before {experiment_id}
# ---------------------------------------------------------------------------

@router.get("/templates")
def get_templates(response: FastAPIResponse):
    """Load cluster experiment templates from config YAML.

    The ``production_baseline`` template is enriched with the currently
    promoted experiment's actual parameters so the UI shows the real
    production config instead of hardcoded defaults.
    """
    set_cache(response, max_age=_TEMPLATE_CACHE_TTL)

    if not _TEMPLATES_PATH.exists():
        return {"templates": []}

    try:
        with open(_TEMPLATES_PATH) as f:
            config = yaml.safe_load(f)
        templates = config.get("templates", []) if config else []
    except (yaml.YAMLError, OSError):
        logger.exception("Failed to load cluster experiment templates")
        raise HTTPException(status_code=500, detail="Failed to load templates")

    # Resolve production_baseline from currently promoted experiment
    promoted_params = _get_promoted_params()
    if promoted_params:
        for tmpl in templates:
            if tmpl.get("id") == "production_baseline":
                tmpl["feature_params"] = promoted_params.get("feature_params")
                tmpl["model_params"] = promoted_params.get("model_params")
                tmpl["label_params"] = promoted_params.get("label_params")
                break

    return {"templates": templates}


def _get_promoted_params() -> dict | None:
    """Fetch feature/model/label params from the currently promoted experiment."""
    sql = """
        SELECT feature_params, model_params, label_params
        FROM cluster_experiment
        WHERE is_promoted = TRUE
        ORDER BY promoted_at DESC
        LIMIT 1
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
            if not row:
                return None
            return {
                "feature_params": _parse_json(row[0]),
                "model_params": _parse_json(row[1]),
                "label_params": _parse_json(row[2]),
            }
    except Exception:
        logger.exception("Failed to fetch promoted experiment params")
        return None


# ---------------------------------------------------------------------------
# 9. GET /cluster-experiments/completed — completed experiments only
# ---------------------------------------------------------------------------

@router.get("/completed")
def list_completed_experiments(response: FastAPIResponse):
    """List only completed cluster experiments (for algorithm experiment dropdown)."""
    set_cache(response, max_age=_LIST_CACHE_TTL)

    sql = f"""
        SELECT {_SELECT_COLS}
        FROM cluster_experiment
        WHERE status = 'completed'
        ORDER BY created_at DESC
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    except Exception:
        logger.exception("Failed to list completed cluster experiments")
        raise HTTPException(status_code=500, detail="Failed to list completed experiments")

    experiments = [_experiment_row_to_dict(r) for r in rows]
    return {"experiments": experiments}


# ---------------------------------------------------------------------------
# 7. GET /cluster-experiments/compare — Compare two experiments
# ---------------------------------------------------------------------------

@router.get("/compare")
def compare_experiments(
    response: FastAPIResponse,
    a_id: int = Query(..., description="First experiment ID"),
    b_id: int = Query(..., description="Second experiment ID"),
):
    """Compare two cluster experiments: quality metrics, profiles, migration matrix.

    Uses a comparison cache table. If not cached, computes migration matrix
    from both experiments' cluster_labels.csv files via pandas crosstab.
    """
    set_cache(response, max_age=_COMPARE_CACHE_TTL)

    if a_id == b_id:
        raise HTTPException(status_code=400, detail="Cannot compare an experiment with itself")

    # Fetch both experiments
    sql = f"""
        SELECT {_SELECT_COLS}
        FROM cluster_experiment
        WHERE experiment_id IN (%s, %s)
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [a_id, b_id])
            rows = cur.fetchall()
    except Exception:
        logger.exception("Failed to fetch experiments for comparison")
        raise HTTPException(status_code=500, detail="Failed to fetch experiments")

    if len(rows) < 2:
        raise HTTPException(status_code=404, detail="One or both experiments not found")

    exp_map = {}
    for r in rows:
        d = _experiment_row_to_dict(r)
        exp_map[d["experiment_id"]] = d

    if a_id not in exp_map or b_id not in exp_map:
        raise HTTPException(status_code=404, detail="One or both experiments not found")

    exp_a = exp_map[a_id]
    exp_b = exp_map[b_id]

    # Check cache
    cache_sql = """
        SELECT migration_matrix, quality_comparison, profile_comparison
        FROM cluster_experiment_comparison
        WHERE experiment_a_id = %s AND experiment_b_id = %s
    """
    cached = None
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(cache_sql, [a_id, b_id])
            cached = cur.fetchone()
    except Exception:
        logger.warning("Failed to check comparison cache", exc_info=True)

    if cached is not None:
        migration = _parse_json(cached[0]) or {}
        total_migrated = 0
        total_unchanged = 0
        for src, targets in migration.items():
            if isinstance(targets, dict):
                for tgt, count in targets.items():
                    if src == tgt:
                        total_unchanged += count
                    else:
                        total_migrated += count
        return {
            "experiment_a": exp_a,
            "experiment_b": exp_b,
            "migration_matrix": migration,
            "quality_comparison": _parse_json(cached[1]),
            "profile_comparison": _parse_json(cached[2]),
            "total_dfus_migrated": total_migrated,
            "total_dfus_unchanged": total_unchanged,
        }

    # Compute comparison
    quality_comparison = _compute_quality_comparison(exp_a, exp_b)
    profile_comparison = _compute_profile_comparison(exp_a, exp_b)
    migration_matrix, total_migrated, total_unchanged = _compute_migration_matrix(exp_a, exp_b)

    # Cache the result
    cache_insert_sql = """
        INSERT INTO cluster_experiment_comparison
            (experiment_a_id, experiment_b_id, migration_matrix,
             quality_comparison, profile_comparison)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (experiment_a_id, experiment_b_id) DO UPDATE
        SET migration_matrix = EXCLUDED.migration_matrix,
            quality_comparison = EXCLUDED.quality_comparison,
            profile_comparison = EXCLUDED.profile_comparison,
            created_at = NOW()
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(cache_insert_sql, [
                a_id, b_id,
                json.dumps(migration_matrix),
                json.dumps(quality_comparison),
                json.dumps(profile_comparison),
            ])
            conn.commit()
    except Exception:
        logger.warning("Failed to cache comparison result", exc_info=True)

    return {
        "experiment_a": exp_a,
        "experiment_b": exp_b,
        "quality_comparison": quality_comparison,
        "profile_comparison": profile_comparison,
        "migration_matrix": migration_matrix,
        "total_dfus_migrated": total_migrated,
        "total_dfus_unchanged": total_unchanged,
    }


def _compute_quality_comparison(
    exp_a: dict[str, Any], exp_b: dict[str, Any],
) -> dict[str, Any]:
    """Compute quality metric deltas between two experiments."""
    sil_a = exp_a.get("silhouette_score")
    sil_b = exp_b.get("silhouette_score")
    inertia_a = exp_a.get("inertia")
    inertia_b = exp_b.get("inertia")
    k_a = exp_a.get("optimal_k")
    k_b = exp_b.get("optimal_k")

    sil_delta = round(sil_b - sil_a, 6) if sil_a is not None and sil_b is not None else None
    inertia_delta = round(inertia_b - inertia_a, 2) if inertia_a is not None and inertia_b is not None else None
    k_delta = k_b - k_a if k_a is not None and k_b is not None else None

    # Determine verdict
    verdict = "mixed"
    if sil_delta is not None:
        if sil_delta > 0.01:
            verdict = "b_better"
        elif sil_delta < -0.01:
            verdict = "a_better"

    return {
        "silhouette_delta": sil_delta,
        "inertia_delta": inertia_delta,
        "k_delta": k_delta,
        "verdict": verdict,
    }


def _compute_profile_comparison(
    exp_a: dict[str, Any], exp_b: dict[str, Any],
) -> dict[str, Any]:
    """Compare cluster profile distributions between two experiments."""
    profiles_a = exp_a.get("profiles") or []
    profiles_b = exp_b.get("profiles") or []

    # Extract cluster labels and counts from profiles
    labels_a: dict[str, int] = {}
    labels_b: dict[str, int] = {}

    for p in profiles_a:
        if isinstance(p, dict):
            label = p.get("label", p.get("cluster_label", ""))
            count = p.get("count", p.get("dfu_count", 0))
            if label:
                labels_a[label] = int(count)

    for p in profiles_b:
        if isinstance(p, dict):
            label = p.get("label", p.get("cluster_label", ""))
            count = p.get("count", p.get("dfu_count", 0))
            if label:
                labels_b[label] = int(count)

    all_labels_a = set(labels_a.keys())
    all_labels_b = set(labels_b.keys())
    common = all_labels_a & all_labels_b
    only_a = sorted(all_labels_a - common)
    only_b = sorted(all_labels_b - common)

    common_clusters = []
    for label in sorted(common):
        common_clusters.append({
            "label": label,
            "count_a": labels_a[label],
            "count_b": labels_b[label],
        })

    return {
        "clusters_only_in_a": only_a,
        "clusters_only_in_b": only_b,
        "common_clusters": common_clusters,
    }


def _compute_migration_matrix(
    exp_a: dict[str, Any], exp_b: dict[str, Any],
) -> tuple[dict[str, dict[str, int]], int, int]:
    """Compute DFU migration matrix from cluster_labels.csv files.

    Returns (matrix, total_migrated, total_unchanged).
    """
    scenario_a = exp_a.get("scenario_id", "")
    scenario_b = exp_b.get("scenario_id", "")

    labels_a_path = _SCENARIOS_DIR / scenario_a / "cluster_labels.csv"
    labels_b_path = _SCENARIOS_DIR / scenario_b / "cluster_labels.csv"

    if not labels_a_path.exists() or not labels_b_path.exists():
        return {}, 0, 0

    try:
        import pandas as pd

        df_a = pd.read_csv(labels_a_path, usecols=["sku_ck", "cluster_label"])
        df_b = pd.read_csv(labels_b_path, usecols=["sku_ck", "cluster_label"])

        merged = df_a.merge(df_b, on="sku_ck", suffixes=("_a", "_b"))

        if merged.empty:
            return {}, 0, 0

        ct = pd.crosstab(merged["cluster_label_a"], merged["cluster_label_b"])

        matrix: dict[str, dict[str, int]] = {}
        for idx_label in ct.index:
            matrix[str(idx_label)] = {}
            for col_label in ct.columns:
                count = int(ct.loc[idx_label, col_label])
                if count > 0:
                    matrix[str(idx_label)][str(col_label)] = count

        total_unchanged = 0
        total_migrated = 0
        for src, targets in matrix.items():
            for tgt, count in targets.items():
                if src == tgt:
                    total_unchanged += count
                else:
                    total_migrated += count

        return matrix, total_migrated, total_unchanged
    except ImportError:
        logger.warning("pandas not available for migration matrix computation")
        return {}, 0, 0
    except Exception:
        logger.exception("Failed to compute migration matrix")
        return {}, 0, 0


# ---------------------------------------------------------------------------
# 2. GET /cluster-experiments/{experiment_id} — Single experiment detail
# ---------------------------------------------------------------------------

@router.get("/{experiment_id}")
def get_experiment(experiment_id: int, response: FastAPIResponse):
    """Get full detail for a single cluster experiment."""
    set_cache(response, max_age=_DETAIL_CACHE_TTL)

    sql = f"""
        SELECT {_SELECT_COLS}
        FROM cluster_experiment
        WHERE experiment_id = %s
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [experiment_id])
            row = cur.fetchone()
    except Exception:
        logger.exception("Failed to get cluster experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Failed to fetch experiment")

    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return _experiment_row_to_dict(row)


# ---------------------------------------------------------------------------
# 3. POST /cluster-experiments — Create + launch experiment
# ---------------------------------------------------------------------------

@router.post("", dependencies=[Depends(require_api_key)])
def create_experiment(body: CreateExperimentBody):
    """Create a new cluster experiment and launch it as a background job.

    Generates a scenario_id, inserts a DB record with status='queued',
    then submits a cluster_scenario job via JobManager.
    Returns 202 with experiment_id, scenario_id, status, job_id.
    """
    from scripts.ml.run_clustering_scenario import generate_scenario_id

    scenario_id = generate_scenario_id()

    feature_json = json.dumps(body.feature_params.model_dump()) if body.feature_params else None
    model_json = json.dumps(body.model_params.model_dump()) if body.model_params else None
    label_json = json.dumps(body.label_params.model_dump()) if body.label_params else None

    insert_sql = """
        INSERT INTO cluster_experiment
            (scenario_id, label, notes, template_id, status,
             feature_params, model_params, label_params)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING experiment_id
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(insert_sql, [
                scenario_id,
                body.label,
                body.notes,
                body.template,
                "queued",
                feature_json,
                model_json,
                label_json,
            ])
            row = cur.fetchone()
            conn.commit()
    except Exception:
        logger.exception("Failed to create cluster experiment")
        raise HTTPException(status_code=500, detail="Failed to create cluster experiment")

    experiment_id = row[0]

    # Build params for the cluster_scenario job
    job_params: dict[str, Any] = {
        "scenario_id": scenario_id,
        "experiment_id": experiment_id,
        "feature_params": body.feature_params.model_dump() if body.feature_params else None,
        "model_params": body.model_params.model_dump() if body.model_params else None,
        "label_params": body.label_params.model_dump() if body.label_params else None,
    }

    # Submit job via JobManager
    job_id: str | None = None
    try:
        from common.services.job_registry import JobManager
        mgr = JobManager()
        job_id = mgr.submit_job(
            job_type="cluster_scenario",
            params=job_params,
            label=f"Cluster Experiment — {body.label}",
        )

        # Store job_id on the experiment record
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE cluster_experiment SET job_id = %s WHERE experiment_id = %s",
                [job_id, experiment_id],
            )
            conn.commit()
    except ValueError as exc:
        logger.warning("Job submission failed for experiment %d: %s", experiment_id, exc)
    except Exception:
        logger.exception("Failed to submit job for cluster experiment %d", experiment_id)

    return JSONResponse(
        status_code=202,
        content={
            "experiment_id": experiment_id,
            "scenario_id": scenario_id,
            "status": "queued",
            "job_id": job_id,
        },
    )


# ---------------------------------------------------------------------------
# 4. PATCH /cluster-experiments/{experiment_id} — Update label/notes
# ---------------------------------------------------------------------------

@router.patch("/{experiment_id}", dependencies=[Depends(require_api_key)])
def update_experiment(experiment_id: int, body: UpdateExperimentBody):
    """Update the label and/or notes of an existing cluster experiment."""
    updates: list[str] = []
    params: list[Any] = []

    if body.label is not None:
        updates.append("label = %s")
        params.append(body.label)
    if body.notes is not None:
        updates.append("notes = %s")
        params.append(body.notes)

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    params.append(experiment_id)
    sql = f"""
        UPDATE cluster_experiment
        SET {', '.join(updates)}
        WHERE experiment_id = %s
        RETURNING experiment_id
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            conn.commit()
    except Exception:
        logger.exception("Failed to update cluster experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Failed to update experiment")

    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    return {"experiment_id": experiment_id, "updated": True}


# ---------------------------------------------------------------------------
# 5. DELETE /cluster-experiments/{experiment_id}
# ---------------------------------------------------------------------------

@router.delete("/{experiment_id}", dependencies=[Depends(require_api_key)])
def delete_experiment(experiment_id: int):
    """Delete a cluster experiment. Returns 409 if running, queued, or promoted."""
    # Check current status and promotion flag
    status_sql = "SELECT status, is_promoted FROM cluster_experiment WHERE experiment_id = %s"

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(status_sql, [experiment_id])
            row = cur.fetchone()
    except Exception:
        logger.exception("Failed to check experiment status for delete")
        raise HTTPException(status_code=500, detail="Failed to check experiment status")

    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    current_status = row[0]
    is_promoted = bool(row[1])
    if is_promoted:
        raise HTTPException(
            status_code=409,
            detail="Cannot delete the promoted production cluster config.",
        )
    if current_status in ("running", "queued"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete experiment with status '{current_status}'. Cancel it first.",
        )

    # Check if any algorithm tuning runs reference this cluster experiment
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM lgbm_tuning_run WHERE cluster_experiment_id = %s",
                [experiment_id],
            )
            ref_count = cur.fetchone()[0]
            if ref_count > 0:
                raise HTTPException(
                    status_code=409,
                    detail=f"Cannot delete: {ref_count} algorithm tuning experiment(s) "
                    f"reference this cluster. Remove those references first.",
                )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to check tuning run references for experiment %d", experiment_id)

    # Delete — cascade removes comparison cache entries
    delete_sql = "DELETE FROM cluster_experiment WHERE experiment_id = %s"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(delete_sql, [experiment_id])
            conn.commit()
    except Exception:
        logger.exception("Failed to delete cluster experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Failed to delete experiment")

    return {"deleted": True, "experiment_id": experiment_id}


# ---------------------------------------------------------------------------
# 6. POST /cluster-experiments/{experiment_id}/promote
# ---------------------------------------------------------------------------

@router.post("/{experiment_id}/promote", dependencies=[Depends(require_api_key)])
def promote_experiment(experiment_id: int):
    """Promote a completed cluster experiment to production.

    Verifies status='completed', clears previous is_promoted flags,
    calls promote_scenario() to update dim_sku.ml_cluster, then sets
    is_promoted=TRUE and promoted_at=NOW() on this experiment.
    """
    # Fetch the experiment
    sql = f"""
        SELECT {_SELECT_COLS}
        FROM cluster_experiment
        WHERE experiment_id = %s
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [experiment_id])
            row = cur.fetchone()
    except Exception:
        logger.exception("Failed to fetch experiment %d for promotion", experiment_id)
        raise HTTPException(status_code=500, detail="Failed to fetch experiment")

    if row is None:
        raise HTTPException(status_code=404, detail="Experiment not found")

    exp = _experiment_row_to_dict(row)

    if exp["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Cannot promote experiment with status '{exp['status']}'. Must be 'completed'.",
        )

    scenario_id = exp["scenario_id"]

    # Promote via the existing clustering scenario promote logic
    try:
        from scripts.ml.run_clustering_scenario import promote_scenario
        result = promote_scenario(scenario_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario artifacts not found for '{scenario_id}'",
        )
    except Exception:
        logger.exception("Failed to promote cluster experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Promotion failed. Check server logs.")

    # Clear previous promoted flags and set this one
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE cluster_experiment SET is_promoted = FALSE, promoted_at = NULL "
                "WHERE is_promoted = TRUE AND experiment_id != %s",
                [experiment_id],
            )
            cur.execute(
                "UPDATE cluster_experiment SET is_promoted = TRUE, promoted_at = NOW() "
                "WHERE experiment_id = %s",
                [experiment_id],
            )
            conn.commit()
    except Exception:
        logger.exception("Failed to update promotion flags for experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Promotion succeeded but flag update failed")

    dfus_updated = result.get("dfus_updated", 0) if isinstance(result, dict) else 0

    return {
        "status": "promoted",
        "experiment_id": experiment_id,
        "scenario_id": scenario_id,
        "dfus_updated": dfus_updated,
    }


# ---------------------------------------------------------------------------
# 10. GET /cluster-experiments/{experiment_id}/used-by
# ---------------------------------------------------------------------------

@router.get("/{experiment_id}/used-by")
def get_experiment_used_by(experiment_id: int, response: FastAPIResponse):
    """List algorithm tuning experiments that reference this cluster experiment."""
    set_cache(response, max_age=_LIST_CACHE_TTL)

    sql = """
        SELECT run_id, run_label, model_id, status, accuracy_pct, started_at
        FROM lgbm_tuning_run
        WHERE cluster_experiment_id = %s
        ORDER BY started_at DESC
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [experiment_id])
            rows = cur.fetchall()
    except Exception:
        logger.exception("Failed to query used-by for experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Failed to query downstream usage")

    runs = [
        {
            "run_id": r[0],
            "run_label": r[1],
            "model_id": r[2],
            "status": r[3],
            "accuracy_pct": float(r[4]) if r[4] is not None else None,
            "started_at": str(r[5]) if r[5] else None,
        }
        for r in rows
    ]

    return {"experiment_id": experiment_id, "runs": runs}
