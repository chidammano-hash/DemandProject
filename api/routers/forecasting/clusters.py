"""DFU clustering + what-if scenario endpoints (features 7, 29, 38)."""
from __future__ import annotations

from typing import Any
import datetime
import json
import logging
import re
import time

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from psycopg import sql as psycopg_sql
from pydantic import BaseModel, Field, field_validator, model_validator

from api.core import get_conn
from api.auth import require_api_key

log = logging.getLogger(__name__)

router = APIRouter(tags=["clustering"])

# Scenario IDs are generated as  sc_YYYYMMDD_HHMMSS_<4hex>  e.g. sc_20250310_142305_a3f1
# Enforcing this pattern at the router level prevents path traversal attacks where an
# attacker could submit  ../../../etc/passwd  as a scenario_id.
_SCENARIO_ID_RE = re.compile(r"^sc_\d{8}_\d{6}_[0-9a-f]{4}$")


def _validate_scenario_id(scenario_id: str) -> str:
    """Raise 400 if scenario_id doesn't match the expected format."""
    if not _SCENARIO_ID_RE.match(scenario_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid scenario_id format. Expected: sc_YYYYMMDD_HHMMSS_<4hex>",
        )
    return scenario_id


# ---------------------------------------------------------------------------
# Cluster summary / profiles / visualization
# ---------------------------------------------------------------------------
@router.get("/domains/sku/clusters")
def dfu_clusters(source: str = Query(default="ml", pattern="^(ml|source)$")):
    """Get cluster summary statistics for DFU clustering.

    source=ml    -> promoted ML clusters (current_sku_cluster_assignment)
    source=source -> original source-file clusters (cluster_assignment column)
    """
    with get_conn() as conn, conn.cursor() as cur:
        if source == "ml":
            cur.execute("""
                WITH cluster_counts AS (
                    SELECT
                        ml_cluster AS cluster_label,
                        COUNT(*) AS dfu_count
                    FROM current_sku_cluster_assignment
                    WHERE ml_cluster IS NOT NULL AND ml_cluster != ''
                    GROUP BY ml_cluster
                ),
                total AS (
                    SELECT SUM(dfu_count) AS total_assigned FROM cluster_counts
                ),
                cluster_demand AS (
                    SELECT
                        ca.ml_cluster AS cluster_label,
                        AVG(s.qty) AS avg_demand,
                        CASE
                            WHEN AVG(s.qty) > 0 THEN COALESCE(STDDEV(s.qty), 0) / AVG(s.qty)
                            ELSE 0
                        END AS cv_demand
                    FROM current_sku_cluster_assignment ca
                    INNER JOIN dim_sku d ON d.sku_ck = ca.sku_ck
                    INNER JOIN fact_sales_monthly s
                        ON s.item_id = d.item_id
                        AND s.customer_group = d.customer_group
                        AND s.loc = d.loc
                    WHERE ca.ml_cluster IS NOT NULL AND ca.ml_cluster != ''
                        AND s.qty IS NOT NULL
                    GROUP BY ca.ml_cluster
                )
                SELECT
                    cc.cluster_label,
                    cc.dfu_count,
                    ROUND(cc.dfu_count * 100.0 / t.total_assigned, 2) AS pct_of_total,
                    COALESCE(cd.avg_demand, 0) AS avg_demand,
                    COALESCE(cd.cv_demand, 0) AS cv_demand
                FROM cluster_counts cc
                CROSS JOIN total t
                LEFT JOIN cluster_demand cd ON cd.cluster_label = cc.cluster_label
                ORDER BY cc.dfu_count DESC
            """)
        else:
            col_id = psycopg_sql.Identifier("cluster_assignment")
            cur.execute(psycopg_sql.SQL("""
                WITH cluster_counts AS (
                    SELECT
                        {col} AS cluster_label,
                        COUNT(*) AS dfu_count
                    FROM dim_sku
                    WHERE {col} IS NOT NULL AND {col} != ''
                    GROUP BY {col}
                ),
                total AS (
                    SELECT SUM(dfu_count) AS total_assigned FROM cluster_counts
                ),
                cluster_demand AS (
                    SELECT
                        d.{col} AS cluster_label,
                        AVG(s.qty) AS avg_demand,
                        CASE
                            WHEN AVG(s.qty) > 0 THEN COALESCE(STDDEV(s.qty), 0) / AVG(s.qty)
                            ELSE 0
                        END AS cv_demand
                    FROM dim_sku d
                    INNER JOIN fact_sales_monthly s
                        ON s.item_id = d.item_id
                        AND s.customer_group = d.customer_group
                        AND s.loc = d.loc
                    WHERE d.{col} IS NOT NULL AND d.{col} != ''
                        AND s.qty IS NOT NULL
                    GROUP BY d.{col}
                )
                SELECT
                    cc.cluster_label,
                    cc.dfu_count,
                    ROUND(cc.dfu_count * 100.0 / t.total_assigned, 2) AS pct_of_total,
                    COALESCE(cd.avg_demand, 0) AS avg_demand,
                    COALESCE(cd.cv_demand, 0) AS cv_demand
                FROM cluster_counts cc
                CROSS JOIN total t
                LEFT JOIN cluster_demand cd ON cd.cluster_label = cc.cluster_label
                ORDER BY cc.dfu_count DESC
            """).format(col=col_id))
        rows = cur.fetchall()

        clusters = []
        total_assigned = 0
        for cluster_label, count, pct, avg_demand, cv_demand in rows:
            total_assigned += int(count)
            clusters.append({
                "cluster_id": cluster_label,
                "label": cluster_label,
                "count": int(count),
                "pct_of_total": float(pct),
                "avg_demand": round(float(avg_demand), 2),
                "cv_demand": round(float(cv_demand), 4),
            })

        return {
            "domain": "sku",
            "source": source,
            "total_assigned": total_assigned,
            "clusters": clusters,
        }


@router.get("/domains/sku/clusters/profiles")
def dfu_cluster_profiles():
    """Get cluster profiles with centroid features and metadata."""
    from common.core.paths import DATA_DIR
    profiles_path = DATA_DIR / "clustering" / "cluster_profiles.json"
    metadata_path = DATA_DIR / "clustering" / "cluster_metadata.json"

    profiles = []
    if profiles_path.exists():
        with open(profiles_path) as f:
            profiles = json.load(f)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return {
        "profiles": profiles,
        "metadata": {
            "optimal_k": metadata.get("optimal_k"),
            "silhouette_score": metadata.get("silhouette_score"),
            "inertia": metadata.get("inertia"),
            "k_selection_results": metadata.get("k_selection_results"),
        },
    }


@router.get("/domains/sku/clusters/visualization/{image_name}")
def dfu_cluster_visualization(image_name: str):
    """Serve clustering visualization images."""
    allowed = {"k_selection_plots.png", "cluster_visualization.png"}
    if image_name not in allowed:
        raise HTTPException(404, f"Image not found: {image_name}")
    from common.core.paths import DATA_DIR
    img_path = DATA_DIR / "clustering" / image_name
    if not img_path.exists():
        raise HTTPException(404, f"Image not generated yet: {image_name}")
    return FileResponse(str(img_path), media_type="image/png")


@router.get("/domains/sku/seasonality-profiles")
def dfu_seasonality_profiles():
    """Get distinct seasonality_profile values with DFU counts."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                COALESCE(seasonality_profile, '(unknown)') AS profile,
                COUNT(*)::bigint AS dfu_count
            FROM dim_sku
            GROUP BY 1
            ORDER BY dfu_count DESC
        """)
        rows = cur.fetchall()
    return {
        "profiles": [
            {"profile": row[0], "count": int(row[1])}
            for row in rows
        ],
    }


# ---------------------------------------------------------------------------
# Core features list (single source: common.ml.clustering.constants)
# Imported from `constants` (lightweight) rather than `training` so the router
# does not pull in matplotlib/sklearn/scipy at app boot.
# ---------------------------------------------------------------------------

from common.ml.clustering.constants import CORE_FEATURES  # noqa: E402


@router.get("/clustering/core-features")
def get_core_features():
    """Return the canonical list of core clustering features."""
    return JSONResponse(
        content={"features": CORE_FEATURES},
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ---------------------------------------------------------------------------
# Clustering what-if scenarios (feature 29)
# ---------------------------------------------------------------------------


class FeatureParams(BaseModel):
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


class ClusteringScenarioRequest(BaseModel):
    feature_params: FeatureParams | None = None
    model_params: ModelParams | None = None
    label_params: LabelParams | None = None
    relabel_only: bool = False
    previous_scenario_id: str | None = None


@router.get("/clustering/defaults")
def get_clustering_defaults():
    """Return current default clustering parameters from promoted experiment or hardcoded defaults."""
    from api.core import get_conn

    _defaults = {
        "feature_params": {
            "time_window_months": 36, "min_months_history": 12,
        },
        "model_params": {
            "k_range": [9, 18], "min_cluster_size_pct": 2.0,
            "use_pca": False, "pca_components": None, "all_features": False,
        },
        "label_params": {
            "volume_high": 0.75, "volume_low": 0.25,
            "cv_steady": 0.4, "cv_volatile": 0.8,
            "seasonality_threshold": 0.3, "zero_demand_threshold": 0.15,
        },
    }

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT feature_params, model_params, label_params "
                "FROM cluster_experiment "
                "WHERE is_promoted = TRUE "
                "ORDER BY promoted_at DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                fp = row[0] if isinstance(row[0], dict) else {}
                mp = row[1] if isinstance(row[1], dict) else {}
                lp = row[2] if isinstance(row[2], dict) else {}
                return {
                    "feature_params": {**_defaults["feature_params"], **fp},
                    "model_params": {**_defaults["model_params"], **mp},
                    "label_params": {**_defaults["label_params"], **lp},
                }
    except Exception:  # noqa: BLE001 — defaults endpoint must never 500; log and fall back
        log.exception("Failed to fetch promoted experiment params for /clustering/defaults")

    return _defaults


@router.get("/clustering/scenario/estimate")
def estimate_scenario_runtime(
    scope: str = Query(default="all"),
    k_min: int = Query(default=3),
    k_max: int = Query(default=12),
):
    """Estimate scenario runtime based on parameters."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM dim_sku")
        dfu_count = int(cur.fetchone()[0])

    k_range = max(1, k_max - k_min + 1)
    max_training_dfus = 20_000  # Pipeline samples if count exceeds this

    # Empirical constants calibrated from observed runs:
    # Feature generation: ~0.001s per DFU (per-DFU loop, runs on ALL DFUs)
    feature_gen_per_sku = 0.001
    # KMeans training: ~0.002s per training DFU per K (runs on sample if large)
    kmeans_per_sku_per_k = 0.002
    # Fixed overhead for SQL query + data loading
    overhead_seconds = 10.0

    training_dfus = min(dfu_count, max_training_dfus)
    feature_gen_time = feature_gen_per_sku * dfu_count
    kmeans_time = kmeans_per_sku_per_k * training_dfus * k_range
    estimated = overhead_seconds + feature_gen_time + kmeans_time

    return {
        "estimated_seconds": round(estimated, 0),
        "dfu_count": dfu_count,
        "training_sample": training_dfus,
        "sampled": dfu_count > max_training_dfus,
        "k_range": k_range,
    }


@router.post("/clustering/scenario", dependencies=[Depends(require_api_key)])
async def run_clustering_scenario(req: ClusteringScenarioRequest):
    """Run a trial clustering pipeline with custom parameters (non-blocking).

    Delegates to the JobManager so the scenario appears in the Jobs tab.
    """
    from common.services.job_registry import JobManager

    manager = JobManager()

    # Build params from the request
    params: dict[str, Any] = {
        "feature_params": req.feature_params.model_dump() if req.feature_params else None,
        "model_params": req.model_params.model_dump() if req.model_params else None,
        "label_params": req.label_params.model_dump() if req.label_params else None,
        "relabel_only": req.relabel_only,
        "previous_scenario_id": req.previous_scenario_id,
    }

    # Embed scenario_id so _run_cluster_scenario uses it
    from scripts.ml.run_clustering_scenario import generate_scenario_id
    scenario_id = generate_scenario_id()
    params["scenario_id"] = scenario_id

    job_id = manager.submit_job("cluster_scenario", params, label="What-If Scenario")

    # Determine if the job is queued or running immediately
    job_status = manager.get_status(job_id)
    is_queued = job_status and job_status.get("status") == "queued"

    return JSONResponse(
        status_code=202,
        content={
            "scenario_id": scenario_id,
            "status": "queued" if is_queued else "running",
            "job_id": job_id,
        },
    )


def _get_job_manager():
    """Lazy import and instantiate JobManager (testable seam)."""
    from common.services.job_registry import JobManager
    return JobManager()


def _find_job_by_scenario_id(scenario_id: str) -> dict[str, Any] | None:
    """Look up a job in job_history DB by scenario_id embedded in params."""
    try:
        manager = _get_job_manager()
        # Search recent jobs (active + completed). 500 covers ~months of job history.
        rows, _ = manager.list_jobs(limit=500)
        for job in rows:
            params = job.get("params")
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    continue
            if params and params.get("scenario_id") == scenario_id:
                return job
    except Exception:  # noqa: BLE001 — job-store lookup: any backend error must degrade to "no running job" rather than 500
        log.warning("Job lookup failed for scenario %s", scenario_id, exc_info=True)
    return None


@router.get("/clustering/scenario/{scenario_id}/status")
def get_scenario_status(scenario_id: str):
    """Poll scenario execution status."""
    _validate_scenario_id(scenario_id)
    # 1. Check for completed/failed result on disk (fastest path)
    from scripts.ml.run_clustering_scenario import get_scenario_result

    result = get_scenario_result(scenario_id)
    if result is not None:
        return {
            "scenario_id": scenario_id,
            "status": result.get("status", "completed"),
            "runtime_seconds": result.get("runtime_seconds", 0),
            "result": result,
        }

    # 2. Check job_history DB for running/queued/failed status
    job = _find_job_by_scenario_id(scenario_id)
    if job is not None:
        job_status = job.get("status", "unknown")
        if job_status == "running":
            # Compute elapsed from submitted_at
            submitted = job.get("submitted_at")
            elapsed = 0.0
            if submitted:
                if isinstance(submitted, str):
                    submitted = datetime.datetime.fromisoformat(submitted)
                if hasattr(submitted, "timestamp"):
                    elapsed = round(time.time() - submitted.timestamp(), 1)
            return {"scenario_id": scenario_id, "status": "running", "elapsed_seconds": elapsed}
        if job_status == "completed" and job.get("result"):
            job_result = job["result"] if isinstance(job["result"], dict) else json.loads(job["result"])
            return {
                "scenario_id": scenario_id,
                "status": job_result.get("status", "completed"),
                "runtime_seconds": job_result.get("runtime_seconds", 0),
                "result": job_result,
            }
        if job_status == "failed":
            return {
                "scenario_id": scenario_id,
                "status": "failed",
                "error": job.get("error", "Scenario failed"),
            }
        if job_status == "queued":
            return {"scenario_id": scenario_id, "status": "queued"}

    raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")


@router.get("/clustering/scenario/{scenario_id}")
def get_clustering_scenario(scenario_id: str):
    """Retrieve a previously run scenario result."""
    _validate_scenario_id(scenario_id)
    from scripts.ml.run_clustering_scenario import get_scenario_result

    result = get_scenario_result(scenario_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")
    return result


@router.post("/clustering/scenario/{scenario_id}/promote", dependencies=[Depends(require_api_key)])
def promote_clustering_scenario(scenario_id: str):
    """Promote a scenario to production (writes sku_cluster_assignment)."""
    _validate_scenario_id(scenario_id)
    from scripts.ml.run_clustering_scenario import promote_scenario

    try:
        result = promote_scenario(scenario_id)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")
    except (ValueError, RuntimeError, OSError) as e:
        log.exception("Promote scenario failed for %s", scenario_id)
        raise HTTPException(status_code=500, detail="Promote failed. Check server logs for details.") from e
