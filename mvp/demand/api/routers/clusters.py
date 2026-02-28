"""DFU clustering + what-if scenario endpoints (features 7, 29, 38)."""
from __future__ import annotations

from typing import Any
import json
import time
import threading

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from api.core import get_conn
from api.auth import require_api_key

router = APIRouter()


# ---------------------------------------------------------------------------
# Cluster summary / profiles / visualization
# ---------------------------------------------------------------------------
@router.get("/domains/dfu/clusters")
def dfu_clusters(source: str = Query(default="ml", pattern="^(ml|source)$")):
    """Get cluster summary statistics for DFU clustering.

    source=ml    -> pipeline-generated clusters (ml_cluster column)
    source=source -> original source-file clusters (cluster_assignment column)
    """
    col = "ml_cluster" if source == "ml" else "cluster_assignment"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""
            WITH cluster_counts AS (
                SELECT
                    {col} AS cluster_label,
                    COUNT(*) AS dfu_count
                FROM dim_dfu
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
                FROM dim_dfu d
                INNER JOIN fact_sales_monthly s
                    ON s.dmdunit = d.dmdunit
                    AND s.dmdgroup = d.dmdgroup
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
        """)
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
            "domain": "dfu",
            "source": source,
            "total_assigned": total_assigned,
            "clusters": clusters,
        }


@router.get("/domains/dfu/clusters/profiles")
def dfu_cluster_profiles():
    """Get cluster profiles with centroid features and metadata."""
    root = Path(__file__).resolve().parents[2]
    profiles_path = root / "data" / "clustering" / "cluster_profiles.json"
    metadata_path = root / "data" / "clustering" / "cluster_metadata.json"

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


@router.get("/domains/dfu/clusters/visualization/{image_name}")
def dfu_cluster_visualization(image_name: str):
    """Serve clustering visualization images."""
    allowed = {"k_selection_plots.png", "cluster_visualization.png"}
    if image_name not in allowed:
        raise HTTPException(404, f"Image not found: {image_name}")
    root = Path(__file__).resolve().parents[2]
    img_path = root / "data" / "clustering" / image_name
    if not img_path.exists():
        raise HTTPException(404, f"Image not generated yet: {image_name}")
    return FileResponse(str(img_path), media_type="image/png")


@router.get("/domains/dfu/seasonality-profiles")
def dfu_seasonality_profiles():
    """Get distinct seasonality_profile values with DFU counts."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT
                COALESCE(seasonality_profile, '(unknown)') AS profile,
                COUNT(*)::bigint AS dfu_count
            FROM dim_dfu
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
# Clustering what-if scenarios (feature 29)
# ---------------------------------------------------------------------------
_scenario_lock = threading.Lock()
_scenario_running = False


class FeatureParams(BaseModel):
    time_window_months: int | str = 24
    min_months_history: int = 1


class ModelParams(BaseModel):
    k_range: list[int] = [3, 12]
    min_cluster_size_pct: float = 2.0
    use_pca: bool = False
    pca_components: int | None = None
    skip_gap: bool = True
    all_features: bool = False


class LabelParams(BaseModel):
    volume_high: float = 0.75
    volume_low: float = 0.25
    cv_steady: float = 0.3
    cv_volatile: float = 0.8
    seasonality_threshold: float = 0.5
    zero_demand_threshold: float = 0.2


class ClusteringScenarioRequest(BaseModel):
    feature_params: FeatureParams | None = None
    model_params: ModelParams | None = None
    label_params: LabelParams | None = None
    relabel_only: bool = False
    previous_scenario_id: str | None = None


@router.get("/clustering/defaults")
def get_clustering_defaults():
    """Return current default clustering parameters from config."""
    import yaml

    config_path = Path(__file__).resolve().parents[2] / "config" / "clustering_config.yaml"
    if not config_path.exists():
        return {
            "feature_params": {"time_window_months": 24, "min_months_history": 1},
            "model_params": {
                "k_range": [3, 12], "min_cluster_size_pct": 2.0,
                "use_pca": False, "pca_components": None,
                "skip_gap": False, "all_features": False,
            },
            "label_params": {
                "volume_high": 0.75, "volume_low": 0.25,
                "cv_steady": 0.3, "cv_volatile": 0.8,
                "seasonality_threshold": 0.5, "zero_demand_threshold": 0.2,
            },
        }

    with open(config_path) as f:
        config = yaml.safe_load(f)

    clustering = config.get("clustering", {})
    labeling = clustering.get("labeling", {})
    vol = labeling.get("volume_thresholds", {})
    cv = labeling.get("cv_thresholds", {})

    return {
        "feature_params": {
            "time_window_months": clustering.get("time_window_months", 24),
            "min_months_history": clustering.get("min_months_history", 1),
        },
        "model_params": {
            "k_range": clustering.get("k_range", [3, 12]),
            "min_cluster_size_pct": clustering.get("min_cluster_size_pct", 2.0),
            "use_pca": clustering.get("use_pca", False),
            "pca_components": clustering.get("pca_components", None),
            "skip_gap": clustering.get("skip_gap", False),
            "all_features": clustering.get("all_features", False),
        },
        "label_params": {
            "volume_high": vol.get("high", 0.75),
            "volume_low": vol.get("low", 0.25),
            "cv_steady": cv.get("steady", 0.3),
            "cv_volatile": cv.get("volatile", 0.8),
            "seasonality_threshold": labeling.get("seasonality_threshold", 0.5),
            "zero_demand_threshold": labeling.get("zero_demand_threshold", 0.2),
        },
    }


_scenario_start_time: float | None = None
_running_scenario_id: str | None = None


@router.get("/clustering/scenario/estimate")
def estimate_scenario_runtime(
    scope: str = Query(default="all"),
    k_min: int = Query(default=3),
    k_max: int = Query(default=12),
    skip_gap: bool = Query(default=True),
):
    """Estimate scenario runtime based on parameters."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM dim_dfu")
        dfu_count = int(cur.fetchone()[0])

    k_range = max(1, k_max - k_min + 1)
    max_training_dfus = 20_000  # Pipeline samples if count exceeds this

    # Empirical constants calibrated from observed runs:
    # Feature generation: ~0.001s per DFU (per-DFU loop, runs on ALL DFUs)
    feature_gen_per_dfu = 0.001
    # KMeans training: ~0.002s per training DFU per K (runs on sample if large)
    kmeans_per_dfu_per_k = 0.002
    gap_multiplier = 2.5 if not skip_gap else 1.0
    # Fixed overhead for SQL query + data loading
    overhead_seconds = 10.0

    training_dfus = min(dfu_count, max_training_dfus)
    feature_gen_time = feature_gen_per_dfu * dfu_count
    kmeans_time = kmeans_per_dfu_per_k * training_dfus * k_range * gap_multiplier
    estimated = overhead_seconds + feature_gen_time + kmeans_time

    return {
        "estimated_seconds": round(estimated, 0),
        "dfu_count": dfu_count,
        "training_sample": training_dfus,
        "sampled": dfu_count > max_training_dfus,
        "k_range": k_range,
        "skip_gap": skip_gap,
    }


@router.post("/clustering/scenario", dependencies=[Depends(require_api_key)])
async def run_clustering_scenario(req: ClusteringScenarioRequest):
    """Run a trial clustering pipeline with custom parameters (non-blocking).

    Delegates to the JobManager so the scenario appears in the Jobs tab.
    """
    global _scenario_running, _scenario_start_time, _running_scenario_id
    from common.job_registry import JobManager

    manager = JobManager()

    # Build params from the request
    params: dict[str, Any] = {
        "feature_params": req.feature_params.model_dump() if req.feature_params else None,
        "model_params": req.model_params.model_dump() if req.model_params else None,
        "label_params": req.label_params.model_dump() if req.label_params else None,
        "relabel_only": req.relabel_only,
        "previous_scenario_id": req.previous_scenario_id,
    }

    # Also embed the scenario_id so _run_cluster_scenario uses it
    from scripts.run_clustering_scenario import generate_scenario_id
    scenario_id = generate_scenario_id()
    params["scenario_id"] = scenario_id

    job_id = manager.submit_job("cluster_scenario", params, label=f"What-If Scenario")

    # Determine if the job is queued or running immediately
    job_status = manager.get_status(job_id)
    is_queued = job_status and job_status.get("status") == "queued"

    if not is_queued:
        # Track for the legacy status polling endpoint
        _running_scenario_id = scenario_id
        _scenario_running = True
        _scenario_start_time = time.time()

        # Wire up legacy state cleanup when job finishes
        _original_start = manager.start_job_in_background

        def _start_with_cleanup(jid: str) -> None:
            """Wrap start to clear legacy globals on completion."""
            _original_start(jid)

            def _wait_and_cleanup():
                import time as _t
                while jid in manager._active_jobs:
                    _t.sleep(1)
                global _scenario_running, _scenario_start_time, _running_scenario_id
                _scenario_running = False
                _scenario_start_time = None
                _running_scenario_id = None

            cleanup_thread = threading.Thread(target=_wait_and_cleanup, daemon=True)
            cleanup_thread.start()

        _start_with_cleanup(job_id)

    return JSONResponse(
        status_code=202,
        content={
            "scenario_id": scenario_id,
            "status": "queued" if is_queued else "running",
            "job_id": job_id,
        },
    )


@router.get("/clustering/scenario/{scenario_id}/status")
def get_scenario_status(scenario_id: str):
    """Poll scenario execution status."""
    # Check if this scenario is currently running
    if _running_scenario_id == scenario_id and _scenario_running:
        elapsed = round(time.time() - (_scenario_start_time or time.time()), 1)
        return {"scenario_id": scenario_id, "status": "running", "elapsed_seconds": elapsed}

    # Check for completed/failed result on disk
    from scripts.run_clustering_scenario import get_scenario_result

    result = get_scenario_result(scenario_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")

    return {
        "scenario_id": scenario_id,
        "status": result.get("status", "completed"),
        "runtime_seconds": result.get("runtime_seconds", 0),
        "result": result,
    }


@router.get("/clustering/scenario/{scenario_id}")
def get_clustering_scenario(scenario_id: str):
    """Retrieve a previously run scenario result."""
    from scripts.run_clustering_scenario import get_scenario_result

    result = get_scenario_result(scenario_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")
    return result


@router.post("/clustering/scenario/{scenario_id}/promote", dependencies=[Depends(require_api_key)])
def promote_clustering_scenario(scenario_id: str):
    """Promote a scenario to production (updates dim_dfu.ml_cluster)."""
    from scripts.run_clustering_scenario import promote_scenario

    try:
        result = promote_scenario(scenario_id)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
