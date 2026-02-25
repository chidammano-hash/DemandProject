"""DFU clustering + what-if scenario endpoints (features 7, 29)."""
from __future__ import annotations

from typing import Any
import json
import threading

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
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


@router.post("/clustering/scenario", dependencies=[Depends(require_api_key)])
async def run_clustering_scenario(req: ClusteringScenarioRequest):
    """Run a trial clustering pipeline with custom parameters."""
    global _scenario_running

    if not _scenario_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A clustering scenario is already running. Please wait.",
        )

    if _scenario_running:
        _scenario_lock.release()
        raise HTTPException(
            status_code=409,
            detail="A clustering scenario is already running. Please wait.",
        )

    _scenario_running = True
    _scenario_lock.release()

    try:
        import asyncio
        loop = asyncio.get_event_loop()

        def _run():
            from scripts.run_clustering_scenario import run_scenario
            return run_scenario(
                feature_params=req.feature_params.model_dump() if req.feature_params else None,
                model_params=req.model_params.model_dump() if req.model_params else None,
                label_params=req.label_params.model_dump() if req.label_params else None,
                relabel_only=req.relabel_only,
                previous_scenario_id=req.previous_scenario_id,
            )

        result = await loop.run_in_executor(None, _run)
        return result
    finally:
        _scenario_running = False


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
