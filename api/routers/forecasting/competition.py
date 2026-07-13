"""Champion / Model Competition endpoints (feature 15).

Uses shared strategy module for leak-free per-DFU per-month selection.
All strategies enforce strict causality: selection for month T uses only
data from months < T.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import get_conn
from common.core.paths import PROJECT_ROOT as _PROJECT_ROOT
from common.core.utils import get_competing_model_ids, load_forecast_pipeline_config, reset_config
from common.ml.champion import STRATEGY_REGISTRY as _STRAT_REG

router = APIRouter(tags=["competition"])
logger = logging.getLogger(__name__)

_PIPELINE_CONFIG_PATH = _PROJECT_ROOT / "config" / "forecasting" / "forecast_pipeline_config.yaml"
_CHAMPION_SUMMARY_PATH = _PROJECT_ROOT / "data" / "champion" / "champion_summary.json"

_VALID_STRATEGIES = set(_STRAT_REG.keys())


class CompetitionConfigUpdate(BaseModel):
    metric: str = "wape"
    lag: str = "execution"
    min_dfu_rows: int = 3
    champion_model_id: str = "champion"
    models: list[str]
    strategy: str = "expanding"
    strategy_params: dict[str, Any] = {}


@router.get("/competition/config")
def get_competition_config():
    """Return current model competition config + available models in DB."""
    pipeline_cfg = load_forecast_pipeline_config()
    champion = pipeline_cfg.get("champion", {})

    # Build config dict matching the legacy response shape
    cfg = {
        "strategy": champion.get("strategy", "expanding"),
        "strategy_params": champion.get("strategy_params", {}),
        "metric": champion.get("metric", "wape"),
        "lag": champion.get("lag", "execution"),
        "min_dfu_rows": champion.get("min_dfu_rows", 3),
        "champion_model_id": champion.get("champion_model_id", "champion"),
        "fallback_model_id": champion.get("fallback_model_id"),
        "models": get_competing_model_ids(),
        "meta_learner": champion.get("meta_learner", {}),
    }

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT model_id FROM fact_external_forecast_monthly ORDER BY 1"
        )
        available = [r[0] for r in cur.fetchall() if r[0]]

    return {"config": cfg, "available_models": available}


@router.put("/competition/config", dependencies=[Depends(require_api_key)])
def update_competition_config(body: CompetitionConfigUpdate):
    """Update model competition config (writes to forecast_pipeline_config.yaml champion section)."""
    import yaml

    if body.metric not in ("wape", "accuracy_pct"):
        raise HTTPException(422, "metric must be 'wape' or 'accuracy_pct'")
    valid_lags = {"execution", "0", "1", "2", "3", "4"}
    if body.lag not in valid_lags:
        raise HTTPException(422, f"lag must be one of: {sorted(valid_lags)}")
    if len(body.models) < 2:
        raise HTTPException(422, "At least 2 models required for competition")
    if body.strategy not in _VALID_STRATEGIES:
        raise HTTPException(422, f"strategy must be one of: {sorted(_VALID_STRATEGIES)}")

    # Read existing pipeline config
    with _PIPELINE_CONFIG_PATH.open() as f:
        pipeline_cfg = yaml.safe_load(f) or {}

    # Update champion section
    champion = pipeline_cfg.setdefault("champion", {})
    champion["metric"] = body.metric
    champion["lag"] = body.lag
    champion["min_dfu_rows"] = body.min_dfu_rows
    champion["champion_model_id"] = body.champion_model_id
    champion["strategy"] = body.strategy
    champion["strategy_params"] = body.strategy_params

    with _PIPELINE_CONFIG_PATH.open("w") as f:
        yaml.dump(pipeline_cfg, f, default_flow_style=False, sort_keys=False)

    reset_config("forecast_pipeline_config.yaml")

    return {"status": "ok", "config": champion}


@router.post("/competition/run", status_code=202, dependencies=[Depends(require_api_key)])
def run_competition():
    """Submit the fail-closed governed champion lifecycle as a managed job."""
    from common.services.job_registry import JobManager

    try:
        job_id = JobManager().submit_job(
            job_type="champion_select",
            params={},
            label="Governed Champion Refresh",
            triggered_by="competition_run",
        )
    except (psycopg.Error, RuntimeError, ValueError):
        logger.exception("Failed to submit governed champion refresh")
        raise HTTPException(
            status_code=500,
            detail="Failed to submit governed champion refresh",
        ) from None
    return {
        "job_id": job_id,
        "job_type": "champion_select",
        "status": "queued",
        "message": "Governed champion refresh submitted",
    }


@router.get("/competition/summary")
def get_competition_summary():
    """Return the last champion selection summary, if available."""
    if not _CHAMPION_SUMMARY_PATH.exists():
        return {"status": "not_run", "summary": None}
    with _CHAMPION_SUMMARY_PATH.open() as f:
        return {"status": "ok", "summary": json.load(f)}
