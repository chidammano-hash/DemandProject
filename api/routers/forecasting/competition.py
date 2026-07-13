"""Champion / Model Competition endpoints (feature 15).

Uses shared strategy module for leak-free per-DFU per-month selection.
All strategies enforce strict causality: selection for month T uses only
data from months < T.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path
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
_CHAMPION_DATA_DIR = _PROJECT_ROOT / "data" / "champion"

_WINNER_REQUIRED_COLUMNS = {
    "item_id",
    "customer_group",
    "loc",
    "startdate",
    "model_id",
}

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
        cur.execute("SELECT DISTINCT model_id FROM fact_external_forecast_monthly ORDER BY 1")
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
    """Return results for the currently promoted champion and its exact winners."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT experiment_id, label, strategy,
                       champion_accuracy, ceiling_accuracy, gap_bps,
                       n_champions, n_dfu_months, promoted_at
                FROM champion_experiment
                WHERE is_promoted = TRUE
                ORDER BY promoted_at DESC, experiment_id DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
    except psycopg.Error:
        logger.exception("Failed to fetch the promoted champion summary")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch the promoted champion summary",
        ) from None

    if row is None:
        return {"status": "not_run", "summary": None}

    (
        experiment_id,
        label,
        strategy,
        champion_accuracy,
        ceiling_accuracy,
        gap_bps,
        expected_dfus,
        expected_rows,
        promoted_at,
    ) = row
    if promoted_at is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Promoted champion experiment #{experiment_id} is missing its promotion timestamp"
            ),
        )
    artifact_path = _CHAMPION_DATA_DIR / f"experiment_{experiment_id}_winners.csv"
    artifact = _read_promoted_winner_artifact(
        artifact_path,
        experiment_id=int(experiment_id),
        expected_dfus=int(expected_dfus) if expected_dfus is not None else None,
        expected_rows=int(expected_rows) if expected_rows is not None else None,
    )

    champion_accuracy_value = float(champion_accuracy) if champion_accuracy is not None else None
    ceiling_accuracy_value = float(ceiling_accuracy) if ceiling_accuracy is not None else None
    return {
        "status": "ok",
        "summary": {
            "experiment_id": int(experiment_id),
            "experiment_label": label,
            "strategy": strategy,
            "artifact_name": artifact_path.name,
            "total_dfus": artifact["total_dfus"],
            "total_dfu_months": artifact["total_rows"],
            "total_champion_rows": artifact["total_rows"],
            "model_wins": artifact["model_wins"],
            "overall_champion_wape": _accuracy_to_wape(champion_accuracy_value),
            "overall_champion_accuracy_pct": champion_accuracy_value,
            "run_ts": promoted_at.isoformat(),
            "overall_ceiling_wape": _accuracy_to_wape(ceiling_accuracy_value),
            "overall_ceiling_accuracy_pct": ceiling_accuracy_value,
            "gap_bps": float(gap_bps) if gap_bps is not None else None,
        },
    }


def _accuracy_to_wape(accuracy: float | None) -> float | None:
    """Convert the persisted accuracy percentage to its WAPE complement."""
    return round(100.0 - accuracy, 4) if accuracy is not None else None


def _read_promoted_winner_artifact(
    path: Path,
    *,
    experiment_id: int,
    expected_dfus: int | None,
    expected_rows: int | None,
) -> dict[str, Any]:
    """Read and validate the exact winner artifact for a promoted experiment."""
    if not path.is_file():
        raise HTTPException(
            status_code=409,
            detail=(
                f"Promoted champion experiment #{experiment_id} is missing its winner artifact"
            ),
        )

    model_wins: Counter[str] = Counter()
    dfus: set[tuple[str, str, str]] = set()
    winner_keys: set[tuple[str, str, str, str]] = set()
    total_rows = 0
    try:
        with path.open(newline="", encoding="utf-8") as artifact_file:
            reader = csv.DictReader(artifact_file)
            columns = set(reader.fieldnames or [])
            missing_columns = sorted(_WINNER_REQUIRED_COLUMNS - columns)
            if missing_columns:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Promoted champion experiment #{experiment_id} has an invalid "
                        f"winner artifact: missing {', '.join(missing_columns)}"
                    ),
                )

            for artifact_row in reader:
                total_rows += 1
                values = {
                    column: (artifact_row.get(column) or "").strip()
                    for column in _WINNER_REQUIRED_COLUMNS
                }
                if any(not value for value in values.values()):
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Promoted champion experiment #{experiment_id} has an invalid "
                            f"winner row at line {reader.line_num}"
                        ),
                    )
                winner_key = (
                    values["item_id"],
                    values["customer_group"],
                    values["loc"],
                    values["startdate"],
                )
                if winner_key in winner_keys:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"Promoted champion experiment #{experiment_id} has duplicate "
                            f"winner rows at line {reader.line_num}"
                        ),
                    )
                winner_keys.add(winner_key)
                model_wins[values["model_id"]] += 1
                dfus.add(
                    (
                        values["item_id"],
                        values["customer_group"],
                        values["loc"],
                    )
                )
    except HTTPException:
        raise
    except (OSError, UnicodeError, csv.Error):
        logger.exception("Failed to read promoted champion winner artifact")
        raise HTTPException(
            status_code=500,
            detail="Failed to read the promoted champion winner artifact",
        ) from None

    if expected_rows is None or total_rows != expected_rows:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Promoted champion experiment #{experiment_id} winner row count does not "
                "match its governed results"
            ),
        )
    if expected_dfus is None or len(dfus) != expected_dfus:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Promoted champion experiment #{experiment_id} winner DFU count does not "
                "match its governed results"
            ),
        )

    ranked_model_wins = dict(sorted(model_wins.items(), key=lambda item: (-item[1], item[0])))
    return {
        "total_dfus": len(dfus),
        "total_rows": total_rows,
        "model_wins": ranked_model_wins,
    }
