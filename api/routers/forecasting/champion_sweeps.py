"""Champion Strategy Sweep (Tournament) API router.

A sweep orchestrates the existing champion-experiment machinery: it fans out a
grid of candidate champion configs, ranks them globally + per demand segment,
assembles a per-segment composite, and recommends a winner. Children are ordinary
``champion_experiment`` rows, so compare/detail reuse that subsystem. Production
promotion is reserved for the governed champion-refresh lifecycle.

All endpoints live under the /champion-sweeps prefix. See spec
``docs/specs/02-forecasting/30-champion-strategy-sweep.md``.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from api.auth import require_api_key
from api.core import get_conn
from api.routers.forecasting._champion_promotion_boundary import (
    raise_manual_champion_promotion_retired,
)
from common.core.sql_helpers import parse_db_json as _parse_json
from common.core.utils import get_competing_model_ids, load_forecast_pipeline_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/champion-sweeps", tags=["champion-sweeps"])

_VALID_MODES = {"global", "per_segment", "both"}
_VALID_AXES = {"demand_class", "ml_cluster", "abc_xyz"}
_VALID_OBJECTIVES = {"accuracy", "gap_to_ceiling", "robust"}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CreateSweepBody(BaseModel):
    """Request body for POST /champion-sweeps."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=200)
    notes: str | None = None
    mode: str = "both"
    segment_axis: str = "demand_class"
    objective: str = "robust"
    grid_spec: dict[str, Any] = Field(
        description="templates × models_variants × metric, or an explicit 'configs' list",
    )
    parallel: bool = False
    baseline_experiment_id: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SWEEP_COLS = """
    sweep_id, label, notes, mode, segment_axis, objective, grid_spec, parallel,
    baseline_experiment_id, status, candidate_count, completed_count, job_id,
    created_at, started_at, completed_at, runtime_seconds,
    best_global_experiment_id, composite_experiment_id, recommended_experiment_id,
    recommended_score, recommended_gate_eligible
"""


def _sweep_row_to_dict(cur: Any, row: tuple[Any, ...]) -> dict[str, Any]:
    cols = [d[0] for d in cur.description]
    d = dict(zip(cols, row, strict=False))
    d["grid_spec"] = _parse_json(d.get("grid_spec"))
    return d


def _expanded_count(grid_spec: dict[str, Any]) -> int:
    """Mirror the runner's expansion arithmetic for the create-time preview/cap.

    Segmentation is NOT an axis, so this is independent of segment_axis.
    """
    if grid_spec.get("configs"):
        return len(grid_spec["configs"])
    templates = grid_spec.get("templates") or []
    variants = grid_spec.get("models_variants") or [None]
    metrics = grid_spec.get("metric") or [None]
    return len(templates) * len(variants) * len(metrics)


def _validate_model_set(models: Any, *, location: str) -> None:
    """Validate one explicit sweep model set against the live competition roster."""
    if not isinstance(models, list) or any(not isinstance(model_id, str) for model_id in models):
        raise HTTPException(
            status_code=422,
            detail=f"{location} must be a list of model ID strings",
        )

    valid_models = get_competing_model_ids()
    unsupported = sorted(set(models) - set(valid_models))
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{location} contains unsupported champion model(s): {unsupported}. "
                f"Valid competing models: {valid_models}"
            ),
        )

    duplicates = sorted({model_id for model_id in models if models.count(model_id) > 1})
    if duplicates:
        raise HTTPException(
            status_code=422,
            detail=f"{location} contains duplicate model(s): {duplicates}",
        )
    if len(models) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"{location} must contain at least 2 distinct competing models",
        )


def _validate_grid_models(grid_spec: dict[str, Any]) -> None:
    """Validate every model set explicitly supplied by a sweep request."""
    configs = grid_spec.get("configs")
    if configs is not None:
        if not isinstance(configs, list):
            raise HTTPException(status_code=422, detail="grid_spec.configs must be a list")
        for index, config in enumerate(configs):
            if not isinstance(config, dict):
                raise HTTPException(
                    status_code=422,
                    detail=f"grid_spec.configs[{index}] must be an object",
                )
            if "models" in config:
                _validate_model_set(
                    config["models"],
                    location=f"grid_spec.configs[{index}].models",
                )

    variants = grid_spec.get("models_variants")
    if variants is not None:
        if not isinstance(variants, list) or not variants:
            raise HTTPException(
                status_code=422,
                detail="grid_spec.models_variants must be a non-empty list of model sets",
            )
        for index, models in enumerate(variants):
            _validate_model_set(
                models,
                location=f"grid_spec.models_variants[{index}]",
            )


def _max_candidates() -> int:
    try:
        cfg = load_forecast_pipeline_config() or {}
    except (FileNotFoundError, KeyError, ValueError):
        return 24
    return int((cfg.get("sweep") or {}).get("max_candidates", 24))


# ---------------------------------------------------------------------------
# Create + launch
# ---------------------------------------------------------------------------

@router.post("", status_code=202, dependencies=[Depends(require_api_key)])
def create_sweep(body: CreateSweepBody):
    """Create a champion sweep and launch it as a background job."""
    if body.mode not in _VALID_MODES:
        raise HTTPException(status_code=422, detail=f"Invalid mode '{body.mode}'")
    if body.segment_axis not in _VALID_AXES:
        raise HTTPException(status_code=422, detail=f"Invalid segment_axis '{body.segment_axis}'")
    if body.objective not in _VALID_OBJECTIVES:
        raise HTTPException(status_code=422, detail=f"Invalid objective '{body.objective}'")

    _validate_grid_models(body.grid_spec)

    count = _expanded_count(body.grid_spec)
    if count == 0:
        raise HTTPException(status_code=422, detail="grid_spec expands to zero candidates")
    cap = _max_candidates()
    if count > cap:
        raise HTTPException(
            status_code=422,
            detail=f"grid_spec expands to {count} candidates, exceeding the cap of {cap}",
        )

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO champion_sweep
                    (label, notes, mode, segment_axis, objective, grid_spec,
                     parallel, baseline_experiment_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING sweep_id
                """,
                (
                    body.label, body.notes, body.mode, body.segment_axis, body.objective,
                    json.dumps(body.grid_spec), body.parallel, body.baseline_experiment_id,
                ),
            )
            sweep_id = cur.fetchone()[0]
            conn.commit()

        from common.services.job_registry import JobManager

        jm = JobManager()
        job_id = jm.submit_job(
            job_type="champion_sweep",
            params={"sweep_id": sweep_id},
            label=f"Champion Sweep: {body.label}",
        )

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE champion_sweep SET job_id = %s WHERE sweep_id = %s",
                (job_id, sweep_id),
            )
            conn.commit()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to create champion sweep")
        raise HTTPException(status_code=500, detail="Failed to create sweep") from None

    return {
        "sweep_id": sweep_id,
        "job_id": job_id,
        "status": "queued",
        "candidate_count": count,
        "label": body.label,
    }


# ---------------------------------------------------------------------------
# List + detail
# ---------------------------------------------------------------------------

@router.get("")
def list_sweeps(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    status: str | None = Query(None),
):
    """List sweeps, newest first."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            if status:
                cur.execute(
                    f"SELECT {_SWEEP_COLS} FROM champion_sweep WHERE status = %s "
                    "ORDER BY created_at DESC OFFSET %s LIMIT %s",
                    (status, offset, limit),
                )
            else:
                cur.execute(
                    f"SELECT {_SWEEP_COLS} FROM champion_sweep "
                    "ORDER BY created_at DESC OFFSET %s LIMIT %s",
                    (offset, limit),
                )
            rows = [_sweep_row_to_dict(cur, r) for r in cur.fetchall()]
    except Exception:
        logger.exception("Failed to list champion sweeps")
        raise HTTPException(status_code=500, detail="Failed to list sweeps") from None
    return {"sweeps": rows, "offset": offset, "limit": limit}


@router.get("/{sweep_id}")
def get_sweep(sweep_id: int):
    """Sweep detail + recommendation summary."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_SWEEP_COLS} FROM champion_sweep WHERE sweep_id = %s",
                (sweep_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Sweep {sweep_id} not found")
            sweep = _sweep_row_to_dict(cur, row)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to load champion sweep %d", sweep_id)
        raise HTTPException(status_code=500, detail="Failed to load sweep") from None
    return sweep


@router.get("/{sweep_id}/leaderboard")
def get_leaderboard(sweep_id: int):
    """Global-ranked members joined to their champion_experiment rows."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM champion_sweep WHERE sweep_id = %s", (sweep_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail=f"Sweep {sweep_id} not found")
            cur.execute(
                """
                SELECT m.experiment_id, m.global_rank, m.global_score, m.gate_eligible,
                       m.is_composite, m.skipped_duplicate,
                       e.label, e.strategy, e.strategy_params, e.models, e.metric,
                       e.champion_accuracy, e.ceiling_accuracy, e.gap_bps, e.status
                FROM champion_sweep_member m
                JOIN champion_experiment e ON e.experiment_id = m.experiment_id
                WHERE m.sweep_id = %s
                ORDER BY m.global_rank NULLS LAST, m.experiment_id
                """,
                (sweep_id,),
            )
            cols = [d[0] for d in cur.description]
            rows = []
            for r in cur.fetchall():
                d = dict(zip(cols, r, strict=False))
                d["strategy_params"] = _parse_json(d.get("strategy_params"))
                d["models"] = _parse_json(d.get("models"))
                rows.append(d)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to load leaderboard for sweep %d", sweep_id)
        raise HTTPException(status_code=500, detail="Failed to load leaderboard") from None
    return {"sweep_id": sweep_id, "members": rows}


@router.get("/{sweep_id}/segments")
def get_segments(sweep_id: int):
    """Per-segment winner map + per-segment scores."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM champion_sweep WHERE sweep_id = %s", (sweep_id,))
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail=f"Sweep {sweep_id} not found")
            cur.execute(
                """
                SELECT s.segment, s.experiment_id, s.n_dfus, s.accuracy, s.score,
                       s.segment_rank, e.label, e.strategy
                FROM champion_sweep_segment_score s
                JOIN champion_experiment e ON e.experiment_id = s.experiment_id
                WHERE s.sweep_id = %s
                ORDER BY s.segment, s.segment_rank NULLS LAST
                """,
                (sweep_id,),
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to load segments for sweep %d", sweep_id)
        raise HTTPException(status_code=500, detail="Failed to load segments") from None
    # Group into per-segment winner (rank 1) + full candidate list.
    by_segment: dict[str, dict[str, Any]] = {}
    for r in rows:
        seg = r["segment"]
        bucket = by_segment.setdefault(seg, {"segment": seg, "winner": None, "candidates": []})
        bucket["candidates"].append(r)
        if r["segment_rank"] == 1:
            bucket["winner"] = r
    return {"sweep_id": sweep_id, "segments": list(by_segment.values())}


# ---------------------------------------------------------------------------
# Cancel + promote-winner + delete
# ---------------------------------------------------------------------------

@router.post("/{sweep_id}/cancel", dependencies=[Depends(require_api_key)])
def cancel_sweep(sweep_id: int):
    """Cancel a queued/running sweep."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT status, job_id FROM champion_sweep WHERE sweep_id = %s",
                (sweep_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Sweep {sweep_id} not found")
            status, job_id = row
            if status not in ("queued", "running"):
                raise HTTPException(
                    status_code=409, detail=f"Cannot cancel a sweep with status '{status}'"
                )
            if job_id:
                from common.services.job_registry import JobManager

                JobManager().cancel_job(job_id)
            cur.execute(
                "UPDATE champion_sweep SET status = 'cancelled', completed_at = NOW() "
                "WHERE sweep_id = %s",
                (sweep_id,),
            )
            conn.commit()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to cancel sweep %d", sweep_id)
        raise HTTPException(status_code=500, detail="Failed to cancel sweep") from None
    return {"sweep_id": sweep_id, "status": "cancelled"}


@router.post("/{sweep_id}/promote-winner", dependencies=[Depends(require_api_key)])
def promote_winner(sweep_id: int):
    """Retain the legacy path as an authenticated, fail-closed boundary."""
    raise_manual_champion_promotion_retired()


@router.delete("/{sweep_id}", dependencies=[Depends(require_api_key)])
def delete_sweep(sweep_id: int):
    """Delete a sweep (not while running). Members cascade; child experiments remain."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT status FROM champion_sweep WHERE sweep_id = %s", (sweep_id,))
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Sweep {sweep_id} not found")
            if row[0] in ("queued", "running"):
                raise HTTPException(
                    status_code=409, detail=f"Cannot delete a sweep with status '{row[0]}'"
                )
            cur.execute("DELETE FROM champion_sweep WHERE sweep_id = %s", (sweep_id,))
            conn.commit()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to delete sweep %d", sweep_id)
        raise HTTPException(status_code=500, detail="Failed to delete sweep") from None
    return {"sweep_id": sweep_id, "deleted": True}
