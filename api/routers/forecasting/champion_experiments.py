"""Champion Experimentation Studio API router.

Provides full CRUD lifecycle for champion selection strategy experiments:
create, list, compare, promote config, load results, and track promotions.

All endpoints live under the /champion-experiments prefix.
"""
from __future__ import annotations

import copy
import json
import logging
import shutil
from typing import Any

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.core import get_conn, set_cache
from common.core.sql_helpers import parse_db_json as _parse_json
from common.core.utils import reset_config
from common.ml.champion import STRATEGY_REGISTRY as _STRAT_REG

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/champion-experiments", tags=["champion-experiments"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from common.core.paths import PROJECT_ROOT as _PROJECT_ROOT

_TEMPLATES_PATH = _PROJECT_ROOT / "config" / "forecasting" / "champion_experiment_templates.yaml"
_PIPELINE_CONFIG_PATH = _PROJECT_ROOT / "config" / "forecasting" / "forecast_pipeline_config.yaml"

_VALID_STRATEGIES = set(_STRAT_REG.keys())
_VALID_METRICS = {"accuracy_pct", "wape"}
_VALID_LAG_MODES = {"execution", "0", "1", "2", "3", "4"}

# Cache TTLs (seconds)
_LIST_CACHE_TTL = 30
_COMPARE_CACHE_TTL = 60
_TEMPLATE_CACHE_TTL = 300
_DETAIL_CACHE_TTL = 30


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class StrategyParams(BaseModel):
    """Strategy-specific parameters."""
    window_months: int | None = None
    decay_factor: float | None = None
    top_k: int | None = None
    weight_method: str | None = None
    min_prior_months: int | None = None


class MetaLearnerParams(BaseModel):
    """Meta-learner configuration."""
    model_type: str = "random_forest"
    n_estimators: int = 200
    max_depth: int = 15
    test_months: int = 3
    performance_window: int = 6


class CreateChampionExperimentBody(BaseModel):
    """Request body for POST /champion-experiments."""
    label: str = Field(min_length=1, max_length=200)
    notes: str | None = None
    template: str | None = None
    strategy: str = "expanding"
    strategy_params: StrategyParams | None = None
    meta_learner_params: MetaLearnerParams | None = None
    models: list[str] = Field(
        default=["lgbm_cluster", "catboost_cluster", "xgboost_cluster", "chronos"],
    )
    metric: str = "accuracy_pct"
    lag_mode: str = "execution"
    min_sku_rows: int = Field(default=3, ge=1, le=24)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SELECT_COLS = """
    experiment_id, label, notes, template_id, status,
    created_at, started_at, completed_at, runtime_seconds, job_id,
    strategy, strategy_params, meta_learner_params, models, metric, lag_mode, min_sku_rows,
    champion_accuracy, ceiling_accuracy, gap_bps, n_champions, n_dfu_months, model_distribution,
    is_promoted, promoted_at, is_results_promoted, results_promoted_at, results_promote_job_id
"""


def _apply_lag_override(target: dict[str, Any], lag_row: tuple) -> None:
    """Override portfolio-level KPIs on *target* dict with lag-specific values.

    *lag_row* columns: (champion_accuracy, ceiling_accuracy, gap_bps,
    n_dfu_months, model_distribution).
    """
    target["champion_accuracy"] = float(lag_row[0]) if lag_row[0] is not None else None
    target["ceiling_accuracy"] = float(lag_row[1]) if lag_row[1] is not None else None
    target["gap_bps"] = float(lag_row[2]) if lag_row[2] is not None else None
    target["n_dfu_months"] = int(lag_row[3]) if lag_row[3] is not None else None
    target["model_distribution"] = _parse_json(lag_row[4])


def _experiment_row_to_dict(row: tuple) -> dict[str, Any]:
    """Convert a champion_experiment row (28 columns) to a response dict."""
    return {
        "experiment_id": row[0],
        "label": row[1],
        "notes": row[2],
        "template_id": row[3],
        "status": row[4],
        "created_at": str(row[5]) if row[5] else None,
        "started_at": str(row[6]) if row[6] else None,
        "completed_at": str(row[7]) if row[7] else None,
        "runtime_seconds": float(row[8]) if row[8] is not None else None,
        "job_id": row[9],
        "strategy": row[10],
        "strategy_params": _parse_json(row[11]),
        "meta_learner_params": _parse_json(row[12]),
        "models": _parse_json(row[13]),
        "metric": row[14],
        "lag_mode": row[15],
        "min_sku_rows": int(row[16]) if row[16] is not None else 3,
        "champion_accuracy": float(row[17]) if row[17] is not None else None,
        "ceiling_accuracy": float(row[18]) if row[18] is not None else None,
        "gap_bps": float(row[19]) if row[19] is not None else None,
        "n_champions": int(row[20]) if row[20] is not None else None,
        "n_dfu_months": int(row[21]) if row[21] is not None else None,
        "model_distribution": _parse_json(row[22]),
        "is_promoted": bool(row[23]),
        "promoted_at": str(row[24]) if row[24] else None,
        "is_results_promoted": bool(row[25]),
        "results_promoted_at": str(row[26]) if row[26] else None,
        "results_promote_job_id": row[27],
    }


# ---------------------------------------------------------------------------
# 1. GET /champion-experiments — List experiments
# ---------------------------------------------------------------------------

@router.get("")
def list_experiments(
    response: FastAPIResponse,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    status: str = Query(default="", max_length=20),
    exec_lag: int | None = Query(default=None, ge=0, le=4),
):
    """List champion experiments with pagination, newest first.

    When *exec_lag* (0-4) is provided, portfolio-level KPIs
    (champion_accuracy, ceiling_accuracy, gap_bps, n_dfu_months,
    model_distribution) are overridden with lag-specific values from
    ``champion_experiment_lag``.
    """
    set_cache(response, max_age=_LIST_CACHE_TTL)

    parts: list[str] = []
    params: list[Any] = []
    if status.strip():
        parts.append("status = %s")
        params.append(status.strip())

    where_sql = f"WHERE {' AND '.join(parts)}" if parts else ""

    count_sql = f"SELECT count(*) FROM champion_experiment {where_sql}"
    data_sql = f"""
        SELECT {_SELECT_COLS}
        FROM champion_experiment
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

            experiments = [_experiment_row_to_dict(r) for r in rows]

            # Override KPIs with lag-specific values when exec_lag is set
            if exec_lag is not None and experiments:
                exp_ids = [e["experiment_id"] for e in experiments]
                cur.execute(
                    """
                    SELECT experiment_id, champion_accuracy, ceiling_accuracy,
                           gap_bps, n_dfu_months, model_distribution
                    FROM champion_experiment_lag
                    WHERE experiment_id = ANY(%s) AND exec_lag = %s
                    """,
                    (exp_ids, exec_lag),
                )
                lag_map: dict[int, tuple] = {}
                for r in cur.fetchall():
                    lag_map[r[0]] = r[1:]  # skip experiment_id
                for exp in experiments:
                    lag_row = lag_map.get(exp["experiment_id"])
                    if lag_row:
                        _apply_lag_override(exp, lag_row)
                    exp["exec_lag_filter"] = exec_lag

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to list champion experiments")
        raise HTTPException(status_code=500, detail="Failed to list champion experiments") from None

    return {
        "experiments": experiments,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


# ---------------------------------------------------------------------------
# 2. GET /champion-experiments/templates (before {id} routes)
# ---------------------------------------------------------------------------

@router.get("/templates")
def get_templates(response: FastAPIResponse):
    """Load champion experiment templates from config YAML."""
    set_cache(response, max_age=_TEMPLATE_CACHE_TTL)

    templates: list[dict[str, Any]] = []
    try:
        with open(_TEMPLATES_PATH) as f:
            config = yaml.safe_load(f)
        templates = config.get("templates", []) if config else []
    except FileNotFoundError:
        logger.info("Champion experiment templates file not found")
    except (yaml.YAMLError, OSError):
        logger.exception("Failed to load champion experiment templates")
        raise HTTPException(status_code=500, detail="Failed to load templates") from None

    # For production_baseline: merge in live config from pipeline config champion section
    for tmpl in templates:
        if tmpl.get("source") in ("model_competition_config", "pipeline_config"):
            try:
                from common.core.utils import get_competing_model_ids, load_forecast_pipeline_config

                pipeline_cfg = load_forecast_pipeline_config()
                champ = pipeline_cfg.get("champion", {})
                tmpl["strategy"] = champ.get("strategy", "expanding")
                tmpl["strategy_params"] = champ.get("strategy_params", {})
                tmpl["meta_learner_params"] = champ.get("meta_learner", {})
                tmpl["models"] = get_competing_model_ids()
                tmpl["metric"] = champ.get("metric", "accuracy_pct")
                tmpl["lag_mode"] = str(champ.get("lag", "execution"))
                tmpl["min_sku_rows"] = champ.get("min_dfu_rows", 3)
            except FileNotFoundError:
                logger.info("Pipeline config not found for template enrichment")
            except (yaml.YAMLError, OSError):
                logger.warning("Failed to load live pipeline config for template")

    return {"templates": templates}


# ---------------------------------------------------------------------------
# 3. GET /champion-experiments/promoted — Current promoted experiment
# ---------------------------------------------------------------------------

@router.get("/promoted")
def get_promoted(response: FastAPIResponse):
    """Get the currently promoted champion experiment."""
    set_cache(response, max_age=_DETAIL_CACHE_TTL)

    sql = f"""
        SELECT {_SELECT_COLS}
        FROM champion_experiment
        WHERE is_promoted = TRUE
        ORDER BY promoted_at DESC
        LIMIT 1
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
    except Exception:
        logger.exception("Failed to fetch promoted champion experiment")
        raise HTTPException(status_code=500, detail="Failed to fetch promoted experiment") from None

    return {"promoted": _experiment_row_to_dict(row) if row else None}


# ---------------------------------------------------------------------------
# 4. GET /champion-experiments/promotions — Promotion audit trail
# ---------------------------------------------------------------------------

@router.get("/promotions")
def list_promotions(
    response: FastAPIResponse,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """Get champion promotion audit trail."""
    set_cache(response, max_age=_LIST_CACHE_TTL)

    sql = """
        SELECT id, experiment_id, promoted_at, promoted_by,
               previous_experiment_id, strategy, champion_accuracy,
               config_snapshot
        FROM champion_promotion_log
        ORDER BY promoted_at DESC
        LIMIT %s OFFSET %s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (limit, offset))
            rows = cur.fetchall()
    except Exception:
        logger.exception("Failed to fetch promotion log")
        raise HTTPException(status_code=500, detail="Failed to fetch promotions") from None

    promotions = []
    for r in rows:
        promotions.append({
            "id": r[0],
            "experiment_id": r[1],
            "promoted_at": str(r[2]) if r[2] else None,
            "promoted_by": r[3],
            "previous_experiment_id": r[4],
            "strategy": r[5],
            "champion_accuracy": float(r[6]) if r[6] is not None else None,
            "config_snapshot": _parse_json(r[7]),
        })
    return {"promotions": promotions}


# ---------------------------------------------------------------------------
# 5. GET /champion-experiments/compare — Compare two experiments
# ---------------------------------------------------------------------------

@router.get("/compare")
def compare_experiments(
    response: FastAPIResponse,
    a_id: int = Query(..., description="First experiment ID"),
    b_id: int = Query(..., description="Second experiment ID"),
    exec_lag: int | None = Query(default=None, ge=0, le=4),
):
    """Compare two champion experiments: overall metrics, per-lag, per-month, model distribution.

    When *exec_lag* (0-4) is provided, overall comparison uses lag-specific
    KPIs instead of portfolio-level values.  Cached results are skipped when
    filtering by lag.
    """
    set_cache(response, max_age=_COMPARE_CACHE_TTL)

    if a_id == b_id:
        raise HTTPException(status_code=400, detail="Cannot compare an experiment with itself")

    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Check cache (skip when lag-filtered — cached results are portfolio-level)
            if exec_lag is None:
                cur.execute(
                    """
                    SELECT overall_comparison, per_lag_comparison, per_month_comparison,
                           model_dist_comparison, config_diffs
                    FROM champion_experiment_comparison
                    WHERE experiment_a_id = %s AND experiment_b_id = %s
                    """,
                    (a_id, b_id),
                )
                cached = cur.fetchone()
                if cached:
                    return {
                        "experiment_a_id": a_id,
                        "experiment_b_id": b_id,
                        "overall_comparison": _parse_json(cached[0]),
                        "per_lag_comparison": _parse_json(cached[1]),
                        "per_month_comparison": _parse_json(cached[2]),
                        "model_dist_comparison": _parse_json(cached[3]),
                        "config_diffs": _parse_json(cached[4]),
                        "source": "cache",
                    }

            # Fetch both experiments
            exp_sql = f"""
                SELECT {_SELECT_COLS}
                FROM champion_experiment
                WHERE experiment_id IN (%s, %s)
            """
            cur.execute(exp_sql, (a_id, b_id))
            rows = cur.fetchall()

            if len(rows) < 2:
                found_ids = {r[0] for r in rows}
                missing = {a_id, b_id} - found_ids
                raise HTTPException(
                    status_code=404,
                    detail=f"Experiment(s) not found: {missing}",
                )

            exp_map = {r[0]: _experiment_row_to_dict(r) for r in rows}
            exp_a = exp_map[a_id]
            exp_b = exp_map[b_id]

            # Override with lag-specific KPIs when exec_lag is set
            if exec_lag is not None:
                cur.execute(
                    """
                    SELECT experiment_id, champion_accuracy, ceiling_accuracy,
                           gap_bps, n_dfu_months, model_distribution
                    FROM champion_experiment_lag
                    WHERE experiment_id IN (%s, %s) AND exec_lag = %s
                    """,
                    (a_id, b_id, exec_lag),
                )
                for r in cur.fetchall():
                    target = exp_a if r[0] == a_id else exp_b
                    _apply_lag_override(target, r[1:])

            # Overall comparison
            overall = _compute_overall_comparison(exp_a, exp_b)

            # Per-lag comparison (single query for both experiments)
            cur.execute(
                """
                SELECT experiment_id, exec_lag, champion_accuracy, ceiling_accuracy,
                       gap_bps, n_dfu_months, model_distribution
                FROM champion_experiment_lag
                WHERE experiment_id IN (%s, %s)
                ORDER BY exec_lag
                """,
                (a_id, b_id),
            )
            a_lags: dict[int, dict[str, Any]] = {}
            b_lags: dict[int, dict[str, Any]] = {}
            for r in cur.fetchall():
                d = _lag_row_to_dict(r[1:])  # skip experiment_id column
                if r[0] == a_id:
                    a_lags[r[1]] = d
                else:
                    b_lags[r[1]] = d

            per_lag = _compute_per_lag_comparison(a_lags, b_lags)

            # Per-month comparison (single query for both experiments)
            cur.execute(
                """
                SELECT experiment_id, month_start, champion_accuracy, ceiling_accuracy,
                       gap_bps, n_champions, model_distribution
                FROM champion_experiment_month
                WHERE experiment_id IN (%s, %s)
                ORDER BY month_start
                """,
                (a_id, b_id),
            )
            a_months: dict[str, dict[str, Any]] = {}
            b_months: dict[str, dict[str, Any]] = {}
            for r in cur.fetchall():
                d = _month_row_to_dict(r[1:])  # skip experiment_id column
                if r[0] == a_id:
                    a_months[str(r[1])] = d
                else:
                    b_months[str(r[1])] = d

            per_month = _compute_per_month_comparison(a_months, b_months)

            # Model distribution comparison
            model_dist = _compute_model_dist_comparison(
                exp_a.get("model_distribution") or {},
                exp_b.get("model_distribution") or {},
            )

            # Config diffs
            config_diffs = _compute_config_diffs(exp_a, exp_b)

            # Cache comparison
            try:
                cur.execute(
                    """
                    INSERT INTO champion_experiment_comparison
                        (experiment_a_id, experiment_b_id,
                         overall_comparison, per_lag_comparison,
                         per_month_comparison, model_dist_comparison, config_diffs)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (experiment_a_id, experiment_b_id) DO UPDATE SET
                        overall_comparison = EXCLUDED.overall_comparison,
                        per_lag_comparison = EXCLUDED.per_lag_comparison,
                        per_month_comparison = EXCLUDED.per_month_comparison,
                        model_dist_comparison = EXCLUDED.model_dist_comparison,
                        config_diffs = EXCLUDED.config_diffs,
                        created_at = NOW()
                    """,
                    (
                        a_id, b_id,
                        json.dumps(overall), json.dumps(per_lag),
                        json.dumps(per_month), json.dumps(model_dist),
                        json.dumps(config_diffs),
                    ),
                )
                conn.commit()
            except Exception:
                logger.warning("Failed to cache comparison for %d vs %d", a_id, b_id)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to compare champion experiments %d vs %d", a_id, b_id)
        raise HTTPException(status_code=500, detail="Failed to compare experiments") from None

    result = {
        "experiment_a_id": a_id,
        "experiment_b_id": b_id,
        "overall_comparison": overall,
        "per_lag_comparison": per_lag,
        "per_month_comparison": per_month,
        "model_dist_comparison": model_dist,
        "config_diffs": config_diffs,
        "source": "computed",
    }
    if exec_lag is not None:
        result["exec_lag_filter"] = exec_lag
    return result


# ---------------------------------------------------------------------------
# 6. GET /champion-experiments/{experiment_id} — Detail
# ---------------------------------------------------------------------------

@router.get("/{experiment_id}")
def get_experiment(
    experiment_id: int,
    response: FastAPIResponse,
    exec_lag: int | None = Query(default=None, ge=0, le=4),
):
    """Get a single champion experiment's details.

    When *exec_lag* (0-4) is provided, portfolio-level KPIs are overridden
    with lag-specific values from ``champion_experiment_lag``.
    """
    set_cache(response, max_age=_DETAIL_CACHE_TTL)

    sql = f"""
        SELECT {_SELECT_COLS}
        FROM champion_experiment
        WHERE experiment_id = %s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (experiment_id,))
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

            result = _experiment_row_to_dict(row)

            if exec_lag is not None:
                cur.execute(
                    """
                    SELECT champion_accuracy, ceiling_accuracy, gap_bps,
                           n_dfu_months, model_distribution
                    FROM champion_experiment_lag
                    WHERE experiment_id = %s AND exec_lag = %s
                    """,
                    (experiment_id, exec_lag),
                )
                lag_row = cur.fetchone()
                if lag_row:
                    _apply_lag_override(result, lag_row)
                result["exec_lag_filter"] = exec_lag

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to fetch champion experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Failed to fetch experiment") from None

    return result


# ---------------------------------------------------------------------------
# 7. GET /champion-experiments/{experiment_id}/lags — Per-lag breakdown
# ---------------------------------------------------------------------------

@router.get("/{experiment_id}/lags")
def get_experiment_lags(
    experiment_id: int,
    response: FastAPIResponse,
):
    """Get per-execution-lag breakdown for a champion experiment."""
    set_cache(response, max_age=_DETAIL_CACHE_TTL)

    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Verify exists
            cur.execute(
                "SELECT experiment_id FROM champion_experiment WHERE experiment_id = %s",
                (experiment_id,),
            )
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

            cur.execute(
                """
                SELECT exec_lag, champion_accuracy, ceiling_accuracy, gap_bps,
                       n_dfu_months, model_distribution
                FROM champion_experiment_lag
                WHERE experiment_id = %s
                ORDER BY exec_lag
                """,
                (experiment_id,),
            )
            rows = cur.fetchall()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to fetch lags for experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Failed to fetch experiment lags") from None

    return {
        "experiment_id": experiment_id,
        "lags": [_lag_row_to_dict(r) for r in rows],
    }


# ---------------------------------------------------------------------------
# 8. GET /champion-experiments/{experiment_id}/months — Per-month breakdown
# ---------------------------------------------------------------------------

@router.get("/{experiment_id}/months")
def get_experiment_months(
    experiment_id: int,
    response: FastAPIResponse,
):
    """Get per-month breakdown for a champion experiment."""
    set_cache(response, max_age=_DETAIL_CACHE_TTL)

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT experiment_id FROM champion_experiment WHERE experiment_id = %s",
                (experiment_id,),
            )
            if cur.fetchone() is None:
                raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

            cur.execute(
                """
                SELECT month_start, champion_accuracy, ceiling_accuracy, gap_bps,
                       n_champions, model_distribution
                FROM champion_experiment_month
                WHERE experiment_id = %s
                ORDER BY month_start
                """,
                (experiment_id,),
            )
            rows = cur.fetchall()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to fetch months for experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Failed to fetch experiment months") from None

    return {
        "experiment_id": experiment_id,
        "months": [_month_row_to_dict(r) for r in rows],
    }


# ---------------------------------------------------------------------------
# 9. GET /champion-experiments/{experiment_id}/logs — Incremental logs
# ---------------------------------------------------------------------------

@router.get("/{experiment_id}/logs")
def get_experiment_logs(
    experiment_id: int,
    offset: int = Query(default=0, ge=0),
):
    """Get experiment logs (offset-based streaming)."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT job_id, status FROM champion_experiment WHERE experiment_id = %s",
                (experiment_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

            job_id, status = row
            if not job_id:
                return {
                    "experiment_id": experiment_id,
                    "log": "",
                    "offset": 0,
                    "next_offset": 0,
                    "status": status,
                    "has_more": False,
                }

            from common.services.job_state import get_job_log
            full_log = get_job_log(job_id)
            chunk = full_log[offset:]
            next_offset = offset + len(chunk)

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to fetch logs for experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Failed to fetch logs") from None

    return {
        "experiment_id": experiment_id,
        "log": chunk,
        "offset": offset,
        "next_offset": next_offset,
        "status": status,
        "has_more": status in ("queued", "running"),
    }


# ---------------------------------------------------------------------------
# 10. POST /champion-experiments — Create + launch
# ---------------------------------------------------------------------------

@router.post("", status_code=202, dependencies=[Depends(require_api_key)])
def create_experiment(body: CreateChampionExperimentBody):
    """Create a new champion experiment and launch it as an async job."""
    # Validate strategy
    if body.strategy not in _VALID_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy '{body.strategy}'. Must be one of: {sorted(_VALID_STRATEGIES)}",
        )
    if body.metric not in _VALID_METRICS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric '{body.metric}'. Must be one of: {sorted(_VALID_METRICS)}",
        )
    if str(body.lag_mode) not in _VALID_LAG_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid lag_mode '{body.lag_mode}'. Must be one of: {sorted(_VALID_LAG_MODES)}",
        )
    if len(body.models) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 models required for champion competition",
        )

    strategy_params = body.strategy_params.model_dump(exclude_none=True) if body.strategy_params else {}
    meta_learner_params = body.meta_learner_params.model_dump() if body.meta_learner_params else None

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO champion_experiment
                    (label, notes, template_id, strategy, strategy_params,
                     meta_learner_params, models, metric, lag_mode, min_sku_rows)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING experiment_id
                """,
                (
                    body.label, body.notes, body.template,
                    body.strategy, json.dumps(strategy_params),
                    json.dumps(meta_learner_params) if meta_learner_params else None,
                    json.dumps(body.models), body.metric,
                    str(body.lag_mode), body.min_sku_rows,
                ),
            )
            experiment_id = cur.fetchone()[0]
            conn.commit()

        # Submit async job
        from common.services.job_registry import JobManager
        jm = JobManager()
        job_id = jm.submit_job(
            job_type="champion_experiment",
            params={"experiment_id": experiment_id},
            label=f"Champion Experiment: {body.label}",
        )

        # Store job_id
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE champion_experiment SET job_id = %s WHERE experiment_id = %s",
                (job_id, experiment_id),
            )
            conn.commit()

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to create champion experiment")
        raise HTTPException(status_code=500, detail="Failed to create experiment") from None

    return {
        "experiment_id": experiment_id,
        "job_id": job_id,
        "status": "queued",
        "strategy": body.strategy,
        "label": body.label,
    }


# ---------------------------------------------------------------------------
# 11. POST /champion-experiments/{experiment_id}/promote — Stage 1
# ---------------------------------------------------------------------------

@router.post("/{experiment_id}/promote", dependencies=[Depends(require_api_key)])
def promote_experiment(experiment_id: int):
    """Promote experiment strategy config to forecast_pipeline_config.yaml champion section."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {_SELECT_COLS} FROM champion_experiment WHERE experiment_id = %s",
                (experiment_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

            exp = _experiment_row_to_dict(row)
            if exp["status"] != "completed":
                raise HTTPException(
                    status_code=409,
                    detail=f"Cannot promote experiment with status '{exp['status']}' (must be completed)",
                )

            # Write to pipeline config champion section
            with open(_PIPELINE_CONFIG_PATH) as f:
                pipeline_cfg = yaml.safe_load(f) or {}

            # Backup
            backup_path = _PIPELINE_CONFIG_PATH.with_suffix(
                f".yaml.bak.{experiment_id}"
            )
            shutil.copy2(_PIPELINE_CONFIG_PATH, backup_path)

            # Update champion section in pipeline config
            new_config = copy.deepcopy(pipeline_cfg)
            champ = new_config.setdefault("champion", {})
            champ["strategy"] = exp["strategy"]
            champ["strategy_params"] = exp["strategy_params"] or {}
            champ["models"] = exp["models"] or []
            champ["metric"] = exp["metric"]
            champ["lag"] = exp["lag_mode"]
            champ["min_dfu_rows"] = exp["min_sku_rows"]
            if exp["meta_learner_params"]:
                champ["meta_learner"] = exp["meta_learner_params"]

            with open(_PIPELINE_CONFIG_PATH, "w") as f:
                yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)

            reset_config("forecast_pipeline_config.yaml")
            config_written = "forecast_pipeline_config.yaml"

            # Find previous promoted
            cur.execute(
                "SELECT experiment_id FROM champion_experiment "
                "WHERE is_promoted = TRUE AND experiment_id != %s",
                (experiment_id,),
            )
            prev_row = cur.fetchone()
            previous_id = prev_row[0] if prev_row else None

            # Clear previous promoted flags
            cur.execute(
                "UPDATE champion_experiment SET is_promoted = FALSE, promoted_at = NULL "
                "WHERE is_promoted = TRUE AND experiment_id != %s",
                (experiment_id,),
            )

            # Set new promoted
            cur.execute(
                "UPDATE champion_experiment SET is_promoted = TRUE, promoted_at = NOW() "
                "WHERE experiment_id = %s",
                (experiment_id,),
            )

            # Audit log — include which config file was written
            snapshot = copy.deepcopy(new_config)
            snapshot["_config_written"] = config_written
            cur.execute(
                """
                INSERT INTO champion_promotion_log
                    (experiment_id, strategy, champion_accuracy,
                     previous_experiment_id, config_snapshot)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    experiment_id, exp["strategy"], exp["champion_accuracy"],
                    previous_id, json.dumps(snapshot),
                ),
            )
            conn.commit()

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to promote champion experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Failed to promote experiment") from None

    return {
        "promoted": True,
        "experiment_id": experiment_id,
        "strategy": exp["strategy"],
        "champion_accuracy": exp["champion_accuracy"],
        "previous_experiment_id": previous_id,
        "backup_path": backup_path.name,
        "config_written": config_written,
    }


# ---------------------------------------------------------------------------
# 12. POST /champion-experiments/{experiment_id}/promote-results — Stage 2
# ---------------------------------------------------------------------------

@router.post("/{experiment_id}/promote-results", status_code=201, dependencies=[Depends(require_api_key)])
def promote_results(experiment_id: int):
    """Submit job to run champion selection and load results into forecast tables."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT status, is_promoted, is_results_promoted "
                "FROM champion_experiment WHERE experiment_id = %s",
                (experiment_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

            status, is_promoted, is_results_promoted = row
            if status != "completed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot load results for experiment with status '{status}'",
                )
            if not is_promoted:
                raise HTTPException(
                    status_code=400,
                    detail="Experiment must be promoted (Stage 1) before loading results",
                )
            if is_results_promoted:
                raise HTTPException(
                    status_code=409,
                    detail="Results already loaded for this experiment",
                )

        # Submit job
        from common.services.job_registry import JobManager
        jm = JobManager()
        job_id = jm.submit_job(
            job_type="champion_results_load",
            params={"experiment_id": experiment_id},
            label=f"Load Champion Results (exp #{experiment_id})",
        )

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE champion_experiment SET results_promote_job_id = %s "
                "WHERE experiment_id = %s",
                (job_id, experiment_id),
            )
            conn.commit()

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to submit results load for experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Failed to submit results load") from None

    return {
        "job_id": job_id,
        "experiment_id": experiment_id,
        "message": "Champion results load job submitted",
    }


# ---------------------------------------------------------------------------
# 13. GET /champion-experiments/{experiment_id}/promote-results/status
# ---------------------------------------------------------------------------

@router.get("/{experiment_id}/promote-results/status")
def get_promote_results_status(experiment_id: int):
    """Check status of results promotion job."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT is_results_promoted, results_promoted_at, results_promote_job_id "
                "FROM champion_experiment WHERE experiment_id = %s",
                (experiment_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

            is_done, done_at, job_id = row

            result: dict[str, Any] = {
                "experiment_id": experiment_id,
                "is_results_promoted": bool(is_done),
                "results_promoted_at": str(done_at) if done_at else None,
            }

            if job_id:
                from common.services.job_registry import JobManager
                job_info = JobManager._db_get(job_id)
                if job_info:
                    result["status"] = job_info.get("status", "unknown")
                    result["progress_pct"] = job_info.get("progress_pct")
                    result["progress_msg"] = job_info.get("progress_msg")
                    result["error"] = job_info.get("error")
                else:
                    result["status"] = "completed" if is_done else "unknown"
            else:
                result["status"] = "not_started"

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to get results status for experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Failed to get results status")

    return result


# ---------------------------------------------------------------------------
# 14. POST /champion-experiments/{experiment_id}/cancel
# ---------------------------------------------------------------------------

@router.post("/{experiment_id}/cancel", dependencies=[Depends(require_api_key)])
def cancel_experiment(experiment_id: int):
    """Cancel a running or queued champion experiment."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT status, job_id FROM champion_experiment WHERE experiment_id = %s",
                (experiment_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

            status, job_id = row
            if status not in ("queued", "running"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot cancel experiment with status '{status}'",
                )

            # Cancel via JobManager if running
            if job_id:
                try:
                    from common.services.job_registry import JobManager
                    jm = JobManager()
                    jm.cancel_job(job_id)
                except Exception:
                    logger.warning("Failed to cancel job %s via JobManager", job_id)

            cur.execute(
                "UPDATE champion_experiment SET status = 'cancelled', completed_at = NOW() "
                "WHERE experiment_id = %s",
                (experiment_id,),
            )
            conn.commit()

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to cancel champion experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Failed to cancel experiment")

    return {
        "cancelled": True,
        "experiment_id": experiment_id,
        "previous_status": status,
    }


# ---------------------------------------------------------------------------
# 15. DELETE /champion-experiments/{experiment_id}
# ---------------------------------------------------------------------------

@router.delete("/{experiment_id}", dependencies=[Depends(require_api_key)])
def delete_experiment(experiment_id: int):
    """Delete a champion experiment (not running/queued/promoted)."""
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT status, is_promoted FROM champion_experiment WHERE experiment_id = %s",
                (experiment_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

            status, is_promoted = row
            if status in ("queued", "running"):
                raise HTTPException(
                    status_code=409,
                    detail=f"Cannot delete experiment with status '{status}' (cancel first)",
                )
            if is_promoted:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Cannot delete promoted experiment {experiment_id}. "
                        "Demote it first (promote a different experiment, or call "
                        "the demote endpoint)."
                    ),
                )

            # Clean up referencing tables before deleting experiment
            cur.execute(
                "DELETE FROM champion_promotion_log WHERE experiment_id = %s",
                (experiment_id,),
            )
            # CASCADE deletes lag, month, comparison rows
            cur.execute(
                "DELETE FROM champion_experiment WHERE experiment_id = %s",
                (experiment_id,),
            )
            conn.commit()

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to delete champion experiment %d", experiment_id)
        raise HTTPException(status_code=500, detail="Failed to delete experiment")

    return {"deleted": True, "experiment_id": experiment_id}


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _lag_row_to_dict(row: tuple) -> dict[str, Any]:
    return {
        "exec_lag": int(row[0]),
        "champion_accuracy": float(row[1]) if row[1] is not None else None,
        "ceiling_accuracy": float(row[2]) if row[2] is not None else None,
        "gap_bps": float(row[3]) if row[3] is not None else None,
        "n_dfu_months": int(row[4]) if row[4] is not None else None,
        "model_distribution": _parse_json(row[5]),
    }


def _month_row_to_dict(row: tuple) -> dict[str, Any]:
    return {
        "month_start": str(row[0]),
        "champion_accuracy": float(row[1]) if row[1] is not None else None,
        "ceiling_accuracy": float(row[2]) if row[2] is not None else None,
        "gap_bps": float(row[3]) if row[3] is not None else None,
        "n_champions": int(row[4]) if row[4] is not None else None,
        "model_distribution": _parse_json(row[5]),
    }


def _safe_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(b - a, 4)


def _compute_overall_comparison(a: dict, b: dict) -> dict[str, Any]:
    delta_acc = _safe_delta(a["champion_accuracy"], b["champion_accuracy"])
    delta_ceil = _safe_delta(a["ceiling_accuracy"], b["ceiling_accuracy"])
    delta_gap = _safe_delta(a["gap_bps"], b["gap_bps"])

    verdict = "mixed"
    if delta_acc is not None:
        if delta_acc > 5:  # > 5 bps improvement
            verdict = "b_better"
        elif delta_acc < -5:
            verdict = "a_better"

    return {
        "experiment_a": {
            "champion_accuracy": a["champion_accuracy"],
            "ceiling_accuracy": a["ceiling_accuracy"],
            "gap_bps": a["gap_bps"],
            "n_dfu_months": a["n_dfu_months"],
        },
        "experiment_b": {
            "champion_accuracy": b["champion_accuracy"],
            "ceiling_accuracy": b["ceiling_accuracy"],
            "gap_bps": b["gap_bps"],
            "n_dfu_months": b["n_dfu_months"],
        },
        "delta_champion_accuracy": delta_acc,
        "delta_ceiling_accuracy": delta_ceil,
        "delta_gap_bps": delta_gap,
        "verdict": verdict,
    }


def _compute_per_lag_comparison(
    a_lags: dict[int, dict], b_lags: dict[int, dict],
) -> list[dict[str, Any]]:
    all_lags = sorted(set(a_lags.keys()) | set(b_lags.keys()))
    result = []
    for lag in all_lags:
        a = a_lags.get(lag, {})
        b = b_lags.get(lag, {})
        result.append({
            "exec_lag": lag,
            "a_champion_accuracy": a.get("champion_accuracy"),
            "b_champion_accuracy": b.get("champion_accuracy"),
            "delta_accuracy": _safe_delta(
                a.get("champion_accuracy"), b.get("champion_accuracy"),
            ),
            "a_gap_bps": a.get("gap_bps"),
            "b_gap_bps": b.get("gap_bps"),
        })
    return result


def _compute_per_month_comparison(
    a_months: dict[str, dict], b_months: dict[str, dict],
) -> list[dict[str, Any]]:
    all_months = sorted(set(a_months.keys()) | set(b_months.keys()))
    result = []
    for month in all_months:
        a = a_months.get(month, {})
        b = b_months.get(month, {})
        result.append({
            "month_start": month,
            "a_champion_accuracy": a.get("champion_accuracy"),
            "b_champion_accuracy": b.get("champion_accuracy"),
            "delta_accuracy": _safe_delta(
                a.get("champion_accuracy"), b.get("champion_accuracy"),
            ),
        })
    return result


def _compute_model_dist_comparison(
    a_dist: dict[str, float], b_dist: dict[str, float],
) -> list[dict[str, Any]]:
    all_models = sorted(set(a_dist.keys()) | set(b_dist.keys()))
    return [
        {
            "model_id": m,
            "a_pct": a_dist.get(m, 0),
            "b_pct": b_dist.get(m, 0),
            "delta_pct": round(b_dist.get(m, 0) - a_dist.get(m, 0), 2),
        }
        for m in all_models
    ]


def _compute_config_diffs(a: dict, b: dict) -> list[dict[str, Any]]:
    keys = ["strategy", "strategy_params", "meta_learner_params",
            "models", "metric", "lag_mode", "min_sku_rows"]
    diffs = []
    for k in keys:
        av = a.get(k)
        bv = b.get(k)
        if av != bv:
            diffs.append({"key": k, "a": av, "b": bv})
    return diffs
