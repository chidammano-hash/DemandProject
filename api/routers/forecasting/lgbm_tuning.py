"""LGBM Tuning endpoints — run tracking, comparison, and analysis."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field

from api.core import get_conn, set_cache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["lgbm-tuning"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class CreateRunBody(BaseModel):
    run_label: str = Field(min_length=1, max_length=200)
    model_id: str = Field(default="lgbm_cluster", max_length=120)
    params: dict[str, Any] | None = None
    features: list[str] | None = None
    notes: str | None = None


class UpdateRunBody(BaseModel):
    status: str | None = Field(default=None, pattern=r"^(running|completed|failed)$")
    completed_at: str | None = None
    accuracy_pct: float | None = None
    wape: float | None = None
    bias: float | None = None
    n_predictions: int | None = None
    n_dfus: int | None = None
    metadata: dict[str, Any] | None = None
    notes: str | None = None
    backup_path: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/lgbm-tuning/runs")
def list_runs(
    response: FastAPIResponse,
    status: str = Query(default="", max_length=20),
    model_id: str = Query(default="", max_length=120),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List LGBM tuning runs, newest first."""
    set_cache(response, max_age=30)

    parts: list[str] = []
    params: list[Any] = []
    if status.strip():
        parts.append("status = %s")
        params.append(status.strip())
    if model_id.strip():
        parts.append("model_id = %s")
        params.append(model_id.strip())

    where_sql = f"WHERE {' AND '.join(parts)}" if parts else ""

    sql = f"""
        SELECT run_id, run_label, model_id, started_at, completed_at,
               status, accuracy_pct, wape, bias, n_predictions, n_dfus, notes,
               is_promoted, promoted_at
        FROM lgbm_tuning_run
        {where_sql}
        ORDER BY started_at DESC
        LIMIT %s OFFSET %s
    """
    params.extend([limit, offset])

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    except Exception:
        logger.exception("Failed to list tuning runs")
        raise HTTPException(status_code=500, detail="Failed to list tuning runs")

    runs = []
    for r in rows:
        runs.append({
            "run_id": r[0],
            "run_label": r[1],
            "model_id": r[2],
            "started_at": str(r[3]) if r[3] else None,
            "completed_at": str(r[4]) if r[4] else None,
            "status": r[5],
            "accuracy_pct": float(r[6]) if r[6] is not None else None,
            "wape": float(r[7]) if r[7] is not None else None,
            "bias": float(r[8]) if r[8] is not None else None,
            "n_predictions": int(r[9]) if r[9] is not None else None,
            "n_dfus": int(r[10]) if r[10] is not None else None,
            "notes": r[11],
            "is_promoted": bool(r[12]),
            "promoted_at": str(r[13]) if r[13] else None,
        })
    return {"runs": runs}


@router.get("/lgbm-tuning/runs/{run_id}")
def get_run(run_id: int, response: FastAPIResponse):
    """Get full detail for a single tuning run, including timeframe breakdowns."""
    set_cache(response, max_age=30)

    run_sql = """
        SELECT run_id, run_label, model_id, started_at, completed_at,
               status, params, feature_count, features,
               accuracy_pct, wape, bias, n_predictions, n_dfus,
               metadata, notes, backup_path
        FROM lgbm_tuning_run
        WHERE run_id = %s
    """
    tf_sql = """
        SELECT id, run_id, timeframe, train_end, predict_start, predict_end,
               n_predictions, accuracy_pct, wape, bias
        FROM lgbm_tuning_timeframe
        WHERE run_id = %s
        ORDER BY timeframe
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(run_sql, [run_id])
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")

        cur.execute(tf_sql, [run_id])
        tf_rows = cur.fetchall()

    def _parse_json(val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, (dict, list)):
            return val
        return json.loads(val)

    run = {
        "run_id": row[0],
        "run_label": row[1],
        "model_id": row[2],
        "started_at": str(row[3]) if row[3] else None,
        "completed_at": str(row[4]) if row[4] else None,
        "status": row[5],
        "params": _parse_json(row[6]),
        "feature_count": row[7],
        "features": _parse_json(row[8]),
        "accuracy_pct": float(row[9]) if row[9] is not None else None,
        "wape": float(row[10]) if row[10] is not None else None,
        "bias": float(row[11]) if row[11] is not None else None,
        "n_predictions": int(row[12]) if row[12] is not None else None,
        "n_dfus": int(row[13]) if row[13] is not None else None,
        "metadata": _parse_json(row[14]),
        "notes": row[15],
        "backup_path": row[16],
    }

    timeframes = []
    for tf in tf_rows:
        timeframes.append({
            "id": tf[0],
            "run_id": tf[1],
            "timeframe": tf[2],
            "train_end": str(tf[3]) if tf[3] else None,
            "predict_start": str(tf[4]) if tf[4] else None,
            "predict_end": str(tf[5]) if tf[5] else None,
            "n_predictions": int(tf[6]) if tf[6] is not None else None,
            "accuracy_pct": float(tf[7]) if tf[7] is not None else None,
            "wape": float(tf[8]) if tf[8] is not None else None,
            "bias": float(tf[9]) if tf[9] is not None else None,
        })

    return {**run, "timeframes": timeframes}


@router.post("/lgbm-tuning/runs", status_code=201)
def create_run(body: CreateRunBody):
    """Register a new tuning run."""
    sql = """
        INSERT INTO lgbm_tuning_run (run_label, model_id, params, feature_count, features, notes)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING run_id
    """
    params_json = json.dumps(body.params) if body.params else None
    features_json = json.dumps(body.features) if body.features else None
    feature_count = len(body.features) if body.features else None

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, [
            body.run_label,
            body.model_id,
            params_json,
            feature_count,
            features_json,
            body.notes,
        ])
        row = cur.fetchone()
        conn.commit()

    return {"run_id": row[0]}


@router.put("/lgbm-tuning/runs/{run_id}")
def update_run(run_id: int, body: UpdateRunBody):
    """Update an existing tuning run (e.g. mark completed with results)."""
    set_parts: list[str] = []
    params: list[Any] = []

    if body.status is not None:
        set_parts.append("status = %s")
        params.append(body.status)
    if body.completed_at is not None:
        set_parts.append("completed_at = %s::timestamptz")
        params.append(body.completed_at)
    if body.accuracy_pct is not None:
        set_parts.append("accuracy_pct = %s")
        params.append(body.accuracy_pct)
    if body.wape is not None:
        set_parts.append("wape = %s")
        params.append(body.wape)
    if body.bias is not None:
        set_parts.append("bias = %s")
        params.append(body.bias)
    if body.n_predictions is not None:
        set_parts.append("n_predictions = %s")
        params.append(body.n_predictions)
    if body.n_dfus is not None:
        set_parts.append("n_dfus = %s")
        params.append(body.n_dfus)
    if body.metadata is not None:
        set_parts.append("metadata = %s")
        params.append(json.dumps(body.metadata))
    if body.notes is not None:
        set_parts.append("notes = %s")
        params.append(body.notes)
    if body.backup_path is not None:
        set_parts.append("backup_path = %s")
        params.append(body.backup_path)

    if not set_parts:
        raise HTTPException(status_code=400, detail="No fields to update")

    params.append(run_id)
    sql = f"UPDATE lgbm_tuning_run SET {', '.join(set_parts)} WHERE run_id = %s"

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        conn.commit()

    return {"updated": True, "run_id": run_id}


@router.get("/lgbm-tuning/compare")
def compare_runs(
    response: FastAPIResponse,
    baseline_id: int = Query(ge=1),
    candidate_id: int = Query(ge=1),
):
    """Compare two tuning runs and return delta metrics."""
    set_cache(response, max_age=60)

    run_sql = """
        SELECT run_id, run_label, model_id, accuracy_pct, wape, bias,
               n_predictions, n_dfus, status, params, features, feature_count, metadata
        FROM lgbm_tuning_run
        WHERE run_id = %s
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(run_sql, [baseline_id])
        baseline = cur.fetchone()
        if baseline is None:
            raise HTTPException(status_code=404, detail="Baseline run not found")

        cur.execute(run_sql, [candidate_id])
        candidate = cur.fetchone()
        if candidate is None:
            raise HTTPException(status_code=404, detail="Candidate run not found")

        # Check for existing comparison
        cur.execute(
            "SELECT id FROM lgbm_tuning_comparison WHERE baseline_run_id = %s AND candidate_run_id = %s",
            [baseline_id, candidate_id],
        )
        existing = cur.fetchone()

    def _parse_json(val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, (dict, list)):
            return val
        return json.loads(val)

    def _run_dict(r: tuple) -> dict[str, Any]:
        return {
            "run_id": r[0],
            "run_label": r[1],
            "model_id": r[2],
            "accuracy_pct": float(r[3]) if r[3] is not None else None,
            "wape": float(r[4]) if r[4] is not None else None,
            "bias": float(r[5]) if r[5] is not None else None,
            "n_predictions": int(r[6]) if r[6] is not None else None,
            "n_dfus": int(r[7]) if r[7] is not None else None,
            "status": r[8],
            "params": _parse_json(r[9]),
            "features": _parse_json(r[10]),
            "feature_count": int(r[11]) if r[11] is not None else None,
            "metadata": _parse_json(r[12]),
        }

    b = _run_dict(baseline)
    c = _run_dict(candidate)

    delta_acc = None
    delta_wape = None
    delta_bias = None
    verdict = "neutral"
    if b["accuracy_pct"] is not None and c["accuracy_pct"] is not None:
        delta_acc = round(c["accuracy_pct"] - b["accuracy_pct"], 2)
        if delta_acc >= 0.05:
            verdict = "improved"
        elif delta_acc <= -0.05:
            verdict = "degraded"
    if b["wape"] is not None and c["wape"] is not None:
        delta_wape = round(c["wape"] - b["wape"], 2)
    if b["bias"] is not None and c["bias"] is not None:
        delta_bias = round(c["bias"] - b["bias"], 4)

    # Fetch cluster and month breakdowns for both runs
    cluster_sql = """
        SELECT cluster_type, cluster_value, n_predictions, n_dfus,
               accuracy_pct, wape, bias
        FROM lgbm_tuning_cluster
        WHERE run_id = %s
        ORDER BY cluster_type, cluster_value
    """
    month_sql = """
        SELECT month_start, n_predictions, n_dfus,
               accuracy_pct, wape, bias
        FROM lgbm_tuning_month
        WHERE run_id = %s
        ORDER BY month_start
    """

    def _cluster_rows(cur_inner: Any, run_id: int) -> list[dict[str, Any]]:
        cur_inner.execute(cluster_sql, [run_id])
        return [
            {
                "cluster_type": r[0], "cluster_value": r[1],
                "n_predictions": r[2], "n_dfus": r[3],
                "accuracy_pct": float(r[4]) if r[4] is not None else None,
                "wape": float(r[5]) if r[5] is not None else None,
                "bias": float(r[6]) if r[6] is not None else None,
            }
            for r in cur_inner.fetchall()
        ]

    def _month_rows(cur_inner: Any, run_id: int) -> list[dict[str, Any]]:
        cur_inner.execute(month_sql, [run_id])
        return [
            {
                "month_start": str(r[0]),
                "n_predictions": r[1], "n_dfus": r[2],
                "accuracy_pct": float(r[3]) if r[3] is not None else None,
                "wape": float(r[4]) if r[4] is not None else None,
                "bias": float(r[5]) if r[5] is not None else None,
            }
            for r in cur_inner.fetchall()
        ]

    with get_conn() as conn2, conn2.cursor() as cur2:
        base_clusters = _cluster_rows(cur2, baseline_id)
        cand_clusters = _cluster_rows(cur2, candidate_id)
        base_months = _month_rows(cur2, baseline_id)
        cand_months = _month_rows(cur2, candidate_id)

    # Build per-cluster comparison (grouped by cluster_type)
    per_cluster: dict[str, list[dict[str, Any]]] = {}
    for ct in ("ml_cluster", "business_cluster"):
        b_map = {r["cluster_value"]: r for r in base_clusters if r["cluster_type"] == ct}
        c_map = {r["cluster_value"]: r for r in cand_clusters if r["cluster_type"] == ct}
        all_vals = sorted(set(b_map.keys()) | set(c_map.keys()))
        items = []
        for val in all_vals:
            br = b_map.get(val, {})
            cr = c_map.get(val, {})
            b_a = br.get("accuracy_pct")
            c_a = cr.get("accuracy_pct")
            items.append({
                "cluster": val,
                "baseline_accuracy": b_a,
                "candidate_accuracy": c_a,
                "delta_accuracy": round(c_a - b_a, 2) if b_a is not None and c_a is not None else None,
                "baseline_wape": br.get("wape"),
                "candidate_wape": cr.get("wape"),
                "baseline_n_dfus": br.get("n_dfus"),
                "candidate_n_dfus": cr.get("n_dfus"),
            })
        per_cluster[ct] = items

    # Build per-month comparison
    b_month_map = {r["month_start"]: r for r in base_months}
    c_month_map = {r["month_start"]: r for r in cand_months}
    all_months = sorted(set(b_month_map.keys()) | set(c_month_map.keys()))
    per_month = []
    for m in all_months:
        br = b_month_map.get(m, {})
        cr = c_month_map.get(m, {})
        b_a = br.get("accuracy_pct")
        c_a = cr.get("accuracy_pct")
        per_month.append({
            "month": m,
            "baseline_accuracy": b_a,
            "candidate_accuracy": c_a,
            "delta_accuracy": round(c_a - b_a, 2) if b_a is not None and c_a is not None else None,
            "baseline_wape": br.get("wape"),
            "candidate_wape": cr.get("wape"),
        })

    # Build parameter comparison — diffs and common values
    param_diffs: list[dict[str, Any]] = []
    param_common: list[dict[str, Any]] = []
    b_params = b.get("params") or {}
    c_params = c.get("params") or {}
    all_keys = sorted(set(b_params.keys()) | set(c_params.keys()))
    for key in all_keys:
        bv = b_params.get(key)
        cv = c_params.get(key)
        if bv != cv:
            param_diffs.append({"param": key, "baseline": bv, "candidate": cv})
        else:
            param_common.append({"param": key, "value": bv})

    # Build feature diff — added / removed / common count
    b_features = set(b.get("features") or [])
    c_features = set(c.get("features") or [])
    features_added = sorted(c_features - b_features)
    features_removed = sorted(b_features - c_features)
    features_common_count = len(b_features & c_features)
    feature_diffs = {
        "baseline_count": b.get("feature_count") or len(b_features),
        "candidate_count": c.get("feature_count") or len(c_features),
        "added": features_added,
        "removed": features_removed,
        "common_count": features_common_count,
    }

    # Build config diff — non-hyperparameter settings from metadata
    _CONFIG_KEYS = [
        "cluster_strategy", "recursive", "shap_select", "shap_threshold",
        "shap_top_n", "shap_sample_size", "tune_inline", "params_source",
    ]
    config_diffs: list[dict[str, Any]] = []
    config_common: list[dict[str, Any]] = []
    b_meta = b.get("metadata") or {}
    c_meta = c.get("metadata") or {}
    for key in _CONFIG_KEYS:
        bv = b_meta.get(key)
        cv = c_meta.get(key)
        if bv is None and cv is None:
            continue
        if bv != cv:
            config_diffs.append({"setting": key, "baseline": bv, "candidate": cv})
        else:
            config_common.append({"setting": key, "value": bv})

    return {
        "baseline": b,
        "candidate": c,
        "delta_accuracy": delta_acc,
        "delta_wape": delta_wape,
        "delta_bias": delta_bias,
        "verdict": verdict,
        "existing_comparison_id": existing[0] if existing else None,
        "param_diffs": param_diffs,
        "param_common": param_common,
        "feature_diffs": feature_diffs,
        "config_diffs": config_diffs,
        "config_common": config_common,
        "per_cluster": per_cluster,
        "per_month": per_month,
        "baseline_has_breakdowns": len(base_clusters) > 0 or len(base_months) > 0,
        "candidate_has_breakdowns": len(cand_clusters) > 0 or len(cand_months) > 0,
    }


@router.get("/lgbm-tuning/runs/{run_id}/clusters")
def get_run_clusters(run_id: int, response: FastAPIResponse):
    """Get per-cluster accuracy breakdowns for a single run."""
    set_cache(response, max_age=60)

    sql = """
        SELECT cluster_type, cluster_value, n_predictions, n_dfus,
               accuracy_pct, wape, bias
        FROM lgbm_tuning_cluster
        WHERE run_id = %s
        ORDER BY cluster_type, accuracy_pct DESC NULLS LAST
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, [run_id])
        rows = cur.fetchall()

    clusters: dict[str, list[dict[str, Any]]] = {"ml_cluster": [], "business_cluster": []}
    for r in rows:
        entry = {
            "cluster_value": r[1],
            "n_predictions": r[2], "n_dfus": r[3],
            "accuracy_pct": float(r[4]) if r[4] is not None else None,
            "wape": float(r[5]) if r[5] is not None else None,
            "bias": float(r[6]) if r[6] is not None else None,
        }
        ct = r[0]
        if ct in clusters:
            clusters[ct].append(entry)
    return {"run_id": run_id, "clusters": clusters}


@router.get("/lgbm-tuning/runs/{run_id}/months")
def get_run_months(run_id: int, response: FastAPIResponse):
    """Get per-month accuracy breakdowns for a single run."""
    set_cache(response, max_age=60)

    sql = """
        SELECT month_start, n_predictions, n_dfus,
               accuracy_pct, wape, bias
        FROM lgbm_tuning_month
        WHERE run_id = %s
        ORDER BY month_start
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, [run_id])
        rows = cur.fetchall()

    months = [
        {
            "month_start": str(r[0]),
            "n_predictions": r[1], "n_dfus": r[2],
            "accuracy_pct": float(r[3]) if r[3] is not None else None,
            "wape": float(r[4]) if r[4] is not None else None,
            "bias": float(r[5]) if r[5] is not None else None,
        }
        for r in rows
    ]
    return {"run_id": run_id, "months": months}


# ---------------------------------------------------------------------------
# Promote to production
# ---------------------------------------------------------------------------

# Known LGBM hyperparameter keys that belong in algorithm_config.yaml
_LGBM_PARAM_KEYS = {
    "n_estimators", "learning_rate", "num_leaves", "min_child_samples",
    "max_depth", "min_gain_to_split", "subsample", "bagging_freq",
    "colsample_bytree", "feature_fraction_bynode", "reg_lambda", "reg_alpha",
    "path_smooth",
}

_ALGO_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "algorithm_config.yaml"


@router.post("/lgbm-tuning/runs/{run_id}/promote")
def promote_run(run_id: int):
    """Promote a tuning run to production — writes params to algorithm_config.yaml."""
    # 1. Fetch run
    sql = """
        SELECT run_id, run_label, status, params, accuracy_pct, backup_path
        FROM lgbm_tuning_run
        WHERE run_id = %s
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, [run_id])
        row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")

    status = row[2]
    if status != "completed":
        raise HTTPException(status_code=400, detail=f"Cannot promote run with status '{status}' — only completed runs")

    params_raw = row[3]
    if params_raw is None:
        raise HTTPException(status_code=400, detail="Run has no params to promote")

    params = params_raw if isinstance(params_raw, dict) else json.loads(params_raw)

    # 2. Filter to known LGBM keys
    lgbm_overrides = {k: v for k, v in params.items() if k in _LGBM_PARAM_KEYS}
    if not lgbm_overrides:
        raise HTTPException(status_code=400, detail="Run params contain no recognized LGBM hyperparameters")

    # 3. Write to algorithm_config.yaml
    try:
        with open(_ALGO_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        lgbm_section = cfg["algorithms"]["lgbm"]
        old_params = {k: lgbm_section.get(k) for k in lgbm_overrides}
        for key, value in lgbm_overrides.items():
            lgbm_section[key] = value

        with open(_ALGO_CONFIG_PATH, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    except (OSError, KeyError, yaml.YAMLError) as exc:
        logger.exception("Failed to write algorithm_config.yaml during promote")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {exc}")

    # 4. Atomically clear previous promoted run and set new one
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE lgbm_tuning_run SET is_promoted = FALSE, promoted_at = NULL WHERE is_promoted = TRUE AND run_id != %s",
            [run_id],
        )
        cur.execute(
            "UPDATE lgbm_tuning_run SET is_promoted = TRUE, promoted_at = NOW() WHERE run_id = %s",
            [run_id],
        )
        conn.commit()

    logger.info("Promoted run #%d to production. Overrides: %s", run_id, lgbm_overrides)

    return {
        "promoted": True,
        "run_id": run_id,
        "run_label": row[1],
        "accuracy_pct": float(row[4]) if row[4] is not None else None,
        "params_written": lgbm_overrides,
        "old_params": old_params,
    }


@router.get("/lgbm-tuning/promoted")
def get_promoted(response: FastAPIResponse):
    """Return the currently promoted run (if any)."""
    set_cache(response, max_age=30)

    sql = """
        SELECT run_id, run_label, model_id, accuracy_pct, wape, bias,
               promoted_at, params
        FROM lgbm_tuning_run
        WHERE is_promoted = TRUE
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()

    if row is None:
        return {"promoted": None}

    return {
        "promoted": {
            "run_id": row[0],
            "run_label": row[1],
            "model_id": row[2],
            "accuracy_pct": float(row[3]) if row[3] is not None else None,
            "wape": float(row[4]) if row[4] is not None else None,
            "bias": float(row[5]) if row[5] is not None else None,
            "promoted_at": str(row[6]) if row[6] else None,
            "params": row[7] if isinstance(row[7], dict) else json.loads(row[7]) if row[7] else None,
        },
    }


@router.get("/lgbm-tuning/comparisons")
def list_comparisons(
    response: FastAPIResponse,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List saved pairwise comparisons."""
    set_cache(response, max_age=30)

    sql = """
        SELECT c.id, c.baseline_run_id, c.candidate_run_id, c.created_at,
               c.delta_accuracy, c.delta_wape, c.delta_bias, c.verdict,
               b.run_label AS baseline_label, d.run_label AS candidate_label
        FROM lgbm_tuning_comparison c
        JOIN lgbm_tuning_run b ON b.run_id = c.baseline_run_id
        JOIN lgbm_tuning_run d ON d.run_id = c.candidate_run_id
        ORDER BY c.created_at DESC
        LIMIT %s OFFSET %s
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, [limit, offset])
        rows = cur.fetchall()

    comparisons = []
    for r in rows:
        comparisons.append({
            "id": r[0],
            "baseline_run_id": r[1],
            "candidate_run_id": r[2],
            "created_at": str(r[3]) if r[3] else None,
            "delta_accuracy": float(r[4]) if r[4] is not None else None,
            "delta_wape": float(r[5]) if r[5] is not None else None,
            "delta_bias": float(r[6]) if r[6] is not None else None,
            "verdict": r[7],
            "baseline_label": r[8],
            "candidate_label": r[9],
        })
    return {"comparisons": comparisons}
