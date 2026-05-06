"""GET /{model}/compare — compare two tuning experiments."""
from __future__ import annotations

import logging
from typing import Any

import psycopg
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache

from ._helpers import _CONFIG_KEYS, _compare_row_to_dict, _validate_model

logger = logging.getLogger(__name__)

router = APIRouter(tags=["model-tuning"])


@router.get("/{model}/compare")
def compare_experiments(
    model: str,
    response: FastAPIResponse,
    baseline_id: int = Query(ge=1),
    candidate_id: int = Query(ge=1),
    exec_lag: int | None = Query(default=None, ge=0, le=4),
):
    """Compare two tuning runs and return delta metrics, with optional exec_lag filtering."""
    _validate_model(model)
    set_cache(response, max_age=60)

    if baseline_id == candidate_id:
        raise HTTPException(status_code=400, detail="Baseline and candidate must be different runs")

    run_sql = """
        SELECT r.run_id, r.run_label, r.model_id, r.accuracy_pct, r.wape, r.bias,
               r.n_predictions, r.n_dfus, r.status, r.params, r.features, r.feature_count,
               r.metadata, r.cluster_source, r.cluster_experiment_id,
               ce.label AS cluster_experiment_label
        FROM lgbm_tuning_run r
        LEFT JOIN cluster_experiment ce ON ce.experiment_id = r.cluster_experiment_id
        WHERE r.run_id = %s
    """

    try:
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
                "SELECT id FROM lgbm_tuning_comparison "
                "WHERE baseline_run_id = %s AND candidate_run_id = %s",
                [baseline_id, candidate_id],
            )
            existing = cur.fetchone()
    except HTTPException:
        raise
    except psycopg.Error:
        logger.exception("Failed to compare runs %d vs %d", baseline_id, candidate_id)
        raise HTTPException(status_code=500, detail="Failed to compare experiments")

    b = _compare_row_to_dict(baseline)
    c = _compare_row_to_dict(candidate)

    # If exec_lag specified, override portfolio metrics with lag-specific metrics
    if exec_lag is not None:
        _apply_lag_metrics(b, baseline_id, exec_lag)
        _apply_lag_metrics(c, candidate_id, exec_lag)

    delta_acc = _safe_delta(b["accuracy_pct"], c["accuracy_pct"])
    delta_wape = _safe_delta(b["wape"], c["wape"])
    delta_bias = _safe_delta(b["bias"], c["bias"], precision=4)

    verdict = "neutral"
    if delta_acc is not None:
        if delta_acc >= 0.05:
            verdict = "improved"
        elif delta_acc <= -0.05:
            verdict = "degraded"

    # Fetch per-lag comparison (always, regardless of exec_lag filter)
    per_lag = _build_per_lag_comparison(baseline_id, candidate_id)

    # Fetch cluster and month breakdowns
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

    try:
        with get_conn() as conn2, conn2.cursor() as cur2:
            cur2.execute(cluster_sql, [baseline_id])
            base_clusters = _parse_cluster_rows(cur2.fetchall())
            cur2.execute(cluster_sql, [candidate_id])
            cand_clusters = _parse_cluster_rows(cur2.fetchall())
            cur2.execute(month_sql, [baseline_id])
            base_months = _parse_month_rows(cur2.fetchall())
            cur2.execute(month_sql, [candidate_id])
            cand_months = _parse_month_rows(cur2.fetchall())
    except psycopg.Error:
        logger.exception("Failed to fetch breakdowns for comparison")
        raise HTTPException(status_code=500, detail="Failed to fetch comparison breakdowns")

    per_cluster = _build_per_cluster_comparison(base_clusters, cand_clusters)
    per_month = _build_per_month_comparison(base_months, cand_months)
    param_diffs, param_common = _build_param_diff(b, c)
    feature_diffs = _build_feature_diff(b, c)
    config_diffs, config_common = _build_config_diff(b, c)

    result: dict[str, Any] = {
        "model": model,
        "baseline": b,
        "candidate": c,
        "delta_accuracy": delta_acc,
        "delta_wape": delta_wape,
        "delta_bias": delta_bias,
        "verdict": verdict,
        "existing_comparison_id": existing[0] if existing else None,
        "per_lag": per_lag,
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
    if exec_lag is not None:
        result["exec_lag_filter"] = exec_lag

    return result


def _apply_lag_metrics(run_dict: dict[str, Any], run_id: int, exec_lag: int) -> None:
    """Override portfolio-level accuracy/wape/bias with lag-specific values in-place."""
    sql = """
        SELECT accuracy_pct, wape, bias, n_predictions
        FROM lgbm_tuning_lag
        WHERE run_id = %s AND exec_lag = %s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [run_id, exec_lag])
            row = cur.fetchone()
    except psycopg.Error:
        logger.warning("Failed to fetch lag %d metrics for run %d", exec_lag, run_id)
        return

    if row is not None:
        run_dict["accuracy_pct"] = float(row[0]) if row[0] is not None else None
        run_dict["wape"] = float(row[1]) if row[1] is not None else None
        run_dict["bias"] = float(row[2]) if row[2] is not None else None
        if row[3] is not None:
            run_dict["n_predictions"] = int(row[3])


def _build_per_lag_comparison(baseline_id: int, candidate_id: int) -> list[dict[str, Any]]:
    """Build per-lag accuracy comparison array for both runs."""
    sql = """
        SELECT exec_lag, accuracy_pct, wape, bias
        FROM lgbm_tuning_lag
        WHERE run_id = %s
        ORDER BY exec_lag
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, [baseline_id])
            b_rows = cur.fetchall()
            cur.execute(sql, [candidate_id])
            c_rows = cur.fetchall()
    except psycopg.Error:
        logger.warning("Failed to fetch per-lag data for comparison")
        return []

    b_map = {r[0]: r for r in b_rows}
    c_map = {r[0]: r for r in c_rows}

    per_lag = []
    for lag in range(5):
        br = b_map.get(lag)
        cr = c_map.get(lag)
        b_acc = float(br[1]) if br and br[1] is not None else None
        c_acc = float(cr[1]) if cr and cr[1] is not None else None
        b_wape = float(br[2]) if br and br[2] is not None else None
        c_wape = float(cr[2]) if cr and cr[2] is not None else None
        b_bias = float(br[3]) if br and br[3] is not None else None
        c_bias = float(cr[3]) if cr and cr[3] is not None else None

        per_lag.append({
            "exec_lag": lag,
            "baseline_acc": b_acc,
            "candidate_acc": c_acc,
            "delta_acc": _safe_delta(b_acc, c_acc),
            "baseline_wape": b_wape,
            "candidate_wape": c_wape,
            "delta_wape": _safe_delta(b_wape, c_wape),
            "baseline_bias": b_bias,
            "candidate_bias": c_bias,
            "delta_bias": _safe_delta(b_bias, c_bias, precision=4),
        })

    return per_lag


def _parse_cluster_rows(rows: list[tuple]) -> list[dict[str, Any]]:
    """Parse cluster result rows into dicts."""
    return [
        {
            "cluster_type": r[0],
            "cluster_value": r[1],
            "n_predictions": int(r[2]) if r[2] is not None else 0,
            "n_dfus": int(r[3]) if r[3] is not None else 0,
            "accuracy_pct": float(r[4]) if r[4] is not None else None,
            "wape": float(r[5]) if r[5] is not None else None,
            "bias": float(r[6]) if r[6] is not None else None,
        }
        for r in rows
    ]


def _parse_month_rows(rows: list[tuple]) -> list[dict[str, Any]]:
    """Parse month result rows into dicts."""
    return [
        {
            "month_start": str(r[0]),
            "n_predictions": int(r[1]) if r[1] is not None else 0,
            "n_dfus": int(r[2]) if r[2] is not None else 0,
            "accuracy_pct": float(r[3]) if r[3] is not None else None,
            "wape": float(r[4]) if r[4] is not None else None,
            "bias": float(r[5]) if r[5] is not None else None,
        }
        for r in rows
    ]


def _safe_delta(a: float | None, b: float | None, precision: int = 2) -> float | None:
    """Return ``round(b - a, precision)`` if both values are non-None, else None."""
    if a is not None and b is not None:
        return round(b - a, precision)
    return None


def _build_per_cluster_comparison(
    base_clusters: list[dict[str, Any]],
    cand_clusters: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build per-cluster accuracy comparison between baseline and candidate."""
    per_cluster: dict[str, list[dict[str, Any]]] = {}
    for ct in ("ml_cluster", "business_cluster"):
        b_map = {r["cluster_value"]: r for r in base_clusters if r["cluster_type"] == ct}
        c_map = {r["cluster_value"]: r for r in cand_clusters if r["cluster_type"] == ct}
        items = []
        for val in sorted(set(b_map.keys()) | set(c_map.keys())):
            br = b_map.get(val, {})
            cr = c_map.get(val, {})
            items.append({
                "cluster": val,
                "baseline_accuracy": br.get("accuracy_pct"),
                "candidate_accuracy": cr.get("accuracy_pct"),
                "delta_accuracy": _safe_delta(br.get("accuracy_pct"), cr.get("accuracy_pct")),
                "baseline_wape": br.get("wape"),
                "candidate_wape": cr.get("wape"),
                "baseline_n_dfus": br.get("n_dfus"),
                "candidate_n_dfus": cr.get("n_dfus"),
            })
        per_cluster[ct] = items
    return per_cluster


def _build_per_month_comparison(
    base_months: list[dict[str, Any]],
    cand_months: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-month accuracy comparison between baseline and candidate."""
    b_map = {r["month_start"]: r for r in base_months}
    c_map = {r["month_start"]: r for r in cand_months}
    per_month = []
    for m in sorted(set(b_map.keys()) | set(c_map.keys())):
        br = b_map.get(m, {})
        cr = c_map.get(m, {})
        per_month.append({
            "month": m,
            "baseline_accuracy": br.get("accuracy_pct"),
            "candidate_accuracy": cr.get("accuracy_pct"),
            "delta_accuracy": _safe_delta(br.get("accuracy_pct"), cr.get("accuracy_pct")),
            "baseline_wape": br.get("wape"),
            "candidate_wape": cr.get("wape"),
        })
    return per_month


def _build_param_diff(
    b: dict[str, Any], c: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare hyperparameters between two run dicts. Returns (diffs, common)."""
    diffs: list[dict[str, Any]] = []
    common: list[dict[str, Any]] = []
    b_params = b.get("params") or {}
    c_params = c.get("params") or {}
    for key in sorted(set(b_params.keys()) | set(c_params.keys())):
        bv = b_params.get(key)
        cv = c_params.get(key)
        if bv != cv:
            diffs.append({"param": key, "baseline": bv, "candidate": cv})
        else:
            common.append({"param": key, "value": bv})
    return diffs, common


def _build_feature_diff(b: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    """Compare feature sets between two run dicts."""
    b_features = set(b.get("features") or [])
    c_features = set(c.get("features") or [])
    return {
        "baseline_count": b.get("feature_count") or len(b_features),
        "candidate_count": c.get("feature_count") or len(c_features),
        "added": sorted(c_features - b_features),
        "removed": sorted(b_features - c_features),
        "common_count": len(b_features & c_features),
    }


def _build_config_diff(
    b: dict[str, Any], c: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare config settings between two run dicts. Returns (diffs, common).

    Checks the ``metadata`` dict first, then falls back to top-level keys
    (e.g. ``cluster_source`` / ``cluster_experiment_id`` stored as direct columns).
    """
    diffs: list[dict[str, Any]] = []
    common: list[dict[str, Any]] = []
    b_meta = b.get("metadata") or {}
    c_meta = c.get("metadata") or {}
    for key in _CONFIG_KEYS:
        bv = b_meta.get(key) if b_meta.get(key) is not None else b.get(key)
        cv = c_meta.get(key) if c_meta.get(key) is not None else c.get(key)
        if bv is None and cv is None:
            continue
        if bv != cv:
            diffs.append({"setting": key, "baseline": bv, "candidate": cv})
        else:
            common.append({"setting": key, "value": bv})
    return diffs, common
