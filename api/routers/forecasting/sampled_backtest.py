"""Sampled backtest endpoints — stratified DFU sampling for fast iteration.

Provides endpoints to preview cluster strata, simulate sample allocations,
and trigger sampled backtest runs that complete in ~3 min instead of ~30 min.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response as FastAPIResponse
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.core import get_conn, set_cache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["lgbm-tuning"])

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SamplePreviewBody(BaseModel):
    target_n: int = Field(default=5000, ge=100, le=20000)
    method: str = Field(default="proportional", pattern=r"^(proportional|equal|sqrt)$")


class SampledRunBody(BaseModel):
    target_n: int = Field(default=5000, ge=100, le=20000)
    method: str = Field(default="proportional", pattern=r"^(proportional|equal|sqrt)$")
    param_overrides: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_stratum(cid: int, s: dict[str, Any]) -> dict[str, Any]:
    """Format a single stratum dict for the API response."""
    return {
        "cluster_id": cid,
        "cluster_label": s.get("cluster_label", ""),
        "n_dfus": s["n_dfus"],
        "mean_demand": round(s["mean_demand"], 2),
        "cv": round(s["cv"], 4),
        "zero_pct": round(s["zero_pct"], 4),
    }


# ---------------------------------------------------------------------------
# GET /lgbm-tuning/sampled/strata
# ---------------------------------------------------------------------------


@router.get("/lgbm-tuning/sampled/strata")
def get_strata(response: FastAPIResponse) -> dict[str, Any]:
    """Return cluster strata with demand statistics.

    Cache: 120s (strata change only after re-clustering).
    """
    set_cache(response, max_age=120)

    try:
        from common.ml.backtest_sampler import compute_cluster_strata

        with get_conn() as conn:
            strata = compute_cluster_strata(conn)
    except psycopg.Error:
        logger.exception("Failed to compute cluster strata")
        raise HTTPException(status_code=500, detail="Failed to compute cluster strata")

    total_dfus = sum(s["n_dfus"] for s in strata.values())
    formatted = [_format_stratum(cid, s) for cid, s in strata.items()]

    return {
        "n_clusters": len(strata),
        "total_dfus": total_dfus,
        "strata": formatted,
    }


# ---------------------------------------------------------------------------
# POST /lgbm-tuning/sampled/preview
# ---------------------------------------------------------------------------


@router.post("/lgbm-tuning/sampled/preview")
def preview_sample(
    body: SamplePreviewBody,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Preview sample allocation without running a backtest.

    Returns per-cluster allocation and estimated accuracy deviation.
    """
    try:
        from common.ml.backtest_sampler import (
            compute_cluster_strata,
            estimate_accuracy_deviation,
        )

        with get_conn() as conn:
            strata = compute_cluster_strata(conn)
    except psycopg.Error:
        logger.exception("Failed to compute cluster strata for preview")
        raise HTTPException(status_code=500, detail="Failed to compute strata")

    if not strata:
        return {
            "target_n": body.target_n,
            "method": body.method,
            "actual_n": 0,
            "estimated_deviation_pct": 0.0,
            "allocation": [],
        }

    from common.ml.backtest_sampler import _ALLOCATORS, _load_sampling_config

    cfg = _load_sampling_config()
    min_per_cluster = cfg.get("min_per_cluster", 10)

    allocator = _ALLOCATORS.get(body.method)
    if allocator is None:
        raise HTTPException(status_code=400, detail=f"Unknown method: {body.method}")

    allocation = allocator(strata, body.target_n, min_per_cluster)

    total_dfus = sum(s["n_dfus"] for s in strata.values())
    actual_n = sum(
        min(n, strata[cid]["n_dfus"]) for cid, n in allocation.items()
    )
    deviation = estimate_accuracy_deviation(actual_n, total_dfus, len(strata))

    alloc_details = []
    for cid, n_alloc in allocation.items():
        s = strata[cid]
        capped = min(n_alloc, s["n_dfus"])
        alloc_details.append({
            "cluster_id": cid,
            "cluster_label": s.get("cluster_label", ""),
            "n_dfus_total": s["n_dfus"],
            "n_sampled": capped,
            "pct_of_cluster": round(100.0 * capped / s["n_dfus"], 1) if s["n_dfus"] > 0 else 0.0,
        })

    return {
        "target_n": body.target_n,
        "method": body.method,
        "actual_n": actual_n,
        "estimated_deviation_pct": deviation,
        "allocation": alloc_details,
    }


# ---------------------------------------------------------------------------
# POST /lgbm-tuning/sampled/run
# ---------------------------------------------------------------------------


@router.post("/lgbm-tuning/sampled/run")
def trigger_sampled_run(
    body: SampledRunBody,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Trigger a sampled backtest run.

    1. Computes stratified sample of DFUs.
    2. Registers a tuning run with note ``sampled_n=<N>``.
    3. Submits the sampled_backtest job (JobManager).
    4. Returns ``{run_id, sample_size, estimated_deviation}``.
    """
    try:
        from common.ml.backtest_sampler import (
            compute_cluster_strata,
            estimate_accuracy_deviation,
            stratified_sample,
        )

        with get_conn() as conn:
            sampled_skus = stratified_sample(
                conn,
                target_n=body.target_n,
                method=body.method,
            )
            strata = compute_cluster_strata(conn)
    except psycopg.Error:
        logger.exception("Failed to compute stratified sample")
        raise HTTPException(status_code=500, detail="Failed to compute stratified sample")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not sampled_skus:
        raise HTTPException(status_code=400, detail="No DFUs available for sampling")

    total_dfus = sum(s["n_dfus"] for s in strata.values())
    deviation = estimate_accuracy_deviation(len(sampled_skus), total_dfus, len(strata))

    # Register the run in tuning tracker
    run_id: int | None = None
    try:
        with get_conn() as conn, conn.cursor() as cur:
            params_json = json.dumps(body.param_overrides) if body.param_overrides else None
            cur.execute(
                """
                INSERT INTO lgbm_tuning_run
                    (run_label, model_id, status, params, notes)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING run_id
                """,
                (
                    f"sampled_{body.method}_{len(sampled_skus)}",
                    "lgbm_cluster",
                    "running",
                    params_json,
                    f"sampled_n={len(sampled_skus)} method={body.method}",
                ),
            )
            row = cur.fetchone()
            if row:
                run_id = row[0]
            conn.commit()
    except psycopg.Error:
        logger.exception("Failed to register sampled run in tuning tracker")
        raise HTTPException(status_code=500, detail="Failed to register sampled run")

    # Write sampled SKU list to a named temp file (auto-cleaned on close/GC)
    sku_tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="sampled_skus_", delete=False,
    )
    sku_tmp.write(json.dumps(sampled_skus))
    sku_tmp.close()
    sku_file = Path(sku_tmp.name)

    # Submit through the JobManager (PID tracking, cancel, log streaming,
    # restart recovery) — never a bare subprocess.Popen.
    try:
        from common.services.job_registry import JobManager

        job_id = JobManager().submit_job(
            "sampled_backtest",
            {
                "run_id": run_id,
                "sku_file": str(sku_file),
                "param_overrides": body.param_overrides or {},
            },
            label=f"Sampled backtest ({len(sampled_skus)} SKUs, {body.method})",
            triggered_by="api",
        )
    except (ValueError, RuntimeError) as exc:
        logger.exception("Failed to submit sampled backtest job")
        # Mark run as failed if we registered one
        if run_id is not None:
            try:
                with get_conn() as conn, conn.cursor() as cur:
                    cur.execute(
                        "UPDATE lgbm_tuning_run SET status = %s WHERE run_id = %s",
                        ("failed", run_id),
                    )
                    conn.commit()
            except psycopg.Error:
                logger.warning("Failed to update run status to failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to submit sampled backtest job",
        ) from exc

    logger.info(
        "Submitted sampled backtest job %s: run_id=%s, n=%d, method=%s",
        job_id, run_id, len(sampled_skus), body.method,
    )

    return {
        "run_id": run_id,
        "job_id": job_id,
        "sample_size": len(sampled_skus),
        "total_dfus": total_dfus,
        "method": body.method,
        "estimated_deviation_pct": deviation,
    }
