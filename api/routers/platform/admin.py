"""Platform admin endpoints — operations tooling (not user-facing).

Currently exposes:
- ``POST /admin/llm/reset``      : force LLM client reinitialization after key rotation
- ``POST /admin/tuning/invalidate-stale``
                                  : clear — or with ``?retune=true`` actually re-tune —
                                    per-cluster tuning profiles flagged stale by a
                                    clustering promotion (``cluster_tuning_profile_state``,
                                    sql/148).

All endpoints are guarded by the ``require_api_key`` dependency so they are
unavailable in environments that do not set ``API_KEY``.
"""
from __future__ import annotations

import logging
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, Query

from api.auth import require_api_key
from api.llm import reset_llm_client

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_api_key)],
)


@router.post("/llm/reset")
def admin_reset_llm() -> dict[str, Any]:
    """Close and clear the cached OpenAI/Anthropic clients.

    Intended for key rotation: after updating ``OPENAI_API_KEY`` /
    ``ANTHROPIC_API_KEY`` in the environment, call this endpoint so the next
    chat completion rebuilds the client with the fresh key rather than
    continuing to use the stale singleton.
    """
    result = reset_llm_client()
    return {"status": "ok", **result}


@router.post("/tuning/invalidate-stale")
def admin_invalidate_stale_tuning(
    retune: bool = Query(
        default=False,
        description="Submit a tune_stale_clusters job (re-tunes stale clusters and "
                    "clears their flags on success) instead of just clearing flags",
    ),
) -> dict[str, Any]:
    """Handle per-cluster tuning profiles flagged stale by a cluster promotion.

    ``promote_scenario`` marks rows in ``cluster_tuning_profile_state`` stale
    whenever a clustering scenario is promoted (sql/148). Two modes:

    - default: clear the stale flags (acknowledge without re-tuning — e.g.
      after a manual full ``make tune-clusters`` run already covered them).
    - ``?retune=true``: submit the ``tune_stale_clusters`` background job,
      which runs ``tune_cluster_hyperparams.py --stale-only``; the script
      clears the flags for the clusters it successfully re-tunes.
    """
    try:
        from api.core import get_conn
    except ImportError as exc:  # defensive; core should always import
        logger.warning("api.core unavailable: %s", exc)
        return {"status": "noop", "reason": "api.core unavailable", "invalidated": 0}

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = %s
                LIMIT 1
                """,
                ("cluster_tuning_profile_state",),
            )
            if cur.fetchone() is None:
                logger.info(
                    "admin/tuning/invalidate-stale: state table not present; no-op"
                )
                return {
                    "status": "noop",
                    "reason": "cluster_tuning_profile_state table not present (apply sql/148)",
                    "invalidated": 0,
                }

            cur.execute(
                """SELECT tuning.cluster_name
                   FROM cluster_tuning_profile_state tuning
                   WHERE tuning.stale = TRUE
                     AND EXISTS (
                         SELECT 1 FROM current_sku_cluster_assignment assignment
                         WHERE assignment.ml_cluster = tuning.cluster_name
                     )
                   ORDER BY tuning.cluster_name"""
            )
            stale_clusters = [row[0] for row in cur.fetchall()]
            if not stale_clusters:
                return {"status": "ok", "invalidated": 0, "stale_clusters": []}

            if retune:
                from common.services.job_registry import JobManager

                job_id = JobManager().submit_job(
                    "tune_stale_clusters",
                    {"model": "lgbm"},
                    label=f"Re-tune {len(stale_clusters)} stale cluster profile(s)",
                    triggered_by="api",
                )
                logger.info(
                    "admin/tuning/invalidate-stale: submitted retune job %s for %d cluster(s)",
                    job_id, len(stale_clusters),
                )
                return {
                    "status": "retune_submitted",
                    "job_id": job_id,
                    "stale_clusters": stale_clusters,
                    "invalidated": 0,
                }

            cur.execute(
                """UPDATE cluster_tuning_profile_state
                   SET stale = FALSE, cleared_at = NOW(), modified_ts = NOW()
                   WHERE stale = TRUE
                     AND cluster_name = ANY(%s)""",
                (stale_clusters,),
            )
            cleared = cur.rowcount
            conn.commit()
            logger.info(
                "admin/tuning/invalidate-stale: cleared stale flag on %d profile(s)",
                cleared,
            )
            return {
                "status": "ok",
                "invalidated": cleared,
                "stale_clusters": stale_clusters,
            }
    except psycopg.Error as exc:
        # DB-level failure (unmigrated schema, permissions, etc.) — degrade to
        # a no-op rather than 500 so the admin tooling stays safe to poll.
        logger.exception("admin/tuning/invalidate-stale DB error (degrading to no-op)")
        return {"status": "noop", "reason": f"db_error: {type(exc).__name__}", "invalidated": 0}
