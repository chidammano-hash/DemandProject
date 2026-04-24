"""Platform admin endpoints — operations tooling (not user-facing).

Currently exposes:
- ``POST /admin/llm/reset``      : force LLM client reinitialization after key rotation
- ``POST /admin/tuning/invalidate-stale``
                                  : ask tuning scheduler to re-tune stale per-cluster
                                    profiles (no-op until the ``stale`` column lands
                                    via Stream F).

All endpoints are guarded by the ``require_api_key`` dependency so they are
unavailable in environments that do not set ``API_KEY``.
"""
from __future__ import annotations

import logging
from typing import Any

import psycopg
from fastapi import APIRouter, Depends

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
def admin_invalidate_stale_tuning() -> dict[str, Any]:
    """Invalidate per-cluster tuning profiles whose cluster has been re-promoted.

    Stream F owns the data-side of this feature: a ``stale`` column on
    ``cluster_tuning_profile`` (or equivalent) that flips ``true`` whenever the
    referenced cluster is replaced.  When the column exists we pick up the
    stale profiles and notify the tuning scheduler so they re-tune on the next
    cycle.

    Until Stream F lands, this endpoint is intentionally a no-op that only
    logs intent — it returns ``{"status": "noop", "reason": "..."}`` so the
    UI / scheduler can wire against it safely.
    """
    try:
        from api.core import get_conn
    except ImportError as exc:  # defensive; core should always import
        logger.warning("api.core unavailable: %s", exc)
        return {"status": "noop", "reason": "api.core unavailable", "invalidated": 0}

    try:
        with get_conn() as conn, conn.cursor() as cur:
            # Is the stale column present yet?
            cur.execute(
                """
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = %s AND column_name = %s
                LIMIT 1
                """,
                ("cluster_tuning_profile", "stale"),
            )
            has_stale = cur.fetchone() is not None
            if not has_stale:
                logger.info(
                    "admin/tuning/invalidate-stale: 'stale' column not present; no-op"
                )
                return {
                    "status": "noop",
                    "reason": "stale column not present (Stream F not landed)",
                    "invalidated": 0,
                }

            # Column exists — count stale rows and reset their stale flag.
            cur.execute(
                "SELECT COUNT(*) FROM cluster_tuning_profile WHERE stale = TRUE"
            )
            row = cur.fetchone()
            count = int(row[0]) if row else 0
            if count == 0:
                return {"status": "ok", "invalidated": 0}

            # Clear the stale flag; scheduler picks these up via its own query.
            cur.execute(
                "UPDATE cluster_tuning_profile SET stale = FALSE WHERE stale = TRUE"
            )
            conn.commit()
            logger.info(
                "admin/tuning/invalidate-stale: cleared stale flag on %d profile(s)",
                count,
            )
            return {"status": "ok", "invalidated": count}
    except psycopg.Error as exc:
        # DB-level failure (unmigrated schema, permissions, etc.) — degrade to
        # a no-op rather than 500 so the admin tooling stays safe to poll.
        logger.exception("admin/tuning/invalidate-stale DB error (degrading to no-op)")
        return {"status": "noop", "reason": f"db_error: {type(exc).__name__}", "invalidated": 0}
