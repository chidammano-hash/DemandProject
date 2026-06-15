"""Read + cleanup helpers for the legacy integration_chain archive.

Since US17d, chains run as JobManager pipelines (see
``common.services.integration_chain_jobs.ChainJobRunner``). The legacy
``integration_chain`` table is a read-only archive: this class only **reads**
old chains (:meth:`get_chain` / :meth:`list_chains`, used by ChainJobRunner's
fallback) and **reaps** stale archive rows (:meth:`reap_orphans`). The
submission/execution path (``submit_chain`` / ``_run_chain`` / ``_run_step``)
was retired in US17e.
"""
from __future__ import annotations

import logging
from typing import Any

import psycopg

from common.services.integration_runner import _row_to_dict

logger = logging.getLogger(__name__)


class IntegrationChainRunner:
    """Read + cleanup surface over the legacy ``integration_chain`` archive."""

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    def get_chain(self, chain_id: str) -> dict[str, Any] | None:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM integration_chain WHERE id = %s", (chain_id,))
            row = cur.fetchone()
            if row is None:
                return None
            chain = _row_to_dict(cur, row)
            cur.execute(
                "SELECT chain_step AS step, id AS job_id, domain, mode, slice, "
                "       status, rows_loaded, rows_inserted, rows_updated, "
                "       rows_deleted, error_message, started_at, completed_at, "
                "       duration_ms "
                "  FROM integration_job WHERE chain_id = %s ORDER BY chain_step",
                (chain_id,),
            )
            chain["jobs"] = [_row_to_dict(cur, r) for r in cur.fetchall()]
            return chain

    def list_chains(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM integration_chain "
                "ORDER BY started_at DESC LIMIT %s",
                (limit,),
            )
            return [_row_to_dict(cur, r) for r in cur.fetchall()]

    def reap_orphans(self) -> int:
        """Mark in-flight legacy chains (and their child jobs) as failed.

        No new chains are written to this table anymore; this only tidies
        pre-cutover archive rows left 'queued'/'running' by a dead worker.
        """
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE integration_chain SET status='failed', "
                    "  completed_at=COALESCE(completed_at, NOW()), "
                    "  duration_ms=COALESCE(duration_ms, "
                    "    EXTRACT(EPOCH FROM (NOW() - started_at))::int * 1000) "
                    "WHERE status IN ('queued','running') RETURNING id"
                )
                reaped_chain_ids = [r[0] for r in cur.fetchall()]
                if reaped_chain_ids:
                    cur.execute(
                        "UPDATE integration_job SET status='failed', "
                        "  error_message=COALESCE(error_message, "
                        "    'abandoned: api restarted while chain was in flight'), "
                        "  completed_at=COALESCE(completed_at, NOW()), "
                        "  duration_ms=COALESCE(duration_ms, "
                        "    EXTRACT(EPOCH FROM (NOW() - started_at))::int * 1000) "
                        "WHERE chain_id = ANY(%s) AND status IN ('queued','running')",
                        (reaped_chain_ids,),
                    )
            reaped = len(reaped_chain_ids)
            if reaped:
                logger.warning("reaped %d orphan integration_chain row(s)", reaped)
            return reaped
        except psycopg.Error as exc:
            logger.warning("reap_orphans (chain) failed: %s", exc)
            return 0
