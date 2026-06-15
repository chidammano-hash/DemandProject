"""Read + cleanup helpers for integration job history.

Since the US17c cutover, ingestion jobs are submitted through JobManager
(``load_domain`` / ``etl_pipeline``) and land in ``job_history``; the legacy
``integration_job`` table is a read-only archive. This module no longer submits
or executes loads — it only:

- **reads** the unified view (``integration_job_unified``) so the
  ``/integration/jobs`` endpoints surface both archived legacy rows and new
  JobManager ingestion jobs (:meth:`IntegrationRunner.list` / :meth:`get`),
- **purges / reaps** stale archive rows (:meth:`purge`, :meth:`reap_orphans`),
- reports backend **health** (:meth:`health`).

The submission/subprocess path (``submit`` / ``_run_job``) was retired in US17e.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg

from common.core.sql_helpers import row_to_dict_from_cursor

logger = logging.getLogger(__name__)


def _to_iso(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    return value


def _row_to_dict(cur: psycopg.Cursor, row: tuple[Any, ...]) -> dict[str, Any]:
    """Convert an integration-job row to a dict with ISO/UUID coercion.

    Wraps the canonical :func:`row_to_dict_from_cursor` helper and then applies
    integration-domain-specific coercions for ``datetime`` and ``UUID`` values.
    """
    return {
        col: _to_iso(val)
        for col, val in row_to_dict_from_cursor(cur, row).items()
    }


class IntegrationRunner:
    """Read + cleanup surface over the integration job archive + unified view.

    Construction takes the shared connection pool. There is no background
    executor — submission moved to JobManager (US17c) and this class no longer
    spawns work.
    """

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    def get(self, job_id: str) -> dict[str, Any] | None:
        # Reads come from the unified view (US17b): legacy integration_job rows
        # plus JobManager ingestion jobs (etl_pipeline / load_domain), all
        # normalized to the integration Job shape.
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM integration_job_unified WHERE id = %s", (job_id,)
            )
            row = cur.fetchone()
            return _row_to_dict(cur, row) if row is not None else None

    def list(
        self,
        domain: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self.pool.connection() as conn, conn.cursor() as cur:
            if domain is not None:
                cur.execute(
                    "SELECT * FROM integration_job_unified WHERE domain = %s "
                    "ORDER BY started_at DESC LIMIT %s",
                    (domain, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM integration_job_unified "
                    "ORDER BY started_at DESC LIMIT %s",
                    (limit,),
                )
            rows = cur.fetchall()
            return [_row_to_dict(cur, r) for r in rows]

    def purge(
        self,
        *,
        older_than_hours: int | None = None,
        statuses: list[str] | None = None,
        domain: str | None = None,
        keep_running: bool = True,
    ) -> int:
        """Delete archived ``integration_job`` rows matching the given filters.

        Defaults are conservative: ``keep_running=True`` always excludes jobs
        currently in 'queued' or 'running' state. Returns the number of rows
        deleted. (Targets the base archive table, not the view.)
        """
        clauses: list[str] = []
        params: list[Any] = []
        if keep_running:
            clauses.append("status NOT IN ('queued', 'running')")
        if statuses:
            clauses.append("status = ANY(%s)")
            params.append(statuses)
        if domain is not None:
            clauses.append("domain = %s")
            params.append(domain)
        if older_than_hours is not None and older_than_hours > 0:
            clauses.append("started_at < NOW() - (%s * INTERVAL '1 hour')")
            params.append(older_than_hours)
        where = " AND ".join(clauses) if clauses else "TRUE"
        sql = f"DELETE FROM integration_job WHERE {where} RETURNING id"
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                deleted = len(cur.fetchall())
            logger.info("purge: deleted %d integration_job row(s)", deleted)
            return deleted
        except psycopg.Error as exc:
            logger.warning("purge failed: %s", exc)
            return 0

    def reap_orphans(self) -> int:
        """Mark any archived 'queued'/'running' row as failed.

        Pre-cutover loads ran in-process; a row left in those states belonged to
        a dead worker. No new rows are written here anymore, so this only tidies
        legacy archive rows. Returns the number of rows reaped.
        """
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE integration_job
                       SET status = 'failed',
                           error_message = COALESCE(error_message,
                                'abandoned: api restarted while job was in flight'),
                           completed_at = COALESCE(completed_at, NOW()),
                           duration_ms  = COALESCE(duration_ms,
                                EXTRACT(EPOCH FROM (NOW() - started_at))::int * 1000)
                     WHERE status IN ('queued', 'running')
                     RETURNING id
                    """,
                )
                reaped = len(cur.fetchall())
            if reaped:
                logger.warning("reaped %d orphan integration_job row(s)", reaped)
            return reaped
        except psycopg.Error as exc:
            logger.warning("reap_orphans failed: %s", exc)
            return 0

    def health(self) -> dict[str, str]:
        pool_status = "degraded"
        try:
            with self.pool.connection() as conn:
                conn.execute("SELECT 1")
            pool_status = "ok"
        except (psycopg.Error, OSError) as exc:
            logger.exception("integration_runner pool health failed: %s", exc)

        table_status = "missing"
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT to_regclass('integration_job') IS NOT NULL")
                row = cur.fetchone()
                if row is not None and bool(row[0]):
                    table_status = "ok"
        except (psycopg.Error, OSError) as exc:
            logger.exception("integration_runner table health failed: %s", exc)

        return {"pool": pool_status, "table": table_status}
