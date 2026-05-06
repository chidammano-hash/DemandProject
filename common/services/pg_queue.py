"""Postgres-backed job queue using ``FOR UPDATE SKIP LOCKED`` (Item 22 pilot).

This module is a *minimal* job queue scaffolded alongside the existing
APScheduler-powered ``JobManager`` (see ``common/services/job_registry.py``).
It is **not** a replacement for APScheduler — it is the cutover surface for
long-running, restart-survivable, multi-instance-safe jobs.

Pilot scope:
  * A single recurring APScheduler job — ``refresh_intramonth`` — has been
    migrated onto this queue. Remaining APScheduler jobs continue to run via
    the APScheduler thread pool. The pattern proven here is the recipe for
    migrating future long-running jobs.

Why pg-queue (and not Celery / RQ / Dramatiq):
  * Postgres-only stack — no broker / no Redis / no extra ops surface.
  * ``FOR UPDATE SKIP LOCKED`` (Postgres 9.5+) is the canonical primitive
    for safe concurrent claim-and-run by multiple workers.
  * Persists across API restarts (the pain APScheduler papers over with
    ``recover_stale_jobs``).

Public API
----------
* :func:`enqueue_job` — INSERT a new row, return its id.
* :func:`claim_next_job` — atomically claim the next due pending job.
* :func:`mark_running` / :func:`mark_completed` / :func:`mark_failed` —
  state-transition helpers used by a worker.
* :func:`requeue_failed_with_backoff` — re-enqueue a failed job with
  exponential backoff if it is still under ``max_attempts``.
* :func:`get_queue_depth` — diagnostic counts grouped by status.

Cutover mechanism for ``refresh_intramonth``
-------------------------------------------
The recurring job is no longer driven by APScheduler's ``schedule_recurring``
path. Instead, a tiny enqueueing entry-point (``make pg-queue-enqueue-recurring``
or a lightweight cron-driven APScheduler job whose only side-effect is calling
:func:`enqueue_job`) drops a row into ``job_queue`` once a day. The actual
work runs whenever a long-lived ``scripts/ops/pg_queue_worker.py`` instance
claims the row. This decouples *scheduling* from *execution* — the API process
is no longer responsible for hosting a 7-20h refresh.
"""
from __future__ import annotations

import json
import logging
import os
import socket
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg import sql

from common.core.db import get_db_params

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

STATUS_PENDING = "pending"
STATUS_CLAIMED = "claimed"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

_TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_FAILED)

# Backoff schedule: 60s, 120s, 240s, ... capped.
_BACKOFF_BASE_SECONDS = 60
_BACKOFF_CAP_SECONDS = 60 * 60  # 1 hour


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


def _connect() -> psycopg.Connection:
    """Open a fresh psycopg connection with autocommit OFF.

    The caller is responsible for committing or rolling back. Worker hot
    paths use explicit transactions because ``FOR UPDATE SKIP LOCKED``
    requires the row lock to live for the duration of the claim UPDATE.
    """
    return psycopg.connect(**get_db_params())


def _default_worker_id() -> str:
    """Best-effort unique identifier for a worker instance.

    Format: ``<hostname>:<pid>`` — good enough for tracing claims back to a
    specific process when troubleshooting. Override via ``PG_QUEUE_WORKER_ID``.
    """
    override = os.environ.get("PG_QUEUE_WORKER_ID")
    if override:
        return override
    try:
        host = socket.gethostname() or "unknown"
    except OSError:
        host = "unknown"
    return f"{host}:{os.getpid()}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enqueue_job(
    job_type: str,
    params: dict[str, Any] | None = None,
    *,
    run_at: datetime | None = None,
    priority: int = 100,
    max_attempts: int = 3,
) -> int:
    """Insert a new pending job and return its id.

    Args:
        job_type: Free-form string identifying the worker handler. Workers
            self-select which ``job_type`` values they handle.
        params: JSON-serialisable dict passed to the handler.
        run_at: When the job becomes eligible to be claimed. Defaults to NOW.
        priority: Lower = higher priority. Default 100.
        max_attempts: Max number of attempts before the job is considered
            permanently failed (dead-letter). Default 3.

    Returns:
        The new row's ``id``.

    Raises:
        ValueError: If ``job_type`` is empty or ``max_attempts < 1``.
        psycopg.Error: On DB failure (caller's responsibility).
    """
    if not job_type:
        raise ValueError("job_type must be a non-empty string")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    payload = json.dumps(params or {})
    effective_run_at = run_at if run_at is not None else datetime.now(UTC)

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO job_queue
                    (job_type, params, status, priority, run_at, max_attempts)
                VALUES (%s, %s::jsonb, %s, %s, %s, %s)
                RETURNING id
                """,
                (job_type, payload, STATUS_PENDING, priority,
                 effective_run_at, max_attempts),
            )
            row = cur.fetchone()
        conn.commit()

    if row is None:  # pragma: no cover — INSERT...RETURNING always returns a row
        raise RuntimeError("INSERT INTO job_queue did not return an id")
    job_id = int(row[0])
    logger.info("Enqueued job %d (type=%s, priority=%d, run_at=%s)",
                job_id, job_type, priority, effective_run_at.isoformat())
    return job_id


def claim_next_job(
    worker_id: str | None = None,
    job_types: list[str] | None = None,
) -> dict[str, Any] | None:
    """Atomically claim the next due pending job.

    Uses ``SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1`` so multiple workers
    can race for the queue without blocking each other. The returned row is
    flipped to status='claimed' inside the same transaction.

    Args:
        worker_id: Identifier stamped onto ``claimed_by``. Defaults to
            ``<hostname>:<pid>`` (or ``PG_QUEUE_WORKER_ID`` env override).
        job_types: Optional whitelist. ``None`` (default) claims any type.

    Returns:
        The claimed job as a dict (id, job_type, params, attempts,
        max_attempts), or ``None`` if no job is currently due.

    Raises:
        psycopg.Error: On DB failure (caller's responsibility).
    """
    effective_worker = worker_id or _default_worker_id()
    now = datetime.now(UTC)

    # Build the SELECT with optional job_type filter — use psycopg.sql to
    # avoid f-string interpolation (Rule 6: no f-string SQL).
    type_filter = sql.SQL("")
    type_params: tuple[Any, ...] = ()
    if job_types:
        type_filter = sql.SQL(" AND job_type = ANY(%s)")
        type_params = (list(job_types),)

    select_query = sql.SQL(
        """
        SELECT id, job_type, params, attempts, max_attempts
        FROM job_queue
        WHERE status = %s
          AND run_at <= %s
          {type_filter}
        ORDER BY priority ASC, run_at ASC, id ASC
        FOR UPDATE SKIP LOCKED
        LIMIT 1
        """
    ).format(type_filter=type_filter)

    with _connect() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(select_query, (STATUS_PENDING, now, *type_params))
                row = cur.fetchone()
                if row is None:
                    conn.rollback()
                    return None
                job_id, job_type, params_raw, attempts, max_attempts = row

                cur.execute(
                    """
                    UPDATE job_queue
                    SET status = %s,
                        claimed_by = %s,
                        claimed_at = %s
                    WHERE id = %s
                    """,
                    (STATUS_CLAIMED, effective_worker, now, job_id),
                )
            conn.commit()
        except psycopg.Error:
            conn.rollback()
            raise

    params_dict = (
        params_raw if isinstance(params_raw, dict)
        else json.loads(params_raw or "{}")
    )
    logger.info("Worker %s claimed job %d (type=%s, attempt=%d/%d)",
                effective_worker, job_id, job_type, attempts + 1, max_attempts)
    return {
        "id": int(job_id),
        "job_type": job_type,
        "params": params_dict,
        "attempts": int(attempts),
        "max_attempts": int(max_attempts),
        "worker_id": effective_worker,
    }


def mark_running(job_id: int) -> None:
    """Transition a claimed job to status='running' and bump attempts.

    Called by the worker just before invoking the handler. Splitting
    ``claimed`` and ``running`` lets diagnostics distinguish a worker that
    grabbed a job and crashed before doing anything from one that died
    mid-execution.
    """
    now = datetime.now(UTC)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE job_queue
                SET status = %s,
                    started_at = %s,
                    attempts = attempts + 1
                WHERE id = %s AND status = %s
                """,
                (STATUS_RUNNING, now, job_id, STATUS_CLAIMED),
            )
        conn.commit()
    logger.debug("Job %d marked running", job_id)


def mark_completed(job_id: int, result: dict[str, Any] | None = None) -> None:
    """Transition a running job to status='completed' with an optional result."""
    now = datetime.now(UTC)
    payload = json.dumps(result) if result is not None else None
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE job_queue
                SET status = %s,
                    completed_at = %s,
                    result = %s::jsonb,
                    last_error = NULL
                WHERE id = %s
                """,
                (STATUS_COMPLETED, now, payload, job_id),
            )
        conn.commit()
    logger.info("Job %d completed", job_id)


def mark_failed(job_id: int, error: str) -> None:
    """Transition a running job to status='failed' with the error string.

    Does not auto-retry; call :func:`requeue_failed_with_backoff` after this
    if the job is still under ``max_attempts``.
    """
    now = datetime.now(UTC)
    truncated = (error or "")[:8000]  # avoid pathological log payloads
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE job_queue
                SET status = %s,
                    completed_at = %s,
                    last_error = %s
                WHERE id = %s
                """,
                (STATUS_FAILED, now, truncated, job_id),
            )
        conn.commit()
    logger.warning("Job %d failed: %s", job_id, truncated[:200])


def requeue_failed_with_backoff(job_id: int) -> bool:
    """Re-enqueue a failed job if it is still under ``max_attempts``.

    Resets status to 'pending' and pushes ``run_at`` into the future using
    exponential backoff (``2^attempts * 60s``, capped at 1 hour).

    Returns:
        ``True`` if the job was re-enqueued, ``False`` if it has hit its
        max-attempts ceiling (and should be treated as a dead-letter).
    """
    with _connect() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT attempts, max_attempts
                    FROM job_queue
                    WHERE id = %s AND status = %s
                    FOR UPDATE
                    """,
                    (job_id, STATUS_FAILED),
                )
                row = cur.fetchone()
                if row is None:
                    conn.rollback()
                    logger.debug("requeue: job %d not in failed status", job_id)
                    return False
                attempts, max_attempts = int(row[0]), int(row[1])
                if attempts >= max_attempts:
                    conn.rollback()
                    logger.info("Job %d exhausted retries (%d/%d) — dead-letter",
                                job_id, attempts, max_attempts)
                    return False

                # Exponential backoff: 60s, 120s, 240s, ... capped.
                delay = min(_BACKOFF_BASE_SECONDS * (2 ** attempts),
                            _BACKOFF_CAP_SECONDS)
                next_run = datetime.now(UTC) + timedelta(seconds=delay)
                cur.execute(
                    """
                    UPDATE job_queue
                    SET status = %s,
                        run_at = %s,
                        claimed_by = NULL,
                        claimed_at = NULL,
                        started_at = NULL
                    WHERE id = %s
                    """,
                    (STATUS_PENDING, next_run, job_id),
                )
            conn.commit()
        except psycopg.Error:
            conn.rollback()
            raise
    logger.info("Re-enqueued job %d with %ds backoff", job_id, delay)
    return True


def get_queue_depth(job_type: str | None = None) -> dict[str, int]:
    """Return counts grouped by status (diagnostic).

    Args:
        job_type: Optional filter to scope the count to one job type.

    Returns:
        Dict like ``{"pending": 3, "claimed": 0, "running": 1,
        "completed": 42, "failed": 2}``. Missing keys are zero.
    """
    if job_type:
        query = """
            SELECT status, COUNT(*) FROM job_queue
            WHERE job_type = %s
            GROUP BY status
        """
        params: tuple[Any, ...] = (job_type,)
    else:
        query = "SELECT status, COUNT(*) FROM job_queue GROUP BY status"
        params = ()

    counts: dict[str, int] = {
        STATUS_PENDING: 0, STATUS_CLAIMED: 0, STATUS_RUNNING: 0,
        STATUS_COMPLETED: 0, STATUS_FAILED: 0,
    }
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            for status, count in cur.fetchall():
                counts[status] = int(count)
        conn.rollback()  # read-only; release any snapshot
    return counts
