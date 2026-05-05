"""Async job runner for the unified ETL load CLI (scripts/etl/load.py).

Persists every submission to the ``integration_job`` table and runs the
underlying ``uv run python -m scripts.etl.load ...`` invocation in a shared
ThreadPoolExecutor so the FastAPI request thread returns immediately.

Public entry point::

    from common.services.integration_runner import IntegrationRunner

    runner = IntegrationRunner(pool)
    job_id = runner.submit(domain="sales", mode="delta", triggered_by="api")
    runner.get(job_id)
    runner.list(domain="sales")
    runner.health()

The subprocess is expected to emit a final JSON line on stdout. Exit codes
drive the final status: 0 -> success, 2 -> skipped, anything else -> failed.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID

import psycopg

logger = logging.getLogger(__name__)

_VALID_MODES = frozenset({"onetime", "delta", "file"})
_SUBPROCESS_TIMEOUT_S = 3600


def _resolve_project_root() -> Path:
    """Walk upward from this file searching for a dir with both ``api`` and
    ``common``. Falls back to ``DEMAND_PROJECT_ROOT`` env or ``parents[2]``.
    """
    env_root = os.environ.get("DEMAND_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "api").is_dir() and (parent / "common").is_dir():
            return parent
    return here.parents[2]


_PROJECT_ROOT = _resolve_project_root()


def _resolve_uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "'uv' not found on PATH; install uv or add it to PATH for the API process"
        )
    return uv


def _to_iso(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    return value


def _row_to_dict(cur: psycopg.Cursor, row: tuple[Any, ...]) -> dict[str, Any]:
    cols = [d[0] for d in cur.description]
    return {col: _to_iso(val) for col, val in zip(cols, row)}


class IntegrationRunner:
    """Async job runner for ETL integration loads.

    Persists job state to integration_job table; spawns subprocess to run
    scripts/etl/load.py and updates status + metrics on completion.
    """

    _executor: ClassVar[ThreadPoolExecutor | None] = None

    def __init__(self, pool: Any) -> None:
        self.pool = pool
        if IntegrationRunner._executor is None:
            IntegrationRunner._executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="integration-runner",
            )

    def submit(
        self,
        domain: str,
        mode: str,
        slice: str | None = None,
        file: str | None = None,
        triggered_by: str = "api",
        reindex: bool = False,
    ) -> str:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"invalid mode {mode!r}; expected one of {sorted(_VALID_MODES)}"
            )
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO integration_job
                    (domain, mode, slice, file_path, status, triggered_by)
                VALUES (%s, %s, %s, %s, 'queued', %s)
                RETURNING id
                """,
                (domain, mode, slice, file, triggered_by),
            )
            row = cur.fetchone()
            if row is None:  # pragma: no cover - defensive
                raise psycopg.Error("INSERT into integration_job returned no row")
            job_id = str(row[0])

        executor = IntegrationRunner._executor
        if executor is None:  # pragma: no cover - constructor guarantees init
            raise RuntimeError("IntegrationRunner executor not initialized")
        executor.submit(self._run_job, job_id, domain, mode, slice, file, reindex)
        logger.info(
            "integration job %s queued: domain=%s mode=%s slice=%s file=%s by=%s reindex=%s",
            job_id, domain, mode, slice, file, triggered_by, reindex,
        )
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM integration_job WHERE id = %s", (job_id,))
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
                    "SELECT * FROM integration_job WHERE domain = %s "
                    "ORDER BY started_at DESC LIMIT %s",
                    (domain, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM integration_job "
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
        """Delete integration_job rows matching the given filters.

        Defaults are conservative: ``keep_running=True`` always excludes jobs
        currently in 'queued' or 'running' state so an in-flight worker can
        still find its row. Returns the number of rows deleted.
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
        """Mark any 'queued' or 'running' job as failed.

        Called on API startup: a fresh process has no in-memory ThreadPoolExecutor
        carrying over from the previous run, so any row left in those states
        belonged to a dead worker and would otherwise stay 'running' forever.
        Returns the number of rows reaped.
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

    def _run_job(
        self,
        job_id: str,
        domain: str,
        mode: str,
        slice: str | None,
        file: str | None,
        reindex: bool = False,
    ) -> None:
        start_monotonic = time.monotonic()
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE integration_job SET status = 'running' WHERE id = %s",
                    (job_id,),
                )
        except psycopg.Error:
            logger.exception("failed to mark job %s running", job_id)
            return

        cmd = [
            _resolve_uv(), "run", "python", "-m", "scripts.etl.load",
            "--domain", domain, "--mode", mode,
        ]
        if slice:
            cmd.extend(["--slice", slice])
        if file:
            cmd.extend(["--file", file])
        if reindex:
            cmd.append("--reindex")

        status = "failed"
        rows_loaded = 0
        rows_inserted: int | None = None
        rows_updated: int | None = None
        rows_deleted: int | None = None
        error_message: str | None = None

        try:
            proc = subprocess.run(  # noqa: S603 - argv built from validated inputs
                cmd,
                cwd=str(_PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT_S,
                check=False,
            )
            parsed = self._parse_final_json(proc.stdout)
            rows_loaded = parsed["rows_loaded"]
            rows_inserted = parsed["rows_inserted"]
            rows_updated = parsed["rows_updated"]
            rows_deleted = parsed["rows_deleted"]
            parsed_error = parsed["error"]
            if proc.returncode == 0:
                status = "success"
            elif proc.returncode == 2:
                status = "skipped"
            else:
                status = "failed"
                error_message = parsed_error or (
                    proc.stderr.strip()[-2000:] if proc.stderr else None
                ) or f"exit code {proc.returncode}"
        except subprocess.TimeoutExpired:
            logger.exception("integration job %s timed out", job_id)
            error_message = f"subprocess timed out after {_SUBPROCESS_TIMEOUT_S}s"
        except (subprocess.SubprocessError, OSError):
            logger.exception("integration job %s subprocess failed", job_id)
            error_message = "subprocess error (see logs)"

        duration_ms = int((time.monotonic() - start_monotonic) * 1000)
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE integration_job
                       SET status = %s, rows_loaded = %s,
                           rows_inserted = %s, rows_updated = %s, rows_deleted = %s,
                           error_message = %s, completed_at = NOW(),
                           duration_ms = %s
                     WHERE id = %s
                    """,
                    (status, rows_loaded, rows_inserted, rows_updated, rows_deleted,
                     error_message, duration_ms, job_id),
                )
        except psycopg.Error:
            logger.exception("failed to finalize integration job %s", job_id)

    @staticmethod
    def _parse_final_json(stdout: str) -> dict[str, Any]:
        """Extract metrics from the dispatcher's last JSON stdout line.

        Returns a dict with keys: rows_loaded, rows_inserted, rows_updated,
        rows_deleted, error. Missing keys default to None (or 0 for rows_loaded).
        """
        empty = {"rows_loaded": 0, "rows_inserted": None,
                 "rows_updated": None, "rows_deleted": None, "error": None}
        if not stdout:
            return empty
        last_line = ""
        for line in stdout.splitlines():
            stripped = line.strip()
            if stripped:
                last_line = stripped
        if not last_line:
            return empty
        try:
            payload = json.loads(last_line)
        except json.JSONDecodeError:
            return empty
        if not isinstance(payload, dict):
            return empty

        def _opt_int(v: Any) -> int | None:
            if v is None:
                return None
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        rows_loaded = _opt_int(payload.get("rows_loaded")) or 0
        err = payload.get("error")
        return {
            "rows_loaded": rows_loaded,
            "rows_inserted": _opt_int(payload.get("rows_inserted")),
            "rows_updated":  _opt_int(payload.get("rows_updated")),
            "rows_deleted":  _opt_int(payload.get("rows_deleted")),
            "error": str(err) if err not in (None, "") else None,
        }
