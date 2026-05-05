"""Sequential chain runner for the unified ETL load CLI (scripts/etl/load.py).

Chains run children one-by-one in a worker thread; on first failure the rest
are marked 'failed' (cancelled: chain halted at step N) and the chain becomes
'halted'. Each step is a normal ``integration_job`` row tagged with
``chain_id`` + ``chain_step``; chain-level state lives in ``integration_chain``.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, TypedDict

import psycopg

from common.services.integration_runner import (
    _PROJECT_ROOT,
    _resolve_uv,
    _row_to_dict,
)

logger = logging.getLogger(__name__)
_VALID_MODES = frozenset({"onetime", "delta", "file"})
_SUBPROCESS_TIMEOUT_S = 3600


class JobSpec(TypedDict, total=False):
    """Specification for a single step in a chain."""
    domain: str
    mode: str
    slice: str | None
    file: str | None


def _parse_final_json(stdout: str) -> dict[str, Any]:
    """Extract metrics from load.py's last JSON stdout line."""
    def _i(v: Any) -> int | None:
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    empty: dict[str, Any] = {
        "rows_loaded": 0, "rows_inserted": None,
        "rows_updated": None, "rows_deleted": None, "error": None,
    }
    if not stdout:
        return empty
    last_line = next(
        (s for s in (ln.strip() for ln in reversed(stdout.splitlines())) if s),
        "",
    )
    if not last_line:
        return empty
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError:
        return empty
    if not isinstance(payload, dict):
        return empty
    err = payload.get("error")
    return {
        "rows_loaded": _i(payload.get("rows_loaded")) or 0,
        "rows_inserted": _i(payload.get("rows_inserted")),
        "rows_updated":  _i(payload.get("rows_updated")),
        "rows_deleted":  _i(payload.get("rows_deleted")),
        "error": str(err) if err not in (None, "") else None,
    }


class IntegrationChainRunner:
    """Runs a list of integration jobs sequentially; halts on first failure."""

    _executor: ClassVar[ThreadPoolExecutor | None] = None

    def __init__(self, pool: Any) -> None:
        self.pool = pool
        if IntegrationChainRunner._executor is None:
            IntegrationChainRunner._executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="integration-chain",
            )

    def submit_chain(
        self, jobs: list[JobSpec], triggered_by: str = "api",
    ) -> dict[str, Any]:
        """Persist chain + one ``integration_job`` per step, then dispatch."""
        if not jobs:
            raise ValueError("submit_chain requires at least one job spec")
        for idx, spec in enumerate(jobs):
            mode = spec.get("mode")
            if mode not in _VALID_MODES:
                raise ValueError(
                    f"invalid mode {mode!r} at step {idx + 1}; "
                    f"expected one of {sorted(_VALID_MODES)}"
                )
            if not spec.get("domain"):
                raise ValueError(f"missing domain at step {idx + 1}")

        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO integration_chain (status, total_steps, triggered_by) "
                "VALUES ('queued', %s, %s) RETURNING id",
                (len(jobs), triggered_by),
            )
            row = cur.fetchone()
            if row is None:  # pragma: no cover - defensive
                raise psycopg.Error("INSERT into integration_chain returned no row")
            chain_id = str(row[0])
            children: list[dict[str, Any]] = []
            for idx, spec in enumerate(jobs):
                step = idx + 1
                cur.execute(
                    "INSERT INTO integration_job "
                    "(domain, mode, slice, file_path, status, "
                    " triggered_by, chain_id, chain_step) "
                    "VALUES (%s, %s, %s, %s, 'queued', %s, %s, %s) RETURNING id",
                    (spec["domain"], spec["mode"], spec.get("slice"),
                     spec.get("file"), triggered_by, chain_id, step),
                )
                child_row = cur.fetchone()
                if child_row is None:  # pragma: no cover - defensive
                    raise psycopg.Error("INSERT into integration_job returned no row")
                children.append({
                    "job_id": str(child_row[0]), "step": step,
                    "domain": spec["domain"], "mode": spec["mode"],
                })
        executor = IntegrationChainRunner._executor
        if executor is None:  # pragma: no cover - constructor guarantees init
            raise RuntimeError("IntegrationChainRunner executor not initialized")
        executor.submit(self._run_chain, chain_id)
        logger.info(
            "integration chain %s queued: steps=%d by=%s",
            chain_id, len(jobs), triggered_by,
        )
        return {"chain_id": chain_id, "status": "queued", "jobs": children}

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
        """Mark in-flight chains (and their child jobs) as failed on startup."""
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

    def _run_chain(self, chain_id: str) -> None:
        chain_start = time.monotonic()
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE integration_chain "
                    "SET status='running', started_at=NOW() WHERE id=%s",
                    (chain_id,),
                )
                cur.execute(
                    "SELECT id, chain_step, domain, mode, slice, file_path "
                    "FROM integration_job WHERE chain_id = %s ORDER BY chain_step",
                    (chain_id,),
                )
                steps = cur.fetchall()
        except psycopg.Error:
            logger.exception("failed to start chain %s", chain_id)
            self._finalize_chain(chain_id, "failed", chain_start, None)
            return
        try:
            for job_row in steps:
                step = int(job_row[1])
                status = self._run_step(
                    str(job_row[0]), job_row[2], job_row[3], job_row[4], job_row[5],
                )
                self._run_sql(
                    "UPDATE integration_chain "
                    "SET completed_steps = completed_steps + 1 WHERE id=%s",
                    (chain_id,),
                    f"failed to bump completed_steps on chain {chain_id}",
                )
                if status == "failed":
                    self._run_sql(
                        "UPDATE integration_job SET status='failed', "
                        "  error_message=%s, completed_at=NOW(), duration_ms=0 "
                        "WHERE chain_id=%s AND status='queued'",
                        (f"cancelled: chain halted at step {step}", chain_id),
                        f"failed to cancel remaining steps on chain {chain_id}",
                    )
                    self._finalize_chain(chain_id, "halted", chain_start, step)
                    return
            self._finalize_chain(chain_id, "success", chain_start, None)
        except (psycopg.Error, subprocess.SubprocessError, OSError):
            logger.exception("chain %s crashed unexpectedly", chain_id)
            self._finalize_chain(chain_id, "failed", chain_start, None)

    def _run_step(
        self, job_id: str, domain: str, mode: str,
        slice_: str | None, file_: str | None,
    ) -> str:
        """Execute one chain step. Returns 'success', 'skipped', or 'failed'."""
        start_monotonic = time.monotonic()
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE integration_job SET status='running' WHERE id=%s",
                    (job_id,),
                )
        except psycopg.Error:
            logger.exception("failed to mark chain step %s running", job_id)
            return "failed"

        cmd = [
            _resolve_uv(), "run", "python", "-m", "scripts.etl.load",
            "--domain", domain, "--mode", mode,
        ]
        if slice_:
            cmd.extend(["--slice", slice_])
        if file_:
            cmd.extend(["--file", file_])
        status, rows_loaded = "failed", 0
        rows_inserted = rows_updated = rows_deleted = None
        error_message: str | None = None
        try:
            proc = subprocess.run(
                cmd, cwd=str(_PROJECT_ROOT), capture_output=True, text=True,
                timeout=_SUBPROCESS_TIMEOUT_S, check=False,
            )
            parsed = _parse_final_json(proc.stdout)
            rows_loaded = parsed["rows_loaded"]
            rows_inserted, rows_updated, rows_deleted = (
                parsed["rows_inserted"], parsed["rows_updated"],
                parsed["rows_deleted"],
            )
            if proc.returncode == 0:
                status = "success"
            elif proc.returncode == 2:
                status = "skipped"
            else:
                status = "failed"
                error_message = parsed["error"] or (
                    proc.stderr.strip()[-2000:] if proc.stderr else None
                ) or f"exit code {proc.returncode}"
        except subprocess.TimeoutExpired:
            logger.exception("chain step %s timed out", job_id)
            error_message = f"subprocess timed out after {_SUBPROCESS_TIMEOUT_S}s"
        except (subprocess.SubprocessError, OSError):
            logger.exception("chain step %s subprocess failed", job_id)
            error_message = "subprocess error (see logs)"
        duration_ms = int((time.monotonic() - start_monotonic) * 1000)
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE integration_job SET status=%s, rows_loaded=%s, "
                    "  rows_inserted=%s, rows_updated=%s, rows_deleted=%s, "
                    "  error_message=%s, completed_at=NOW(), duration_ms=%s "
                    "WHERE id=%s",
                    (status, rows_loaded, rows_inserted, rows_updated,
                     rows_deleted, error_message, duration_ms, job_id),
                )
        except psycopg.Error:
            logger.exception("failed to finalize chain step %s", job_id)
            return "failed"
        return status

    def _run_sql(self, sql: str, params: tuple[Any, ...], err_msg: str) -> None:
        """Run a one-shot UPDATE; log+swallow psycopg errors so the chain
        worker can keep its remaining cleanup steps best-effort."""
        try:
            with self.pool.connection() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
        except psycopg.Error:
            logger.exception(err_msg)

    def _finalize_chain(
        self, chain_id: str, status: str, chain_start: float,
        failed_step: int | None,
    ) -> None:
        self._run_sql(
            "UPDATE integration_chain SET status=%s, failed_step=%s, "
            "completed_at=NOW(), duration_ms=%s WHERE id=%s",
            (status, failed_step,
             int((time.monotonic() - chain_start) * 1000), chain_id),
            f"failed to finalize chain {chain_id}",
        )
