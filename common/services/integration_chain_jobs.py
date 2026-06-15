"""US17d — integration chains on the unified JobManager backend.

A chain is submitted as a JobManager *pipeline* of ``load_domain`` steps
(``chain_id == pipeline_id``); per-step state lives in ``job_history`` rows
(``pipeline_step`` column). Reads merge the stored step plan (US17d
``chain_shape``) with the actual job rows, normalized through the same
status/shape adapters as the single-job path (``job_shape``/``chain_shape``).

Legacy ``integration_chain`` chains stay readable: ``get_chain`` / ``list_chains``
fall back to :class:`IntegrationChainRunner` when no pipeline rows exist (no
migration/deletion here — that is US17e).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import psycopg

from common.services.chain_shape import (
    PLAN_KEY,
    chain_jobs,
    chain_summary,
    to_load_domain_steps,
)
from common.services.integration_chain_runner import IntegrationChainRunner
from common.services.job_shape import job_history_to_integration_job

logger = logging.getLogger(__name__)

# How many recent pipeline job rows to scan when building the chain list. A
# chain has a handful of steps, so this comfortably covers the default page.
_LIST_ROW_CAP = 500


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def _row_to_actual(row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Map a job_history pipeline row dict to a (step, ChainJob-shaped) pair."""
    mapped = job_history_to_integration_job(row)
    step = int(row.get("pipeline_step") or 0)
    return step, {
        "step": step,
        "job_id": mapped["id"],
        "domain": mapped["domain"],
        "mode": mapped["mode"],
        "slice": mapped["slice"],
        "status": mapped["status"],
        "rows_loaded": mapped["rows_loaded"],
        "rows_inserted": mapped["rows_inserted"],
        "rows_updated": mapped["rows_updated"],
        "rows_deleted": mapped["rows_deleted"],
        "error_message": mapped["error_message"],
        "started_at": _iso(mapped["started_at"]),
        "completed_at": _iso(mapped["completed_at"]),
        "duration_ms": mapped["duration_ms"],
    }


# Explicit column lists (literal SQL — never interpolate values; psycopg3 %s).
_GET_CHAIN_SQL = (
    "SELECT pipeline_id, pipeline_step, job_id, status, params, result, error, "
    "started_at, completed_at, triggered_by FROM job_history "
    "WHERE pipeline_id = %s AND job_type = 'load_domain' ORDER BY pipeline_step"
)
_LIST_CHAINS_SQL = (
    "SELECT pipeline_id, pipeline_step, job_id, status, params, result, error, "
    "started_at, completed_at, triggered_by, submitted_at FROM job_history "
    "WHERE pipeline_id IS NOT NULL AND job_type = 'load_domain' "
    "ORDER BY submitted_at DESC LIMIT %s"
)


class ChainJobRunner:
    """Submits/reads integration chains via JobManager pipelines.

    Mirrors the IntegrationChainRunner public surface used by the chain router
    (``submit_chain`` / ``get_chain`` / ``list_chains``) so the endpoints swap
    one dependency for another with no response-shape change.
    """

    def __init__(self, pool: Any) -> None:
        self.pool = pool
        self._legacy = IntegrationChainRunner(pool)

    # -- submit -------------------------------------------------------------
    def submit_chain(
        self, jobs: list[dict[str, Any]], triggered_by: str = "api",
    ) -> dict[str, Any]:
        """Build a JobManager pipeline of load_domain steps and dispatch it."""
        if not jobs:
            raise ValueError("submit_chain requires at least one job spec")
        for idx, spec in enumerate(jobs, start=1):
            if spec.get("mode") not in ("onetime", "delta", "file"):
                raise ValueError(
                    f"invalid mode {spec.get('mode')!r} at step {idx}"
                )
            if not spec.get("domain"):
                raise ValueError(f"missing domain at step {idx}")

        steps, plan = to_load_domain_steps(jobs)

        from common.services.job_registry import JobManager
        chain_id = JobManager().submit_pipeline(
            steps,
            label="Integration chain",
            triggered_by=triggered_by,
        )
        # Only step 1 has a real job row yet; later steps are submitted on
        # completion. Return synthetic step ids (unique) so the UI can key the
        # list — the real ids surface on GET /chains/{id} polls.
        children = [
            {"job_id": f"{chain_id}#{p['step']}", "step": p["step"],
             "domain": p["domain"], "mode": p["mode"]}
            for p in plan
        ]
        logger.info(
            "integration chain %s queued via JobManager: steps=%d by=%s",
            chain_id, len(plan), triggered_by,
        )
        return {"chain_id": chain_id, "status": "queued", "jobs": children}

    # -- reads --------------------------------------------------------------
    def _fetch_pipeline_rows(self, chain_id: str) -> list[dict[str, Any]]:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(_GET_CHAIN_SQL, (chain_id,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]

    @staticmethod
    def _plan_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        for row in rows:
            params = row.get("params") or {}
            if isinstance(params, dict) and params.get(PLAN_KEY):
                return params[PLAN_KEY]
        return None

    def get_chain(self, chain_id: str) -> dict[str, Any] | None:
        """Fetch one chain (pipeline) with per-step jobs; legacy fallback."""
        rows = self._fetch_pipeline_rows(chain_id)
        if not rows:
            # Not a JobManager pipeline — may be a legacy integration_chain row.
            return self._legacy.get_chain(chain_id)

        plan = self._plan_from_rows(rows)
        actual = dict(_row_to_actual(r) for r in rows)
        triggered_by = next(
            (r.get("triggered_by") for r in rows if r.get("triggered_by")), "api"
        )
        summary = chain_summary(chain_id, plan, actual, triggered_by=triggered_by)
        summary["jobs"] = chain_jobs(chain_id, plan, actual)
        return summary

    def list_chains(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent chains across JobManager pipelines + legacy chains."""
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.execute(_LIST_CHAINS_SQL, (_LIST_ROW_CAP,))
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r, strict=False)) for r in cur.fetchall()]

        # Group by pipeline, preserving recency (rows already newest-first).
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(row["pipeline_id"], []).append(row)

        summaries: list[dict[str, Any]] = []
        for chain_id, group in grouped.items():
            plan = self._plan_from_rows(group)
            actual = dict(_row_to_actual(r) for r in group)
            triggered_by = next(
                (r.get("triggered_by") for r in group if r.get("triggered_by")), "api"
            )
            summaries.append(chain_summary(chain_id, plan, actual, triggered_by=triggered_by))

        # Merge legacy chains (read-compat during transition).
        try:
            summaries.extend(self._legacy.list_chains(limit=limit) or [])
        except psycopg.Error as exc:
            logger.warning("legacy list_chains failed: %s", exc)

        summaries.sort(key=lambda s: (s.get("started_at") or ""), reverse=True)
        return summaries[:limit]

    def reap_orphans(self) -> int:
        """Delegate legacy-chain orphan reaping (JobManager has its own recovery)."""
        return self._legacy.reap_orphans()
