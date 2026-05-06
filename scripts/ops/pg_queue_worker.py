"""Long-running pg-queue worker (Item 22 pilot).

Polls ``job_queue`` for pending jobs and runs the matching handler. One
worker = one process; run multiple instances for parallelism. The
``FOR UPDATE SKIP LOCKED`` claim path in :mod:`common.services.pg_queue`
guarantees no two workers pick the same row.

Pilot scope: this worker handles exactly one job type today —
``refresh_intramonth`` — by delegating to the existing job-state runner.
Add more handlers to the ``HANDLERS`` table as more APScheduler jobs
graduate to the queue.

Usage:
    uv run python scripts/ops/pg_queue_worker.py
    uv run python scripts/ops/pg_queue_worker.py --types refresh_intramonth
    uv run python scripts/ops/pg_queue_worker.py --poll-interval 5

SIGTERM / SIGINT triggers a graceful shutdown — the worker finishes its
current job (if any) before exiting.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from collections.abc import Callable
from typing import Any

# CLI bootstrap: ensure the project root is importable when invoked as
# `python scripts/ops/pg_queue_worker.py`. Once the path is set, all real
# imports go through `common.core.paths` (Rule 2).
try:
    from common.core.paths import PROJECT_ROOT
except ModuleNotFoundError:  # bootstrap path: project root not yet on sys.path
    from pathlib import Path
    _here = Path(__file__).resolve()
    for _candidate in _here.parents:
        if (_candidate / "common" / "core" / "paths.py").is_file():
            sys.path.insert(0, str(_candidate))
            break
    from common.core.paths import PROJECT_ROOT

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.services import pg_queue
from common.services.job_state import _run_refresh_intramonth

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handler dispatch table
# ---------------------------------------------------------------------------

# Each handler takes (params, progress_cb=None, cancel_event=None, job_id=None)
# and returns a JSON-serialisable dict. Reuse the existing job_state runners
# so the pg-queue handlers are byte-for-byte equivalent to APScheduler runs.
HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "refresh_intramonth": _run_refresh_intramonth,
}


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


class _ShutdownFlag:
    """Flag toggled by SIGTERM/SIGINT so the main loop can exit cleanly."""

    def __init__(self) -> None:
        self.requested = False

    def request(self, signum: int, frame: Any) -> None:
        logger.info("Shutdown signal %d received — finishing current job", signum)
        self.requested = True


def _install_signal_handlers(flag: _ShutdownFlag) -> None:
    signal.signal(signal.SIGTERM, flag.request)
    signal.signal(signal.SIGINT, flag.request)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _execute_one(job: dict[str, Any]) -> None:
    """Run a single claimed job through state transitions.

    Errors from the handler are caught, logged, and converted to
    ``mark_failed`` + ``requeue_failed_with_backoff``. Errors from the
    state-transition calls themselves propagate — they indicate a broken
    DB and the worker should crash so an operator notices.
    """
    job_id = job["id"]
    job_type = job["job_type"]
    handler = HANDLERS.get(job_type)
    if handler is None:
        msg = f"No handler registered for job_type='{job_type}'"
        logger.error("%s — failing job %d", msg, job_id)
        pg_queue.mark_running(job_id)
        pg_queue.mark_failed(job_id, msg)
        # Don't retry: missing handler is a config bug, not a transient fault.
        return

    pg_queue.mark_running(job_id)
    try:
        result = handler(job["params"]) or {}
        if not isinstance(result, dict):
            result = {"output": str(result)}
        pg_queue.mark_completed(job_id, result)
    except (RuntimeError, ValueError, OSError) as exc:
        logger.exception("Job %d (%s) failed: %s", job_id, job_type, exc)
        pg_queue.mark_failed(job_id, f"{type(exc).__name__}: {exc}")
        pg_queue.requeue_failed_with_backoff(job_id)


def run_worker(
    job_types: list[str] | None,
    poll_interval: float,
    shutdown: _ShutdownFlag,
) -> int:
    """Main polling loop. Returns the number of jobs processed."""
    processed = 0
    logger.info(
        "pg-queue worker started (types=%s, poll_interval=%.1fs)",
        job_types or "ALL", poll_interval,
    )
    while not shutdown.requested:
        try:
            job = pg_queue.claim_next_job(job_types=job_types)
        except Exception:  # noqa: BLE001 — top-level worker loop must survive any DB/network fault
            logger.exception("claim_next_job raised — backing off %.1fs", poll_interval)
            time.sleep(poll_interval)
            continue

        if job is None:
            # No work — sleep in small slices so SIGTERM is responsive.
            slept = 0.0
            while slept < poll_interval and not shutdown.requested:
                time.sleep(min(0.5, poll_interval - slept))
                slept += 0.5
            continue

        _execute_one(job)
        processed += 1

    logger.info("pg-queue worker exiting after %d jobs", processed)
    return processed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pg-queue worker")
    parser.add_argument(
        "--types",
        nargs="*",
        default=None,
        help="Whitelist of job_type values to handle (default: all known)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds to wait between polls when the queue is empty",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    # Default to handlers we actually know about.
    types = args.types if args.types is not None else list(HANDLERS.keys())

    flag = _ShutdownFlag()
    _install_signal_handlers(flag)
    run_worker(job_types=types, poll_interval=args.poll_interval, shutdown=flag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
