"""Generate one resumable, batch-persisted customer forecast run."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from typing import Any

import psycopg

from common.core.db import get_db_params
from common.core.paths import SCRIPTS_DIR
from common.services.customer_forecast import (
    get_customer_forecast_settings,
    mark_customer_forecast_run_terminal,
)
from common.services.customer_forecast_batches import (
    finalize_customer_forecast_batches,
    initialize_customer_forecast_batches,
    load_customer_forecast_progress,
    run_customer_forecast_worker,
)

logger = logging.getLogger(__name__)
_PROGRESS_PREFIX = "JOB_PROGRESS_JSON:"


def _format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "calculating"
    hours, remainder = divmod(max(seconds, 0), 3600)
    minutes = remainder // 60
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _emit_progress(progress: dict[str, Any]) -> None:
    message = (
        f"{int(progress['completed_series']):,} / {int(progress['total_series']):,} "
        f"customer-SKUs completed · {int(progress['completed_batches']):,} / "
        f"{int(progress['total_batches']):,} batches · ETA "
        f"{_format_duration(progress['eta_seconds'])}"
    )
    logger.info(
        "%s%s",
        _PROGRESS_PREFIX,
        json.dumps(
            {"pct": int(progress["progress_pct"]), "msg": message},
            separators=(",", ":"),
        ),
    )


def _worker_command(run_id: str, route_model_ids: list[str]) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "forecasting" / "generate_customer_forecasts.py"),
        "--run-id",
        run_id,
    ]
    for model_id in route_model_ids:
        command.extend(["--worker-model", model_id])
    return command


def _terminate_workers(workers: list[subprocess.Popen[str]]) -> None:
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
    deadline = time.monotonic() + 10
    for worker in workers:
        if worker.poll() is not None:
            continue
        remaining = max(deadline - time.monotonic(), 0.1)
        try:
            worker.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=5)


def _run_parallel_workers(run_id: str, settings: dict[str, Any]) -> None:
    workers: list[subprocess.Popen[str]] = []
    try:
        with psycopg.connect(**get_db_params()) as conn:
            progress = load_customer_forecast_progress(conn, run_id)
        route_counts = progress["model_route_counts"]
        cpu_routes = [
            str(route)
            for route in settings["route_model_ids"]
            if int(route_counts.get(route, 0)) > 0
        ]
        cpu_series = sum(int(route_counts.get(route, 0)) for route in cpu_routes)
        if cpu_series > 0:
            cpu_worker_count = min(
                int(settings["cpu_workers"]),
                max(1, int(progress["total_batches"])),
            )
            for _worker_index in range(cpu_worker_count):
                workers.append(
                    subprocess.Popen(
                        _worker_command(run_id, cpu_routes),
                        text=True,
                    )
                )

        while workers and any(worker.poll() is None for worker in workers):
            with psycopg.connect(**get_db_params()) as conn:
                _emit_progress(load_customer_forecast_progress(conn, run_id))
            time.sleep(float(settings["progress_interval_seconds"]))
        failures = [worker.returncode for worker in workers if worker.returncode != 0]
        if failures:
            raise RuntimeError("One or more customer forecast batch workers failed")
    finally:
        _terminate_workers(workers)


def _run_parent(run_id: str) -> dict[str, Any]:
    settings = get_customer_forecast_settings()
    with psycopg.connect(**get_db_params()) as conn:
        logger.info(
            "%s%s",
            _PROGRESS_PREFIX,
            json.dumps(
                {
                    "pct": 5,
                    "msg": "Preparing resumable customer-SKU batch manifest",
                },
                separators=(",", ":"),
            ),
        )
        progress = initialize_customer_forecast_batches(conn, run_id)
        _emit_progress(progress)
    _run_parallel_workers(run_id, settings)
    with psycopg.connect(**get_db_params()) as conn:
        result = finalize_customer_forecast_batches(conn, run_id)
    _emit_progress(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-model", action="append", default=[])
    args = parser.parse_args()

    if args.worker_model:
        with psycopg.connect(**get_db_params()) as conn:
            completed_batches = run_customer_forecast_worker(
                conn,
                args.run_id,
                args.worker_model,
            )
        logger.info(
            "Customer forecast worker completed %s batches for %s",
            completed_batches,
            ", ".join(args.worker_model),
        )
        return

    try:
        result = _run_parent(args.run_id)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, psycopg.Error):
        try:
            with psycopg.connect(**get_db_params()) as conn:
                mark_customer_forecast_run_terminal(
                    conn,
                    args.run_id,
                    "failed",
                    "customer forecast generation failed",
                )
        except psycopg.Error:
            logger.exception("Marking customer forecast run failed")
        logger.exception("Generating customer forecasts failed")
        raise
    logger.info(
        "Customer forecast run %s completed: %s rows across %s series",
        args.run_id,
        result["row_count"],
        result["total_series"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main()
