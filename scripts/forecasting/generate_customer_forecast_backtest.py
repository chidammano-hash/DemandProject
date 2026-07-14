"""Generate rolling Croston and bottom-up blend accuracy evidence."""

from __future__ import annotations

import argparse
import logging
from uuid import UUID

import psycopg

from common.core.db import get_db_params
from common.services.customer_forecast_backtest import (
    generate_customer_forecast_backtest,
)

logger = logging.getLogger(__name__)


def _mark_failed(run_id: UUID) -> None:
    try:
        with psycopg.connect(**get_db_params()) as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE customer_forecast_backtest_run
                   SET run_status = 'failed', error_summary = %s,
                       completed_at = NOW()
                   WHERE run_id = %s::uuid
                     AND run_status IN ('queued', 'generating', 'failed')""",
                ("customer forecast backtest generation failed", str(run_id)),
            )
            conn.commit()
    except psycopg.Error:
        logger.exception("Marking customer forecast backtest failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, type=UUID)
    args = parser.parse_args()
    try:
        with psycopg.connect(**get_db_params()) as conn:
            result = generate_customer_forecast_backtest(conn, run_id=args.run_id)
    except (RuntimeError, TypeError, ValueError, psycopg.Error):
        _mark_failed(args.run_id)
        logger.exception("Generating customer forecast backtest failed")
        raise
    logger.info(
        "Customer forecast backtest %s completed: %s rows, blend gate=%s",
        result.run_id,
        result.component_rows,
        result.comparison.gate_passed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
