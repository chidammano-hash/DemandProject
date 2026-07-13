"""Generate one immutable customer-level Chronos forecast run."""

from __future__ import annotations

import argparse
import logging

import psycopg

from common.core.db import get_db_params
from common.services.customer_forecast import (
    generate_customer_forecast,
    mark_customer_forecast_run_terminal,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    with psycopg.connect(**get_db_params()) as conn:
        try:
            result = generate_customer_forecast(conn, args.run_id)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError, psycopg.Error) as exc:
            conn.rollback()
            try:
                mark_customer_forecast_run_terminal(conn, args.run_id, "failed", str(exc))
            except psycopg.Error:
                logger.exception("Marking customer forecast run failed")
            logger.exception("Generating customer forecasts failed")
            raise
    logger.info(
        "Customer forecast run %s completed: %s rows across %s series",
        args.run_id,
        result["row_count"],
        result["eligible_series"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
