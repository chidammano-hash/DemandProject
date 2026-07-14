"""Generate a governed customer bottom-up blended champion candidate."""

from __future__ import annotations

import argparse
import logging
from uuid import UUID

import psycopg

from common.core.db import get_db_params
from common.services.customer_forecast_blend import generate_customer_bottom_up_blend

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, type=UUID)
    parser.add_argument("--customer-run-id", type=UUID)
    args = parser.parse_args()
    with psycopg.connect(**get_db_params()) as conn:
        result = generate_customer_bottom_up_blend(
            conn,
            run_id=args.run_id,
            customer_run_id=args.customer_run_id,
        )
    logger.info(
        "Customer bottom-up blend %s completed: %s rows across %s warehouse-items",
        result.run_id,
        result.row_count,
        result.dfu_count,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
