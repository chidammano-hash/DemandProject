"""Refresh materialized views via the central dependency map.

The single operator entry point for MV refreshes — backs the Make targets
``refresh-mvs-tiered`` (--all) and ``refresh-accuracy-mvs`` (--tables ...).
The MV list and ordering come from ``common/core/mv_refresh.py``; this script
never hardcodes view names.

Usage (module form so the project root resolves without PYTHONPATH):
    uv run python -m scripts.db.refresh_mvs --all
    uv run python -m scripts.db.refresh_mvs \
        --tables fact_external_forecast_monthly,backtest_lag_archive
    uv run python -m scripts.db.refresh_mvs --mvs agg_sales_monthly,agg_forecast_monthly
    uv run python -m scripts.db.refresh_mvs --all --parallel 3 --skip-heavy
"""
from __future__ import annotations

import argparse
import logging
import sys

from common.core.mv_refresh import (
    HEAVY_MVS,
    MV_SOURCES,
    all_mvs,
    mvs_for_tables,
    refresh_materialized_views,
    refresh_materialized_views_parallel,
)

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh materialized views")
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--all", action="store_true", help="Refresh every known MV")
    scope.add_argument(
        "--tables",
        help="Comma-separated source tables; refreshes their dependent MVs",
    )
    scope.add_argument("--mvs", help="Comma-separated explicit MV names")
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="Skip HEAVY_MVS (e.g. mv_intramonth_stockout) when deriving from --all/--tables",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        metavar="N",
        help="Refresh with N workers per dependency tier (default: sequential)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.all:
        mvs = [m for m in all_mvs() if not (args.skip_heavy and m in HEAVY_MVS)]
    elif args.tables:
        tables = [t.strip() for t in args.tables.split(",") if t.strip()]
        mvs = mvs_for_tables(tables, include_heavy=not args.skip_heavy)
    else:
        requested = [m.strip() for m in args.mvs.split(",") if m.strip()]
        unknown = [m for m in requested if m not in MV_SOURCES]
        if unknown:
            logger.error("Unknown MV(s): %s. Known: %s", unknown, all_mvs())
            return 2
        mvs = [m for m in all_mvs() if m in set(requested)]

    if not mvs:
        logger.info("Nothing to refresh.")
        return 0

    logger.info("Refreshing %d MV(s): %s", len(mvs), ", ".join(mvs))
    if args.parallel > 1:
        result = refresh_materialized_views_parallel(mvs, max_workers=args.parallel)
    else:
        result = refresh_materialized_views(mvs)

    if result["failed"]:
        logger.error("Failed MVs: %s", ", ".join(result["failed"]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
