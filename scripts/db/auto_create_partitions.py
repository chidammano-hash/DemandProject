"""Idempotently create the next N partitions for partitioned fact tables.

Background
----------
Several fact tables in this project are RANGE-partitioned by a date column
with concrete partitions provisioned up front in their migrations:

* ``fact_inventory_snapshot``        (sql/088, partition key ``snapshot_date``)
* ``fact_customer_demand_monthly``   (sql/110, partition key ``startdate``)
* ``fact_external_signal``           (sql/141, partition key ``event_ts``)

When the calendar advances past the last hardcoded partition, new rows fall
into the table's DEFAULT partition. That defeats partition pruning and slowly
turns the default partition into a giant grab-bag.

This script keeps a rolling window of ``--horizon`` partitions ahead of today
for every registered partitioned table. Each registry entry chooses an
``interval`` of ``month`` (default) or ``week``. Weekly partitions follow
ISO-8601 weeks (Mon–Sun) and are named ``<prefix>_YYYYwWW``.

It uses ``CREATE TABLE IF NOT EXISTS ... PARTITION OF`` so it is fully
idempotent — re-running it is always safe.

Run it on a schedule (monthly for monthly tables, weekly for weekly tables),
OR manually before any large backfill that may write rows into a future window.

Usage
-----
    python scripts/db/auto_create_partitions.py                 # use each table's default horizon
    python scripts/db/auto_create_partitions.py --horizon 6     # override horizon for all tables
    python scripts/db/auto_create_partitions.py --interval week # restrict to weekly tables
    python scripts/db/auto_create_partitions.py --dry-run       # print DDL only
    python scripts/db/auto_create_partitions.py --table fact_inventory_snapshot
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import psycopg

from common.core.db import get_db_params
from common.core.planning_date import get_planning_date

logger = logging.getLogger(__name__)


Interval = Literal["month", "week"]


# ---------------------------------------------------------------------------
# Registry of RANGE-partitioned fact tables.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PartitionedTable:
    """Definition of a RANGE-partitioned table."""

    name: str
    """Parent table name (e.g. fact_inventory_snapshot)."""

    partition_prefix: str
    """Prefix used for child partition names. Children are named
    ``<prefix>_YYYY_MM`` (monthly) or ``<prefix>_YYYYwWW`` (weekly).
    Often equal to the parent name."""

    column_type: str
    """SQL type used in the FROM/TO bounds. ``date`` for date columns,
    ``timestamptz`` for timestamp columns."""

    interval: Interval = "month"
    """Partition cadence: ``month`` (default) or ``week`` (ISO Mon–Sun)."""

    default_horizon: int | None = None
    """Default number of partitions to provision ahead. If None, falls back
    to the CLI default (12 for month, 12 for week)."""


# Tables to manage. Add new partitioned tables here.
#
# NOTE on the inventory/customer-demand cutover (sql/<next>+sql/<next+1>):
# After the weekly cutover migrations are applied, the same parent table will
# have BOTH historical monthly partitions AND new weekly partitions. The auto-
# creation logic only adds NEW future partitions (ahead of today), so flipping
# the registered ``interval`` to ``week`` for these two tables is the supported
# way to switch the rolling window over to weekly cadence.
PARTITIONED_TABLES: tuple[PartitionedTable, ...] = (
    PartitionedTable(
        name="fact_inventory_snapshot",
        partition_prefix="fact_inventory_snapshot",
        column_type="date",
        interval="month",
    ),
    PartitionedTable(
        name="fact_customer_demand_monthly",
        partition_prefix="fact_customer_demand_monthly",
        column_type="date",
        interval="month",
    ),
    PartitionedTable(
        name="fact_external_signal",
        partition_prefix="fact_external_signal",
        column_type="timestamptz",
        interval="month",
    ),
)


# ---------------------------------------------------------------------------
# Date math helpers — pure, no DB access.
# ---------------------------------------------------------------------------

def _add_months(d: date, n: int) -> date:
    """Return the first day of the month ``n`` months after ``d``."""
    year = d.year + (d.month - 1 + n) // 12
    month = (d.month - 1 + n) % 12 + 1
    return date(year, month, 1)


def iso_week_start(d: date) -> date:
    """Return the Monday (ISO weekday 1) of the ISO week containing ``d``.

    ISO-8601 weeks run Mon–Sun. ``isoweekday()`` returns 1=Mon..7=Sun, so
    subtracting ``isoweekday()-1`` always yields the Monday of the same week.
    """
    return d - timedelta(days=d.isoweekday() - 1)


def add_weeks(d: date, n: int) -> date:
    """Return ``d + n weeks`` (exact 7-day arithmetic; safe across DST/year boundaries)."""
    return d + timedelta(weeks=n)


def iso_week_partition_suffix(week_start: date) -> str:
    """Return the ``YYYYwWW`` suffix for a Monday-aligned ISO week start.

    Uses ISO-8601 calendar (NOT the ordinary ``%Y/%U`` strftime tokens, which
    use Sunday as week start and disagree with ISO numbering).
    """
    iso_year, iso_week, _ = week_start.isocalendar()
    return f"{iso_year:04d}w{iso_week:02d}"


def _format_bound(d: date, column_type: str) -> str:
    """Render a partition bound literal for the given column type."""
    iso = d.isoformat()
    if column_type == "timestamptz":
        return f"TIMESTAMPTZ '{iso} 00:00:00+00'"
    return f"DATE '{iso}'"


# ---------------------------------------------------------------------------
# DDL builders — pure, no DB access.
# ---------------------------------------------------------------------------

def build_monthly_partition_ddl(
    table: PartitionedTable,
    month_start: date,
) -> tuple[str, str]:
    """Return (partition_name, DDL) for a single monthly partition."""
    next_month = _add_months(month_start, 1)
    partition_name = f"{table.partition_prefix}_{month_start.year:04d}_{month_start.month:02d}"
    from_bound = _format_bound(month_start, table.column_type)
    to_bound = _format_bound(next_month, table.column_type)
    ddl = (
        f"CREATE TABLE IF NOT EXISTS {partition_name} "
        f"PARTITION OF {table.name} "
        f"FOR VALUES FROM ({from_bound}) TO ({to_bound});"
    )
    return partition_name, ddl


def build_weekly_partition_ddl(
    table: PartitionedTable,
    week_start: date,
) -> tuple[str, str]:
    """Return (partition_name, DDL) for a single weekly (ISO Mon–Sun) partition.

    ``week_start`` MUST be a Monday (ISO weekday 1). Callers can use
    :func:`iso_week_start` to align an arbitrary date.
    """
    if week_start.isoweekday() != 1:
        raise ValueError(
            f"week_start must be a Monday (ISO weekday 1); got "
            f"{week_start} (weekday={week_start.isoweekday()})"
        )
    next_week = add_weeks(week_start, 1)
    suffix = iso_week_partition_suffix(week_start)
    partition_name = f"{table.partition_prefix}_{suffix}"
    from_bound = _format_bound(week_start, table.column_type)
    to_bound = _format_bound(next_week, table.column_type)
    ddl = (
        f"CREATE TABLE IF NOT EXISTS {partition_name} "
        f"PARTITION OF {table.name} "
        f"FOR VALUES FROM ({from_bound}) TO ({to_bound});"
    )
    return partition_name, ddl


def build_partition_ddl(
    table: PartitionedTable,
    period_start: date,
) -> tuple[str, str]:
    """Dispatch to the monthly/weekly DDL builder based on table.interval."""
    if table.interval == "week":
        return build_weekly_partition_ddl(table, period_start)
    return build_monthly_partition_ddl(table, period_start)


# ---------------------------------------------------------------------------
# DB-touching helpers
# ---------------------------------------------------------------------------

def _table_exists(conn: psycopg.Connection, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_class WHERE relname = %s AND relkind IN ('r', 'p')",
            (table,),
        )
        return cur.fetchone() is not None


def _resolve_horizon(table: PartitionedTable, cli_horizon: int | None) -> int:
    """CLI override > table default > 12."""
    if cli_horizon is not None:
        return cli_horizon
    if table.default_horizon is not None:
        return table.default_horizon
    return 12


def ensure_partitions(
    conn: psycopg.Connection,
    table: PartitionedTable,
    horizon: int,
    today: date | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Ensure ``horizon`` partitions exist starting at the current period.

    Returns the list of DDL statements emitted (whether or not the partition
    already existed; CREATE TABLE IF NOT EXISTS makes the call cheap).
    """
    if not _table_exists(conn, table.name):
        logger.warning("Skipping %s: parent table not present in database", table.name)
        return []

    today = today or get_planning_date()
    if table.interval == "week":
        start = iso_week_start(today)
        step = add_weeks
        unit = "week"
    else:
        start = date(today.year, today.month, 1)
        step = _add_months
        unit = "month"

    statements: list[str] = []
    for offset in range(horizon):
        period_start = step(start, offset)
        partition_name, ddl = build_partition_ddl(table, period_start)
        statements.append(ddl)
        if dry_run:
            logger.info("[dry-run] %s", ddl)
        else:
            with conn.cursor() as cur:
                cur.execute(ddl)
            logger.info("Ensured %s partition %s", unit, partition_name)
    return statements


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Number of partitions to provision ahead from the current period. "
             "Default: each table's configured default (12 for both month and week).",
    )
    parser.add_argument(
        "--table",
        action="append",
        default=None,
        help="Restrict to a specific parent table. May be passed multiple times. "
             "Default: every table in the registry.",
    )
    parser.add_argument(
        "--interval",
        choices=("month", "week"),
        default=None,
        help="Restrict to tables registered with this interval. "
             "Default: all intervals.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print DDL without executing.",
    )
    return parser.parse_args(argv)


def _select_tables(
    args: argparse.Namespace,
) -> tuple[PartitionedTable, ...] | int:
    """Resolve the registry slice the user asked for. Returns an exit code on error."""
    selected: tuple[PartitionedTable, ...] = PARTITIONED_TABLES
    if args.table:
        chosen = set(args.table)
        selected = tuple(t for t in selected if t.name in chosen)
        unknown = chosen - {t.name for t in selected}
        if unknown:
            logger.error(
                "Unknown table(s): %s. Known: %s",
                ", ".join(sorted(unknown)),
                ", ".join(t.name for t in PARTITIONED_TABLES),
            )
            return 2
    if args.interval:
        selected = tuple(t for t in selected if t.interval == args.interval)
        if not selected:
            logger.error(
                "No tables match --interval=%s. Registered intervals: %s",
                args.interval,
                ", ".join(sorted({t.interval for t in PARTITIONED_TABLES})),
            )
            return 2
    return selected


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = _parse_args(argv)

    if args.horizon is not None and args.horizon < 1:
        logger.error("--horizon must be at least 1")
        return 2

    selected = _select_tables(args)
    if isinstance(selected, int):
        return selected

    db_params = get_db_params()
    with psycopg.connect(**db_params) as conn:
        conn.autocommit = False
        for table in selected:
            try:
                ensure_partitions(
                    conn,
                    table,
                    horizon=_resolve_horizon(table, args.horizon),
                    dry_run=args.dry_run,
                )
            except psycopg.Error:
                logger.exception("Failed to provision partitions for %s", table.name)
                conn.rollback()
                return 1
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
    logger.info(
        "Done: ensured partitions ahead for %d table(s)%s",
        len(selected),
        " (dry-run, no changes committed)" if args.dry_run else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
