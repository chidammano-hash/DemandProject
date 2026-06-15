"""Shared ETL load helpers (US3).

Single home for index/constraint management used by the ETL loaders. Before
this module the logic was copy-pasted across:

* ``scripts/etl/load_dataset_postgres.py`` — generic introspection-based
  drop/recreate (the canonical, table-agnostic approach).
* ``scripts/etl/load_backtest_forecasts.py`` — hardcoded forecast/archive
  index + constraint specs.
* ``scripts/etl/load_ext_ml_forecasts.py`` — a byte-identical copy of the
  forecast/archive specs.

This module hosts both flavours so a schema or perf fix is made once.

Identifiers in the generic helpers are quoted with :func:`qident`; the
forecast specs below are hardcoded module constants (never user input).
"""

from __future__ import annotations

import logging
from datetime import date

from common.core.sql_helpers import qident
from common.engines.medallion import complete_batch, create_batch, fail_batch

logger = logging.getLogger(__name__)

# Default unmatched-DFU warning threshold (%) when etl_config is unavailable.
_DEFAULT_UNMATCHED_WARN_PCT = 10.0


# ---------------------------------------------------------------------------
# Staging table naming — one convention across all loaders
# ---------------------------------------------------------------------------

def staging_table_name(domain: str) -> str:
    """Canonical staging table name for a domain (replaces the per-loader
    ``_stg_*`` / ``_stg_<domain>_bulk`` / ``_slice_stg`` variants)."""
    return f"stg_{domain}"


# ---------------------------------------------------------------------------
# Monthly partition management — for declaratively-partitioned fact tables
# (fact_inventory_snapshot, fact_customer_demand_monthly, ...). Partition
# *field* metadata lives in common/core/domain_partition.py; the cursor DDL
# operations live here so they're not re-implemented per loader.
# ---------------------------------------------------------------------------

def is_pg_partitioned(cur, table: str) -> bool:
    """True when ``table`` is declaratively partitioned (pg_class.relkind = 'p')."""
    cur.execute(
        "SELECT relkind = 'p' FROM pg_class WHERE relname = %s "
        "AND relnamespace = 'public'::regnamespace",
        (table,),
    )
    row = cur.fetchone()
    return bool(row and row[0])


def monthly_partition_name(parent: str, month_start: date) -> str:
    """Deterministic monthly partition name: ``<parent>_<YYYY>_<MM>``."""
    return f"{parent}_{month_start:%Y_%m}"


def month_bounds(month_start: date) -> tuple[str, str]:
    """Half-open ISO date range ``[month_start, next_month_start)``."""
    year, month = month_start.year, month_start.month
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return month_start.isoformat(), end.isoformat()


def create_monthly_partition(cur, parent: str, month_start: date) -> str:
    """Unconditionally create the monthly partition of ``parent``. Returns its name.

    Callers that already know the partition is absent use this directly to avoid
    a redundant existence check.
    """
    part_name = monthly_partition_name(parent, month_start)
    start_str, end_str = month_bounds(month_start)
    # DDL can't bind %s; start/end are validated YYYY-MM-DD literals.
    cur.execute(
        f"CREATE TABLE {qident(part_name)} PARTITION OF {qident(parent)} "
        f"FOR VALUES FROM ('{start_str}') TO ('{end_str}')"
    )
    return part_name


def ensure_monthly_partition(cur, parent: str, month_start: date) -> str:
    """Create the monthly partition of ``parent`` if absent. Returns its name."""
    part_name = monthly_partition_name(parent, month_start)
    cur.execute(
        "SELECT 1 FROM pg_class WHERE relname = %s "
        "AND relnamespace = 'public'::regnamespace",
        (part_name,),
    )
    if not cur.fetchone():
        create_monthly_partition(cur, parent, month_start)
    return part_name


def drop_monthly_partition(cur, parent: str, month_start: date) -> None:
    """Drop the monthly partition of ``parent`` if it exists."""
    cur.execute(
        f"DROP TABLE IF EXISTS {qident(monthly_partition_name(parent, month_start))}"
    )


def delete_partition_range(cur, table: str, date_col: str, start, end) -> int:
    """Delete rows in the half-open range ``[start, end)``. Returns rowcount.

    ``start``/``end`` may be ``date`` objects or ISO ``YYYY-MM-DD`` strings.
    """
    cur.execute(
        f"DELETE FROM {qident(table)} "
        f"WHERE {qident(date_col)} >= %s AND {qident(date_col)} < %s",
        (str(start), str(end)),
    )
    return cur.rowcount


# ---------------------------------------------------------------------------
# Generic introspection-based index / constraint management
# (table-agnostic; used by the standard dataset loader)
# ---------------------------------------------------------------------------

def get_secondary_indexes(cur, table: str) -> list[tuple[str, str]]:
    """Return (index_name, index_def) for non-PK, non-constraint-backing indexes."""
    cur.execute("""
        SELECT i.indexname, i.indexdef
        FROM pg_indexes i
        WHERE i.tablename = %s
          AND i.schemaname = 'public'
          AND i.indexname NOT LIKE '%%_pkey'
          AND NOT EXISTS (
              SELECT 1 FROM pg_constraint c
              WHERE c.conindid = (
                  SELECT oid FROM pg_class
                  WHERE relname = i.indexname
                    AND relnamespace = 'public'::regnamespace
              )
          )
        ORDER BY i.indexname
    """, (table,))
    return cur.fetchall()


def get_unique_constraints(cur, table: str) -> list[tuple[str, str, list[str]]]:
    """Return (constraint_name, type, [columns]) for UNIQUE constraints."""
    cur.execute("""
        SELECT con.conname, con.contype::text,
               array_agg(att.attname ORDER BY u.pos)
        FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS u(attnum, pos) ON true
        JOIN pg_attribute att ON att.attrelid = con.conrelid AND att.attnum = u.attnum
        WHERE rel.relname = %s AND con.contype = 'u'
        GROUP BY con.conname, con.contype
    """, (table,))
    return [(r[0], r[1], r[2]) for r in cur.fetchall()]


def drop_indexes(cur, indexes: list[tuple[str, str]]) -> None:
    for idx_name, _ in indexes:
        cur.execute(f"DROP INDEX IF EXISTS {qident(idx_name)}")


def drop_unique_constraints(cur, table: str,
                            constraints: list[tuple[str, str, list[str]]]) -> None:
    for con_name, _, _ in constraints:
        cur.execute(
            f"ALTER TABLE {qident(table)} DROP CONSTRAINT IF EXISTS {qident(con_name)}"
        )


def recreate_indexes(cur, indexes: list[tuple[str, str]]) -> None:
    for _, idx_def in indexes:
        cur.execute(idx_def + ";")


def recreate_unique_constraints(cur, table: str,
                                constraints: list[tuple[str, str, list[str]]]) -> None:
    for con_name, _, cols in constraints:
        col_list = ", ".join(qident(c) for c in cols)
        cur.execute(
            f"ALTER TABLE {qident(table)} ADD CONSTRAINT {qident(con_name)} "
            f"UNIQUE ({col_list})"
        )


# ---------------------------------------------------------------------------
# Forecast / archive table index + constraint specs
# (shared by load_backtest_forecasts.py and load_ext_ml_forecasts.py)
# ---------------------------------------------------------------------------

FORECAST_TABLE = "fact_external_forecast_monthly"
FORECAST_ARCHIVE_TABLE = "backtest_lag_archive"

FORECAST_SECONDARY_INDEXES = [
    "idx_fact_external_forecast_monthly_item",
    "idx_fact_external_forecast_monthly_loc",
    "idx_fact_external_forecast_monthly_fcstdate",
    "idx_fact_external_forecast_monthly_startdate",
    "idx_fact_external_forecast_monthly_lag",
    "idx_fact_external_forecast_monthly_model_id",
]
FORECAST_INDEX_DDL = [
    "CREATE INDEX {name} ON fact_external_forecast_monthly (item_id)",
    "CREATE INDEX {name} ON fact_external_forecast_monthly (loc)",
    "CREATE INDEX {name} ON fact_external_forecast_monthly (fcstdate)",
    "CREATE INDEX {name} ON fact_external_forecast_monthly (startdate)",
    "CREATE INDEX {name} ON fact_external_forecast_monthly (lag)",
    "CREATE INDEX {name} ON fact_external_forecast_monthly (model_id)",
]
FORECAST_CHECK_CONSTRAINTS = [
    "chk_fact_external_forecast_monthly_lag_0_4",
    "chk_fact_external_forecast_monthly_fcst_month_start",
    "chk_fact_external_forecast_monthly_start_month_start",
]
FORECAST_UNIQUE_CONSTRAINT = "uq_forecast_ck_model"

FORECAST_ARCHIVE_SECONDARY_INDEXES = [
    "idx_backtest_lag_archive_model_id",
    "idx_backtest_lag_archive_item_id",
    "idx_backtest_lag_archive_startdate",
    "idx_backtest_lag_archive_lag",
]
FORECAST_ARCHIVE_INDEX_DDL = [
    "CREATE INDEX {name} ON backtest_lag_archive (model_id)",
    "CREATE INDEX {name} ON backtest_lag_archive (item_id)",
    "CREATE INDEX {name} ON backtest_lag_archive (startdate)",
    "CREATE INDEX {name} ON backtest_lag_archive (lag)",
]
FORECAST_ARCHIVE_CHECK_CONSTRAINTS = [
    "chk_backtest_lag_archive_lag_0_4",
    "chk_backtest_lag_archive_fcst_month_start",
    "chk_backtest_lag_archive_start_month_start",
]
FORECAST_ARCHIVE_UNIQUE_CONSTRAINT = "uq_backtest_lag_archive_ck"


def drop_forecast_indexes_and_constraints(cur) -> None:
    """Drop main forecast table secondary indexes + constraints for fast bulk insert."""
    for idx in FORECAST_SECONDARY_INDEXES:
        cur.execute(f"DROP INDEX IF EXISTS {idx}")
    cur.execute(
        f"ALTER TABLE {FORECAST_TABLE} DROP CONSTRAINT IF EXISTS {FORECAST_UNIQUE_CONSTRAINT}"
    )
    for ck in FORECAST_CHECK_CONSTRAINTS:
        cur.execute(f"ALTER TABLE {FORECAST_TABLE} DROP CONSTRAINT IF EXISTS {ck}")


def recreate_forecast_indexes_and_constraints(cur) -> None:
    """Recreate main forecast table indexes + constraints after bulk insert."""
    logger.info("  Recreating UNIQUE constraint...")
    cur.execute(
        f"ALTER TABLE {FORECAST_TABLE} ADD CONSTRAINT {FORECAST_UNIQUE_CONSTRAINT} "
        f"UNIQUE (forecast_ck, model_id)"
    )
    logger.info("  Recreating secondary indexes...")
    for name, ddl in zip(FORECAST_SECONDARY_INDEXES, FORECAST_INDEX_DDL, strict=True):
        cur.execute(ddl.format(name=name))
    logger.info("  Recreating CHECK constraints...")
    cur.execute(f"""ALTER TABLE {FORECAST_TABLE}
        ADD CONSTRAINT chk_fact_external_forecast_monthly_lag_0_4
            CHECK (lag BETWEEN 0 AND 4),
        ADD CONSTRAINT chk_fact_external_forecast_monthly_fcst_month_start
            CHECK (fcstdate = date_trunc('month', fcstdate)::date),
        ADD CONSTRAINT chk_fact_external_forecast_monthly_start_month_start
            CHECK (startdate = date_trunc('month', startdate)::date)
    """)


def drop_forecast_archive_indexes_and_constraints(cur) -> None:
    """Drop archive table indexes + constraints for fast bulk insert."""
    for idx in FORECAST_ARCHIVE_SECONDARY_INDEXES:
        cur.execute(f"DROP INDEX IF EXISTS {idx}")
    cur.execute(
        f"ALTER TABLE {FORECAST_ARCHIVE_TABLE} DROP CONSTRAINT IF EXISTS "
        f"{FORECAST_ARCHIVE_UNIQUE_CONSTRAINT}"
    )
    for ck in FORECAST_ARCHIVE_CHECK_CONSTRAINTS:
        cur.execute(f"ALTER TABLE {FORECAST_ARCHIVE_TABLE} DROP CONSTRAINT IF EXISTS {ck}")


def recreate_forecast_archive_indexes_and_constraints(cur) -> None:
    """Recreate archive table indexes + constraints after bulk insert."""
    logger.info("  Recreating archive UNIQUE constraint...")
    cur.execute(
        f"ALTER TABLE {FORECAST_ARCHIVE_TABLE} ADD CONSTRAINT "
        f"{FORECAST_ARCHIVE_UNIQUE_CONSTRAINT} UNIQUE (forecast_ck, model_id, lag)"
    )
    logger.info("  Recreating archive secondary indexes...")
    for name, ddl in zip(FORECAST_ARCHIVE_SECONDARY_INDEXES, FORECAST_ARCHIVE_INDEX_DDL, strict=True):
        cur.execute(ddl.format(name=name))
    logger.info("  Recreating archive CHECK constraints...")
    cur.execute(f"""ALTER TABLE {FORECAST_ARCHIVE_TABLE}
        ADD CONSTRAINT chk_backtest_lag_archive_lag_0_4
            CHECK (lag BETWEEN 0 AND 4),
        ADD CONSTRAINT chk_backtest_lag_archive_fcst_month_start
            CHECK (fcstdate = date_trunc('month', fcstdate)::date),
        ADD CONSTRAINT chk_backtest_lag_archive_start_month_start
            CHECK (startdate = date_trunc('month', startdate)::date)
    """)


# ---------------------------------------------------------------------------
# DFU match + FK orphan filters (shared by the dataset loader)
# ---------------------------------------------------------------------------

# Domains that require a matching DFU in dim_sku to be loaded.
DFU_MATCH_DOMAINS = {"sales", "forecast", "inventory"}

# Map: domain → [(staging_column, dimension_table, dimension_column), ...]
FK_CHECKS: dict[str, list[tuple[str, str, str]]] = {
    "sales":     [("loc", "dim_location", "location_id"), ("item_id", "dim_item", "item_id")],
    "forecast":  [("loc", "dim_location", "location_id"), ("item_id", "dim_item", "item_id")],
    "inventory": [("loc", "dim_location", "location_id"), ("item_id", "dim_item", "item_id")],
}


def filter_unmatched_dfus(cur, stg_table: str, domain: str) -> int:
    """Delete staging rows that have no matching DFU in dim_sku. Returns deleted count.

    Sales/forecast match on item_id + customer_group + loc (full sku_ck).
    Inventory matches on item_id + loc only (no customer_group in inventory data).
    """
    cur.execute("""
        SELECT EXISTS(
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'dim_sku' AND table_schema = 'public'
        )
    """)
    if not cur.fetchone()[0]:
        logger.warning("dim_sku not found — skipping DFU match filter for %s", domain)
        return 0

    if domain == "inventory":
        cur.execute(f"""
            DELETE FROM {qident(stg_table)} s
            WHERE NOT EXISTS (
                SELECT 1 FROM dim_sku d
                WHERE d.item_id = trim(s."item_id")
                  AND d.loc = trim(s."loc")
            )
        """)
    else:
        cur.execute(f"""
            DELETE FROM {qident(stg_table)} s
            WHERE NOT EXISTS (
                SELECT 1 FROM dim_sku d
                WHERE d.sku_ck = trim(s."item_id") || '_' || trim(s."customer_group") || '_' || trim(s."loc")
            )
        """)

    deleted = cur.rowcount
    if deleted:
        logger.info("  Deleted %s staging rows with no matching DFU in dim_sku",
                    f"{deleted:,}")
    return deleted


def filter_fk_orphans(cur, stg_table: str, domain: str) -> int:
    """Delete staging rows referencing missing dimension values. Returns deleted count."""
    checks = FK_CHECKS.get(domain)
    if not checks:
        return 0

    total_deleted = 0
    for stg_col, dim_table, dim_col in checks:
        cur.execute("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.tables
                WHERE table_name = %s AND table_schema = 'public'
            )
        """, (dim_table,))
        if not cur.fetchone()[0]:
            continue

        cur.execute("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.columns
                WHERE table_name = %s AND column_name = %s
            )
        """, (stg_table, stg_col))
        if not cur.fetchone()[0]:
            continue

        cur.execute(f"""
            DELETE FROM {qident(stg_table)} s
            WHERE trim(s.{qident(stg_col)}) IS NOT NULL
              AND trim(s.{qident(stg_col)}) != ''
              AND NOT EXISTS (
                SELECT 1 FROM {qident(dim_table)} d
                WHERE d.{qident(dim_col)} = trim(s.{qident(stg_col)})
            )
        """)
        deleted = cur.rowcount
        if deleted:
            logger.info("  Removed %s staging rows: %s not in %s.%s",
                        f"{deleted:,}", stg_col, dim_table, dim_col)
            total_deleted += deleted

    return total_deleted


def unmatched_warn_pct() -> float:
    """Unmatched-DFU warning threshold (%) from etl_config.yaml, with default."""
    try:
        from common.core.utils import load_config
        cfg = load_config("etl/etl_config.yaml") or {}
    except (FileNotFoundError, OSError, ValueError):
        return _DEFAULT_UNMATCHED_WARN_PCT
    return float(
        (cfg.get("filters") or {}).get("unmatched_dfu_warn_pct", _DEFAULT_UNMATCHED_WARN_PCT)
    )


# ---------------------------------------------------------------------------
# Audit batch lineage — single entry point for every loader
# ---------------------------------------------------------------------------

def record_load_batch(cur, domain: str, *, source_file: str | None = None,
                      source_hash: str | None = None, rows_in: int = 0,
                      rows_out: int = 0, status: str = "completed",
                      error: str | None = None) -> int:
    """Record a complete load batch in audit_load_batch in one call.

    Wraps medallion.create_batch + complete_batch/fail_batch so loaders that
    don't track a long-running batch (e.g. customer_demand) still register
    lineage and a source_hash, making them visible to change detection.
    Returns the batch_id.
    """
    batch_id = create_batch(cur, domain, source_file, source_hash)
    if status == "failed":
        fail_batch(cur, batch_id, error or "load failed")
    else:
        complete_batch(cur, batch_id, rows_in, rows_out)
    return batch_id
