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

from common.core.sql_helpers import qident

logger = logging.getLogger(__name__)


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
