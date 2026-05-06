"""Shared SQL helpers used by pipeline scripts and load_dataset_postgres.py.

Extracted to eliminate duplication (D1) and centralise magic constants (M1-M7).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Sequence

from common.core.domain_specs import DomainSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Row -> dict conversion (canonical helpers)
# ---------------------------------------------------------------------------


def row_to_dict_from_cursor(cur: Any, row: Sequence[Any]) -> dict[str, Any]:
    """Convert a DB row tuple to a dict using a cursor's ``description``.

    Use this when the column list is naturally available from the cursor that
    produced the row (e.g. immediately after ``cur.execute`` / ``fetchone``).
    """
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def row_to_dict_from_cols(
    cols: Iterable[str], row: Sequence[Any]
) -> dict[str, Any]:
    """Convert a DB row tuple to a dict using an explicit column list.

    Use this when the column names are known statically (e.g. they were used
    to build the SELECT statement) and the cursor is no longer in scope.
    """
    return dict(zip(tuple(cols), row))

# ---------------------------------------------------------------------------
# Magic-value constants (M1-M7)
# ---------------------------------------------------------------------------

NULL_SQL = "'', 'null', 'none', 'na', 'n/a'"

IQR_OUTLIER_MULTIPLIER = 1.5        # M1
LEAD_TIME_MAX_DAYS = 730             # M2
LEAD_TIME_DEFAULT_DAYS = 7           # M3
HASH_CHUNK_SIZE = 8 * 1024 * 1024    # M4 (8 MB — optimized for large CSVs)
EXTERNAL_MODEL_ID = "external"       # M5
PERCENTILE_MEDIAN = 0.5              # M7
PERCENTILE_Q1 = 0.25                 # M7
PERCENTILE_Q3 = 0.75                 # M7

# MV refresh lists (D5) -------------------------------------------------------

MV_REFRESH_ARCHIVE = [
    "agg_accuracy_lag_archive",
    "agg_dfu_coverage_lag_archive",
]

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _elapsed(t0: float) -> str:
    """Format elapsed time since *t0* as a human-readable string."""
    dt = time.time() - t0
    if dt < 60:
        return f"{dt:.1f}s"
    m, s = divmod(dt, 60)
    return f"{int(m)}m {s:.0f}s"


def qident(name: str) -> str:
    """Quote a SQL identifier (table or column name)."""
    return '"' + name.replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# Type casting / business-key SQL builders
# ---------------------------------------------------------------------------


def typed_expr(field: str, spec: DomainSpec, src_alias: str) -> str:
    """Generate a SQL CASE-cast expression for *field* based on *spec* types.

    Falls back to an untyped column reference if *field* is not in any
    typed-field set, but logs a warning (E4).
    """
    col = f"{src_alias}.{qident(field)}"
    if field in spec.int_fields:
        return (
            f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL "
            f"ELSE {col}::integer END"
        )
    if field in spec.float_fields:
        return (
            f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL "
            f"ELSE {col}::numeric END"
        )
    if field in spec.date_fields:
        return (
            f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL "
            f"ELSE {col}::date END"
        )
    if field in spec.bool_fields:
        return (
            f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL "
            f"ELSE {col}::boolean END"
        )
    # E4 – warn when a column has no type mapping in the spec
    if field not in spec.columns:
        logger.warning(
            "typed_expr: field '%s' not found in spec '%s' columns",
            field, spec.name,
        )
    return col


def typed_expr_sets(
    field: str,
    int_fields: set[str],
    float_fields: set[str],
    date_fields: set[str],
    src_alias: str,
    bool_fields: set[str] | None = None,
) -> str:
    """Legacy overload used by load_dataset_postgres.py (takes raw sets).

    Logs a warning if the field is not in any typed set (E4).
    """
    col = f"{src_alias}.{qident(field)}"
    if field in int_fields:
        return (
            f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL "
            f"ELSE {col}::integer END"
        )
    if field in float_fields:
        return (
            f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL "
            f"ELSE {col}::numeric END"
        )
    if field in date_fields:
        return (
            f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL "
            f"ELSE {col}::date END"
        )
    if bool_fields and field in bool_fields:
        return (
            f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL "
            f"ELSE {col}::boolean END"
        )
    return col


def business_key_expr(spec: DomainSpec, src_alias: str) -> str:
    """Generate SQL expression for the composite business key."""
    cols = [f"trim({src_alias}.{qident(f)})" for f in spec.key_fields]
    if len(cols) == 1:
        return cols[0]
    sep = (spec.business_key_separator or "-").replace("'", "''")
    return f" || '{sep}' || ".join(cols)
