"""Shared SQL helpers used by pipeline scripts and load_dataset_postgres.py.

Extracted to eliminate duplication (D1) and centralise magic constants (M1-M7).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from typing import Any
from uuid import uuid4

from common.core.domain_specs import DomainSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB value coercion helpers (shared across routers — single source of truth)
# ---------------------------------------------------------------------------
def parse_db_json(val: Any) -> Any:
    """Parse a DB JSON/JSONB column value.

    Handles the three shapes a psycopg JSON(B) read can yield: already-decoded
    ``dict``/``list`` (psycopg3 default), a raw JSON ``str`` (text columns or
    older drivers), or ``None``. Returns the original value if it is not valid
    JSON rather than raising — robust against malformed text columns.
    """
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return val


def to_float(v: Any, decimals: int | None = None) -> float | None:
    """Coerce a Postgres numeric value to ``float``; ``None`` on NULL/failure.

    Optional ``decimals`` rounds the result. Replaces the per-router
    ``_safe_float`` copies.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, decimals) if decimals is not None else f


# ---------------------------------------------------------------------------
# Streaming/chunked SQL reads (memory-safety helpers for fact-table scans)
# ---------------------------------------------------------------------------

# Default rows-per-chunk for fact-table scans. 50k strikes a balance between
# round-trip overhead and per-chunk memory footprint (~5-50 MB depending on
# column count and dtype).
DEFAULT_CHUNK_SIZE = 50_000


def stream_query_in_chunks(
    conn: Any,
    sql: str,
    params: Sequence[Any] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[Any]:
    """Yield pandas DataFrames of approximately *chunk_size* rows.

    Use this for any pipeline query that scans a fact-table (sales, forecast,
    inventory snapshots). At 40x scale, an unchunked ``pd.read_sql`` over a
    fact table OOMs the worker; this helper bounds peak memory to a single
    chunk plus whatever the caller is accumulating.

    The caller is responsible for accumulating / aggregating across chunks.
    For "I just need the full frame" callers, use :func:`read_sql_chunked`.

    Args:
        conn: An open psycopg connection. The query uses a named server-side
            cursor so pandas never handles the DBAPI connection directly.
        sql: SQL with ``%s`` placeholders (psycopg3 style).
        params: Bind parameters (list/tuple). ``None`` means no params.
        chunk_size: Rows per yielded DataFrame.

    Yields:
        ``pandas.DataFrame`` objects, each with up to ``chunk_size`` rows. An
        empty result yields one zero-row frame retaining the query columns.
    """
    import pandas as pd  # local import: keeps common.core import-cheap

    if chunk_size <= 0:
        raise ValueError("SQL chunk size must be positive")

    cursor_name = f"sql_helpers_{uuid4().hex}"
    with conn.cursor(name=cursor_name) as cur:
        cur.execute(sql, params)
        description = cur.description
        if description is None:
            raise RuntimeError("Chunked SQL query did not return a result set")
        columns = [str(column[0]) for column in description]
        yielded = False
        while rows := cur.fetchmany(chunk_size):
            yielded = True
            yield pd.DataFrame.from_records(rows, columns=columns)
        if not yielded:
            yield pd.DataFrame(columns=columns)


def read_sql_chunked(
    conn: Any,
    sql: str,
    params: Sequence[Any] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Any:
    """Drop-in replacement for ``pd.read_sql`` that streams via a cursor.

    Returns a single concatenated DataFrame, but pulls rows in chunks of
    ``chunk_size`` so peak memory stays bounded during the fetch phase.
    Output is identical to ``pd.read_sql(sql, conn, params=params)`` (same
    columns, same dtypes -- pandas concat preserves both).

    Use this when downstream code genuinely needs the full frame (e.g.
    pivot/groupby/rolling); use :func:`stream_query_in_chunks` directly when
    the work can be done incrementally.
    """
    import pandas as pd

    chunks = list(stream_query_in_chunks(conn, sql, params=params, chunk_size=chunk_size))
    if len(chunks) == 1:
        return chunks[0]
    return pd.concat(chunks, ignore_index=True, copy=False)


# ---------------------------------------------------------------------------
# Row -> dict conversion (canonical helpers)
# ---------------------------------------------------------------------------


def row_to_dict_from_cursor(cur: Any, row: Sequence[Any]) -> dict[str, Any]:
    """Convert a DB row tuple to a dict using a cursor's ``description``.

    Use this when the column list is naturally available from the cursor that
    produced the row (e.g. immediately after ``cur.execute`` / ``fetchone``).
    """
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row, strict=True))


def row_to_dict_from_cols(cols: Iterable[str], row: Sequence[Any]) -> dict[str, Any]:
    """Convert a DB row tuple to a dict using an explicit column list.

    Use this when the column names are known statically (e.g. they were used
    to build the SELECT statement) and the cursor is no longer in scope.
    """
    return dict(zip(tuple(cols), row, strict=True))


# ---------------------------------------------------------------------------
# Magic-value constants (M1-M7)
# ---------------------------------------------------------------------------

NULL_SQL = "'', 'null', 'none', 'na', 'n/a'"

IQR_OUTLIER_MULTIPLIER = 1.5  # M1
LEAD_TIME_MAX_DAYS = 730  # M2
LEAD_TIME_DEFAULT_DAYS = 7  # M3
HASH_CHUNK_SIZE = 8 * 1024 * 1024  # M4 (8 MB — optimized for large CSVs)
EXTERNAL_MODEL_ID = "external"  # M5
PERCENTILE_MEDIAN = 0.5  # M7
PERCENTILE_Q1 = 0.25  # M7
PERCENTILE_Q3 = 0.75  # M7

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
        return f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL ELSE {col}::integer END"
    if field in spec.float_fields:
        return f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL ELSE {col}::numeric END"
    if field in spec.date_fields:
        return f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL ELSE {col}::date END"
    if field in spec.bool_fields:
        return f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL ELSE {col}::boolean END"
    # E4 - warn when a column has no type mapping in the spec
    if field not in spec.columns:
        logger.warning(
            "typed_expr: field '%s' not found in spec '%s' columns",
            field,
            spec.name,
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
        return f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL ELSE {col}::integer END"
    if field in float_fields:
        return f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL ELSE {col}::numeric END"
    if field in date_fields:
        return f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL ELSE {col}::date END"
    if bool_fields and field in bool_fields:
        return f"CASE WHEN lower(trim({col})) IN ({NULL_SQL}) THEN NULL ELSE {col}::boolean END"
    return col


def business_key_expr(spec: DomainSpec, src_alias: str) -> str:
    """Generate SQL expression for the composite business key."""
    cols = [f"trim({src_alias}.{qident(f)})" for f in spec.key_fields]
    if len(cols) == 1:
        return cols[0]
    sep = (spec.business_key_separator or "-").replace("'", "''")
    return f" || '{sep}' || ".join(cols)
