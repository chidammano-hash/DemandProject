"""Shared utilities for the Supply Chain Command Center API.

Provides connection pool management, SQL helpers, domain spec wrappers,
pagination (fetch_page / list_domain), and metric expression builders.

Connection pool and OpenAI client are implemented in dedicated sub-modules
(api.pool, api.llm) and re-exported here for backward compatibility.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any
from datetime import date
import json

from fastapi import HTTPException

from common.core.constants import FORECAST_QTY_COL
from common.core.domain_specs import DomainSpec, get_spec

# ---------------------------------------------------------------------------
# Re-exports from sub-modules (backward compatibility)
# ---------------------------------------------------------------------------
from api.pool import (  # noqa: F401
    _get_async_pool,
    _get_async_read_pool,
    _get_pool,
    _get_read_pool,
    _read_replica_configured,
)
from api.llm import get_openai  # noqa: F401

logger = logging.getLogger(__name__)


def get_conn():
    """Return a connection context manager from the shared pool.

    Defined here (not re-exported) so that ``patch("api.core._get_pool")``
    in tests correctly intercepts the pool used by this function.
    """
    return _get_pool().connection()


@contextmanager
def get_read_only_conn():
    """Yield a sync connection for read-only analytics queries.

    Routes to the read replica pool when ``READ_REPLICA_URL`` is configured;
    otherwise falls back to the primary pool — the env-unset code path is
    behaviour-identical to ``get_conn()``.

    Sets ``SET TRANSACTION READ ONLY`` at the session level as a safety net
    so a caller that accidentally issues a write gets a clear error rather
    than silently succeeding on the primary fallback. If setting the flag
    fails (rare; e.g. mock pools in tests), we log + continue rather than
    fail the request — read-only is a defensive guard, not a correctness
    requirement.

    Lag awareness: read replicas can lag the primary by seconds. Use ONLY
    for queries that tolerate eventual consistency (analytics dashboards,
    long-window aggregates). Do NOT use for read-after-write flows where
    the user expects to see their own write reflected.

    For async handlers use :func:`get_async_read_only_conn` instead.
    """
    pool = _get_read_pool() if _read_replica_configured() else _get_pool()
    with pool.connection() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
        except Exception as exc:  # noqa: BLE001 — defensive, never block the request
            logger.debug("Could not set TRANSACTION READ ONLY (likely a mock pool): %s", exc)
        yield conn


@asynccontextmanager
async def get_async_read_only_conn():
    """Async sibling of :func:`get_read_only_conn`.

    Routes to the async read-replica pool when ``READ_REPLICA_URL`` is set;
    otherwise falls back to the primary async pool. The env-unset code path
    is behaviour-identical to ``get_async_conn()`` so existing handlers can
    opt in by swapping the helper name and nothing else.

    Same lag-awareness contract as :func:`get_read_only_conn`: callers MUST
    only use this for queries that tolerate eventual consistency.
    """
    pool = _get_async_read_pool() if _read_replica_configured() else _get_async_pool()
    async with pool.connection() as conn:
        try:
            async with conn.cursor() as cur:
                await cur.execute("SET TRANSACTION READ ONLY")
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.debug("Could not set TRANSACTION READ ONLY on async conn: %s", exc)
        yield conn


def get_async_conn():
    """Return an async connection context manager from the async pool.

    Used by handlers converted to ``async def`` (Item 19 pilot —
    customer_analytics + GET endpoints in inv_planning_insights).

    Defined here (not re-exported) so that
    ``patch("api.core._get_async_pool")`` in tests correctly intercepts the
    pool used by this function. Returns the value of ``pool.connection()``
    directly — caller uses ``async with get_async_conn() as conn:``.
    """
    return _get_async_pool().connection()


# ---------------------------------------------------------------------------
# Nullable type coercions (shared across router modules)
# ---------------------------------------------------------------------------

def _f(v: Any) -> float | None:
    """Coerce a Postgres numeric value to float, returning None for NULL."""
    return float(v) if v is not None else None


def _s(v: Any) -> str | None:
    """Coerce a Postgres value to str, returning None for NULL."""
    return str(v) if v is not None else None


# ---------------------------------------------------------------------------
# Domain spec wrapper (raises 404 instead of ValueError)
# ---------------------------------------------------------------------------
def get_spec_or_404(domain: str) -> DomainSpec:
    """Resolve a domain name to its DomainSpec, raising HTTP 404 on unknown domains."""
    try:
        return get_spec(domain)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------
def set_cache(response, max_age: int, stale_while_revalidate: int = 0):
    """Set Cache-Control header on a response."""
    parts = [f"max-age={max_age}"]
    if stale_while_revalidate:
        parts.append(f"stale-while-revalidate={stale_while_revalidate}")
    response.headers["Cache-Control"] = ", ".join(parts)


# ---------------------------------------------------------------------------
# Column name mapping (reserved word workaround: class → class_)
# ---------------------------------------------------------------------------
def to_api_col(col: str) -> str:
    return "class_" if col == "class" else col


def to_sql_col(spec: DomainSpec, col: str) -> str:
    c = (col or "").strip()
    if c == "class_" and "class" in spec.columns_with_ck:
        return "class"
    return c


def domain_row_to_dict(spec: DomainSpec, row: tuple[Any, ...]) -> dict[str, Any]:
    """Convert a row to a dict using a :class:`DomainSpec` column list.

    Domain-spec-aware: applies :func:`to_api_col` (e.g. ``class`` -> ``class_``)
    so the resulting keys match the public API field names. Distinct from the
    generic :func:`common.core.sql_helpers.row_to_dict_from_cols` helper
    because of the spec-driven name translation.
    """
    out: dict[str, Any] = {}
    for idx, name in enumerate(spec.columns_with_ck):
        out[to_api_col(name)] = row[idx]
    return out


# ---------------------------------------------------------------------------
# SQL identifier quoting
# ---------------------------------------------------------------------------
def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def dotted_qident(*parts: str) -> str:
    return ".".join(qident(p) for p in parts if p)


# ---------------------------------------------------------------------------
# Spec field introspection (cached by spec.name)
# ---------------------------------------------------------------------------
def ordered_subset(cols: list[str], subset: set[str]) -> list[str]:
    return [c for c in cols if c in subset]


_field_cache: dict[str, dict[str, list[str]]] = {}


def _get_field_cache(spec: DomainSpec) -> dict[str, list[str]]:
    if spec.name not in _field_cache:
        float_fields = ordered_subset(spec.columns, spec.float_fields)
        int_fields = ordered_subset(spec.columns, spec.int_fields)
        numeric = [*float_fields, *[c for c in int_fields if c not in float_fields]]
        dates = ordered_subset(spec.columns, spec.date_fields)
        numeric_set = set(numeric)
        date_set = set(dates)
        category = [c for c in spec.columns if c not in numeric_set and c not in date_set]
        _field_cache[spec.name] = {"numeric": numeric, "date": dates, "category": category}
    return _field_cache[spec.name]


def numeric_fields_for_spec(spec: DomainSpec) -> list[str]:
    return _get_field_cache(spec)["numeric"]


def date_fields_for_spec(spec: DomainSpec) -> list[str]:
    return _get_field_cache(spec)["date"]


def category_fields_for_spec(spec: DomainSpec) -> list[str]:
    return _get_field_cache(spec)["category"]


def item_field_for_spec(spec: DomainSpec) -> str:
    if "item_id" in spec.columns_with_ck:
        return "item_id"
    return ""


def location_field_for_spec(spec: DomainSpec) -> str:
    for candidate in ("loc", "location_id"):
        if candidate in spec.columns_with_ck:
            return candidate
    return ""


def default_date_field_for_spec(spec: DomainSpec) -> str:
    dates = date_fields_for_spec(spec)
    for candidate in ("startdate", "fcstdate", "date_key"):
        if candidate in dates:
            return candidate
    return dates[0] if dates else ""


def default_trend_metric_for_spec(spec: DomainSpec) -> str:
    numeric = numeric_fields_for_spec(spec)
    excluded = {"type", "lag", "execution_lag"}
    for col in numeric:
        if col not in excluded:
            return col
    return numeric[0] if numeric else "__count__"


# ---------------------------------------------------------------------------
# Filter / WHERE clause building
# ---------------------------------------------------------------------------
def parse_filters_json(filters: str) -> dict[str, str]:
    if not filters.strip():
        return {}
    try:
        parsed = json.loads(filters)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="Invalid filters JSON") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="filters must be a JSON object")
    out: dict[str, str] = {}
    for raw_key, val in parsed.items():
        key = str(raw_key).strip()
        sval = str(val).strip() if val is not None else ""
        if key and sval:
            out[key] = sval
    return out


def parse_filters_safe(filters: str) -> dict[str, str]:
    if not filters.strip():
        return {}
    return parse_filters_json(filters)


def _col_type(spec: DomainSpec, col: str) -> str:
    """Return 'int', 'float', 'date', or 'text' for a column."""
    if col in spec.int_fields:
        return "int"
    if col in spec.float_fields:
        return "float"
    if col in spec.date_fields:
        return "date"
    return "text"


def _typed_eq_clause(spec: DomainSpec, col: str, val: str, where: list[str], params: list[Any]) -> None:
    """Append a type-aware equality clause. Uses native types to leverage B-tree indexes."""
    ctype = _col_type(spec, col)
    qcol = qident(col)
    if ctype == "text":
        where.append(f"{qcol} = %s")
        params.append(val)
    elif ctype == "int":
        try:
            where.append(f"{qcol} = %s")
            params.append(int(val))
        except ValueError:
            where.append(f"{qcol}::text = %s")
            params.append(val)
    elif ctype == "float":
        try:
            where.append(f"{qcol} = %s")
            params.append(float(val))
        except ValueError:
            where.append(f"{qcol}::text = %s")
            params.append(val)
    elif ctype == "date":
        try:
            date.fromisoformat(val)
            where.append(f"{qcol} = %s::date")
            params.append(val)
        except ValueError:
            where.append(f"{qcol}::text = %s")
            params.append(val)


def _typed_like_clause(spec: DomainSpec, col: str, val: str, where: list[str], params: list[Any]) -> None:
    """Append a type-aware substring clause. Avoids ::text cast on text columns for index use."""
    ctype = _col_type(spec, col)
    qcol = qident(col)
    if ctype == "text":
        where.append(f"{qcol} ILIKE %s")
        params.append(f"%{val}%")
    elif ctype == "int":
        try:
            where.append(f"{qcol} = %s")
            params.append(int(val))
        except ValueError:
            where.append(f"{qcol}::text ILIKE %s")
            params.append(f"%{val}%")
    elif ctype == "float":
        try:
            where.append(f"{qcol} = %s")
            params.append(float(val))
        except ValueError:
            where.append(f"{qcol}::text ILIKE %s")
            params.append(f"%{val}%")
    elif ctype == "date":
        where.append(f"{qcol}::text LIKE %s")
        params.append(f"%{val}%")


def build_where(spec: DomainSpec, q: str, filters: str) -> tuple[str, list[Any]]:
    where: list[str] = []
    params: list[Any] = []

    if q.strip():
        term = f"%{q.strip()}%"
        clauses = []
        for c in spec.search_fields:
            if _col_type(spec, c) == "text":
                clauses.append(f"{qident(c)} ILIKE %s")
            else:
                clauses.append(f"{qident(c)}::text ILIKE %s")
        where.append("(" + " OR ".join(clauses) + ")")
        params.extend([term] * len(spec.search_fields))

    if filters.strip():
        parsed = parse_filters_json(filters)
        allowed = set(spec.columns_with_ck)
        for raw_key, sval in parsed.items():
            col = to_sql_col(spec, str(raw_key).strip())
            if not col:
                continue
            if col not in allowed:
                raise HTTPException(status_code=422, detail=f"Invalid filter column: {raw_key}")
            if sval.startswith("="):
                exact = sval[1:].strip()
                if not exact:
                    continue
                _typed_eq_clause(spec, col, exact, where, params)
            else:
                _typed_like_clause(spec, col, sval, where, params)

    return (f"WHERE {' AND '.join(where)}" if where else "", params)


# ---------------------------------------------------------------------------
# Metric expression builders
# ---------------------------------------------------------------------------
def grouped_metric_expr(metric_col: str) -> str:
    if metric_col == "__count__":
        return "count(*)::double precision"
    return f"coalesce(sum({qident(metric_col)}), 0)::double precision"


def grouped_metric_expr_with_count(metric_col: str, count_col: str | None = None) -> str:
    if metric_col == "__count__":
        if count_col:
            return f"coalesce(sum({qident(count_col)}), 0)::double precision"
        return "count(*)::double precision"
    return f"coalesce(sum({qident(metric_col)}), 0)::double precision"


def forecast_accuracy_expr(forecast_col: str = FORECAST_QTY_COL, actual_col: str = "tothist_dmd") -> str:
    return (
        "CASE WHEN abs(coalesce(sum("
        + qident(actual_col)
        + "), 0)) > 0 THEN "
        + "(100.0 - (100.0 * coalesce(sum(abs("
        + qident(forecast_col)
        + " - "
        + qident(actual_col)
        + ")), 0) / abs(coalesce(sum("
        + qident(actual_col)
        + "), 0)))) "
        + "ELSE NULL END::double precision"
    )


# ---------------------------------------------------------------------------
# KPI computation
# ---------------------------------------------------------------------------
def compute_kpis(sum_forecast: float, sum_actual: float, sum_abs_error: float, dfu_count: int = 0) -> dict[str, Any]:
    wape = (100.0 * sum_abs_error / abs(sum_actual)) if sum_actual != 0 else None
    bias = ((sum_forecast / sum_actual) - 1.0) if sum_actual != 0 else None
    accuracy_pct = (100.0 - wape) if wape is not None else None
    return {
        "accuracy_pct": round(accuracy_pct, 4) if accuracy_pct is not None else None,
        "wape": round(wape, 4) if wape is not None else None,
        "bias": round(bias, 4) if bias is not None else None,
        "sum_forecast": round(sum_forecast, 2),
        "sum_actual": round(sum_actual, 2),
        "dfu_count": dfu_count,
        "sku_count": dfu_count,
    }


# ---------------------------------------------------------------------------
# Aggregated trend source builder
# ---------------------------------------------------------------------------
def build_agg_trend_source(spec: DomainSpec, trend_metrics: list[str], filters: str) -> tuple[str, str, str, list[Any], str | None] | None:
    agg_table_map = {
        "sales": ("agg_sales_monthly", {"qty_shipped", "qty_ordered", "qty"}),
        "forecast": ("agg_forecast_monthly", {FORECAST_QTY_COL, "tothist_dmd", "accuracy_pct"}),
    }
    if spec.name not in agg_table_map:
        return None

    agg_table, allowed_metrics = agg_table_map[spec.name]
    if any(m != "__count__" and m not in allowed_metrics for m in trend_metrics):
        return None

    parsed = parse_filters_safe(filters)
    allowed_filter_cols = {"item_id", "loc", "startdate", "model_id"}
    agg_col_types = {"month_start": "date"}
    where: list[str] = []
    params: list[Any] = []
    for raw_key, sval in parsed.items():
        source_col = to_sql_col(spec, raw_key)
        if source_col not in allowed_filter_cols:
            return None
        target_col = "month_start" if source_col == "startdate" else source_col
        ctype = agg_col_types.get(target_col, "text")
        if sval.startswith("="):
            exact = sval[1:].strip()
            if not exact:
                continue
            if ctype == "text":
                where.append(f"{qident(target_col)} = %s")
                params.append(exact)
            elif ctype == "date":
                try:
                    date.fromisoformat(exact)
                    where.append(f"{qident(target_col)} = %s::date")
                    params.append(exact)
                except ValueError:
                    where.append(f"{qident(target_col)}::text = %s")
                    params.append(exact)
        else:
            if ctype == "text":
                where.append(f"{qident(target_col)} ILIKE %s")
                params.append(f"%{sval}%")
            else:
                where.append(f"{qident(target_col)}::text LIKE %s")
                params.append(f"%{sval}%")

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    return agg_table, "month_start", where_sql, params, "row_count"


# ---------------------------------------------------------------------------
# Generic row helpers (used across all router modules)
# ---------------------------------------------------------------------------

def rows_to_dicts(cols: list[str], rows: list[tuple]) -> list[dict[str, Any]]:
    """Convert a list of DB row tuples to a list of dicts using a column name list."""
    return [dict(zip(cols, r)) for r in rows]


def add_cross_dim_filters(
    where: list[str],
    params: list[Any],
    *,
    brand: str | None = None,
    category: str | None = None,
    market: str | None = None,
    item_col: str = "t.item_id",
    loc_col: str = "t.loc",
) -> None:
    """Append brand/category/market EXISTS subquery filters to WHERE parts.

    This is the shared implementation for the global filter bar's cross-dimension
    filters (brand → dim_item.brand_name, category → dim_item.class,
    market → dim_location.state_id).  Each value may be comma-separated for
    multi-select.

    Args:
        where: Mutable list of WHERE clause fragments.
        params: Mutable list of query parameters, appended in tandem with *where*.
        brand: Comma-separated brand_name values (dim_item).
        category: Comma-separated class values (dim_item.class).
        market: Comma-separated state_id values (dim_location).
        item_col: Qualified column expression for the item identifier in the outer
            query (e.g. ``"t.item_id"``, ``"s.item_id"``, ``"f.item_id"``).
        loc_col: Qualified column expression for the location identifier in the
            outer query (e.g. ``"t.loc"``, ``"f.loc"``).
    """
    if brand:
        values = [v.strip() for v in brand.split(",") if v.strip()]
        if values:
            params.append(values)
            where.append(
                f"EXISTS (SELECT 1 FROM dim_item di WHERE di.item_id = {item_col} AND di.brand_name = ANY(%s))"
            )
    if category:
        values = [v.strip() for v in category.split(",") if v.strip()]
        if values:
            params.append(values)
            where.append(
                f"EXISTS (SELECT 1 FROM dim_item di WHERE di.item_id = {item_col} AND di.class = ANY(%s))"
            )
    if market:
        values = [v.strip() for v in market.split(",") if v.strip()]
        if values:
            params.append(values)
            where.append(
                f"EXISTS (SELECT 1 FROM dim_location dl WHERE dl.location_id = {loc_col} AND dl.state_id = ANY(%s))"
            )


def add_item_loc_filters(
    where: list[str],
    params: list[Any],
    item: str | None,
    loc: str | None,
    *,
    table_alias: str = "",
    item_col: str = "item_id",
    loc_col: str = "loc",
) -> None:
    """Append ILIKE filter clauses for item and location to an in-progress WHERE parts list.

    Args:
        where: Mutable list of WHERE clause fragments (e.g. ``["status = %s"]``).
        params: Mutable list of query parameters, appended in tandem with *where*.
        item: Optional item filter value; adds ``ILIKE '%value%'`` clause when non-empty.
        loc: Optional location filter value; adds ``ILIKE '%value%'`` clause when non-empty.
        table_alias: Optional table alias prefix, e.g. ``"s"`` produces ``s.item_id``.
        item_col: Column name for the item identifier (default ``"item_id"``).
        loc_col: Column name for the location identifier (default ``"loc"``).
    """
    prefix = f"{table_alias}." if table_alias else ""
    if item:
        where.append(f"{prefix}{item_col} ILIKE %s")
        params.append(f"%{item}%")
    if loc:
        where.append(f"{prefix}{loc_col} ILIKE %s")
        params.append(f"%{loc}%")


# ---------------------------------------------------------------------------
# Pagination helpers
# ---------------------------------------------------------------------------
_LARGE_TABLES = {"fact_external_forecast_monthly", "fact_sales_monthly"}
_MAX_COUNT_SCAN = 100_001


def fetch_page(
    spec: DomainSpec,
    limit: int,
    offset: int,
    q: str,
    filters: str,
    sort_by: str,
    sort_dir: str,
) -> dict[str, Any]:
    allowed_sort = set(spec.columns_with_ck)
    sort_sql = to_sql_col(spec, sort_by)
    order_col = sort_sql if sort_sql in allowed_sort else spec.default_sort
    order_dir = "DESC" if sort_dir.lower() == "desc" else "ASC"
    tie_breaker = f", {spec.default_sort} ASC" if order_col != spec.default_sort else ""

    where_sql, params = build_where(spec, q, filters)
    select_cols = ", ".join(spec.columns_with_ck)

    total_approximate = False
    if not where_sql:
        count_sql = "SELECT COALESCE(c.reltuples, 0)::bigint FROM pg_class c WHERE c.relname = %s"
        count_params: list[Any] = [spec.table]
    elif spec.table in _LARGE_TABLES:
        count_sql = f"SELECT count(*) FROM (SELECT 1 FROM {spec.table} {where_sql} LIMIT {_MAX_COUNT_SCAN}) _sub"
        count_params = list(params)
    else:
        count_sql = f"SELECT count(*) FROM {spec.table} {where_sql}"
        count_params = list(params)

    data_sql = f"""
      SELECT {select_cols}
      FROM {spec.table}
      {where_sql}
      ORDER BY {order_col} {order_dir}{tie_breaker}
      LIMIT %s OFFSET %s
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(count_sql, count_params)
        total = int(cur.fetchone()[0])
        if total >= _MAX_COUNT_SCAN:
            total_approximate = True
        cur.execute(data_sql, [*params, limit, offset])
        rows = cur.fetchall()

    records = [domain_row_to_dict(spec, r) for r in rows]
    result: dict[str, Any] = {
        "total": total,
        "limit": limit,
        "offset": offset,
        spec.plural: records,
    }
    if total_approximate:
        result["total_approximate"] = True
    return result


def list_domain(spec: DomainSpec, limit: int) -> list[dict[str, Any]]:
    page = fetch_page(
        spec=spec,
        limit=limit,
        offset=0,
        q="",
        filters="",
        sort_by=spec.default_sort,
        sort_dir="asc",
    )
    return page[spec.plural]

