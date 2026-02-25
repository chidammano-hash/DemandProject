"""Shared utilities for the Demand Studio API.

Provides connection pool management, SQL helpers, domain spec wrappers,
pagination (fetch_page / list_domain), and metric expression builders.
"""
from __future__ import annotations

from typing import Any
from datetime import date
import json
import math
import os
import subprocess
import time

import psycopg
from psycopg_pool import ConnectionPool
from fastapi import HTTPException

from common.domain_specs import DOMAIN_SPECS, DomainSpec, get_spec

# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------
_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        conninfo = (
            f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
            f"port={os.getenv('POSTGRES_PORT', '5440')} "
            f"dbname={os.getenv('POSTGRES_DB', 'demand_mvp')} "
            f"user={os.getenv('POSTGRES_USER', 'demand')} "
            f"password={os.getenv('POSTGRES_PASSWORD', 'demand')}"
        )
        _pool = ConnectionPool(conninfo, min_size=2, max_size=10, open=True)
    return _pool


def get_conn():
    return _get_pool().connection()


# ---------------------------------------------------------------------------
# OpenAI client (shared by chat + intel routers)
# ---------------------------------------------------------------------------
_openai_client = None


def get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-..."):
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


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


def row_to_dict(spec: DomainSpec, row: tuple[Any, ...]) -> dict[str, Any]:
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
# Spec field introspection
# ---------------------------------------------------------------------------
def ordered_subset(cols: list[str], subset: set[str]) -> list[str]:
    return [c for c in cols if c in subset]


def numeric_fields_for_spec(spec: DomainSpec) -> list[str]:
    float_fields = ordered_subset(spec.columns, spec.float_fields)
    int_fields = ordered_subset(spec.columns, spec.int_fields)
    return [*float_fields, *[c for c in int_fields if c not in float_fields]]


def date_fields_for_spec(spec: DomainSpec) -> list[str]:
    return ordered_subset(spec.columns, spec.date_fields)


def category_fields_for_spec(spec: DomainSpec) -> list[str]:
    numeric = set(numeric_fields_for_spec(spec))
    dates = set(date_fields_for_spec(spec))
    return [c for c in spec.columns if c not in numeric and c not in dates]


def item_field_for_spec(spec: DomainSpec) -> str:
    for candidate in ("dmdunit", "item_no"):
        if candidate in spec.columns_with_ck:
            return candidate
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


def forecast_accuracy_expr(forecast_col: str = "basefcst_pref", actual_col: str = "tothist_dmd") -> str:
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
    }


# ---------------------------------------------------------------------------
# Aggregated trend source builder
# ---------------------------------------------------------------------------
def build_agg_trend_source(spec: DomainSpec, trend_metrics: list[str], filters: str) -> tuple[str, str, str, list[Any], str | None] | None:
    agg_table_map = {
        "sales": ("agg_sales_monthly", {"qty_shipped", "qty_ordered", "qty"}),
        "forecast": ("agg_forecast_monthly", {"basefcst_pref", "tothist_dmd", "accuracy_pct"}),
    }
    if spec.name not in agg_table_map:
        return None

    agg_table, allowed_metrics = agg_table_map[spec.name]
    if any(m != "__count__" and m not in allowed_metrics for m in trend_metrics):
        return None

    parsed = parse_filters_safe(filters)
    allowed_filter_cols = {"dmdunit", "loc", "startdate", "model_id"}
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

    records = [row_to_dict(spec, r) for r in rows]
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


# ---------------------------------------------------------------------------
# Benchmark timing helpers
# ---------------------------------------------------------------------------
def quote_literal(value: str) -> str:
    """Escape a string for inline SQL literals.

    Used only by the benchmark endpoint which must build identical SQL for both
    Postgres and Trino (Trino's docker exec path cannot use parameterized queries).
    """
    return "'" + value.replace("'", "''") + "'"


def parse_optional_iso_date(value: str, field_name: str) -> str:
    v = value.strip()
    if not v:
        return ""
    try:
        return date.fromisoformat(v).isoformat()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid {field_name}; expected YYYY-MM-DD") from exc


def percentile_ms(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil((p / 100.0) * len(ordered)) - 1))
    return round(ordered[idx], 3)


def summary_stats_ms(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"runs": 0, "avg_ms": None, "min_ms": None, "max_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "runs": len(values),
        "avg_ms": round(sum(values) / len(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
        "p50_ms": percentile_ms(values, 50),
        "p95_ms": percentile_ms(values, 95),
    }


def timed_postgres_query(sql: str, runs: int, warmup: int) -> list[float]:
    timings: list[float] = []
    with get_conn() as conn, conn.cursor() as cur:
        for _ in range(warmup):
            cur.execute(sql)
            cur.fetchall()
        for _ in range(runs):
            t0 = time.perf_counter()
            cur.execute(sql)
            cur.fetchall()
            timings.append((time.perf_counter() - t0) * 1000.0)
    return timings


def timed_trino_query(sql: str, runs: int, warmup: int, trino_container: str) -> list[float]:
    timings: list[float] = []
    cmd = [
        "docker",
        "exec",
        "-i",
        trino_container,
        "trino",
        "--output-format",
        "CSV_HEADER",
        "--execute",
        sql,
    ]
    for _ in range(warmup):
        warmup_proc = subprocess.run(cmd, capture_output=True, text=True)
        if warmup_proc.returncode != 0:
            err = (warmup_proc.stderr or warmup_proc.stdout or "unknown trino error").strip()
            raise RuntimeError(err)
    for _ in range(runs):
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "unknown trino error").strip()
            raise RuntimeError(err)
        timings.append(elapsed)
    return timings
