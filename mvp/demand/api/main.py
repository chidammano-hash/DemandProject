from __future__ import annotations

from typing import Any
from datetime import date
import json
import math
import os
import subprocess
import time

import re

from dotenv import load_dotenv
load_dotenv()

import psycopg
from psycopg_pool import ConnectionPool
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from common.domain_specs import DOMAIN_SPECS, DomainSpec, get_spec


app = FastAPI(title="Demand Unified MVP API")

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


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def dotted_qident(*parts: str) -> str:
    return ".".join(qident(p) for p in parts if p)


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


def build_where(spec: DomainSpec, q: str, filters: str) -> tuple[str, list[Any]]:
    where: list[str] = []
    params: list[Any] = []

    if q.strip():
        term = f"%{q.strip()}%"
        where.append("(" + " OR ".join([f"{c}::text ILIKE %s" for c in spec.search_fields]) + ")")
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
            # Exact-match mode for UI filter values with "=" prefix (example: "=109101")
            if sval.startswith("="):
                exact = sval[1:].strip()
                if not exact:
                    continue
                where.append(f"{col}::text = %s")
                params.append(exact)
            else:
                where.append(f"{col}::text ILIKE %s")
                params.append(f"%{sval}%")

    return (f"WHERE {' AND '.join(where)}" if where else "", params)


@app.get("/domains/{domain}/suggest")
def domain_suggest(
    domain: str,
    field: str = Query(..., min_length=1),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    limit: int = Query(default=12, ge=1, le=100),
):
    spec = get_spec(domain)
    target_col = to_sql_col(spec, field)
    allowed = set(spec.columns_with_ck)
    if target_col not in allowed:
        raise HTTPException(status_code=422, detail=f"Invalid suggest field: {field}")

    where: list[str] = []
    params: list[Any] = []
    if q.strip():
        where.append(f"{qident(target_col)}::text ILIKE %s")
        params.append(f"{q.strip()}%")

    scoped_filters = parse_filters_json(filters)
    for raw_key, sval in scoped_filters.items():
        col = to_sql_col(spec, raw_key)
        if col == target_col:
            continue
        if col not in allowed:
            raise HTTPException(status_code=422, detail=f"Invalid filter column: {raw_key}")
        if sval.startswith("="):
            exact = sval[1:].strip()
            if not exact:
                continue
            where.append(f"{qident(col)}::text = %s")
            params.append(exact)
        else:
            where.append(f"{qident(col)}::text ILIKE %s")
            params.append(f"%{sval}%")

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    sql = f"""
      SELECT DISTINCT {qident(target_col)}::text AS val
      FROM {qident(spec.table)}
      {where_sql}
      ORDER BY 1
      LIMIT %s
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, [*params, limit])
        values = [r[0] for r in cur.fetchall() if r and r[0] is not None]
    return {
        "domain": spec.name,
        "field": to_api_col(target_col),
        "values": values,
    }


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

    count_sql = f"SELECT count(*) FROM {spec.table} {where_sql}"
    data_sql = f"""
      SELECT {select_cols}
      FROM {spec.table}
      {where_sql}
      ORDER BY {order_col} {order_dir}{tie_breaker}
      LIMIT %s OFFSET %s
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(count_sql, params)
        total = int(cur.fetchone()[0])
        cur.execute(data_sql, [*params, limit, offset])
        rows = cur.fetchall()

    records = [row_to_dict(spec, r) for r in rows]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        spec.plural: records,
    }


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


@app.get("/", include_in_schema=False)
def root():
    return {
        "status": "ok",
        "ui_url": "http://127.0.0.1:5173",
        "api_docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "domains": sorted(DOMAIN_SPECS.keys()),
    }


@app.get("/domains")
def list_domains():
    return {
        "domains": sorted(DOMAIN_SPECS.keys()),
    }


@app.get("/domains/{domain}/meta")
def domain_meta(domain: str):
    spec = get_spec(domain)
    return {
        "name": spec.name,
        "plural": spec.plural,
        "table": spec.table,
        "default_sort": to_api_col(spec.default_sort),
        "columns": [to_api_col(c) for c in spec.columns_with_ck],
        "numeric_fields": [to_api_col(c) for c in numeric_fields_for_spec(spec)],
        "date_fields": [to_api_col(c) for c in date_fields_for_spec(spec)],
        "category_fields": [to_api_col(c) for c in category_fields_for_spec(spec)],
    }


@app.get("/domains/{domain}/sample-pair")
def domain_sample_pair(domain: str):
    spec = get_spec(domain)
    item_col = item_field_for_spec(spec)
    location_col = location_field_for_spec(spec)
    if not item_col or not location_col:
        raise HTTPException(status_code=404, detail=f"Domain '{spec.name}' does not support item+location sampling")

    table_sql = qident(spec.table)
    sample_sql = f"""
      SELECT {qident(item_col)}::text AS item_value, {qident(location_col)}::text AS location_value
      FROM {table_sql}
      TABLESAMPLE SYSTEM (1)
      WHERE {qident(item_col)} IS NOT NULL
        AND {qident(location_col)} IS NOT NULL
      LIMIT 1
    """
    fallback_sql = f"""
      SELECT {qident(item_col)}::text AS item_value, {qident(location_col)}::text AS location_value
      FROM {table_sql}
      WHERE {qident(item_col)} IS NOT NULL
        AND {qident(location_col)} IS NOT NULL
      ORDER BY random()
      LIMIT 1
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sample_sql)
        row = cur.fetchone()
        if not row:
            cur.execute(fallback_sql)
            row = cur.fetchone()

    if not row:
        return {
            "domain": spec.name,
            "item_field": to_api_col(item_col),
            "location_field": to_api_col(location_col),
            "item": None,
            "location": None,
        }

    return {
        "domain": spec.name,
        "item_field": to_api_col(item_col),
        "location_field": to_api_col(location_col),
        "item": row[0],
        "location": row[1],
    }


@app.get("/domains/{domain}")
def list_domain_records(domain: str, limit: int = Query(default=50, ge=1, le=1000)):
    spec = get_spec(domain)
    return list_domain(spec, limit)


@app.get("/domains/{domain}/page")
def list_domain_records_page(
    domain: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default=""),
    sort_dir: str = Query(default="asc"),
):
    spec = get_spec(domain)
    return fetch_page(
        spec=spec,
        limit=limit,
        offset=offset,
        q=q,
        filters=filters,
        sort_by=sort_by or to_api_col(spec.default_sort),
        sort_dir=sort_dir,
    )


def grouped_metric_expr(metric_col: str) -> str:
    if metric_col == "__count__":
        return "count(*)::double precision"
    return f"coalesce(sum({qident(metric_col)}), 0)::double precision"


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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


def parse_filters_safe(filters: str) -> dict[str, str]:
    if not filters.strip():
        return {}
    return parse_filters_json(filters)


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
    where: list[str] = []
    params: list[Any] = []
    for raw_key, sval in parsed.items():
        source_col = to_sql_col(spec, raw_key)
        if source_col not in allowed_filter_cols:
            return None
        target_col = "month_start" if source_col == "startdate" else source_col
        if sval.startswith("="):
            exact = sval[1:].strip()
            if not exact:
                continue
            where.append(f"{qident(target_col)}::text = %s")
            params.append(exact)
        else:
            where.append(f"{qident(target_col)}::text ILIKE %s")
            params.append(f"%{sval}%")

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    return agg_table, "month_start", where_sql, params, "row_count"


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


@app.get("/bench/compare")
def benchmark_postgres_vs_trino(
    domain: str = Query(default="sales", min_length=1, max_length=40),
    runs: int = Query(default=5, ge=1, le=30),
    warmup: int = Query(default=1, ge=0, le=5),
    limit: int = Query(default=200, ge=10, le=1000),
    points: int = Query(default=24, ge=3, le=120),
    item: str = Query(default="", max_length=120),
    location: str = Query(default="", max_length=120),
    start_date: str = Query(default="", max_length=10),
    end_date: str = Query(default="", max_length=10),
    trino_container: str = Query(default="demand-mvp-trino", min_length=1, max_length=120),
    trino_catalog: str = Query(default="iceberg", min_length=1, max_length=80),
    trino_schema: str = Query(default="silver", min_length=1, max_length=80),
):
    try:
        spec = get_spec(domain)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    item_col = item_field_for_spec(spec)
    location_col = location_field_for_spec(spec)
    date_col = default_date_field_for_spec(spec)
    trend_metric = default_trend_metric_for_spec(spec)

    where_clauses: list[str] = []
    if item.strip():
        if not item_col:
            raise HTTPException(status_code=422, detail=f"Domain '{spec.name}' does not have an item field")
        where_clauses.append(f"CAST({qident(item_col)} AS VARCHAR) = {quote_literal(item.strip())}")
    if location.strip():
        if not location_col:
            raise HTTPException(status_code=422, detail=f"Domain '{spec.name}' does not have a location field")
        where_clauses.append(f"CAST({qident(location_col)} AS VARCHAR) = {quote_literal(location.strip())}")

    start_iso = parse_optional_iso_date(start_date, "start_date")
    end_iso = parse_optional_iso_date(end_date, "end_date")
    if (start_iso or end_iso) and not date_col:
        raise HTTPException(status_code=422, detail=f"Domain '{spec.name}' does not have a date field")
    if start_iso:
        where_clauses.append(f"CAST({qident(date_col)} AS DATE) >= DATE {quote_literal(start_iso)}")
    if end_iso:
        where_clauses.append(f"CAST({qident(date_col)} AS DATE) <= DATE {quote_literal(end_iso)}")
    if start_iso and end_iso and start_iso > end_iso:
        raise HTTPException(status_code=422, detail="start_date must be <= end_date")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    pg_table_sql = qident(spec.table)
    trino_table_sql = dotted_qident(trino_catalog.strip(), trino_schema.strip(), spec.table)
    sort_sql = qident(spec.default_sort)
    select_cols_sql = ", ".join(qident(c) for c in spec.columns_with_ck[: min(len(spec.columns_with_ck), 10)])

    query_templates: dict[str, str] = {
        "count": "SELECT count(*) AS cnt FROM {table} {where_sql}",
        "page": (
            f"SELECT {select_cols_sql} "
            "FROM {table} {where_sql} "
            f"ORDER BY {sort_sql} ASC LIMIT {limit}"
        ),
    }
    if date_col:
        metric_sql = "count(*)" if trend_metric == "__count__" else f"coalesce(sum({qident(trend_metric)}), 0)"
        query_templates["trend"] = (
            f"SELECT date_trunc('month', CAST({qident(date_col)} AS DATE)) AS bucket, {metric_sql} AS metric "
            "FROM {table} {where_sql} "
            "GROUP BY 1 "
            "ORDER BY 1 DESC "
            f"LIMIT {points}"
        )

    results: list[dict[str, Any]] = []
    for name, template in query_templates.items():
        pg_sql = template.format(table=pg_table_sql, where_sql=where_sql)
        trino_sql = template.format(table=trino_table_sql, where_sql=where_sql)
        pg_runs = timed_postgres_query(pg_sql, runs=runs, warmup=warmup)
        try:
            trino_runs = timed_trino_query(trino_sql, runs=runs, warmup=warmup, trino_container=trino_container)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Trino benchmark failed for query '{name}': {exc}",
            ) from exc

        pg_stats = summary_stats_ms(pg_runs)
        trino_stats = summary_stats_ms(trino_runs)
        pg_avg = pg_stats["avg_ms"]
        trino_avg = trino_stats["avg_ms"]

        faster = "tie"
        speedup: float | None = 1.0
        if isinstance(pg_avg, float) and isinstance(trino_avg, float):
            if pg_avg < trino_avg:
                faster = "postgres"
                speedup = round(trino_avg / pg_avg, 3) if pg_avg > 0 else None
            elif trino_avg < pg_avg:
                faster = "trino"
                speedup = round(pg_avg / trino_avg, 3) if trino_avg > 0 else None

        results.append(
            {
                "query": name,
                "postgres_sql": pg_sql,
                "trino_sql": trino_sql,
                "postgres": {"runs_ms": [round(x, 3) for x in pg_runs], "stats": pg_stats},
                "trino": {"runs_ms": [round(x, 3) for x in trino_runs], "stats": trino_stats},
                "faster_backend": faster,
                "speedup_factor": speedup,
            }
        )

    return {
        "domain": spec.name,
        "table": spec.table,
        "iceberg_table": f"{trino_catalog.strip()}.{trino_schema.strip()}.{spec.table}",
        "filters": {
            "item_field": item_col or None,
            "item": item.strip() or None,
            "location_field": location_col or None,
            "location": location.strip() or None,
            "date_field": date_col or None,
            "start_date": start_iso or None,
            "end_date": end_iso or None,
        },
        "config": {
            "runs": runs,
            "warmup": warmup,
            "limit": limit,
            "points": points,
            "trend_metric": to_api_col(trend_metric) if trend_metric != "__count__" else "__count__",
            "trino_container": trino_container,
            "trino_catalog": trino_catalog.strip(),
            "trino_schema": trino_schema.strip(),
        },
        "results": results,
    }


@app.get("/domains/forecast/models")
def forecast_models():
    """Return distinct model_id values from the forecast table."""
    spec = get_spec("forecast")
    sql = f"SELECT DISTINCT model_id FROM {qident(spec.table)} ORDER BY 1"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        models = [r[0] for r in cur.fetchall() if r[0] is not None]
    return {"domain": "forecast", "models": models}


_ACCURACY_SLICE_DIMS = {
    "cluster_assignment",
    "ml_cluster",
    "supplier_desc",
    "abc_vol",
    "region",
    "brand_desc",
    "dfu_execution_lag",
    "month_start",
    "lag",
    "model_id",
}


def _compute_kpis(sum_forecast: float, sum_actual: float, sum_abs_error: float) -> dict[str, Any]:
    wape = (100.0 * sum_abs_error / abs(sum_actual)) if sum_actual != 0 else None
    bias = ((sum_forecast / sum_actual) - 1.0) if sum_actual != 0 else None
    accuracy_pct = (100.0 - wape) if wape is not None else None
    return {
        "accuracy_pct": round(accuracy_pct, 4) if accuracy_pct is not None else None,
        "wape": round(wape, 4) if wape is not None else None,
        "bias": round(bias, 4) if bias is not None else None,
        "sum_forecast": round(sum_forecast, 2),
        "sum_actual": round(sum_actual, 2),
    }


@app.get("/forecast/accuracy/slice")
def forecast_accuracy_slice(
    group_by: str = Query(default="cluster_assignment", max_length=64),
    models: str = Query(default="", max_length=500),
    lag: int = Query(default=-1, ge=-1, le=4),
    month_from: str = Query(default="", max_length=20),
    month_to: str = Query(default="", max_length=20),
    cluster_assignment: str = Query(default="", max_length=120),
    supplier_desc: str = Query(default="", max_length=120),
    abc_vol: str = Query(default="", max_length=40),
    region: str = Query(default="", max_length=120),
):
    """Return accuracy KPIs grouped by a chosen DFU-attribute dimension.

    Uses pre-aggregated agg_accuracy_by_dim for O(1) performance at aggregate level.
    Falls back gracefully if the view has no data yet.
    """
    if group_by not in _ACCURACY_SLICE_DIMS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid group_by '{group_by}'. Valid: {sorted(_ACCURACY_SLICE_DIMS)}",
        )

    model_list = [m.strip() for m in models.split(",") if m.strip()] if models.strip() else []

    where_parts: list[str] = []
    params: list[Any] = []

    if model_list:
        placeholders = ",".join(["%s"] * len(model_list))
        where_parts.append(f"model_id IN ({placeholders})")
        params.extend(model_list)
    if lag == -1:
        # execution lag: lag column equals dfu_execution_lag
        where_parts.append("lag::text = dfu_execution_lag")
    elif lag >= 0:
        where_parts.append("lag = %s")
        params.append(lag)
    if month_from.strip():
        where_parts.append("month_start >= %s::date")
        params.append(month_from.strip())
    if month_to.strip():
        where_parts.append("month_start <= %s::date")
        params.append(month_to.strip())
    if cluster_assignment.strip():
        where_parts.append("cluster_assignment = %s")
        params.append(cluster_assignment.strip())
    if supplier_desc.strip():
        where_parts.append("supplier_desc = %s")
        params.append(supplier_desc.strip())
    if abc_vol.strip():
        where_parts.append("abc_vol = %s")
        params.append(abc_vol.strip())
    if region.strip():
        where_parts.append("region = %s")
        params.append(region.strip())

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    sql = f"""
        SELECT
            {group_by} AS bucket,
            model_id,
            SUM(row_count)::bigint   AS n_rows,
            SUM(sum_forecast)        AS sum_forecast,
            SUM(sum_actual)          AS sum_actual,
            SUM(sum_abs_error)       AS sum_abs_error
        FROM agg_accuracy_by_dim
        {where_sql}
        GROUP BY 1, 2
        ORDER BY 1 ASC NULLS LAST, 2 ASC
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        db_rows = cur.fetchall()

    # Pivot: bucket → {model_id: kpis}
    pivot: dict[str, dict[str, Any]] = {}
    for bucket, model_id, n_rows, sf, sa, sae in db_rows:
        b = str(bucket) if bucket is not None else "(unknown)"
        if b not in pivot:
            pivot[b] = {"bucket": b, "n_rows": 0, "by_model": {}}
        pivot[b]["n_rows"] = int(pivot[b]["n_rows"]) + int(n_rows or 0)
        pivot[b]["by_model"][model_id] = _compute_kpis(float(sf or 0), float(sa or 0), float(sae or 0))

    return {
        "group_by": group_by,
        "lag_filter": lag,
        "models": model_list or None,
        "rows": sorted(pivot.values(), key=lambda r: r["bucket"]),
        "source": "agg_accuracy_by_dim",
    }


@app.get("/forecast/accuracy/lag-curve")
def forecast_accuracy_lag_curve(
    models: str = Query(default="", max_length=500),
    cluster_assignment: str = Query(default="", max_length=120),
    supplier_desc: str = Query(default="", max_length=120),
    abc_vol: str = Query(default="", max_length=40),
    region: str = Query(default="", max_length=120),
    month_from: str = Query(default="", max_length=20),
    month_to: str = Query(default="", max_length=20),
):
    """Return accuracy by lag (0–4) for each model.

    Uses pre-aggregated agg_accuracy_lag_archive for performance.
    Returns a list ordered by lag for easy charting.
    """
    model_list = [m.strip() for m in models.split(",") if m.strip()] if models.strip() else []

    where_parts: list[str] = []
    params: list[Any] = []

    if model_list:
        placeholders = ",".join(["%s"] * len(model_list))
        where_parts.append(f"model_id IN ({placeholders})")
        params.extend(model_list)
    if cluster_assignment.strip():
        where_parts.append("cluster_assignment = %s")
        params.append(cluster_assignment.strip())
    if supplier_desc.strip():
        where_parts.append("supplier_desc = %s")
        params.append(supplier_desc.strip())
    if abc_vol.strip():
        where_parts.append("abc_vol = %s")
        params.append(abc_vol.strip())
    if region.strip():
        where_parts.append("region = %s")
        params.append(region.strip())
    if month_from.strip():
        where_parts.append("month_start >= %s::date")
        params.append(month_from.strip())
    if month_to.strip():
        where_parts.append("month_start <= %s::date")
        params.append(month_to.strip())

    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    sql = f"""
        SELECT
            model_id,
            lag,
            SUM(row_count)::bigint   AS n_rows,
            SUM(sum_forecast)        AS sum_forecast,
            SUM(sum_actual)          AS sum_actual,
            SUM(sum_abs_error)       AS sum_abs_error
        FROM agg_accuracy_lag_archive
        {where_sql}
        GROUP BY 1, 2
        ORDER BY 2 ASC, 1 ASC
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        db_rows = cur.fetchall()

    # Group by lag: {lag: {model_id: kpis}}
    by_lag: dict[int, dict[str, Any]] = {}
    for model_id, lag_val, n_rows, sf, sa, sae in db_rows:
        lag_key = int(lag_val)
        if lag_key not in by_lag:
            by_lag[lag_key] = {"lag": lag_key, "by_model": {}}
        by_lag[lag_key]["by_model"][model_id] = _compute_kpis(
            float(sf or 0), float(sa or 0), float(sae or 0)
        )

    return {
        "models": model_list or None,
        "by_lag": [by_lag[k] for k in sorted(by_lag.keys())],
        "source": "agg_accuracy_lag_archive",
    }


# ---------------------------------------------------------------------------
# Champion / Model Competition endpoints (feature15)
# ---------------------------------------------------------------------------

_COMPETITION_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "model_competition.yaml",
)

_CHAMPION_SUMMARY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "champion", "champion_summary.json",
)


class CompetitionConfigUpdate(BaseModel):
    metric: str = "wape"
    lag: str = "execution"
    min_dfu_rows: int = 3
    champion_model_id: str = "champion"
    models: list[str]


@app.get("/competition/config")
def get_competition_config():
    """Return current model competition config + available models in DB."""
    import yaml

    if not os.path.exists(_COMPETITION_CONFIG_PATH):
        raise HTTPException(404, "Competition config not found")
    with open(_COMPETITION_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("competition", {})

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT model_id FROM fact_external_forecast_monthly ORDER BY 1"
        )
        available = [r[0] for r in cur.fetchall() if r[0]]

    return {"config": cfg, "available_models": available}


@app.put("/competition/config")
def update_competition_config(body: CompetitionConfigUpdate):
    """Update model competition config (writes YAML to disk)."""
    import yaml

    if body.metric not in ("wape", "accuracy_pct"):
        raise HTTPException(422, "metric must be 'wape' or 'accuracy_pct'")
    valid_lags = {"execution", "0", "1", "2", "3", "4"}
    if body.lag not in valid_lags:
        raise HTTPException(422, f"lag must be one of: {sorted(valid_lags)}")
    if len(body.models) < 2:
        raise HTTPException(422, "At least 2 models required for competition")

    cfg = {
        "competition": {
            "name": "default",
            "metric": body.metric,
            "lag": body.lag,
            "min_dfu_rows": body.min_dfu_rows,
            "champion_model_id": body.champion_model_id,
            "models": body.models,
        }
    }
    os.makedirs(os.path.dirname(_COMPETITION_CONFIG_PATH), exist_ok=True)
    with open(_COMPETITION_CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return {"status": "ok", "config": cfg["competition"]}


@app.post("/competition/run")
def run_competition():
    """Execute champion model selection and return summary."""
    import io
    import yaml
    from datetime import datetime, timezone

    # 1. Load config
    if not os.path.exists(_COMPETITION_CONFIG_PATH):
        raise HTTPException(404, "Competition config not found")
    with open(_COMPETITION_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("competition", {})

    models = cfg.get("models", [])
    metric = cfg.get("metric", "wape")
    lag_mode = str(cfg.get("lag", "execution"))
    min_rows = int(cfg.get("min_dfu_rows", 3))
    champion_id = cfg.get("champion_model_id", "champion")

    if len(models) < 2:
        raise HTTPException(422, "At least 2 models required for competition")

    placeholders = ",".join(["%s"] * len(models))
    params: list[Any] = list(models)

    if lag_mode == "execution":
        lag_cond = "lag::text = execution_lag::text"
    else:
        lag_cond = "lag = %s"
        params.append(int(lag_mode))

    # 2. Compute DFU-level winners
    winner_sql = f"""
    WITH dfu_model_wape AS (
        SELECT
            dmdunit, dmdgroup, loc, model_id,
            SUM(ABS(basefcst_pref - tothist_dmd))
                / NULLIF(ABS(SUM(tothist_dmd)), 0) AS wape,
            COUNT(*) AS n_rows
        FROM fact_external_forecast_monthly
        WHERE model_id IN ({placeholders})
          AND {lag_cond}
          AND basefcst_pref IS NOT NULL
          AND tothist_dmd IS NOT NULL
        GROUP BY dmdunit, dmdgroup, loc, model_id
        HAVING COUNT(*) >= %s
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY dmdunit, dmdgroup, loc
                ORDER BY wape ASC NULLS LAST
            ) AS rn
        FROM dfu_model_wape
        WHERE wape IS NOT NULL
    )
    SELECT dmdunit, dmdgroup, loc, model_id, wape, n_rows
    FROM ranked
    WHERE rn = 1
    ORDER BY dmdunit, dmdgroup, loc
    """
    params.append(min_rows)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(winner_sql, params)
        winners = cur.fetchall()

    if not winners:
        raise HTTPException(404, "No qualifying DFUs found with current config")

    # 3. Bulk insert champion rows
    with get_conn() as conn, conn.cursor() as cur:
        # Delete old champion rows
        cur.execute(
            "DELETE FROM fact_external_forecast_monthly WHERE model_id = %s",
            (champion_id,),
        )

        # Temp table with winners
        cur.execute("""
            CREATE TEMP TABLE _champion_winners (
                dmdunit TEXT NOT NULL,
                dmdgroup TEXT NOT NULL,
                loc TEXT NOT NULL,
                winning_model_id TEXT NOT NULL
            ) ON COMMIT DROP
        """)

        buf = io.StringIO()
        for dmdunit, dmdgroup, loc, model_id, _wape, _n in winners:
            buf.write(f"{dmdunit}\t{dmdgroup}\t{loc}\t{model_id}\n")
        buf.seek(0)
        with cur.copy("COPY _champion_winners FROM STDIN") as copy:
            copy.write(buf.read())

        # Bulk INSERT ... SELECT
        cur.execute(
            """
            INSERT INTO fact_external_forecast_monthly
                (forecast_ck, dmdunit, dmdgroup, loc, fcstdate, startdate,
                 lag, execution_lag, basefcst_pref, tothist_dmd, model_id)
            SELECT
                f.forecast_ck, f.dmdunit, f.dmdgroup, f.loc, f.fcstdate, f.startdate,
                f.lag, f.execution_lag, f.basefcst_pref, f.tothist_dmd,
                %s
            FROM fact_external_forecast_monthly f
            INNER JOIN _champion_winners w
                ON f.dmdunit = w.dmdunit
               AND f.dmdgroup = w.dmdgroup
               AND f.loc = w.loc
               AND f.model_id = w.winning_model_id
            """,
            (champion_id,),
        )
        inserted = cur.rowcount
        conn.commit()

    # 4. Compute ceiling (oracle) — best model per DFU per month
    ceiling_id = cfg.get("ceiling_model_id", "ceiling")
    ceil_placeholders = ",".join(["%s"] * len(models))
    ceil_params: list[Any] = list(models)
    if lag_mode == "execution":
        ceil_lag_cond = "lag::text = execution_lag::text"
    else:
        ceil_lag_cond = "lag = %s"
        ceil_params.append(int(lag_mode))

    ceiling_sql = f"""
    WITH monthly_ranked AS (
        SELECT
            dmdunit, dmdgroup, loc, startdate, model_id,
            ABS(basefcst_pref - tothist_dmd) AS abs_err,
            basefcst_pref, tothist_dmd,
            ROW_NUMBER() OVER (
                PARTITION BY dmdunit, dmdgroup, loc, startdate
                ORDER BY ABS(basefcst_pref - tothist_dmd) ASC NULLS LAST
            ) AS rn
        FROM fact_external_forecast_monthly
        WHERE model_id IN ({ceil_placeholders})
          AND {ceil_lag_cond}
          AND basefcst_pref IS NOT NULL
          AND tothist_dmd IS NOT NULL
    )
    SELECT dmdunit, dmdgroup, loc, startdate, model_id,
           abs_err, basefcst_pref, tothist_dmd
    FROM monthly_ranked
    WHERE rn = 1
    ORDER BY dmdunit, dmdgroup, loc, startdate
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(ceiling_sql, ceil_params)
        ceiling_rows = cur.fetchall()

    # 5. Bulk insert ceiling rows
    ceiling_inserted = 0
    if ceiling_rows:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM fact_external_forecast_monthly WHERE model_id = %s",
                (ceiling_id,),
            )
            cur.execute("""
                CREATE TEMP TABLE _ceiling_winners (
                    dmdunit TEXT NOT NULL,
                    dmdgroup TEXT NOT NULL,
                    loc TEXT NOT NULL,
                    startdate DATE NOT NULL,
                    winning_model_id TEXT NOT NULL
                ) ON COMMIT DROP
            """)

            buf2 = io.StringIO()
            for dmdunit, dmdgroup, loc, startdate, model_id, *_ in ceiling_rows:
                buf2.write(f"{dmdunit}\t{dmdgroup}\t{loc}\t{startdate}\t{model_id}\n")
            buf2.seek(0)
            with cur.copy("COPY _ceiling_winners FROM STDIN") as copy:
                copy.write(buf2.read())

            cur.execute(
                """
                INSERT INTO fact_external_forecast_monthly
                    (forecast_ck, dmdunit, dmdgroup, loc, fcstdate, startdate,
                     lag, execution_lag, basefcst_pref, tothist_dmd, model_id)
                SELECT
                    f.forecast_ck, f.dmdunit, f.dmdgroup, f.loc, f.fcstdate, f.startdate,
                    f.lag, f.execution_lag, f.basefcst_pref, f.tothist_dmd,
                    %s
                FROM fact_external_forecast_monthly f
                INNER JOIN _ceiling_winners w
                    ON f.dmdunit = w.dmdunit
                   AND f.dmdgroup = w.dmdgroup
                   AND f.loc = w.loc
                   AND f.startdate = w.startdate
                   AND f.model_id = w.winning_model_id
                """,
                (ceiling_id,),
            )
            ceiling_inserted = cur.rowcount
            conn.commit()

    # 6. Refresh materialized views
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '512MB'")
        cur.execute("REFRESH MATERIALIZED VIEW agg_forecast_monthly")
        cur.execute("REFRESH MATERIALIZED VIEW agg_accuracy_by_dim")
        conn.commit()

    # 7. Build summary (champion)
    model_wins: dict[str, int] = {}
    total_wape_num = 0.0
    total_wape_denom = 0.0
    for _u, _g, _l, mid, wape, n_rows in winners:
        model_wins[mid] = model_wins.get(mid, 0) + 1
        if wape is not None:
            total_wape_num += float(wape) * float(n_rows)
            total_wape_denom += float(n_rows)

    overall_wape = (total_wape_num / total_wape_denom * 100) if total_wape_denom else None
    overall_acc = (100.0 - overall_wape) if overall_wape is not None else None

    summary: dict[str, Any] = {
        "config": {
            "metric": metric,
            "lag": lag_mode,
            "min_dfu_rows": min_rows,
            "champion_model_id": champion_id,
            "models": models,
        },
        "total_dfus": len(winners),
        "total_champion_rows": inserted,
        "model_wins": dict(sorted(model_wins.items(), key=lambda x: -x[1])),
        "overall_champion_wape": round(overall_wape, 4) if overall_wape is not None else None,
        "overall_champion_accuracy_pct": round(overall_acc, 4) if overall_acc is not None else None,
        "run_ts": datetime.now(timezone.utc).isoformat(),
    }

    # 8. Ceiling metrics
    if ceiling_rows:
        ceil_wins: dict[str, int] = {}
        ceil_abs_err_sum = 0.0
        ceil_actual_sum = 0.0
        for _u, _g, _l, _sd, mid, abs_err, _fcst, actual in ceiling_rows:
            ceil_wins[mid] = ceil_wins.get(mid, 0) + 1
            ceil_abs_err_sum += float(abs_err)
            ceil_actual_sum += abs(float(actual))

        ceil_wape = (ceil_abs_err_sum / ceil_actual_sum * 100) if ceil_actual_sum else None
        ceil_acc = (100.0 - ceil_wape) if ceil_wape is not None else None

        summary["total_ceiling_rows"] = ceiling_inserted
        summary["ceiling_model_wins"] = dict(sorted(ceil_wins.items(), key=lambda x: -x[1]))
        summary["overall_ceiling_wape"] = round(ceil_wape, 4) if ceil_wape is not None else None
        summary["overall_ceiling_accuracy_pct"] = round(ceil_acc, 4) if ceil_acc is not None else None

    # Save summary to disk
    summary_dir = os.path.dirname(_CHAMPION_SUMMARY_PATH)
    os.makedirs(summary_dir, exist_ok=True)
    with open(_CHAMPION_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


@app.get("/competition/summary")
def get_competition_summary():
    """Return the last champion selection summary, if available."""
    if not os.path.exists(_CHAMPION_SUMMARY_PATH):
        return {"status": "not_run", "summary": None}
    with open(_CHAMPION_SUMMARY_PATH) as f:
        return {"status": "ok", "summary": json.load(f)}


@app.get("/domains/dfu/clusters")
def dfu_clusters(source: str = Query(default="ml", regex="^(ml|source)$")):
    """Get cluster summary statistics for DFU clustering.

    source=ml    → pipeline-generated clusters (ml_cluster column)
    source=source → original source-file clusters (cluster_assignment column)
    """
    col = "ml_cluster" if source == "ml" else "cluster_assignment"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"""
            WITH cluster_counts AS (
                SELECT
                    {col} AS cluster_label,
                    COUNT(*) AS dfu_count
                FROM dim_dfu
                WHERE {col} IS NOT NULL AND {col} != ''
                GROUP BY {col}
            ),
            total AS (
                SELECT SUM(dfu_count) AS total_assigned FROM cluster_counts
            ),
            cluster_demand AS (
                SELECT
                    d.{col} AS cluster_label,
                    AVG(s.qty) AS avg_demand,
                    CASE
                        WHEN AVG(s.qty) > 0 THEN COALESCE(STDDEV(s.qty), 0) / AVG(s.qty)
                        ELSE 0
                    END AS cv_demand
                FROM dim_dfu d
                INNER JOIN fact_sales_monthly s
                    ON s.dmdunit = d.dmdunit
                    AND s.dmdgroup = d.dmdgroup
                    AND s.loc = d.loc
                WHERE d.{col} IS NOT NULL AND d.{col} != ''
                    AND s.qty IS NOT NULL
                GROUP BY d.{col}
            )
            SELECT
                cc.cluster_label,
                cc.dfu_count,
                ROUND(cc.dfu_count * 100.0 / t.total_assigned, 2) AS pct_of_total,
                COALESCE(cd.avg_demand, 0) AS avg_demand,
                COALESCE(cd.cv_demand, 0) AS cv_demand
            FROM cluster_counts cc
            CROSS JOIN total t
            LEFT JOIN cluster_demand cd ON cd.cluster_label = cc.cluster_label
            ORDER BY cc.dfu_count DESC
        """)
        rows = cur.fetchall()

        clusters = []
        total_assigned = 0
        for cluster_label, count, pct, avg_demand, cv_demand in rows:
            total_assigned += int(count)
            clusters.append({
                "cluster_id": cluster_label,
                "label": cluster_label,
                "count": int(count),
                "pct_of_total": float(pct),
                "avg_demand": round(float(avg_demand), 2),
                "cv_demand": round(float(cv_demand), 4),
            })

        return {
            "domain": "dfu",
            "source": source,
            "total_assigned": total_assigned,
            "clusters": clusters,
        }


@app.get("/domains/dfu/clusters/profiles")
def dfu_cluster_profiles():
    """Get cluster profiles with centroid features and metadata."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    profiles_path = root / "data" / "clustering" / "cluster_profiles.json"
    metadata_path = root / "data" / "clustering" / "cluster_metadata.json"

    profiles = []
    if profiles_path.exists():
        with open(profiles_path) as f:
            profiles = json.load(f)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return {
        "profiles": profiles,
        "metadata": {
            "optimal_k": metadata.get("optimal_k"),
            "silhouette_score": metadata.get("silhouette_score"),
            "inertia": metadata.get("inertia"),
            "k_selection_results": metadata.get("k_selection_results"),
        },
    }


@app.get("/domains/dfu/clusters/visualization/{image_name}")
def dfu_cluster_visualization(image_name: str):
    """Serve clustering visualization images."""
    from pathlib import Path
    allowed = {"k_selection_plots.png", "cluster_visualization.png"}
    if image_name not in allowed:
        raise HTTPException(404, f"Image not found: {image_name}")
    root = Path(__file__).resolve().parents[1]
    img_path = root / "data" / "clustering" / image_name
    if not img_path.exists():
        raise HTTPException(404, f"Image not generated yet: {image_name}")
    return FileResponse(str(img_path), media_type="image/png")


@app.get("/domains/{domain}/analytics")
def domain_analytics(
    domain: str,
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    metric: str = Query(default=""),
    metrics: str = Query(default="", max_length=1500),
    date_field: str = Query(default=""),
    category_field: str = Query(default=""),
    points: int = Query(default=24, ge=3, le=120),
    top_n: int = Query(default=12, ge=3, le=50),
    kpi_months: int = Query(default=12, ge=1, le=12),
    model: str = Query(default="", max_length=120),
):
    spec = get_spec(domain)

    # Inject model_id filter for forecast domain
    if spec.name == "forecast" and model.strip():
        merged = parse_filters_safe(filters)
        merged["model_id"] = f"={model.strip()}"
        filters = json.dumps(merged)
    if spec.name == "time":
        raise HTTPException(status_code=404, detail="Analytics is disabled for domain: time")
    numeric = numeric_fields_for_spec(spec)
    dates = date_fields_for_spec(spec)
    categories = category_fields_for_spec(spec)
    virtual_metrics = {"accuracy_pct"} if spec.name == "forecast" else set()

    metric_sql = to_sql_col(spec, metric)
    date_sql = to_sql_col(spec, date_field)
    category_sql = to_sql_col(spec, category_field)
    metrics_sql = [
        to_sql_col(spec, m.strip())
        for m in metrics.split(",")
        if m and m.strip()
    ]

    if metric_sql and metric_sql != "__count__" and metric_sql not in numeric and metric_sql not in virtual_metrics:
        raise HTTPException(status_code=422, detail=f"Invalid metric field: {metric}")
    for m_sql in metrics_sql:
        if m_sql and m_sql != "__count__" and m_sql not in numeric and m_sql not in virtual_metrics:
            raise HTTPException(status_code=422, detail=f"Invalid metric field: {m_sql}")
    if date_sql and date_sql not in dates:
        raise HTTPException(status_code=422, detail=f"Invalid date field: {date_field}")
    if category_sql and category_sql not in categories:
        raise HTTPException(status_code=422, detail=f"Invalid category field: {category_field}")

    chosen_metric = metric_sql or (numeric[0] if numeric else "__count__")
    trend_metrics = [m for m in metrics_sql if m] or [chosen_metric]
    if chosen_metric not in trend_metrics:
        trend_metrics = [chosen_metric, *trend_metrics]
    chosen_date = date_sql or (dates[0] if dates else "")
    chosen_category = category_sql or (categories[0] if categories else "")
    summary_metric = chosen_metric if chosen_metric not in virtual_metrics else (numeric[0] if numeric else "__count__")

    where_sql, params = build_where(spec, q, filters)
    metric_expr = grouped_metric_expr(summary_metric)

    summary_sql = (
        "SELECT count(*)::bigint, "
        + metric_expr
        + ", "
        + ("avg(" + qident(summary_metric) + ")::double precision" if summary_metric != "__count__" else "NULL::double precision")
        + ", "
        + ("min(" + qident(summary_metric) + ")::double precision" if summary_metric != "__count__" else "NULL::double precision")
        + ", "
        + ("max(" + qident(summary_metric) + ")::double precision" if summary_metric != "__count__" else "NULL::double precision")
        + f" FROM {qident(spec.table)} {where_sql}"
    )

    category_sql_stmt = ""
    if chosen_category:
        category_sql_stmt = f"""
          SELECT {qident(chosen_category)}::text AS bucket, {metric_expr} AS value
          FROM {qident(spec.table)}
          {where_sql}
          GROUP BY 1
          ORDER BY 2 DESC NULLS LAST, 1 ASC
          LIMIT %s
        """

    min_date = None
    max_date = None
    trend_points: list[dict[str, Any]] = []
    trend_multi: dict[str, list[dict[str, Any]]] = {}
    category_points: list[dict[str, Any]] = []
    kpis: dict[str, Any] = {}

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(summary_sql, params)
        total_rows, metric_total, metric_avg, metric_min, metric_max = cur.fetchone()

        if spec.name == "forecast":
            forecast_kpi_sql = f"""
              WITH monthly AS (
                SELECT
                  date_trunc('month', startdate)::date AS month_start,
                  coalesce(sum(basefcst_pref), 0)::double precision AS forecast_sum,
                  coalesce(sum(tothist_dmd), 0)::double precision AS actual_sum,
                  coalesce(sum(abs(basefcst_pref - tothist_dmd)), 0)::double precision AS abs_error_sum,
                  avg(
                    CASE
                      WHEN tothist_dmd IS NOT NULL AND tothist_dmd <> 0
                        THEN abs(basefcst_pref - tothist_dmd) / abs(tothist_dmd)
                      ELSE NULL
                    END
                  )::double precision AS mape_ratio
                FROM {qident(spec.table)}
                {where_sql}
                GROUP BY 1
              ),
              windowed AS (
                SELECT *
                FROM monthly
                ORDER BY month_start DESC
                LIMIT %s
              )
              SELECT
                count(*)::int AS months_covered,
                coalesce(sum(forecast_sum), 0)::double precision AS total_forecast,
                coalesce(sum(actual_sum), 0)::double precision AS total_actual,
                coalesce(sum(abs_error_sum), 0)::double precision AS abs_error_sum,
                (CASE WHEN abs(sum(actual_sum)) > 0 THEN (sum(forecast_sum) / sum(actual_sum)) - 1 ELSE NULL END)::double precision AS bias,
                avg(CASE WHEN abs(actual_sum) > 0 THEN (abs_error_sum / abs(actual_sum)) * 100.0 ELSE NULL END)::double precision AS avg_wape_pct,
                avg(CASE WHEN mape_ratio IS NOT NULL THEN mape_ratio * 100.0 ELSE NULL END)::double precision AS avg_mape_pct,
                avg(CASE WHEN abs(actual_sum) > 0 THEN 100.0 - ((abs_error_sum / abs(actual_sum)) * 100.0) ELSE NULL END)::double precision AS avg_accuracy_pct
              FROM windowed
            """
            cur.execute(forecast_kpi_sql, [*params, kpi_months])
            (
                months_covered,
                total_fcst,
                total_actual,
                abs_error_sum,
                bias,
                avg_wape_pct,
                avg_mape_pct,
                avg_accuracy_pct,
            ) = cur.fetchone()
            kpis = {
                "months_window": int(kpi_months),
                "months_covered": int(months_covered or 0),
                "total_forecast": float(total_fcst or 0),
                "total_actual": float(total_actual or 0),
                "abs_error": float(abs_error_sum or 0),
                "bias": float(bias) if bias is not None else None,
                "wape_pct": float(avg_wape_pct) if avg_wape_pct is not None else None,
                "mape_pct": float(avg_mape_pct) if avg_mape_pct is not None else None,
                "accuracy_pct": float(avg_accuracy_pct) if avg_accuracy_pct is not None else None,
            }

        if chosen_date:
            cur.execute(
                f"SELECT min({qident(chosen_date)}), max({qident(chosen_date)}) FROM {qident(spec.table)} {where_sql}",
                params,
            )
            min_date, max_date = cur.fetchone()

            trend_table = qident(spec.table)
            trend_date_col = chosen_date
            trend_where_sql = where_sql
            trend_params = params
            trend_count_col: str | None = None

            # Use pre-aggregated monthly tables for facts when the request shape is compatible.
            if not q.strip() and chosen_date == "startdate":
                agg_source = build_agg_trend_source(spec, trend_metrics, filters)
                if agg_source:
                    agg_table, agg_date_col, agg_where_sql, agg_params, agg_count_col = agg_source
                    cur.execute("SELECT to_regclass(%s)", [agg_table])
                    if cur.fetchone()[0]:
                        trend_table = qident(agg_table)
                        trend_date_col = agg_date_col
                        trend_where_sql = agg_where_sql
                        trend_params = agg_params
                        trend_count_col = agg_count_col

            for m in trend_metrics:
                value_expr = grouped_metric_expr_with_count(m, trend_count_col)
                if spec.name == "forecast" and m == "accuracy_pct":
                    value_expr = forecast_accuracy_expr()
                trend_sql = f"""
                  SELECT date_trunc('month', {qident(trend_date_col)})::date AS bucket, {value_expr} AS value
                  FROM {trend_table}
                  {trend_where_sql}
                  GROUP BY 1
                  ORDER BY 1 DESC
                  LIMIT %s
                """
                cur.execute(trend_sql, [*trend_params, points])
                trend_rows = list(reversed(cur.fetchall()))
                series = [{"x": str(x), "y": float(y or 0)} for x, y in trend_rows]
                trend_multi[to_api_col(m)] = series

            trend_points = trend_multi.get(to_api_col(chosen_metric), [])

        if chosen_category:
            cur.execute(category_sql_stmt, [*params, top_n])
            category_points = [{"x": str(x or "(blank)"), "y": float(y or 0)} for x, y in cur.fetchall()]

    return {
        "domain": spec.name,
        "config": {
            "metric": to_api_col(chosen_metric),
            "trend_metrics": [to_api_col(m) for m in trend_metrics],
            "date_field": to_api_col(chosen_date) if chosen_date else "",
            "category_field": to_api_col(chosen_category) if chosen_category else "",
            "points": points,
            "top_n": top_n,
            "kpi_months": kpi_months,
        },
        "available": {
            "metrics": ["__count__", *[to_api_col(c) for c in numeric], *[to_api_col(c) for c in virtual_metrics]],
            "date_fields": [to_api_col(c) for c in dates],
            "category_fields": [to_api_col(c) for c in categories],
        },
        "summary": {
            "total_rows": int(total_rows or 0),
            "metric_total": float(metric_total or 0),
            "metric_avg": float(metric_avg or 0) if metric_avg is not None else None,
            "metric_min": float(metric_min or 0) if metric_min is not None else None,
            "metric_max": float(metric_max or 0) if metric_max is not None else None,
            "min_date": str(min_date) if min_date else None,
            "max_date": str(max_date) if max_date else None,
        },
        "trend": trend_points,
        "trend_multi": trend_multi,
        "top_categories": category_points,
        "kpis": kpis,
    }


# Backward-compatible aliases
@app.get("/items")
def list_items(limit: int = Query(default=50, ge=1, le=1000)):
    return list_domain(get_spec("item"), limit)


@app.get("/locations")
def list_locations(limit: int = Query(default=50, ge=1, le=1000)):
    return list_domain(get_spec("location"), limit)


@app.get("/customers")
def list_customers(limit: int = Query(default=50, ge=1, le=1000)):
    return list_domain(get_spec("customer"), limit)


@app.get("/times")
def list_times(limit: int = Query(default=50, ge=1, le=1000)):
    return list_domain(get_spec("time"), limit)


@app.get("/dfus")
def list_dfus(limit: int = Query(default=50, ge=1, le=1000)):
    return list_domain(get_spec("dfu"), limit)


@app.get("/sales")
def list_sales(limit: int = Query(default=50, ge=1, le=1000)):
    return list_domain(get_spec("sales"), limit)


@app.get("/forecasts")
def list_forecasts(limit: int = Query(default=50, ge=1, le=1000)):
    return list_domain(get_spec("forecast"), limit)


@app.get("/items/page")
def list_items_page(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default="item_no"),
    sort_dir: str = Query(default="asc"),
):
    return fetch_page(get_spec("item"), limit, offset, q, filters, sort_by, sort_dir)


@app.get("/locations/page")
def list_locations_page(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default="location_id"),
    sort_dir: str = Query(default="asc"),
):
    return fetch_page(get_spec("location"), limit, offset, q, filters, sort_by, sort_dir)


@app.get("/customers/page")
def list_customers_page(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default="customer_ck"),
    sort_dir: str = Query(default="asc"),
):
    return fetch_page(get_spec("customer"), limit, offset, q, filters, sort_by, sort_dir)


@app.get("/times/page")
def list_times_page(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default="date_key"),
    sort_dir: str = Query(default="asc"),
):
    return fetch_page(get_spec("time"), limit, offset, q, filters, sort_by, sort_dir)


@app.get("/dfus/page")
def list_dfus_page(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default="dmdunit"),
    sort_dir: str = Query(default="asc"),
):
    return fetch_page(get_spec("dfu"), limit, offset, q, filters, sort_by, sort_dir)


@app.get("/sales/page")
def list_sales_page(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default="startdate"),
    sort_dir: str = Query(default="asc"),
):
    return fetch_page(get_spec("sales"), limit, offset, q, filters, sort_by, sort_dir)


@app.get("/forecasts/page")
def list_forecasts_page(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default="fcstdate"),
    sort_dir: str = Query(default="asc"),
):
    return fetch_page(get_spec("forecast"), limit, offset, q, filters, sort_by, sort_dir)


# ---------------------------------------------------------------------------
# Chat / Natural Language Queries (Feature 12)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    domain: str = ""


_openai_client = None


def _get_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-..."):
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _build_schema_summary() -> str:
    """Compact schema summary for the system prompt."""
    lines: list[str] = []
    for spec in DOMAIN_SPECS.values():
        cols = []
        for c in spec.columns:
            if c in spec.int_fields:
                cols.append(f"{c} (int)")
            elif c in spec.float_fields:
                cols.append(f"{c} (numeric)")
            elif c in spec.date_fields:
                cols.append(f"{c} (date)")
            else:
                cols.append(f"{c} (text)")
        lines.append(f"Table: {spec.table}  PK: {spec.ck_field}  Key: {', '.join(spec.key_fields)}")
        lines.append(f"  Columns: {', '.join(cols)}")
    return "\n".join(lines)


def _vector_search(question_embedding: list[float], top_k: int = 10) -> list[str]:
    """Retrieve most relevant schema context via pgvector cosine similarity."""
    sql = """
        SELECT source_text
        FROM chat_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (str(question_embedding), top_k))
            return [r[0] for r in cur.fetchall()]
    except Exception:
        return []


CHAT_SYSTEM_PROMPT = """You are a SQL expert for a demand forecasting PostgreSQL database called Demand Studio.

## Schema
{schema}

## Retrieved Context
{context}

## Business Rules
- Forecast accuracy formula: 100 - (100 * SUM(ABS(basefcst_pref - tothist_dmd)) / NULLIF(ABS(SUM(tothist_dmd)), 0))
- Bias formula: (SUM(basefcst_pref) / NULLIF(SUM(tothist_dmd), 0)) - 1
- WAPE formula: 100 * SUM(ABS(basefcst_pref - tothist_dmd)) / NULLIF(ABS(SUM(tothist_dmd)), 0)
- Only sales rows with type=1 exist in fact_sales_monthly
- All startdate/fcstdate values are month-start dates (first day of month)
- Forecast lag is 0-4 months (startdate - fcstdate in months)
- model_id on forecasts defaults to 'external' for source-system forecasts

## Instructions
1. Answer the user's question about the demand data.
2. If you need to query data, generate a PostgreSQL SELECT statement.
3. ONLY generate SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or any DDL/DML.
4. Always include LIMIT 500 at the end of your queries.
5. Use proper column names exactly as listed in the schema.
6. For date filtering, use ISO format: '2024-01-01'.

Respond in JSON format:
{{"answer": "your natural language answer", "sql": "SELECT ... LIMIT 500"}}

If no SQL is needed (e.g., the question is about schema or definitions), set sql to null:
{{"answer": "your explanation", "sql": null}}"""


def _is_safe_sql(sql: str) -> bool:
    """Check that SQL is a SELECT statement only."""
    cleaned = re.sub(r'--[^\n]*', '', sql)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip().upper()
    if not cleaned.startswith("SELECT"):
        return False
    forbidden = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE", "COPY"}
    tokens = set(re.findall(r'\b[A-Z]+\b', cleaned))
    if tokens & forbidden:
        return False
    return True


def _execute_readonly_sql(sql: str) -> tuple[list[str], list[list[Any]]]:
    """Execute SQL in a read-only transaction with timeout. Returns (columns, rows)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL statement_timeout = '5000'")
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute(sql)
        if cur.description is None:
            return [], []
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchmany(500)
        return columns, [list(r) for r in rows]


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    client = _get_openai()

    # 1. Embed the question
    embed_resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
    )
    question_embedding = embed_resp.data[0].embedding

    # 2. Vector search for relevant context
    context_chunks = _vector_search(question_embedding, top_k=10)
    context_text = "\n".join(f"- {chunk}" for chunk in context_chunks) if context_chunks else "(no embeddings available)"

    # 3. Build prompt
    schema_summary = _build_schema_summary()
    system_prompt = CHAT_SYSTEM_PROMPT.format(schema=schema_summary, context=context_text)

    user_msg = question
    if req.domain.strip():
        user_msg = f"[Current domain context: {req.domain.strip()}] {question}"

    # 4. Call GPT-4o
    chat_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    raw_content = chat_resp.choices[0].message.content or "{}"

    # 5. Parse response
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        parsed = {"answer": raw_content, "sql": None}

    answer = parsed.get("answer", "I couldn't generate an answer.")
    sql = parsed.get("sql")
    data = None
    columns: list[str] = []
    row_count = None
    error_msg = None

    # 6. Execute SQL if present and safe
    if sql and isinstance(sql, str) and sql.strip():
        sql = sql.strip().rstrip(";")
        if not _is_safe_sql(sql):
            error_msg = "Generated SQL was blocked for safety reasons (only SELECT allowed)."
            sql = None
        else:
            try:
                columns, raw_rows = _execute_readonly_sql(sql)
                row_count = len(raw_rows)
                data = [
                    {col: (str(val) if val is not None else None) for col, val in zip(columns, row)}
                    for row in raw_rows
                ]
            except Exception as exc:
                error_msg = f"SQL execution error: {exc}"

    result: dict[str, Any] = {
        "answer": answer,
        "sql": sql,
        "data": data,
        "columns": columns,
        "row_count": row_count,
    }
    if error_msg:
        result["error"] = error_msg
    return result
