"""Postgres vs Trino/Iceberg benchmarking endpoint (feature 26)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api.core import (
    get_spec_or_404,
    qident,
    dotted_qident,
    to_api_col,
    item_field_for_spec,
    location_field_for_spec,
    default_date_field_for_spec,
    default_trend_metric_for_spec,
    quote_literal,
    parse_optional_iso_date,
    summary_stats_ms,
    timed_postgres_query,
    timed_trino_query,
)

router = APIRouter()


@router.get("/bench/compare")
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
    spec = get_spec_or_404(domain)
    item_col = item_field_for_spec(spec)
    location_col = location_field_for_spec(spec)
    date_col = default_date_field_for_spec(spec)
    trend_metric = default_trend_metric_for_spec(spec)

    # NOTE: quote_literal is used here because the same SQL must run against
    # both Postgres (psycopg) and Trino (docker exec), and the Trino path
    # does not support parameterized queries.
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
