"""Domain CRUD, analytics, suggest, sample-pair, and backward-compatible alias routes."""
from __future__ import annotations

from typing import Any
import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse

from common.core.domain_specs import DOMAIN_SPECS
from api.core import (
    get_conn,
    get_spec_or_404,
    set_cache,
    to_api_col,
    to_sql_col,
    qident,
    numeric_fields_for_spec,
    date_fields_for_spec,
    category_fields_for_spec,
    item_field_for_spec,
    location_field_for_spec,
    build_where,
    parse_filters_json,
    parse_filters_safe,
    _col_type,
    _typed_eq_clause,
    _typed_like_clause,
    grouped_metric_expr,
    grouped_metric_expr_with_count,
    forecast_accuracy_expr,
    build_agg_trend_source,
    fetch_page,
    list_domain,
)

router = APIRouter(tags=["domains"])


# ---------------------------------------------------------------------------
# Root / health / list domains
# ---------------------------------------------------------------------------
@router.get("/", include_in_schema=False)
def root():
    return {
        "status": "ok",
        "ui_url": "http://127.0.0.1:5173",
        "api_docs": "/docs",
    }


@router.get("/health")
def health(response: FastAPIResponse):
    set_cache(response, max_age=60)
    return {
        "status": "ok",
        "domains": sorted(DOMAIN_SPECS.keys()),
    }


@router.get("/domains")
def list_domains(response: FastAPIResponse):
    set_cache(response, max_age=3600)
    return {
        "domains": sorted(DOMAIN_SPECS.keys()),
    }


# ---------------------------------------------------------------------------
# Domain meta / suggest / sample-pair / CRUD
# ---------------------------------------------------------------------------
@router.get("/domains/{domain}/meta")
def domain_meta(domain: str, response: FastAPIResponse):
    set_cache(response, max_age=600)
    spec = get_spec_or_404(domain)
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


@router.get("/domains/{domain}/suggest")
def domain_suggest(
    domain: str,
    response: FastAPIResponse,
    field: str = Query(..., min_length=1),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    limit: int = Query(default=12, ge=1, le=100),
):
    set_cache(response, max_age=300)
    spec = get_spec_or_404(domain)
    target_col = to_sql_col(spec, field)
    allowed = set(spec.columns_with_ck)
    if target_col not in allowed:
        raise HTTPException(status_code=422, detail=f"Invalid suggest field: {field}")

    where: list[str] = []
    params: list[Any] = []
    if q.strip():
        if _col_type(spec, target_col) == "text":
            where.append(f"{qident(target_col)} ILIKE %s")
        else:
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
            _typed_eq_clause(spec, col, exact, where, params)
        else:
            _typed_like_clause(spec, col, sval, where, params)

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


@router.get("/domains/{domain}/sample-pair")
def domain_sample_pair(domain: str):
    spec = get_spec_or_404(domain)
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


@router.get("/domains/{domain}")
def list_domain_records(domain: str, limit: int = Query(default=50, ge=1, le=1000)):
    spec = get_spec_or_404(domain)
    return list_domain(spec, limit)


@router.get("/domains/{domain}/page")
def list_domain_records_page(
    domain: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default=""),
    sort_dir: str = Query(default="asc"),
):
    spec = get_spec_or_404(domain)
    return fetch_page(
        spec=spec,
        limit=limit,
        offset=offset,
        q=q,
        filters=filters,
        sort_by=sort_by or to_api_col(spec.default_sort),
        sort_dir=sort_dir,
    )


# ---------------------------------------------------------------------------
# Forecast models
# ---------------------------------------------------------------------------
@router.get("/domains/forecast/models")
def forecast_models():
    """Return distinct model_id values from both the forecast table and archive."""
    spec = get_spec_or_404("forecast")
    sql = f"""
        SELECT DISTINCT model_id FROM {qident(spec.table)} WHERE model_id IS NOT NULL
        UNION
        SELECT DISTINCT model_id FROM backtest_lag_archive WHERE model_id IS NOT NULL
        ORDER BY 1
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        models = [r[0] for r in cur.fetchall()]
    return {"domain": "forecast", "models": models}


# ---------------------------------------------------------------------------
# Domain analytics
# ---------------------------------------------------------------------------
@router.get("/domains/{domain}/analytics")
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
    spec = get_spec_or_404(domain)

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


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------
@router.get("/items")
def list_items(limit: int = Query(default=50, ge=1, le=1000)):
    return list_domain(get_spec_or_404("item"), limit)


@router.get("/locations")
def list_locations(limit: int = Query(default=50, ge=1, le=1000)):
    return list_domain(get_spec_or_404("location"), limit)


@router.get("/customers")
def list_customers(limit: int = Query(default=50, ge=1, le=1000)):
    return list_domain(get_spec_or_404("customer"), limit)


@router.get("/times")
def list_times(limit: int = Query(default=50, ge=1, le=1000)):
    return list_domain(get_spec_or_404("time"), limit)


@router.get("/dfus")
def list_dfus(limit: int = Query(default=50, ge=1, le=1000)):
    return list_domain(get_spec_or_404("sku"), limit)


@router.get("/sales")
def list_sales(limit: int = Query(default=50, ge=1, le=1000)):
    return list_domain(get_spec_or_404("sales"), limit)


@router.get("/forecasts")
def list_forecasts(limit: int = Query(default=50, ge=1, le=1000)):
    return list_domain(get_spec_or_404("forecast"), limit)


@router.get("/items/page")
def list_items_page(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default="item_id"),
    sort_dir: str = Query(default="asc"),
):
    return fetch_page(get_spec_or_404("item"), limit, offset, q, filters, sort_by, sort_dir)


@router.get("/locations/page")
def list_locations_page(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default="location_id"),
    sort_dir: str = Query(default="asc"),
):
    return fetch_page(get_spec_or_404("location"), limit, offset, q, filters, sort_by, sort_dir)


@router.get("/customers/page")
def list_customers_page(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default="customer_ck"),
    sort_dir: str = Query(default="asc"),
):
    return fetch_page(get_spec_or_404("customer"), limit, offset, q, filters, sort_by, sort_dir)


@router.get("/times/page")
def list_times_page(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default="date_key"),
    sort_dir: str = Query(default="asc"),
):
    return fetch_page(get_spec_or_404("time"), limit, offset, q, filters, sort_by, sort_dir)


@router.get("/dfus/page")
def list_dfus_page(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default="item_id"),
    sort_dir: str = Query(default="asc"),
):
    return fetch_page(get_spec_or_404("sku"), limit, offset, q, filters, sort_by, sort_dir)


@router.get("/sales/page")
def list_sales_page(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default="startdate"),
    sort_dir: str = Query(default="asc"),
):
    return fetch_page(get_spec_or_404("sales"), limit, offset, q, filters, sort_by, sort_dir)


@router.get("/forecasts/page")
def list_forecasts_page(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default="", max_length=120),
    filters: str = Query(default="", max_length=4000),
    sort_by: str = Query(default="fcstdate"),
    sort_dir: str = Query(default="asc"),
):
    return fetch_page(get_spec_or_404("forecast"), limit, offset, q, filters, sort_by, sort_dir)


# ---------------------------------------------------------------------------
# DFU count with optional global filters (used by GlobalFilterBar badge)
# ---------------------------------------------------------------------------

@router.get("/domains/sku/count")
def dfu_count(
    response: FastAPIResponse,
    brand: str = Query(default="", max_length=500),
    category: str = Query(default="", max_length=500),
    item: str = Query(default="", max_length=500),
    location: str = Query(default="", max_length=500),
    market: str = Query(default="", max_length=500),
    channel: str = Query(default="", max_length=500),
    cluster: str = Query(default="", max_length=500),
) -> dict:
    """Count distinct DFUs matching the active global filter combination."""
    set_cache(response, max_age=60)

    conditions: list[str] = []
    params: list = []

    if brand:
        params.append(brand.split(","))
        conditions.append("EXISTS (SELECT 1 FROM dim_item di WHERE di.item_id = d.item_id AND di.brand_name = ANY(%s))")
    if category:
        params.append(category.split(","))
        conditions.append("EXISTS (SELECT 1 FROM dim_item di WHERE di.item_id = d.item_id AND di.class = ANY(%s))")
    if item:
        params.append(item.split(","))
        conditions.append("d.item_id = ANY(%s)")
    if location:
        params.append(location.split(","))
        conditions.append("d.loc = ANY(%s)")
    if market:
        params.append(market.split(","))
        conditions.append("EXISTS (SELECT 1 FROM dim_location dl WHERE dl.location_id = d.loc AND dl.state_id = ANY(%s))")
    if channel:
        params.append(channel.split(","))
        conditions.append(
            "EXISTS (SELECT 1 FROM dim_customer dc "
            "JOIN fact_sales_monthly fsm ON fsm.cust_grp = dc.customer_group "
            "WHERE fsm.item_id = d.item_id AND fsm.loc = d.loc AND dc.rpt_channel_desc = ANY(%s))"
        )
    if cluster:
        params.append(cluster.split(","))
        conditions.append("d.cluster_assignment = ANY(%s)")

    where_sql = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"SELECT COUNT(*) FROM dim_sku d {where_sql}"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()

    return {"count": int(row[0]) if row else 0}


# ---------------------------------------------------------------------------
# Distinct values (used by global filter dropdowns)
# ---------------------------------------------------------------------------

_DISTINCT_ALLOWED: dict[str, list[str]] = {
    "item": ["brand_name", "class", "item_desc", "item_id"],
    "location": ["state_id", "site_desc", "location_id"],
    "customer": ["rpt_channel_desc"],
    "sku": ["cluster_assignment"],
}


# Mapping: (domain, column) → SQL expression to extract value through dim_sku joins
_CASCADING_EXPR: dict[tuple[str, str], tuple[str, str]] = {
    # (domain, column) → (join_clause, select_expr)
    ("item", "brand_name"): (
        "JOIN dim_item di ON di.item_id = d.item_id",
        "di.brand_name",
    ),
    ("item", "class"): (
        "JOIN dim_item di ON di.item_id = d.item_id",
        "di.class",
    ),
    ("item", "item_id"): (
        "",  # item_id lives on dim_sku
        "d.item_id",
    ),
    ("location", "location_id"): (
        "",  # loc lives on dim_sku
        "d.loc",
    ),
    ("location", "state_id"): (
        "JOIN dim_location dl ON dl.location_id = d.loc",
        "dl.state_id",
    ),
    ("customer", "rpt_channel_desc"): (
        "JOIN fact_sales_monthly fsm ON fsm.item_id = d.item_id AND fsm.loc = d.loc "
        "JOIN dim_customer dc ON dc.customer_group = fsm.cust_grp",
        "dc.rpt_channel_desc",
    ),
}


def _build_cascade_conditions(
    params: list[Any],
    *,
    brand: str = "",
    category: str = "",
    item: str = "",
    location: str = "",
    market: str = "",
    channel: str = "",
    cluster: str = "",
) -> list[str]:
    """Build WHERE conditions for cascading filter narrowing via dim_sku d."""
    conds: list[str] = []
    if brand:
        params.append(brand.split(","))
        conds.append("EXISTS (SELECT 1 FROM dim_item _di WHERE _di.item_id = d.item_id AND _di.brand_name = ANY(%s))")
    if category:
        params.append(category.split(","))
        conds.append("EXISTS (SELECT 1 FROM dim_item _di WHERE _di.item_id = d.item_id AND _di.class = ANY(%s))")
    if item:
        params.append(item.split(","))
        conds.append("d.item_id = ANY(%s)")
    if location:
        params.append(location.split(","))
        conds.append("d.loc = ANY(%s)")
    if market:
        params.append(market.split(","))
        conds.append("EXISTS (SELECT 1 FROM dim_location _dl WHERE _dl.location_id = d.loc AND _dl.state_id = ANY(%s))")
    if channel:
        params.append(channel.split(","))
        conds.append(
            "EXISTS (SELECT 1 FROM dim_customer _dc "
            "JOIN fact_sales_monthly _fsm ON _fsm.cust_grp = _dc.customer_group "
            "WHERE _fsm.item_id = d.item_id AND _fsm.loc = d.loc AND _dc.rpt_channel_desc = ANY(%s))"
        )
    if cluster:
        params.append(cluster.split(","))
        conds.append("d.cluster_assignment = ANY(%s)")
    return conds


@router.get("/domains/{domain}/distinct")
def domain_distinct(
    domain: str,
    response: FastAPIResponse,
    column: str = Query(..., min_length=1, max_length=120),
    limit: int = Query(default=100, ge=1, le=500),
    search: str = Query(default="", max_length=120),
    # Cascading filter params — narrow results by other active filters
    brand: str = Query(default="", max_length=500),
    category: str = Query(default="", max_length=500),
    item: str = Query(default="", max_length=500),
    location: str = Query(default="", max_length=500),
    market: str = Query(default="", max_length=500),
    channel: str = Query(default="", max_length=500),
    cluster: str = Query(default="", max_length=500),
):
    """Distinct values for a column -- used by global filter dropdowns.

    When cascading filter params are provided, results are narrowed to only
    values that co-exist with the active filter selection (via dim_sku joins).
    """
    set_cache(response, max_age=120)
    spec = get_spec_or_404(domain)
    allowed = _DISTINCT_ALLOWED.get(spec.name, [])
    sql_col = to_sql_col(spec, column)
    if sql_col not in allowed:
        raise HTTPException(400, f"Column '{column}' not allowed for distinct on domain '{domain}'")

    bool(brand or category or item or location or market or channel or cluster)
    cascade_key = (spec.name, sql_col)

    # Always query through dim_sku for mapped columns so that items/locations
    # with zero DFUs never appear in the filter dropdowns.
    if cascade_key in _CASCADING_EXPR:
        join_clause, select_expr = _CASCADING_EXPR[cascade_key]
        params: list[Any] = []
        conds = [f"{select_expr} IS NOT NULL"]
        conds.extend(_build_cascade_conditions(
            params, brand=brand, category=category, item=item,
            location=location, market=market, channel=channel, cluster=cluster,
        ))
        if search.strip():
            conds.append(f"{select_expr}::text ILIKE %s")
            params.append(f"{search.strip()}%")
        where_sql = "WHERE " + " AND ".join(conds)
        sql = f"SELECT DISTINCT {select_expr} FROM dim_sku d {join_clause} {where_sql} ORDER BY {select_expr} LIMIT %s"
        params.append(limit)
    else:
        # Original non-cascading path
        params = []
        where = f"WHERE {sql_col} IS NOT NULL"
        if search.strip():
            where += f" AND {sql_col}::text ILIKE %s"
            params.append(f"{search.strip()}%")
        sql = f"SELECT DISTINCT {sql_col} FROM {spec.table} {where} ORDER BY {sql_col} LIMIT %s"
        params.append(limit)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        values = [str(r[0]) for r in cur.fetchall()]

    return {"column": column, "values": values, "total": len(values)}
