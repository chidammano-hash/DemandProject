"""Forecast accuracy slicing and lag-curve endpoints."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import add_cross_dim_filters, compute_kpis, get_conn, set_cache
from common.services.metrics import compute_unweighted_accuracy

router = APIRouter(tags=["accuracy"])

# Dimensions that the per-DFU decomposition can group by. These are exactly the
# DFU-constant attribute columns carried in agg_accuracy_by_dfu (sql/193);
# month_start/lag/model_id are filters, not group keys, so they are excluded.
_DECOMP_GROUP_DIMS = {
    "cluster_assignment",
    "ml_cluster",
    "supplier_desc",
    "abc_vol",
    "region",
    "brand_desc",
    "seasonality_profile",
    "dfu_execution_lag",
}


def _add_dim_filters(
    where_parts: list[str],
    params: list[Any],
    *,
    cluster_assignment: str = "",
    supplier_desc: str = "",
    abc_vol: str = "",
    region: str = "",
    seasonality_profile: str = "",
    table_alias: str = "",
) -> None:
    """Append the 5 standard DFU dimension filters to where_parts/params in-place."""
    prefix = f"{table_alias}." if table_alias else ""
    if cluster_assignment.strip():
        where_parts.append(f"{prefix}cluster_assignment = %s")
        params.append(cluster_assignment.strip())
    if supplier_desc.strip():
        where_parts.append(f"{prefix}supplier_desc = %s")
        params.append(supplier_desc.strip())
    if abc_vol.strip():
        where_parts.append(f"{prefix}abc_vol = %s")
        params.append(abc_vol.strip())
    if region.strip():
        where_parts.append(f"{prefix}region = %s")
        params.append(region.strip())
    if seasonality_profile.strip():
        where_parts.append(f"{prefix}seasonality_profile = %s")
        params.append(seasonality_profile.strip())


def _add_item_location_filters(
    where_parts: list[str],
    params: list[Any],
    *,
    item_id_col: str,
    loc_col: str,
    item: Optional[str] = None,
    location: Optional[str] = None,
) -> None:
    """Append direct item/location IN-list filters."""
    if item:
        values = [v.strip() for v in item.split(",") if v.strip()]
        if values:
            ph = ",".join(["%s"] * len(values))
            where_parts.append(f"{item_id_col} IN ({ph})")
            params.extend(values)
    if location:
        values = [v.strip() for v in location.split(",") if v.strip()]
        if values:
            ph = ",".join(["%s"] * len(values))
            where_parts.append(f"{loc_col} IN ({ph})")
            params.extend(values)


_ACCURACY_SLICE_DIMS = {
    "cluster_assignment",
    "ml_cluster",
    "supplier_desc",
    "abc_vol",
    "region",
    "brand_desc",
    "sku_execution_lag",
    "month_start",
    "lag",
    "model_id",
    "seasonality_profile",
}

_RAW_BUCKET_EXPR: dict[str, str] = {
    "cluster_assignment": "COALESCE(d.cluster_assignment, '(unassigned)')",
    "ml_cluster": "COALESCE(d.ml_cluster, '(unassigned)')",
    "supplier_desc": "COALESCE(d.supplier_desc, '(unknown)')",
    "abc_vol": "COALESCE(d.abc_vol, '(unknown)')",
    "region": "COALESCE(d.region, '(unknown)')",
    "brand_desc": "COALESCE(d.brand_desc, '(unknown)')",
    "sku_execution_lag": "COALESCE(d.execution_lag::text, '(none)')",
    "seasonality_profile": "COALESCE(d.seasonality_profile, '(unknown)')",
    "month_start": "date_trunc('month', f.startdate)::date",
    "lag": "f.lag",
    "model_id": "f.model_id",
}


@router.get("/forecast/accuracy/slice")
def forecast_accuracy_slice(
    response: FastAPIResponse,
    group_by: str = Query(default="cluster_assignment", max_length=64),
    models: str = Query(default="", max_length=500),
    lag: int = Query(default=-1, ge=-1, le=4),
    month_from: str = Query(default="", max_length=20),
    month_to: str = Query(default="", max_length=20),
    cluster_assignment: str = Query(default="", max_length=120),
    supplier_desc: str = Query(default="", max_length=120),
    abc_vol: str = Query(default="", max_length=40),
    region: str = Query(default="", max_length=120),
    seasonality_profile: str = Query(default="", max_length=120),
    common_dfus: bool = Query(default=False),
    include_dfu_count: bool = Query(default=False),
    brand: Optional[str] = Query(default=None, max_length=500),
    category: Optional[str] = Query(default=None, max_length=500),
    market: Optional[str] = Query(default=None, max_length=500),
    item: Optional[str] = Query(default=None, max_length=500),
    location: Optional[str] = Query(default=None, max_length=500),
):
    set_cache(response, max_age=120, stale_while_revalidate=300)
    """Return accuracy KPIs grouped by a chosen DFU-attribute dimension."""
    if group_by not in _ACCURACY_SLICE_DIMS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid group_by '{group_by}'. Valid: {sorted(_ACCURACY_SLICE_DIMS)}",
        )

    model_list = [m.strip() for m in models.split(",") if m.strip()] if models.strip() else []
    if len(model_list) > 20:
        raise HTTPException(status_code=422, detail="models: max 20 values allowed")

    # ── Common-DFUs path: raw fact table with intersection CTE ──────────
    use_common = common_dfus and len(model_list) >= 2
    # Use raw fact table when cross-dim filters (brand/category/market/item/location) are set,
    # because the pre-aggregated view does not expose item_id/loc for direct filtering.
    use_raw = bool(brand or category or market or item or location)

    if use_common:
        bucket_expr = _RAW_BUCKET_EXPR.get(group_by)
        if bucket_expr is None:
            raise HTTPException(422, f"group_by '{group_by}' not supported with common_dfus filter")

        cte_ph = ",".join(["%s"] * len(model_list))

        where_parts: list[str] = [f"f.model_id IN ({cte_ph})"]
        main_params: list[Any] = list(model_list)

        if lag == -1:
            where_parts.append("f.lag = d.execution_lag")
        elif lag >= 0:
            where_parts.append("f.lag = %s")
            main_params.append(lag)
        if month_from.strip():
            where_parts.append("date_trunc('month', f.startdate)::date >= %s::date")
            main_params.append(month_from.strip())
        if month_to.strip():
            where_parts.append("date_trunc('month', f.startdate)::date <= %s::date")
            main_params.append(month_to.strip())
        if cluster_assignment.strip():
            where_parts.append("COALESCE(d.cluster_assignment, '(unassigned)') = %s")
            main_params.append(cluster_assignment.strip())
        if supplier_desc.strip():
            where_parts.append("COALESCE(d.supplier_desc, '(unknown)') = %s")
            main_params.append(supplier_desc.strip())
        if abc_vol.strip():
            where_parts.append("COALESCE(d.abc_vol, '(unknown)') = %s")
            main_params.append(abc_vol.strip())
        if region.strip():
            where_parts.append("COALESCE(d.region, '(unknown)') = %s")
            main_params.append(region.strip())
        if seasonality_profile.strip():
            where_parts.append("COALESCE(d.seasonality_profile, '(unknown)') = %s")
            main_params.append(seasonality_profile.strip())
        _add_item_location_filters(where_parts, main_params,
                                   item_id_col="f.item_id", loc_col="f.loc",
                                   item=item, location=location)
        add_cross_dim_filters(where_parts, main_params,
                              item_col="f.item_id", loc_col="f.loc",
                              brand=brand, category=category, market=market)

        where_sql = " AND ".join(where_parts)
        cte_params: list[Any] = list(model_list) + [len(model_list)]
        all_params = cte_params + main_params

        sql = f"""
            WITH cd AS (
                SELECT item_id, customer_group, loc
                FROM fact_external_forecast_monthly
                WHERE model_id IN ({cte_ph})
                  AND tothist_dmd IS NOT NULL AND basefcst_pref IS NOT NULL
                GROUP BY 1, 2, 3
                HAVING COUNT(DISTINCT model_id) = %s
            )
            SELECT
                {bucket_expr} AS bucket,
                f.model_id,
                COUNT(DISTINCT (f.item_id, f.customer_group, f.loc))::bigint AS dfu_count,
                SUM(f.basefcst_pref)                     AS sum_forecast,
                SUM(f.tothist_dmd)                       AS sum_actual,
                SUM(ABS(f.basefcst_pref - f.tothist_dmd)) AS sum_abs_error
            FROM fact_external_forecast_monthly f
            JOIN dim_sku d
              ON f.item_id = d.item_id AND f.customer_group = d.customer_group AND f.loc = d.loc
            WHERE (f.item_id, f.customer_group, f.loc) IN (SELECT item_id, customer_group, loc FROM cd)
              AND f.tothist_dmd IS NOT NULL AND f.basefcst_pref IS NOT NULL
              AND {where_sql}
            GROUP BY 1, 2
            ORDER BY 1 ASC NULLS LAST, 2 ASC
        """

        count_sql = f"""
            SELECT COUNT(*)::bigint FROM (
                SELECT 1 FROM fact_external_forecast_monthly
                WHERE model_id IN ({cte_ph})
                  AND tothist_dmd IS NOT NULL AND basefcst_pref IS NOT NULL
                GROUP BY item_id, customer_group, loc
                HAVING COUNT(DISTINCT model_id) = %s
            ) sub
        """
        per_model_sql = f"""
            SELECT model_id, COUNT(DISTINCT (item_id, customer_group, loc))::bigint
            FROM fact_external_forecast_monthly
            WHERE model_id IN ({cte_ph})
              AND tothist_dmd IS NOT NULL AND basefcst_pref IS NOT NULL
            GROUP BY 1
        """

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, all_params)
            db_rows = cur.fetchall()
            cur.execute(count_sql, list(model_list) + [len(model_list)])
            common_count = int(cur.fetchone()[0])  # type: ignore[index]
            cur.execute(per_model_sql, list(model_list))
            dfu_counts = {r[0]: int(r[1]) for r in cur.fetchall()}

        pivot: dict[str, dict[str, Any]] = {}
        for bucket, mid, dfu_cnt, sf, sa, sae in db_rows:
            b = str(bucket) if bucket is not None else "(unknown)"
            if b not in pivot:
                pivot[b] = {"bucket": b, "n_rows": 0, "by_model": {}}
            pivot[b]["n_rows"] = int(pivot[b]["n_rows"]) + int(dfu_cnt or 0)
            pivot[b]["by_model"][mid] = compute_kpis(float(sf or 0), float(sa or 0), float(sae or 0), int(dfu_cnt or 0))

        return {
            "group_by": group_by,
            "lag_filter": lag,
            "models": model_list,
            "common_dfus": True,
            "common_dfu_count": common_count,
            "common_sku_count": common_count,
            "dfu_counts": dfu_counts,
            "sku_counts": dfu_counts,
            "rows": sorted(pivot.values(), key=lambda r: r["bucket"]),
            "source": "fact_external_forecast_monthly",
        }

    if use_raw:
        # ── Raw fact table path: needed when brand/category/market filters are set ──
        bucket_expr = _RAW_BUCKET_EXPR.get(group_by)
        if bucket_expr is None:
            raise HTTPException(422, f"group_by '{group_by}' not supported with brand/category/market filter")

        where_parts_raw: list[str] = ["f.tothist_dmd IS NOT NULL", "f.basefcst_pref IS NOT NULL"]
        raw_params: list[Any] = []

        if model_list:
            ph = ",".join(["%s"] * len(model_list))
            where_parts_raw.append(f"f.model_id IN ({ph})")
            raw_params.extend(model_list)
        if lag == -1:
            where_parts_raw.append("f.lag = d.execution_lag")
        elif lag >= 0:
            where_parts_raw.append("f.lag = %s")
            raw_params.append(lag)
        if month_from.strip():
            where_parts_raw.append("date_trunc('month', f.startdate)::date >= %s::date")
            raw_params.append(month_from.strip())
        if month_to.strip():
            where_parts_raw.append("date_trunc('month', f.startdate)::date <= %s::date")
            raw_params.append(month_to.strip())
        if cluster_assignment.strip():
            where_parts_raw.append("COALESCE(d.cluster_assignment, '(unassigned)') = %s")
            raw_params.append(cluster_assignment.strip())
        if supplier_desc.strip():
            where_parts_raw.append("COALESCE(d.supplier_desc, '(unknown)') = %s")
            raw_params.append(supplier_desc.strip())
        if abc_vol.strip():
            where_parts_raw.append("COALESCE(d.abc_vol, '(unknown)') = %s")
            raw_params.append(abc_vol.strip())
        if region.strip():
            where_parts_raw.append("COALESCE(d.region, '(unknown)') = %s")
            raw_params.append(region.strip())
        if seasonality_profile.strip():
            where_parts_raw.append("COALESCE(d.seasonality_profile, '(unknown)') = %s")
            raw_params.append(seasonality_profile.strip())
        _add_item_location_filters(where_parts_raw, raw_params,
                                   item_id_col="f.item_id", loc_col="f.loc",
                                   item=item, location=location)
        add_cross_dim_filters(where_parts_raw, raw_params,
                              item_col="f.item_id", loc_col="f.loc",
                              brand=brand, category=category, market=market)

        where_sql_raw = " AND ".join(where_parts_raw)

        sql_raw = f"""
            SELECT
                {bucket_expr} AS bucket,
                f.model_id,
                COUNT(DISTINCT (f.item_id, f.customer_group, f.loc))::bigint AS dfu_count,
                SUM(f.basefcst_pref)                       AS sum_forecast,
                SUM(f.tothist_dmd)                         AS sum_actual,
                SUM(ABS(f.basefcst_pref - f.tothist_dmd))  AS sum_abs_error
            FROM fact_external_forecast_monthly f
            JOIN dim_sku d
              ON f.item_id = d.item_id AND f.customer_group = d.customer_group AND f.loc = d.loc
            WHERE {where_sql_raw}
            GROUP BY 1, 2
            ORDER BY 1 ASC NULLS LAST, 2 ASC
        """

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql_raw, raw_params)
            db_rows = cur.fetchall()

        pivot_raw: dict[str, dict[str, Any]] = {}
        for bucket, mid, dfu_cnt, sf, sa, sae in db_rows:
            b = str(bucket) if bucket is not None else "(unknown)"
            if b not in pivot_raw:
                pivot_raw[b] = {"bucket": b, "n_rows": 0, "by_model": {}}
            pivot_raw[b]["n_rows"] = int(pivot_raw[b]["n_rows"]) + int(dfu_cnt or 0)
            pivot_raw[b]["by_model"][mid] = compute_kpis(float(sf or 0), float(sa or 0), float(sae or 0), int(dfu_cnt or 0))

        return {
            "group_by": group_by,
            "lag_filter": lag,
            "models": model_list or None,
            "rows": sorted(pivot_raw.values(), key=lambda r: r["bucket"]),
            "source": "fact_external_forecast_monthly",
        }

    # ── Standard path: pre-aggregated view ──────────────────────────────
    where_parts = []
    params: list[Any] = []

    if model_list:
        placeholders = ",".join(["%s"] * len(model_list))
        where_parts.append(f"model_id IN ({placeholders})")
        params.extend(model_list)
    if lag == -1:
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
    _add_dim_filters(where_parts, params,
                     cluster_assignment=cluster_assignment, supplier_desc=supplier_desc,
                     abc_vol=abc_vol, region=region, seasonality_profile=seasonality_profile)

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

    dfu_map: dict[tuple[str, str], int] = {}
    if include_dfu_count:
        dfu_where: list[str] = []
        dfu_params: list[Any] = []
        if model_list:
            dfu_where.append(f"model_id IN ({','.join(['%s'] * len(model_list))})")
            dfu_params.extend(model_list)
        if lag == -1:
            dfu_where.append("lag::text = dfu_execution_lag")
        elif lag >= 0:
            dfu_where.append("lag = %s")
            dfu_params.append(lag)
        if month_from.strip():
            dfu_where.append("max_month >= %s::date")
            dfu_params.append(month_from.strip())
        if month_to.strip():
            dfu_where.append("min_month <= %s::date")
            dfu_params.append(month_to.strip())
        _add_dim_filters(dfu_where, dfu_params,
                         cluster_assignment=cluster_assignment, supplier_desc=supplier_desc,
                         abc_vol=abc_vol, region=region, seasonality_profile=seasonality_profile)

        dfu_where_sql = ("WHERE " + " AND ".join(dfu_where)) if dfu_where else ""
        dfu_sql = f"""
            SELECT {group_by} AS bucket, model_id, COUNT(*)::bigint AS dfu_count
            FROM agg_dfu_coverage
            {dfu_where_sql}
            GROUP BY 1, 2
        """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        db_rows = cur.fetchall()
        if include_dfu_count:
            cur.execute(dfu_sql, dfu_params)
            for b_raw, mid, cnt in cur.fetchall():
                dfu_map[(str(b_raw) if b_raw is not None else "(unknown)", mid)] = int(cnt)

    pivot = {}
    for bucket, model_id, n_rows, sf, sa, sae in db_rows:
        b = str(bucket) if bucket is not None else "(unknown)"
        if b not in pivot:
            pivot[b] = {"bucket": b, "n_rows": 0, "by_model": {}}
        pivot[b]["n_rows"] = int(pivot[b]["n_rows"]) + int(n_rows or 0)
        dfu_cnt = dfu_map.get((b, model_id), 0)
        pivot[b]["by_model"][model_id] = compute_kpis(float(sf or 0), float(sa or 0), float(sae or 0), dfu_cnt)

    return {
        "group_by": group_by,
        "lag_filter": lag,
        "models": model_list or None,
        "rows": sorted(pivot.values(), key=lambda r: r["bucket"]),
        "source": "agg_accuracy_by_dim",
    }


@router.get("/forecast/accuracy/lag-curve")
def forecast_accuracy_lag_curve(
    response: FastAPIResponse,
    models: str = Query(default="", max_length=500),
    cluster_assignment: str = Query(default="", max_length=120),
    supplier_desc: str = Query(default="", max_length=120),
    abc_vol: str = Query(default="", max_length=40),
    region: str = Query(default="", max_length=120),
    seasonality_profile: str = Query(default="", max_length=120),
    month_from: str = Query(default="", max_length=20),
    month_to: str = Query(default="", max_length=20),
    common_dfus: bool = Query(default=False),
    include_dfu_count: bool = Query(default=False),
    brand: Optional[str] = Query(default=None, max_length=500),
    category: Optional[str] = Query(default=None, max_length=500),
    market: Optional[str] = Query(default=None, max_length=500),
    item: Optional[str] = Query(default=None, max_length=500),
    location: Optional[str] = Query(default=None, max_length=500),
):
    set_cache(response, max_age=120, stale_while_revalidate=300)
    """Return accuracy by lag (0-4) for each model."""
    model_list = [m.strip() for m in models.split(",") if m.strip()] if models.strip() else []
    if len(model_list) > 20:
        raise HTTPException(status_code=422, detail="models: max 20 values allowed")

    use_common = common_dfus and len(model_list) >= 2
    use_raw = bool(brand or category or market or item or location)

    if use_common:
        cte_ph = ",".join(["%s"] * len(model_list))

        where_parts: list[str] = [f"a.model_id IN ({cte_ph})"]
        main_params: list[Any] = list(model_list)

        if cluster_assignment.strip():
            where_parts.append("COALESCE(d.cluster_assignment, '(unassigned)') = %s")
            main_params.append(cluster_assignment.strip())
        if supplier_desc.strip():
            where_parts.append("COALESCE(d.supplier_desc, '(unknown)') = %s")
            main_params.append(supplier_desc.strip())
        if abc_vol.strip():
            where_parts.append("COALESCE(d.abc_vol, '(unknown)') = %s")
            main_params.append(abc_vol.strip())
        if region.strip():
            where_parts.append("COALESCE(d.region, '(unknown)') = %s")
            main_params.append(region.strip())
        if seasonality_profile.strip():
            where_parts.append("COALESCE(d.seasonality_profile, '(unknown)') = %s")
            main_params.append(seasonality_profile.strip())
        if month_from.strip():
            where_parts.append("date_trunc('month', a.startdate)::date >= %s::date")
            main_params.append(month_from.strip())
        if month_to.strip():
            where_parts.append("date_trunc('month', a.startdate)::date <= %s::date")
            main_params.append(month_to.strip())
        _add_item_location_filters(where_parts, main_params,
                                   item_id_col="a.item_id", loc_col="a.loc",
                                   item=item, location=location)
        add_cross_dim_filters(where_parts, main_params,
                              item_col="a.item_id", loc_col="a.loc",
                              brand=brand, category=category, market=market)

        where_sql = " AND ".join(where_parts)
        cte_params: list[Any] = list(model_list) + [len(model_list)]
        all_params = cte_params + main_params

        sql = f"""
            WITH cd AS (
                SELECT item_id, customer_group, loc
                FROM backtest_lag_archive
                WHERE model_id IN ({cte_ph})
                  AND tothist_dmd IS NOT NULL AND basefcst_pref IS NOT NULL
                GROUP BY 1, 2, 3
                HAVING COUNT(DISTINCT model_id) = %s
            )
            SELECT
                a.model_id,
                a.lag,
                COUNT(DISTINCT (a.item_id, a.customer_group, a.loc))::bigint AS dfu_count,
                SUM(a.basefcst_pref)          AS sum_forecast,
                SUM(a.tothist_dmd)            AS sum_actual,
                SUM(ABS(a.basefcst_pref - a.tothist_dmd)) AS sum_abs_error
            FROM backtest_lag_archive a
            JOIN dim_sku d
              ON a.item_id = d.item_id AND a.customer_group = d.customer_group AND a.loc = d.loc
            WHERE (a.item_id, a.customer_group, a.loc) IN (SELECT item_id, customer_group, loc FROM cd)
              AND a.tothist_dmd IS NOT NULL AND a.basefcst_pref IS NOT NULL
              AND {where_sql}
            GROUP BY 1, 2
            ORDER BY 2 ASC, 1 ASC
        """

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, all_params)
            db_rows = cur.fetchall()

        by_lag: dict[int, dict[str, Any]] = {}
        for mid, lag_val, dfu_cnt, sf, sa, sae in db_rows:
            lag_key = int(lag_val)
            if lag_key not in by_lag:
                by_lag[lag_key] = {"lag": lag_key, "by_model": {}}
            by_lag[lag_key]["by_model"][mid] = compute_kpis(
                float(sf or 0), float(sa or 0), float(sae or 0), int(dfu_cnt or 0)
            )

        return {
            "models": model_list,
            "common_dfus": True,
            "by_lag": [by_lag[k] for k in sorted(by_lag.keys())],
            "source": "backtest_lag_archive",
        }

    if use_raw:
        # ── Raw fact table path: needed when brand/category/market filters are set ──
        where_parts_raw: list[str] = ["a.tothist_dmd IS NOT NULL", "a.basefcst_pref IS NOT NULL"]
        raw_params: list[Any] = []

        if model_list:
            ph = ",".join(["%s"] * len(model_list))
            where_parts_raw.append(f"a.model_id IN ({ph})")
            raw_params.extend(model_list)
        if cluster_assignment.strip():
            where_parts_raw.append("COALESCE(d.cluster_assignment, '(unassigned)') = %s")
            raw_params.append(cluster_assignment.strip())
        if supplier_desc.strip():
            where_parts_raw.append("COALESCE(d.supplier_desc, '(unknown)') = %s")
            raw_params.append(supplier_desc.strip())
        if abc_vol.strip():
            where_parts_raw.append("COALESCE(d.abc_vol, '(unknown)') = %s")
            raw_params.append(abc_vol.strip())
        if region.strip():
            where_parts_raw.append("COALESCE(d.region, '(unknown)') = %s")
            raw_params.append(region.strip())
        if seasonality_profile.strip():
            where_parts_raw.append("COALESCE(d.seasonality_profile, '(unknown)') = %s")
            raw_params.append(seasonality_profile.strip())
        if month_from.strip():
            where_parts_raw.append("date_trunc('month', a.startdate)::date >= %s::date")
            raw_params.append(month_from.strip())
        if month_to.strip():
            where_parts_raw.append("date_trunc('month', a.startdate)::date <= %s::date")
            raw_params.append(month_to.strip())
        _add_item_location_filters(where_parts_raw, raw_params,
                                   item_id_col="a.item_id", loc_col="a.loc",
                                   item=item, location=location)
        add_cross_dim_filters(where_parts_raw, raw_params,
                              item_col="a.item_id", loc_col="a.loc",
                              brand=brand, category=category, market=market)

        where_sql_raw = " AND ".join(where_parts_raw)

        sql_raw = f"""
            SELECT
                a.model_id,
                a.lag,
                COUNT(DISTINCT (a.item_id, a.customer_group, a.loc))::bigint AS dfu_count,
                SUM(a.basefcst_pref)          AS sum_forecast,
                SUM(a.tothist_dmd)            AS sum_actual,
                SUM(ABS(a.basefcst_pref - a.tothist_dmd)) AS sum_abs_error
            FROM backtest_lag_archive a
            JOIN dim_sku d
              ON a.item_id = d.item_id AND a.customer_group = d.customer_group AND a.loc = d.loc
            WHERE {where_sql_raw}
            GROUP BY 1, 2
            ORDER BY 2 ASC, 1 ASC
        """

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql_raw, raw_params)
            db_rows = cur.fetchall()

        by_lag_raw: dict[int, dict[str, Any]] = {}
        for mid, lag_val, dfu_cnt, sf, sa, sae in db_rows:
            lag_key = int(lag_val)
            if lag_key not in by_lag_raw:
                by_lag_raw[lag_key] = {"lag": lag_key, "by_model": {}}
            by_lag_raw[lag_key]["by_model"][mid] = compute_kpis(
                float(sf or 0), float(sa or 0), float(sae or 0), int(dfu_cnt or 0)
            )

        return {
            "models": model_list or None,
            "by_lag": [by_lag_raw[k] for k in sorted(by_lag_raw.keys())],
            "source": "backtest_lag_archive",
        }

    # ── Standard path: pre-aggregated view ──────────────────────────────
    where_parts = []
    params: list[Any] = []

    if model_list:
        placeholders = ",".join(["%s"] * len(model_list))
        where_parts.append(f"model_id IN ({placeholders})")
        params.extend(model_list)
    _add_dim_filters(where_parts, params,
                     cluster_assignment=cluster_assignment, supplier_desc=supplier_desc,
                     abc_vol=abc_vol, region=region, seasonality_profile=seasonality_profile)
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

    dfu_map: dict[tuple[str, int], int] = {}
    if include_dfu_count:
        dfu_where: list[str] = []
        dfu_params: list[Any] = []
        if model_list:
            dfu_where.append(f"model_id IN ({','.join(['%s'] * len(model_list))})")
            dfu_params.extend(model_list)
        _add_dim_filters(dfu_where, dfu_params,
                         cluster_assignment=cluster_assignment, supplier_desc=supplier_desc,
                         abc_vol=abc_vol, region=region, seasonality_profile=seasonality_profile)
        if month_from.strip():
            dfu_where.append("max_month >= %s::date")
            dfu_params.append(month_from.strip())
        if month_to.strip():
            dfu_where.append("min_month <= %s::date")
            dfu_params.append(month_to.strip())

        dfu_where_sql = ("WHERE " + " AND ".join(dfu_where)) if dfu_where else ""
        dfu_sql = f"""
            SELECT model_id, lag, COUNT(*)::bigint AS dfu_count
            FROM agg_dfu_coverage_lag_archive
            {dfu_where_sql}
            GROUP BY 1, 2
        """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        db_rows = cur.fetchall()
        if include_dfu_count:
            cur.execute(dfu_sql, dfu_params)
            for mid, lag, cnt in cur.fetchall():
                dfu_map[(mid, int(lag))] = int(cnt)

    by_lag = {}
    for model_id, lag_val, n_rows, sf, sa, sae in db_rows:
        lag_key = int(lag_val)
        if lag_key not in by_lag:
            by_lag[lag_key] = {"lag": lag_key, "by_model": {}}
        dfu_cnt = dfu_map.get((model_id, lag_key), 0)
        by_lag[lag_key]["by_model"][model_id] = compute_kpis(
            float(sf or 0), float(sa or 0), float(sae or 0), dfu_cnt
        )

    return {
        "models": model_list or None,
        "by_lag": [by_lag[k] for k in sorted(by_lag.keys())],
        "source": "agg_accuracy_lag_archive",
    }


@router.get("/forecast/accuracy/lag-leaderboard")
def forecast_accuracy_lag_leaderboard(
    response: FastAPIResponse,
    month_from: str = Query(default="", max_length=20),
    month_to: str = Query(default="", max_length=20),
    limit: int = Query(default=10, ge=1, le=50),
):
    """Per-lag model leaderboard ranked by accuracy (data: agg_accuracy_lag_archive).

    Returns WAPE and bias for each model at execution lags 0-4. Pinball loss
    requires quantile forecast rows and is not computed here.
    """
    set_cache(response, max_age=120, stale_while_revalidate=300)

    where_parts: list[str] = []
    params: list[Any] = []
    if month_from.strip():
        where_parts.append("month_start >= %s::date")
        params.append(month_from.strip())
    if month_to.strip():
        where_parts.append("month_start <= %s::date")
        params.append(month_to.strip())
    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    sql = f"""
        SELECT
            lag,
            model_id,
            SUM(row_count)::bigint   AS n_rows,
            SUM(sum_forecast)        AS sum_forecast,
            SUM(sum_actual)          AS sum_actual,
            SUM(sum_abs_error)       AS sum_abs_error
        FROM agg_accuracy_lag_archive
        {where_sql}
        GROUP BY 1, 2
        ORDER BY 1 ASC, 3 DESC
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        db_rows = cur.fetchall()

    by_lag: dict[int, list[dict[str, Any]]] = {lag: [] for lag in range(5)}
    for lag_val, model_id, n_rows, sf, sa, sae in db_rows:
        lag_key = int(lag_val)
        if lag_key not in by_lag:
            continue
        kpis = compute_kpis(float(sf or 0), float(sa or 0), float(sae or 0), int(n_rows or 0))
        by_lag[lag_key].append({
            "model_id": model_id,
            "accuracy_pct": kpis["accuracy_pct"],
            "wape": kpis["wape"],
            "bias": kpis["bias"],
            "n_rows": int(n_rows or 0),
        })

    lags_out: list[dict[str, Any]] = []
    for lag_key in sorted(by_lag.keys()):
        ranked = sorted(
            by_lag[lag_key],
            key=lambda r: (r["accuracy_pct"] is not None, r["accuracy_pct"] or -1.0),
            reverse=True,
        )[:limit]
        for rank, entry in enumerate(ranked, start=1):
            entry["rank"] = rank
        lags_out.append({"lag": lag_key, "rankings": ranked})

    return {
        "lags": lags_out,
        "limit": limit,
        "source": "agg_accuracy_lag_archive",
    }


# ───────────────────────────────────────────────────────────────────────────
# Per-DFU accuracy decomposition (diagnostic layer)
#
# The endpoints above all return the VOLUME-WEIGHTED aggregate WAPE — the headline
# ~72%. They cannot answer "where is the error concentrated?" because they sum
# error across every DFU before dividing, so big SKUs dominate and the long tail
# is invisible. The two endpoints below read agg_accuracy_by_dfu (sql/193), which
# preserves the individual DFU, and expose BOTH the volume-weighted metric and the
# unweighted per-DFU mean/median, plus a Pareto view of error contribution.
# ───────────────────────────────────────────────────────────────────────────


def _build_accuracy_by_dfu_where(
    *,
    model_list: list[str],
    lag: int,
    month_from: str,
    month_to: str,
    cluster_assignment: str,
    supplier_desc: str,
    abc_vol: str,
    region: str,
    seasonality_profile: str,
) -> tuple[str, list[Any]]:
    """Build the WHERE clause for agg_accuracy_by_dfu (single MV, no table alias).

    ``month_from``/``month_to`` apply a *coarse* overlap on each DFU's active
    month range (min_month/max_month), matching agg_dfu_coverage semantics — the
    per-DFU sums are over the DFU's full active period, not an exact month slice.
    """
    where_parts: list[str] = []
    params: list[Any] = []
    if model_list:
        placeholders = ",".join(["%s"] * len(model_list))
        where_parts.append(f"model_id IN ({placeholders})")
        params.extend(model_list)
    if lag == -1:
        where_parts.append("lag::text = dfu_execution_lag")
    elif lag >= 0:
        where_parts.append("lag = %s")
        params.append(lag)
    if month_from.strip():
        where_parts.append("max_month >= %s::date")
        params.append(month_from.strip())
    if month_to.strip():
        where_parts.append("min_month <= %s::date")
        params.append(month_to.strip())
    _add_dim_filters(where_parts, params,
                     cluster_assignment=cluster_assignment, supplier_desc=supplier_desc,
                     abc_vol=abc_vol, region=region, seasonality_profile=seasonality_profile)
    where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    return where_sql, params


@router.get("/forecast/accuracy/decomposition")
def forecast_accuracy_decomposition(
    response: FastAPIResponse,
    group_by: str = Query(default="seasonality_profile", max_length=64),
    models: str = Query(default="", max_length=500),
    lag: int = Query(default=-1, ge=-1, le=4),
    month_from: str = Query(default="", max_length=20),
    month_to: str = Query(default="", max_length=20),
    cluster_assignment: str = Query(default="", max_length=120),
    supplier_desc: str = Query(default="", max_length=120),
    abc_vol: str = Query(default="", max_length=40),
    region: str = Query(default="", max_length=120),
    seasonality_profile: str = Query(default="", max_length=120),
):
    set_cache(response, max_age=120, stale_while_revalidate=300)
    """Per-bucket accuracy with BOTH weightings plus error-contribution share.

    For each bucket x model returns:
      • ``volume_weighted`` — the headline KPIs (compute_kpis), big SKUs dominate.
      • ``unweighted`` — per-DFU WAPE then mean/median (compute_unweighted_accuracy),
        every DFU equal, with ``n_undefined`` for zero-actual DFUs.
      • ``error_contribution_pct`` — bucket's share of the model's total absolute
        error (Pareto): the buckets that own the error.
    Data: agg_accuracy_by_dfu (sql/193), one row per DFU x model x lag.
    """
    if group_by not in _DECOMP_GROUP_DIMS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid group_by '{group_by}'. Valid: {sorted(_DECOMP_GROUP_DIMS)}",
        )
    model_list = [m.strip() for m in models.split(",") if m.strip()] if models.strip() else []
    if len(model_list) > 20:
        raise HTTPException(status_code=422, detail="models: max 20 values allowed")

    where_sql, params = _build_accuracy_by_dfu_where(
        model_list=model_list, lag=lag, month_from=month_from, month_to=month_to,
        cluster_assignment=cluster_assignment, supplier_desc=supplier_desc,
        abc_vol=abc_vol, region=region, seasonality_profile=seasonality_profile)

    # One row per DFU (after the lag filter); group_by is whitelisted above.
    sql = f"""
        SELECT {group_by} AS bucket, model_id, sum_forecast, sum_actual, sum_abs_error
        FROM agg_accuracy_by_dfu
        {where_sql}
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        db_rows = cur.fetchall()

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    model_total_err: dict[str, float] = {}
    for bucket, model_id, sf, sa, sae in db_rows:
        b = str(bucket) if bucket is not None else "(unknown)"
        g = groups.setdefault((b, model_id), {
            "sum_forecast": 0.0, "sum_actual": 0.0, "sum_abs_error": 0.0, "per_dfu": []})
        sf_f, sa_f, sae_f = float(sf or 0), float(sa or 0), float(sae or 0)
        g["sum_forecast"] += sf_f
        g["sum_actual"] += sa_f
        g["sum_abs_error"] += sae_f
        g["per_dfu"].append((sa_f, sae_f))
        model_total_err[model_id] = model_total_err.get(model_id, 0.0) + sae_f

    pivot: dict[str, dict[str, Any]] = {}
    for (b, model_id), g in groups.items():
        row = pivot.setdefault(b, {"bucket": b, "by_model": {}})
        total_err = model_total_err.get(model_id, 0.0)
        row["by_model"][model_id] = {
            "volume_weighted": compute_kpis(
                g["sum_forecast"], g["sum_actual"], g["sum_abs_error"], len(g["per_dfu"])),
            "unweighted": compute_unweighted_accuracy(g["per_dfu"]),
            "error_contribution_pct": (
                round(100.0 * g["sum_abs_error"] / total_err, 4) if total_err > 0 else None),
            "n_dfus": len(g["per_dfu"]),
        }

    return {
        "group_by": group_by,
        "lag_filter": lag,
        "models": model_list or None,
        "rows": sorted(pivot.values(), key=lambda r: r["bucket"]),
        "source": "agg_accuracy_by_dfu",
    }


@router.get("/forecast/accuracy/error-contributors")
def forecast_accuracy_error_contributors(
    response: FastAPIResponse,
    models: str = Query(default="", max_length=500),
    lag: int = Query(default=-1, ge=-1, le=4),
    limit: int = Query(default=20, ge=1, le=200),
    month_from: str = Query(default="", max_length=20),
    month_to: str = Query(default="", max_length=20),
    cluster_assignment: str = Query(default="", max_length=120),
    supplier_desc: str = Query(default="", max_length=120),
    abc_vol: str = Query(default="", max_length=40),
    region: str = Query(default="", max_length=120),
    seasonality_profile: str = Query(default="", max_length=120),
):
    set_cache(response, max_age=120, stale_while_revalidate=300)
    """Pareto ranking of the DFUs that own the most absolute error ("fix these first").

    Returns the top-``limit`` DFUs by share of total absolute error, each with its
    actual volume, WAPE/accuracy, bias direction (over/under-forecast), and the
    running cumulative error share. Pin a single model (e.g. ``models=champion``)
    for a coherent list; multiple models are summed per DFU. Data: agg_accuracy_by_dfu.
    """
    model_list = [m.strip() for m in models.split(",") if m.strip()] if models.strip() else []
    if len(model_list) > 20:
        raise HTTPException(status_code=422, detail="models: max 20 values allowed")

    where_sql, params = _build_accuracy_by_dfu_where(
        model_list=model_list, lag=lag, month_from=month_from, month_to=month_to,
        cluster_assignment=cluster_assignment, supplier_desc=supplier_desc,
        abc_vol=abc_vol, region=region, seasonality_profile=seasonality_profile)

    # Dim columns are DFU-constant, so MAX() just picks the single value per DFU.
    top_sql = f"""
        SELECT
            item_id, customer_group, loc,
            MAX(cluster_assignment) AS cluster_assignment,
            MAX(region)             AS region,
            MAX(abc_vol)            AS abc_vol,
            MAX(seasonality_profile) AS seasonality_profile,
            SUM(sum_forecast)       AS sum_forecast,
            SUM(sum_actual)         AS sum_actual,
            SUM(sum_abs_error)      AS sum_abs_error
        FROM agg_accuracy_by_dfu
        {where_sql}
        GROUP BY item_id, customer_group, loc
        ORDER BY SUM(sum_abs_error) DESC
        LIMIT %s
    """
    total_sql = f"""
        SELECT COALESCE(SUM(sum_abs_error), 0)::double precision,
               COUNT(DISTINCT (item_id, customer_group, loc))::bigint
        FROM agg_accuracy_by_dfu
        {where_sql}
    """

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(top_sql, [*params, limit])
        top_rows = cur.fetchall()
        cur.execute(total_sql, params)
        total_abs_error, total_dfus = cur.fetchone()

    total_err = float(total_abs_error or 0)
    contributors: list[dict[str, Any]] = []
    cumulative = 0.0
    for (item_id, customer_group, loc, cluster, region_v, abc, season,
         sf, sa, sae) in top_rows:
        sae_f = float(sae or 0)
        share = (100.0 * sae_f / total_err) if total_err > 0 else None
        if share is not None:
            cumulative += share
        kpis = compute_kpis(float(sf or 0), float(sa or 0), sae_f, 1)
        bias = kpis["bias"]
        bias_direction = "over" if (bias or 0) > 0 else "under" if (bias or 0) < 0 else "even"
        contributors.append({
            "item_id": item_id,
            "customer_group": customer_group,
            "loc": loc,
            "cluster_assignment": cluster,
            "region": region_v,
            "abc_vol": abc,
            "seasonality_profile": season,
            "sum_actual": kpis["sum_actual"],
            "sum_abs_error": round(sae_f, 2),
            "accuracy_pct": kpis["accuracy_pct"],
            "wape": kpis["wape"],
            "bias": bias,
            "bias_direction": bias_direction if sa else "n/a",
            "error_contribution_pct": round(share, 4) if share is not None else None,
            "cumulative_contribution_pct": round(cumulative, 4) if share is not None else None,
        })

    return {
        "models": model_list or None,
        "lag_filter": lag,
        "limit": limit,
        "total_abs_error": round(total_err, 2),
        "total_dfus": int(total_dfus or 0),
        "contributors": contributors,
        "source": "agg_accuracy_by_dfu",
    }
