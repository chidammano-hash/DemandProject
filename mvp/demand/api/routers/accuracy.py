"""Forecast accuracy slicing and lag-curve endpoints."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache, compute_kpis

router = APIRouter(tags=["accuracy"])


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


def _add_cross_dim_filters(
    where_parts: list[str],
    params: list[Any],
    *,
    dmdunit_col: str,
    loc_col: str,
    brand: Optional[str] = None,
    category: Optional[str] = None,
    market: Optional[str] = None,
) -> None:
    """Append EXISTS subquery filters for brand (dim_item), category (dim_item), market (dim_location)."""
    if brand:
        values = [v.strip() for v in brand.split(",") if v.strip()]
        if values:
            ph = ",".join(["%s"] * len(values))
            where_parts.append(
                f"EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = {dmdunit_col} AND di.brand_name = ANY(ARRAY[{ph}]))"
            )
            params.extend(values)
    if category:
        values = [v.strip() for v in category.split(",") if v.strip()]
        if values:
            ph = ",".join(["%s"] * len(values))
            where_parts.append(
                f"EXISTS (SELECT 1 FROM dim_item di WHERE di.item_no = {dmdunit_col} AND di.class_ = ANY(ARRAY[{ph}]))"
            )
            params.extend(values)
    if market:
        values = [v.strip() for v in market.split(",") if v.strip()]
        if values:
            ph = ",".join(["%s"] * len(values))
            where_parts.append(
                f"EXISTS (SELECT 1 FROM dim_location dl WHERE dl.loc = {loc_col} AND dl.state_id = ANY(ARRAY[{ph}]))"
            )
            params.extend(values)


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
    "seasonality_profile",
}

_RAW_BUCKET_EXPR: dict[str, str] = {
    "cluster_assignment": "COALESCE(d.cluster_assignment, '(unassigned)')",
    "ml_cluster": "COALESCE(d.ml_cluster, '(unassigned)')",
    "supplier_desc": "COALESCE(d.supplier_desc, '(unknown)')",
    "abc_vol": "COALESCE(d.abc_vol, '(unknown)')",
    "region": "COALESCE(d.region, '(unknown)')",
    "brand_desc": "COALESCE(d.brand_desc, '(unknown)')",
    "dfu_execution_lag": "COALESCE(d.execution_lag::text, '(none)')",
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
    # Use raw fact table when cross-dim filters (brand/category/market) are set,
    # because the pre-aggregated view does not expose dmdunit/loc for EXISTS subqueries.
    use_raw = bool(brand or category or market)

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
        _add_cross_dim_filters(where_parts, main_params,
                               dmdunit_col="f.dmdunit", loc_col="f.loc",
                               brand=brand, category=category, market=market)

        where_sql = " AND ".join(where_parts)
        cte_params: list[Any] = list(model_list) + [len(model_list)]
        all_params = cte_params + main_params

        sql = f"""
            WITH cd AS (
                SELECT dmdunit, dmdgroup, loc
                FROM fact_external_forecast_monthly
                WHERE model_id IN ({cte_ph})
                  AND tothist_dmd IS NOT NULL AND basefcst_pref IS NOT NULL
                GROUP BY 1, 2, 3
                HAVING COUNT(DISTINCT model_id) = %s
            )
            SELECT
                {bucket_expr} AS bucket,
                f.model_id,
                COUNT(DISTINCT (f.dmdunit, f.dmdgroup, f.loc))::bigint AS dfu_count,
                SUM(f.basefcst_pref)                     AS sum_forecast,
                SUM(f.tothist_dmd)                       AS sum_actual,
                SUM(ABS(f.basefcst_pref - f.tothist_dmd)) AS sum_abs_error
            FROM fact_external_forecast_monthly f
            JOIN dim_dfu d
              ON f.dmdunit = d.dmdunit AND f.dmdgroup = d.dmdgroup AND f.loc = d.loc
            WHERE (f.dmdunit, f.dmdgroup, f.loc) IN (SELECT dmdunit, dmdgroup, loc FROM cd)
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
                GROUP BY dmdunit, dmdgroup, loc
                HAVING COUNT(DISTINCT model_id) = %s
            ) sub
        """
        per_model_sql = f"""
            SELECT model_id, COUNT(DISTINCT (dmdunit, dmdgroup, loc))::bigint
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
            "dfu_counts": dfu_counts,
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
        _add_cross_dim_filters(where_parts_raw, raw_params,
                               dmdunit_col="f.dmdunit", loc_col="f.loc",
                               brand=brand, category=category, market=market)

        where_sql_raw = " AND ".join(where_parts_raw)

        sql_raw = f"""
            SELECT
                {bucket_expr} AS bucket,
                f.model_id,
                COUNT(DISTINCT (f.dmdunit, f.dmdgroup, f.loc))::bigint AS dfu_count,
                SUM(f.basefcst_pref)                       AS sum_forecast,
                SUM(f.tothist_dmd)                         AS sum_actual,
                SUM(ABS(f.basefcst_pref - f.tothist_dmd))  AS sum_abs_error
            FROM fact_external_forecast_monthly f
            JOIN dim_dfu d
              ON f.dmdunit = d.dmdunit AND f.dmdgroup = d.dmdgroup AND f.loc = d.loc
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
):
    set_cache(response, max_age=120, stale_while_revalidate=300)
    """Return accuracy by lag (0-4) for each model."""
    model_list = [m.strip() for m in models.split(",") if m.strip()] if models.strip() else []
    if len(model_list) > 20:
        raise HTTPException(status_code=422, detail="models: max 20 values allowed")

    use_common = common_dfus and len(model_list) >= 2
    use_raw = bool(brand or category or market)

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
        _add_cross_dim_filters(where_parts, main_params,
                               dmdunit_col="a.dmdunit", loc_col="a.loc",
                               brand=brand, category=category, market=market)

        where_sql = " AND ".join(where_parts)
        cte_params: list[Any] = list(model_list) + [len(model_list)]
        all_params = cte_params + main_params

        sql = f"""
            WITH cd AS (
                SELECT dmdunit, dmdgroup, loc
                FROM backtest_lag_archive
                WHERE model_id IN ({cte_ph})
                  AND tothist_dmd IS NOT NULL AND basefcst_pref IS NOT NULL
                GROUP BY 1, 2, 3
                HAVING COUNT(DISTINCT model_id) = %s
            )
            SELECT
                a.model_id,
                a.lag,
                COUNT(DISTINCT (a.dmdunit, a.dmdgroup, a.loc))::bigint AS dfu_count,
                SUM(a.basefcst_pref)          AS sum_forecast,
                SUM(a.tothist_dmd)            AS sum_actual,
                SUM(ABS(a.basefcst_pref - a.tothist_dmd)) AS sum_abs_error
            FROM backtest_lag_archive a
            JOIN dim_dfu d
              ON a.dmdunit = d.dmdunit AND a.dmdgroup = d.dmdgroup AND a.loc = d.loc
            WHERE (a.dmdunit, a.dmdgroup, a.loc) IN (SELECT dmdunit, dmdgroup, loc FROM cd)
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
        _add_cross_dim_filters(where_parts_raw, raw_params,
                               dmdunit_col="a.dmdunit", loc_col="a.loc",
                               brand=brand, category=category, market=market)

        where_sql_raw = " AND ".join(where_parts_raw)

        sql_raw = f"""
            SELECT
                a.model_id,
                a.lag,
                COUNT(DISTINCT (a.dmdunit, a.dmdgroup, a.loc))::bigint AS dfu_count,
                SUM(a.basefcst_pref)          AS sum_forecast,
                SUM(a.tothist_dmd)            AS sum_actual,
                SUM(ABS(a.basefcst_pref - a.tothist_dmd)) AS sum_abs_error
            FROM backtest_lag_archive a
            JOIN dim_dfu d
              ON a.dmdunit = d.dmdunit AND a.dmdgroup = d.dmdgroup AND a.loc = d.loc
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
            for mid, l, cnt in cur.fetchall():
                dfu_map[(mid, int(l))] = int(cnt)

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
