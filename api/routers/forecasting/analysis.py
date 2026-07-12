"""DFU Analysis endpoint (feature 17) — sales vs multi-model forecast overlay."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api.core import get_conn
from common.core.sql_helpers import EXTERNAL_MODEL_ID
from common.core.utils import get_algorithm_roster

router = APIRouter(tags=["dfu-analysis"])

_DFU_ANALYSIS_MODES = {"item_location", "all_items_at_location", "item_at_all_locations"}
_ANALYSIS_REFERENCE_MODELS = {EXTERNAL_MODEL_ID, "champion", "ceiling"}


def _visible_analysis_model_ids() -> set[str]:
    """Configured forecast models plus intentional comparison/reference series."""
    return set(get_algorithm_roster(stage="forecast")) | _ANALYSIS_REFERENCE_MODELS


@router.get("/sku/analysis")
def sku_analysis(
    mode: str = Query(default="item_location", max_length=30),
    item: str = Query(default="", max_length=120),
    location: str = Query(default="", max_length=120),
    points: int = Query(default=36, ge=3, le=120),
    seasonality_profile: str = Query(default="", max_length=120),
):
    """Unified DFU analysis: overlay sales history + multi-model forecasts with KPIs."""
    if mode not in _DFU_ANALYSIS_MODES:
        raise HTTPException(422, f"Invalid mode '{mode}'. Valid: {sorted(_DFU_ANALYSIS_MODES)}")

    item_val = item.strip()
    loc_val = location.strip()
    if mode == "item_location" and (not item_val or not loc_val):
        raise HTTPException(422, "item_location mode requires both item and location")
    if mode == "all_items_at_location" and not loc_val:
        raise HTTPException(422, "all_items_at_location mode requires location")
    if mode == "item_at_all_locations" and not item_val:
        raise HTTPException(422, "item_at_all_locations mode requires item")

    # Build WHERE clause based on mode
    where_parts: list[str] = []
    params: list[Any] = []
    if mode == "item_location":
        where_parts.append("item_id = %s")
        params.append(item_val)
        where_parts.append("loc = %s")
        params.append(loc_val)
    elif mode == "all_items_at_location":
        where_parts.append("loc = %s")
        params.append(loc_val)
    elif mode == "item_at_all_locations":
        where_parts.append("item_id = %s")
        params.append(item_val)

    sp_val = seasonality_profile.strip()
    if sp_val:
        where_parts.append(
            "(item_id, loc) IN (SELECT item_id, loc FROM dim_sku WHERE seasonality_profile = %s)"
        )
        params.append(sp_val)

    where_sql = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    with get_conn() as conn, conn.cursor() as cur:
        visible_model_ids = _visible_analysis_model_ids()
        # 1. Sales measures from agg_sales_monthly
        sales_measures_sql = f"""
            SELECT month_start,
                   SUM(qty_shipped)::double precision AS qty_shipped,
                   SUM(qty_ordered)::double precision AS qty_ordered,
                   SUM(qty)::double precision AS sales_qty
            FROM agg_sales_monthly
            {where_sql}
            GROUP BY 1 ORDER BY 1 ASC
        """
        cur.execute(sales_measures_sql, params)
        shipped_by_month: dict[str, float] = {}
        ordered_by_month: dict[str, float] = {}
        sales_qty_by_month: dict[str, float] = {}
        for row in cur.fetchall():
            month_key = str(row[0])
            shipped_by_month[month_key] = float(row[1] or 0)
            ordered_by_month[month_key] = float(row[2] or 0)
            sales_qty_by_month[month_key] = float(row[3] or 0)

        # 2. Forecast trend from agg_forecast_monthly (all models)
        forecast_sql = f"""
            SELECT month_start, model_id,
                   SUM(basefcst_pref)::double precision AS forecast_value,
                   SUM(tothist_dmd)::double precision AS actual_value
            FROM agg_forecast_monthly
            {where_sql}
            GROUP BY 1, 2 ORDER BY 1 ASC
        """
        cur.execute(forecast_sql, params)
        forecast_by_month: dict[str, dict[str, float]] = {}
        model_monthly_data: dict[str, dict[str, dict[str, float]]] = {}
        model_set: set[str] = set()
        for row in cur.fetchall():
            month_key = str(row[0])
            model_id = str(row[1])
            if model_id not in visible_model_ids:
                continue
            forecast_val = float(row[2] or 0)
            actual_val = float(row[3] or 0)
            model_set.add(model_id)
            if month_key not in forecast_by_month:
                forecast_by_month[month_key] = {}
            forecast_by_month[month_key][model_id] = forecast_val
            if month_key not in model_monthly_data:
                model_monthly_data[month_key] = {}
            model_monthly_data[month_key][model_id] = {"forecast": forecast_val, "actual": actual_val}

        # 2b. Actual demand — deduplicated via single model per DFU
        #     FIX: parameterized query instead of f-string interpolation
        dedup_model = EXTERNAL_MODEL_ID if EXTERNAL_MODEL_ID in model_set else (sorted(model_set)[0] if model_set else None)
        dedup_clause = ""
        dedup_params: list[Any] = []
        if dedup_model:
            dedup_clause = "AND model_id = %s"
            dedup_params = [dedup_model]

        actual_sql = f"""
            SELECT month_start,
                   SUM(tothist_dmd)::double precision AS actual_value
            FROM agg_forecast_monthly
            {where_sql}
              {dedup_clause}
            GROUP BY 1 ORDER BY 1 ASC
        """
        cur.execute(actual_sql, [*params, *dedup_params])
        actual_by_month: dict[str, float] = {}
        for row in cur.fetchall():
            actual_by_month[str(row[0])] = float(row[1] or 0)

        models = sorted(model_set)

        # 2c. Champion's per-month winning model (item_location only).
        #     The champion is selected per DFU per month; agg_forecast_monthly
        #     collapses it to model_id='champion' and loses the source. We read
        #     source_model_id straight from the fact table so the UI can label
        #     the champion line "champion (N-BEATS)" for a single DFU. Only
        #     meaningful for one DFU — across DFUs the source differs per item.
        #     source_mix carries the blend composition for blended champions, so
        #     the tooltip can show "champion (40% NBEATS, 35% LGBM, 25% Chronos)".
        champion_source_by_month: dict[str, str] = {}
        champion_mix_by_month: dict[str, Any] = {}
        if mode == "item_location" and "champion" in model_set:
            cur.execute(
                """
                SELECT startdate::text AS month, source_model_id, source_mix
                FROM fact_external_forecast_monthly
                WHERE item_id = %s AND loc = %s
                  AND model_id = 'champion'
                  AND (source_model_id IS NOT NULL OR source_mix IS NOT NULL)
                GROUP BY startdate, source_model_id, source_mix
                ORDER BY startdate
                """,
                [item_val, loc_val],
            )
            for row in cur.fetchall():
                month_key = str(row[0])
                if row[1] is not None:
                    champion_source_by_month[month_key] = str(row[1])
                if row[2] is not None:
                    # psycopg returns JSONB already parsed to Python list/dict.
                    champion_mix_by_month[month_key] = row[2]

        # 3. Merge into series
        all_months = sorted(
            set(shipped_by_month.keys())
            | set(ordered_by_month.keys())
            | set(actual_by_month.keys())
            | set(forecast_by_month.keys())
            | set(sales_qty_by_month.keys())
        )
        if len(all_months) > points:
            all_months = all_months[-points:]

        series: list[dict[str, Any]] = []
        for month in all_months:
            point: dict[str, Any] = {"month": month}
            if month in shipped_by_month:
                point["qty_shipped"] = shipped_by_month[month]
            if month in ordered_by_month:
                point["qty_ordered"] = ordered_by_month[month]
            if month in sales_qty_by_month:
                point["sales_qty"] = sales_qty_by_month[month]
            if month in actual_by_month:
                point["tothist_dmd"] = actual_by_month[month]
            for model_id in models:
                fcst = forecast_by_month.get(month, {}).get(model_id)
                if fcst is not None:
                    point[f"forecast_{model_id}"] = fcst
            # Per-month champion source model + blend mix (item_location only).
            if month in champion_source_by_month:
                point["champion_source"] = champion_source_by_month[month]
            if month in champion_mix_by_month:
                point["champion_mix"] = champion_mix_by_month[month]
            series.append(point)

        # 4. Build model_monthly for client-side KPI computation
        model_monthly: dict[str, list[dict[str, Any]]] = {}
        for month in sorted(model_monthly_data.keys(), reverse=True):
            for mid, vals in model_monthly_data[month].items():
                if mid not in model_monthly:
                    model_monthly[mid] = []
                model_monthly[mid].append({
                    "month": month,
                    "forecast": vals["forecast"],
                    "actual": vals["actual"],
                })

        # 5. DFU attributes from dim_sku
        dfu_attrs: list[dict[str, Any]] = []
        dfu_where_parts: list[str] = []
        dfu_params: list[Any] = []
        if item_val:
            dfu_where_parts.append("item_id = %s")
            dfu_params.append(item_val)
        if loc_val:
            dfu_where_parts.append("loc = %s")
            dfu_params.append(loc_val)
        if dfu_where_parts:
            dfu_cols = [
                "item_id", "customer_group", "loc", "brand", "brand_desc",
                "abc_vol", "prod_cat_desc", "prod_class_desc", "subclass_desc",
                "prod_subgrp_desc", "size", "brand_size", "bot_type_desc",
                "region", "state_plan", "cnty", "premise", "supergroup",
                "supplier_desc", "producer_desc", "service_lvl_grp",
                "execution_lag", "total_lt", "otc_status", "sales_div",
                "dom_imp_opt", "alcoh_pct", "proof", "grape_vrty_desc",
                "material", "vintage", "cluster_assignment",
                "histstart", "sop_ref",
                "seasonality_profile", "seasonality_strength",
                "is_yearly_seasonal", "peak_month", "trough_month",
                "peak_trough_ratio",
            ]
            dfu_sql = f"""
                SELECT {', '.join(dfu_cols)}
                FROM dim_sku
                WHERE {' AND '.join(dfu_where_parts)}
                ORDER BY item_id, loc
                LIMIT 20
            """
            cur.execute(dfu_sql, dfu_params)
            for row in cur.fetchall():
                dfu_attrs.append({
                    col: (str(val) if val is not None else None)
                    for col, val in zip(dfu_cols, row)
                })

        # U3.5 — the item's human-readable description (dim_item.item_desc) so the
        # Item Analysis breadcrumb can render "185690 — DAMMANN JARDIN BLEU TEA"
        # rather than a bare numeric code. Resolved once per item, not per DFU row.
        item_desc: str | None = None
        if item_val:
            cur.execute(
                "SELECT item_desc FROM dim_item WHERE item_id = %s LIMIT 1",
                [item_val],
            )
            desc_row = cur.fetchone()
            if desc_row and desc_row[0]:
                item_desc = str(desc_row[0])

    # Dominant champion source (most months) — lets the UI label the champion
    # line/legend with a single model name even though the pick varies by month.
    champion_dominant_source: str | None = None
    if champion_source_by_month:
        counts: dict[str, int] = {}
        for src in champion_source_by_month.values():
            counts[src] = counts.get(src, 0) + 1
        champion_dominant_source = max(counts, key=counts.get)

    return {
        "mode": mode,
        "item": item_val,
        "location": loc_val,
        "points": points,
        "models": models,
        "series": series,
        "model_monthly": model_monthly,
        "dfu_attributes": dfu_attrs,
        "item_desc": item_desc,
        "champion_source_by_month": champion_source_by_month,
        "champion_dominant_source": champion_dominant_source,
        "champion_mix_by_month": champion_mix_by_month,
    }
