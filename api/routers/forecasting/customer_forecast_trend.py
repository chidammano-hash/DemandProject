"""Exact-lineage Portfolio trend for customer bottom-up blend evidence."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import psycopg
from fastapi import APIRouter, HTTPException, Query
from psycopg import sql

from api.core import get_read_only_conn
from common.services.cache import cached_sync

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/customer-forecast", tags=["customer-forecast"])


def _split_filter_values(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()][:50]


def _customer_blend_trend_filter(
    *,
    item_id: str,
    location_id: str,
    brand: str,
    category: str,
    market: str,
    cluster_assignment: str,
) -> tuple[str, str, list[Any], dict[str, list[str]]]:
    """Build fixed SQL fragments shared by historical and staged evidence."""
    clauses: list[str] = []
    params: list[Any] = []
    applied: dict[str, list[str]] = {}
    joins = ""
    brands = _split_filter_values(brand)
    categories = _split_filter_values(category)
    markets = _split_filter_values(market)
    items = _split_filter_values(item_id)
    locations = _split_filter_values(location_id)
    clusters = _split_filter_values(cluster_assignment)
    if brands or categories:
        joins += " JOIN dim_item i ON i.item_id = f.item_id"
    if markets:
        joins += " JOIN dim_location lo ON lo.location_id = f.loc"
    for key, values, expression in (
        ("brand", brands, "i.brand_name = ANY(%s)"),
        ("category", categories, 'i."class" = ANY(%s)'),
        ("market", markets, "lo.state_id = ANY(%s)"),
        ("item", items, "f.item_id = ANY(%s)"),
        ("location", locations, "f.loc = ANY(%s)"),
    ):
        if values:
            clauses.append(expression)
            params.append(values)
            applied[key] = values
    if clusters:
        clauses.append(
            "EXISTS (SELECT 1 FROM dim_sku sku "
            "WHERE sku.item_id = f.item_id AND sku.loc = f.loc "
            "AND sku.cluster_assignment = ANY(%s))"
        )
        params.append(clusters)
        applied["cluster"] = clusters
    where = " AND " + " AND ".join(clauses) if clauses else ""
    return joins, where, params, applied


def _wape_pct(absolute_error: float, actual_qty: float) -> float | None:
    if actual_qty <= 0:
        return None
    return round(100.0 * absolute_error / actual_qty, 6)


@router.get("/blend/trend")
@cached_sync(ttl=120, group="customer_forecast")
def get_customer_blend_trend(
    run_id: uuid.UUID,
    window: int = Query(default=12, ge=1, le=24),
    item_id: str = Query(default=""),
    location_id: str = Query(default=""),
    brand: str = Query(default=""),
    category: str = Query(default=""),
    market: str = Query(default=""),
    channel: str = Query(default=""),
    cluster_assignment: str = Query(default=""),
) -> dict[str, Any]:
    """Return exact-lineage customer backtest history and staged future totals."""
    joins, where_extra, filter_params, applied_filters = _customer_blend_trend_filter(
        item_id=item_id,
        location_id=location_id,
        brand=brand,
        category=category,
        market=market,
        cluster_assignment=cluster_assignment,
    )
    try:
        with get_read_only_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT run_status, forecast_month_generated, completed_at, metadata
                   FROM forecast_generation_run
                   WHERE run_id = %s::uuid
                     AND metadata ? 'customer_bottom_up_blend'
                     AND run_status IN ('ready', 'promoted')""",
                (str(run_id),),
            )
            manifest = cur.fetchone()
            if manifest is None:
                raise HTTPException(status_code=404, detail="Customer blend run not found")
            metadata = dict(manifest[3]) if isinstance(manifest[3], dict) else {}
            lineage = metadata.get("customer_bottom_up_blend")
            lineage = lineage if isinstance(lineage, dict) else {}
            backtest_run_id = lineage.get("backtest_run_id")
            bottom_up_staging_run_id = lineage.get("bottom_up_staging_run_id")
            if not backtest_run_id or not bottom_up_staging_run_id:
                raise HTTPException(
                    status_code=409,
                    detail="Customer blend staging lineage is incomplete",
                )
            planning_month = manifest[1]
            trend_sql = sql.SQL(
                """WITH historical AS (
                         SELECT f.forecast_month,
                                'backtest'::text AS phase,
                                SUM(f.actual_qty) FILTER (
                                    WHERE f.normalized_customer_qty IS NOT NULL
                                )::numeric AS actual_qty,
                                SUM(f.normalized_customer_qty)::numeric
                                    AS customer_bottom_up_qty,
                                SUM(f.champion_qty) FILTER (
                                    WHERE f.normalized_customer_qty IS NOT NULL
                                )::numeric AS source_champion_qty,
                                SUM(f.blended_qty) FILTER (
                                    WHERE f.normalized_customer_qty IS NOT NULL
                                )::numeric AS customer_blend_qty,
                                NULL::numeric AS lower_bound,
                                NULL::numeric AS upper_bound,
                                COUNT(*) FILTER (
                                    WHERE f.coverage_status = 'blended'
                                      AND f.normalized_customer_qty IS NOT NULL
                                )::integer AS blended_dfu_count,
                                COUNT(*) FILTER (
                                    WHERE f.coverage_status = 'champion_fallback'
                                      AND f.normalized_customer_qty IS NOT NULL
                                )::integer AS fallback_dfu_count,
                                SUM(f.actual_qty) FILTER (
                                    WHERE f.normalized_customer_qty IS NOT NULL
                                )::numeric AS common_actual_qty,
                                SUM(ABS(f.normalized_customer_qty - f.actual_qty)) FILTER (
                                    WHERE f.normalized_customer_qty IS NOT NULL
                                )::numeric AS customer_absolute_error,
                                SUM(ABS(f.champion_qty - f.actual_qty)) FILTER (
                                    WHERE f.normalized_customer_qty IS NOT NULL
                                )::numeric AS champion_absolute_error,
                                SUM(ABS(f.blended_qty - f.actual_qty)) FILTER (
                                    WHERE f.normalized_customer_qty IS NOT NULL
                                )::numeric AS blend_absolute_error
                         FROM customer_bottom_up_backtest_component f
                         {joins}
                         WHERE f.backtest_run_id = %s::uuid
                           AND f.forecast_month >= %s::date
                               - (%s * INTERVAL '1 month')
                           AND f.forecast_month < %s::date
                           {where_extra}
                         GROUP BY f.forecast_month
                     ), staged AS (
                         SELECT f.forecast_month,
                                'staged'::text AS phase,
                                NULL::numeric AS actual_qty,
                                SUM(bottom_up.forecast_qty)::numeric
                                    AS customer_bottom_up_qty,
                                SUM(f.champion_qty)::numeric AS source_champion_qty,
                                SUM(blend.forecast_qty)::numeric AS customer_blend_qty,
                                SUM(blend.forecast_qty_lower)::numeric AS lower_bound,
                                SUM(blend.forecast_qty_upper)::numeric AS upper_bound,
                                COUNT(*) FILTER (
                                    WHERE f.coverage_status = 'blended'
                                )::integer AS blended_dfu_count,
                                COUNT(*) FILTER (
                                    WHERE f.coverage_status = 'champion_fallback'
                                )::integer AS fallback_dfu_count,
                                NULL::numeric AS common_actual_qty,
                                NULL::numeric AS customer_absolute_error,
                                NULL::numeric AS champion_absolute_error,
                                NULL::numeric AS blend_absolute_error
                         FROM customer_bottom_up_blend_component f
                         JOIN fact_production_forecast_staging blend
                           ON blend.run_id = f.run_id
                          AND blend.generation_purpose = 'release_candidate'
                          AND blend.candidate_model_id = 'champion'
                          AND blend.model_id = 'customer_bottom_up_blend'
                          AND blend.item_id = f.item_id
                          AND blend.loc = f.loc
                          AND blend.forecast_month = f.forecast_month
                         LEFT JOIN fact_production_forecast_staging bottom_up
                           ON bottom_up.run_id = %s::uuid
                          AND bottom_up.generation_purpose = 'shadow_candidate'
                          AND bottom_up.candidate_model_id = 'customer_bottom_up'
                          AND bottom_up.model_id = 'customer_bottom_up'
                          AND bottom_up.item_id = f.item_id
                          AND bottom_up.loc = f.loc
                          AND bottom_up.forecast_month = f.forecast_month
                         {joins}
                         WHERE f.run_id = %s::uuid
                           AND f.forecast_month >= %s::date
                           AND f.forecast_month < %s::date
                               + (%s * INTERVAL '1 month')
                           {where_extra}
                         GROUP BY f.forecast_month
                     )
                     SELECT * FROM historical
                     UNION ALL
                     SELECT * FROM staged
                     ORDER BY forecast_month, phase"""
            ).format(
                joins=sql.SQL(joins),
                where_extra=sql.SQL(where_extra),
            )
            cur.execute(
                trend_sql,
                (
                    str(backtest_run_id),
                    planning_month,
                    window,
                    planning_month,
                    *filter_params,
                    str(bottom_up_staging_run_id),
                    str(run_id),
                    planning_month,
                    planning_month,
                    window,
                    *filter_params,
                ),
            )
            rows = cur.fetchall()
    except HTTPException:
        raise
    except psycopg.Error as exc:
        logger.exception("Loading customer blend portfolio trend failed")
        raise HTTPException(status_code=500, detail="customer blend trend lookup failed") from exc

    common_actual = sum(float(row[10] or 0) for row in rows)
    customer_error = sum(float(row[11] or 0) for row in rows)
    champion_error = sum(float(row[12] or 0) for row in rows)
    blend_error = sum(float(row[13] or 0) for row in rows)
    months = [
        {
            "month": row[0].isoformat(),
            "phase": row[1],
            "actual_qty": float(row[2]) if row[2] is not None else None,
            "customer_bottom_up_qty": float(row[3]) if row[3] is not None else None,
            "source_champion_qty": float(row[4]) if row[4] is not None else None,
            "customer_blend_qty": float(row[5]) if row[5] is not None else None,
            "lower_bound": float(row[6]) if row[6] is not None else None,
            "upper_bound": float(row[7]) if row[7] is not None else None,
            "blended_dfu_count": int(row[8] or 0),
            "fallback_dfu_count": int(row[9] or 0),
        }
        for row in rows
    ]
    staged_months = [month for month in months if month["phase"] == "staged"]
    filter_notes = []
    if channel.strip():
        filter_notes.append("Channel does not apply to warehouse-item customer blend comparisons.")
    return {
        "run_id": str(run_id),
        "status": manifest[0],
        "planning_month": planning_month.isoformat(),
        "completed_at": manifest[2].isoformat() if manifest[2] else None,
        "backtest_run_id": str(backtest_run_id),
        "bottom_up_staging_run_id": str(bottom_up_staging_run_id),
        "backtest_gate": lineage.get("backtest_gate"),
        "filters_applied": applied_filters,
        "filter_notes": filter_notes,
        "accuracy": {
            "common_actual_qty": common_actual,
            "customer_bottom_up_wape_pct": _wape_pct(customer_error, common_actual),
            "source_champion_wape_pct": _wape_pct(champion_error, common_actual),
            "customer_blend_wape_pct": _wape_pct(blend_error, common_actual),
        },
        "coverage": {
            "blended_rows": sum(int(month["blended_dfu_count"]) for month in staged_months),
            "champion_fallback_rows": sum(
                int(month["fallback_dfu_count"]) for month in staged_months
            ),
            "global_customer_only_excluded_count": int(
                lineage.get("excluded_customer_dfu_count") or 0
            ),
        },
        "months": months,
    }
