"""Cluster EDA (Exploratory Data Analysis) endpoints.

Provides demand profiling, cluster statistics, error concentration analysis,
and residual diagnostics for the LGBM tuning UI.
"""
from __future__ import annotations

import logging
from typing import Any

import psycopg
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse

from api.core import get_conn, set_cache, _f

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cluster-eda"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_query(cur: Any, sql: str, params: list[Any] | None = None) -> list[tuple]:
    """Execute a query, returning [] if the underlying table does not exist."""
    try:
        cur.execute(sql, params or [])
        return cur.fetchall()
    except psycopg.errors.UndefinedTable:
        cur.connection.rollback()
        return []
    except psycopg.errors.UndefinedColumn:
        cur.connection.rollback()
        return []


def _safe_fetchone(cur: Any, sql: str, params: list[Any] | None = None) -> tuple | None:
    """Execute a query expecting a single row, returning None on missing table."""
    try:
        cur.execute(sql, params or [])
        return cur.fetchone()
    except psycopg.errors.UndefinedTable:
        cur.connection.rollback()
        return None
    except psycopg.errors.UndefinedColumn:
        cur.connection.rollback()
        return None


# ---------------------------------------------------------------------------
# 1. GET /cluster-eda/profile — Cluster Profile Matrix
# ---------------------------------------------------------------------------

@router.get("/cluster-eda/profile")
def cluster_profile(response: FastAPIResponse):
    """Per-cluster demand statistics including CV, zero-pct, and accuracy."""
    set_cache(response, max_age=120)

    profile_sql = """
        SELECT
            s.ml_cluster,
            COUNT(DISTINCT s.sku_ck)                                          AS n_dfus,
            ROUND(AVG(CASE WHEN f.tothist_dmd > 0
                            THEN f.tothist_dmd END)::numeric, 1)             AS mean_demand,
            ROUND((STDDEV(f.tothist_dmd)
                   / NULLIF(AVG(f.tothist_dmd), 0))::numeric, 2)            AS cv_demand,
            ROUND((SUM(CASE WHEN f.tothist_dmd = 0 THEN 1 ELSE 0 END)::float
                   / NULLIF(COUNT(*), 0))::numeric, 3)                       AS zero_pct,
            ROUND(AVG(f.tothist_dmd)::numeric, 1)                            AS overall_mean,
            ROUND(STDDEV(f.tothist_dmd)::numeric, 1)                         AS demand_std
        FROM dim_sku s
        JOIN fact_sales_monthly f ON s.sku_ck = f.sku_ck
        WHERE s.ml_cluster IS NOT NULL
        GROUP BY s.ml_cluster
        ORDER BY s.ml_cluster
    """

    accuracy_sql = """
        SELECT
            s.ml_cluster,
            ROUND((100 - 100.0 * SUM(ABS(ef.basefcst_pref - f.tothist_dmd))
                   / NULLIF(ABS(SUM(f.tothist_dmd)), 0))::numeric, 2)       AS accuracy_pct,
            ROUND((SUM(ABS(ef.basefcst_pref - f.tothist_dmd))::float
                   / NULLIF(ABS(SUM(f.tothist_dmd)), 0))::numeric, 4)       AS wape
        FROM dim_sku s
        JOIN fact_sales_monthly f ON s.sku_ck = f.sku_ck
        JOIN fact_external_forecast_monthly ef
             ON ef.sku_ck = f.sku_ck AND ef.actual_month = f.month_start
        WHERE s.ml_cluster IS NOT NULL
          AND ef.lag = 0
        GROUP BY s.ml_cluster
        ORDER BY s.ml_cluster
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            profile_rows = _safe_query(cur, profile_sql)
            accuracy_rows = _safe_query(cur, accuracy_sql)
    except psycopg.Error:
        logger.exception("Failed to fetch cluster profile")
        raise HTTPException(status_code=500, detail="Failed to fetch cluster profile")

    # Index accuracy by cluster
    acc_map: dict[Any, dict[str, float | None]] = {}
    for r in accuracy_rows:
        acc_map[r[0]] = {"accuracy_pct": _f(r[1]), "wape": _f(r[2])}

    clusters = []
    for r in profile_rows:
        cluster_id = r[0]
        acc = acc_map.get(cluster_id, {})
        clusters.append({
            "ml_cluster": cluster_id,
            "n_dfus": int(r[1]) if r[1] is not None else 0,
            "mean_demand": _f(r[2]),
            "cv_demand": _f(r[3]),
            "zero_pct": _f(r[4]),
            "overall_mean": _f(r[5]),
            "demand_std": _f(r[6]),
            "accuracy_pct": acc.get("accuracy_pct"),
            "wape": acc.get("wape"),
        })

    return {
        "clusters": clusters,
        "warning": "No data available" if not clusters else None,
    }


# ---------------------------------------------------------------------------
# 2. GET /cluster-eda/error-concentration — Error Concentration Analysis
# ---------------------------------------------------------------------------

@router.get("/cluster-eda/error-concentration")
def error_concentration(response: FastAPIResponse):
    """Show where forecast errors concentrate: top DFUs, by month, by cluster, by ABC."""
    set_cache(response, max_age=120)

    # Top error DFUs — cumulative share of total absolute error
    top_error_sql = """
        WITH dfu_errors AS (
            SELECT
                ef.sku_ck,
                SUM(ABS(ef.basefcst_pref - f.tothist_dmd)) AS abs_error
            FROM fact_external_forecast_monthly ef
            JOIN fact_sales_monthly f
                 ON ef.sku_ck = f.sku_ck AND ef.actual_month = f.month_start
            WHERE ef.lag = 0
            GROUP BY ef.sku_ck
        ),
        ranked AS (
            SELECT
                abs_error,
                SUM(abs_error) OVER () AS total_error,
                ROW_NUMBER() OVER (ORDER BY abs_error DESC) AS rn,
                COUNT(*) OVER () AS n_total
            FROM dfu_errors
        )
        SELECT
            ROUND((SUM(CASE WHEN rn <= CEIL(n_total * 0.10)
                             THEN abs_error ELSE 0 END)
                   / NULLIF(MAX(total_error), 0))::numeric, 4) AS top_10pct_share,
            ROUND((SUM(CASE WHEN rn <= CEIL(n_total * 0.20)
                             THEN abs_error ELSE 0 END)
                   / NULLIF(MAX(total_error), 0))::numeric, 4) AS top_20pct_share
        FROM ranked
    """

    # Error by month
    error_by_month_sql = """
        SELECT
            EXTRACT(MONTH FROM f.month_start)::int AS month_num,
            ROUND((SUM(ABS(ef.basefcst_pref - f.tothist_dmd))::float
                   / NULLIF(ABS(SUM(f.tothist_dmd)), 0))::numeric, 2) AS wape,
            ROUND(((SUM(ef.basefcst_pref)::float
                    / NULLIF(SUM(f.tothist_dmd), 0)) - 1)::numeric, 4) AS bias
        FROM fact_external_forecast_monthly ef
        JOIN fact_sales_monthly f
             ON ef.sku_ck = f.sku_ck AND ef.actual_month = f.month_start
        WHERE ef.lag = 0
        GROUP BY EXTRACT(MONTH FROM f.month_start)
        ORDER BY month_num
    """

    # Error by cluster
    error_by_cluster_sql = """
        WITH cluster_errs AS (
            SELECT
                s.ml_cluster,
                SUM(ABS(ef.basefcst_pref - f.tothist_dmd)) AS abs_err,
                SUM(f.tothist_dmd) AS total_actual,
                SUM(ef.basefcst_pref) AS total_fcst
            FROM fact_external_forecast_monthly ef
            JOIN fact_sales_monthly f
                 ON ef.sku_ck = f.sku_ck AND ef.actual_month = f.month_start
            JOIN dim_sku s ON s.sku_ck = f.sku_ck
            WHERE ef.lag = 0 AND s.ml_cluster IS NOT NULL
            GROUP BY s.ml_cluster
        ),
        totals AS (
            SELECT SUM(abs_err) AS grand_total FROM cluster_errs
        )
        SELECT
            ce.ml_cluster,
            ROUND((ce.abs_err::float / NULLIF(ABS(ce.total_actual), 0))::numeric, 2) AS wape,
            ROUND(((ce.total_fcst::float / NULLIF(ce.total_actual, 0)) - 1)::numeric, 4) AS bias,
            ROUND((ce.abs_err::float / NULLIF(t.grand_total, 0))::numeric, 4) AS share_of_total_error
        FROM cluster_errs ce
        CROSS JOIN totals t
        ORDER BY ce.ml_cluster
    """

    # Error by ABC class
    error_by_abc_sql = """
        SELECT
            COALESCE(s.abc_class, 'Unknown') AS abc_class,
            ROUND((SUM(ABS(ef.basefcst_pref - f.tothist_dmd))::float
                   / NULLIF(ABS(SUM(f.tothist_dmd)), 0))::numeric, 2) AS wape,
            ROUND(((SUM(ef.basefcst_pref)::float
                    / NULLIF(SUM(f.tothist_dmd), 0)) - 1)::numeric, 4) AS bias
        FROM fact_external_forecast_monthly ef
        JOIN fact_sales_monthly f
             ON ef.sku_ck = f.sku_ck AND ef.actual_month = f.month_start
        JOIN dim_sku s ON s.sku_ck = f.sku_ck
        WHERE ef.lag = 0
        GROUP BY COALESCE(s.abc_class, 'Unknown')
        ORDER BY abc_class
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            top_rows = _safe_query(cur, top_error_sql)
            month_rows = _safe_query(cur, error_by_month_sql)
            cluster_rows = _safe_query(cur, error_by_cluster_sql)
            abc_rows = _safe_query(cur, error_by_abc_sql)
    except psycopg.Error:
        logger.exception("Failed to fetch error concentration")
        raise HTTPException(status_code=500, detail="Failed to fetch error concentration")

    # Build top_error_dfus
    if top_rows and top_rows[0][0] is not None:
        top_error_dfus = {
            "top_10pct_share": _f(top_rows[0][0]),
            "top_20pct_share": _f(top_rows[0][1]),
        }
    else:
        top_error_dfus = {"top_10pct_share": None, "top_20pct_share": None}

    error_by_month = [
        {"month": r[0], "wape": _f(r[1]), "bias": _f(r[2])}
        for r in month_rows
    ]

    error_by_cluster = [
        {
            "cluster": r[0],
            "wape": _f(r[1]),
            "bias": _f(r[2]),
            "share_of_total_error": _f(r[3]),
        }
        for r in cluster_rows
    ]

    error_by_abc = [
        {"abc_class": r[0], "wape": _f(r[1]), "bias": _f(r[2])}
        for r in abc_rows
    ]

    return {
        "top_error_dfus": top_error_dfus,
        "error_by_month": error_by_month,
        "error_by_cluster": error_by_cluster,
        "error_by_abc": error_by_abc,
        "warning": "No data available" if not month_rows else None,
    }


# ---------------------------------------------------------------------------
# 3. GET /cluster-eda/demand-distribution/{cluster_id}
# ---------------------------------------------------------------------------

@router.get("/cluster-eda/demand-distribution/{cluster_id}")
def demand_distribution(cluster_id: int, response: FastAPIResponse):
    """Per-cluster demand distribution — histogram, percentiles, top DFUs."""
    set_cache(response, max_age=120)

    count_sql = """
        SELECT COUNT(DISTINCT s.sku_ck)
        FROM dim_sku s
        WHERE s.ml_cluster = %s
    """

    histogram_sql = """
        SELECT
            CASE
                WHEN f.tothist_dmd = 0 THEN '0'
                WHEN f.tothist_dmd BETWEEN 1 AND 10 THEN '1-10'
                WHEN f.tothist_dmd BETWEEN 11 AND 50 THEN '11-50'
                WHEN f.tothist_dmd BETWEEN 51 AND 100 THEN '51-100'
                WHEN f.tothist_dmd BETWEEN 101 AND 500 THEN '101-500'
                WHEN f.tothist_dmd BETWEEN 501 AND 1000 THEN '501-1000'
                ELSE '1000+'
            END AS bucket,
            COUNT(*) AS cnt
        FROM fact_sales_monthly f
        JOIN dim_sku s ON s.sku_ck = f.sku_ck
        WHERE s.ml_cluster = %s
        GROUP BY bucket
        ORDER BY MIN(f.tothist_dmd)
    """

    percentile_sql = """
        SELECT
            ROUND(PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY f.tothist_dmd)::numeric, 1),
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY f.tothist_dmd)::numeric, 1),
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY f.tothist_dmd)::numeric, 1),
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY f.tothist_dmd)::numeric, 1),
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY f.tothist_dmd)::numeric, 1),
            ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY f.tothist_dmd)::numeric, 1)
        FROM fact_sales_monthly f
        JOIN dim_sku s ON s.sku_ck = f.sku_ck
        WHERE s.ml_cluster = %s
    """

    top_dfus_sql = """
        SELECT
            f.sku_ck,
            ROUND(AVG(f.tothist_dmd)::numeric, 1)                            AS mean_demand,
            ROUND((STDDEV(f.tothist_dmd)
                   / NULLIF(AVG(f.tothist_dmd), 0))::numeric, 2)            AS cv
        FROM fact_sales_monthly f
        JOIN dim_sku s ON s.sku_ck = f.sku_ck
        WHERE s.ml_cluster = %s
        GROUP BY f.sku_ck
        ORDER BY AVG(f.tothist_dmd) DESC
        LIMIT 20
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            count_row = _safe_fetchone(cur, count_sql, [cluster_id])
            hist_rows = _safe_query(cur, histogram_sql, [cluster_id])
            pct_row = _safe_fetchone(cur, percentile_sql, [cluster_id])
            top_rows = _safe_query(cur, top_dfus_sql, [cluster_id])
    except psycopg.Error:
        logger.exception("Failed to fetch demand distribution for cluster %s", cluster_id)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch demand distribution for cluster {cluster_id}",
        )

    n_dfus = int(count_row[0]) if count_row and count_row[0] is not None else 0

    histogram = [
        {"bucket": r[0], "count": int(r[1])}
        for r in hist_rows
    ]

    if pct_row and pct_row[0] is not None:
        percentiles = {
            "p10": _f(pct_row[0]),
            "p25": _f(pct_row[1]),
            "p50": _f(pct_row[2]),
            "p75": _f(pct_row[3]),
            "p90": _f(pct_row[4]),
            "p99": _f(pct_row[5]),
        }
    else:
        percentiles = {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "p99": None}

    top_dfus = [
        {"sku_ck": r[0], "mean_demand": _f(r[1]), "cv": _f(r[2])}
        for r in top_rows
    ]

    return {
        "cluster_id": cluster_id,
        "n_dfus": n_dfus,
        "histogram": histogram,
        "percentiles": percentiles,
        "top_dfus": top_dfus,
        "warning": "No data available" if n_dfus == 0 else None,
    }


# ---------------------------------------------------------------------------
# 4. GET /cluster-eda/residual-analysis — Residual Analysis
# ---------------------------------------------------------------------------

@router.get("/cluster-eda/residual-analysis")
def residual_analysis(
    response: FastAPIResponse,
    model_id: str = Query(default="lgbm_cluster", max_length=120),
    cluster_id: int | None = Query(default=None),
):
    """Residual diagnostics: distribution stats, by horizon, worst DFUs, bias by cluster."""
    set_cache(response, max_age=120)

    use_cluster = cluster_id is not None

    # -- Static SQL variants (no f-string interpolation) ----------------------
    _STATS_BASE = """
        SELECT
            ROUND(AVG(ef.basefcst_pref - f.tothist_dmd)::numeric, 2)                   AS mean_err,
            ROUND(STDDEV(ef.basefcst_pref - f.tothist_dmd)::numeric, 2)                AS std_err,
            ROUND((
                (COUNT(*) * SUM(POWER(ef.basefcst_pref - f.tothist_dmd, 3)))
                / NULLIF(POWER(NULLIF(COUNT(*), 0) * NULLIF(SUM(POWER(ef.basefcst_pref - f.tothist_dmd, 2)), 0), 0), 0)
            )::numeric, 2)                                                               AS skew_approx,
            ROUND((
                (COUNT(*) * SUM(POWER(ef.basefcst_pref - f.tothist_dmd, 4)))
                / NULLIF(POWER(NULLIF(SUM(POWER(ef.basefcst_pref - f.tothist_dmd, 2)), 0), 2), 0)
            )::numeric, 2)                                                               AS kurtosis_approx
        FROM fact_external_forecast_monthly ef
        JOIN fact_sales_monthly f
             ON ef.sku_ck = f.sku_ck AND ef.actual_month = f.month_start
        JOIN dim_sku s ON s.sku_ck = f.sku_ck
        WHERE ef.model_id = %s
          AND ef.lag = 0
    """
    _STATS_CLUSTER = _STATS_BASE + " AND s.ml_cluster = %s"

    _HORIZON_BASE = """
        SELECT
            ef.lag,
            ROUND(AVG(ef.basefcst_pref - f.tothist_dmd)::numeric, 2)                   AS mean_error,
            ROUND(SQRT(AVG(POWER(ef.basefcst_pref - f.tothist_dmd, 2)))::numeric, 2)   AS rmse
        FROM fact_external_forecast_monthly ef
        JOIN fact_sales_monthly f
             ON ef.sku_ck = f.sku_ck AND ef.actual_month = f.month_start
        JOIN dim_sku s ON s.sku_ck = f.sku_ck
        WHERE ef.model_id = %s
        GROUP BY ef.lag
        ORDER BY ef.lag
    """
    _HORIZON_CLUSTER = """
        SELECT
            ef.lag,
            ROUND(AVG(ef.basefcst_pref - f.tothist_dmd)::numeric, 2)                   AS mean_error,
            ROUND(SQRT(AVG(POWER(ef.basefcst_pref - f.tothist_dmd, 2)))::numeric, 2)   AS rmse
        FROM fact_external_forecast_monthly ef
        JOIN fact_sales_monthly f
             ON ef.sku_ck = f.sku_ck AND ef.actual_month = f.month_start
        JOIN dim_sku s ON s.sku_ck = f.sku_ck
        WHERE ef.model_id = %s AND s.ml_cluster = %s
        GROUP BY ef.lag
        ORDER BY ef.lag
    """

    _WORST_BASE = """
        SELECT
            ef.sku_ck,
            ROUND(AVG(ABS(ef.basefcst_pref - f.tothist_dmd))::numeric, 2) AS mean_abs_error,
            ROUND(((SUM(ef.basefcst_pref)::float
                    / NULLIF(SUM(f.tothist_dmd), 0)) - 1)::numeric, 4)    AS bias,
            s.ml_cluster
        FROM fact_external_forecast_monthly ef
        JOIN fact_sales_monthly f
             ON ef.sku_ck = f.sku_ck AND ef.actual_month = f.month_start
        JOIN dim_sku s ON s.sku_ck = f.sku_ck
        WHERE ef.model_id = %s
          AND ef.lag = 0
        GROUP BY ef.sku_ck, s.ml_cluster
        ORDER BY AVG(ABS(ef.basefcst_pref - f.tothist_dmd)) DESC
        LIMIT 20
    """
    _WORST_CLUSTER = """
        SELECT
            ef.sku_ck,
            ROUND(AVG(ABS(ef.basefcst_pref - f.tothist_dmd))::numeric, 2) AS mean_abs_error,
            ROUND(((SUM(ef.basefcst_pref)::float
                    / NULLIF(SUM(f.tothist_dmd), 0)) - 1)::numeric, 4)    AS bias,
            s.ml_cluster
        FROM fact_external_forecast_monthly ef
        JOIN fact_sales_monthly f
             ON ef.sku_ck = f.sku_ck AND ef.actual_month = f.month_start
        JOIN dim_sku s ON s.sku_ck = f.sku_ck
        WHERE ef.model_id = %s
          AND ef.lag = 0
          AND s.ml_cluster = %s
        GROUP BY ef.sku_ck, s.ml_cluster
        ORDER BY AVG(ABS(ef.basefcst_pref - f.tothist_dmd)) DESC
        LIMIT 20
    """

    stats_sql = _STATS_CLUSTER if use_cluster else _STATS_BASE
    horizon_sql = _HORIZON_CLUSTER if use_cluster else _HORIZON_BASE
    worst_sql = _WORST_CLUSTER if use_cluster else _WORST_BASE
    params: list[Any] = [model_id, cluster_id] if use_cluster else [model_id]

    # Bias by cluster
    bias_sql = """
        SELECT
            s.ml_cluster,
            ROUND(((SUM(ef.basefcst_pref)::float
                    / NULLIF(SUM(f.tothist_dmd), 0)) - 1)::numeric, 4) AS bias
        FROM fact_external_forecast_monthly ef
        JOIN fact_sales_monthly f
             ON ef.sku_ck = f.sku_ck AND ef.actual_month = f.month_start
        JOIN dim_sku s ON s.sku_ck = f.sku_ck
        WHERE ef.model_id = %s
          AND ef.lag = 0
          AND s.ml_cluster IS NOT NULL
        GROUP BY s.ml_cluster
        ORDER BY s.ml_cluster
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            stats_row = _safe_fetchone(cur, stats_sql, params)
            horizon_rows = _safe_query(cur, horizon_sql, params)
            worst_rows = _safe_query(cur, worst_sql, params)
            bias_rows = _safe_query(cur, bias_sql, [model_id])
    except psycopg.Error:
        logger.exception("Failed to fetch residual analysis")
        raise HTTPException(status_code=500, detail="Failed to fetch residual analysis")

    if stats_row and stats_row[0] is not None:
        residual_stats = {
            "mean": _f(stats_row[0]),
            "std": _f(stats_row[1]),
            "skew": _f(stats_row[2]),
            "kurtosis": _f(stats_row[3]),
        }
    else:
        residual_stats = {"mean": None, "std": None, "skew": None, "kurtosis": None}

    residual_by_horizon = [
        {"lag": int(r[0]), "mean_error": _f(r[1]), "rmse": _f(r[2])}
        for r in horizon_rows
    ]

    worst_dfus = [
        {
            "sku_ck": r[0],
            "mean_abs_error": _f(r[1]),
            "bias": _f(r[2]),
            "cluster": r[3],
        }
        for r in worst_rows
    ]

    bias_by_cluster = [
        {
            "cluster": r[0],
            "bias": _f(r[1]),
            "direction": "over" if r[1] is not None and float(r[1]) > 0 else "under",
        }
        for r in bias_rows
    ]

    return {
        "residual_stats": residual_stats,
        "residual_by_horizon": residual_by_horizon,
        "worst_dfus": worst_dfus,
        "bias_by_cluster": bias_by_cluster,
        "warning": "No data available" if not horizon_rows else None,
    }


# ---------------------------------------------------------------------------
# 5. GET /cluster-eda/seasonality-heatmap — Seasonality patterns per cluster
# ---------------------------------------------------------------------------

@router.get("/cluster-eda/seasonality-heatmap")
def seasonality_heatmap(response: FastAPIResponse):
    """Month x cluster WAPE matrix for seasonality pattern analysis."""
    set_cache(response, max_age=300)

    sql = """
        SELECT
            s.ml_cluster,
            EXTRACT(MONTH FROM f.month_start)::int AS month_num,
            ROUND((SUM(ABS(ef.basefcst_pref - f.tothist_dmd))::float
                   / NULLIF(ABS(SUM(f.tothist_dmd)), 0))::numeric, 2) AS wape
        FROM fact_external_forecast_monthly ef
        JOIN fact_sales_monthly f
             ON ef.sku_ck = f.sku_ck AND ef.actual_month = f.month_start
        JOIN dim_sku s ON s.sku_ck = f.sku_ck
        WHERE ef.lag = 0
          AND s.ml_cluster IS NOT NULL
        GROUP BY s.ml_cluster, EXTRACT(MONTH FROM f.month_start)
        ORDER BY s.ml_cluster, month_num
    """

    try:
        with get_conn() as conn, conn.cursor() as cur:
            rows = _safe_query(cur, sql)
    except psycopg.Error:
        logger.exception("Failed to fetch seasonality heatmap")
        raise HTTPException(status_code=500, detail="Failed to fetch seasonality heatmap")

    if not rows:
        return {
            "clusters": [],
            "months": list(range(1, 13)),
            "values": [],
            "warning": "No data available",
        }

    # Build cluster -> month -> wape mapping
    cluster_month_map: dict[Any, dict[int, float | None]] = {}
    for r in rows:
        cl = r[0]
        mo = r[1]
        wape = _f(r[2])
        if cl not in cluster_month_map:
            cluster_month_map[cl] = {}
        cluster_month_map[cl][mo] = wape

    clusters = sorted(cluster_month_map.keys())
    months = list(range(1, 13))

    values = []
    for cl in clusters:
        row_vals = [cluster_month_map[cl].get(m) for m in months]
        values.append(row_vals)

    return {
        "clusters": clusters,
        "months": months,
        "values": values,
    }
