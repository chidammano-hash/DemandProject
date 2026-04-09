"""SKU feature endpoints — computed demand features for dim_sku.

Serves summary statistics, paginated feature lists, histogram
distributions, and on-demand computation trigger for the SKU Features tab.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.core import get_conn

log = logging.getLogger(__name__)

router = APIRouter(prefix="/sku-features", tags=["sku-features"])

# ── Feature columns from dim_sku (migration 120 + base DDL) ──────────────
_FEATURE_COLS = [
    "demand_cv",
    "demand_mean",
    "demand_std",
    "demand_mad",
    "demand_p50",
    "demand_p90",
    "demand_skewness",
    "demand_kurtosis",
    "zero_demand_months",
    "total_demand_months",
    "intermittency_ratio",
    "iqr_demand",
    "median_demand",
    "min_demand",
    "max_demand",
    "total_demand",
    "trend_slope",
    "trend_slope_norm",
    "trend_r2",
    "trend_pct_change",
    "trend_direction",
    "seasonal_amplitude",
    "seasonal_r2",
    "yoy_correlation",
    "seasonal_index_std",
    "periodicity_strength",
    "adi",
    "cagr",
    "recency_ratio",
    "acceleration",
    "outlier_count",
    "acf_lag12",
    "seasonality_strength",
    "peak_month",
    "trough_month",
    "peak_trough_ratio",
]

# Histogram features — keys match frontend HISTOGRAM_FEATURES constants.
# Maps frontend key -> dim_sku column name.
_HISTOGRAM_FEATURES: dict[str, str] = {
    "cv_demand":        "demand_cv",
    "seasonal_amplitude": "seasonal_amplitude",
    "trend_r2":         "trend_r2",
    "zero_demand_pct":  "intermittency_ratio",
    "adi":              "adi",
    "cagr":             "cagr",
}

# Columns returned per row in the list endpoint
_LIST_SELECT = (
    "sku_ck, item_id, loc, ml_cluster, seasonality_profile, variability_class, "
    "trend_direction, features_computed_ts, "
    + ", ".join(_FEATURE_COLS)
)

# Valid sort columns (prevent SQL injection via allowlist)
_SORTABLE = {
    "item_id", "loc", "ml_cluster", "seasonality_profile", "variability_class",
    "trend_direction", "features_computed_ts",
} | set(_FEATURE_COLS)

_CACHE_SHORT = "public, max-age=120"
_CACHE_MEDIUM = "public, max-age=300"


# ────────────────────────────────────────────────────────────────────────────
# 1. GET /sku-features/summary
# ────────────────────────────────────────────────────────────────────────────
@router.get("/summary")
def sku_features_summary():
    """Aggregate statistics for computed SKU features."""
    with get_conn() as conn, conn.cursor() as cur:
        # Total SKUs with features computed + latest timestamp
        cur.execute("""
            SELECT
                COUNT(*)::bigint,
                MAX(features_computed_ts)
            FROM dim_sku
            WHERE features_computed_ts IS NOT NULL
        """)
        total_row = cur.fetchone()
        total_skus = int(total_row[0]) if total_row else 0
        latest_ts = total_row[1].isoformat() if total_row and total_row[1] else None

        # Distribution by seasonality_profile
        cur.execute("""
            SELECT COALESCE(seasonality_profile, '(unknown)') AS profile,
                   COUNT(*)::bigint AS cnt
            FROM dim_sku
            WHERE features_computed_ts IS NOT NULL
            GROUP BY 1
            ORDER BY cnt DESC
        """)
        seasonality_dist = [
            {"profile": r[0], "count": int(r[1])} for r in cur.fetchall()
        ]

        # Distribution by variability_class
        cur.execute("""
            SELECT COALESCE(variability_class, '(unknown)') AS vclass,
                   COUNT(*)::bigint AS cnt
            FROM dim_sku
            WHERE features_computed_ts IS NOT NULL
            GROUP BY 1
            ORDER BY cnt DESC
        """)
        variability_dist = [
            {"class": r[0], "count": int(r[1])} for r in cur.fetchall()
        ]

        # Distribution by trend_direction
        cur.execute("""
            SELECT COALESCE(trend_direction, 0) AS td,
                   COUNT(*)::bigint AS cnt
            FROM dim_sku
            WHERE features_computed_ts IS NOT NULL
            GROUP BY 1
            ORDER BY td
        """)
        trend_dist = [
            {"direction": int(r[0]), "count": int(r[1])} for r in cur.fetchall()
        ]

        # Average values for key metrics
        cur.execute("""
            SELECT
                AVG(demand_cv)              AS avg_cv_demand,
                AVG(seasonal_amplitude)     AS avg_seasonal_amplitude,
                AVG(trend_r2)               AS avg_trend_r2,
                AVG(intermittency_ratio)    AS avg_zero_demand_pct,
                AVG(adi)                    AS avg_adi,
                AVG(cagr)                   AS avg_cagr,
                AVG(demand_mean)            AS avg_demand_mean,
                AVG(seasonality_strength)   AS avg_seasonality_strength
            FROM dim_sku
            WHERE features_computed_ts IS NOT NULL
        """)
        avgs_row = cur.fetchone()

    def _safe_float(v, decimals: int = 4) -> float | None:
        return round(float(v), decimals) if v is not None else None

    averages = {}
    if avgs_row:
        labels = [
            "cv_demand", "seasonal_amplitude", "trend_r2",
            "zero_demand_pct", "adi", "cagr",
            "demand_mean", "seasonality_strength",
        ]
        for i, label in enumerate(labels):
            averages[label] = _safe_float(avgs_row[i])

    # Convert arrays to Record<string, number> for frontend compatibility
    seasonality_record = {r["profile"]: r["count"] for r in seasonality_dist}
    variability_record = {r["class"]: r["count"] for r in variability_dist}
    trend_record = {str(r["direction"]): r["count"] for r in trend_dist}

    return JSONResponse(
        content={
            "total_skus": total_skus,
            "last_computed": latest_ts,
            "distributions": {
                "seasonality_profile": seasonality_record,
                "variability_class": variability_record,
                "trend_direction": trend_record,
            },
            "averages": averages,
        },
        headers={"Cache-Control": _CACHE_SHORT},
    )


# ────────────────────────────────────────────────────────────────────────────
# 2. GET /sku-features/list
# ────────────────────────────────────────────────────────────────────────────
@router.get("/list")
def sku_features_list(
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    sort_by: str = Query(default="item_id"),
    sort_dir: Literal["asc", "desc"] = Query(default="asc"),
    seasonality_profile: str | None = Query(default=None),
    variability_class: str | None = Query(default=None),
    trend_direction: int | None = Query(default=None),
    search: str | None = Query(default=None, max_length=200),
):
    """Paginated list of SKU features from dim_sku."""
    if sort_by not in _SORTABLE:
        sort_by = "item_id"

    # Build WHERE clause
    conditions = ["features_computed_ts IS NOT NULL"]
    params: list = []

    if seasonality_profile is not None:
        conditions.append("seasonality_profile = %s")
        params.append(seasonality_profile)
    if variability_class is not None:
        conditions.append("variability_class = %s")
        params.append(variability_class)
    if trend_direction is not None:
        conditions.append("trend_direction = %s")
        params.append(trend_direction)
    if search:
        conditions.append("item_id ILIKE %s")
        params.append(f"%{search}%")

    where = " AND ".join(conditions)

    # sort_by is validated against _SORTABLE allowlist above — safe to interpolate
    order = f"{sort_by} {sort_dir} NULLS LAST"

    with get_conn() as conn, conn.cursor() as cur:
        # Total matching count
        cur.execute(f"SELECT COUNT(*)::bigint FROM dim_sku WHERE {where}", params)
        total = int(cur.fetchone()[0])

        # Paginated rows
        cur.execute(
            f"SELECT {_LIST_SELECT} FROM dim_sku WHERE {where} "
            f"ORDER BY {order} LIMIT %s OFFSET %s",
            [*params, limit, offset],
        )
        rows = cur.fetchall()

    # Build response rows
    col_names = [
        "sku_ck", "item_id", "loc", "ml_cluster",
        "seasonality_profile", "variability_class",
        "trend_direction", "features_computed_ts",
        *_FEATURE_COLS,
    ]
    result_rows = []
    for row in rows:
        obj = {}
        for i, col in enumerate(col_names):
            val = row[i]
            if col == "features_computed_ts":
                obj[col] = val.isoformat() if val else None
            elif isinstance(val, (int, float, str, type(None), bool)):
                obj[col] = val
            else:
                # Decimal -> float
                obj[col] = float(val) if val is not None else None
        result_rows.append(obj)

    return JSONResponse(
        content={
            "total": total,
            "limit": limit,
            "offset": offset,
            "rows": result_rows,
        },
        headers={"Cache-Control": _CACHE_SHORT},
    )


# ────────────────────────────────────────────────────────────────────────────
# 3. GET /sku-features/distributions
# ────────────────────────────────────────────────────────────────────────────
@router.get("/distributions")
def sku_features_distributions(
    bins: int = Query(default=20, ge=5, le=100),
):
    """Histogram distributions for key SKU features.

    Returns equal-width bins computed server-side via
    ``width_bucket`` for each feature in _HISTOGRAM_FEATURES.
    """
    distributions: dict = {}

    with get_conn() as conn, conn.cursor() as cur:
        for frontend_key, db_col in _HISTOGRAM_FEATURES.items():
            # Get min/max for the feature
            cur.execute(
                f"SELECT MIN({db_col})::float, MAX({db_col})::float "
                f"FROM dim_sku "
                f"WHERE features_computed_ts IS NOT NULL AND {db_col} IS NOT NULL"
            )
            minmax = cur.fetchone()
            if not minmax or minmax[0] is None or minmax[1] is None:
                distributions[frontend_key] = []
                continue

            feat_min, feat_max = float(minmax[0]), float(minmax[1])

            # Handle edge case where all values are identical
            if feat_min == feat_max:
                cur.execute(
                    f"SELECT COUNT(*)::bigint FROM dim_sku "
                    f"WHERE features_computed_ts IS NOT NULL AND {db_col} IS NOT NULL"
                )
                cnt = int(cur.fetchone()[0])
                distributions[frontend_key] = [
                    {"bin_start": feat_min, "bin_end": feat_max, "count": cnt}
                ]
                continue

            bin_width = (feat_max - feat_min) / bins

            # Use width_bucket for efficient server-side binning
            cur.execute(
                f"SELECT "
                f"  width_bucket({db_col}::float, %s, %s, %s) AS bucket, "
                f"  COUNT(*)::bigint AS cnt "
                f"FROM dim_sku "
                f"WHERE features_computed_ts IS NOT NULL AND {db_col} IS NOT NULL "
                f"GROUP BY bucket ORDER BY bucket",
                [feat_min, feat_max, bins],
            )
            bucket_rows = cur.fetchall()

            histogram = []
            for bucket_id, count in bucket_rows:
                bucket_id = int(bucket_id)
                # width_bucket returns 1..bins; 0 = below min, bins+1 = above max
                if bucket_id <= 0:
                    b_start = feat_min
                    b_end = feat_min + bin_width
                elif bucket_id > bins:
                    b_start = feat_max - bin_width
                    b_end = feat_max
                else:
                    b_start = feat_min + (bucket_id - 1) * bin_width
                    b_end = feat_min + bucket_id * bin_width
                histogram.append({
                    "bin_start": round(b_start, 6),
                    "bin_end": round(b_end, 6),
                    "count": int(count),
                })

            distributions[frontend_key] = histogram

    return JSONResponse(
        content={"bins": bins, "features": distributions},
        headers={"Cache-Control": _CACHE_MEDIUM},
    )


# ────────────────────────────────────────────────────────────────────────────
# 4. POST /sku-features/compute — trigger feature computation job
# ────────────────────────────────────────────────────────────────────────────

class ComputeFeaturesRequest(BaseModel):
    time_window_months: int = Field(default=36, ge=6, le=120)


@router.post("/compute", dependencies=[Depends(require_api_key)])
def compute_sku_features(req: ComputeFeaturesRequest | None = None):
    """Submit a background job to compute all SKU features.

    Returns 202 with a job_id that can be polled via GET /jobs/{job_id}.
    """
    from common.services.job_registry import JobManager

    params = {"time_window_months": req.time_window_months if req else 36}
    mgr = JobManager()
    job_id = mgr.submit_job(
        "compute_sku_features",
        params,
        label="Compute SKU Features",
        triggered_by="api",
    )

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued"},
    )
