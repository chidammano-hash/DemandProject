"""SKU feature persistence — bulk write computed features to dim_sku.

Uses psycopg3 COPY for efficient bulk loading via a temporary staging table,
then merges into dim_sku with a single UPDATE ... FROM statement.
"""

from __future__ import annotations

import io
import logging
from datetime import UTC, datetime

import pandas as pd
import psycopg
from psycopg import sql

logger = logging.getLogger(__name__)

# ── Column mapping ──────────────────────────────────────────────────────────
# Maps feature names (from compute_time_series_features output) to dim_sku
# column names.  Most names match 1:1; only a few need renaming.
#
# pipeline output name -> dim_sku column name
_FEATURE_TO_COLUMNS: dict[str, str] = {
    # Renamed columns
    "mean_demand":      "demand_mean",
    "std_demand":       "demand_std",
    "cv_demand":        "demand_cv",
    "median_demand":    "demand_p50",
    "zero_demand_pct":  "intermittency_ratio",
    "months_available": "total_demand_months",
    # 1:1 columns (pipeline name == dim_sku name)
    "demand_mad":            "demand_mad",
    "demand_p90":            "demand_p90",
    "demand_skewness":       "demand_skewness",
    "demand_kurtosis":       "demand_kurtosis",
    "iqr_demand":            "iqr_demand",
    "min_demand":            "min_demand",
    "max_demand":            "max_demand",
    "total_demand":          "total_demand",
    "trend_slope":           "trend_slope",
    "trend_slope_norm":      "trend_slope_norm",
    "trend_r2":              "trend_r2",
    "trend_pct_change":      "trend_pct_change",
    "trend_direction":       "trend_direction",
    "seasonality_strength":  "seasonality_strength",
    "seasonal_amplitude":    "seasonal_amplitude",
    "seasonal_r2":           "seasonal_r2",
    "yoy_correlation":       "yoy_correlation",
    "seasonal_index_std":    "seasonal_index_std",
    "periodicity_strength":  "periodicity_strength",
    "peak_month":            "peak_month",
    "trough_month":          "trough_month",
    "peak_trough_ratio":     "peak_trough_ratio",
    "adi":                   "adi",
    "cagr":                  "cagr",
    "recency_ratio":         "recency_ratio",
    "acceleration":          "acceleration",
    "outlier_count":         "outlier_count",
    "acf_lag12":             "acf_lag12",
    # Classification labels (TEXT columns)
    "seasonality_profile":   "seasonality_profile",
    "variability_class":     "variability_class",
}

# dim_sku columns that are written (deduplicated, deterministic order)
_DIM_SKU_COLUMNS: list[str] = sorted(set(_FEATURE_TO_COLUMNS.values()))

# Type classification for dim_sku target columns
_TEXT_COLS = {"seasonality_profile", "variability_class"}
_INTEGER_COLS = {"peak_month", "trough_month", "total_demand_months",
                 "outlier_count", "trend_direction"}

# Staging table columns: sku_ck + every dim_sku target column + timestamps
_STAGING_COLS: list[str] = ["sku_ck", *_DIM_SKU_COLUMNS, "demand_profile_ts", "features_computed_ts"]


def _build_staging_df(features_df: pd.DataFrame) -> pd.DataFrame:
    """Build a staging DataFrame mapped to dim_sku column names.

    Reads feature columns from ``features_df`` and maps them to dim_sku
    target column names.  Missing source columns are filled with ``None``.
    A ``demand_profile_ts`` column is appended with the current UTC timestamp.
    """
    now = datetime.now(UTC)
    staging: dict[str, object] = {"sku_ck": features_df["sku_ck"]}

    for feature_name, target_col in _FEATURE_TO_COLUMNS.items():
        if feature_name in features_df.columns:
            staging[target_col] = features_df[feature_name]
        else:
            staging[target_col] = pd.Series([None] * len(features_df), dtype=object)

    staging["demand_profile_ts"] = now
    staging["features_computed_ts"] = now

    staging_df = pd.DataFrame(staging)

    # Ensure column order matches _STAGING_COLS
    return staging_df[_STAGING_COLS]


def _copy_to_temp_table(
    cur: psycopg.Cursor,
    staging_df: pd.DataFrame,
    temp_table: str,
) -> None:
    """Create a temp table and COPY staging data into it using psycopg3 COPY."""
    # Build CREATE TEMP TABLE with text columns (COPY handles type coercion
    # and the UPDATE casts to the target column types)
    col_defs = ", ".join(f"{c} TEXT" for c in _STAGING_COLS)
    cur.execute(
        sql.SQL("CREATE TEMP TABLE {} ({}) ON COMMIT DROP").format(
            sql.Identifier(temp_table),
            sql.SQL(col_defs),
        )
    )

    # Columns that target INTEGER in dim_sku — must serialize as "2" not "2.0"
    int_col_indices = {
        i for i, c in enumerate(_STAGING_COLS)
        if c in _INTEGER_COLS
    }

    # Build tab-separated data for COPY
    buf = io.StringIO()
    for row in staging_df.itertuples(index=False, name=None):
        parts: list[str] = []
        for i, v in enumerate(row):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                parts.append("\\N")
            elif i in int_col_indices:
                parts.append(str(int(float(v))))
            else:
                parts.append(str(v))
        buf.write("\t".join(parts) + "\n")
    buf.seek(0)

    copy_sql = sql.SQL("COPY {} ({}) FROM STDIN").format(
        sql.Identifier(temp_table),
        sql.SQL(", ").join(sql.Identifier(c) for c in _STAGING_COLS),
    )

    with cur.copy(copy_sql) as copy:
        while True:
            chunk = buf.read(65536)
            if not chunk:
                break
            copy.write(chunk.encode("utf-8"))


def write_features_to_dim_sku(
    features_df: pd.DataFrame,
    db_params: dict,
) -> dict:
    """Bulk update ``dim_sku`` with computed SKU features.

    Uses a temporary staging table loaded via psycopg3 COPY, then performs
    a single ``UPDATE ... FROM`` to merge into ``dim_sku``.

    Parameters
    ----------
    features_df:
        DataFrame with ``sku_ck`` column plus feature columns as produced
        by ``compute_all_sku_features()``.
    db_params:
        Connection parameters for psycopg.

    Returns
    -------
    Dict with ``{"updated": <count>}``.
    """
    if features_df.empty or "sku_ck" not in features_df.columns:
        logger.warning("Empty or missing sku_ck — nothing to write.")
        return {"updated": 0}

    staging_df = _build_staging_df(features_df)
    temp_table = "_tmp_sku_features"

    # Build SET clauses — cast TEXT staging columns to match dim_sku types.

    set_parts: list[sql.Composable] = []
    for c in _DIM_SKU_COLUMNS:
        if c in _TEXT_COLS:
            # TEXT — no cast needed
            set_parts.append(
                sql.SQL("{col} = {tmp}.{col}").format(
                    col=sql.Identifier(c),
                    tmp=sql.Identifier(temp_table),
                )
            )
        elif c in _INTEGER_COLS:
            set_parts.append(
                sql.SQL("{col} = {tmp}.{col}::INTEGER").format(
                    col=sql.Identifier(c),
                    tmp=sql.Identifier(temp_table),
                )
            )
        else:
            # NUMERIC (default for all feature columns)
            set_parts.append(
                sql.SQL("{col} = {tmp}.{col}::NUMERIC").format(
                    col=sql.Identifier(c),
                    tmp=sql.Identifier(temp_table),
                )
            )

    # Timestamps
    for ts_col in ("demand_profile_ts", "features_computed_ts"):
        set_parts.append(
            sql.SQL("{col} = {tmp}.{col}::TIMESTAMPTZ").format(
                col=sql.Identifier(ts_col),
                tmp=sql.Identifier(temp_table),
            )
        )
    set_parts.append(sql.SQL("modified_ts = NOW()"))

    update_sql = sql.SQL(
        "UPDATE dim_sku SET {sets} "
        "FROM {tmp} "
        "WHERE dim_sku.sku_ck = {tmp}.sku_ck"
    ).format(
        sets=sql.SQL(", ").join(set_parts),
        tmp=sql.Identifier(temp_table),
    )

    logger.info(
        "Writing features for %d SKUs to dim_sku via COPY + UPDATE.",
        len(staging_df),
    )

    with psycopg.connect(**db_params) as conn:
        with conn.cursor() as cur:
            _copy_to_temp_table(cur, staging_df, temp_table)
            cur.execute(update_sql)
            updated = cur.rowcount
        conn.commit()

    logger.info("Updated %d rows in dim_sku.", updated)
    return {"updated": updated}
