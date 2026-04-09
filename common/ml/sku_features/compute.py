"""SKU feature computation — main orchestration module.

Loads monthly sales data from the database and computes time-series features
for every SKU using the shared feature library from ``common.ml.clustering.features``.
"""

from __future__ import annotations

import logging
import multiprocessing

import pandas as pd
import psycopg

from common.ml.clustering.features import _compute_features_for_group
from common.planning_date import get_planning_date

logger = logging.getLogger(__name__)


def load_sales_from_db(
    db_params: dict,
    time_window_months: int = 36,
    min_months_history: int = 1,
) -> pd.DataFrame:
    """Load sales data from ``fact_sales_monthly`` joined with ``dim_sku``.

    Parameters
    ----------
    db_params:
        Connection parameters for psycopg (host, dbname, user, password, port).
    time_window_months:
        Number of months of sales history to load, counting backward from the
        planning date.
    min_months_history:
        Minimum number of non-null sales months a SKU must have to be included
        in the result.  SKUs with fewer months are excluded early to save
        downstream computation.

    Returns
    -------
    DataFrame with columns ``sku_ck``, ``startdate``, ``qty`` — sorted by
    ``sku_ck`` then ``startdate``.
    """
    planning_date = get_planning_date()

    sql = """
        SELECT d.sku_ck,
               s.startdate,
               s.qty
        FROM fact_sales_monthly s
        INNER JOIN dim_sku d
            ON  d.item_id         = s.item_id
            AND d.customer_group  = s.customer_group
            AND d.loc             = s.loc
        WHERE s.qty IS NOT NULL
          AND s.startdate >= %s - (%s || ' months')::INTERVAL
          AND s.startdate <  %s
        ORDER BY d.sku_ck, s.startdate
    """

    logger.info(
        "Loading sales data (window=%d months, min_history=%d, planning_date=%s)",
        time_window_months,
        min_months_history,
        planning_date,
    )

    with psycopg.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (planning_date, str(time_window_months), planning_date))
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]

    if not rows:
        logger.warning("No sales data returned from the database.")
        return pd.DataFrame(columns=["sku_ck", "startdate", "qty"])

    df = pd.DataFrame(rows, columns=colnames)
    df["startdate"] = pd.to_datetime(df["startdate"])
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)

    # Filter out SKUs with insufficient history
    month_counts = df.groupby("sku_ck")["startdate"].nunique()
    valid_skus = month_counts[month_counts >= min_months_history].index
    df = df[df["sku_ck"].isin(valid_skus)].reset_index(drop=True)

    logger.info(
        "Loaded %d rows for %d SKUs (filtered from %d total SKUs, min_history=%d).",
        len(df),
        df["sku_ck"].nunique(),
        len(month_counts),
        min_months_history,
    )
    return df


def compute_all_sku_features(
    sales_df: pd.DataFrame,
    min_months_history: int = 1,
    workers: int | None = None,
) -> pd.DataFrame:
    """Compute all time-series features for every SKU in the sales DataFrame.

    Groups by ``sku_ck``, calls ``compute_time_series_features()`` per group,
    and uses multiprocessing for parallel computation when the number of groups
    exceeds a threshold.

    Parameters
    ----------
    sales_df:
        DataFrame with columns ``sku_ck``, ``startdate``, ``qty``.
    min_months_history:
        SKU groups with fewer rows than this are skipped.
    workers:
        Number of parallel workers.  ``None`` defaults to ``min(cpu_count, 8)``.

    Returns
    -------
    DataFrame with ``sku_ck`` plus all feature columns produced by
    ``compute_time_series_features()``.
    """
    if sales_df.empty:
        logger.warning("Empty sales DataFrame — returning empty features.")
        return pd.DataFrame(columns=["sku_ck"])

    grouped = sales_df.groupby("sku_ck", sort=False)

    # Filter groups by minimum history
    work_items: list[tuple[str, dict]] = []
    skipped = 0
    for sku_ck, grp in grouped:
        if len(grp) < min_months_history:
            skipped += 1
            continue
        work_items.append(
            (
                sku_ck,
                {
                    "startdate": grp["startdate"].values,
                    "qty": grp["qty"].values,
                },
            )
        )

    if skipped > 0:
        logger.info(
            "Skipped %d SKUs with fewer than %d months of history.",
            skipped,
            min_months_history,
        )

    if not work_items:
        logger.warning("No SKUs with sufficient history — returning empty features.")
        return pd.DataFrame(columns=["sku_ck"])

    n_workers = workers if workers is not None else min(multiprocessing.cpu_count(), 8)

    # Use multiprocessing for large workloads, serial for small ones
    parallel_threshold = 500
    if len(work_items) > parallel_threshold and n_workers > 1:
        logger.info(
            "Computing features for %d SKUs using %d workers (parallel).",
            len(work_items),
            n_workers,
        )
        chunksize = max(1, len(work_items) // (n_workers * 4))
        with multiprocessing.Pool(n_workers) as pool:
            results = pool.map(
                _compute_features_for_group,
                work_items,
                chunksize=chunksize,
            )
    else:
        logger.info(
            "Computing features for %d SKUs (serial).",
            len(work_items),
        )
        results = [_compute_features_for_group(item) for item in work_items]

    features_df = pd.DataFrame(results)

    logger.info(
        "Computed %d features for %d SKUs.",
        len(features_df.columns) - 1,  # exclude sku_ck
        len(features_df),
    )
    return features_df
