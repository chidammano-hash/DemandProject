"""Golden set creation: stratified sampling and data loading.

Creates a mixed golden set of 5000 SKUs stratified by cluster,
loads their sales data, DFU attributes, and item attributes.
"""

import logging
from pathlib import Path

import pandas as pd
import psycopg

from common.core.db import get_db_params
from common.core.planning_date import get_planning_date
from common.ml.backtest_sampler import stratified_sample

logger = logging.getLogger(__name__)


def create_loc_golden_set(
    loc: str,
    output_dir: Path | None = None,
) -> list[str]:
    """Fetch all DFUs for a specific location — no sampling, uses every DFU at that loc.

    Args:
        loc: Location code, e.g. '1401-BULK'.
        output_dir: If provided, saves golden_set_skus.csv here.

    Returns:
        List of sku_ck strings for all DFUs at the given location.
    """
    db = get_db_params()
    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT sku_ck FROM dim_sku WHERE loc = %s ORDER BY sku_ck",
            (loc,),
        )
        rows = cur.fetchall()
    golden_skus = [row[0] for row in rows]

    if not golden_skus:
        logger.warning("No DFUs found for loc='%s'", loc)
    else:
        logger.info("Loc filter '%s': %d DFUs found", loc, len(golden_skus))
        _log_cluster_distribution(golden_skus)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "golden_set_skus.csv"
        pd.DataFrame({"sku_ck": golden_skus}).to_csv(out_path, index=False)
        logger.info("Saved golden set to %s", out_path)

    return golden_skus


def create_golden_set(
    n_dfus: int = 5000,
    sampling_method: str = "proportional",
    seed: int = 42,
    output_dir: Path | None = None,
) -> list[str]:
    """Sample a stratified golden set of SKUs.

    Args:
        n_dfus: Number of DFUs to sample.
        sampling_method: 'proportional', 'equal', or 'sqrt'.
        seed: Random seed for reproducibility.
        output_dir: If provided, saves golden_set_skus.csv here.

    Returns:
        List of sku_ck strings.
    """
    db = get_db_params()
    with psycopg.connect(**db) as conn:
        golden_skus = stratified_sample(
            conn, target_n=n_dfus, method=sampling_method, seed=seed
        )

    logger.info("Golden set: %d SKUs sampled (method=%s, seed=%d)",
                len(golden_skus), sampling_method, seed)

    # Log cluster distribution from the sku_ck format (item|cust|loc)
    if golden_skus:
        _log_cluster_distribution(golden_skus)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "golden_set_skus.csv"
        pd.DataFrame({"sku_ck": golden_skus}).to_csv(out_path, index=False)
        logger.info("Saved golden set to %s", out_path)

    return golden_skus


def _log_cluster_distribution(golden_skus: list[str]) -> None:
    """Query cluster assignments for sampled SKUs and log distribution."""
    db = get_db_params()
    try:
        with psycopg.connect(**db) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(ml_cluster::text, 'unassigned') AS cluster,
                       COUNT(*) AS cnt
                FROM dim_sku
                WHERE sku_ck = ANY(%s)
                GROUP BY ml_cluster
                ORDER BY cnt DESC
                """,
                (golden_skus,),
            )
            rows = cur.fetchall()
        for cluster, cnt in rows:
            logger.info("  cluster %s: %d SKUs", cluster, cnt)
    except psycopg.Error:
        logger.warning("Could not query cluster distribution", exc_info=True)


def load_golden_set_data(
    golden_skus: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load sales, DFU attributes, and item attributes for the golden set.

    Args:
        golden_skus: List of sku_ck values from create_golden_set().

    Returns:
        (sales_df, dfu_attrs, item_attrs) filtered to golden set only.

        sales_df columns: sku_ck, item_id, customer_group, loc, startdate, qty
        dfu_attrs columns: sku_ck, item_id, customer_group, loc, execution_lag,
                          ml_cluster, region, brand, abc_vol, seasonality_profile,
                          variability_class, abc_xyz_segment, ...
        item_attrs columns: item_id, case_weight, bpc, item_proof, category,
                           brand_name, class, sub_class, ...
    """
    db = get_db_params()
    planning_cutoff = get_planning_date().replace(day=1)

    with psycopg.connect(**db) as conn:
        # 1. Sales data
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.sku_ck, s.item_id, s.customer_group, s.loc,
                       s.startdate, s.qty
                FROM (
                    SELECT item_id || '_' || customer_group || '_' || loc AS sku_ck,
                           item_id, customer_group, loc, startdate,
                           COALESCE(qty, 0) AS qty
                    FROM fact_sales_monthly
                    WHERE startdate < %s
                ) s
                WHERE s.sku_ck = ANY(%s)
                ORDER BY s.sku_ck, s.startdate
                """,
                (planning_cutoff, golden_skus),
            )
            cols = [d[0] for d in cur.description]
            sales_df = pd.DataFrame(cur.fetchall(), columns=cols)

        # 2. DFU attributes
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sku_ck, item_id, customer_group, loc,
                       COALESCE(execution_lag, 0) AS execution_lag,
                       COALESCE(ml_cluster, '0') AS ml_cluster,
                       region, brand, abc_vol,
                       seasonality_profile, variability_class, abc_xyz_segment
                FROM dim_sku
                WHERE sku_ck = ANY(%s)
                """,
                (golden_skus,),
            )
            cols = [d[0] for d in cur.description]
            dfu_attrs = pd.DataFrame(cur.fetchall(), columns=cols)

        # 3. Item attributes
        item_ids = dfu_attrs["item_id"].unique().tolist() if not dfu_attrs.empty else []
        if item_ids:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT item_id, case_weight, bpc, item_proof,
                           category, brand_name, class
                    FROM dim_item
                    WHERE item_id = ANY(%s)
                    """,
                    (item_ids,),
                )
                cols = [d[0] for d in cur.description]
                item_attrs = pd.DataFrame(cur.fetchall(), columns=cols)
        else:
            item_attrs = pd.DataFrame()

    # Type conversions
    if not sales_df.empty:
        sales_df["startdate"] = pd.to_datetime(sales_df["startdate"])
        sales_df["qty"] = pd.to_numeric(sales_df["qty"], errors="coerce").fillna(0)

    logger.info(
        "Golden set data loaded: %d sales rows, %d DFUs, %d items",
        len(sales_df), len(dfu_attrs), len(item_attrs),
    )
    return sales_df, dfu_attrs, item_attrs


def load_external_forecast(
    golden_skus: list[str],
    months: list[pd.Timestamp],
) -> pd.DataFrame:
    """Load external forecast for golden set as comparison baseline.

    Args:
        golden_skus: List of sku_ck values.
        months: List of target months to load forecasts for.

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref
        Uses the lowest available lag per DFU-month (closest to actuals).
    """
    db = get_db_params()
    month_dates = [m.date() if hasattr(m, "date") else m for m in months]

    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT item_id || '_' || customer_group || '_' || loc AS sku_ck,
                   startdate,
                   basefcst_pref
            FROM fact_external_forecast_monthly
            WHERE model_id = 'external'
              AND startdate = ANY(%s)
              AND item_id || '_' || customer_group || '_' || loc = ANY(%s)
            ORDER BY sku_ck, startdate
            """,
            (month_dates, golden_skus),
        )
        cols = [d[0] for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)

    if not df.empty:
        df["startdate"] = pd.to_datetime(df["startdate"])
        df["basefcst_pref"] = pd.to_numeric(
            df["basefcst_pref"], errors="coerce"
        ).fillna(0)
        # Dedup: multiple lags may exist per (sku_ck, startdate) — keep mean
        df = df.groupby(["sku_ck", "startdate"], sort=False, as_index=False)[
            "basefcst_pref"
        ].mean()

    logger.info(
        "External forecast loaded: %d rows for %d months",
        len(df), len(months),
    )
    return df


def load_existing_predictions(
    golden_skus: list[str],
    months: list[pd.Timestamp],
    model_ids: list[str],
) -> pd.DataFrame:
    """Load existing model predictions from backtest_lag_archive.

    Args:
        golden_skus: List of sku_ck values.
        months: List of target months to load predictions for.
        model_ids: List of model identifiers (e.g. ['lgbm', 'catboost']).

    Returns:
        DataFrame with columns: sku_ck, startdate, basefcst_pref,
        tothist_dmd, model_id
    """
    db = get_db_params()
    month_dates = [m.date() if hasattr(m, "date") else m for m in months]

    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT item_id || '_' || customer_group || '_' || loc AS sku_ck,
                   startdate,
                   basefcst_pref,
                   tothist_dmd,
                   model_id
            FROM backtest_lag_archive
            WHERE lag = 0
              AND startdate = ANY(%s)
              AND model_id = ANY(%s)
              AND item_id || '_' || customer_group || '_' || loc = ANY(%s)
            ORDER BY sku_ck, startdate, model_id
            """,
            (month_dates, model_ids, golden_skus),
        )
        cols = [d[0] for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)

    if not df.empty:
        df["startdate"] = pd.to_datetime(df["startdate"])
        df["basefcst_pref"] = pd.to_numeric(
            df["basefcst_pref"], errors="coerce"
        ).fillna(0)
        df["tothist_dmd"] = pd.to_numeric(
            df["tothist_dmd"], errors="coerce"
        ).fillna(0)

    logger.info(
        "Existing predictions loaded: %d rows for %d models, %d months",
        len(df), len(model_ids), len(months),
    )
    return df
