"""
Generate clustering features from historical sales data, DFU attributes, and item attributes.

This script is a thin CLI wrapper around the library functions in
``common.ml.clustering.features``.  All feature-engineering logic lives there;
this file provides the argparse entry-point, DB loading, and the merge/save step.

Feature engineering covers six dimensions:
  - Volume         : mean, CV, IQR (robust spread)
  - Trend          : scale-invariant normalized slope, R², CAGR
  - Seasonality    : amplitude ratio, OLS seasonal R² (STL-lite), YoY correlation
  - Periodicity    : FFT dominant-component strength
  - Intermittency  : zero-demand pct, Croston ADI
  - Lifecycle      : months available, recency ratio (last-6m / full history)
"""

import argparse
import multiprocessing
import os
import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import psycopg

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params
from common.core.planning_date import get_planning_date
from common.services.perf_profiler import profiled_section

# ── Re-export library functions for backward compatibility ───────────────────
# Existing scripts and tests import these names from this module.
from common.ml.clustering.features import (  # noqa: F401
    _adi,
    _compute_features_for_group,
    _periodicity_strength,
    _seasonal_r2,
    compute_time_series_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clustering features from sales history")
    parser.add_argument(
        "--min-months", type=int, default=1,
        help="Minimum months of history (1 = include all DFUs with any sales)",
    )
    parser.add_argument(
        "--time-window", type=str, default=None,
        help="Months to include (number or 'all'); default: from config",
    )
    parser.add_argument(
        "--output", type=str, default="data/clustering_features.csv",
        help="Output file path",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: min(cpu_count, 8); 1 for serial)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional clustering config YAML (defaults resolved from DB)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")

    # Load config
    config_path = root / args.config
    cfg = {}
    if config_path.exists():
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f).get("clustering", {})

    db = get_db_params()

    # Determine time window — CLI overrides config
    time_window = args.time_window or str(cfg.get("time_window_months", 36))
    if time_window.lower() == "all":
        cutoff_date = None
    else:
        try:
            months = int(time_window)
            cutoff_date = get_planning_date() - timedelta(days=months * 30)
        except ValueError:
            print(f"Invalid time-window: {time_window}. Use a number or 'all'")
            sys.exit(1)

    # Determine min_months — CLI overrides config
    min_months = args.min_months if args.min_months != 1 else cfg.get("min_months_history", 12)

    print(f"Fetching sales data (min_months={min_months}, time_window={time_window})...")

    # Query sales data aggregated by DFU
    with profiled_section("load_data_from_db"):
        with psycopg.connect(**db) as conn:
            # Get all DFUs
            dfu_query = """
                SELECT sku_ck, item_id, customer_group, loc
                FROM dim_sku
            """
            with conn.cursor() as cur:
                cur.execute(dfu_query)
                cols = [desc[0] for desc in cur.description]
                dfus = pd.DataFrame(cur.fetchall(), columns=cols)
            print(f"Found {len(dfus)} DFUs")

            # Get sales data — only for DFUs that exist in dim_sku
            sales_query = """
                SELECT d.sku_ck, s.startdate, s.qty
                FROM fact_sales_monthly s
                INNER JOIN dim_sku d
                    ON d.item_id = s.item_id
                    AND d.customer_group = s.customer_group
                    AND d.loc = s.loc
                WHERE s.qty IS NOT NULL
            """
            params: dict[str, object] = {}
            if cutoff_date:
                sales_query += " AND s.startdate >= %(cutoff)s"
                params["cutoff"] = cutoff_date
            # Cap at planning date — exclude any data beyond current planning horizon
            planning_upper = get_planning_date().replace(day=1)
            sales_query += " AND s.startdate <= %(planning_upper)s"
            params["planning_upper"] = planning_upper
            sales_query += " ORDER BY d.sku_ck, s.startdate"

            with conn.cursor() as cur:
                cur.execute(sales_query, params if params else None)
                cols = [desc[0] for desc in cur.description]
                sales_df = pd.DataFrame(cur.fetchall(), columns=cols)
            sales_df["startdate"] = pd.to_datetime(sales_df["startdate"])
            print(f"Found {len(sales_df)} sales records")
            print(f"Unique DFUs with sales: {sales_df['sku_ck'].nunique()}")

            # Get DFU attributes (only for qualified DFUs)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT sku_ck, brand, region, state_plan, sales_div,"
                    " prod_cat_desc, prod_class_desc, subclass_desc,"
                    " abc_vol, service_lvl_grp, supergroup,"
                    " execution_lag, total_lt, alcoh_pct, proof, vintage"
                    " FROM dim_sku"
                )
                cols = [desc[0] for desc in cur.description]
                dfu_attrs = pd.DataFrame(cur.fetchall(), columns=cols)

            # Get item attributes (join via item_id)
            item_query = """
                SELECT DISTINCT
                    i.item_id AS item_id,
                    i.category, i.class, i.sub_class,
                    i.brand_name, i.country, i.national_service_model,
                    i.case_weight, i.item_proof,
                    i.bpc, i.bottle_pack, i.pack_case
                FROM dim_item i
                INNER JOIN dim_sku d ON i.item_id = d.item_id
            """
            with conn.cursor() as cur:
                cur.execute(item_query)
                cols = [desc[0] for desc in cur.description]
                item_attrs = pd.DataFrame(cur.fetchall(), columns=cols)

    print("Computing time series features...")

    # Group sales by sku_ck once (single pass)
    sales_df = sales_df.sort_values(["sku_ck", "startdate"])
    grouped = sales_df.groupby("sku_ck", sort=False)
    n_groups = grouped.ngroups

    # Determine worker count
    n_workers = args.workers if args.workers is not None else min(os.cpu_count() or 1, 8)
    n_workers = max(1, n_workers)

    with profiled_section("compute_time_series_features"):
        if n_workers == 1 or n_groups < 500:
            # Serial path — avoids multiprocessing overhead for small datasets or when requested
            ts_features_list = []
            report_interval = max(1, n_groups // 20)  # ~5% progress steps
            for idx, (sku_ck, dfu_sales) in enumerate(grouped):
                if idx == 0 or (idx + 1) % report_interval == 0 or (idx + 1) == n_groups:
                    print(f"  Processing DFU {idx + 1}/{n_groups} ({(idx + 1) / n_groups * 100:.0f}%)...")
                ts_features = compute_time_series_features(dfu_sales)
                ts_features["sku_ck"] = sku_ck
                ts_features_list.append(ts_features)
        else:
            # Parallel path — distribute DFU groups across worker processes
            print(f"  Using {n_workers} parallel workers...")
            # Pre-extract group data as lightweight dicts (avoid pickling DataFrames)
            work_items = []
            for sku_ck, dfu_sales in grouped:
                work_items.append((
                    sku_ck,
                    {
                        "startdate": dfu_sales["startdate"].values,
                        "qty": dfu_sales["qty"].values,
                    },
                ))

            try:
                with multiprocessing.Pool(processes=n_workers) as pool:
                    ts_features_list = pool.map(
                        _compute_features_for_group,
                        work_items,
                        chunksize=max(1, n_groups // (n_workers * 4)),
                    )
            except Exception as exc:
                print(f"ERROR: Worker process failed during feature computation: {exc}")
                raise

            print(f"  Completed {len(ts_features_list)}/{n_groups} DFUs across {n_workers} workers.")

    ts_features_df = pd.DataFrame(ts_features_list)
    print(f"Computed features for {len(ts_features_df)} DFUs")

    # Merge with DFU attributes
    print("Merging DFU attributes...")
    with profiled_section("merge_attributes"):
        feature_df = ts_features_df.merge(dfu_attrs, on="sku_ck", how="left")

        # Merge with item attributes
        print("Merging item attributes...")
        feature_df = feature_df.merge(
            dfus[["sku_ck", "item_id"]].merge(item_attrs, on="item_id", how="left"),
            on="sku_ck",
            how="left",
            suffixes=("", "_item"),
        )

        # Handle missing values
        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
        feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0)

    # Save feature matrix
    with profiled_section("save_feature_matrix"):
        output_path = root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(output_path, index=False)
    print(f"Saved feature matrix to {output_path}")
    print(f"Feature matrix shape: {feature_df.shape}")
    print(f"Features: {list(feature_df.columns)}")


if __name__ == "__main__":
    main()
