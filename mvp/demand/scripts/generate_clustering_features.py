"""
Generate clustering features from historical sales data, DFU attributes, and item attributes.

This script extracts time series features, item features, and DFU features to create
a feature matrix for clustering analysis.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.domain_specs import DFU_SPEC, ITEM_SPEC


def get_db_conn() -> dict[str, Any]:
    """Get database connection parameters."""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5440")),
        "dbname": os.getenv("POSTGRES_DB", "demand_mvp"),
        "user": os.getenv("POSTGRES_USER", "demand"),
        "password": os.getenv("POSTGRES_PASSWORD", "demand"),
    }


def compute_time_series_features(df: pd.DataFrame) -> pd.Series:
    """Compute time series features from monthly demand data."""
    features = {}
    
    if len(df) == 0:
        return pd.Series()
    
    # Use .to_numpy() for a single contiguous array (faster than .values for some ops)
    demand_values = np.asarray(df["qty"].fillna(0), dtype=np.float64)
    
    # Volume metrics
    features["mean_demand"] = np.mean(demand_values)
    features["median_demand"] = np.median(demand_values)
    features["std_demand"] = np.std(demand_values) if len(demand_values) > 1 else 0.0
    features["cv_demand"] = features["std_demand"] / features["mean_demand"] if features["mean_demand"] > 0 else 0.0
    features["min_demand"] = np.min(demand_values)
    features["max_demand"] = np.max(demand_values)
    features["total_demand"] = np.sum(demand_values)
    
    # Trend features
    if len(df) > 1:
        x = np.arange(len(df))
        y = demand_values
        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0.0
        features["trend_slope"] = slope
        features["trend_pct_change"] = (
            ((demand_values[-1] - demand_values[0]) / demand_values[0] * 100)
            if demand_values[0] > 0 else 0.0
        )
        features["trend_direction"] = 1 if slope > 0.01 else (-1 if slope < -0.01 else 0)
    else:
        features["trend_slope"] = 0.0
        features["trend_pct_change"] = 0.0
        features["trend_direction"] = 0
    
    # Seasonality features
    if len(df) >= 12:
        df_sorted = df.sort_values("startdate")
        monthly_means = df_sorted.groupby(df_sorted["startdate"].dt.month)["qty"].mean()
        if len(monthly_means) > 1:
            seasonal_std = monthly_means.std()
            seasonal_mean = monthly_means.mean()
            features["seasonality_strength"] = seasonal_std / seasonal_mean if seasonal_mean > 0 else 0.0
            features["peak_month"] = monthly_means.idxmax()
            features["seasonal_index_std"] = seasonal_std
        else:
            features["seasonality_strength"] = 0.0
            features["peak_month"] = 1
            features["seasonal_index_std"] = 0.0
        
        # Year-over-year correlation
        if len(df_sorted) >= 24:
            df_sorted["year"] = df_sorted["startdate"].dt.year
            df_sorted["month"] = df_sorted["startdate"].dt.month
            pivot = df_sorted.pivot_table(values="qty", index="month", columns="year", aggfunc="mean")
            if pivot.shape[1] >= 2:
                corr_matrix = pivot.corr()
                # Get correlation excluding diagonal
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                features["year_over_year_correlation"] = corr_matrix.where(mask).stack().mean() if mask.sum() > 0 else 0.0
            else:
                features["year_over_year_correlation"] = 0.0
        else:
            features["year_over_year_correlation"] = 0.0
    else:
        features["seasonality_strength"] = 0.0
        features["peak_month"] = 1
        features["seasonal_index_std"] = 0.0
        features["year_over_year_correlation"] = 0.0
    
    # Volatility features
    zero_count = np.sum(demand_values == 0)
    features["zero_demand_pct"] = zero_count / len(demand_values) if len(demand_values) > 0 else 0.0
    features["sparsity_score"] = zero_count / len(demand_values) if len(demand_values) > 0 else 1.0
    features["demand_stability"] = 1.0 / (1.0 + features["cv_demand"]) if features["cv_demand"] > 0 else 1.0
    
    # Outlier count (> 2 std from mean)
    if features["std_demand"] > 0:
        outliers = np.abs(demand_values - features["mean_demand"]) > (2 * features["std_demand"])
        features["outlier_count"] = np.sum(outliers)
    else:
        features["outlier_count"] = 0
    
    # Growth patterns
    if len(df) >= 12:
        # CAGR: compound annual growth rate
        first_half = demand_values[:len(demand_values)//2].mean()
        second_half = demand_values[len(demand_values)//2:].mean()
        if first_half > 0 and len(demand_values) >= 12:
            periods = len(demand_values) / 12.0
            features["growth_rate"] = ((second_half / first_half) ** (1.0 / periods) - 1) * 100
        else:
            features["growth_rate"] = 0.0
        
        # Recent vs historical
        if len(demand_values) >= 12:
            recent = demand_values[-6:].mean()
            historical = demand_values[:-6].mean() if len(demand_values) > 6 else recent
            features["recent_vs_historical"] = recent / historical if historical > 0 else 1.0
        else:
            features["recent_vs_historical"] = 1.0
        
        # Acceleration (second derivative approximation)
        if len(df) >= 3:
            first_third = demand_values[:len(demand_values)//3].mean()
            middle_third = demand_values[len(demand_values)//3:2*len(demand_values)//3].mean()
            last_third = demand_values[2*len(demand_values)//3:].mean()
            if first_third > 0:
                growth1 = (middle_third - first_third) / first_third
                growth2 = (last_third - middle_third) / middle_third if middle_third > 0 else 0
                features["acceleration"] = growth2 - growth1
            else:
                features["acceleration"] = 0.0
        else:
            features["acceleration"] = 0.0
    else:
        features["growth_rate"] = 0.0
        features["recent_vs_historical"] = 1.0
        features["acceleration"] = 0.0
    
    return pd.Series(features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clustering features from sales history")
    parser.add_argument("--min-months", type=int, default=12, help="Minimum months of history required")
    parser.add_argument("--time-window", type=str, default="24", help="Months to include (number or 'all')")
    parser.add_argument("--output", type=str, default="data/clustering_features.csv", help="Output file path")
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    
    db = get_db_conn()
    
    # Determine time window
    if args.time_window.lower() == "all":
        cutoff_date = None
    else:
        try:
            months = int(args.time_window)
            cutoff_date = date.today() - timedelta(days=months * 30)
        except ValueError:
            print(f"Invalid time-window: {args.time_window}. Use a number or 'all'")
            sys.exit(1)
    
    print(f"Fetching sales data (min_months={args.min_months}, time_window={args.time_window})...")
    
    # Query sales data aggregated by DFU
    with psycopg.connect(**db) as conn:
        # Get all DFUs
        dfu_query = f"""
            SELECT dfu_ck, dmdunit, dmdgroup, loc
            FROM dim_dfu
        """
        dfus = pd.read_sql(dfu_query, conn)
        print(f"Found {len(dfus)} DFUs")
        
        # Get sales data
        sales_query = """
            SELECT 
                CONCAT(dmdunit, '_', dmdgroup, '_', loc) AS dfu_ck,
                startdate,
                qty
            FROM fact_sales_monthly
            WHERE qty IS NOT NULL
        """
        if cutoff_date:
            sales_query += f" AND startdate >= '{cutoff_date}'"
        sales_query += " ORDER BY dfu_ck, startdate"
        
        sales_df = pd.read_sql(sales_query, conn)
        sales_df["startdate"] = pd.to_datetime(sales_df["startdate"])
        print(f"Found {len(sales_df)} sales records")
        
        # Get DFU attributes
        dfu_attrs_query = f"""
            SELECT 
                dfu_ck,
                brand, region, state_plan, sales_div,
                prod_cat_desc, prod_class_desc, subclass_desc,
                abc_vol, service_lvl_grp, supergroup,
                execution_lag, total_lt, alcoh_pct, proof, vintage
            FROM dim_dfu
        """
        dfu_attrs = pd.read_sql(dfu_attrs_query, conn)
        
        # Get item attributes (join via dmdunit)
        item_query = f"""
            SELECT DISTINCT
                i.item_no AS dmdunit,
                i.category, i.class, i.sub_class,
                i.brand_name, i.country, i.national_service_model,
                i.case_weight, i.item_proof,
                i.bpc, i.bottle_pack, i.pack_case
            FROM dim_item i
            INNER JOIN dim_dfu d ON i.item_no = d.dmdunit
        """
        item_attrs = pd.read_sql(item_query, conn)
    
    print("Computing time series features...")
    
    # Group sales by dfu_ck once (single pass); then iterate over groups to avoid O(N) filter per DFU
    sales_df = sales_df.sort_values(["dfu_ck", "startdate"])
    grouped = sales_df.groupby("dfu_ck", sort=False)
    n_groups = grouped.ngroups
    ts_features_list = []
    for idx, (dfu_ck, dfu_sales) in enumerate(grouped):
        if (idx + 1) % 500 == 0 or idx == 0:
            print(f"  Processing DFU {idx + 1}/{n_groups}...")
        if len(dfu_sales) < args.min_months:
            continue
        ts_features = compute_time_series_features(dfu_sales)
        ts_features["dfu_ck"] = dfu_ck
        ts_features_list.append(ts_features)
    
    ts_features_df = pd.DataFrame(ts_features_list)
    print(f"Computed features for {len(ts_features_df)} DFUs")
    
    # Merge with DFU attributes
    print("Merging DFU attributes...")
    feature_df = ts_features_df.merge(dfu_attrs, on="dfu_ck", how="left")
    
    # Merge with item attributes
    print("Merging item attributes...")
    feature_df = feature_df.merge(
        dfus[["dfu_ck", "dmdunit"]].merge(item_attrs, on="dmdunit", how="left"),
        on="dfu_ck",
        how="left",
        suffixes=("", "_item")
    )
    
    # Handle missing values
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0)
    
    # Save feature matrix
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    print(f"Saved feature matrix to {output_path}")
    print(f"Feature matrix shape: {feature_df.shape}")
    print(f"Features: {list(feature_df.columns)}")


if __name__ == "__main__":
    main()
