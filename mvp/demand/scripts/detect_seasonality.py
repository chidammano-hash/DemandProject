"""
Detect seasonality patterns in DFU monthly sales history.

Computes per-DFU seasonality metrics: strength (CV of monthly means),
year-over-year correlation, autocorrelation at lag 12, peak/trough analysis,
and classifies each DFU into a seasonality profile tier.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def get_db_conn() -> dict[str, Any]:
    """Get database connection parameters."""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5440")),
        "dbname": os.getenv("POSTGRES_DB", "demand_mvp"),
        "user": os.getenv("POSTGRES_USER", "demand"),
        "password": os.getenv("POSTGRES_PASSWORD", "demand"),
    }


def load_config(config_path: str = "config/seasonality_config.yaml") -> dict:
    """Load seasonality configuration."""
    path = ROOT / config_path
    with open(path) as f:
        return yaml.safe_load(f)["seasonality"]


def compute_acf_lag12(series: np.ndarray) -> float:
    """Compute autocorrelation at lag 12."""
    if len(series) < 25:  # Need at least 24 + 1 observations
        return 0.0
    mean = np.mean(series)
    var = np.var(series)
    if var == 0:
        return 0.0
    n = len(series)
    cov = np.sum((series[:n - 12] - mean) * (series[12:] - mean)) / n
    return float(cov / var)


def compute_seasonality_metrics(
    dfu_sales: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """Compute all seasonality metrics for a single DFU.

    Parameters
    ----------
    dfu_sales : DataFrame with columns [startdate, qty], sorted by startdate
    config : seasonality config dict

    Returns
    -------
    dict with seasonality metrics and classification
    """
    min_months = config["min_months_history"]
    thresholds = config["thresholds"]
    confirmation = config["confirmation"]
    peak_trough_min = config["peak_trough_min_ratio"]

    qty = dfu_sales["qty"].fillna(0).values.astype(np.float64)
    months_available = len(qty)

    # Insufficient history check
    if months_available < min_months:
        return {
            "seasonality_profile": "insufficient_history",
            "seasonality_strength": None,
            "is_yearly_seasonal": None,
            "peak_month": None,
            "trough_month": None,
            "peak_trough_ratio": None,
            "yoy_correlation": None,
            "acf_lag12": None,
            "months_available": months_available,
        }

    # Step 1: Monthly means
    dfu_sales = dfu_sales.copy()
    dfu_sales["month"] = dfu_sales["startdate"].dt.month
    monthly_means = dfu_sales.groupby("month")["qty"].mean()
    # Fill missing months with 0
    monthly_means = monthly_means.reindex(range(1, 13), fill_value=0.0)
    mm_values = monthly_means.values.astype(np.float64)

    # Step 2: Seasonality strength (CV of monthly means)
    mm_mean = np.mean(mm_values)
    mm_std = np.std(mm_values)
    seasonality_strength = float(mm_std / mm_mean) if mm_mean > 0 else 0.0

    # Step 3: Year-over-year correlation
    dfu_sales["year"] = dfu_sales["startdate"].dt.year
    pivot = dfu_sales.pivot_table(values="qty", index="month", columns="year", aggfunc="mean")
    if pivot.shape[1] >= 2:
        corr_matrix = pivot.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper = corr_matrix.where(mask).stack()
        yoy_correlation = float(upper.mean()) if len(upper) > 0 else 0.0
    else:
        yoy_correlation = 0.0

    # Step 4: Autocorrelation at lag 12
    acf_lag12 = compute_acf_lag12(qty)

    # Step 5: Peak and trough
    peak_month = int(monthly_means.idxmax())
    trough_month = int(monthly_means.idxmin())
    trough_val = mm_values[trough_month - 1]
    peak_val = mm_values[peak_month - 1]
    peak_trough_ratio = float(peak_val / trough_val) if trough_val > 0 else None

    # Step 6: Profile classification
    has_confirmation = (
        yoy_correlation >= confirmation["yoy_correlation"]
        or acf_lag12 >= confirmation["acf_lag12"]
    )

    if seasonality_strength >= thresholds["high"] and has_confirmation:
        profile = "high"
    elif seasonality_strength >= thresholds["medium"] and has_confirmation:
        profile = "medium"
    elif seasonality_strength >= thresholds["low"]:
        profile = "low"
    else:
        profile = "none"

    # Step 7: Yearly seasonal flag
    is_yearly_seasonal = (
        seasonality_strength >= thresholds["low"]
        and has_confirmation
        and (peak_trough_ratio is not None and peak_trough_ratio >= peak_trough_min)
    )

    return {
        "seasonality_profile": profile,
        "seasonality_strength": round(seasonality_strength, 4),
        "is_yearly_seasonal": is_yearly_seasonal,
        "peak_month": peak_month,
        "trough_month": trough_month,
        "peak_trough_ratio": round(peak_trough_ratio, 4) if peak_trough_ratio is not None else None,
        "yoy_correlation": round(yoy_correlation, 4),
        "acf_lag12": round(acf_lag12, 4),
        "months_available": months_available,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect seasonality patterns in DFU sales")
    parser.add_argument("--config", type=str, default="config/seasonality_config.yaml", help="Config file path")
    parser.add_argument("--min-months", type=int, default=None, help="Override minimum months required")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--verbose", action="store_true", help="Print per-DFU diagnostics")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = load_config(args.config)

    if args.min_months is not None:
        config["min_months_history"] = args.min_months

    output_path = ROOT / (args.output or config["output_path"])

    print(f"Seasonality detection (min_months={config['min_months_history']})")
    print(f"Thresholds: low={config['thresholds']['low']}, "
          f"medium={config['thresholds']['medium']}, high={config['thresholds']['high']}")

    db = get_db_conn()

    with psycopg.connect(**db) as conn:
        print("Loading sales data...")
        sales_df = pd.read_sql(
            """
            SELECT d.dfu_ck, s.startdate, s.qty
            FROM fact_sales_monthly s
            INNER JOIN dim_dfu d
                ON d.dmdunit = s.dmdunit
                AND d.dmdgroup = s.dmdgroup
                AND d.loc = s.loc
            WHERE s.qty IS NOT NULL
            ORDER BY d.dfu_ck, s.startdate
            """,
            conn,
        )

    sales_df["startdate"] = pd.to_datetime(sales_df["startdate"])
    print(f"Loaded {len(sales_df)} sales records for {sales_df['dfu_ck'].nunique()} DFUs")

    # Process each DFU
    grouped = sales_df.groupby("dfu_ck", sort=False)
    n_groups = grouped.ngroups
    results = []

    for idx, (dfu_ck, dfu_sales) in enumerate(grouped):
        if (idx + 1) % 2000 == 0 or idx == 0:
            print(f"  Processing DFU {idx + 1}/{n_groups}...")

        metrics = compute_seasonality_metrics(dfu_sales.sort_values("startdate"), config)
        metrics["dfu_ck"] = dfu_ck
        results.append(metrics)

        if args.verbose and metrics["seasonality_profile"] in ("high", "medium"):
            print(f"    {dfu_ck}: {metrics['seasonality_profile']} "
                  f"(strength={metrics['seasonality_strength']}, "
                  f"yoy={metrics['yoy_correlation']}, acf12={metrics['acf_lag12']})")

    results_df = pd.DataFrame(results)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results_df)} DFU seasonality profiles to {output_path}")

    # Print summary
    print("\nProfile distribution:")
    profile_counts = results_df["seasonality_profile"].value_counts()
    for profile, count in profile_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {profile}: {count} ({pct:.1f}%)")

    seasonal_count = results_df["is_yearly_seasonal"].sum()
    print(f"\nDFUs with yearly seasonal cycle: {seasonal_count} "
          f"({seasonal_count / len(results_df) * 100:.1f}%)")

    # Top seasonal DFUs
    ranked = results_df.dropna(subset=["seasonality_strength"])
    if len(ranked) > 0:
        top10 = ranked.nlargest(10, "seasonality_strength")
        print("\nTop 10 most seasonal DFUs:")
        for _, row in top10.iterrows():
            print(f"  {row['dfu_ck']}: strength={row['seasonality_strength']}, "
                  f"profile={row['seasonality_profile']}, "
                  f"peak={row['peak_month']}, trough={row['trough_month']}")


if __name__ == "__main__":
    main()
