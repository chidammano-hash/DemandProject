"""
Run Prophet backtesting with expanding-window timeframes.

Prophet fits a per-DFU individual time series model with native Fourier
seasonality decomposition and piecewise linear trend.  Unlike tree-based models
(LGBM, CatBoost, XGBoost), Prophet operates on raw (ds, y) series — no
hand-engineered lag or rolling features needed.

Supports three strategies:
  - global:      Independent Prophet fit per DFU (model_id=prophet_global)
  - per_cluster: Fit only DFUs within assigned clusters (model_id=prophet_cluster)
  - pooled:      Aggregate by cluster → fit → disaggregate proportionally (model_id=prophet_pooled)

Produces two CSVs:
  - backtest_predictions.csv: execution-lag only (for fact_external_forecast_monthly)
  - backtest_predictions_all_lags.csv: lag 0-4 archive (for backtest_lag_archive)
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import psycopg

# Suppress Prophet's verbose logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*cmdstan.*")
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ts() -> str:
    """Timestamp prefix for log messages."""
    return time.strftime("%H:%M:%S")


# ── DB connection ────────────────────────────────────────────────────────────


def get_db_conn() -> dict[str, Any]:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5440")),
        "dbname": os.getenv("POSTGRES_DB", "demand_mvp"),
        "user": os.getenv("POSTGRES_USER", "demand"),
        "password": os.getenv("POSTGRES_PASSWORD", "demand"),
    }


# ── Timeframe generation ────────────────────────────────────────────────────


def generate_timeframes(
    earliest: pd.Timestamp,
    latest: pd.Timestamp,
    n: int = 10,
) -> list[dict]:
    """Generate N expanding-window timeframes.

    For timeframe i (A=0 .. J=9):
      train_end   = latest - (N - i) months
      predict     = [train_end + 1 month, latest]
    """
    timeframes = []
    for i in range(n):
        train_end = latest - pd.DateOffset(months=(n - i))
        train_end = train_end.normalize()
        predict_start = train_end + pd.DateOffset(months=1)
        label = chr(ord("A") + i)
        timeframes.append({
            "label": label,
            "index": i,
            "train_start": earliest,
            "train_end": train_end,
            "predict_start": predict_start,
            "predict_end": latest,
        })
    return timeframes


# ── Per-DFU Prophet fitting ──────────────────────────────────────────────────


def _fit_single_dfu(args_tuple: tuple) -> list[dict] | None:
    """Fit Prophet for one DFU and return predictions.

    Designed to be called via multiprocessing.Pool.map.
    Returns list of dicts with (dfu_ck, dmdunit, dmdgroup, loc, startdate, basefcst_pref)
    or None if fitting fails.
    """
    dfu_ck, dmdunit, dmdgroup, loc, train_series, predict_months, prophet_kwargs = args_tuple

    # Import Prophet inside worker to avoid pickling issues
    from prophet import Prophet

    if len(train_series) < 2:
        # Prophet needs at least 2 data points
        return [
            {
                "dfu_ck": dfu_ck,
                "dmdunit": dmdunit,
                "dmdgroup": dmdgroup,
                "loc": loc,
                "startdate": pm,
                "basefcst_pref": 0.0,
            }
            for pm in predict_months
        ]

    # Build Prophet input DataFrame
    df_prophet = pd.DataFrame({
        "ds": train_series.index,
        "y": train_series.values.astype(float),
    })
    df_prophet = df_prophet.sort_values("ds").reset_index(drop=True)

    try:
        m = Prophet(**prophet_kwargs)
        m.fit(df_prophet)

        # Create future DataFrame for prediction months
        future = pd.DataFrame({"ds": predict_months})
        forecast = m.predict(future)

        results = []
        for _, row in forecast.iterrows():
            results.append({
                "dfu_ck": dfu_ck,
                "dmdunit": dmdunit,
                "dmdgroup": dmdgroup,
                "loc": loc,
                "startdate": row["ds"],
                "basefcst_pref": max(float(row["yhat"]), 0),  # floor at 0
            })
        return results

    except Exception:
        # If Prophet fails for this DFU, return zeros
        return [
            {
                "dfu_ck": dfu_ck,
                "dmdunit": dmdunit,
                "dmdgroup": dmdgroup,
                "loc": loc,
                "startdate": pm,
                "basefcst_pref": 0.0,
            }
            for pm in predict_months
        ]


def fit_prophet_parallel(
    sales_df: pd.DataFrame,
    dfu_keys: pd.DataFrame,
    train_end: pd.Timestamp,
    predict_months: list[pd.Timestamp],
    n_workers: int = 4,
    prophet_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Fit Prophet per DFU in parallel and return predictions."""
    if prophet_kwargs is None:
        prophet_kwargs = {}

    # Build per-DFU training series
    train_sales = sales_df[sales_df["startdate"] <= train_end].copy()
    grouped = train_sales.groupby("dfu_ck").apply(
        lambda g: g.set_index("startdate")["qty"], include_groups=False
    )

    # Build work items
    work_items = []
    dfu_lookup = dfu_keys.set_index("dfu_ck")
    for dfu_ck in dfu_keys["dfu_ck"].values:
        row = dfu_lookup.loc[dfu_ck]
        dmdunit = row["dmdunit"]
        dmdgroup = row["dmdgroup"]
        loc = row["loc"]

        if dfu_ck in grouped.index:
            series = grouped.loc[dfu_ck]
            if isinstance(series, pd.DataFrame):
                series = series.squeeze()
        else:
            series = pd.Series(dtype=float)

        work_items.append((dfu_ck, dmdunit, dmdgroup, loc, series, predict_months, prophet_kwargs))

    # Run in parallel
    actual_workers = min(n_workers, len(work_items), cpu_count())
    if actual_workers <= 1:
        all_results = [_fit_single_dfu(item) for item in work_items]
    else:
        with Pool(processes=actual_workers) as pool:
            all_results = pool.map(_fit_single_dfu, work_items)

    # Flatten results
    rows = []
    for result in all_results:
        if result is not None:
            rows.extend(result)

    if not rows:
        return pd.DataFrame(columns=["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"])

    return pd.DataFrame(rows)


# ── Pooled cluster strategy ──────────────────────────────────────────────────


def _fit_single_cluster_pooled(args_tuple: tuple) -> list[dict] | None:
    """Fit Prophet on aggregated cluster-level sales and return cluster-level predictions."""
    cluster_label, train_series, predict_months, prophet_kwargs = args_tuple

    from prophet import Prophet

    if len(train_series) < 2:
        return [
            {"cluster": cluster_label, "startdate": pm, "cluster_forecast": 0.0}
            for pm in predict_months
        ]

    df_prophet = pd.DataFrame({
        "ds": train_series.index,
        "y": train_series.values.astype(float),
    })
    df_prophet = df_prophet.sort_values("ds").reset_index(drop=True)

    try:
        m = Prophet(**prophet_kwargs)
        m.fit(df_prophet)

        future = pd.DataFrame({"ds": predict_months})
        forecast = m.predict(future)

        results = []
        for _, row in forecast.iterrows():
            results.append({
                "cluster": cluster_label,
                "startdate": row["ds"],
                "cluster_forecast": max(float(row["yhat"]), 0),
            })
        return results

    except Exception:
        return [
            {"cluster": cluster_label, "startdate": pm, "cluster_forecast": 0.0}
            for pm in predict_months
        ]


def fit_prophet_pooled(
    sales_df: pd.DataFrame,
    dfu_keys: pd.DataFrame,
    cluster_map: dict[str, str],
    train_end: pd.Timestamp,
    predict_months: list[pd.Timestamp],
    n_workers: int = 4,
    prophet_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Pooled cluster strategy: aggregate by cluster → fit → disaggregate.

    1. Aggregate sales by cluster (sum qty per cluster per month).
    2. Fit one Prophet model per cluster on aggregated series.
    3. Disaggregate cluster-level forecast to DFU level using historical
       demand proportions within each cluster.
    """
    if prophet_kwargs is None:
        prophet_kwargs = {}

    train_sales = sales_df[sales_df["startdate"] <= train_end].copy()
    train_sales["cluster"] = train_sales["dfu_ck"].map(cluster_map)

    # Only include DFUs with a cluster assignment
    train_sales = train_sales[train_sales["cluster"].notna() & (train_sales["cluster"] != "__unknown__")]

    # Step 1: Aggregate by cluster
    cluster_agg = train_sales.groupby(["cluster", "startdate"])["qty"].sum().reset_index()

    # Step 2: Fit Prophet per cluster
    clusters = sorted(cluster_agg["cluster"].unique())
    work_items = []
    for cluster_label in clusters:
        c_data = cluster_agg[cluster_agg["cluster"] == cluster_label]
        series = c_data.set_index("startdate")["qty"]
        work_items.append((cluster_label, series, predict_months, prophet_kwargs))

    actual_workers = min(n_workers, len(work_items), cpu_count())
    if actual_workers <= 1:
        all_results = [_fit_single_cluster_pooled(item) for item in work_items]
    else:
        with Pool(processes=actual_workers) as pool:
            all_results = pool.map(_fit_single_cluster_pooled, work_items)

    # Build cluster-level forecast DataFrame
    cluster_rows = []
    for result in all_results:
        if result is not None:
            cluster_rows.extend(result)

    if not cluster_rows:
        return pd.DataFrame(columns=["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"])

    cluster_fcst = pd.DataFrame(cluster_rows)

    # Step 3: Disaggregate to DFU level using historical proportions
    dfu_cluster_sales = train_sales.groupby(["dfu_ck", "cluster"])["qty"].sum().reset_index()
    cluster_total = dfu_cluster_sales.groupby("cluster")["qty"].sum().reset_index().rename(
        columns={"qty": "cluster_total"}
    )
    dfu_cluster_sales = dfu_cluster_sales.merge(cluster_total, on="cluster")
    dfu_cluster_sales["proportion"] = np.where(
        dfu_cluster_sales["cluster_total"] > 0,
        dfu_cluster_sales["qty"] / dfu_cluster_sales["cluster_total"],
        0.0,
    )

    # Join DFU proportions with cluster forecasts
    dfu_lookup = dfu_keys.set_index("dfu_ck")[["dmdunit", "dmdgroup", "loc"]]
    dfu_props = dfu_cluster_sales[["dfu_ck", "cluster", "proportion"]].copy()

    disagg = dfu_props.merge(cluster_fcst, on="cluster")
    disagg["basefcst_pref"] = np.maximum(disagg["proportion"] * disagg["cluster_forecast"], 0)

    # Add DFU dimensions
    disagg = disagg.merge(dfu_lookup, left_on="dfu_ck", right_index=True)

    return disagg[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"]].copy()


# ── Execution-lag assignment ─────────────────────────────────────────────────


def assign_execution_lag(
    pred_df: pd.DataFrame,
    execution_lag_map: dict[str, int],
) -> pd.DataFrame:
    """Assign each prediction its DFU's execution lag and compute fcstdate."""
    t0 = time.time()
    result = pred_df.copy()
    result["execution_lag"] = result["dfu_ck"].map(execution_lag_map).fillna(0).astype(int)
    result["lag"] = result["execution_lag"]
    result["fcstdate"] = result.apply(
        lambda r: r["startdate"] - pd.DateOffset(months=int(r["lag"])), axis=1
    )
    result["forecast_ck"] = (
        result["dmdunit"].astype(str) + "_"
        + result["dmdgroup"].astype(str) + "_"
        + result["loc"].astype(str) + "_"
        + result["fcstdate"].dt.strftime("%Y-%m-%d") + "_"
        + result["startdate"].dt.strftime("%Y-%m-%d")
    )
    print(f"  [{_ts()}] Execution-lag assignment done ({time.time() - t0:.1f}s)")
    return result


# ── All-lag expansion (archive) ──────────────────────────────────────────────


def expand_to_all_lags(
    pred_df: pd.DataFrame,
    max_lag: int,
    execution_lag_map: dict[str, int],
) -> pd.DataFrame:
    """Expand each prediction to lag 0..max_lag rows for the archive table."""
    t0 = time.time()
    dfs = []
    for lag in range(max_lag + 1):
        df = pred_df.copy()
        df["lag"] = lag
        df["fcstdate"] = df["startdate"] - pd.DateOffset(months=lag)
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    result["execution_lag"] = result["dfu_ck"].map(execution_lag_map).fillna(0).astype(int)
    result["forecast_ck"] = (
        result["dmdunit"].astype(str) + "_"
        + result["dmdgroup"].astype(str) + "_"
        + result["loc"].astype(str) + "_"
        + result["fcstdate"].dt.strftime("%Y-%m-%d") + "_"
        + result["startdate"].dt.strftime("%Y-%m-%d")
    )
    print(f"  [{_ts()}] All-lag expansion (0-{max_lag}) done: {len(result):,} rows ({time.time() - t0:.1f}s)")
    return result


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Prophet backtest with expanding-window timeframes")
    parser.add_argument("--cluster-strategy", choices=["global", "per_cluster", "pooled"], default="global",
                        help="global: per-DFU fits, per_cluster: fit only clustered DFUs, pooled: aggregate by cluster → fit → disaggregate")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model_id (default: prophet_global, prophet_cluster, or prophet_pooled)")
    parser.add_argument("--n-timeframes", type=int, default=10, help="Number of expanding windows")
    parser.add_argument("--output-dir", type=str, default="data/backtest", help="Output directory")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of parallel workers for per-DFU fitting")

    # Prophet hyperparameters
    parser.add_argument("--yearly-seasonality", type=str, default="auto",
                        help="Yearly seasonality: auto, True, False, or integer Fourier terms")
    parser.add_argument("--weekly-seasonality", action="store_true", default=False,
                        help="Enable weekly seasonality (default: off for monthly data)")
    parser.add_argument("--daily-seasonality", action="store_true", default=False,
                        help="Enable daily seasonality (default: off for monthly data)")
    parser.add_argument("--changepoint-prior-scale", type=float, default=0.05,
                        help="Flexibility of the trend changepoints")
    parser.add_argument("--seasonality-prior-scale", type=float, default=10.0,
                        help="Strength of the seasonality model")
    parser.add_argument("--growth", type=str, default="linear",
                        choices=["linear", "logistic", "flat"],
                        help="Prophet growth model")
    args = parser.parse_args()

    t_start = time.time()
    load_dotenv(ROOT / ".env")
    db = get_db_conn()

    _default_model_ids = {
        "global": "prophet_global",
        "per_cluster": "prophet_cluster",
        "pooled": "prophet_pooled",
    }
    model_id = args.model_id or _default_model_ids[args.cluster_strategy]
    print(f"[{_ts()}] Backtest: strategy={args.cluster_strategy}, model_id={model_id}, "
          f"n_timeframes={args.n_timeframes}, n_workers={args.n_workers}")

    # Build Prophet kwargs
    yearly = args.yearly_seasonality
    if yearly == "auto":
        pass  # keep "auto"
    elif yearly.lower() in ("true", "false"):
        yearly = yearly.lower() == "true"
    else:
        yearly = int(yearly)

    prophet_kwargs = {
        "yearly_seasonality": yearly,
        "weekly_seasonality": args.weekly_seasonality,
        "daily_seasonality": args.daily_seasonality,
        "changepoint_prior_scale": args.changepoint_prior_scale,
        "seasonality_prior_scale": args.seasonality_prior_scale,
        "growth": args.growth,
    }
    print(f"[{_ts()}] Prophet: {prophet_kwargs}")

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print(f"\n[{_ts()}] Step 1: Loading data from Postgres...")
    t1 = time.time()
    with psycopg.connect(**db) as conn:
        sales_df = pd.read_sql("""
            SELECT d.dfu_ck, s.dmdunit, s.dmdgroup, s.loc, s.startdate, s.qty
            FROM fact_sales_monthly s
            INNER JOIN dim_dfu d
                ON d.dmdunit = s.dmdunit AND d.dmdgroup = s.dmdgroup AND d.loc = s.loc
            WHERE s.qty IS NOT NULL
            ORDER BY d.dfu_ck, s.startdate
        """, conn)

        dfu_attrs = pd.read_sql("""
            SELECT dfu_ck, dmdunit, dmdgroup, loc,
                   execution_lag, total_lt, ml_cluster
            FROM dim_dfu
        """, conn)

    sales_df["startdate"] = pd.to_datetime(sales_df["startdate"])
    sales_df["qty"] = pd.to_numeric(sales_df["qty"], errors="coerce").fillna(0)

    # Only keep DFUs that have sales
    dfus_with_sales = set(sales_df["dfu_ck"].unique())
    dfu_attrs = dfu_attrs[dfu_attrs["dfu_ck"].isin(dfus_with_sales)].copy()

    print(f"  [{_ts()}] Sales: {len(sales_df):,} rows, {len(dfus_with_sales):,} DFUs ({time.time() - t1:.1f}s)")
    print(f"  [{_ts()}] DFU attrs: {len(dfu_attrs):,}")

    # Execution lag lookup
    exec_lag_map = dfu_attrs.set_index("dfu_ck")["execution_lag"].fillna(0).astype(int).to_dict()

    # Cluster map for per_cluster and pooled strategies
    cluster_map = dfu_attrs.set_index("dfu_ck")["ml_cluster"].to_dict()

    # DFU keys for prediction output
    dfu_keys = dfu_attrs[["dfu_ck", "dmdunit", "dmdgroup", "loc"]].drop_duplicates()

    # ── Step 2: Generate timeframes ──────────────────────────────────────────
    latest_month = sales_df["startdate"].max()
    earliest_month = sales_df["startdate"].min()
    print(f"  [{_ts()}] Date range: {earliest_month.date()} → {latest_month.date()}")

    timeframes = generate_timeframes(earliest_month, latest_month, args.n_timeframes)
    print(f"\n[{_ts()}] Step 2: Generated {len(timeframes)} timeframes:")
    for tf in timeframes:
        print(f"  {tf['label']}: train [{tf['train_start'].date()} → {tf['train_end'].date()}], "
              f"predict [{tf['predict_start'].date()} → {tf['predict_end'].date()}]")

    all_months = sorted(sales_df["startdate"].unique())

    # ── Step 3: Train & predict per timeframe ────────────────────────────────
    print(f"\n[{_ts()}] Step 3: Running {len(timeframes)} timeframe backtests...")
    all_predictions = []

    for ti, tf in enumerate(timeframes):
        label = tf["label"]
        train_end = tf["train_end"]
        predict_start = tf["predict_start"]
        predict_end = tf["predict_end"]
        tf_start = time.time()

        print(f"\n── Timeframe {label} ({ti + 1}/{len(timeframes)}) ──")

        predict_months = sorted([m for m in all_months if predict_start <= m <= predict_end])
        if not predict_months:
            print(f"  [{_ts()}] No predict months — skipping")
            continue

        train_months = [m for m in all_months if earliest_month <= m <= train_end]
        if len(train_months) < 3:
            print(f"  [{_ts()}] Insufficient training months ({len(train_months)}) — need 3 min — skipping")
            continue

        # Select DFUs based on strategy
        if args.cluster_strategy == "per_cluster":
            active_dfus = dfu_keys[
                dfu_keys["dfu_ck"].map(cluster_map).notna()
                & (dfu_keys["dfu_ck"].map(cluster_map) != "__unknown__")
            ]
            n_skipped = len(dfu_keys) - len(active_dfus)
            if n_skipped > 0:
                print(f"  [{_ts()}] Per-cluster: fitting {len(active_dfus):,} clustered DFUs "
                      f"(skipping {n_skipped} unassigned)")
        else:
            active_dfus = dfu_keys

        if args.cluster_strategy == "pooled":
            print(f"  [{_ts()}] Pooled strategy: aggregating by cluster, fitting Prophet per cluster...")
            preds = fit_prophet_pooled(
                sales_df, active_dfus, cluster_map,
                train_end, predict_months,
                n_workers=args.n_workers,
                prophet_kwargs=prophet_kwargs,
            )
        else:
            print(f"  [{_ts()}] Fitting Prophet per DFU ({len(active_dfus):,} DFUs, "
                  f"{len(predict_months)} predict months, {args.n_workers} workers)...")
            preds = fit_prophet_parallel(
                sales_df, active_dfus, train_end, predict_months,
                n_workers=args.n_workers,
                prophet_kwargs=prophet_kwargs,
            )

        if len(preds) == 0:
            print(f"  [{_ts()}] No predictions — skipping")
            continue

        preds["model_id"] = model_id
        preds["timeframe"] = label
        preds["timeframe_idx"] = tf["index"]
        all_predictions.append(preds)
        print(f"  [{_ts()}] Timeframe {label} complete: {len(preds):,} predictions ({time.time() - tf_start:.1f}s)")

    if not all_predictions:
        print(f"\n[{_ts()}] No predictions generated. Check data range and timeframe count.")
        sys.exit(1)

    # ── Step 4: Combine, assign execution lag, attach actuals ────────────────
    print(f"\n[{_ts()}] Step 4: Combining predictions...")
    combined = pd.concat(all_predictions, ignore_index=True)
    print(f"  [{_ts()}] Total raw predictions: {len(combined):,}")

    combined["startdate"] = pd.to_datetime(combined["startdate"])

    print(f"  [{_ts()}] Assigning execution lag per DFU...")
    expanded = assign_execution_lag(combined, exec_lag_map)
    print(f"  [{_ts()}] Rows after execution-lag assignment: {len(expanded):,}")

    # Deduplicate: for same (forecast_ck, model_id), keep latest timeframe
    expanded = expanded.sort_values("timeframe_idx")
    expanded = expanded.drop_duplicates(subset=["forecast_ck", "model_id"], keep="last")
    print(f"  [{_ts()}] After dedup: {len(expanded):,}")

    # Attach actuals
    print(f"  [{_ts()}] Attaching actuals...")
    t1 = time.time()
    actuals = sales_df.drop_duplicates(subset=["dfu_ck", "startdate"])[["dfu_ck", "startdate", "qty"]].rename(
        columns={"qty": "tothist_dmd"}
    )
    expanded = expanded.merge(actuals, on=["dfu_ck", "startdate"], how="left")
    print(f"  [{_ts()}] Actuals attached ({time.time() - t1:.1f}s)")

    # ── Step 5: Save output ──────────────────────────────────────────────────
    print(f"\n[{_ts()}] Step 5: Saving output...")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    out_cols = [
        "forecast_ck", "dmdunit", "dmdgroup", "loc",
        "fcstdate", "startdate", "lag", "execution_lag",
        "basefcst_pref", "tothist_dmd", "model_id",
    ]
    output_df = expanded[out_cols].copy()
    output_df["fcstdate"] = output_df["fcstdate"].dt.strftime("%Y-%m-%d")
    output_df["startdate"] = output_df["startdate"].dt.strftime("%Y-%m-%d")

    output_path = output_dir / "backtest_predictions.csv"
    output_df.to_csv(output_path, index=False)
    print(f"  [{_ts()}] Saved {len(output_df):,} predictions to {output_path}")

    # ── Step 5b: All-lags archive CSV ────────────────────────────────────────
    print(f"  [{_ts()}] Generating all-lags archive (lag 0-4)...")
    archive_expanded = expand_to_all_lags(combined, 4, exec_lag_map)

    archive_expanded = archive_expanded.sort_values("timeframe_idx")
    archive_expanded = archive_expanded.drop_duplicates(
        subset=["forecast_ck", "model_id", "lag"], keep="last"
    )
    print(f"  [{_ts()}] Archive after dedup: {len(archive_expanded):,}")

    archive_expanded = archive_expanded.merge(actuals, on=["dfu_ck", "startdate"], how="left")

    archive_cols = [
        "forecast_ck", "dmdunit", "dmdgroup", "loc",
        "fcstdate", "startdate", "lag", "execution_lag",
        "basefcst_pref", "tothist_dmd", "model_id", "timeframe",
    ]
    archive_df = archive_expanded[archive_cols].copy()
    archive_df["fcstdate"] = archive_df["fcstdate"].dt.strftime("%Y-%m-%d")
    archive_df["startdate"] = archive_df["startdate"].dt.strftime("%Y-%m-%d")

    archive_path = output_dir / "backtest_predictions_all_lags.csv"
    archive_df.to_csv(archive_path, index=False)
    print(f"  [{_ts()}] Saved {len(archive_df):,} archive rows to {archive_path}")

    # Save metadata
    metadata = {
        "model_id": model_id,
        "cluster_strategy": args.cluster_strategy,
        "n_timeframes": args.n_timeframes,
        "n_workers": args.n_workers,
        "prophet_kwargs": {k: str(v) for k, v in prophet_kwargs.items()},
        "n_predictions": len(output_df),
        "n_dfus": int(output_df["dmdunit"].nunique()),
        "date_range": {
            "earliest": str(earliest_month.date()),
            "latest": str(latest_month.date()),
        },
        "timeframes": [
            {
                "label": tf["label"],
                "train_end": str(tf["train_end"].date()),
                "predict_start": str(tf["predict_start"].date()),
                "predict_end": str(tf["predict_end"].date()),
            }
            for tf in timeframes
        ],
    }

    # Compute accuracy summary
    at_exec = output_df.copy()
    at_exec["basefcst_pref"] = pd.to_numeric(at_exec["basefcst_pref"], errors="coerce")
    at_exec["tothist_dmd"] = pd.to_numeric(at_exec["tothist_dmd"], errors="coerce")
    at_exec = at_exec.dropna(subset=["basefcst_pref", "tothist_dmd"])

    if len(at_exec) > 0 and at_exec["tothist_dmd"].abs().sum() > 0:
        total_f = at_exec["basefcst_pref"].sum()
        total_a = at_exec["tothist_dmd"].sum()
        abs_error = (at_exec["basefcst_pref"] - at_exec["tothist_dmd"]).abs().sum()
        wape = 100 * abs_error / abs(total_a) if abs(total_a) > 0 else None
        bias = (total_f / total_a) - 1 if abs(total_a) > 0 else None
        accuracy = 100 - wape if wape is not None else None
        metadata["accuracy_at_execution_lag"] = {
            "n_rows": len(at_exec),
            "wape": round(float(wape), 2) if wape else None,
            "bias": round(float(bias), 4) if bias else None,
            "accuracy_pct": round(float(accuracy), 2) if accuracy else None,
        }
        print(f"\n  Accuracy at execution lag ({len(at_exec):,} rows):")
        print(f"    WAPE: {wape:.2f}%")
        print(f"    Bias: {bias:.4f}")
        print(f"    Accuracy: {accuracy:.2f}%")

    meta_path = output_dir / "backtest_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  [{_ts()}] Saved metadata to {meta_path}")

    # ── Step 6: MLflow logging ───────────────────────────────────────────────
    try:
        import mlflow

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5003")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("demand_backtest")

        with mlflow.start_run():
            mlflow.set_tag("model_type", "prophet_backtest")
            mlflow.set_tag("cluster_strategy", args.cluster_strategy)
            mlflow.set_tag("model_id", model_id)

            _mlflow_params = {
                "n_timeframes": args.n_timeframes,
                "cluster_strategy": args.cluster_strategy,
                "n_workers": args.n_workers,
                "growth": args.growth,
                "changepoint_prior_scale": args.changepoint_prior_scale,
                "seasonality_prior_scale": args.seasonality_prior_scale,
            }
            mlflow.log_params(_mlflow_params)

            mlflow.log_metrics({
                "n_predictions": len(output_df),
                "n_dfus": int(output_df["dmdunit"].nunique()),
            })
            if "accuracy_at_execution_lag" in metadata:
                acc = metadata["accuracy_at_execution_lag"]
                if acc.get("wape"):
                    mlflow.log_metric("wape", acc["wape"])
                if acc.get("accuracy_pct"):
                    mlflow.log_metric("accuracy_pct", acc["accuracy_pct"])
                if acc.get("bias"):
                    mlflow.log_metric("bias", acc["bias"])

            mlflow.log_artifact(str(output_path))
            mlflow.log_artifact(str(archive_path))
            mlflow.log_artifact(str(meta_path))

            print(f"\n  [{_ts()}] Logged to MLflow: {mlflow.get_artifact_uri()}")
    except Exception as e:
        print(f"\n  [{_ts()}] MLflow logging skipped: {e}")

    elapsed = time.time() - t_start
    print(f"\n[{_ts()}] Prophet backtest complete in {elapsed:.0f}s ({elapsed / 60:.1f}m)")


if __name__ == "__main__":
    main()
