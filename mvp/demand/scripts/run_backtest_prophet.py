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
import logging
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Suppress Prophet's verbose logging
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*cmdstan.*")
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.backtest_framework import (
    generate_timeframes,
    load_backtest_data,
    postprocess_predictions,
    save_backtest_output,
)
from common.db import get_db_params
from common.mlflow_utils import log_backtest_run


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ── Per-DFU Prophet fitting ──────────────────────────────────────────────────


def _fit_single_dfu(args_tuple: tuple) -> list[dict] | None:
    """Fit Prophet for one DFU and return predictions.

    Designed to be called via multiprocessing.Pool.map.
    Returns list of dicts with (dfu_ck, dmdunit, dmdgroup, loc, startdate, basefcst_pref)
    or None if fitting fails.
    """
    dfu_ck, dmdunit, dmdgroup, loc, train_series, predict_months, prophet_kwargs = args_tuple

    from prophet import Prophet

    if len(train_series) < 2:
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
                "dfu_ck": dfu_ck,
                "dmdunit": dmdunit,
                "dmdgroup": dmdgroup,
                "loc": loc,
                "startdate": row["ds"],
                "basefcst_pref": max(float(row["yhat"]), 0),
            })
        return results

    except Exception:
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

    train_sales = sales_df[sales_df["startdate"] <= train_end].copy()
    grouped = train_sales.groupby("dfu_ck").apply(
        lambda g: g.set_index("startdate")["qty"], include_groups=False
    )

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

    actual_workers = min(n_workers, len(work_items), cpu_count())
    if actual_workers <= 1:
        all_results = [_fit_single_dfu(item) for item in work_items]
    else:
        with Pool(processes=actual_workers) as pool:
            all_results = pool.map(_fit_single_dfu, work_items)

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

    train_sales = train_sales[train_sales["cluster"].notna() & (train_sales["cluster"] != "__unknown__")]

    cluster_agg = train_sales.groupby(["cluster", "startdate"])["qty"].sum().reset_index()

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

    cluster_rows = []
    for result in all_results:
        if result is not None:
            cluster_rows.extend(result)

    if not cluster_rows:
        return pd.DataFrame(columns=["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"])

    cluster_fcst = pd.DataFrame(cluster_rows)

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

    dfu_lookup = dfu_keys.set_index("dfu_ck")[["dmdunit", "dmdgroup", "loc"]]
    dfu_props = dfu_cluster_sales[["dfu_ck", "cluster", "proportion"]].copy()

    disagg = dfu_props.merge(cluster_fcst, on="cluster")
    disagg["basefcst_pref"] = np.maximum(disagg["proportion"] * disagg["cluster_forecast"], 0)

    disagg = disagg.merge(dfu_lookup, left_on="dfu_ck", right_index=True)

    return disagg[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"]].copy()


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
    db = get_db_params()
    sales_df, dfu_attrs, _ = load_backtest_data(db, include_item_attrs=False)

    exec_lag_map = dfu_attrs.set_index("dfu_ck")["execution_lag"].fillna(0).astype(int).to_dict()
    cluster_map = dfu_attrs.set_index("dfu_ck")["ml_cluster"].to_dict()
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
    expanded, archive_expanded, combined = postprocess_predictions(
        all_predictions, sales_df, exec_lag_map
    )

    # ── Step 5: Save output ──────────────────────────────────────────────────
    print(f"\n[{_ts()}] Step 5: Saving output...")
    output_dir = ROOT / args.output_dir
    output_path, archive_path, meta_path, metadata = save_backtest_output(
        output_df=expanded,
        archive_df=archive_expanded,
        output_dir=output_dir,
        model_id=model_id,
        cluster_strategy=args.cluster_strategy,
        n_timeframes=args.n_timeframes,
        model_params=prophet_kwargs,
        model_params_key="prophet_kwargs",
        timeframes=timeframes,
        earliest_month=earliest_month,
        latest_month=latest_month,
        extra_metadata={"n_workers": args.n_workers},
    )

    # ── Step 6: MLflow logging ───────────────────────────────────────────────
    log_backtest_run(
        model_type="prophet_backtest",
        model_id=model_id,
        cluster_strategy=args.cluster_strategy,
        hyperparams={
            "n_timeframes": args.n_timeframes,
            "cluster_strategy": args.cluster_strategy,
            "n_workers": args.n_workers,
            "growth": args.growth,
            "changepoint_prior_scale": args.changepoint_prior_scale,
            "seasonality_prior_scale": args.seasonality_prior_scale,
        },
        metrics={
            "n_predictions": len(expanded),
            "n_dfus": int(expanded["dmdunit"].nunique()),
        },
        metadata=metadata,
        artifact_paths=[str(output_path), str(archive_path), str(meta_path)],
    )

    elapsed = time.time() - t_start
    print(f"\n[{_ts()}] Prophet backtest complete in {elapsed:.0f}s ({elapsed / 60:.1f}m)")


if __name__ == "__main__":
    main()
