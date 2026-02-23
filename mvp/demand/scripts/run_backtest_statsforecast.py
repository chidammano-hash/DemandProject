"""
Run StatsForecast backtesting with expanding-window timeframes.

StatsForecast (Nixtla) provides vectorized statistical models that process
ALL time series as a single DataFrame — no per-DFU fitting loop needed.
This makes it ~100x faster than Prophet for large-scale backtesting.

Supports three strategies:
  - global:      Fit all DFUs at once (model_id=statsforecast_global)
  - per_cluster: Fit only DFUs within assigned clusters (model_id=statsforecast_cluster)
  - pooled:      Aggregate by cluster -> fit -> disaggregate proportionally (model_id=statsforecast_pooled)

Produces two CSVs:
  - backtest_predictions.csv: execution-lag only (for fact_external_forecast_monthly)
  - backtest_predictions_all_lags.csv: lag 0-4 archive (for backtest_lag_archive)
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

logging.disable(logging.INFO)
warnings.filterwarnings("ignore")

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


# -- StatsForecast fitting ---------------------------------------------------


def _fit_and_predict_batch(
    sales_df: pd.DataFrame,
    dfu_keys: pd.DataFrame,
    train_end: pd.Timestamp,
    predict_months: list[pd.Timestamp],
    season_length: int = 12,
    n_jobs: int = -1,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Fit StatsForecast models on all DFUs in a single batch call.

    Returns DataFrame with columns: dfu_ck, dmdunit, dmdgroup, loc, startdate, basefcst_pref
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive

    if model_names is None:
        model_names = ["AutoARIMA", "AutoETS"]

    # Build model list
    models = []
    for name in model_names:
        if name == "AutoARIMA":
            models.append(AutoARIMA(season_length=season_length))
        elif name == "AutoETS":
            models.append(AutoETS(season_length=season_length))
        elif name == "SeasonalNaive":
            models.append(SeasonalNaive(season_length=season_length))

    if not models:
        models = [AutoARIMA(season_length=season_length)]

    # Prepare training data in StatsForecast format: unique_id, ds, y
    train_sales = sales_df[sales_df["startdate"] <= train_end].copy()

    # Filter to DFUs in dfu_keys
    valid_dfus = set(dfu_keys["dfu_ck"].values)
    train_sales = train_sales[train_sales["dfu_ck"].isin(valid_dfus)]

    if len(train_sales) == 0:
        return pd.DataFrame(columns=["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"])

    sf_df = train_sales[["dfu_ck", "startdate", "qty"]].rename(
        columns={"dfu_ck": "unique_id", "startdate": "ds", "qty": "y"}
    )
    sf_df = sf_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Filter series with fewer than 3 observations
    counts = sf_df.groupby("unique_id").size()
    valid_ids = counts[counts >= 3].index
    sf_df = sf_df[sf_df["unique_id"].isin(valid_ids)]

    if len(sf_df) == 0:
        return pd.DataFrame(columns=["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"])

    # Fit and predict
    h = len(predict_months)
    sf = StatsForecast(models=models, freq="MS", n_jobs=n_jobs)

    try:
        forecast = sf.forecast(df=sf_df, h=h)
    except Exception as e:
        print(f"    [{_ts()}] StatsForecast batch fitting failed: {e}", flush=True)
        return pd.DataFrame(columns=["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"])

    # forecast has index=unique_id, columns include ds and model columns
    forecast = forecast.reset_index()

    # Pick best model column: prefer AutoARIMA, fallback to first available
    model_cols = [c for c in forecast.columns if c not in ("unique_id", "ds")]
    primary_col = None
    for preferred in ["AutoARIMA", "AutoETS", "SeasonalNaive"]:
        if preferred in model_cols:
            primary_col = preferred
            break
    if primary_col is None and model_cols:
        primary_col = model_cols[0]

    if primary_col is None:
        return pd.DataFrame(columns=["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"])

    # Build result
    result = forecast[["unique_id", "ds", primary_col]].copy()
    result.columns = ["dfu_ck", "startdate", "basefcst_pref"]
    result["basefcst_pref"] = result["basefcst_pref"].clip(lower=0)

    # Merge with DFU metadata
    dfu_lookup = dfu_keys.set_index("dfu_ck")[["dmdunit", "dmdgroup", "loc"]]
    result = result.merge(dfu_lookup, left_on="dfu_ck", right_index=True, how="left")

    # Add zero predictions for DFUs that were filtered out (too few observations)
    fitted_dfus = set(result["dfu_ck"].unique())
    missing_dfus = dfu_keys[~dfu_keys["dfu_ck"].isin(fitted_dfus)]
    if len(missing_dfus) > 0:
        zero_rows = []
        for _, row in missing_dfus.iterrows():
            for pm in predict_months:
                zero_rows.append({
                    "dfu_ck": row["dfu_ck"],
                    "dmdunit": row["dmdunit"],
                    "dmdgroup": row["dmdgroup"],
                    "loc": row["loc"],
                    "startdate": pm,
                    "basefcst_pref": 0.0,
                })
        if zero_rows:
            result = pd.concat([result, pd.DataFrame(zero_rows)], ignore_index=True)

    return result[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"]]


def _fit_and_predict_pooled(
    sales_df: pd.DataFrame,
    dfu_keys: pd.DataFrame,
    cluster_map: dict[str, str],
    train_end: pd.Timestamp,
    predict_months: list[pd.Timestamp],
    season_length: int = 12,
    n_jobs: int = -1,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Pooled cluster strategy: aggregate by cluster -> fit -> disaggregate."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive

    if model_names is None:
        model_names = ["AutoARIMA", "AutoETS"]

    models = []
    for name in model_names:
        if name == "AutoARIMA":
            models.append(AutoARIMA(season_length=season_length))
        elif name == "AutoETS":
            models.append(AutoETS(season_length=season_length))
        elif name == "SeasonalNaive":
            models.append(SeasonalNaive(season_length=season_length))
    if not models:
        models = [AutoARIMA(season_length=season_length)]

    train_sales = sales_df[sales_df["startdate"] <= train_end].copy()
    train_sales["cluster"] = train_sales["dfu_ck"].map(cluster_map)
    train_sales = train_sales[train_sales["cluster"].notna() & (train_sales["cluster"] != "__unknown__")]

    # Aggregate by cluster
    cluster_agg = train_sales.groupby(["cluster", "startdate"])["qty"].sum().reset_index()

    sf_df = cluster_agg.rename(columns={"cluster": "unique_id", "startdate": "ds", "qty": "y"})
    sf_df = sf_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Filter clusters with fewer than 3 observations
    counts = sf_df.groupby("unique_id").size()
    valid_ids = counts[counts >= 3].index
    sf_df = sf_df[sf_df["unique_id"].isin(valid_ids)]

    if len(sf_df) == 0:
        return pd.DataFrame(columns=["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"])

    h = len(predict_months)
    sf = StatsForecast(models=models, freq="MS", n_jobs=n_jobs)

    try:
        forecast = sf.forecast(df=sf_df, h=h)
    except Exception as e:
        print(f"    [{_ts()}] StatsForecast pooled fitting failed: {e}", flush=True)
        return pd.DataFrame(columns=["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"])

    forecast = forecast.reset_index()

    model_cols = [c for c in forecast.columns if c not in ("unique_id", "ds")]
    primary_col = None
    for preferred in ["AutoARIMA", "AutoETS", "SeasonalNaive"]:
        if preferred in model_cols:
            primary_col = preferred
            break
    if primary_col is None and model_cols:
        primary_col = model_cols[0]

    if primary_col is None:
        return pd.DataFrame(columns=["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"])

    cluster_fcst = forecast[["unique_id", "ds", primary_col]].copy()
    cluster_fcst.columns = ["cluster", "startdate", "cluster_forecast"]
    cluster_fcst["cluster_forecast"] = cluster_fcst["cluster_forecast"].clip(lower=0)

    # Compute DFU proportions within clusters
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

    # Disaggregate
    dfu_lookup = dfu_keys.set_index("dfu_ck")[["dmdunit", "dmdgroup", "loc"]]
    dfu_props = dfu_cluster_sales[["dfu_ck", "cluster", "proportion"]].copy()

    disagg = dfu_props.merge(cluster_fcst, on="cluster")
    disagg["basefcst_pref"] = np.maximum(disagg["proportion"] * disagg["cluster_forecast"], 0)
    disagg = disagg.merge(dfu_lookup, left_on="dfu_ck", right_index=True)

    return disagg[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"]].copy()


# -- Main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run StatsForecast backtest with expanding-window timeframes")
    parser.add_argument("--cluster-strategy", choices=["global", "per_cluster", "pooled"], default="global",
                        help="global: fit all DFUs, per_cluster: fit only clustered DFUs, pooled: aggregate by cluster")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model_id (default: statsforecast_global, etc.)")
    parser.add_argument("--n-timeframes", type=int, default=10, help="Number of expanding windows")
    parser.add_argument("--output-dir", type=str, default="data/backtest", help="Output directory")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel jobs for StatsForecast (-1 = all CPUs)")

    # StatsForecast-specific parameters
    parser.add_argument("--models", type=str, default="AutoARIMA,AutoETS",
                        help="Comma-separated model names: AutoARIMA, AutoETS, SeasonalNaive")
    parser.add_argument("--season-length", type=int, default=12,
                        help="Seasonal period (12 for monthly data)")

    args = parser.parse_args()

    t_start = time.time()
    load_dotenv(ROOT / ".env")

    _default_model_ids = {
        "global": "statsforecast_global",
        "per_cluster": "statsforecast_cluster",
        "pooled": "statsforecast_pooled",
    }
    model_id = args.model_id or _default_model_ids[args.cluster_strategy]
    model_names = [m.strip() for m in args.models.split(",")]

    print(f"[{_ts()}] Backtest: strategy={args.cluster_strategy}, model_id={model_id}, "
          f"n_timeframes={args.n_timeframes}, models={model_names}")
    print(f"[{_ts()}] StatsForecast: season_length={args.season_length}, n_jobs={args.n_jobs}")

    print(f"[{_ts()}] Loading data...")
    db = get_db_params()
    sales_df, dfu_attrs, _ = load_backtest_data(db, include_item_attrs=False)

    exec_lag_map = dfu_attrs.set_index("dfu_ck")["execution_lag"].fillna(0).astype(int).to_dict()
    cluster_map = dfu_attrs.set_index("dfu_ck")["ml_cluster"].to_dict()
    dfu_keys = dfu_attrs[["dfu_ck", "dmdunit", "dmdgroup", "loc"]].drop_duplicates()

    latest_month = sales_df["startdate"].max()
    earliest_month = sales_df["startdate"].min()
    print(f"[{_ts()}] {len(dfu_keys):,} DFUs, range {earliest_month.date()}->{latest_month.date()}")

    timeframes = generate_timeframes(earliest_month, latest_month, args.n_timeframes)
    print(f"[{_ts()}] {len(timeframes)} timeframes: {timeframes[0]['label']}-{timeframes[-1]['label']}")

    all_months = sorted(sales_df["startdate"].unique())

    sf_kwargs = {
        "season_length": args.season_length,
        "n_jobs": args.n_jobs,
        "model_names": model_names,
    }

    # -- Train & predict per timeframe ----------------------------------------
    all_predictions = []

    for ti, tf in enumerate(timeframes):
        label = tf["label"]
        train_end = tf["train_end"]
        predict_start = tf["predict_start"]
        predict_end = tf["predict_end"]
        tf_start = time.time()

        predict_months = sorted([m for m in all_months if predict_start <= m <= predict_end])
        if not predict_months:
            continue

        train_months = [m for m in all_months if earliest_month <= m <= train_end]
        if len(train_months) < 3:
            continue

        # Select DFUs based on strategy
        if args.cluster_strategy == "per_cluster":
            active_dfus = dfu_keys[
                dfu_keys["dfu_ck"].map(cluster_map).notna()
                & (dfu_keys["dfu_ck"].map(cluster_map) != "__unknown__")
            ]
        else:
            active_dfus = dfu_keys

        print(f"  [{_ts()}] TF {label} ({ti + 1}/{len(timeframes)}): {len(active_dfus):,} DFUs ...", end="", flush=True)

        if args.cluster_strategy == "pooled":
            preds = _fit_and_predict_pooled(
                sales_df, active_dfus, cluster_map,
                train_end, predict_months,
                **sf_kwargs,
            )
        else:
            preds = _fit_and_predict_batch(
                sales_df, active_dfus, train_end, predict_months,
                **sf_kwargs,
            )

        if len(preds) == 0:
            print(" skipped")
            continue

        preds["model_id"] = model_id
        preds["timeframe"] = label
        preds["timeframe_idx"] = tf["index"]
        all_predictions.append(preds)
        tf_elapsed = time.time() - tf_start
        total_elapsed = time.time() - t_start
        remaining_tfs = len(timeframes) - (ti + 1)
        eta_min = (total_elapsed / (ti + 1) * remaining_tfs) / 60
        print(f" {len(preds):,} preds ({tf_elapsed:.0f}s) | ETA ~{eta_min:.0f}m")

    if not all_predictions:
        print(f"\n[{_ts()}] No predictions generated. Check data range and timeframe count.")
        sys.exit(1)

    print(f"[{_ts()}] Post-processing...")
    expanded, archive_expanded, combined = postprocess_predictions(
        all_predictions, sales_df, exec_lag_map
    )

    print(f"[{_ts()}] Saving output...")
    output_dir = ROOT / args.output_dir
    output_path, archive_path, meta_path, metadata = save_backtest_output(
        output_df=expanded,
        archive_df=archive_expanded,
        output_dir=output_dir,
        model_id=model_id,
        cluster_strategy=args.cluster_strategy,
        n_timeframes=args.n_timeframes,
        model_params=sf_kwargs,
        model_params_key="statsforecast_kwargs",
        timeframes=timeframes,
        earliest_month=earliest_month,
        latest_month=latest_month,
        extra_metadata={"models": model_names},
    )

    log_backtest_run(
        model_type="statsforecast_backtest",
        model_id=model_id,
        cluster_strategy=args.cluster_strategy,
        hyperparams={
            "n_timeframes": args.n_timeframes,
            "cluster_strategy": args.cluster_strategy,
            "season_length": args.season_length,
            "models": model_names,
            "n_jobs": args.n_jobs,
        },
        metrics={
            "n_predictions": len(expanded),
            "n_dfus": int(expanded["dmdunit"].nunique()),
        },
        metadata=metadata,
        artifact_paths=[str(output_path), str(archive_path), str(meta_path)],
    )

    elapsed = time.time() - t_start
    print(f"\n[{_ts()}] StatsForecast backtest complete in {elapsed:.0f}s ({elapsed / 60:.1f}m)")


if __name__ == "__main__":
    main()
