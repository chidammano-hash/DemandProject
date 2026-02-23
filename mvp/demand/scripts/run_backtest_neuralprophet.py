"""
Run NeuralProphet backtesting with expanding-window timeframes.

NeuralProphet is a PyTorch-based successor to Prophet that supports GPU
acceleration (Apple MPS, NVIDIA CUDA) while maintaining a Prophet-compatible
API.  Like Prophet, it fits per-DFU individual time series models with
native trend and seasonality decomposition, but uses neural network
components for potentially better non-linear pattern capture.

Supports three strategies:
  - global:      Independent NeuralProphet fit per DFU (model_id=neuralprophet_global)
  - per_cluster: Fit only DFUs within assigned clusters (model_id=neuralprophet_cluster)
  - pooled:      Aggregate by cluster -> fit -> disaggregate proportionally (model_id=neuralprophet_pooled)

Produces two CSVs:
  - backtest_predictions.csv: execution-lag only (for fact_external_forecast_monthly)
  - backtest_predictions_all_lags.csv: lag 0-4 archive (for backtest_lag_archive)
"""

import argparse
import logging
import sys
import time
import warnings
import multiprocessing
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Suppress verbose logging (main process)
logging.disable(logging.INFO)
warnings.filterwarnings("ignore")


def _get_mp_context():
    """Return a 'spawn' multiprocessing context.

    PyTorch's MPS backend crashes with fork() on macOS:
      +[MPSGraphObject initialize] may have been in progress in another
      thread when fork() was called.
    Using 'spawn' avoids this by starting fresh worker processes.
    """
    return multiprocessing.get_context("spawn")


def _init_worker():
    """Silence all logging and warnings in spawned worker processes."""
    import os

    logging.disable(logging.INFO)
    warnings.filterwarnings("ignore")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # Force CPU in workers — MPS hangs in spawned subprocesses and adds
    # overhead for tiny per-DFU series anyway.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_MPS_FORCE_CPU"] = "1"
    # Suppress NeuralProphet/PyTorch Lightning logging
    logging.getLogger("neuralprophet").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.WARNING)


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


# -- Per-DFU NeuralProphet fitting -------------------------------------------


def _fit_single_dfu(args_tuple: tuple) -> list[dict] | None:
    """Fit NeuralProphet for one DFU and return predictions.

    Designed to be called via multiprocessing.Pool.map.
    Returns list of dicts with (dfu_ck, dmdunit, dmdgroup, loc, startdate, basefcst_pref)
    or None if fitting fails.
    """
    dfu_ck, dmdunit, dmdgroup, loc, train_series, predict_months, np_kwargs = args_tuple

    from neuralprophet import NeuralProphet, set_log_level

    set_log_level("ERROR")

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

    df_np = pd.DataFrame({
        "ds": train_series.index,
        "y": train_series.values.astype(float),
    })
    df_np = df_np.sort_values("ds").reset_index(drop=True)

    try:
        # Always use CPU in worker processes — MPS hangs in subprocesses
        # and adds overhead for tiny per-DFU time series.
        accelerator = np_kwargs.pop("accelerator", "auto")
        trainer_config = {"accelerator": "cpu"}

        m = NeuralProphet(
            growth=np_kwargs.get("growth", "linear"),
            yearly_seasonality=np_kwargs.get("yearly_seasonality", "auto"),
            weekly_seasonality=np_kwargs.get("weekly_seasonality", False),
            daily_seasonality=np_kwargs.get("daily_seasonality", False),
            n_lags=np_kwargs.get("n_lags", 0),
            learning_rate=np_kwargs.get("learning_rate", 0.1),
            epochs=np_kwargs.get("epochs", 100),
            batch_size=np_kwargs.get("batch_size", 64),
            trainer_config=trainer_config,
        )
        # Restore accelerator for next call
        np_kwargs["accelerator"] = accelerator

        m.fit(df_np, freq="MS")

        future = m.make_future_dataframe(df_np, periods=len(predict_months))
        forecast = m.predict(future)

        # NeuralProphet uses 'yhat1' for the forecast column
        yhat_col = "yhat1" if "yhat1" in forecast.columns else "yhat"

        results = []
        # Get only the future predictions (last N rows)
        future_fcst = forecast.tail(len(predict_months))
        for i, (_, row) in enumerate(future_fcst.iterrows()):
            if i < len(predict_months):
                results.append({
                    "dfu_ck": dfu_ck,
                    "dmdunit": dmdunit,
                    "dmdgroup": dmdgroup,
                    "loc": loc,
                    "startdate": predict_months[i],
                    "basefcst_pref": max(float(row[yhat_col]), 0),
                })
        return results

    except Exception as exc:
        # Log first failure to aid debugging (once per worker)
        if not getattr(_fit_single_dfu, "_logged_error", False):
            import traceback
            print(f"    [NP] DFU {dfu_ck} failed: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            _fit_single_dfu._logged_error = True
        # Restore accelerator if popped
        if "accelerator" not in np_kwargs:
            np_kwargs["accelerator"] = "auto"
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


def fit_neuralprophet_parallel(
    sales_df: pd.DataFrame,
    dfu_keys: pd.DataFrame,
    train_end: pd.Timestamp,
    predict_months: list[pd.Timestamp],
    n_workers: int = 4,
    np_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Fit NeuralProphet per DFU in parallel and return predictions."""
    if np_kwargs is None:
        np_kwargs = {}

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

        work_items.append((dfu_ck, dmdunit, dmdgroup, loc, series, predict_months, np_kwargs.copy()))

    actual_workers = min(n_workers, len(work_items), cpu_count())
    total = len(work_items)
    rows = []
    done = 0
    failed = 0

    def _report_progress(result, dfu_idx):
        nonlocal done, failed
        done += 1
        if result is None:
            failed += 1
        if done % 10_000 == 0 or done == total:
            print(f"    [{_ts()}] {done:,}/{total:,} ({done * 100 // total}%)", flush=True)

    if actual_workers <= 1:
        for i, item in enumerate(work_items):
            result = _fit_single_dfu(item)
            _report_progress(result, i)
            if result is not None:
                rows.extend(result)
    else:
        ctx = _get_mp_context()
        with ctx.Pool(processes=actual_workers, initializer=_init_worker) as pool:
            for i, result in enumerate(pool.imap_unordered(_fit_single_dfu, work_items, chunksize=10)):
                _report_progress(result, i)
                if result is not None:
                    rows.extend(result)

    if not rows:
        return pd.DataFrame(columns=["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"])

    return pd.DataFrame(rows)


# -- Pooled cluster strategy --------------------------------------------------


def _fit_single_cluster_pooled(args_tuple: tuple) -> list[dict] | None:
    """Fit NeuralProphet on aggregated cluster-level sales."""
    cluster_label, train_series, predict_months, np_kwargs = args_tuple

    from neuralprophet import NeuralProphet, set_log_level

    set_log_level("ERROR")

    if len(train_series) < 2:
        return [
            {"cluster": cluster_label, "startdate": pm, "cluster_forecast": 0.0}
            for pm in predict_months
        ]

    df_np = pd.DataFrame({
        "ds": train_series.index,
        "y": train_series.values.astype(float),
    })
    df_np = df_np.sort_values("ds").reset_index(drop=True)

    try:
        accelerator = np_kwargs.pop("accelerator", "auto")
        trainer_config = {"accelerator": "cpu"}

        m = NeuralProphet(
            growth=np_kwargs.get("growth", "linear"),
            yearly_seasonality=np_kwargs.get("yearly_seasonality", "auto"),
            weekly_seasonality=np_kwargs.get("weekly_seasonality", False),
            daily_seasonality=np_kwargs.get("daily_seasonality", False),
            n_lags=np_kwargs.get("n_lags", 0),
            learning_rate=np_kwargs.get("learning_rate", 0.1),
            epochs=np_kwargs.get("epochs", 100),
            batch_size=np_kwargs.get("batch_size", 64),
            trainer_config=trainer_config,
        )
        np_kwargs["accelerator"] = accelerator

        m.fit(df_np, freq="MS")

        future = m.make_future_dataframe(df_np, periods=len(predict_months))
        forecast = m.predict(future)

        yhat_col = "yhat1" if "yhat1" in forecast.columns else "yhat"

        results = []
        future_fcst = forecast.tail(len(predict_months))
        for i, (_, row) in enumerate(future_fcst.iterrows()):
            if i < len(predict_months):
                results.append({
                    "cluster": cluster_label,
                    "startdate": predict_months[i],
                    "cluster_forecast": max(float(row[yhat_col]), 0),
                })
        return results

    except Exception as exc:
        if not getattr(_fit_single_cluster_pooled, "_logged_error", False):
            import traceback
            print(f"    [NP] Cluster {cluster_label} failed: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            _fit_single_cluster_pooled._logged_error = True
        if "accelerator" not in np_kwargs:
            np_kwargs["accelerator"] = "auto"
        return [
            {"cluster": cluster_label, "startdate": pm, "cluster_forecast": 0.0}
            for pm in predict_months
        ]


def fit_neuralprophet_pooled(
    sales_df: pd.DataFrame,
    dfu_keys: pd.DataFrame,
    cluster_map: dict[str, str],
    train_end: pd.Timestamp,
    predict_months: list[pd.Timestamp],
    n_workers: int = 4,
    np_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Pooled cluster strategy: aggregate by cluster -> fit -> disaggregate."""
    if np_kwargs is None:
        np_kwargs = {}

    train_sales = sales_df[sales_df["startdate"] <= train_end].copy()
    train_sales["cluster"] = train_sales["dfu_ck"].map(cluster_map)
    train_sales = train_sales[train_sales["cluster"].notna() & (train_sales["cluster"] != "__unknown__")]

    cluster_agg = train_sales.groupby(["cluster", "startdate"])["qty"].sum().reset_index()

    clusters = sorted(cluster_agg["cluster"].unique())
    work_items = []
    for cluster_label in clusters:
        c_data = cluster_agg[cluster_agg["cluster"] == cluster_label]
        series = c_data.set_index("startdate")["qty"]
        work_items.append((cluster_label, series, predict_months, np_kwargs.copy()))

    actual_workers = min(n_workers, len(work_items), cpu_count())
    total_clusters = len(work_items)
    cluster_rows = []
    done = 0

    if actual_workers <= 1:
        for item in work_items:
            result = _fit_single_cluster_pooled(item)
            done += 1
            if result is not None:
                cluster_rows.extend(result)
    else:
        ctx = _get_mp_context()
        with ctx.Pool(processes=actual_workers, initializer=_init_worker) as pool:
            for result in pool.imap_unordered(_fit_single_cluster_pooled, work_items, chunksize=2):
                done += 1
                if result is not None:
                    cluster_rows.extend(result)
    print(f"    [{_ts()}] {done}/{total_clusters} clusters fitted", flush=True)

    if not cluster_rows:
        return pd.DataFrame(columns=["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"])

    cluster_fcst = pd.DataFrame(cluster_rows)

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

    dfu_lookup = dfu_keys.set_index("dfu_ck")[["dmdunit", "dmdgroup", "loc"]]
    dfu_props = dfu_cluster_sales[["dfu_ck", "cluster", "proportion"]].copy()

    disagg = dfu_props.merge(cluster_fcst, on="cluster")
    disagg["basefcst_pref"] = np.maximum(disagg["proportion"] * disagg["cluster_forecast"], 0)
    disagg = disagg.merge(dfu_lookup, left_on="dfu_ck", right_index=True)

    return disagg[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "basefcst_pref"]].copy()


# -- Main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NeuralProphet backtest with expanding-window timeframes")
    parser.add_argument("--cluster-strategy", choices=["global", "per_cluster", "pooled"], default="global",
                        help="global: per-DFU fits, per_cluster: fit only clustered DFUs, pooled: aggregate by cluster")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model_id (default: neuralprophet_global, etc.)")
    parser.add_argument("--n-timeframes", type=int, default=10, help="Number of expanding windows")
    parser.add_argument("--output-dir", type=str, default="data/backtest", help="Output directory")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Number of parallel workers (default: all CPU cores)")

    # NeuralProphet hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per model")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--n-lags", type=int, default=0,
                        help="Number of autoregressive lags (0 = pure decomposition like Prophet)")
    parser.add_argument("--yearly-seasonality", type=str, default="auto",
                        help="Yearly seasonality: auto, True, False")
    parser.add_argument("--weekly-seasonality", action="store_true", default=False,
                        help="Enable weekly seasonality (default: off for monthly data)")
    parser.add_argument("--daily-seasonality", action="store_true", default=False,
                        help="Enable daily seasonality (default: off for monthly data)")
    parser.add_argument("--growth", type=str, default="linear",
                        choices=["linear", "discontinuous", "off"],
                        help="NeuralProphet growth model")
    parser.add_argument("--accelerator", type=str, default="auto",
                        choices=["auto", "cpu", "gpu", "mps"],
                        help="Hardware accelerator for training")
    args = parser.parse_args()

    t_start = time.time()
    load_dotenv(ROOT / ".env")

    _default_model_ids = {
        "global": "neuralprophet_global",
        "per_cluster": "neuralprophet_cluster",
        "pooled": "neuralprophet_pooled",
    }
    model_id = args.model_id or _default_model_ids[args.cluster_strategy]
    n_workers = args.n_workers if args.n_workers is not None else cpu_count()

    print(f"[{_ts()}] Backtest: strategy={args.cluster_strategy}, model_id={model_id}, "
          f"n_timeframes={args.n_timeframes}, n_workers={n_workers} (CPUs: {cpu_count()})")

    # Parse yearly_seasonality
    yearly = args.yearly_seasonality
    if yearly == "auto":
        pass
    elif yearly.lower() in ("true", "false"):
        yearly = yearly.lower() == "true"
    else:
        yearly = int(yearly)

    np_kwargs = {
        "yearly_seasonality": yearly,
        "weekly_seasonality": args.weekly_seasonality,
        "daily_seasonality": args.daily_seasonality,
        "growth": args.growth,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "n_lags": args.n_lags,
        "accelerator": args.accelerator,
    }
    print(f"[{_ts()}] NeuralProphet: growth={args.growth}, epochs={args.epochs}, "
          f"lr={args.learning_rate}, accelerator={args.accelerator}")

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
            preds = fit_neuralprophet_pooled(
                sales_df, active_dfus, cluster_map,
                train_end, predict_months,
                n_workers=n_workers,
                np_kwargs=np_kwargs.copy(),
            )
        else:
            preds = fit_neuralprophet_parallel(
                sales_df, active_dfus, train_end, predict_months,
                n_workers=n_workers,
                np_kwargs=np_kwargs.copy(),
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
        model_params=np_kwargs,
        model_params_key="neuralprophet_kwargs",
        timeframes=timeframes,
        earliest_month=earliest_month,
        latest_month=latest_month,
        extra_metadata={"n_workers": n_workers},
    )

    log_backtest_run(
        model_type="neuralprophet_backtest",
        model_id=model_id,
        cluster_strategy=args.cluster_strategy,
        hyperparams={
            "n_timeframes": args.n_timeframes,
            "cluster_strategy": args.cluster_strategy,
            "n_workers": n_workers,
            "growth": args.growth,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "n_lags": args.n_lags,
            "accelerator": args.accelerator,
        },
        metrics={
            "n_predictions": len(expanded),
            "n_dfus": int(expanded["dmdunit"].nunique()),
        },
        metadata=metadata,
        artifact_paths=[str(output_path), str(archive_path), str(meta_path)],
    )

    elapsed = time.time() - t_start
    print(f"\n[{_ts()}] NeuralProphet backtest complete in {elapsed:.0f}s ({elapsed / 60:.1f}m)")


if __name__ == "__main__":
    main()
