"""
Run Chronos foundation model backtest on ALL DFUs with expanding-window timeframes.

Chronos is a zero-shot time-series foundation model — no feature engineering,
no per-cluster training, no SHAP. Just raw historical demand in, forecasts out.

Produces two CSVs under data/backtest/chronos/:
  - backtest_predictions.csv          (execution-lag row for DB load)
  - backtest_predictions_all_lags.csv (lag 0-4 archive)
  - backtest_metadata.json            (accuracy stats)
"""

import argparse
import logging
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adv_algorithm_testing.foundation_models import run_foundation_models
from common.core.db import get_db_params
from common.core.planning_date import get_planning_date
from common.ml.backtest_framework import (
    BacktestCheckpointer,
    generate_timeframes,
    load_backtest_data,
    postprocess_predictions,
    save_backtest_output,
)
from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)

MODEL_ID = "chronos"


# ---------------------------------------------------------------------------
# Worker function for parallel timeframe processing
# ---------------------------------------------------------------------------

def _process_timeframe(
    tf: dict,
    ti: int,
    n_total: int,
    sales_path: str,
    all_months_ser: list,
    chronos_params: dict,
    model_id: str,
    item_id_map_path: str,
    cg_map_path: str,
    loc_map_path: str,
) -> pd.DataFrame | None:
    """Process a single timeframe in a worker process.

    Reads data from parquet files to avoid large pickle overhead.
    Each worker loads its own Chronos pipeline (cached within the process).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    wlog = logging.getLogger(f"chronos.worker.{ti}")

    label = tf["label"]
    train_end = pd.Timestamp(tf["train_end"])
    predict_start = pd.Timestamp(tf["predict_start"])
    predict_end = pd.Timestamp(tf["predict_end"])

    wlog.info("Timeframe %s (%d/%d) — worker started", label, ti + 1, n_total)

    all_months = [pd.Timestamp(m) for m in all_months_ser]
    predict_months = [m for m in all_months if predict_start <= m <= predict_end]
    if not predict_months:
        wlog.info("  No predict months — skipping")
        return None

    sales_df = pd.read_parquet(sales_path)
    train_sales = sales_df[sales_df["startdate"] <= train_end].copy()
    if train_sales.empty:
        wlog.info("  No training data — skipping")
        return None

    wlog.info(
        "  Train: %s rows (%s DFUs), Predict months: %d (%s -> %s)",
        f"{len(train_sales):,}",
        f"{train_sales['sku_ck'].nunique():,}",
        len(predict_months),
        predict_months[0].date(),
        predict_months[-1].date(),
    )

    t0 = time.time()
    preds = run_foundation_models(
        train_sales[["sku_ck", "startdate", "qty"]],
        predict_months,
        {"chronos": chronos_params},
    )

    if preds.empty:
        wlog.warning("  Timeframe %s: no predictions produced", label)
        return None

    if "algorithm_id" in preds.columns:
        preds = preds.drop(columns=["algorithm_id"])

    # Enrich with DFU attributes
    item_id_map = pd.read_parquet(item_id_map_path).squeeze()
    cg_map = pd.read_parquet(cg_map_path).squeeze()
    loc_map_s = pd.read_parquet(loc_map_path).squeeze()

    preds["item_id"] = preds["sku_ck"].map(item_id_map).fillna("")
    preds["customer_group"] = preds["sku_ck"].map(cg_map).fillna("")
    preds["loc"] = preds["sku_ck"].map(loc_map_s).fillna("")
    preds["model_id"] = model_id
    preds["timeframe_idx"] = tf["index"]
    preds["timeframe_label"] = label

    wlog.info(
        "  Timeframe %s: %s predictions for %s DFUs (%.1fs)",
        label, f"{len(preds):,}", f"{preds['sku_ck'].nunique():,}",
        time.time() - t0,
    )
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Chronos foundation model backtest (zero-shot, all DFUs)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to algorithm_config.yaml (default: config/algorithm_config.yaml)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: data/backtest from config)",
    )
    parser.add_argument(
        "--skip-load", action="store_true",
        help="Skip DB data loading (for debugging with pre-loaded data)",
    )
    parser.add_argument(
        "--loc", type=str, default=None,
        help="Optional location filter (e.g. 'DC01') to restrict DFUs",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers for timeframe processing (default: 1 = sequential)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoints if a previous run crashed",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    # ── Load config ─────────────────────────────────────────────────────────
    with profiled_section("load_config"):
        config_path = (
            Path(args.config) if args.config
            else ROOT / "config" / "algorithm_config.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        chronos_cfg = cfg.get("algorithms", {}).get("chronos", {})
        if not chronos_cfg.get("enabled", True):
            logger.info("Chronos is disabled in config; exiting")
            return

        backtest_cfg = cfg.get("backtest", {})
        n_timeframes = backtest_cfg.get("n_timeframes", 10)
        embargo_months = backtest_cfg.get("embargo_months", 0)

        output_dir = (
            Path(args.output_dir) if args.output_dir
            else ROOT / backtest_cfg.get("output_dir", "data/backtest")
        )

        model_id = chronos_cfg.get("model_id", MODEL_ID)

        # Chronos-specific params forwarded to run_foundation_models
        chronos_params = {
            "model_size": chronos_cfg.get("model_size", "small"),
            "device": chronos_cfg.get("device", "auto"),
            "batch_size": chronos_cfg.get("batch_size", 256),
            "num_samples": chronos_cfg.get("num_samples", 20),
            "prediction_length": chronos_cfg.get("prediction_length", 6),
        }

    # CLI --workers takes priority; fallback to YAML config
    if args.workers > 1:
        n_workers_cfg = args.workers
    else:
        n_workers_cfg = chronos_cfg.get("num_workers", 1)
        if n_workers_cfg > 1:
            args.workers = n_workers_cfg

    logger.info(
        "Chronos backtest config: model_id=%s, n_timeframes=%d, "
        "embargo_months=%d, model_size=%s, batch_size=%d, workers=%d",
        model_id, n_timeframes, embargo_months,
        chronos_params["model_size"], chronos_params["batch_size"],
        args.workers,
    )

    # ── Step 1: Load data ───────────────────────────────────────────────────
    logger.info("Step 1: Loading data from Postgres...")
    t_start = time.time()
    db = get_db_params()

    with profiled_section("load_data"):
        sales_df, dfu_attrs, _item_attrs = load_backtest_data(
            db, include_item_attrs=False,
        )

    # Apply optional location filter
    if args.loc:
        loc_filter = args.loc.strip()
        dfu_attrs = dfu_attrs[dfu_attrs["loc"] == loc_filter].copy()
        valid_skus = set(dfu_attrs["sku_ck"])
        sales_df = sales_df[sales_df["sku_ck"].isin(valid_skus)].copy()
        logger.info(
            "Location filter '%s': %d DFUs, %d sales rows retained",
            loc_filter, len(dfu_attrs), len(sales_df),
        )

    if sales_df.empty:
        logger.warning("No sales data found; exiting")
        return

    # Execution lag lookup
    exec_lag_map = (
        dfu_attrs.set_index("sku_ck")["execution_lag"]
        .fillna(0).astype(int).to_dict()
    )

    # DFU cohort map for per-cohort accuracy reporting
    dfu_cohort_map: dict[str, str] | None = None
    if "cohort" in dfu_attrs.columns:
        dfu_cohort_map = dfu_attrs.set_index("sku_ck")["cohort"].to_dict()

    logger.info(
        "Data loaded: %s sales rows, %s DFUs (%.1fs)",
        f"{len(sales_df):,}",
        f"{sales_df['sku_ck'].nunique():,}",
        time.time() - t_start,
    )

    # ── Step 2: Generate timeframes ─────────────────────────────────────────
    planning_dt = pd.Timestamp(get_planning_date())
    planning_cutoff = planning_dt.normalize().replace(day=1)
    latest_month = min(sales_df["startdate"].max(), planning_cutoff)
    earliest_month = sales_df["startdate"].min()
    sales_df = sales_df[sales_df["startdate"] <= latest_month].copy()

    logger.info(
        "Date range: %s -> %s (planning date: %s)",
        earliest_month.date(), latest_month.date(), planning_dt.date(),
    )

    with profiled_section("generate_timeframes"):
        timeframes = generate_timeframes(
            earliest_month, latest_month, n_timeframes,
            embargo_months=embargo_months,
        )

    if embargo_months:
        logger.info(
            "Embargo gap: %d month(s) between train_end and predict_start",
            embargo_months,
        )
    logger.info("Step 2: Generated %d timeframes:", len(timeframes))
    for tf in timeframes:
        logger.info(
            "  %s: train [%s -> %s], predict [%s -> %s]",
            tf["label"],
            tf["train_start"].date(), tf["train_end"].date(),
            tf["predict_start"].date(), tf["predict_end"].date(),
        )

    all_months = sorted(sales_df["startdate"].unique())

    # ── Checkpoint manager for incremental saves ─────────────────────────────
    ckpt = BacktestCheckpointer(output_dir, model_id, resume=args.resume)

    # ── Step 3: Run Chronos per timeframe ───────────────────────────────────
    n_workers = args.workers
    logger.info(
        "Step 3: Running Chronos inference across %d timeframes (%d worker%s)...",
        len(timeframes), n_workers, "s" if n_workers > 1 else "",
    )
    all_predictions: list[pd.DataFrame] = []

    # Build DFU attribute lookup Series for enriching predictions (vectorized)
    dfu_indexed = dfu_attrs.set_index("sku_ck")
    item_id_map = dfu_indexed["item_id"]
    cg_map = dfu_indexed["customer_group"]
    loc_map = dfu_indexed["loc"]

    # Load any existing checkpoints first (resume from crash)
    all_predictions.extend(ckpt.load_all_existing())

    if n_workers > 1:
        # ── Parallel path ──────────────────────────────────────────────
        # Write shared data to temp parquet files (avoids large pickle)
        tmp_dir = tempfile.mkdtemp(prefix="chronos_bt_")
        sales_path = os.path.join(tmp_dir, "sales.parquet")
        item_path = os.path.join(tmp_dir, "item_id.parquet")
        cg_path = os.path.join(tmp_dir, "cg.parquet")
        loc_path = os.path.join(tmp_dir, "loc.parquet")

        with profiled_section("write_shared_data"):
            sales_df.to_parquet(sales_path)
            item_id_map.to_frame().to_parquet(item_path)
            cg_map.to_frame().to_parquet(cg_path)
            loc_map.to_frame().to_parquet(loc_path)
        logger.info("Shared data written to %s", tmp_dir)

        # Serialize timeframes (Timestamps -> ISO strings for pickling)
        pending_tfs = []
        for tf in timeframes:
            if ckpt.exists(tf["index"]):
                continue
            pending_tfs.append({
                "label": tf["label"],
                "index": tf["index"],
                "train_start": tf["train_start"].isoformat(),
                "train_end": tf["train_end"].isoformat(),
                "predict_start": tf["predict_start"].isoformat(),
                "predict_end": tf["predict_end"].isoformat(),
            })
        all_months_ser = [m.isoformat() for m in all_months]

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _process_timeframe,
                    stf, ti, len(timeframes),
                    sales_path, all_months_ser, chronos_params,
                    model_id, item_path, cg_path, loc_path,
                ): stf
                for ti, stf in enumerate(pending_tfs)
            }
            for fut in as_completed(futures):
                stf = futures[fut]
                try:
                    result = fut.result()
                    if result is not None and not result.empty:
                        ckpt.save(result, stf["index"])
                        all_predictions.append(result)
                except Exception:
                    logger.exception("Worker failed for timeframe %s", stf["label"])

        # Cleanup temp files
        for p in [sales_path, item_path, cg_path, loc_path]:
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
    else:
        # ── Sequential path ────────────────────────────────────────────
        for ti, tf in enumerate(timeframes):
            if ckpt.exists(tf["index"]):
                logger.info("Timeframe %s (%d/%d) — checkpoint exists, skipping",
                            tf["label"], ti + 1, len(timeframes))
                continue

            label = tf["label"]
            train_end = tf["train_end"]
            predict_start = tf["predict_start"]
            predict_end = tf["predict_end"]
            tf_start = time.time()

            logger.info("Timeframe %s (%d/%d)", label, ti + 1, len(timeframes))

            predict_months = [m for m in all_months if predict_start <= m <= predict_end]
            if not predict_months:
                logger.info("  No predict months -- skipping")
                continue

            train_sales = sales_df[sales_df["startdate"] <= train_end].copy()
            if train_sales.empty:
                logger.info("  No training data -- skipping")
                continue

            logger.info(
                "  Train: %s rows (%s DFUs), Predict months: %d (%s -> %s)",
                f"{len(train_sales):,}",
                f"{train_sales['sku_ck'].nunique():,}",
                len(predict_months),
                predict_months[0].date(),
                predict_months[-1].date(),
            )

            with profiled_section(f"chronos_tf_{label}"):
                preds = run_foundation_models(
                    train_sales[["sku_ck", "startdate", "qty"]],
                    predict_months,
                    {"chronos": chronos_params},
                )

            if preds.empty:
                logger.warning("  Timeframe %s: no predictions produced", label)
                continue

            if "algorithm_id" in preds.columns:
                preds = preds.drop(columns=["algorithm_id"])

            preds["item_id"] = preds["sku_ck"].map(item_id_map).fillna("")
            preds["customer_group"] = preds["sku_ck"].map(cg_map).fillna("")
            preds["loc"] = preds["sku_ck"].map(loc_map).fillna("")
            preds["model_id"] = model_id
            preds["timeframe_idx"] = tf["index"]
            preds["timeframe_label"] = label

            # Persist to disk immediately — survives OOM/crash
            ckpt.save(preds, tf["index"])
            all_predictions.append(preds)
            logger.info(
                "  Timeframe %s: %s predictions for %s DFUs (%.1fs) [checkpointed]",
                label,
                f"{len(preds):,}",
                f"{preds['sku_ck'].nunique():,}",
                time.time() - tf_start,
            )

    if not all_predictions:
        logger.warning("No predictions produced across any timeframe; exiting")
        return

    # ── Step 4: Post-process ────────────────────────────────────────────────
    logger.info("Step 4: Post-processing predictions...")
    with profiled_section("postprocess"):
        output_df, archive_df, combined_raw = postprocess_predictions(
            all_predictions, sales_df, exec_lag_map, timeframes=timeframes,
        )

    # Ensure model_id is set consistently
    output_df["model_id"] = model_id
    archive_df["model_id"] = model_id

    logger.info(
        "Post-process complete: %s output rows, %s archive rows",
        f"{len(output_df):,}", f"{len(archive_df):,}",
    )

    # ── Step 5: Save output ─────────────────────────────────────────────────
    logger.info("Step 5: Saving backtest output...")
    with profiled_section("save_output"):
        _out_path, _arch_path, _meta_path, metadata = save_backtest_output(
            output_df=output_df,
            archive_df=archive_df,
            output_dir=output_dir,
            model_id=model_id,
            cluster_strategy="global",
            n_timeframes=n_timeframes,
            model_params=chronos_params,
            model_params_key="chronos_params",
            timeframes=timeframes,
            earliest_month=earliest_month,
            latest_month=latest_month,
            extra_metadata={
                "params_source": "algorithm_config",
                "model_type": "foundation_model",
                "zero_shot": True,
            },
            dfu_cohort_map=dfu_cohort_map,
        )

    ckpt.cleanup()

    total_time = time.time() - t_start
    accuracy = metadata.get("accuracy_overall")
    logger.info(
        "Chronos backtest complete: accuracy=%.2f%%, %s predictions, %.1f min total",
        accuracy if accuracy is not None else 0.0,
        f"{len(output_df):,}",
        total_time / 60,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
