"""
Run MSTL (Multiple Seasonal-Trend decomposition using LOESS) backtest on ALL DFUs.

MSTL is a statistical model from the statsforecast library that decomposes
time series into trend + multiple seasonal components. Per-DFU fitting
with parallel workers.

Produces two CSVs under data/backtest/mstl/:
  - backtest_predictions.csv          (execution-lag row for DB load)
  - backtest_predictions_all_lags.csv (lag 0-4 archive)
  - backtest_metadata.json            (accuracy stats)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params  # noqa: E402 — after CLI path bootstrap
from common.core.planning_date import get_planning_date  # noqa: E402 — after CLI path bootstrap
from common.ml.backtest_framework import (  # noqa: E402 — after CLI path bootstrap
    BacktestCheckpointer,
    generate_timeframes,
    load_backtest_data,
    postprocess_predictions,
    save_backtest_output,
)
from common.ml.backtest_config import (  # noqa: E402 — after CLI path bootstrap
    BACKTEST_CONFIG_METADATA_KEY,
    build_backtest_config_snapshot,
)
from common.ml.monthly_history import (  # noqa: E402 — after CLI path bootstrap
    select_bounded_history,
)
from common.ml.mstl import run_mstl  # noqa: E402 — after CLI path bootstrap
from common.services.perf_profiler import profiled_section  # noqa: E402 — after CLI path bootstrap

logger = logging.getLogger(__name__)

MODEL_ID = "mstl"
CONFIG_KEY = "mstl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MSTL statistical backtest (per-DFU, all DFUs)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: config/forecasting/forecast_pipeline_config.yaml)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: data/backtest from config)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override configured parallel workers for per-DFU fitting",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoints if a previous run crashed",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    load_dotenv(ROOT / ".env")

    # ── Load config ─────────────────────────────────────────────────────────
    with profiled_section("load_config"):
        if args.config:
            config_path = Path(args.config)
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        else:
            from common.core.utils import load_forecast_pipeline_config
            cfg = load_forecast_pipeline_config()

        backtest_config_snapshot = build_backtest_config_snapshot(cfg, MODEL_ID)

        algo_entry = cfg["algorithms"][CONFIG_KEY]
        mstl_cfg = algo_entry["params"]
        if not algo_entry["enabled"]:
            logger.info("MSTL is disabled in config; exiting")
            return

        backtest_cfg = cfg["backtest"]
        history_lookback_months = int(cfg["production_forecast"]["lookback_months"])
        n_timeframes = int(backtest_cfg["n_timeframes"])
        embargo_months = int(backtest_cfg["embargo_months"])

        output_dir = (
            Path(args.output_dir) if args.output_dir
            else ROOT / backtest_cfg["output_dir"]
        )

        model_id = MODEL_ID
        mstl_params = {
            "season_length": int(mstl_cfg["season_length"]),
            "min_history": int(mstl_cfg["min_history"]),
            "num_workers": int(mstl_cfg["num_workers"]),
        }
        n_workers = args.workers if args.workers is not None else mstl_params["num_workers"]

    logger.info(
        "MSTL backtest config: model_id=%s, n_timeframes=%d, "
        "embargo_months=%d, workers=%d, season_length=%d",
        model_id, n_timeframes, embargo_months,
        n_workers, mstl_params["season_length"],
    )

    # ── Step 1: Load data ───────────────────────────────────────────────────
    logger.info("Step 1: Loading data from Postgres...")
    t_start = time.time()
    db = get_db_params()

    with profiled_section("load_data"):
        sales_df, dfu_attrs, _item_attrs = load_backtest_data(
            db, include_item_attrs=False,
        )

    if sales_df.empty:
        raise RuntimeError("MSTL backtest cannot run without sales data")

    exec_lag_map = (
        dfu_attrs.set_index("sku_ck")["execution_lag"]
        .fillna(0).astype(int).to_dict()
    )

    dfu_cohort_map: dict[str, str] | None = None
    if "cohort" in dfu_attrs.columns:
        dfu_cohort_map = dfu_attrs.set_index("sku_ck")["cohort"].to_dict()

    dfu_indexed = dfu_attrs.set_index("sku_ck")
    item_id_map = dfu_indexed["item_id"]
    cg_map = dfu_indexed["customer_group"]
    loc_map = dfu_indexed["loc"]

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

    logger.info("Step 2: Generated %d timeframes:", len(timeframes))
    for tf in timeframes:
        logger.info(
            "  %s: train [%s -> %s], predict [%s -> %s]",
            tf["label"],
            tf["train_start"].date(), tf["train_end"].date(),
            tf["predict_start"].date(), tf["predict_end"].date(),
        )

    all_months = sorted(sales_df["startdate"].unique())

    # ── Checkpoint manager ──────────────────────────────────────────────────
    ckpt = BacktestCheckpointer(output_dir, model_id, resume=args.resume)

    # ── Step 3: Run MSTL per timeframe ──────────────────────────────────────
    logger.info(
        "Step 3: Running MSTL inference across %d timeframes (%d workers)...",
        len(timeframes), n_workers,
    )
    all_predictions: list[pd.DataFrame] = []
    all_predictions.extend(ckpt.load_all_existing())

    for ti, tf in enumerate(timeframes):
        if ckpt.exists(tf["index"]):
            logger.info(
                "Timeframe %s (%d/%d) — checkpoint exists, skipping",
                tf["label"], ti + 1, len(timeframes),
            )
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

        train_sales = select_bounded_history(
            sales_df,
            history_end=train_end,
            lookback_months=history_lookback_months,
        )
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

        with profiled_section(f"mstl_tf_{label}"):
            preds = run_mstl(
                train_sales[["sku_ck", "startdate", "qty"]],
                predict_months,
                season_length=mstl_params["season_length"],
                min_history=mstl_params["min_history"],
                n_workers=n_workers,
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
        raise RuntimeError("MSTL produced no predictions across any timeframe")

    # ── Step 4: Post-process ────────────────────────────────────────────────
    logger.info("Step 4: Post-processing predictions...")
    with profiled_section("postprocess"):
        output_df, archive_df, _combined_raw = postprocess_predictions(
            all_predictions, sales_df, exec_lag_map, timeframes=timeframes,
        )

    output_df["model_id"] = model_id
    archive_df["model_id"] = model_id

    if output_df.empty or archive_df.empty:
        raise RuntimeError("MSTL post-processing produced empty output artifacts")

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
            model_params=mstl_params,
            model_params_key="mstl_params",
            timeframes=timeframes,
            earliest_month=earliest_month,
            latest_month=latest_month,
            extra_metadata={
                "params_source": "forecast_pipeline_config",
                "model_type": "statistical_upgrade",
                "architecture": "mstl",
                "per_dfu": True,
                "history_lookback_months": history_lookback_months,
                BACKTEST_CONFIG_METADATA_KEY: {
                    model_id: backtest_config_snapshot.as_metadata()
                },
            },
            dfu_cohort_map=dfu_cohort_map,
        )

    ckpt.cleanup()

    total_time = time.time() - t_start
    accuracy = metadata.get("accuracy_overall")
    logger.info(
        "MSTL backtest complete: accuracy=%.2f%%, %s predictions, %.1f min total",
        accuracy if accuracy is not None else 0.0,
        f"{len(output_df):,}",
        total_time / 60,
    )


if __name__ == "__main__":
    main()
