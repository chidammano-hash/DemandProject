"""
Run Chronos 2 Enriched backtest — Chronos 2 with covariate features.

Unlike the zero-shot Chronos 2 backtest, this version passes:
  - past_covariates: lag/rolling/croston/cluster features + categoricals
  - future_covariates: calendar/fourier features (known for any future date)

Produces CSVs under data/backtest/chronos2_enriched/.
"""

import argparse
import logging
import sys
import time
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
from common.ml.feature_engineering import build_feature_matrix, mask_future_sales
from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)

MODEL_ID = "chronos2_enriched"
CONFIG_KEY = "chronos2_enriched"
DISPATCHER_KEY = "chronos2_enriched"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Chronos 2 Enriched backtest (with covariates, all DFUs)",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--loc", type=str, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoints if a previous run crashed")
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

        c2e_cfg = cfg.get("algorithms", {}).get(CONFIG_KEY, {})
        if not c2e_cfg.get("enabled", True):
            logger.info("Chronos 2 Enriched is disabled in config; exiting")
            return

        backtest_cfg = cfg.get("backtest", {})
        n_timeframes = backtest_cfg.get("n_timeframes", 10)
        embargo_months = backtest_cfg.get("embargo_months", 0)

        output_dir = (
            Path(args.output_dir) if args.output_dir
            else ROOT / backtest_cfg.get("output_dir", "data/backtest")
        )

        model_id = c2e_cfg.get("model_id", MODEL_ID)

        c2e_params = {
            "device": c2e_cfg.get("device", "auto"),
            "batch_size": c2e_cfg.get("batch_size", 512),
            "prediction_length": c2e_cfg.get("prediction_length", 6),
        }

    logger.info(
        "Chronos 2 Enriched config: model_id=%s, n_timeframes=%d, "
        "embargo_months=%d, batch_size=%d",
        model_id, n_timeframes, embargo_months, c2e_params["batch_size"],
    )

    # ── Step 1: Load data ───────────────────────────────────────────────────
    logger.info("Step 1: Loading data from Postgres...")
    t_start = time.time()
    db = get_db_params()

    with profiled_section("load_data"):
        sales_df, dfu_attrs, item_attrs = load_backtest_data(db)

    if args.loc:
        loc_filter = args.loc.strip()
        dfu_attrs = dfu_attrs[dfu_attrs["loc"] == loc_filter].copy()
        valid_skus = set(dfu_attrs["sku_ck"])
        sales_df = sales_df[sales_df["sku_ck"].isin(valid_skus)].copy()
        logger.info("Location filter '%s': %d DFUs", loc_filter, len(dfu_attrs))

    if sales_df.empty:
        logger.warning("No sales data found; exiting")
        return

    exec_lag_map = (
        dfu_attrs.set_index("sku_ck")["execution_lag"]
        .fillna(0).astype(int).to_dict()
    )

    dfu_cohort_map: dict[str, str] | None = None
    if "cohort" in dfu_attrs.columns:
        dfu_cohort_map = dfu_attrs.set_index("sku_ck")["cohort"].to_dict()

    logger.info(
        "Data loaded: %s sales rows, %s DFUs (%.1fs)",
        f"{len(sales_df):,}", f"{sales_df['sku_ck'].nunique():,}",
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

    # ── Step 3: Build feature matrix ONCE ───────────────────────────────────
    logger.info("Step 3: Building feature matrix (one-time)...")
    with profiled_section("build_features"):
        full_grid = build_feature_matrix(
            sales_df, dfu_attrs, item_attrs, all_months, cat_dtype="str",
        )
    logger.info("Feature matrix: %s", full_grid.shape)

    # ── Step 4: Run Chronos 2 Enriched per timeframe ───────────────────────
    ckpt = BacktestCheckpointer(output_dir, model_id, resume=args.resume)

    logger.info("Step 4: Running Chronos 2 Enriched across %d timeframes...", len(timeframes))
    all_predictions: list[pd.DataFrame] = []

    dfu_indexed = dfu_attrs.set_index("sku_ck")
    item_id_map = dfu_indexed["item_id"]
    cg_map = dfu_indexed["customer_group"]
    loc_map = dfu_indexed["loc"]

    all_predictions.extend(ckpt.load_all_existing())

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

        # Mask future sales and get feature grid for this timeframe
        with profiled_section(f"mask_tf_{label}"):
            masked_grid = mask_future_sales(full_grid, train_end)

        # Only pass training-period rows as covariates
        train_grid = masked_grid[masked_grid["startdate"] <= train_end].copy()

        train_sales = sales_df[sales_df["startdate"] <= train_end].copy()
        if train_sales.empty:
            logger.info("  No training data -- skipping")
            continue

        logger.info(
            "  Train: %s rows (%s DFUs), Predict months: %d, Features: %d cols",
            f"{len(train_sales):,}",
            f"{train_sales['sku_ck'].nunique():,}",
            len(predict_months),
            train_grid.shape[1],
        )

        with profiled_section(f"c2e_tf_{label}"):
            preds = run_foundation_models(
                train_sales[["sku_ck", "startdate", "qty"]],
                predict_months,
                {DISPATCHER_KEY: c2e_params},
                feature_grid=train_grid,
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
            label, f"{len(preds):,}", f"{preds['sku_ck'].nunique():,}",
            time.time() - tf_start,
        )

    if not all_predictions:
        logger.warning("No predictions produced across any timeframe; exiting")
        return

    # ── Step 5: Post-process ────────────────────────────────────────────────
    logger.info("Step 5: Post-processing predictions...")
    with profiled_section("postprocess"):
        output_df, archive_df, combined_raw = postprocess_predictions(
            all_predictions, sales_df, exec_lag_map, timeframes=timeframes,
        )

    output_df["model_id"] = model_id
    archive_df["model_id"] = model_id

    logger.info(
        "Post-process complete: %s output rows, %s archive rows",
        f"{len(output_df):,}", f"{len(archive_df):,}",
    )

    # ── Step 6: Save output ─────────────────────────────────────────────────
    logger.info("Step 6: Saving backtest output...")
    with profiled_section("save_output"):
        _out_path, _arch_path, _meta_path, metadata = save_backtest_output(
            output_df=output_df,
            archive_df=archive_df,
            output_dir=output_dir,
            model_id=model_id,
            cluster_strategy="global",
            n_timeframes=n_timeframes,
            model_params=c2e_params,
            model_params_key="chronos2_enriched_params",
            timeframes=timeframes,
            earliest_month=earliest_month,
            latest_month=latest_month,
            extra_metadata={
                "params_source": "algorithm_config",
                "model_type": "foundation_model",
                "architecture": "chronos2_enriched",
                "zero_shot": False,
                "covariates": {
                    "past_numeric": [
                        "qty_lag_1", "qty_lag_2", "qty_lag_3", "qty_lag_6", "qty_lag_12",
                        "qty_rolling_mean_3", "qty_rolling_mean_6", "qty_rolling_mean_12",
                        "mom_growth", "demand_accel", "volatility_ratio",
                        "croston_demand_size", "croston_demand_interval", "croston_probability",
                        "cluster_mean_lag1", "cluster_total_lag1", "cluster_demand_trend",
                    ],
                    "past_categorical": ["ml_cluster", "brand", "region", "abc_vol"],
                    "future": [
                        "month", "quarter", "is_quarter_end", "is_year_end", "days_in_month",
                        "fourier_sin_12", "fourier_cos_12", "fourier_sin_6", "fourier_cos_6",
                        "fourier_sin_4", "fourier_cos_4", "fourier_sin_3", "fourier_cos_3",
                    ],
                },
            },
            dfu_cohort_map=dfu_cohort_map,
        )

    ckpt.cleanup()

    total_time = time.time() - t_start
    accuracy = metadata.get("accuracy_overall")
    logger.info(
        "Chronos 2 Enriched complete: accuracy=%.2f%%, %s predictions, %.1f min total",
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
