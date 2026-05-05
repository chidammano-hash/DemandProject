"""
Shared scaffolding for foundation model backtests.

All four foundation model backtest scripts (chronos, chronos2, chronos_bolt,
chronos2_enriched) share >90% identical structure.  This module extracts the
common workflow:
  - CLI argument parsing
  - Config loading + model enablement check
  - Data loading from Postgres
  - Location filtering
  - Timeframe generation
  - Checkpoint/resume management
  - Parallel (ProcessPoolExecutor) and sequential timeframe loops
  - DFU attribute enrichment
  - Post-processing and output saving

Each script becomes a thin wrapper that defines:
  - FoundationModelSpec — model identity and config extraction
  - Optionally, a pre-timeframe hook (e.g. feature engineering for enriched)
  - Optionally, a per-timeframe hook (e.g. masking features for enriched)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
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


# ---------------------------------------------------------------------------
# Model spec dataclass — each script populates one of these
# ---------------------------------------------------------------------------

@dataclass
class FoundationModelSpec:
    """Everything that differs between foundation model backtest scripts.

    Attributes:
        model_id: Default model identifier (e.g. "chronos", "chronos_bolt").
        config_key: Key under ``algorithms`` in forecast_pipeline_config.yaml.
        dispatcher_key: Key passed to ``run_foundation_models`` params dict.
        display_name: Human-readable name for log messages.
        extract_params: Callable(cfg_section) -> dict of model-specific params.
        model_params_key: Key name for params in saved metadata JSON.
        extra_metadata: Additional metadata dict merged into backtest output.
        log_config_summary: Callable(model_id, n_timeframes, embargo_months,
            params, n_workers) -> str for the config log line.
        supports_parallel: Whether this model supports parallel workers.
            Set False for enriched models that need feature grids.
        include_item_attrs: Whether to request item_attrs from load_backtest_data.
        include_customer_features: Whether to load customer features from DB.
        pre_timeframe_hook: Optional callable invoked once after data loading
            and before the timeframe loop.  Receives a PreTimeframeContext and
            returns arbitrary state passed to per_timeframe_hook.
        per_timeframe_hook: Optional callable invoked per timeframe (sequential
            path only) to produce model predictions.  When set, replaces the
            default ``run_foundation_models`` call.  Receives a
            PerTimeframeContext and the state from pre_timeframe_hook, and
            returns a DataFrame of predictions (or empty DataFrame).
        profiler_prefix: Prefix for profiled_section labels (e.g. "chronos",
            "c2", "bolt").  Defaults to model_id.
        tmp_dir_prefix: Prefix for tempdir in parallel mode.
    """

    model_id: str
    config_key: str
    dispatcher_key: str
    display_name: str
    extract_params: Callable[[dict], dict]
    model_params_key: str
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    log_config_summary: Callable[..., str] | None = None
    supports_parallel: bool = True
    include_item_attrs: bool = False
    include_customer_features: bool = False
    pre_timeframe_hook: Callable[..., Any] | None = None
    per_timeframe_hook: Callable[..., pd.DataFrame] | None = None
    profiler_prefix: str = ""
    tmp_dir_prefix: str = "foundation_bt_"

    def __post_init__(self) -> None:
        if not self.profiler_prefix:
            self.profiler_prefix = self.model_id


# ---------------------------------------------------------------------------
# Context dataclasses for hooks
# ---------------------------------------------------------------------------

@dataclass
class PreTimeframeContext:
    """Passed to spec.pre_timeframe_hook after data loading."""

    sales_df: pd.DataFrame
    dfu_attrs: pd.DataFrame
    item_attrs: Any  # may be None
    customer_features: Any  # may be None
    all_months: list
    timeframes: list[dict]
    model_params: dict


@dataclass
class PerTimeframeContext:
    """Passed to spec.per_timeframe_hook for each timeframe."""

    tf: dict
    ti: int
    n_total: int
    train_sales: pd.DataFrame
    predict_months: list
    model_params: dict
    dispatcher_key: str
    label: str


# ---------------------------------------------------------------------------
# Worker for parallel timeframe processing
# ---------------------------------------------------------------------------

def _process_timeframe_worker(
    tf: dict,
    ti: int,
    n_total: int,
    sales_path: str,
    all_months_ser: list,
    model_params: dict,
    model_id: str,
    dispatcher_key: str,
    worker_log_name: str,
    item_id_map_path: str,
    cg_map_path: str,
    loc_map_path: str,
) -> pd.DataFrame | None:
    """Process a single timeframe in a worker process.

    Reads data from parquet files to avoid large pickle overhead.
    Each worker loads its own model pipeline (cached within the process).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    wlog = logging.getLogger(f"{worker_log_name}.{ti}")

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
        {dispatcher_key: model_params},
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
        label,
        f"{len(preds):,}",
        f"{preds['sku_ck'].nunique():,}",
        time.time() - t0,
    )
    return preds


# ---------------------------------------------------------------------------
# Enrichment helper (used in both parallel and sequential paths)
# ---------------------------------------------------------------------------

def _enrich_predictions(
    preds: pd.DataFrame,
    item_id_map: pd.Series,
    cg_map: pd.Series,
    loc_map: pd.Series,
    model_id: str,
    tf: dict,
) -> pd.DataFrame:
    """Add DFU attribute columns and timeframe metadata to predictions."""
    if "algorithm_id" in preds.columns:
        preds = preds.drop(columns=["algorithm_id"])

    preds["item_id"] = preds["sku_ck"].map(item_id_map).fillna("")
    preds["customer_group"] = preds["sku_ck"].map(cg_map).fillna("")
    preds["loc"] = preds["sku_ck"].map(loc_map).fillna("")
    preds["model_id"] = model_id
    preds["timeframe_idx"] = tf["index"]
    preds["timeframe_label"] = tf["label"]
    return preds


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_argparser(description: str, *, supports_parallel: bool = True) -> argparse.ArgumentParser:
    """Build the standard CLI argument parser for foundation backtests."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: config/forecasting/forecast_pipeline_config.yaml)",
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
    if supports_parallel:
        parser.add_argument(
            "--workers", type=int, default=1,
            help="Parallel workers for timeframe processing (default: 1 = sequential)",
        )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoints if a previous run crashed",
    )
    return parser


def run_foundation_backtest(spec: FoundationModelSpec, args: argparse.Namespace) -> None:
    """Execute the full foundation model backtest workflow.

    This is the shared main() body.  Each script creates a FoundationModelSpec
    and passes it here with the parsed CLI args.
    """
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

        algo_entry = cfg.get("algorithms", {}).get(spec.config_key, {})
        # Support pipeline config format (params sub-dict) or flat legacy format
        algo_cfg = algo_entry.get("params", algo_entry)
        if not algo_entry.get("enabled", True):
            logger.info("%s is disabled in config; exiting", spec.display_name)
            return

        backtest_cfg = cfg.get("backtest", {})
        n_timeframes = backtest_cfg.get("n_timeframes", 10)
        embargo_months = backtest_cfg.get("embargo_months", 0)

        output_dir = (
            Path(args.output_dir) if args.output_dir
            else ROOT / backtest_cfg.get("output_dir", "data/backtest")
        )

        model_id = algo_entry.get("model_id", algo_cfg.get("model_id", spec.model_id))

        # Extract model-specific params via the spec's callback
        model_params = spec.extract_params(algo_cfg)

    # CLI --workers takes priority; fallback to YAML config
    n_workers = getattr(args, "workers", 1)
    if spec.supports_parallel:
        if n_workers <= 1:
            n_workers_cfg = algo_cfg.get("num_workers", 1)
            if n_workers_cfg > 1:
                n_workers = n_workers_cfg
    else:
        n_workers = 1

    # Log config summary
    if spec.log_config_summary:
        summary = spec.log_config_summary(
            model_id, n_timeframes, embargo_months, model_params, n_workers,
        )
        logger.info("%s", summary)
    else:
        logger.info(
            "%s backtest config: model_id=%s, n_timeframes=%d, "
            "embargo_months=%d, workers=%d",
            spec.display_name, model_id, n_timeframes, embargo_months, n_workers,
        )

    # ── Step 1: Load data ──────────────────────────────────────────────────
    logger.info("Step 1: Loading data from Postgres...")
    t_start = time.time()
    db = get_db_params()

    with profiled_section("load_data"):
        if spec.include_customer_features:
            result = load_backtest_data(db, include_customer_features=True)
            sales_df, dfu_attrs, item_attrs = result[0], result[1], result[2]
            customer_features = result[3] if len(result) > 3 else None
        else:
            sales_df, dfu_attrs, _item_attrs = load_backtest_data(
                db, include_item_attrs=spec.include_item_attrs,
            )
            item_attrs = _item_attrs if spec.include_item_attrs else None
            customer_features = None

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

    # ── Step 2: Generate timeframes ────────────────────────────────────────
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

    # ── Pre-timeframe hook (e.g. build feature matrix for enriched) ────────
    hook_state = None
    if spec.pre_timeframe_hook:
        ctx = PreTimeframeContext(
            sales_df=sales_df,
            dfu_attrs=dfu_attrs,
            item_attrs=item_attrs,
            customer_features=customer_features,
            all_months=all_months,
            timeframes=timeframes,
            model_params=model_params,
        )
        hook_state = spec.pre_timeframe_hook(ctx)

    # ── Checkpoint manager ─────────────────────────────────────────────────
    ckpt = BacktestCheckpointer(output_dir, model_id, resume=args.resume)

    # ── Step 3: Run model per timeframe ────────────────────────────────────
    step_label = "Step 3" if not spec.pre_timeframe_hook else "Step 4"
    logger.info(
        "%s: Running %s inference across %d timeframes (%d worker%s)...",
        step_label, spec.display_name,
        len(timeframes), n_workers, "s" if n_workers > 1 else "",
    )
    all_predictions: list[pd.DataFrame] = []

    # Build DFU attribute lookup Series for enriching predictions
    dfu_indexed = dfu_attrs.set_index("sku_ck")
    item_id_map = dfu_indexed["item_id"]
    cg_map = dfu_indexed["customer_group"]
    loc_map = dfu_indexed["loc"]

    # Load any existing checkpoints first (resume from crash)
    all_predictions.extend(ckpt.load_all_existing())

    if n_workers > 1 and spec.supports_parallel:
        # ── Parallel path ──────────────────────────────────────────────
        tmp_dir = tempfile.mkdtemp(prefix=spec.tmp_dir_prefix)
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

        worker_log_name = f"{spec.model_id}.worker"
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _process_timeframe_worker,
                    stf, ti, len(timeframes),
                    sales_path, all_months_ser, model_params,
                    model_id, spec.dispatcher_key, worker_log_name,
                    item_path, cg_path, loc_path,
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
                    logger.exception(
                        "Worker failed for timeframe %s", stf["label"],
                    )

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

            predict_months = [
                m for m in all_months if predict_start <= m <= predict_end
            ]
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

            # Use per-timeframe hook if provided, else default path
            if spec.per_timeframe_hook:
                ctx = PerTimeframeContext(
                    tf=tf, ti=ti, n_total=len(timeframes),
                    train_sales=train_sales, predict_months=predict_months,
                    model_params=model_params, dispatcher_key=spec.dispatcher_key,
                    label=label,
                )
                preds = spec.per_timeframe_hook(ctx, hook_state)
            else:
                with profiled_section(f"{spec.profiler_prefix}_tf_{label}"):
                    preds = run_foundation_models(
                        train_sales[["sku_ck", "startdate", "qty"]],
                        predict_months,
                        {spec.dispatcher_key: model_params},
                    )

            if preds.empty:
                logger.warning("  Timeframe %s: no predictions produced", label)
                continue

            preds = _enrich_predictions(
                preds, item_id_map, cg_map, loc_map, model_id, tf,
            )

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

    # ── Post-process ───────────────────────────────────────────────────────
    post_step = "Step 4" if not spec.pre_timeframe_hook else "Step 5"
    logger.info("%s: Post-processing predictions...", post_step)
    with profiled_section("postprocess"):
        output_df, archive_df, _combined_raw = postprocess_predictions(
            all_predictions, sales_df, exec_lag_map, timeframes=timeframes,
        )

    # Ensure model_id is set consistently
    output_df["model_id"] = model_id
    archive_df["model_id"] = model_id

    logger.info(
        "Post-process complete: %s output rows, %s archive rows",
        f"{len(output_df):,}", f"{len(archive_df):,}",
    )

    # ── Save output ────────────────────────────────────────────────────────
    save_step = "Step 5" if not spec.pre_timeframe_hook else "Step 6"
    logger.info("%s: Saving backtest output...", save_step)
    with profiled_section("save_output"):
        _out_path, _arch_path, _meta_path, metadata = save_backtest_output(
            output_df=output_df,
            archive_df=archive_df,
            output_dir=output_dir,
            model_id=model_id,
            cluster_strategy="global",
            n_timeframes=n_timeframes,
            model_params=model_params,
            model_params_key=spec.model_params_key,
            timeframes=timeframes,
            earliest_month=earliest_month,
            latest_month=latest_month,
            extra_metadata=spec.extra_metadata,
            dfu_cohort_map=dfu_cohort_map,
        )

    ckpt.cleanup()

    total_time = time.time() - t_start
    accuracy = metadata.get("accuracy_overall")
    logger.info(
        "%s backtest complete: accuracy=%.2f%%, %s predictions, %.1f min total",
        spec.display_name,
        accuracy if accuracy is not None else 0.0,
        f"{len(output_df):,}",
        total_time / 60,
    )
