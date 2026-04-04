"""
Hierarchical Chronos Bolt backtest: customer-level bottom-up + top-down reconciliation.

1. Load true demand (demand_qty) from fact_customer_demand_monthly
2. Run Bolt on each customer × item × loc series (bottom-up)
3. Run Bolt on aggregated item × loc series (top-down)
4. Reconcile: α·BU + (1-α)·TD (weighted average, default α=0.6)
5. Map reconciled item×loc forecasts to DFU grain (item×customer_group×loc)
6. Output in standard backtest format → champion selection sees it as any other model

Produces:
  data/backtest/bolt_hierarchical/backtest_predictions.csv
  data/backtest/bolt_hierarchical/backtest_predictions_all_lags.csv
  data/backtest/bolt_hierarchical/backtest_metadata.json
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adv_algorithm_testing.foundation_models import run_foundation_models
from adv_algorithm_testing.reconciliation import reconcile_two_level
from common.core.db import get_db_params
from common.core.planning_date import get_planning_date
from common.ml.backtest_framework import (
    BacktestCheckpointer,
    generate_timeframes,
    postprocess_predictions,
    save_backtest_output,
)
from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)

MODEL_ID = "bolt_hierarchical"
DISPATCHER_KEY = "chronos_bolt"


# ---------------------------------------------------------------------------
# Data loading — customer-level demand (not constrained sales)
# ---------------------------------------------------------------------------

def _load_customer_demand(db: dict) -> pd.DataFrame:
    """Load customer demand from fact_customer_demand_monthly."""
    import psycopg

    sql = """
        SELECT f.item_id,
               f.customer_no,
               f.location_id AS loc,
               f.startdate,
               f.demand_qty AS qty
        FROM fact_customer_demand_monthly f
        WHERE f.demand_qty > 0
        ORDER BY f.item_id, f.location_id, f.customer_no, f.startdate
    """
    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=cols)
    df["startdate"] = pd.to_datetime(df["startdate"])
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype("float32")
    return df


def _load_dfu_mapping(db: dict) -> pd.DataFrame:
    """Load DFU grain mapping from dim_sku: item×loc → item×customer_group×loc."""
    import psycopg

    sql = """
        SELECT sku_ck, item_id, customer_group, loc, execution_lag
        FROM dim_sku
    """
    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=cols)
    df["execution_lag"] = pd.to_numeric(df["execution_lag"], errors="coerce").fillna(0).astype(int)
    return df


def _load_sales_actuals(db: dict) -> pd.DataFrame:
    """Load fact_sales_monthly actuals for fair comparison with other models."""
    import psycopg

    sql = """
        SELECT d.sku_ck, s.item_id, s.customer_group, s.loc, s.startdate, s.qty
        FROM fact_sales_monthly s
        JOIN dim_sku d ON d.item_id = s.item_id
            AND d.customer_group = s.customer_group
            AND d.loc = s.loc
        WHERE s.qty IS NOT NULL
        ORDER BY d.sku_ck, s.startdate
    """
    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=cols)
    df["startdate"] = pd.to_datetime(df["startdate"])
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
    return df


# ---------------------------------------------------------------------------
# Series preparation
# ---------------------------------------------------------------------------

def _build_customer_series(
    demand_df: pd.DataFrame,
    train_end: pd.Timestamp,
    min_nonzero_months: int = 3,
    max_customers_per_item_loc: int = 100,
) -> pd.DataFrame:
    """Build bottom-level customer series for Bolt inference.

    Returns DataFrame with [sku_ck, startdate, qty] where sku_ck encodes
    item×customer×loc at the customer grain.
    """
    train = demand_df[demand_df["startdate"] <= train_end].copy()
    if train.empty:
        return pd.DataFrame(columns=["sku_ck", "startdate", "qty"])

    # Build customer series key
    train["cust_key"] = train["item_id"] + "__" + train["customer_no"] + "__" + train["loc"]

    # Filter: keep series with >= min_nonzero_months non-zero months
    nonzero_counts = train[train["qty"] > 0].groupby("cust_key")["startdate"].nunique()
    valid_keys = nonzero_counts[nonzero_counts >= min_nonzero_months].index
    train = train[train["cust_key"].isin(valid_keys)]

    # Cap: keep top N customers per item×loc by total demand
    cust_totals = train.groupby(["item_id", "loc", "cust_key"])["qty"].sum().reset_index()
    cust_totals["rank"] = cust_totals.groupby(["item_id", "loc"])["qty"].rank(
        ascending=False, method="first"
    )
    keep_keys = set(cust_totals[cust_totals["rank"] <= max_customers_per_item_loc]["cust_key"])
    train = train[train["cust_key"].isin(keep_keys)]

    return train[["cust_key", "startdate", "qty"]].rename(columns={"cust_key": "sku_ck"})


def _build_agg_series(
    demand_df: pd.DataFrame,
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build top-level aggregated item×loc series for Bolt inference."""
    train = demand_df[demand_df["startdate"] <= train_end].copy()
    if train.empty:
        return pd.DataFrame(columns=["sku_ck", "startdate", "qty"])

    agg = train.groupby(["item_id", "loc", "startdate"])["qty"].sum().reset_index()
    agg["sku_ck"] = agg["item_id"] + "__AGG__" + agg["loc"]
    return agg[["sku_ck", "startdate", "qty"]]


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def _reconcile(
    bu_preds: pd.DataFrame,
    td_preds: pd.DataFrame,
    bu_weight: float = 0.6,
    method: str = "weighted_average",
    actuals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reconcile bottom-up and top-down forecasts at item×loc level.

    Delegates to reconciliation module which supports weighted_average and mint_shrink.
    """
    return reconcile_two_level(
        bu_item_loc=bu_preds,
        td_item_loc=td_preds,
        actuals_item_loc=actuals,
        method=method,
        bu_weight=bu_weight,
    )


# ---------------------------------------------------------------------------
# DFU grain mapping
# ---------------------------------------------------------------------------

def _map_to_dfu_grain(
    reconciled: pd.DataFrame,
    dfu_map: pd.DataFrame,
    sales_df: pd.DataFrame,
) -> pd.DataFrame:
    """Map item×loc reconciled forecasts to item×customer_group×loc DFU grain.

    Uses historical sales share to proportionally allocate when multiple
    customer_groups exist for the same (item, loc).
    """
    # Build shares: fraction of sales per customer_group within each (item, loc)
    if not sales_df.empty:
        cg_totals = sales_df.groupby(["item_id", "loc", "customer_group"])["qty"].sum().reset_index()
        il_totals = cg_totals.groupby(["item_id", "loc"])["qty"].sum().reset_index(name="il_total")
        cg_totals = cg_totals.merge(il_totals, on=["item_id", "loc"])
        cg_totals["share"] = cg_totals["qty"] / cg_totals["il_total"].clip(lower=1e-9)
    else:
        cg_totals = pd.DataFrame(columns=["item_id", "loc", "customer_group", "share"])

    # Get unique DFU mappings
    dfu_keys = dfu_map[["sku_ck", "item_id", "customer_group", "loc"]].drop_duplicates()

    # Merge reconciled with DFU keys
    expanded = reconciled.merge(
        dfu_keys[["item_id", "customer_group", "loc", "sku_ck"]],
        on=["item_id", "loc"],
        how="inner",
    )

    # Apply proportional split if multiple customer_groups per (item, loc)
    if not cg_totals.empty:
        expanded = expanded.merge(
            cg_totals[["item_id", "loc", "customer_group", "share"]],
            on=["item_id", "loc", "customer_group"],
            how="left",
        )
        expanded["share"] = expanded["share"].fillna(1.0)
    else:
        expanded["share"] = 1.0

    # Count customer_groups per (item, loc, startdate)
    n_cg = expanded.groupby(["item_id", "loc", "startdate"])["customer_group"].transform("nunique")
    expanded.loc[n_cg > 1, "basefcst_pref"] = (
        expanded.loc[n_cg > 1, "basefcst_pref"] * expanded.loc[n_cg > 1, "share"]
    )

    return expanded[["sku_ck", "item_id", "customer_group", "loc", "startdate", "basefcst_pref"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical Bolt: customer-level bottom-up + top-down reconciliation",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-timeframes", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    # ── Config ──────────────────────────────────────────────────────────────
    with profiled_section("load_config"):
        config_path = Path(args.config) if args.config else ROOT / "config" / "algorithm_config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        hier_cfg = cfg.get("algorithms", {}).get("bolt_hierarchical", {})
        if not hier_cfg.get("enabled", True):
            logger.info("bolt_hierarchical is disabled in config; exiting")
            return

        backtest_cfg = cfg.get("backtest", {})
        n_timeframes = args.n_timeframes or hier_cfg.get("n_timeframes", 5)
        embargo_months = backtest_cfg.get("embargo_months", 0)

        output_dir = (
            Path(args.output_dir) if args.output_dir
            else ROOT / backtest_cfg.get("output_dir", "data/backtest")
        )
        model_id = hier_cfg.get("model_id", MODEL_ID)

        bolt_params = {
            "model_size": hier_cfg.get("model_size", "base"),
            "device": hier_cfg.get("device", "auto"),
            "batch_size": hier_cfg.get("batch_size", 2048),
            "num_samples": hier_cfg.get("num_samples", 12),
            "prediction_length": hier_cfg.get("prediction_length", 6),
        }
        min_nonzero = hier_cfg.get("min_nonzero_months", 3)
        max_custs = hier_cfg.get("max_customers_per_item_loc", 100)
        bu_weight = hier_cfg.get("bu_weight", 0.6)
        reconciliation_method = hier_cfg.get("reconciliation_method", "weighted_average")

    logger.info(
        "Bolt Hierarchical config: model_id=%s, n_timeframes=%d, "
        "batch=%d, bu_weight=%.2f, max_custs=%d",
        model_id, n_timeframes, bolt_params["batch_size"], bu_weight, max_custs,
    )

    # ── Step 1: Load data ───────────────────────────────────────────────────
    logger.info("Step 1: Loading customer demand + DFU mapping + sales actuals...")
    t_start = time.time()
    db = get_db_params()

    with profiled_section("load_data"):
        demand_df = _load_customer_demand(db)
        dfu_map = _load_dfu_mapping(db)
        sales_df = _load_sales_actuals(db)

    logger.info(
        "Data loaded: %s demand rows (%s item×loc pairs), %s DFUs, %s sales rows (%.1fs)",
        f"{len(demand_df):,}",
        f"{demand_df.groupby(['item_id', 'loc']).ngroups:,}",
        f"{len(dfu_map):,}",
        f"{len(sales_df):,}",
        time.time() - t_start,
    )

    if demand_df.empty:
        logger.warning("No customer demand data; exiting")
        return

    # Build exec_lag and cohort maps from DFU mapping
    exec_lag_map = dfu_map.set_index("sku_ck")["execution_lag"].to_dict()

    # ── Step 2: Generate timeframes ─────────────────────────────────────────
    planning_dt = pd.Timestamp(get_planning_date())
    planning_cutoff = planning_dt.normalize().replace(day=1)
    latest_month = min(demand_df["startdate"].max(), planning_cutoff)
    earliest_month = demand_df["startdate"].min()
    demand_df = demand_df[demand_df["startdate"] <= latest_month].copy()

    logger.info("Date range: %s -> %s", earliest_month.date(), latest_month.date())

    with profiled_section("generate_timeframes"):
        timeframes = generate_timeframes(
            earliest_month, latest_month, n_timeframes,
            embargo_months=embargo_months,
        )

    logger.info("Step 2: Generated %d timeframes", len(timeframes))
    for tf in timeframes:
        logger.info(
            "  %s: train -> %s, predict [%s -> %s]",
            tf["label"], tf["train_end"].date(),
            tf["predict_start"].date(), tf["predict_end"].date(),
        )

    all_months = sorted(demand_df["startdate"].unique())

    # ── Checkpoint manager ──────────────────────────────────────────────────
    ckpt = BacktestCheckpointer(output_dir, model_id, resume=args.resume)
    all_predictions: list[pd.DataFrame] = []
    all_predictions.extend(ckpt.load_all_existing())

    # ── Step 3: Per-timeframe inference + reconciliation ────────────────────
    logger.info("Step 3: Running hierarchical Bolt inference...")

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

        predict_months = [m for m in all_months if predict_start <= m <= predict_end]
        if not predict_months:
            logger.info("  Timeframe %s: no predict months, skipping", label)
            continue

        logger.info("Timeframe %s (%d/%d)", label, ti + 1, len(timeframes))

        # --- Bottom-up: customer-level Bolt ---
        with profiled_section(f"bu_build_{label}"):
            cust_series = _build_customer_series(
                demand_df, train_end, min_nonzero, max_custs,
            )
        n_cust_series = cust_series["sku_ck"].nunique()
        logger.info("  Bottom-up: %s customer series", f"{n_cust_series:,}")

        with profiled_section(f"bu_bolt_{label}"):
            if not cust_series.empty:
                bu_raw = run_foundation_models(
                    cust_series[["sku_ck", "startdate", "qty"]],
                    predict_months,
                    {DISPATCHER_KEY: bolt_params},
                )
            else:
                bu_raw = pd.DataFrame()

        # Parse customer key back to item×loc
        if not bu_raw.empty and "algorithm_id" in bu_raw.columns:
            bu_raw = bu_raw.drop(columns=["algorithm_id"])
        if not bu_raw.empty:
            parts = bu_raw["sku_ck"].str.split("__", expand=True)
            bu_raw["item_id"] = parts[0]
            bu_raw["loc"] = parts[2]

        # --- Top-down: aggregated item×loc Bolt ---
        with profiled_section(f"td_build_{label}"):
            agg_series = _build_agg_series(demand_df, train_end)
        n_agg_series = agg_series["sku_ck"].nunique()
        logger.info("  Top-down: %s aggregated series", f"{n_agg_series:,}")

        with profiled_section(f"td_bolt_{label}"):
            if not agg_series.empty:
                td_raw = run_foundation_models(
                    agg_series[["sku_ck", "startdate", "qty"]],
                    predict_months,
                    {DISPATCHER_KEY: bolt_params},
                )
            else:
                td_raw = pd.DataFrame()

        if not td_raw.empty and "algorithm_id" in td_raw.columns:
            td_raw = td_raw.drop(columns=["algorithm_id"])
        if not td_raw.empty:
            parts = td_raw["sku_ck"].str.split("__AGG__", expand=True)
            td_raw["item_id"] = parts[0]
            td_raw["loc"] = parts[1]

        # --- Reconcile ---
        if bu_raw.empty and td_raw.empty:
            logger.warning("  Timeframe %s: no predictions from either level", label)
            continue

        # Build actuals for MinTrace reconciliation (aggregated demand at item×loc)
        agg_actuals = demand_df[demand_df["startdate"] <= train_end].groupby(
            ["item_id", "loc", "startdate"]
        )["qty"].sum().reset_index()

        with profiled_section(f"reconcile_{label}"):
            reconciled = _reconcile(
                bu_raw, td_raw,
                bu_weight=bu_weight,
                method=reconciliation_method,
                actuals=agg_actuals,
            )

        logger.info(
            "  Reconciled: %s item×loc×month predictions (BU=%.0f%%, TD=%.0f%%)",
            f"{len(reconciled):,}", bu_weight * 100, (1 - bu_weight) * 100,
        )

        # --- Map to DFU grain ---
        with profiled_section(f"dfu_map_{label}"):
            preds = _map_to_dfu_grain(reconciled, dfu_map, sales_df)

        if preds.empty:
            logger.warning("  Timeframe %s: no predictions after DFU mapping", label)
            continue

        preds["model_id"] = model_id
        preds["timeframe_idx"] = tf["index"]
        preds["timeframe_label"] = label

        ckpt.save(preds, tf["index"])
        all_predictions.append(preds)

        logger.info(
            "  Timeframe %s: %s DFU predictions (%.1fs) [checkpointed]",
            label, f"{len(preds):,}", time.time() - tf_start,
        )

    if not all_predictions:
        logger.warning("No predictions produced; exiting")
        return

    # ── Step 4: Post-process ────────────────────────────────────────────────
    logger.info("Step 4: Post-processing predictions...")
    with profiled_section("postprocess"):
        output_df, archive_df, _combined = postprocess_predictions(
            all_predictions, sales_df, exec_lag_map, timeframes=timeframes,
        )

    output_df["model_id"] = model_id
    archive_df["model_id"] = model_id

    logger.info(
        "Post-process: %s output rows, %s archive rows",
        f"{len(output_df):,}", f"{len(archive_df):,}",
    )

    # ── Step 5: Save ────────────────────────────────────────────────────────
    logger.info("Step 5: Saving backtest output...")
    with profiled_section("save_output"):
        _out, _arch, _meta, metadata = save_backtest_output(
            output_df=output_df,
            archive_df=archive_df,
            output_dir=output_dir,
            model_id=model_id,
            cluster_strategy="global",
            n_timeframes=n_timeframes,
            model_params={**bolt_params, "bu_weight": bu_weight},
            model_params_key="bolt_hierarchical_params",
            timeframes=timeframes,
            earliest_month=earliest_month,
            latest_month=latest_month,
            extra_metadata={
                "params_source": "algorithm_config",
                "model_type": "foundation_model",
                "architecture": "bolt_hierarchical",
                "reconciliation": "weighted_average",
                "bu_weight": bu_weight,
                "max_customers_per_item_loc": max_custs,
                "data_source": "fact_customer_demand_monthly",
                "demand_column": "demand_qty",
            },
        )

    ckpt.cleanup()
    total_time = time.time() - t_start
    accuracy = metadata.get("accuracy_overall")
    logger.info(
        "Bolt Hierarchical complete: accuracy=%.2f%%, %s predictions, %.1f min",
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
