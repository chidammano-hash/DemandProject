"""
Hierarchical Chronos Bolt backtest: customer-level bottom-up + location-level top-down.

Architecture:
  BU: Bolt on every customer×item×loc series from fact_customer_demand_monthly
  TD: Bolt on aggregated item×loc series (demand summed across customers)
  Reconcile: Dispatch via config — weighted_average (Phase 1) or mint_shrink (Phase 2)
  Map: Reconciled item×loc forecasts → DFU grain via demand-based share

Memory strategy (designed for 128 GB system):
  - Bolt model loaded FIRST while memory is clean (~1.5 GB with bfloat16)
  - Customer demand loaded via server-side cursor in 500K-row chunks
  - String columns stored as categorical dtype (10x memory reduction)
  - Demand pre-aggregated to item×loc once; TD uses filtered view per timeframe
  - BU inference chunked at 200K series; each chunk aggregated immediately
  - Predictions written to disk per-timeframe via checkpointer; NOT accumulated
  - Peak memory: ~12-15 GB (model + demand_df + one active chunk)

Produces:
  data/backtest/bolt_hierarchical/backtest_predictions.csv
  data/backtest/bolt_hierarchical/backtest_predictions_all_lags.csv
  data/backtest/bolt_hierarchical/backtest_metadata.json
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ml.expert_panel.foundation_models import (
    _clear_pipeline_cache,
    _get_chronos_pipeline,
    _resolve_device,
    run_foundation_models,
)
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

MODEL_ID = "bolt_hierarchical"
DISPATCHER_KEY = "chronos_bolt"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_customer_demand(db: dict) -> pd.DataFrame:
    """Load customer demand from fact_customer_demand_monthly.

    Uses server-side cursor to avoid materializing full result set.
    Loads ALL rows (including zero demand) so series can be re-indexed to a
    regular monthly grid — Bolt needs consecutive months, not sparse non-zero.
    String columns stored as categorical dtype to reduce memory ~10x.
    """
    import psycopg

    sql = """
        SELECT f.item_id,
               f.customer_no,
               f.location_id AS loc,
               f.startdate,
               GREATEST(f.demand_qty, 0) AS qty
        FROM fact_customer_demand_monthly f
        ORDER BY f.item_id, f.location_id, f.customer_no, f.startdate
    """
    chunks: list[pd.DataFrame] = []
    with psycopg.connect(**db) as conn:
        with conn.cursor(name="cust_demand_cursor") as cur:
            cur.itersize = 500_000
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            while True:
                batch = cur.fetchmany(500_000)
                if not batch:
                    break
                chunks.append(pd.DataFrame(batch, columns=cols))

    if not chunks:
        return pd.DataFrame(columns=["item_id", "customer_no", "loc", "startdate", "qty", "cust_key"])

    df = pd.concat(chunks, ignore_index=True)
    del chunks

    df["startdate"] = pd.to_datetime(df["startdate"])
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype("float32")
    # Categorical string columns — ~10x memory reduction vs object dtype
    for col in ("item_id", "customer_no", "loc"):
        df[col] = df[col].astype("category")
    # Pre-build customer key as categorical
    df["cust_key"] = (
        df["item_id"].astype(str) + "__"
        + df["customer_no"].astype(str) + "__"
        + df["loc"].astype(str)
    ).astype("category")
    return df


def _load_dfu_mapping(db: dict) -> pd.DataFrame:
    """Load DFU grain mapping from dim_sku."""
    import psycopg

    sql = "SELECT sku_ck, item_id, customer_group, loc, execution_lag FROM dim_sku"
    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=cols)
    df["execution_lag"] = pd.to_numeric(df["execution_lag"], errors="coerce").fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Series preparation
# ---------------------------------------------------------------------------

def _build_customer_series(
    demand_df: pd.DataFrame,
    train_end: pd.Timestamp,
    min_nonzero_months: int = 6,
    max_customers_per_item_loc: int | None = None,
) -> pd.DataFrame:
    """Build customer-level series for Bolt BU inference.

    Expects demand_df to have a pre-built 'cust_key' column.
    Returns [sku_ck, startdate, qty] where sku_ck = item__customer__loc.

    Series are returned on a regular monthly grid (zero-filled gaps) so Bolt
    sees correct temporal structure. Sparse-only rows from the source already
    include zeros because _load_customer_demand loads all rows.

    Caps customers per item×loc at max_customers_per_item_loc; excess demand
    is aggregated into an '__OTHER__' bucket series per item×loc.
    """
    mask = demand_df["startdate"] <= train_end
    train = demand_df.loc[mask, ["cust_key", "item_id", "customer_no", "loc", "startdate", "qty"]].copy()
    if train.empty:
        return pd.DataFrame(columns=["sku_ck", "startdate", "qty"])

    # Filter: keep series with >= min_nonzero_months non-zero months
    nz = train.loc[train["qty"] > 0].groupby("cust_key", observed=True)["startdate"].nunique()
    valid_keys = nz.index[nz >= min_nonzero_months]
    train = train.loc[train["cust_key"].isin(valid_keys)]

    if train.empty:
        return pd.DataFrame(columns=["sku_ck", "startdate", "qty"])

    # Cap customers per item×loc — keep top-N by demand, aggregate rest to __OTHER__
    if max_customers_per_item_loc and max_customers_per_item_loc > 0:
        cust_totals = train.groupby(["item_id", "loc", "customer_no"], observed=True)["qty"].sum().reset_index()
        cust_totals["rank"] = cust_totals.groupby(["item_id", "loc"], observed=True)["qty"].rank(
            method="first", ascending=False,
        )
        # Customers within cap
        keep = cust_totals.loc[cust_totals["rank"] <= max_customers_per_item_loc]
        keep_set = set(zip(keep["item_id"], keep["loc"], keep["customer_no"]))

        is_kept = train.apply(
            lambda r: (r["item_id"], r["loc"], r["customer_no"]) in keep_set, axis=1,
        )
        kept_df = train[is_kept]
        overflow_df = train[~is_kept]

        if not overflow_df.empty:
            # Aggregate overflow into __OTHER__ bucket per item×loc×month
            other_agg = overflow_df.groupby(
                ["item_id", "loc", "startdate"], observed=True, sort=False,
            )["qty"].sum().reset_index()
            other_agg["cust_key"] = (
                other_agg["item_id"].astype(str) + "____OTHER____"
                + other_agg["loc"].astype(str)
            )
            other_out = other_agg[["cust_key", "startdate", "qty"]]
            kept_out = kept_df[["cust_key", "startdate", "qty"]]
            train = pd.concat([kept_out, other_out], ignore_index=True)
        else:
            train = kept_df[["cust_key", "startdate", "qty"]]
    else:
        train = train[["cust_key", "startdate", "qty"]]

    return train.rename(columns={"cust_key": "sku_ck"})


def _pre_aggregate_demand(demand_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-aggregate demand to item×loc×month once before the timeframe loop.

    Returns [sku_ck, item_id, loc, startdate, qty] aggregated across customers.
    This avoids re-scanning all demand rows every timeframe.
    """
    if demand_df.empty:
        return pd.DataFrame(columns=["sku_ck", "item_id", "loc", "startdate", "qty"])

    agg = demand_df.groupby(
        ["item_id", "loc", "startdate"], observed=True, sort=False,
    )["qty"].sum().reset_index()
    agg["sku_ck"] = agg["item_id"].astype(str) + "__AGG__" + agg["loc"].astype(str)
    return agg[["sku_ck", "item_id", "loc", "startdate", "qty"]]


def _build_agg_series(
    agg_demand: pd.DataFrame,
    train_end: pd.Timestamp,
    min_history_months: int = 6,
) -> pd.DataFrame:
    """Filter pre-aggregated item×loc series for a timeframe.

    Applies minimum history filter to exclude item-locs with too few months.
    """
    train = agg_demand.loc[agg_demand["startdate"] <= train_end]
    if train.empty:
        return pd.DataFrame(columns=["sku_ck", "startdate", "qty"])

    # Filter: keep item-locs with >= min_history_months of non-zero data
    nz = train.loc[train["qty"] > 0].groupby("sku_ck", observed=True)["startdate"].nunique()
    valid_keys = nz.index[nz >= min_history_months]
    train = train.loc[train["sku_ck"].isin(valid_keys)]

    return train[["sku_ck", "startdate", "qty"]]


def _aggregate_bu_to_item_loc(bu_preds: pd.DataFrame) -> pd.DataFrame:
    """Aggregate customer-level BU predictions to item×loc.

    Parses item_id and loc from sku_ck (item__customer__loc), sums across customers.
    """
    if bu_preds.empty:
        return pd.DataFrame(columns=["item_id", "loc", "startdate", "basefcst_pref"])

    # Right-split once to isolate loc (last segment), then left-split for item_id
    right = bu_preds["sku_ck"].str.rsplit("__", n=1, expand=True)
    left = right.iloc[:, 0].str.split("__", n=1, expand=True)
    bu_preds = bu_preds.assign(item_id=left.iloc[:, 0], loc=right.iloc[:, 1])

    return bu_preds.groupby(
        ["item_id", "loc", "startdate"], observed=True, sort=False,
    )["basefcst_pref"].sum().reset_index()


# ---------------------------------------------------------------------------
# Chunked BU inference
# ---------------------------------------------------------------------------

def _run_bu_chunked(
    cust_series: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    bolt_params: dict,
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    """Run Bolt on customer series in memory-safe chunks.

    Each chunk is inferred, aggregated to item×loc immediately, then freed.
    Returns aggregated BU predictions at item×loc grain.
    """
    if cust_series.empty:
        return pd.DataFrame(columns=["item_id", "loc", "startdate", "basefcst_pref"])

    # Sort by sku_ck so we can use positional slicing (O(1) per chunk)
    cust_series = cust_series.sort_values("sku_ck").reset_index(drop=True)
    sku_vals = cust_series["sku_ck"].values
    unique_skus, first_idx = np.unique(sku_vals, return_index=True)
    # End boundaries: next key's start, or end of array
    end_idx = np.append(first_idx[1:], len(sku_vals))

    n_total = len(unique_skus)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    agg_parts: list[pd.DataFrame] = []

    logger.info("  BU: %d series in %d chunks of %d", n_total, n_chunks, chunk_size)

    for ci in range(n_chunks):
        cs = ci * chunk_size
        ce = min(cs + chunk_size, n_total)

        # Positional slice — O(1), no boolean scan over 8M rows
        row_start = first_idx[cs]
        row_end = end_idx[ce - 1]
        chunk_df = cust_series.iloc[row_start:row_end]

        chunk_preds = run_foundation_models(
            chunk_df, predict_months,
            {DISPATCHER_KEY: bolt_params},
            keep_model_loaded=True,
        )

        if not chunk_preds.empty:
            if "algorithm_id" in chunk_preds.columns:
                chunk_preds = chunk_preds.drop(columns=["algorithm_id"])
            # Aggregate to item×loc immediately — frees customer-level predictions
            agg_parts.append(_aggregate_bu_to_item_loc(chunk_preds))

        del chunk_df, chunk_preds
        gc.collect()

        logger.info("  BU chunk %d/%d done (%d-%d)", ci + 1, n_chunks, cs, ce)

    if not agg_parts:
        return pd.DataFrame(columns=["item_id", "loc", "startdate", "basefcst_pref"])

    # Merge aggregated chunks — same item×loc can span multiple chunks
    bu_agg = pd.concat(agg_parts, ignore_index=True)
    del agg_parts
    bu_agg = bu_agg.groupby(
        ["item_id", "loc", "startdate"], observed=True, sort=False,
    )["basefcst_pref"].sum().reset_index()

    return bu_agg


# ---------------------------------------------------------------------------
# Reconciliation + DFU mapping
# ---------------------------------------------------------------------------

def _compute_adaptive_bu_weight(
    bu_agg: pd.DataFrame,
    demand_df: pd.DataFrame,
    train_end: pd.Timestamp,
    default_weight: float = 0.6,
) -> pd.DataFrame:
    """Compute per-item×loc BU weight based on active customer count.

    More customers → higher BU weight (diverse signal).
    Fewer customers → lower BU weight (noisy, trust TD more).

    Returns [item_id, loc, bu_weight].
    """
    train = demand_df.loc[
        (demand_df["startdate"] <= train_end) & (demand_df["qty"] > 0)
    ]
    if train.empty:
        return pd.DataFrame(columns=["item_id", "loc", "bu_weight"])

    n_cust = train.groupby(["item_id", "loc"], observed=True)["customer_no"].nunique().reset_index(
        name="n_customers",
    )
    # Heuristic: >20 customers → 0.7, 5-20 → 0.5, <5 → 0.3
    conditions = [
        n_cust["n_customers"] > 20,
        n_cust["n_customers"] >= 5,
    ]
    choices = [0.7, 0.5]
    n_cust["bu_weight"] = np.select(conditions, choices, default=0.3)
    return n_cust[["item_id", "loc", "bu_weight"]]


def _reconcile_and_map(
    bu_agg: pd.DataFrame,
    td_preds: pd.DataFrame,
    bu_weight: float,
    dfu_map: pd.DataFrame,
    demand_share: pd.DataFrame,
    reconciliation_method: str = "weighted_average",
    actuals_item_loc: pd.DataFrame | None = None,
    adaptive_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reconcile BU+TD at item×loc, map to DFU grain via demand share.

    Args:
        bu_agg: BU at item×loc [item_id, loc, startdate, basefcst_pref]
        td_preds: TD predictions [sku_ck (AGG key), startdate, basefcst_pref]
        bu_weight: Default BU weight (overridden by adaptive_weights if provided)
        dfu_map: [sku_ck, item_id, customer_group, loc]
        demand_share: Per-timeframe [item_id, loc, customer_group, share]
        reconciliation_method: "weighted_average" or "mint_shrink"
        actuals_item_loc: Actuals for mint_shrink [item_id, loc, startdate, qty]
        adaptive_weights: Per-item×loc [item_id, loc, bu_weight]

    Returns:
        DFU-grain DataFrame [sku_ck, item_id, customer_group, loc, startdate, basefcst_pref]
    """
    from common.ml.expert_panel.reconciliation import reconcile_two_level

    if bu_agg.empty:
        return pd.DataFrame(
            columns=["sku_ck", "item_id", "customer_group", "loc", "startdate", "basefcst_pref"],
        )

    # Parse TD keys to item×loc
    if not td_preds.empty:
        td = td_preds.copy()
        parts = td["sku_ck"].str.split("__AGG__", expand=True)
        td["item_id"] = parts[0]
        td["loc"] = parts[1]
        td = td[["item_id", "loc", "startdate", "basefcst_pref"]]
    else:
        td = pd.DataFrame(columns=["item_id", "loc", "startdate", "basefcst_pref"])

    # Dispatch to reconciliation module
    if reconciliation_method == "mint_shrink" and actuals_item_loc is not None:
        reconciled = reconcile_two_level(
            bu_item_loc=bu_agg,
            td_item_loc=td,
            actuals_item_loc=actuals_item_loc,
            method="mint_shrink",
            bu_weight=bu_weight,
        )
    elif adaptive_weights is not None and not adaptive_weights.empty:
        # Adaptive weighted average: per-item×loc BU weight
        merged = bu_agg.rename(columns={"basefcst_pref": "bu_fcst"})
        if not td.empty:
            td_r = td.rename(columns={"basefcst_pref": "td_fcst"})
            merged = merged.merge(td_r, on=["item_id", "loc", "startdate"], how="left")
            merged["td_fcst"] = merged["td_fcst"].fillna(0.0)
        else:
            merged["td_fcst"] = 0.0

        merged = merged.merge(adaptive_weights, on=["item_id", "loc"], how="left")
        merged["bu_weight"] = merged["bu_weight"].fillna(bu_weight)

        has_td = merged["td_fcst"] > 0
        merged["basefcst_pref"] = np.where(
            has_td,
            merged["bu_weight"] * merged["bu_fcst"] + (1 - merged["bu_weight"]) * merged["td_fcst"],
            merged["bu_fcst"],
        )
        reconciled = merged
    else:
        # Simple weighted average via reconciliation module
        reconciled = reconcile_two_level(
            bu_item_loc=bu_agg,
            td_item_loc=td,
            method="weighted_average",
            bu_weight=bu_weight,
        )

    reconciled["basefcst_pref"] = np.maximum(reconciled["basefcst_pref"], 0.0)

    # Expand to DFU grain
    dfu_keys = dfu_map[["sku_ck", "item_id", "customer_group", "loc"]].drop_duplicates()
    expanded = reconciled[["item_id", "loc", "startdate", "basefcst_pref"]].merge(
        dfu_keys, on=["item_id", "loc"], how="inner",
    )

    # Apply proportional split for multiple DFUs per item×loc
    n_cg = expanded.groupby(
        ["item_id", "loc", "startdate"], observed=True,
    )["customer_group"].transform("nunique")

    if not demand_share.empty:
        expanded = expanded.merge(
            demand_share, on=["item_id", "loc", "customer_group"], how="left",
        )
        expanded["share"] = expanded["share"].fillna(1.0 / n_cg)
    else:
        expanded["share"] = 1.0 / n_cg

    expanded["basefcst_pref"] = expanded["basefcst_pref"] * expanded["share"]

    return expanded[["sku_ck", "item_id", "customer_group", "loc", "startdate", "basefcst_pref"]]


def _compute_demand_share(
    sales_df: pd.DataFrame,
    train_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute customer_group share within each item×loc from sales data.

    Uses data up to train_end to avoid data leakage across backtest timeframes.

    Returns [item_id, loc, customer_group, share].
    """
    if sales_df.empty:
        return pd.DataFrame(columns=["item_id", "loc", "customer_group", "share"])

    df = sales_df
    if train_end is not None:
        df = df[df["startdate"] <= train_end]
        if df.empty:
            return pd.DataFrame(columns=["item_id", "loc", "customer_group", "share"])

    cg = df.groupby(["item_id", "loc", "customer_group"], observed=True)["qty"].sum().reset_index()
    il = cg.groupby(["item_id", "loc"], observed=True)["qty"].sum().reset_index(name="il_total")
    cg = cg.merge(il, on=["item_id", "loc"])
    cg["share"] = cg["qty"] / cg["il_total"].clip(lower=1e-9)
    return cg[["item_id", "loc", "customer_group", "share"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical Bolt: customer BU + location TD with adaptive reconciliation",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-timeframes", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--loc", type=str, default=None,
                        help="Filter to a single location (e.g. 1401-BULK)")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    # ── Config ──────────────────────────────────────────────────────────────
    with profiled_section("load_config"):
        if args.config:
            config_path = Path(args.config)
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        else:
            from common.core.utils import load_forecast_pipeline_config
            cfg = load_forecast_pipeline_config()

        algo_entry = cfg.get("algorithms", {}).get("bolt_hierarchical", {})
        # Support pipeline config format (params sub-dict) or flat legacy format
        hier_cfg = algo_entry.get("params", algo_entry)
        if not algo_entry.get("enabled", True):
            logger.info("bolt_hierarchical is disabled in config; exiting")
            return

        backtest_cfg = cfg.get("backtest", {})
        n_timeframes = args.n_timeframes or hier_cfg.get("n_timeframes", 5)
        embargo_months = backtest_cfg.get("embargo_months", 0)

        output_dir = (
            Path(args.output_dir) if args.output_dir
            else ROOT / backtest_cfg.get("output_dir", "data/backtest")
        )
        model_id = algo_entry.get("model_id", hier_cfg.get("model_id", MODEL_ID))

        bolt_params = {
            "model_size": hier_cfg.get("model_size", "base"),
            "device": hier_cfg.get("device", "auto"),
            "batch_size": hier_cfg.get("batch_size", 2048),
            "prediction_length": hier_cfg.get("prediction_length", 6),
        }
        bu_weight = hier_cfg.get("bu_weight", 0.6)
        min_nonzero = hier_cfg.get("min_nonzero_months", 6)
        min_history = hier_cfg.get("min_history_months", 6)
        max_cust = hier_cfg.get("max_customers_per_item_loc", None)
        bu_chunk_size = hier_cfg.get("bu_chunk_size", 200_000)
        reconciliation_method = hier_cfg.get("reconciliation_method", "weighted_average")

    logger.info(
        "Config: model=%s, timeframes=%d, batch=%d, bu_weight=%.2f, "
        "chunk=%d, reconciliation=%s, max_cust=%s",
        model_id, n_timeframes, bolt_params["batch_size"], bu_weight,
        bu_chunk_size, reconciliation_method, max_cust,
    )

    # ── Pre-load Bolt model while memory is clean ───────────────────────────
    bolt_device = _resolve_device(bolt_params.get("device", "auto"))
    bolt_model_name = f"amazon/chronos-bolt-{bolt_params.get('model_size', 'base')}"
    logger.info("Pre-loading Bolt model (%s on %s)...", bolt_model_name, bolt_device)
    _get_chronos_pipeline(bolt_model_name, bolt_device)

    # ── Step 1: Load data ───────────────────────────────────────────────────
    logger.info("Step 1: Loading data...")
    t_start = time.time()
    db = get_db_params()

    with profiled_section("load_data"):
        demand_df = _load_customer_demand(db)
        sales_df, _dfu_attrs, _item_attrs = load_backtest_data(db)
        dfu_map = _load_dfu_mapping(db)

    # Optional location filter
    if args.loc:
        demand_df = demand_df[demand_df["loc"] == args.loc]
        sales_df = sales_df[sales_df["loc"] == args.loc]
        dfu_map = dfu_map[dfu_map["loc"] == args.loc]
        logger.info(
            "Filtered to loc=%s: %s demand, %s sales, %s DFUs",
            args.loc, f"{len(demand_df):,}",
            f"{len(sales_df):,}", f"{len(dfu_map):,}",
        )

    n_cust_keys = demand_df["cust_key"].nunique()
    logger.info(
        "Data: %s demand rows (%s customer series), %s sales rows, %s DFUs (%.1fs)",
        f"{len(demand_df):,}", f"{n_cust_keys:,}",
        f"{len(sales_df):,}", f"{len(dfu_map):,}",
        time.time() - t_start,
    )

    if demand_df.empty:
        logger.warning("No customer demand data; exiting")
        return

    exec_lag_map = dfu_map.set_index("sku_ck")["execution_lag"].to_dict()
    dfu_item_loc = dfu_map[["sku_ck", "item_id", "customer_group", "loc"]].drop_duplicates()

    # ── Pre-aggregate demand for TD (once, reused across timeframes) ────────
    with profiled_section("pre_aggregate_demand"):
        agg_demand = _pre_aggregate_demand(demand_df)
    logger.info("Pre-aggregated: %s item×loc×month rows", f"{len(agg_demand):,}")

    # ── Step 2: Generate timeframes ─────────────────────────────────────────
    planning_dt = pd.Timestamp(get_planning_date())
    planning_cutoff = planning_dt.normalize().replace(day=1)
    latest_month = min(demand_df["startdate"].max(), planning_cutoff)
    earliest_month = demand_df["startdate"].min()
    demand_df = demand_df[demand_df["startdate"] <= latest_month]
    agg_demand = agg_demand[agg_demand["startdate"] <= latest_month]

    logger.info("Date range: %s -> %s", earliest_month.date(), latest_month.date())

    with profiled_section("generate_timeframes"):
        timeframes = generate_timeframes(
            earliest_month, latest_month, n_timeframes,
            embargo_months=embargo_months,
        )

    logger.info("Step 2: %d timeframes", len(timeframes))
    for tf in timeframes:
        logger.info(
            "  %s: train -> %s, predict [%s -> %s]",
            tf["label"], tf["train_end"].date(),
            tf["predict_start"].date(), tf["predict_end"].date(),
        )

    all_months = sorted(demand_df["startdate"].unique())

    # ── Checkpoint manager ──────────────────────────────────────────────────
    ckpt = BacktestCheckpointer(output_dir, model_id, resume=args.resume)

    # ── Step 3: Per-timeframe inference + reconciliation ────────────────────
    logger.info(
        "Step 3: Running hierarchical Bolt (reconciliation=%s)...",
        reconciliation_method,
    )

    for ti, tf in enumerate(timeframes):
        if ckpt.exists(tf["index"]):
            logger.info(
                "Timeframe %s (%d/%d) — checkpointed, skipping",
                tf["label"], ti + 1, len(timeframes),
            )
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

        # --- Per-timeframe demand share (avoids data leakage) ---
        with profiled_section(f"demand_share_{label}"):
            demand_share = _compute_demand_share(sales_df, train_end=train_end)

        # --- Adaptive BU weight per item×loc ---
        with profiled_section(f"adaptive_weight_{label}"):
            adaptive_weights = _compute_adaptive_bu_weight(
                None, demand_df, train_end, default_weight=bu_weight,
            )

        # --- BU: customer-level Bolt (chunked) ---
        with profiled_section(f"bu_build_{label}"):
            cust_series = _build_customer_series(
                demand_df, train_end, min_nonzero, max_cust,
            )
        n_cust = cust_series["sku_ck"].nunique()
        logger.info("  BU: %s customer series", f"{n_cust:,}")

        with profiled_section(f"bu_bolt_{label}"):
            bu_agg = _run_bu_chunked(
                cust_series, predict_months, bolt_params, bu_chunk_size,
            )
        del cust_series
        gc.collect()

        logger.info("  BU aggregated: %s item×loc×month rows", f"{len(bu_agg):,}")

        # --- TD: aggregated item×loc Bolt (from pre-aggregated demand) ---
        with profiled_section(f"td_build_{label}"):
            agg_series = _build_agg_series(agg_demand, train_end, min_history)
        n_agg = agg_series["sku_ck"].nunique()
        logger.info("  TD: %s item×loc series", f"{n_agg:,}")

        with profiled_section(f"td_bolt_{label}"):
            td_raw = run_foundation_models(
                agg_series[["sku_ck", "startdate", "qty"]],
                predict_months,
                {DISPATCHER_KEY: bolt_params},
                keep_model_loaded=True,
            )
        if not td_raw.empty and "algorithm_id" in td_raw.columns:
            td_raw = td_raw.drop(columns=["algorithm_id"])

        # --- Reconcile + map to DFU grain ---
        if bu_agg.empty and td_raw.empty:
            logger.warning("  Timeframe %s: no predictions", label)
            continue

        # Prepare actuals for mint_shrink reconciliation
        actuals_il = None
        if reconciliation_method == "mint_shrink":
            actuals_il = agg_demand.loc[
                agg_demand["startdate"] <= train_end,
                ["item_id", "loc", "startdate", "qty"],
            ]

        with profiled_section(f"reconcile_{label}"):
            preds = _reconcile_and_map(
                bu_agg, td_raw, bu_weight, dfu_map, demand_share,
                reconciliation_method=reconciliation_method,
                actuals_item_loc=actuals_il,
                adaptive_weights=adaptive_weights,
            )
        del bu_agg, td_raw
        gc.collect()

        if preds.empty:
            logger.warning("  Timeframe %s: empty after reconciliation", label)
            continue

        preds["model_id"] = model_id
        preds["timeframe_idx"] = tf["index"]
        preds["timeframe_label"] = label

        # Write to disk immediately — don't accumulate in memory
        ckpt.save(preds, tf["index"])

        logger.info(
            "  Timeframe %s: %s predictions (%.1fs) [checkpointed]",
            label, f"{len(preds):,}", time.time() - tf_start,
        )
        del preds

    # Free model after all timeframes
    _clear_pipeline_cache()

    # ── Step 4: Post-process (read from disk checkpoints) ──────────────────
    logger.info("Step 4: Post-processing from checkpoints...")
    all_predictions: list[pd.DataFrame] = []
    for ckpt_df in ckpt.load_all_existing():
        if "item_id" not in ckpt_df.columns:
            ckpt_df = ckpt_df.merge(
                dfu_item_loc.set_index("sku_ck"), on="sku_ck", how="left",
            )
        all_predictions.append(ckpt_df)

    if not all_predictions:
        logger.warning("No predictions produced; exiting")
        return

    with profiled_section("postprocess"):
        output_df, archive_df, _combined = postprocess_predictions(
            all_predictions, sales_df, exec_lag_map, timeframes=timeframes,
        )
    del all_predictions

    output_df["model_id"] = model_id
    archive_df["model_id"] = model_id

    logger.info(
        "Post-process: %s output, %s archive",
        f"{len(output_df):,}", f"{len(archive_df):,}",
    )

    # ── Step 5: Save ────────────────────────────────────────────────────────
    logger.info("Step 5: Saving...")
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
                "params_source": "forecast_pipeline_config",
                "model_type": "foundation_model",
                "architecture": "bolt_hierarchical_adaptive",
                "reconciliation": reconciliation_method,
                "bu_weight_default": bu_weight,
                "bu_level": "customer",
                "td_level": "item_loc",
                "data_source": "fact_customer_demand_monthly",
                "max_customers_per_item_loc": max_cust,
                "min_nonzero_months": min_nonzero,
                "min_history_months": min_history,
            },
        )

    ckpt.cleanup()
    total_time = time.time() - t_start
    accuracy = metadata.get("accuracy_overall")
    logger.info(
        "Done: accuracy=%.2f%%, %s predictions, %.1f min",
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
