"""Generate customer-derived features from fact_customer_demand_monthly.

Aggregates individual-customer demand data to item x location x month grain
and computes 25 features: concentration, dynamics, true demand, channel mix,
and cross-customer signals.

Output: Upserts into customer_features_monthly table.

Usage:
    python -m scripts.ml.generate_customer_features [--months 36] [--workers 4]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

# Suppress noisy pandas FutureWarnings from .clip() and .fillna() downcasting
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params
from common.services.perf_profiler import profiled_section

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_customer_demand(db: dict, lookback_months: int = 36) -> pd.DataFrame:
    """Load raw customer demand data from Postgres.

    Uses two fast queries instead of one big JOIN:
      1. Fact table only (partition-pruned, no join overhead)
      2. Channel lookup from dim_customer (small, one-shot)
    Then merges in Python which is much faster than a DB hash-join on millions of rows.
    """
    import psycopg

    # Query 1: fact table only — partition-pruned, no join
    sql_fact = """
        SELECT item_id, location_id AS loc, startdate,
               customer_no, site, demand_qty, sales_qty, oos_qty
        FROM fact_customer_demand_monthly
        WHERE startdate >= (CURRENT_DATE - INTERVAL '%s months')::date
    """ % lookback_months

    # Query 2: customer attributes lookup (small table, fast)
    sql_cust = """
        SELECT DISTINCT customer_no, site,
               rpt_channel_desc AS channel,
               store_type_desc AS store_type,
               chain_type_desc AS chain_type,
               rpt_sub_channel_desc AS sub_channel,
               status AS cust_status,
               delivery_freq_code AS delivery_freq,
               premise_code
        FROM dim_customer
    """

    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        logger.info("  Query 1/2: Loading fact_customer_demand_monthly (partition-pruned)...")
        t0 = time.time()
        cur.execute(sql_fact)
        cols_fact = [d[0] for d in cur.description]
        rows_fact = cur.fetchall()
        logger.info("  Query 1 done: %s rows in %.1fs", f"{len(rows_fact):,}", time.time() - t0)

        logger.info("  Query 2/2: Loading dim_customer attributes lookup...")
        t1 = time.time()
        cur.execute(sql_cust)
        cols_ch = [d[0] for d in cur.description]
        rows_ch = cur.fetchall()
        logger.info("  Query 2 done: %s rows in %.1fs", f"{len(rows_ch):,}", time.time() - t1)

    logger.info("  Building DataFrames and merging customer attributes...")
    df = pd.DataFrame(rows_fact, columns=cols_fact)
    ch = pd.DataFrame(rows_ch, columns=cols_ch)

    df["startdate"] = pd.to_datetime(df["startdate"])
    for c in ["demand_qty", "sales_qty", "oos_qty"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")

    # Merge customer attributes (fast Python merge on pre-loaded small table)
    df = df.merge(ch, on=["customer_no", "site"], how="left")
    for col in ["channel", "store_type", "chain_type", "sub_channel", "cust_status", "delivery_freq", "premise_code"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    df.drop(columns=["site"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Vectorized feature computation for a single (item, loc) group
# ---------------------------------------------------------------------------

def _compute_group_features(grp: pd.DataFrame, item_id: str, loc: str) -> list[dict]:
    """Compute all 25 features for one (item_id, loc) across all its months.

    Uses vectorized pandas where possible, falls back to loops only for
    inherently sequential features (churn, retention).
    """
    grp = grp.sort_values("startdate")
    item_months = grp["startdate"].unique()
    results = []

    # Pre-compute first order per customer (once per group, not per month)
    first_orders = grp.groupby("customer_no")["startdate"].min()

    for target_month in item_months:
        m3_start = target_month - pd.DateOffset(months=2)
        m6_start = target_month - pd.DateOffset(months=5)
        m1_prev = target_month - pd.DateOffset(months=1)
        m6_only_start = target_month - pd.DateOffset(months=5)
        m6_only_end = target_month - pd.DateOffset(months=3)

        w3 = grp[(grp["startdate"] >= m3_start) & (grp["startdate"] <= target_month)]
        if w3.empty:
            results.append({"item_id": item_id, "loc": loc, "startdate": target_month})
            continue

        w6 = grp[(grp["startdate"] >= m6_start) & (grp["startdate"] <= target_month)]
        w1 = grp[grp["startdate"] == target_month]
        w1_prev = grp[grp["startdate"] == m1_prev]
        w6_only = grp[(grp["startdate"] >= m6_only_start) & (grp["startdate"] <= m6_only_end)]

        td3 = float(w3["demand_qty"].sum())
        ts3 = float(w3["sales_qty"].sum())
        to3 = float(w3["oos_qty"].sum())

        # --- Concentration (vectorized) ---
        cust_demand = w3.groupby("customer_no")["demand_qty"].sum()
        n3 = len(cust_demand)
        n6 = w6["customer_no"].nunique() if not w6.empty else 0
        shares = cust_demand / max(td3, 1e-9)
        hhi = float((shares ** 2).sum())
        top1 = float(shares.max()) if n3 > 0 else 0
        top3 = float(shares.nlargest(3).sum()) if n3 > 0 else 0

        if n3 > 1:
            ss = np.sort(shares.values)
            ss_sum = ss.sum()
            if ss_sum > 0:
                idx = np.arange(1, n3 + 1)
                gini = max(0.0, float((2 * (idx * ss).sum() / (n3 * ss_sum)) - (n3 + 1) / n3))
            else:
                gini = 0.0
        else:
            gini = 0.0

        # --- Dynamics ---
        new_custs = first_orders[first_orders >= m3_start]
        new_demand = float(cust_demand.reindex(new_custs.index).sum()) if len(new_custs) > 0 else 0
        new_share = new_demand / max(td3, 1e-9)

        custs_prior = set(w6_only["customer_no"].unique()) if not w6_only.empty else set()
        custs_recent = set(w3["customer_no"].unique())
        churned = custs_prior - custs_recent
        prior_demand = float(w6_only[w6_only["customer_no"].isin(churned)]["demand_qty"].sum()) if churned else 0
        prior_total = float(w6_only["demand_qty"].sum()) if not w6_only.empty else 1
        churned_share = prior_demand / max(prior_total, 1e-9)

        n_curr = w1["customer_no"].nunique() if not w1.empty else 0
        n_prev = w1_prev["customer_no"].nunique() if not w1_prev.empty else 0
        cust_mom = (n_curr - n_prev) / max(n_prev, 1) if n_prev > 0 else 0.0

        c_curr = set(w1["customer_no"].unique()) if not w1.empty else set()
        c_prev = set(w1_prev["customer_no"].unique()) if not w1_prev.empty else set()
        retention = len(c_curr & c_prev) / max(len(c_prev), 1)

        active_first = first_orders.reindex(cust_demand.index).dropna()
        if len(active_first) > 0:
            tenure = np.maximum(((target_month - active_first).dt.days / 30.44).values, 0)
            tenure_mean = float(tenure.mean())
        else:
            tenure_mean = 0.0

        # --- True demand ---
        td_ratio = td3 / max(ts3, 1e-9)
        oos_rate_3m = to3 / max(td3, 1e-9)
        oos_custs = w3[w3["oos_qty"] > 0]["customer_no"].nunique()
        oos_pct = oos_custs / max(n3, 1)
        gap = td3 - ts3

        td6 = float(w6["demand_qty"].sum()) if not w6.empty else 0
        to6 = float(w6["oos_qty"].sum()) if not w6.empty else 0
        oos_rate_6m = to6 / max(td6, 1e-9)
        oos_trend = (oos_rate_3m - oos_rate_6m) / max(oos_rate_6m, 1e-9) if oos_rate_6m > 0 else 0

        prev_m = grp[grp["startdate"] == m1_prev]
        demand_lag1 = float(prev_m["demand_qty"].sum()) if not prev_m.empty else 0
        lag_vals = []
        for off in range(1, 4):
            m = grp[grp["startdate"] == target_month - pd.DateOffset(months=off)]
            lag_vals.append(float(m["demand_qty"].sum()) if not m.empty else 0)
        demand_lag3_mean = float(np.mean(lag_vals))

        # --- Channel mix ---
        ch_demand = w3.groupby("channel")["demand_qty"].sum()
        ch_shares = ch_demand / max(td3, 1e-9)
        ch_entropy = float(-np.sum(ch_shares[ch_shares > 0] * np.log(ch_shares[ch_shares > 0])))
        dom_ch = float(ch_shares.max()) if len(ch_shares) > 0 else 0
        on_prem = float(ch_demand.get("On Premise", 0) / max(td3, 1e-9))

        ch_prior = w6_only.groupby("channel")["demand_qty"].sum() if not w6_only.empty else pd.Series(dtype=float)
        if not ch_prior.empty:
            ch_p_shares = ch_prior / max(ch_prior.sum(), 1e-9)
            all_ch = set(ch_shares.index) | set(ch_p_shares.index)
            mix_shift = sum(abs(ch_shares.get(c, 0) - ch_p_shares.get(c, 0)) for c in all_ch)
        else:
            mix_shift = 0.0

        # --- Cross-customer (simplified for speed) ---
        top_c = cust_demand.nlargest(min(10, n3)).index
        cm = w6[w6["customer_no"].isin(top_c)].groupby(["customer_no", "startdate"])["demand_qty"].sum().unstack(fill_value=0)
        if not cm.empty and cm.shape[1] >= 2:
            cvs = cm.std(axis=1) / np.maximum(cm.mean(axis=1).values, 1e-9)
            cv_mean = float(cvs.mean())
        else:
            cv_mean = 0.0

        top5 = cust_demand.nlargest(min(5, n3)).index
        sm = w6[w6["customer_no"].isin(top5)].groupby(["customer_no", "startdate"])["demand_qty"].sum().unstack(fill_value=0)
        if sm.shape[0] >= 2 and sm.shape[1] >= 3:
            corr = sm.T.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            sync = float(corr.where(mask).mean().mean())
            sync = 0.0 if np.isnan(sync) else sync
        else:
            sync = 0.0

        if not w1.empty and not w1_prev.empty:
            sc = w1.groupby("customer_no")["demand_qty"].sum()
            sp = w1_prev.groupby("customer_no")["demand_qty"].sum()
            tc = max(float(sc.sum()), 1e-9)
            tp = max(float(sp.sum()), 1e-9)
            all_c = set(sc.index) | set(sp.index)
            max_delta = max(abs(sc.get(c, 0) / tc - sp.get(c, 0) / tp) for c in all_c) if all_c else 0
        else:
            max_delta = 0.0

        # --- Store type mix (same pattern as channel) ---
        def _entropy_and_dom(series, demand_col="demand_qty"):
            grp_dem = series.groupby(series.name if hasattr(series, 'name') else series)[demand_col].sum() if hasattr(series, 'groupby') else pd.Series(dtype=float)
            return 0.0, 0.0  # fallback

        st_demand = w3.groupby("store_type")["demand_qty"].sum() if "store_type" in w3.columns else pd.Series(dtype=float)
        st_shares = st_demand / max(td3, 1e-9) if not st_demand.empty else pd.Series(dtype=float)
        st_entropy = float(-np.sum(st_shares[st_shares > 0] * np.log(st_shares[st_shares > 0]))) if not st_shares.empty else 0
        dom_st = float(st_shares.max()) if len(st_shares) > 0 else 0

        # --- Chain vs independent ratio ---
        if "chain_type" in w3.columns:
            chain_demand = w3.groupby("chain_type")["demand_qty"].sum()
            chain_total = chain_demand.sum()
            # "Independent" or similar = not chained
            indep_keys = [k for k in chain_demand.index if "indep" in str(k).lower()]
            indep_demand = sum(chain_demand.get(k, 0) for k in indep_keys)
            chain_ratio = 1.0 - (indep_demand / max(chain_total, 1e-9))
        else:
            chain_ratio = 0.0

        # --- Top chain concentration ---
        if "chain_type" in w3.columns and not chain_demand.empty:
            top_chain_share = float(chain_demand.max() / max(chain_total, 1e-9))
        else:
            top_chain_share = 0.0

        # --- Sub-channel diversity ---
        sc_demand = w3.groupby("sub_channel")["demand_qty"].sum() if "sub_channel" in w3.columns else pd.Series(dtype=float)
        sc_shares = sc_demand / max(td3, 1e-9) if not sc_demand.empty else pd.Series(dtype=float)
        sc_entropy = float(-np.sum(sc_shares[sc_shares > 0] * np.log(sc_shares[sc_shares > 0]))) if not sc_shares.empty else 0

        # --- Active customer percentage ---
        if "cust_status" in w3.columns:
            status_counts = w3.drop_duplicates("customer_no")["cust_status"].value_counts()
            active_keys = [k for k in status_counts.index if "act" in str(k).lower() or k == "A"]
            n_active_status = sum(status_counts.get(k, 0) for k in active_keys)
            active_pct = n_active_status / max(status_counts.sum(), 1) if status_counts.sum() > 0 else 1.0
        else:
            active_pct = 1.0

        # --- Avg delivery frequency score ---
        if "delivery_freq" in w3.columns:
            freq_map = {"D": 5, "W": 4, "2W": 3, "M": 2, "Q": 1}
            freqs = w3.drop_duplicates("customer_no")["delivery_freq"].map(freq_map)
            avg_delivery_freq = float(freqs.mean()) if not freqs.dropna().empty else 0
        else:
            avg_delivery_freq = 0.0

        results.append({
            "item_id": item_id, "loc": loc, "startdate": target_month,
            "n_active_cust": n3, "n_active_cust_6m": n6,
            "hhi_demand": round(hhi, 4), "top1_cust_share": round(top1, 4),
            "top3_cust_share": round(top3, 4), "cust_gini": round(gini, 4),
            "new_cust_demand_share": round(new_share, 4),
            "churned_cust_demand_share": round(churned_share, 4),
            "cust_count_mom": round(cust_mom, 4),
            "cust_retention_rate": round(retention, 4),
            "cust_tenure_mean": round(tenure_mean, 1),
            "true_demand_ratio": round(td_ratio, 4),
            "oos_rate": round(oos_rate_3m, 4), "oos_cust_pct": round(oos_pct, 4),
            "demand_sales_gap_3m": round(gap, 1), "oos_trend": round(oos_trend, 4),
            "demand_qty_lag1": round(demand_lag1, 1),
            "demand_qty_lag3_mean": round(demand_lag3_mean, 1),
            "channel_entropy": round(ch_entropy, 4),
            "dominant_channel_share": round(dom_ch, 4),
            "channel_mix_shift": round(float(mix_shift), 4),
            "on_premise_share": round(on_prem, 4),
            "cust_demand_cv_mean": round(cv_mean, 4),
            "cust_demand_sync": round(sync, 4),
            "max_cust_share_delta": round(float(max_delta), 4),
            # New attribute-based features
            "store_type_entropy": round(st_entropy, 4),
            "dominant_store_type_share": round(dom_st, 4),
            "chain_ratio": round(chain_ratio, 4),
            "top_chain_share": round(top_chain_share, 4),
            "sub_channel_entropy": round(sc_entropy, 4),
            "active_cust_pct": round(active_pct, 4),
            "avg_delivery_freq": round(avg_delivery_freq, 2),
            # Premise code: on-premise vs off-premise at account level
            "on_premise_acct_share": round(
                float(w3[w3["premise_code"].str.upper().isin(["ON", "O", "ON-PREMISE", "ON PREMISE"])]["demand_qty"].sum() / max(td3, 1e-9))
                if "premise_code" in w3.columns else 0.0, 4),
            "premise_diversity": round(
                float(w3["premise_code"].nunique() / max(n3, 1))
                if "premise_code" in w3.columns else 0.0, 4),
        })

    return results


# ---------------------------------------------------------------------------
# Parallel chunk processor
# ---------------------------------------------------------------------------

def _process_chunk(chunk_df: pd.DataFrame, chunk_idx: int, n_chunks: int) -> pd.DataFrame:
    """Process a chunk of (item, loc) groups. Called in worker processes."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    wlog = logging.getLogger(f"cust_feat.worker.{chunk_idx}")

    groups = list(chunk_df.groupby(["item_id", "loc"]))
    n_groups = len(groups)
    all_results: list[dict] = []

    for i, ((item_id, loc), grp) in enumerate(groups):
        if (i + 1) % 500 == 0 or i == 0:
            wlog.info("  Chunk %d/%d: group %d/%d (%s × %s)",
                      chunk_idx + 1, n_chunks, i + 1, n_groups, item_id, loc)
        all_results.extend(_compute_group_features(grp, item_id, loc))

    wlog.info("  Chunk %d/%d complete: %d groups → %d feature rows",
              chunk_idx + 1, n_chunks, n_groups, len(all_results))
    return pd.DataFrame(all_results)


def compute_features(df: pd.DataFrame, n_workers: int = 1) -> pd.DataFrame:
    """Compute all 34 customer features at item × loc × month grain.

    Optionally parallelizes across (item, loc) groups using ProcessPoolExecutor.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(["item_id", "loc", "startdate", "customer_no"])
    groups = df.groupby(["item_id", "loc"]).ngroups
    logger.info("Computing features for %s item×loc groups (workers=%d)...", f"{groups:,}", n_workers)

    if n_workers <= 1:
        # Sequential: process all groups in main process with progress logging
        all_results: list[dict] = []
        group_list = list(df.groupby(["item_id", "loc"]))
        n_groups = len(group_list)
        t0 = time.time()

        for i, ((item_id, loc), grp) in enumerate(group_list):
            all_results.extend(_compute_group_features(grp, item_id, loc))
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_groups - i - 1) / max(rate, 0.01)
                logger.info(
                    "  Progress: %d/%d groups (%.0f%%) — %.1f groups/sec — ETA %.0fs",
                    i + 1, n_groups, (i + 1) / n_groups * 100, rate, eta,
                )

        logger.info("  Sequential complete: %d feature rows from %d groups",
                    len(all_results), n_groups)
        return pd.DataFrame(all_results)

    # Parallel: split by unique (item, loc) keys into N chunks
    unique_keys = df[["item_id", "loc"]].drop_duplicates()
    chunk_size = max(1, len(unique_keys) // n_workers)
    key_chunks = [unique_keys.iloc[i:i + chunk_size] for i in range(0, len(unique_keys), chunk_size)]

    logger.info("Splitting %d groups into %d chunks (chunk_size~%d)...",
                groups, len(key_chunks), chunk_size)

    # Partition data by chunk
    data_chunks = []
    for keys_chunk in key_chunks:
        mask = df.set_index(["item_id", "loc"]).index.isin(
            keys_chunk.set_index(["item_id", "loc"]).index
        )
        data_chunks.append(df[mask].copy())

    results: list[pd.DataFrame] = []
    n_chunks = len(data_chunks)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_chunk, chunk, ci, n_chunks): ci
            for ci, chunk in enumerate(data_chunks)
        }
        for fut in as_completed(futures):
            ci = futures[fut]
            try:
                chunk_result = fut.result()
                results.append(chunk_result)
                logger.info("  Chunk %d/%d returned %s rows", ci + 1, n_chunks, f"{len(chunk_result):,}")
            except Exception:
                logger.exception("  Chunk %d failed", ci)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# DB upsert (batched for speed)
# ---------------------------------------------------------------------------

def _upsert_features(db: dict, features_df: pd.DataFrame) -> int:
    """Upsert features into customer_features_monthly using batch COPY."""
    import psycopg
    import io

    cols = [c for c in features_df.columns if c not in ("load_ts",)]

    # Convert types for COPY
    out = features_df[cols].copy()
    for c in out.columns:
        if c == "startdate":
            out[c] = pd.to_datetime(out[c]).dt.date
        elif out[c].dtype in ("float32", "float64"):
            out[c] = out[c].round(4)

    # Write to temp table via COPY, then upsert
    col_list = ", ".join(cols)
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in ("item_id", "loc", "startdate"))

    logger.info("  Upserting %s rows to customer_features_monthly...", f"{len(out):,}")
    with psycopg.connect(**db) as conn:
        with conn.cursor() as cur:
            # Create temp table
            cur.execute("CREATE TEMP TABLE _cust_feat_stage (LIKE customer_features_monthly INCLUDING DEFAULTS) ON COMMIT DROP")

            # COPY data in
            buf = io.StringIO()
            out.to_csv(buf, index=False, header=False, sep="\t", na_rep="\\N")
            buf.seek(0)
            stage_cols = ", ".join(cols)
            with cur.copy(f"COPY _cust_feat_stage ({stage_cols}) FROM STDIN") as copy:
                for line in buf:
                    copy.write(line.encode())

            logger.info("  COPY to staging complete, running upsert...")
            cur.execute(f"""
                INSERT INTO customer_features_monthly ({col_list})
                SELECT {col_list} FROM _cust_feat_stage
                ON CONFLICT (item_id, loc, startdate) DO UPDATE SET {update_set}
            """)
            n = cur.rowcount
        conn.commit()
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate customer-derived features")
    parser.add_argument("--months", type=int, default=36, help="Lookback months (default 36)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default 4)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    db = get_db_params()

    logger.info("=" * 60)
    logger.info("Customer Feature Generation — Start")
    logger.info("=" * 60)

    # Step 1: Load
    logger.info("Step 1/3: Loading customer demand data (lookback=%d months)...", args.months)
    t0 = time.time()
    with profiled_section("load_customer_demand"):
        df = _load_customer_demand(db, lookback_months=args.months)

    n_il = df.groupby(["item_id", "loc"]).ngroups
    logger.info(
        "  Loaded: %s rows, %s item×loc pairs, %s months, %s customers (%.1fs)",
        f"{len(df):,}", f"{n_il:,}",
        f"{df['startdate'].nunique()}", f"{df['customer_no'].nunique():,}",
        time.time() - t0,
    )

    if df.empty:
        logger.warning("No customer demand data found; exiting")
        return

    # Step 2: Compute
    logger.info("Step 2/3: Computing 34 customer features (%d workers)...", args.workers)
    t1 = time.time()
    with profiled_section("compute_features"):
        features = compute_features(df, n_workers=args.workers)

    logger.info("  Computed: %s feature rows in %.1fs", f"{len(features):,}", time.time() - t1)

    if features.empty:
        logger.warning("No features computed; exiting")
        return

    # Step 3: Upsert
    logger.info("Step 3/3: Upserting to customer_features_monthly...")
    t2 = time.time()
    with profiled_section("upsert_features"):
        n = _upsert_features(db, features)

    logger.info("  Upserted: %s rows in %.1fs", f"{n:,}", time.time() - t2)
    logger.info("=" * 60)
    logger.info("Customer Feature Generation — Done (%.1fs total)", time.time() - t0)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
