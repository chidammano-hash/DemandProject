"""
Run XGBoost backtesting with expanding-window timeframes.

Supports three strategies:
  - global:      One XGBoost for all DFUs, ml_cluster as categorical feature (model_id=xgboost_global)
  - per_cluster:  Separate XGBoost per ml_cluster (model_id=xgboost_cluster)
  - transfer:    Global base model (no ml_cluster) → per-cluster fine-tune via xgb_model (model_id=xgboost_transfer)

Produces two CSVs:
  - backtest_predictions.csv: execution-lag only (for fact_external_forecast_monthly)
  - backtest_predictions_all_lags.csv: lag 0-4 archive (for backtest_lag_archive)
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import psycopg
import xgboost as xgb
import mlflow

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ts() -> str:
    """Timestamp prefix for log messages."""
    return time.strftime("%H:%M:%S")


# ── DB connection ───────────────────────────────────────────────────────────

def get_db_conn() -> dict[str, Any]:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5440")),
        "dbname": os.getenv("POSTGRES_DB", "demand_mvp"),
        "user": os.getenv("POSTGRES_USER", "demand"),
        "password": os.getenv("POSTGRES_PASSWORD", "demand"),
    }


# ── Timeframe generation ───────────────────────────────────────────────────

def generate_timeframes(
    earliest: pd.Timestamp,
    latest: pd.Timestamp,
    n: int = 10,
) -> list[dict]:
    """Generate N expanding-window timeframes.

    For timeframe i (A=0 .. J=9):
      train_end   = latest - (N - i) months
      predict     = [train_end + 1 month, latest]
    """
    timeframes = []
    for i in range(n):
        train_end = latest - pd.DateOffset(months=(n - i))
        train_end = train_end.normalize()  # midnight
        predict_start = train_end + pd.DateOffset(months=1)
        label = chr(ord("A") + i)
        timeframes.append({
            "label": label,
            "index": i,
            "train_start": earliest,
            "train_end": train_end,
            "predict_start": predict_start,
            "predict_end": latest,
        })
    return timeframes


# ── Feature engineering ─────────────────────────────────────────────────────

CAT_FEATURES = ["ml_cluster", "region", "brand", "abc_vol"]
NUMERIC_DFU_FEATURES = ["execution_lag", "total_lt"]
NUMERIC_ITEM_FEATURES = ["case_weight", "item_proof", "bpc"]

LAG_RANGE = range(1, 13)  # qty_lag_1 .. qty_lag_12
ROLLING_WINDOWS = [3, 6, 12]


def build_feature_matrix(
    sales_df: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
    item_attrs: pd.DataFrame,
    all_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """Build FULL feature matrix: one row per (dfu_ck, month).

    Features are strictly causal — only data from months < target month used.
    Built ONCE for all timeframes; per-timeframe masking done externally.
    """
    t0 = time.time()
    dfu_keys = dfu_attrs[["dfu_ck", "dmdunit", "dmdgroup", "loc"]].drop_duplicates()
    n_dfus = len(dfu_keys)
    n_months = len(all_months)
    print(f"  [{_ts()}] Building grid: {n_dfus:,} DFUs × {n_months} months = {n_dfus * n_months:,} rows")

    # Build complete grid via MultiIndex (faster than cross-join merge)
    idx = pd.MultiIndex.from_product(
        [dfu_keys["dfu_ck"].values, all_months],
        names=["dfu_ck", "startdate"],
    )
    grid = pd.DataFrame(index=idx).reset_index()
    grid = grid.merge(dfu_keys, on="dfu_ck", how="left")
    print(f"  [{_ts()}] Grid built: {len(grid):,} rows ({time.time() - t0:.1f}s)")

    # Join sales (full — no cutoff; masking done per timeframe)
    t1 = time.time()
    grid = grid.merge(
        sales_df[["dfu_ck", "startdate", "qty"]],
        on=["dfu_ck", "startdate"],
        how="left",
    )
    grid["qty"] = grid["qty"].fillna(0)
    print(f"  [{_ts()}] Sales joined ({time.time() - t1:.1f}s)")

    # Sort for lag/rolling operations
    t1 = time.time()
    grid = grid.sort_values(["dfu_ck", "startdate"]).reset_index(drop=True)
    print(f"  [{_ts()}] Sorted ({time.time() - t1:.1f}s)")

    # Lag features (vectorized groupby shift)
    t1 = time.time()
    g = grid.groupby("dfu_ck", sort=False)["qty"]
    for lag_n in LAG_RANGE:
        grid[f"qty_lag_{lag_n}"] = g.shift(lag_n)
    print(f"  [{_ts()}] Lag features 1-12 done ({time.time() - t1:.1f}s)")

    # Rolling stats (vectorized — no lambda)
    t1 = time.time()
    shifted = g.shift(1)
    for w in ROLLING_WINDOWS:
        rolling = shifted.groupby(grid["dfu_ck"], sort=False).rolling(w, min_periods=1)
        grid[f"rolling_mean_{w}m"] = rolling.mean().reset_index(level=0, drop=True)
        grid[f"rolling_std_{w}m"] = rolling.std().fillna(0).reset_index(level=0, drop=True)
    print(f"  [{_ts()}] Rolling stats done ({time.time() - t1:.1f}s)")

    # Calendar features
    grid["month"] = grid["startdate"].dt.month
    grid["quarter"] = grid["startdate"].dt.quarter
    grid["month_sin"] = np.sin(2 * np.pi * grid["month"] / 12)
    grid["month_cos"] = np.cos(2 * np.pi * grid["month"] / 12)

    # DFU attributes
    t1 = time.time()
    dfu_feat_cols = ["dfu_ck"] + CAT_FEATURES + NUMERIC_DFU_FEATURES
    dfu_feat_cols = [c for c in dfu_feat_cols if c in dfu_attrs.columns]
    grid = grid.merge(dfu_attrs[dfu_feat_cols], on="dfu_ck", how="left")

    # Item attributes
    if len(item_attrs) > 0:
        grid = grid.merge(item_attrs, on="dmdunit", how="left")
    print(f"  [{_ts()}] Attributes joined ({time.time() - t1:.1f}s)")

    # Fill missing numerics
    for col in NUMERIC_DFU_FEATURES + NUMERIC_ITEM_FEATURES:
        if col in grid.columns:
            grid[col] = grid[col].fillna(0)

    # XGBoost supports pandas Categorical dtype natively (enable_categorical=True)
    for col in CAT_FEATURES:
        if col in grid.columns:
            grid[col] = grid[col].fillna("__unknown__").astype("category")

    print(f"  [{_ts()}] Feature matrix complete: {grid.shape} ({time.time() - t0:.1f}s total)")
    return grid


def get_feature_columns(grid: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (everything except metadata/target)."""
    exclude = {"dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate", "qty", "_k"}
    return [c for c in grid.columns if c not in exclude]


def mask_future_sales(grid: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Zero out qty and recompute lag/rolling features for rows after cutoff."""
    df = grid.copy()

    # Mask future sales
    future_mask = df["startdate"] > cutoff
    df.loc[future_mask, "qty"] = 0

    # Recompute lags and rolling on the masked data
    g = df.groupby("dfu_ck", sort=False)["qty"]
    for lag_n in LAG_RANGE:
        df[f"qty_lag_{lag_n}"] = g.shift(lag_n)

    shifted = g.shift(1)
    for w in ROLLING_WINDOWS:
        rolling = shifted.groupby(df["dfu_ck"], sort=False).rolling(w, min_periods=1)
        df[f"rolling_mean_{w}m"] = rolling.mean().reset_index(level=0, drop=True)
        df[f"rolling_std_{w}m"] = rolling.std().fillna(0).reset_index(level=0, drop=True)

    return df


# ── Training & prediction ──────────────────────────────────────────────────

def train_and_predict_global(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict,
) -> pd.DataFrame:
    """Train one global XGBoost and predict."""
    t0 = time.time()
    X_train = train_df[feature_cols]
    y_train = train_df["qty"]
    X_pred = predict_df[feature_cols]

    print(f"    [{_ts()}] Training XGBoost global ({len(X_train):,} rows, {len(feature_cols)} features)...")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    print(f"    [{_ts()}] Training done ({time.time() - t0:.1f}s)")

    print(f"    [{_ts()}] Predicting {len(X_pred):,} rows...")
    preds = model.predict(X_pred)
    result = predict_df[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
    result["basefcst_pref"] = np.maximum(preds, 0)  # floor at 0
    print(f"    [{_ts()}] Prediction done ({time.time() - t0:.1f}s total)")
    return result, model


def train_and_predict_per_cluster(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict,
) -> pd.DataFrame:
    """Train separate XGBoost per ml_cluster."""
    # Exclude ml_cluster from features (it's the grouping key)
    feat_cols_no_cluster = [c for c in feature_cols if c != "ml_cluster"]

    all_results = []
    models = {}

    clusters = sorted(train_df["ml_cluster"].dropna().unique())
    print(f"    [{_ts()}] Training {len(clusters)} per-cluster models...")
    for ci, cluster_label in enumerate(clusters, 1):
        train_c = train_df[train_df["ml_cluster"] == cluster_label]
        pred_c = predict_df[predict_df["ml_cluster"] == cluster_label]

        if len(train_c) < 50 or len(pred_c) == 0:
            if len(pred_c) > 0:
                print(f"    [{_ts()}] Cluster {ci}/{len(clusters)} '{cluster_label}': skipped (train={len(train_c)}), "
                      f"zeroing {len(pred_c)} predictions")
                result = pred_c[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
                result["basefcst_pref"] = 0.0
                all_results.append(result)
            continue

        X_train = train_c[feat_cols_no_cluster]
        y_train = train_c["qty"]
        X_pred = pred_c[feat_cols_no_cluster]

        t0 = time.time()
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_pred)

        result = pred_c[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
        result["basefcst_pref"] = np.maximum(preds, 0)
        all_results.append(result)
        models[cluster_label] = model
        print(f"    [{_ts()}] Cluster {ci}/{len(clusters)} '{cluster_label}': "
              f"train={len(train_c):,}, pred={len(pred_c):,} ({time.time() - t0:.1f}s)")

    # Handle DFUs with no cluster assignment
    no_cluster = predict_df[predict_df["ml_cluster"].isna() | (predict_df["ml_cluster"] == "__unknown__")]
    if len(no_cluster) > 0:
        print(f"    [{_ts()}] {len(no_cluster)} predict rows with no cluster → using 0")
        result = no_cluster[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
        result["basefcst_pref"] = 0.0
        all_results.append(result)

    return pd.concat(all_results, ignore_index=True), models


def train_and_predict_transfer(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    params: dict,
    transfer_n_estimators: int = 100,
    transfer_min_rows: int = 20,
) -> pd.DataFrame:
    """Transfer learning: global base model (no ml_cluster) → per-cluster fine-tune.

    Phase 1: Train a base XGBoost on ALL training data, excluding ml_cluster from features.
    Phase 2: For each cluster with >= transfer_min_rows: fine-tune from the base model
             with transfer_n_estimators additional trees via xgb_model.
    Fallback: Clusters < transfer_min_rows or unassigned DFUs use base model predictions.
    """
    # Feature sets without ml_cluster (base and per-cluster use same features)
    feat_cols_no_cluster = [c for c in feature_cols if c != "ml_cluster"]

    # ── Phase 1: Train base model on ALL data (no ml_cluster) ──
    t0 = time.time()
    X_train_all = train_df[feat_cols_no_cluster]
    y_train_all = train_df["qty"]

    print(f"    [{_ts()}] Phase 1: Training base XGBoost ({len(X_train_all):,} rows, "
          f"{len(feat_cols_no_cluster)} features, no ml_cluster)...")
    base_model = xgb.XGBRegressor(**params)
    base_model.fit(X_train_all, y_train_all)
    print(f"    [{_ts()}] Base model trained ({time.time() - t0:.1f}s)")

    # ── Phase 2: Fine-tune per cluster ──
    all_results = []
    models = {"__base__": base_model}

    clusters = sorted(train_df["ml_cluster"].dropna().unique())
    clusters = [c for c in clusters if c != "__unknown__"]
    print(f"    [{_ts()}] Phase 2: Fine-tuning {len(clusters)} clusters "
          f"(min_rows={transfer_min_rows}, extra_trees={transfer_n_estimators})...")

    for ci, cluster_label in enumerate(clusters, 1):
        train_c = train_df[train_df["ml_cluster"] == cluster_label]
        pred_c = predict_df[predict_df["ml_cluster"] == cluster_label]

        if len(pred_c) == 0:
            continue

        if len(train_c) < transfer_min_rows:
            # Fallback: use base model (not zero)
            print(f"    [{_ts()}] Cluster {ci}/{len(clusters)} '{cluster_label}': "
                  f"train={len(train_c)} < {transfer_min_rows} → base model fallback")
            X_pred = pred_c[feat_cols_no_cluster]
            preds = base_model.predict(X_pred)
            result = pred_c[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
            result["basefcst_pref"] = np.maximum(preds, 0)
            all_results.append(result)
            continue

        # Fine-tune: new XGBoost initialized from base model's booster
        X_train_c = train_c[feat_cols_no_cluster]
        y_train_c = train_c["qty"]
        X_pred = pred_c[feat_cols_no_cluster]

        t1 = time.time()
        ft_params = {**params, "n_estimators": transfer_n_estimators}
        ft_model = xgb.XGBRegressor(**ft_params)
        ft_model.fit(
            X_train_c, y_train_c,
            xgb_model=base_model.get_booster(),
        )
        preds = ft_model.predict(X_pred)

        result = pred_c[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
        result["basefcst_pref"] = np.maximum(preds, 0)
        all_results.append(result)
        models[cluster_label] = ft_model
        print(f"    [{_ts()}] Cluster {ci}/{len(clusters)} '{cluster_label}': "
              f"train={len(train_c):,}, pred={len(pred_c):,}, fine-tuned ({time.time() - t1:.1f}s)")

    # Handle DFUs with no cluster assignment → base model fallback
    no_cluster = predict_df[predict_df["ml_cluster"].isna() | (predict_df["ml_cluster"] == "__unknown__")]
    if len(no_cluster) > 0:
        print(f"    [{_ts()}] {len(no_cluster)} predict rows with no cluster → base model fallback")
        X_pred = no_cluster[feat_cols_no_cluster]
        preds = base_model.predict(X_pred)
        result = no_cluster[["dfu_ck", "dmdunit", "dmdgroup", "loc", "startdate"]].copy()
        result["basefcst_pref"] = np.maximum(preds, 0)
        all_results.append(result)

    return pd.concat(all_results, ignore_index=True), models


# ── Execution-lag assignment ────────────────────────────────────────────────

def assign_execution_lag(
    pred_df: pd.DataFrame,
    execution_lag_map: dict[str, int],
) -> pd.DataFrame:
    """Assign each prediction its DFU's execution lag and compute fcstdate."""
    t0 = time.time()
    result = pred_df.copy()

    result["execution_lag"] = result["dfu_ck"].map(execution_lag_map).fillna(0).astype(int)
    result["lag"] = result["execution_lag"]
    result["fcstdate"] = result.apply(
        lambda r: r["startdate"] - pd.DateOffset(months=int(r["lag"])), axis=1
    )

    result["forecast_ck"] = (
        result["dmdunit"].astype(str) + "_"
        + result["dmdgroup"].astype(str) + "_"
        + result["loc"].astype(str) + "_"
        + result["fcstdate"].dt.strftime("%Y-%m-%d") + "_"
        + result["startdate"].dt.strftime("%Y-%m-%d")
    )

    print(f"  [{_ts()}] Execution-lag assignment done ({time.time() - t0:.1f}s)")
    return result


# ── All-lag expansion (archive) ────────────────────────────────────────────

def expand_to_all_lags(
    pred_df: pd.DataFrame,
    max_lag: int,
    execution_lag_map: dict[str, int],
) -> pd.DataFrame:
    """Expand each prediction to lag 0 .. max_lag rows for the archive table."""
    t0 = time.time()
    dfs = []
    for lag in range(max_lag + 1):
        df = pred_df.copy()
        df["lag"] = lag
        df["fcstdate"] = df["startdate"] - pd.DateOffset(months=lag)
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    result["execution_lag"] = result["dfu_ck"].map(execution_lag_map).fillna(0).astype(int)

    result["forecast_ck"] = (
        result["dmdunit"].astype(str) + "_"
        + result["dmdgroup"].astype(str) + "_"
        + result["loc"].astype(str) + "_"
        + result["fcstdate"].dt.strftime("%Y-%m-%d") + "_"
        + result["startdate"].dt.strftime("%Y-%m-%d")
    )

    print(f"  [{_ts()}] All-lag expansion (0-{max_lag}) done: {len(result):,} rows ({time.time() - t0:.1f}s)")
    return result


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run XGBoost backtest with expanding-window timeframes")
    parser.add_argument("--cluster-strategy", choices=["global", "per_cluster", "transfer"], default="global",
                        help="global: one model, per_cluster: model per ml_cluster, transfer: global base → per-cluster fine-tune")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model_id (default: xgboost_global, xgboost_cluster, or xgboost_transfer)")
    parser.add_argument("--n-timeframes", type=int, default=10, help="Number of expanding windows")
    parser.add_argument("--output-dir", type=str, default="data/backtest", help="Output directory")
    # Transfer learning params
    parser.add_argument("--transfer-n-estimators", type=int, default=100,
                        help="Number of additional trees for per-cluster fine-tuning (transfer strategy)")
    parser.add_argument("--transfer-min-rows", type=int, default=20,
                        help="Minimum cluster rows for fine-tuning; smaller clusters use base model (transfer strategy)")
    # XGBoost hyperparams
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=int, default=5)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--verbosity", type=int, default=0, help="XGBoost verbosity (0=silent)")
    args = parser.parse_args()

    t_start = time.time()
    load_dotenv(ROOT / ".env")
    db = get_db_conn()

    _default_model_ids = {
        "global": "xgboost_global",
        "per_cluster": "xgboost_cluster",
        "transfer": "xgboost_transfer",
    }
    model_id = args.model_id or _default_model_ids[args.cluster_strategy]
    print(f"[{_ts()}] Backtest: strategy={args.cluster_strategy}, model_id={model_id}, "
          f"n_timeframes={args.n_timeframes}")

    # Detect GPU support for XGBoost
    _use_gpu = False
    try:
        _test = xgb.XGBRegressor(device="cuda", n_estimators=1, verbosity=0)
        _test.fit([[0]], [0])
        _use_gpu = True
        print(f"[{_ts()}] Using GPU (CUDA) for XGBoost")
    except Exception:
        print(f"[{_ts()}] GPU not available, falling back to CPU")

    xgb_params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "verbosity": args.verbosity,
        "random_state": 42,
        "n_jobs": -1,
        "enable_categorical": True,
        "tree_method": "hist",
    }
    if _use_gpu:
        xgb_params["device"] = "cuda"

    # ── Step 1: Load data ───────────────────────────────────────────────────
    print(f"\n[{_ts()}] Step 1: Loading data from Postgres...")
    t1 = time.time()
    with psycopg.connect(**db) as conn:
        sales_df = pd.read_sql("""
            SELECT d.dfu_ck, s.dmdunit, s.dmdgroup, s.loc, s.startdate, s.qty
            FROM fact_sales_monthly s
            INNER JOIN dim_dfu d
                ON d.dmdunit = s.dmdunit AND d.dmdgroup = s.dmdgroup AND d.loc = s.loc
            WHERE s.qty IS NOT NULL
            ORDER BY d.dfu_ck, s.startdate
        """, conn)

        dfu_attrs = pd.read_sql("""
            SELECT dfu_ck, dmdunit, dmdgroup, loc,
                   execution_lag, total_lt, ml_cluster,
                   brand, region, abc_vol
            FROM dim_dfu
        """, conn)

        item_attrs = pd.read_sql("""
            SELECT DISTINCT i.item_no AS dmdunit,
                   i.case_weight, i.item_proof, i.bpc
            FROM dim_item i
            INNER JOIN dim_dfu d ON i.item_no = d.dmdunit
        """, conn)

    sales_df["startdate"] = pd.to_datetime(sales_df["startdate"])
    sales_df["qty"] = pd.to_numeric(sales_df["qty"], errors="coerce").fillna(0)

    # Only keep DFUs that have sales
    dfus_with_sales = set(sales_df["dfu_ck"].unique())
    dfu_attrs = dfu_attrs[dfu_attrs["dfu_ck"].isin(dfus_with_sales)].copy()

    print(f"  [{_ts()}] Sales: {len(sales_df):,} rows, {len(dfus_with_sales):,} DFUs ({time.time() - t1:.1f}s)")
    print(f"  [{_ts()}] DFU attrs: {len(dfu_attrs):,}, Item attrs: {len(item_attrs):,}")

    # Execution lag lookup
    exec_lag_map = dfu_attrs.set_index("dfu_ck")["execution_lag"].fillna(0).astype(int).to_dict()

    # ── Step 2: Generate timeframes ─────────────────────────────────────────
    latest_month = sales_df["startdate"].max()
    earliest_month = sales_df["startdate"].min()
    print(f"  [{_ts()}] Date range: {earliest_month.date()} → {latest_month.date()}")

    timeframes = generate_timeframes(earliest_month, latest_month, args.n_timeframes)
    print(f"\n[{_ts()}] Step 2: Generated {len(timeframes)} timeframes:")
    for tf in timeframes:
        print(f"  {tf['label']}: train [{tf['train_start'].date()} → {tf['train_end'].date()}], "
              f"predict [{tf['predict_start'].date()} → {tf['predict_end'].date()}]")

    # All months in the data range
    all_months = sorted(sales_df["startdate"].unique())

    # ── Step 3: Build feature matrix ONCE ────────────────────────────────────
    print(f"\n[{_ts()}] Step 3: Building feature matrix (one-time)...")
    full_grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, all_months)
    feature_cols = get_feature_columns(full_grid)
    cat_cols = [c for c in CAT_FEATURES if c in feature_cols and c in full_grid.columns]
    print(f"  [{_ts()}] Features: {len(feature_cols)} columns, cat: {cat_cols}")

    # ── Step 4: Train & predict per timeframe ────────────────────────────────
    print(f"\n[{_ts()}] Step 4: Running {len(timeframes)} timeframe backtests...")
    all_predictions = []

    for ti, tf in enumerate(timeframes):
        label = tf["label"]
        train_end = tf["train_end"]
        predict_start = tf["predict_start"]
        predict_end = tf["predict_end"]
        tf_start = time.time()

        print(f"\n── Timeframe {label} ({ti + 1}/{len(timeframes)}) ──")

        predict_months = [m for m in all_months if predict_start <= m <= predict_end]
        if not predict_months:
            print(f"  [{_ts()}] No predict months — skipping")
            continue

        train_months = [m for m in all_months if earliest_month <= m <= train_end]
        if len(train_months) < 13:
            print(f"  [{_ts()}] Insufficient training months ({len(train_months)}) — need 13 min — skipping")
            continue

        # Mask future sales and recompute lag/rolling features
        print(f"  [{_ts()}] Masking sales after {train_end.date()} and recomputing features...")
        t1 = time.time()
        masked_grid = mask_future_sales(full_grid, train_end)
        print(f"  [{_ts()}] Masking done ({time.time() - t1:.1f}s)")

        # Split train / predict
        train_mask = masked_grid["startdate"] <= train_end
        predict_mask = masked_grid["startdate"].isin(predict_months)

        # Drop rows with NaN in lag features (first few months of each DFU)
        train_data = masked_grid[train_mask].dropna(subset=[f"qty_lag_{lag}" for lag in LAG_RANGE])
        predict_data = masked_grid[predict_mask].copy()

        # Fill NaN lag features in predict data with 0 (skip categoricals)
        for col in feature_cols:
            if col in predict_data.columns and col not in cat_cols:
                predict_data[col] = predict_data[col].fillna(0)

        print(f"  [{_ts()}] Train: {len(train_data):,} rows, Predict: {len(predict_data):,} rows")

        if len(train_data) == 0 or len(predict_data) == 0:
            print(f"  [{_ts()}] Empty train or predict — skipping")
            continue

        # Train & predict
        if args.cluster_strategy == "global":
            preds, model = train_and_predict_global(
                train_data, predict_data, feature_cols, cat_cols, xgb_params
            )
        elif args.cluster_strategy == "transfer":
            preds, models = train_and_predict_transfer(
                train_data, predict_data, feature_cols, cat_cols, xgb_params,
                transfer_n_estimators=args.transfer_n_estimators,
                transfer_min_rows=args.transfer_min_rows,
            )
        else:
            preds, models = train_and_predict_per_cluster(
                train_data, predict_data, feature_cols, cat_cols, xgb_params
            )

        preds["model_id"] = model_id
        preds["timeframe"] = label
        preds["timeframe_idx"] = tf["index"]
        all_predictions.append(preds)
        print(f"  [{_ts()}] Timeframe {label} complete: {len(preds):,} predictions ({time.time() - tf_start:.1f}s)")

    if not all_predictions:
        print(f"\n[{_ts()}] No predictions generated. Check data range and timeframe count.")
        sys.exit(1)

    # ── Step 5: Combine, assign execution lag, attach actuals ───────────────
    print(f"\n[{_ts()}] Step 5: Combining predictions...")
    combined = pd.concat(all_predictions, ignore_index=True)
    print(f"  [{_ts()}] Total raw predictions: {len(combined):,}")

    # Assign execution lag and compute fcstdate (one row per prediction)
    print(f"  [{_ts()}] Assigning execution lag per DFU...")
    expanded = assign_execution_lag(combined, exec_lag_map)
    print(f"  [{_ts()}] Rows after execution-lag assignment: {len(expanded):,}")

    # Deduplicate: for same (forecast_ck, model_id), keep latest timeframe
    expanded = expanded.sort_values("timeframe_idx")
    expanded = expanded.drop_duplicates(subset=["forecast_ck", "model_id"], keep="last")
    print(f"  [{_ts()}] After dedup: {len(expanded):,}")

    # Attach actuals via merge
    print(f"  [{_ts()}] Attaching actuals...")
    t1 = time.time()
    actuals = sales_df.drop_duplicates(subset=["dfu_ck", "startdate"])[["dfu_ck", "startdate", "qty"]].rename(
        columns={"qty": "tothist_dmd"}
    )
    expanded = expanded.merge(actuals, on=["dfu_ck", "startdate"], how="left")
    print(f"  [{_ts()}] Actuals attached ({time.time() - t1:.1f}s)")

    # ── Step 6: Save output ─────────────────────────────────────────────────
    print(f"\n[{_ts()}] Step 6: Saving output...")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    out_cols = [
        "forecast_ck", "dmdunit", "dmdgroup", "loc",
        "fcstdate", "startdate", "lag", "execution_lag",
        "basefcst_pref", "tothist_dmd", "model_id",
    ]
    output_df = expanded[out_cols].copy()
    output_df["fcstdate"] = output_df["fcstdate"].dt.strftime("%Y-%m-%d")
    output_df["startdate"] = output_df["startdate"].dt.strftime("%Y-%m-%d")

    output_path = output_dir / "backtest_predictions.csv"
    output_df.to_csv(output_path, index=False)
    print(f"  [{_ts()}] Saved {len(output_df):,} predictions to {output_path}")

    # ── Step 6b: All-lags archive CSV ─────────────────────────────────────
    print(f"  [{_ts()}] Generating all-lags archive (lag 0-4)...")
    archive_expanded = expand_to_all_lags(combined, 4, exec_lag_map)

    # Deduplicate: for same (forecast_ck, model_id, lag), keep latest timeframe
    archive_expanded = archive_expanded.sort_values("timeframe_idx")
    archive_expanded = archive_expanded.drop_duplicates(
        subset=["forecast_ck", "model_id", "lag"], keep="last"
    )
    print(f"  [{_ts()}] Archive after dedup: {len(archive_expanded):,}")

    # Attach actuals
    archive_expanded = archive_expanded.merge(actuals, on=["dfu_ck", "startdate"], how="left")

    archive_cols = [
        "forecast_ck", "dmdunit", "dmdgroup", "loc",
        "fcstdate", "startdate", "lag", "execution_lag",
        "basefcst_pref", "tothist_dmd", "model_id", "timeframe",
    ]
    archive_df = archive_expanded[archive_cols].copy()
    archive_df["fcstdate"] = archive_df["fcstdate"].dt.strftime("%Y-%m-%d")
    archive_df["startdate"] = archive_df["startdate"].dt.strftime("%Y-%m-%d")

    archive_path = output_dir / "backtest_predictions_all_lags.csv"
    archive_df.to_csv(archive_path, index=False)
    print(f"  [{_ts()}] Saved {len(archive_df):,} archive rows to {archive_path}")

    # Save metadata
    metadata = {
        "model_id": model_id,
        "cluster_strategy": args.cluster_strategy,
        "n_timeframes": args.n_timeframes,
        "xgboost_params": {k: v for k, v in xgb_params.items()},
        **({"transfer_n_estimators": args.transfer_n_estimators,
            "transfer_min_rows": args.transfer_min_rows} if args.cluster_strategy == "transfer" else {}),
        "n_predictions": len(output_df),
        "n_dfus": int(output_df["dmdunit"].nunique()),
        "date_range": {
            "earliest": str(earliest_month.date()),
            "latest": str(latest_month.date()),
        },
        "timeframes": [
            {
                "label": tf["label"],
                "train_end": str(tf["train_end"].date()),
                "predict_start": str(tf["predict_start"].date()),
                "predict_end": str(tf["predict_end"].date()),
            }
            for tf in timeframes
        ],
    }

    # Compute accuracy summary
    at_exec = output_df.copy()
    at_exec["basefcst_pref"] = pd.to_numeric(at_exec["basefcst_pref"], errors="coerce")
    at_exec["tothist_dmd"] = pd.to_numeric(at_exec["tothist_dmd"], errors="coerce")
    at_exec = at_exec.dropna(subset=["basefcst_pref", "tothist_dmd"])

    if len(at_exec) > 0 and at_exec["tothist_dmd"].abs().sum() > 0:
        total_f = at_exec["basefcst_pref"].sum()
        total_a = at_exec["tothist_dmd"].sum()
        abs_error = (at_exec["basefcst_pref"] - at_exec["tothist_dmd"]).abs().sum()
        wape = 100 * abs_error / abs(total_a) if abs(total_a) > 0 else None
        bias = (total_f / total_a) - 1 if abs(total_a) > 0 else None
        accuracy = 100 - wape if wape is not None else None
        metadata["accuracy_at_execution_lag"] = {
            "n_rows": len(at_exec),
            "wape": round(float(wape), 2) if wape else None,
            "bias": round(float(bias), 4) if bias else None,
            "accuracy_pct": round(float(accuracy), 2) if accuracy else None,
        }
        print(f"\n  Accuracy at execution lag ({len(at_exec):,} rows):")
        print(f"    WAPE: {wape:.2f}%")
        print(f"    Bias: {bias:.4f}")
        print(f"    Accuracy: {accuracy:.2f}%")

    meta_path = output_dir / "backtest_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  [{_ts()}] Saved metadata to {meta_path}")

    # Feature importance (from last timeframe's model)
    if args.cluster_strategy == "global" and "model" in dir():
        try:
            importance = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            imp_path = output_dir / "feature_importance.csv"
            importance.to_csv(imp_path, index=False)
            print(f"  [{_ts()}] Saved feature importance to {imp_path}")
        except Exception:
            pass

    # ── Step 7: MLflow logging ──────────────────────────────────────────────
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5003")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("demand_backtest")

        with mlflow.start_run():
            mlflow.set_tag("model_type", "xgboost_backtest")
            mlflow.set_tag("cluster_strategy", args.cluster_strategy)
            mlflow.set_tag("model_id", model_id)

            _mlflow_params = {
                "n_timeframes": args.n_timeframes,
                "cluster_strategy": args.cluster_strategy,
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
                "min_child_weight": args.min_child_weight,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
            }
            if args.cluster_strategy == "transfer":
                _mlflow_params["transfer_n_estimators"] = args.transfer_n_estimators
                _mlflow_params["transfer_min_rows"] = args.transfer_min_rows
            mlflow.log_params(_mlflow_params)

            mlflow.log_metrics({
                "n_predictions": len(output_df),
                "n_dfus": int(output_df["dmdunit"].nunique()),
            })
            if "accuracy_at_execution_lag" in metadata:
                acc = metadata["accuracy_at_execution_lag"]
                if acc.get("wape"):
                    mlflow.log_metric("wape", acc["wape"])
                if acc.get("accuracy_pct"):
                    mlflow.log_metric("accuracy_pct", acc["accuracy_pct"])
                if acc.get("bias"):
                    mlflow.log_metric("bias", acc["bias"])

            mlflow.log_artifact(str(output_path))
            mlflow.log_artifact(str(archive_path))
            mlflow.log_artifact(str(meta_path))

            print(f"\n  [{_ts()}] Logged to MLflow: {mlflow.get_artifact_uri()}")
    except Exception as e:
        print(f"\n  [{_ts()}] MLflow logging skipped: {e}")

    elapsed = time.time() - t_start
    print(f"\n[{_ts()}] Backtest complete in {elapsed:.0f}s ({elapsed / 60:.1f}m)")


if __name__ == "__main__":
    main()
