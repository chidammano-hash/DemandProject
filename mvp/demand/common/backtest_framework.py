"""Shared backtest orchestration framework.

Provides common logic for all backtest scripts:
- Timeframe generation
- Data loading from Postgres
- Execution-lag assignment and forecast_ck construction
- All-lag expansion for archive tables
- Output saving (CSV + metadata JSON)
- Accuracy computation
- MLflow logging

Model-specific scripts implement only the training/prediction functions.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import psycopg

from common.constants import (
    ARCHIVE_COLS,
    CAT_FEATURES,
    LAG_RANGE,
    MAX_ARCHIVE_LAG,
    MIN_TRAINING_MONTHS,
    OUTPUT_COLS,
)
from common.db import get_db_params
from common.metrics import compute_accuracy_metrics
from common.mlflow_utils import log_backtest_run


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ── Timeframe generation ─────────────────────────────────────────────────────


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


# ── Data loading ─────────────────────────────────────────────────────────────


def load_backtest_data(
    db: dict[str, Any],
    include_item_attrs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load sales, DFU attributes, and item attributes from Postgres.

    Returns (sales_df, dfu_attrs, item_attrs).
    """
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

        if include_item_attrs:
            item_attrs = pd.read_sql("""
                SELECT DISTINCT i.item_no AS dmdunit,
                       i.case_weight, i.item_proof, i.bpc
                FROM dim_item i
                INNER JOIN dim_dfu d ON i.item_no = d.dmdunit
            """, conn)
        else:
            item_attrs = pd.DataFrame()

    sales_df["startdate"] = pd.to_datetime(sales_df["startdate"])
    sales_df["qty"] = pd.to_numeric(sales_df["qty"], errors="coerce").fillna(0)

    # Only keep DFUs that have sales
    dfus_with_sales = set(sales_df["dfu_ck"].unique())
    dfu_attrs = dfu_attrs[dfu_attrs["dfu_ck"].isin(dfus_with_sales)].copy()

    print(f"  [{_ts()}] Sales: {len(sales_df):,} rows, {len(dfus_with_sales):,} DFUs ({time.time() - t1:.1f}s)")
    print(f"  [{_ts()}] DFU attrs: {len(dfu_attrs):,}, Item attrs: {len(item_attrs):,}")

    return sales_df, dfu_attrs, item_attrs


# ── Execution-lag assignment ─────────────────────────────────────────────────


def assign_execution_lag(
    pred_df: pd.DataFrame,
    execution_lag_map: dict[str, int],
) -> pd.DataFrame:
    """Assign each prediction its DFU's execution lag and compute fcstdate.

    Only one row per prediction — at the DFU's execution lag.
    fcstdate = startdate - execution_lag months.
    """
    t0 = time.time()
    result = pred_df.copy()

    # Execution lag from DFU dimension
    result["execution_lag"] = result["dfu_ck"].map(execution_lag_map).fillna(0).astype(int)
    result["lag"] = result["execution_lag"]
    result["fcstdate"] = result.apply(
        lambda r: r["startdate"] - pd.DateOffset(months=int(r["lag"])), axis=1
    )

    # Build forecast_ck (vectorized string concat)
    result["forecast_ck"] = (
        result["dmdunit"].astype(str) + "_"
        + result["dmdgroup"].astype(str) + "_"
        + result["loc"].astype(str) + "_"
        + result["fcstdate"].dt.strftime("%Y-%m-%d") + "_"
        + result["startdate"].dt.strftime("%Y-%m-%d")
    )

    print(f"  [{_ts()}] Execution-lag assignment done ({time.time() - t0:.1f}s)")
    return result


# ── All-lag expansion (archive) ──────────────────────────────────────────────


def expand_to_all_lags(
    pred_df: pd.DataFrame,
    max_lag: int,
    execution_lag_map: dict[str, int],
) -> pd.DataFrame:
    """Expand each prediction to lag 0 .. max_lag rows for the archive table.

    fcstdate = startdate - lag months.
    """
    t0 = time.time()
    dfs = []
    for lag in range(max_lag + 1):
        df = pred_df.copy()
        df["lag"] = lag
        df["fcstdate"] = df["startdate"] - pd.DateOffset(months=lag)
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    result["execution_lag"] = result["dfu_ck"].map(execution_lag_map).fillna(0).astype(int)

    # Build forecast_ck
    result["forecast_ck"] = (
        result["dmdunit"].astype(str) + "_"
        + result["dmdgroup"].astype(str) + "_"
        + result["loc"].astype(str) + "_"
        + result["fcstdate"].dt.strftime("%Y-%m-%d") + "_"
        + result["startdate"].dt.strftime("%Y-%m-%d")
    )

    print(f"  [{_ts()}] All-lag expansion (0-{max_lag}) done: {len(result):,} rows ({time.time() - t0:.1f}s)")
    return result


# ── Post-processing: combine, dedup, attach actuals ─────────────────────────


def postprocess_predictions(
    all_predictions: list[pd.DataFrame],
    sales_df: pd.DataFrame,
    exec_lag_map: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine timeframe predictions, assign execution lag, dedup, attach actuals.

    Returns (output_df, archive_df, combined_raw).
    """
    combined = pd.concat(all_predictions, ignore_index=True)
    print(f"  [{_ts()}] Total raw predictions: {len(combined):,}")

    # Ensure startdate is datetime
    combined["startdate"] = pd.to_datetime(combined["startdate"])

    # Assign execution lag and compute fcstdate (one row per prediction)
    print(f"  [{_ts()}] Assigning execution lag per DFU...")
    expanded = assign_execution_lag(combined, exec_lag_map)
    print(f"  [{_ts()}] Rows after execution-lag assignment: {len(expanded):,}")

    # Deduplicate: for same (forecast_ck, model_id), keep latest timeframe
    expanded = expanded.sort_values("timeframe_idx")
    expanded = expanded.drop_duplicates(subset=["forecast_ck", "model_id"], keep="last")
    print(f"  [{_ts()}] After dedup: {len(expanded):,}")

    # Attach actuals via merge (vectorized — not row-by-row apply)
    print(f"  [{_ts()}] Attaching actuals...")
    t1 = time.time()
    actuals = sales_df.drop_duplicates(subset=["dfu_ck", "startdate"])[["dfu_ck", "startdate", "qty"]].rename(
        columns={"qty": "tothist_dmd"}
    )
    expanded = expanded.merge(actuals, on=["dfu_ck", "startdate"], how="left")
    print(f"  [{_ts()}] Actuals attached ({time.time() - t1:.1f}s)")

    # ── All-lags archive ──────────────────────────────────────────────────
    print(f"  [{_ts()}] Generating all-lags archive (lag 0-{MAX_ARCHIVE_LAG})...")
    archive_expanded = expand_to_all_lags(combined, MAX_ARCHIVE_LAG, exec_lag_map)

    # Deduplicate
    archive_expanded = archive_expanded.sort_values("timeframe_idx")
    archive_expanded = archive_expanded.drop_duplicates(
        subset=["forecast_ck", "model_id", "lag"], keep="last"
    )
    print(f"  [{_ts()}] Archive after dedup: {len(archive_expanded):,}")

    # Attach actuals
    archive_expanded = archive_expanded.merge(actuals, on=["dfu_ck", "startdate"], how="left")

    return expanded, archive_expanded, combined


# ── Output saving ────────────────────────────────────────────────────────────


def save_backtest_output(
    output_df: pd.DataFrame,
    archive_df: pd.DataFrame,
    output_dir: Path,
    model_id: str,
    cluster_strategy: str,
    n_timeframes: int,
    model_params: dict[str, Any],
    model_params_key: str,
    timeframes: list[dict],
    earliest_month: pd.Timestamp,
    latest_month: pd.Timestamp,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path, dict]:
    """Save predictions CSV, archive CSV, and metadata JSON.

    Returns (output_path, archive_path, meta_path, metadata_dict).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select and order columns for fact_external_forecast_monthly
    out = output_df[OUTPUT_COLS].copy()
    out["fcstdate"] = out["fcstdate"].dt.strftime("%Y-%m-%d")
    out["startdate"] = out["startdate"].dt.strftime("%Y-%m-%d")

    output_path = output_dir / "backtest_predictions.csv"
    out.to_csv(output_path, index=False)
    print(f"  [{_ts()}] Saved {len(out):,} predictions to {output_path}")

    # Archive CSV
    arch = archive_df[ARCHIVE_COLS].copy()
    arch["fcstdate"] = arch["fcstdate"].dt.strftime("%Y-%m-%d")
    arch["startdate"] = arch["startdate"].dt.strftime("%Y-%m-%d")

    archive_path = output_dir / "backtest_predictions_all_lags.csv"
    arch.to_csv(archive_path, index=False)
    print(f"  [{_ts()}] Saved {len(arch):,} archive rows to {archive_path}")

    # Build metadata
    metadata = {
        "model_id": model_id,
        "cluster_strategy": cluster_strategy,
        "n_timeframes": n_timeframes,
        model_params_key: {k: v for k, v in model_params.items()},
        **(extra_metadata or {}),
        "n_predictions": len(out),
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
    acc = compute_accuracy_metrics(
        pd.to_numeric(out["basefcst_pref"], errors="coerce"),
        pd.to_numeric(out["tothist_dmd"], errors="coerce"),
    )
    if acc["wape"] is not None:
        metadata["accuracy_at_execution_lag"] = acc
        print(f"\n  Accuracy at execution lag ({acc['n_rows']:,} rows):")
        print(f"    WAPE: {acc['wape']:.2f}%")
        print(f"    Bias: {acc['bias']:.4f}")
        print(f"    Accuracy: {acc['accuracy_pct']:.2f}%")

    meta_path = output_dir / "backtest_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  [{_ts()}] Saved metadata to {meta_path}")

    return output_path, archive_path, meta_path, metadata


# ── Feature importance ───────────────────────────────────────────────────────


def save_feature_importance(
    model: Any,
    feature_cols: list[str],
    output_dir: Path,
    importance_attr: str = "feature_importances_",
) -> Path | None:
    """Save feature importance CSV from a trained model. Returns path or None."""
    try:
        if hasattr(model, importance_attr):
            importances = getattr(model, importance_attr)
        elif hasattr(model, "get_feature_importance"):
            importances = model.get_feature_importance()
        else:
            return None

        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        imp_path = output_dir / "feature_importance.csv"
        importance.to_csv(imp_path, index=False)
        print(f"  [{_ts()}] Saved feature importance to {imp_path}")
        return imp_path
    except Exception:
        return None


# ── Tree-model backtest runner ───────────────────────────────────────────────

# Type alias for train-and-predict functions
TrainFn = Callable[
    [pd.DataFrame, pd.DataFrame, list[str], list[str], dict],
    tuple[pd.DataFrame, Any],
]


def run_tree_backtest(
    *,
    model_id: str,
    cluster_strategy: str,
    n_timeframes: int,
    output_dir: Path,
    model_params: dict[str, Any],
    model_params_key: str,
    model_type_tag: str,
    train_fn_global: TrainFn,
    train_fn_per_cluster: TrainFn,
    train_fn_transfer: TrainFn,
    transfer_kwargs: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    cat_dtype: str = "category",
    min_training_months: int = MIN_TRAINING_MONTHS,
) -> None:
    """Run a complete tree-based backtest (LGBM, CatBoost, XGBoost).

    This is the main orchestrator that replaces the duplicated main() in each script.
    """
    from common.feature_engineering import (
        build_feature_matrix,
        get_feature_columns,
        mask_future_sales,
    )

    t_start = time.time()
    db = get_db_params()

    print(f"[{_ts()}] Backtest: strategy={cluster_strategy}, model_id={model_id}, "
          f"n_timeframes={n_timeframes}")

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print(f"\n[{_ts()}] Step 1: Loading data from Postgres...")
    sales_df, dfu_attrs, item_attrs = load_backtest_data(db)

    # Execution lag lookup
    exec_lag_map = dfu_attrs.set_index("dfu_ck")["execution_lag"].fillna(0).astype(int).to_dict()

    # ── Step 2: Generate timeframes ──────────────────────────────────────────
    latest_month = sales_df["startdate"].max()
    earliest_month = sales_df["startdate"].min()
    print(f"  [{_ts()}] Date range: {earliest_month.date()} → {latest_month.date()}")

    timeframes = generate_timeframes(earliest_month, latest_month, n_timeframes)
    print(f"\n[{_ts()}] Step 2: Generated {len(timeframes)} timeframes:")
    for tf in timeframes:
        print(f"  {tf['label']}: train [{tf['train_start'].date()} → {tf['train_end'].date()}], "
              f"predict [{tf['predict_start'].date()} → {tf['predict_end'].date()}]")

    all_months = sorted(sales_df["startdate"].unique())

    # ── Step 3: Build feature matrix ONCE ────────────────────────────────────
    print(f"\n[{_ts()}] Step 3: Building feature matrix (one-time)...")
    full_grid = build_feature_matrix(sales_df, dfu_attrs, item_attrs, all_months, cat_dtype=cat_dtype)
    feature_cols = get_feature_columns(full_grid)
    cat_cols = [c for c in CAT_FEATURES if c in feature_cols and c in full_grid.columns]
    print(f"  [{_ts()}] Features: {len(feature_cols)} columns, cat: {cat_cols}")

    # ── Step 4: Train & predict per timeframe ────────────────────────────────
    print(f"\n[{_ts()}] Step 4: Running {len(timeframes)} timeframe backtests...")
    all_predictions = []
    last_global_model = None

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
        if len(train_months) < min_training_months:
            print(f"  [{_ts()}] Insufficient training months ({len(train_months)}) — need {min_training_months} min — skipping")
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

        # Train & predict based on strategy
        if cluster_strategy == "global":
            preds, model = train_fn_global(
                train_data, predict_data, feature_cols, cat_cols, model_params
            )
            last_global_model = model
        elif cluster_strategy == "transfer":
            preds, models = train_fn_transfer(
                train_data, predict_data, feature_cols, cat_cols, model_params,
                **(transfer_kwargs or {}),
            )
        else:
            preds, models = train_fn_per_cluster(
                train_data, predict_data, feature_cols, cat_cols, model_params
            )

        preds["model_id"] = model_id
        preds["timeframe"] = label
        preds["timeframe_idx"] = tf["index"]
        all_predictions.append(preds)
        print(f"  [{_ts()}] Timeframe {label} complete: {len(preds):,} predictions ({time.time() - tf_start:.1f}s)")

    if not all_predictions:
        print(f"\n[{_ts()}] No predictions generated. Check data range and timeframe count.")
        sys.exit(1)

    # ── Step 5: Combine, assign execution lag, attach actuals ────────────────
    print(f"\n[{_ts()}] Step 5: Combining predictions...")
    expanded, archive_expanded, combined = postprocess_predictions(
        all_predictions, sales_df, exec_lag_map
    )

    # ── Step 6: Save output ──────────────────────────────────────────────────
    print(f"\n[{_ts()}] Step 6: Saving output...")
    output_path, archive_path, meta_path, metadata = save_backtest_output(
        output_df=expanded,
        archive_df=archive_expanded,
        output_dir=output_dir,
        model_id=model_id,
        cluster_strategy=cluster_strategy,
        n_timeframes=n_timeframes,
        model_params=model_params,
        model_params_key=model_params_key,
        timeframes=timeframes,
        earliest_month=earliest_month,
        latest_month=latest_month,
        extra_metadata=extra_metadata,
    )

    # Feature importance (from last timeframe's global model)
    if cluster_strategy == "global" and last_global_model is not None:
        save_feature_importance(last_global_model, feature_cols, output_dir)

    # ── Step 7: MLflow logging ───────────────────────────────────────────────
    mlflow_params = {
        "n_timeframes": n_timeframes,
        "cluster_strategy": cluster_strategy,
        **{k: v for k, v in model_params.items() if not callable(v)},
    }
    if transfer_kwargs:
        mlflow_params.update(transfer_kwargs)

    log_backtest_run(
        model_type=model_type_tag,
        model_id=model_id,
        cluster_strategy=cluster_strategy,
        hyperparams=mlflow_params,
        metrics={
            "n_predictions": len(expanded),
            "n_dfus": int(expanded["dmdunit"].nunique()),
        },
        metadata=metadata,
        artifact_paths=[str(output_path), str(archive_path), str(meta_path)],
    )

    elapsed = time.time() - t_start
    print(f"\n[{_ts()}] Backtest complete in {elapsed:.0f}s ({elapsed / 60:.1f}m)")
