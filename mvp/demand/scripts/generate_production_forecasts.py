"""
F1.1 — Production Forecast Generation Pipeline

Runs champion ML models over a forward horizon and writes predictions to
fact_production_forecast. This is the bridge between the backtesting engine
(which evaluates historical accuracy) and operational planning (which needs
future-period demand signals).

The key difference from backtesting:
  - Backtest: predicts historical months where actuals exist (for model evaluation)
  - Production: predicts future months where no actuals exist yet (for planning)

Usage:
    uv run python scripts/generate_production_forecasts.py
    uv run python scripts/generate_production_forecasts.py --horizon 6
    uv run python scripts/generate_production_forecasts.py --dfu 100320 1401-BULK
    uv run python scripts/generate_production_forecasts.py --dry-run

Algorithm:
    For each DFU in champion assignments:
        1. Load the cluster's .pkl model artifact
        2. Build an inference grid (feature matrix for T+1 through T+horizon)
        3. Generate predictions recursively: lag_1 for T+2 = model's T+1 prediction
        4. Write to fact_production_forecast with upsert on (plan_version, item_no, loc, forecast_month)
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import uuid
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params
from common.constants import CAT_FEATURES, LAG_RANGE, ROLLING_WINDOWS

import psycopg

CONFIG_PATH = ROOT / "config" / "production_forecast_config.yaml"


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_active_models(model_id: str, config: dict) -> dict[str, dict]:
    """Load pickled model artifacts for all clusters of a given model_id.

    Returns:
        {cluster_label: {"model": <model>, "feature_cols": [...]}}
    """
    base_path = config.get("model_registry", {}).get("base_path", "data/models")
    model_dir = ROOT / base_path / model_id

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. "
            f"Run 'make backtest-{model_id.replace('_cluster', '')}' to train and persist models."
        )

    models = {}
    for fname in sorted(model_dir.glob("cluster_*.pkl")):
        cluster_label_raw = fname.stem.replace("cluster_", "")
        # Cluster labels may be integers or strings (e.g. "0", "3")
        try:
            cluster_label = int(cluster_label_raw)
        except ValueError:
            cluster_label = cluster_label_raw

        with open(fname, "rb") as f:
            artifact = pickle.load(f)
        models[cluster_label] = artifact

    print(f"  [{_ts()}] Loaded {len(models)} {model_id} cluster models from {model_dir}/")
    return models


# ---------------------------------------------------------------------------
# Champion assignment query
# ---------------------------------------------------------------------------


def get_champion_assignments(conn, item_no: str | None = None, loc: str | None = None) -> pd.DataFrame:
    """Return the most recent champion model assignment per DFU.

    Returns `source_model_id` — the underlying algorithm (e.g. lgbm_cluster) whose
    artifacts are used for production inference.  Populated by champion selection
    via the source_model_id column (added in F1.1 via sql/041_add_source_model_id.sql).
    When NULL (champion rows from before this column was added), the caller falls
    back to `fallback_model_id` from config.

    Returns DataFrame with columns: item_no, loc, source_model_id, cluster_id, dmdgroup.
    """
    where_clauses = ["f.model_id = 'champion'"]
    params: list = []

    if item_no:
        where_clauses.append("f.dmdunit = %s")
        params.append(item_no)
    if loc:
        where_clauses.append("f.loc = %s")
        params.append(loc)

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT DISTINCT ON (f.dmdunit, f.loc)
            f.dmdunit                   AS item_no,
            f.loc,
            f.source_model_id,
            d.ml_cluster                AS cluster_id,
            d.dmdgroup
        FROM fact_external_forecast_monthly f
        JOIN dim_dfu d ON d.dmdunit = f.dmdunit AND d.loc = f.loc
        WHERE {where_sql}
        ORDER BY f.dmdunit, f.loc, f.startdate DESC
    """

    df = pd.read_sql(sql, conn, params=params)
    with_src = int(df["source_model_id"].notna().sum())
    print(f"  [{_ts()}] Champion assignments loaded: {len(df):,} DFUs "
          f"({with_src:,} with source_model_id)")
    return df


# ---------------------------------------------------------------------------
# Historical sales loading
# ---------------------------------------------------------------------------


def load_recent_sales(conn, item_no: str | None = None, loc: str | None = None,
                      lookback_months: int = 24) -> pd.DataFrame:
    """Load the last N months of sales for all DFUs (or a specific DFU).

    Returns DataFrame with columns: item_no, loc, startdate, qty.
    """
    cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(months=lookback_months)

    where_clauses = ["s.startdate >= %s", "s.type = 1"]
    params: list = [cutoff.date()]

    if item_no:
        where_clauses.append("s.dmdunit = %s")
        params.append(item_no)
    if loc:
        where_clauses.append("s.loc = %s")
        params.append(loc)

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT s.dmdunit AS item_no, s.loc, s.startdate, COALESCE(s.qty, 0) AS qty
        FROM fact_sales_monthly s
        WHERE {where_sql}
        ORDER BY s.dmdunit, s.loc, s.startdate
    """

    df = pd.read_sql(sql, conn, params=params)
    df["startdate"] = pd.to_datetime(df["startdate"])
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
    print(f"  [{_ts()}] Recent sales loaded: {len(df):,} rows, {df.groupby(['item_no','loc']).ngroups:,} DFUs")
    return df


# ---------------------------------------------------------------------------
# DFU attributes loading
# ---------------------------------------------------------------------------


def load_dfu_attrs(conn, item_no: str | None = None, loc: str | None = None) -> pd.DataFrame:
    """Load DFU attributes needed for feature construction."""
    where_clauses = []
    params: list = []

    if item_no:
        where_clauses.append("dmdunit = %s")
        params.append(item_no)
    if loc:
        where_clauses.append("loc = %s")
        params.append(loc)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT dmdunit AS item_no, dmdgroup, loc, ml_cluster,
               execution_lag, total_lt, brand, region, abc_vol
        FROM dim_dfu
        {where_sql}
    """

    df = pd.read_sql(sql, conn, params=params)
    print(f"  [{_ts()}] DFU attributes loaded: {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Inference grid construction
# ---------------------------------------------------------------------------


def build_inference_grid(
    item_no: str,
    loc: str,
    cluster_id: int | str,
    sales_history: pd.DataFrame,
    dfu_attrs: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame | None:
    """Build a feature matrix for recursive inference over the next `horizon` months.

    For T+1: lag_1 = last known actual
    For T+2: lag_1 = model's T+1 prediction (written back after each step)
    For T+N: lag_1 = model's T+(N-1) prediction

    Returns DataFrame with `horizon` rows ready for model.predict(), or None if
    insufficient history.
    """
    dfu_sales = sales_history[
        (sales_history["item_no"] == item_no) & (sales_history["loc"] == loc)
    ].sort_values("startdate").copy()

    max_lag = max(LAG_RANGE)
    if len(dfu_sales) < max_lag:
        return None  # insufficient history

    # Last closed month = most recent actual
    last_month = dfu_sales["startdate"].max()

    # Build a series of future months T+1 ... T+horizon
    future_months = [last_month + pd.DateOffset(months=h) for h in range(1, horizon + 1)]

    # Build rows for each future month
    rows = []
    qty_series = list(dfu_sales["qty"].values)  # chronological

    # DFU attributes for categorical features
    dfu_row = dfu_attrs[
        (dfu_attrs["item_no"] == item_no) & (dfu_attrs["loc"] == loc)
    ]
    attrs = dfu_row.iloc[0].to_dict() if len(dfu_row) > 0 else {}

    for h, fmonth in enumerate(future_months):
        # At step h=0 (T+1), qty_series contains all actuals
        # At step h=1+ (T+2+), qty_series[-1] is the model's T+h prediction
        n = len(qty_series)
        row: dict = {}

        # Lag features: lag_k = qty at (current position - k)
        for lag_n in LAG_RANGE:
            idx = n - lag_n
            row[f"qty_lag_{lag_n}"] = float(qty_series[idx]) if idx >= 0 else 0.0

        # Rolling features
        for w in ROLLING_WINDOWS:
            window_vals = qty_series[max(0, n - w):n]
            if window_vals:
                row[f"rolling_mean_{w}m"] = float(np.mean(window_vals))
                row[f"rolling_std_{w}m"] = float(np.std(window_vals)) if len(window_vals) > 1 else 0.0
            else:
                row[f"rolling_mean_{w}m"] = 0.0
                row[f"rolling_std_{w}m"] = 0.0

        # Calendar features
        month_num = fmonth.month
        row["month"] = month_num
        row["quarter"] = (month_num - 1) // 3 + 1
        row["month_sin"] = float(np.sin(2 * np.pi * month_num / 12))
        row["month_cos"] = float(np.cos(2 * np.pi * month_num / 12))

        # DFU attributes (categorical and numeric)
        for col in CAT_FEATURES:
            row[col] = attrs.get(col, "__unknown__")
        row["execution_lag"] = float(attrs.get("execution_lag", 0) or 0)
        row["total_lt"] = float(attrs.get("total_lt", 14) or 14)

        row["_forecast_month"] = fmonth
        row["_horizon"] = h + 1
        row["_lag_source"] = "actual" if h == 0 else "predicted"
        rows.append(row)

        # Placeholder — will be filled during recursive inference
        qty_series.append(0.0)

    if not rows:
        return None

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Recursive inference
# ---------------------------------------------------------------------------


def generate_forecast_recursive(
    model,
    feature_cols: list[str],
    grid: pd.DataFrame,
    horizon: int,
    item_no: str,
    loc: str,
    plan_version: str,
    run_id: str,
    model_id: str,
    cluster_id: int | str,
) -> list[dict]:
    """Run recursive inference for one DFU over the horizon.

    Predicts month-by-month, writing each prediction back into the grid as
    lag_1 for the next month (recursive write-back).

    Returns list of row dicts ready for DB insert.
    """
    rows = []
    predicted_values: list[float] = []

    # Identify feature columns available in grid (exclude metadata cols)
    meta_cols = {"_forecast_month", "_horizon", "_lag_source"}
    available_features = [c for c in feature_cols if c in grid.columns and c not in meta_cols]

    for h in range(horizon):
        row_data = grid.iloc[[h]].copy()

        # Write-back: update lag_1 with previous prediction for h >= 1
        if h > 0 and predicted_values:
            row_data["qty_lag_1"] = predicted_values[-1]
            # Update lag_2, lag_3 etc. from prior predictions
            for k in range(2, min(h + 2, max(LAG_RANGE) + 1)):
                lag_col = f"qty_lag_{k}"
                prev_idx = h - k
                if prev_idx >= 0 and lag_col in row_data.columns:
                    row_data[lag_col] = predicted_values[prev_idx]
            # Recompute rolling features from updated lag values
            for w in ROLLING_WINDOWS:
                lag_vals = []
                for k in range(1, w + 1):
                    lag_col = f"qty_lag_{k}"
                    if lag_col in row_data.columns:
                        lag_vals.append(float(row_data[lag_col].iloc[0]))
                if lag_vals:
                    row_data[f"rolling_mean_{w}m"] = np.mean(lag_vals)
                    row_data[f"rolling_std_{w}m"] = np.std(lag_vals) if len(lag_vals) > 1 else 0.0

        # Predict using only the features the model was trained on
        X = row_data[available_features].fillna(0)
        pred = float(model.predict(X)[0])
        pred = max(0.0, round(pred, 2))  # no negative forecasts

        predicted_values.append(pred)

        forecast_month = grid.iloc[h]["_forecast_month"]
        if hasattr(forecast_month, "date"):
            forecast_month_date = forecast_month.date().replace(day=1)
        else:
            forecast_month_date = forecast_month

        rows.append({
            "plan_version": plan_version,
            "item_no": item_no,
            "loc": loc,
            "forecast_month": forecast_month_date,
            "forecast_qty": pred,
            "forecast_qty_lower": None,
            "forecast_qty_upper": None,
            "model_id": model_id,
            "cluster_id": int(cluster_id) if cluster_id is not None else None,
            "horizon_months": h + 1,
            "is_recursive": h > 0,
            "lag_source": "actual" if h == 0 else "predicted",
            "run_id": run_id,
            "generated_at": datetime.utcnow(),
        })

    return rows


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------


def write_forecast(rows: list[dict], conn, dry_run: bool = False) -> int:
    """Upsert forecast rows into fact_production_forecast.

    Uses ON CONFLICT DO UPDATE so repeated runs for the same plan_version
    overwrite rather than error.
    """
    if not rows:
        return 0

    if dry_run:
        print(f"  [{_ts()}] [DRY RUN] Would insert {len(rows):,} rows")
        return len(rows)

    sql = """
        INSERT INTO fact_production_forecast
            (plan_version, item_no, loc, forecast_month, forecast_qty,
             forecast_qty_lower, forecast_qty_upper, model_id, cluster_id,
             horizon_months, is_recursive, lag_source, run_id, generated_at)
        VALUES
            (%(plan_version)s, %(item_no)s, %(loc)s, %(forecast_month)s, %(forecast_qty)s,
             %(forecast_qty_lower)s, %(forecast_qty_upper)s, %(model_id)s, %(cluster_id)s,
             %(horizon_months)s, %(is_recursive)s, %(lag_source)s, %(run_id)s, %(generated_at)s)
        ON CONFLICT (plan_version, item_no, loc, forecast_month)
        DO UPDATE SET
            forecast_qty        = EXCLUDED.forecast_qty,
            forecast_qty_lower  = EXCLUDED.forecast_qty_lower,
            forecast_qty_upper  = EXCLUDED.forecast_qty_upper,
            model_id            = EXCLUDED.model_id,
            cluster_id          = EXCLUDED.cluster_id,
            horizon_months      = EXCLUDED.horizon_months,
            is_recursive        = EXCLUDED.is_recursive,
            lag_source          = EXCLUDED.lag_source,
            generated_at        = EXCLUDED.generated_at
    """

    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Old version cleanup
# ---------------------------------------------------------------------------


def purge_old_versions(conn, keep_n: int, dry_run: bool = False) -> None:
    """Delete plan versions beyond the most recent keep_n."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT plan_version
            FROM fact_production_forecast
            GROUP BY plan_version
            ORDER BY MIN(generated_at) DESC
        """)
        versions = [r[0] for r in cur.fetchall()]

    to_delete = versions[keep_n:]
    if not to_delete:
        return

    for v in to_delete:
        if dry_run:
            print(f"  [{_ts()}] [DRY RUN] Would delete version {v}")
        else:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM fact_production_forecast WHERE plan_version = %s", [v])
            conn.commit()
            print(f"  [{_ts()}] Purged old plan version: {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate production forecasts for future months (F1.1)"
    )
    parser.add_argument("--horizon", type=int, default=None,
                        help="Months ahead to forecast (default from config)")
    parser.add_argument("--dfu", nargs=2, metavar=("ITEM", "LOC"),
                        help="Run for a single DFU only: --dfu 100320 1401-BULK")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing to DB")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model_id (default: champion assignment per DFU)")
    args = parser.parse_args()

    config = load_config()
    horizon = args.horizon or config["inference"]["horizon_months"]
    keep_n = config["plan_version"]["keep_last_n_versions"]
    fallback_model_id = config["model_selection"]["fallback_model_id"]

    plan_version = datetime.utcnow().strftime(config["plan_version"]["format"])
    run_id = str(uuid.uuid4())

    item_filter = args.dfu[0] if args.dfu else None
    loc_filter = args.dfu[1] if args.dfu else None

    print(f"[{_ts()}] Production Forecast Generation — F1.1")
    print(f"[{_ts()}] plan_version={plan_version}, horizon={horizon}, run_id={run_id[:8]}...")
    if args.dry_run:
        print(f"[{_ts()}] DRY RUN — no data will be written")

    t_start = time.time()
    db = get_db_params()

    with psycopg.connect(**db) as conn:
        # Load data
        print(f"\n[{_ts()}] Step 1: Loading data...")
        champion_df = get_champion_assignments(conn, item_filter, loc_filter)
        if len(champion_df) == 0:
            print(f"[{_ts()}] No champion assignments found. Run 'make champion-select' first.")
            return

        sales_df = load_recent_sales(conn, item_filter, loc_filter)
        dfu_attrs = load_dfu_attrs(conn, item_filter, loc_filter)

        # Determine which model_ids we need to load.
        # source_model_id = the underlying algorithm that won per DFU (e.g. lgbm_cluster).
        # Populated by champion selection after sql/041_add_source_model_id.sql is applied.
        # Always include fallback_model_id so every DFU has at least one model to use.
        if args.model_id:
            model_ids_needed = {args.model_id}
        else:
            src_ids = set(champion_df["source_model_id"].dropna().unique())
            model_ids_needed = src_ids | {fallback_model_id}

        # Load model artifacts
        print(f"\n[{_ts()}] Step 2: Loading model artifacts for: {model_ids_needed}")
        loaded_models: dict[str, dict] = {}
        for mid in model_ids_needed:
            try:
                loaded_models[mid] = load_active_models(mid, config)
            except FileNotFoundError as e:
                print(f"  [{_ts()}] Warning: {e}")

        if not loaded_models:
            print(f"[{_ts()}] No model artifacts found for: {model_ids_needed}")
            print(f"[{_ts()}] Run 'make backtest-lgbm' (or backtest-catboost / backtest-xgboost) "
                  f"to train and persist model weights, then re-run this script.")
            return

        # Generate forecasts DFU by DFU
        print(f"\n[{_ts()}] Step 3: Generating forecasts for {len(champion_df):,} DFUs...")
        all_rows: list[dict] = []
        skipped = 0

        for idx, champ in champion_df.iterrows():
            item_no = champ["item_no"]
            loc = champ["loc"]

            # Determine model_id: CLI override > source_model_id from champion row > fallback
            if args.model_id:
                model_id = args.model_id
            else:
                model_id = champ["source_model_id"] or fallback_model_id

            cluster_id = champ["cluster_id"]

            # Route to the correct model artifact
            cluster_models = loaded_models.get(model_id)
            if cluster_models is None:
                cluster_models = loaded_models.get(fallback_model_id)
            if cluster_models is None:
                skipped += 1
                continue

            # Try to find the cluster's model, fall back to cluster 0 or first available
            artifact = cluster_models.get(cluster_id)
            if artifact is None:
                # Try integer version of cluster_id
                try:
                    artifact = cluster_models.get(int(cluster_id))
                except (ValueError, TypeError):
                    pass
            if artifact is None:
                # Fall back to first available cluster model
                artifact = next(iter(cluster_models.values()), None)
            if artifact is None:
                skipped += 1
                continue

            model = artifact["model"]
            feature_cols = artifact["feature_cols"]

            # Build inference grid
            grid = build_inference_grid(
                item_no=item_no,
                loc=loc,
                cluster_id=cluster_id,
                sales_history=sales_df,
                dfu_attrs=dfu_attrs,
                horizon=horizon,
            )
            if grid is None:
                skipped += 1
                continue

            # Generate recursive forecast
            try:
                rows = generate_forecast_recursive(
                    model=model,
                    feature_cols=feature_cols,
                    grid=grid,
                    horizon=horizon,
                    item_no=item_no,
                    loc=loc,
                    plan_version=plan_version,
                    run_id=run_id,
                    model_id=model_id,
                    cluster_id=cluster_id,
                )
                all_rows.extend(rows)
            except Exception as exc:
                print(f"  [{_ts()}] Warning: DFU {item_no}/{loc} failed: {exc}")
                skipped += 1
                continue

            if (idx + 1) % 1000 == 0:
                print(f"  [{_ts()}] Processed {idx + 1:,} / {len(champion_df):,} DFUs "
                      f"({len(all_rows):,} forecast rows)")

        print(f"\n[{_ts()}] Step 3 complete: {len(all_rows):,} rows, {skipped:,} skipped")

        # Write to DB
        print(f"\n[{_ts()}] Step 4: Writing to fact_production_forecast...")
        written = write_forecast(all_rows, conn, dry_run=args.dry_run)
        print(f"  [{_ts()}] Written: {written:,} rows")

        # Purge old plan versions
        if not args.dry_run:
            purge_old_versions(conn, keep_n=keep_n, dry_run=args.dry_run)

    elapsed = time.time() - t_start
    print(f"\n[{_ts()}] Production forecast complete in {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"[{_ts()}] plan_version={plan_version}, rows={written:,}, skipped={skipped:,}")


if __name__ == "__main__":
    main()
