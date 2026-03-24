"""SHAP feature importance API endpoints (Feature 42).

Reads SHAP CSV outputs written by run_tree_backtest() + save_shap_outputs()
from data/backtest/<model_id>/shap/. No database queries — all data is
served directly from the filesystem CSVs.

Also provides on-demand per-DFU SHAP computation endpoint that loads
persisted pkl model artifacts and runs SHAP for a specific item+location pair.

Endpoints:
    GET /forecast/shap/models
    GET /forecast/shap/{model_id}/summary
    GET /forecast/shap/{model_id}/timeframes
    GET /forecast/shap/{model_id}/timeframe/{idx}
    GET /forecast/shap/{model_id}/dfu
"""
from __future__ import annotations

import os
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.core import get_conn
from common.constants import CAT_FEATURES, LAG_RANGE, ROLLING_WINDOWS

router = APIRouter(tags=["shap"])

# Root of the model-scoped backtest output directories
_BACKTEST_DATA_DIR = Path(os.environ.get("BACKTEST_DATA_DIR", "data/backtest"))

# Root of persisted model artifacts (written by generate_production_forecasts.py)
_MODELS_DIR = Path(os.environ.get("MODEL_REGISTRY_DIR", "data/models"))


# ---------------------------------------------------------------------------
# Filter → cluster resolution helper
# ---------------------------------------------------------------------------


def _resolve_filter_clusters(
    item: str | None = None,
    location: str | None = None,
    brand: str | None = None,
    category: str | None = None,
    market: str | None = None,
) -> list[str] | None:
    """Resolve global filter params to a list of matching cluster labels.

    Returns None if no filters are active (meaning use all data).
    Returns an empty list if filters are active but no clusters match.
    """
    if not any([item, location, brand, category, market]):
        return None

    conditions: list[str] = []
    params: list[str] = []
    need_item_join = False
    need_loc_join = False

    if item:
        items = [i.strip() for i in item.split(",")]
        conditions.append(f"d.item_id IN ({','.join(['%s'] * len(items))})")
        params.extend(items)
    if location:
        locs = [loc.strip() for loc in location.split(",")]
        conditions.append(f"d.loc IN ({','.join(['%s'] * len(locs))})")
        params.extend(locs)
    if brand:
        brands = [b.strip() for b in brand.split(",")]
        conditions.append(f"d.brand IN ({','.join(['%s'] * len(brands))})")
        params.extend(brands)
    if category:
        cats = [c.strip() for c in category.split(",")]
        conditions.append(f"i.category IN ({','.join(['%s'] * len(cats))})")
        params.extend(cats)
        need_item_join = True
    if market:
        markets = [m.strip() for m in market.split(",")]
        conditions.append(f"l.market IN ({','.join(['%s'] * len(markets))})")
        params.extend(markets)
        need_loc_join = True

    sql = "SELECT DISTINCT d.ml_cluster::TEXT FROM dim_sku d"
    if need_item_join:
        sql += " LEFT JOIN dim_item i ON i.item_id = d.item_id"
    if need_loc_join:
        sql += " LEFT JOIN dim_location l ON l.loc = d.loc"
    sql += f" WHERE {' AND '.join(conditions)} AND d.ml_cluster IS NOT NULL"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [str(r[0]) for r in rows] if rows else []


def _aggregate_cluster_shap(df: pd.DataFrame, clusters: list[str]) -> pd.DataFrame:
    """Filter a per-timeframe SHAP DataFrame to specific clusters and re-aggregate.

    If no cluster column or no matching rows, returns the pooled ("all") rows unchanged.
    """
    if "cluster" not in df.columns:
        return df

    filtered = df[df["cluster"].isin(clusters)]
    if filtered.empty:
        # Fall back to pooled "all" rows
        pooled = df[df["cluster"] == "all"]
        return pooled if not pooled.empty else df

    # Average mean_abs_shap across matching clusters per feature
    agg = (
        filtered.groupby("feature", sort=False)
        .agg(mean_abs_shap=("mean_abs_shap", "mean"))
        .reset_index()
    )
    agg = agg.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    agg["rank"] = range(1, len(agg) + 1)
    agg["selected"] = True  # re-derive: mark top features as selected

    # Preserve timeframe / cutoff_date from the original data
    for col in ("timeframe", "cutoff_date"):
        if col in filtered.columns:
            agg[col] = filtered[col].iloc[0]
    agg["cluster"] = "filtered"
    return agg


def _shap_dir(model_id: str) -> Path:
    return _BACKTEST_DATA_DIR / model_id / "shap"


def _summary_csv(model_id: str) -> Path:
    return _shap_dir(model_id) / "shap_summary.csv"


def _timeframe_csv(model_id: str, idx: int) -> Path:
    return _shap_dir(model_id) / f"shap_timeframe_{idx:02d}.csv"


# ---------------------------------------------------------------------------
# GET /forecast/shap/models
# ---------------------------------------------------------------------------


@router.get("/forecast/shap/models")
async def shap_models() -> dict:
    """List model IDs that have SHAP outputs available."""
    if not _BACKTEST_DATA_DIR.exists():
        return {"models": []}
    models = [
        d.name
        for d in sorted(_BACKTEST_DATA_DIR.iterdir())
        if d.is_dir() and _summary_csv(d.name).exists()
    ]
    return {"models": models}


# ---------------------------------------------------------------------------
# GET /forecast/shap/{model_id}/summary
# ---------------------------------------------------------------------------


@router.get("/forecast/shap/{model_id}/summary")
async def shap_summary(
    model_id: str,
    top_n: int = Query(default=15, ge=1, le=200),
    item: str | None = Query(default=None),
    location: str | None = Query(default=None),
    brand: str | None = Query(default=None),
    category: str | None = Query(default=None),
    market: str | None = Query(default=None),
) -> dict:
    """Cross-timeframe SHAP importance summary for a model.

    Returns features sorted by mean_abs_shap_across_timeframes descending.
    When global filter params are provided, resolves matching DFU clusters
    and re-aggregates SHAP from per-cluster data in timeframe CSVs.
    """
    filter_clusters = _resolve_filter_clusters(item, location, brand, category, market)

    # When filters are active and clusters resolved, re-compute summary from
    # per-timeframe CSVs filtered to matching clusters.
    if filter_clusters is not None:
        shap_d = _shap_dir(model_id)
        if not shap_d.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No SHAP outputs found for model '{model_id}'.",
            )
        timeframe_files = sorted(shap_d.glob("shap_timeframe_*.csv"))
        if not timeframe_files:
            raise HTTPException(
                status_code=404,
                detail=f"No SHAP timeframe data found for model '{model_id}'.",
            )
        # Re-aggregate per-timeframe data filtered by matching clusters
        filtered_dfs: list[pd.DataFrame] = []
        for f in timeframe_files:
            tf_df = pd.read_csv(f)
            agg_df = _aggregate_cluster_shap(tf_df, filter_clusters)
            if not agg_df.empty:
                filtered_dfs.append(agg_df)

        if not filtered_dfs:
            return {"model_id": model_id, "total_features": 0, "features": []}

        # Build cross-timeframe summary from filtered data
        combined = pd.concat(filtered_dfs, ignore_index=True)
        summary = (
            combined.groupby("feature", sort=False)
            .agg(
                mean_abs_shap_across_timeframes=("mean_abs_shap", "mean"),
                mean_rank=("rank", "mean"),
                selected_count=("selected", "sum"),
            )
            .reset_index()
        )
        summary["n_timeframes"] = len(filtered_dfs)
        summary = summary.sort_values("mean_abs_shap_across_timeframes", ascending=False).reset_index(drop=True)
        features = summary.head(top_n).to_dict(orient="records")
        return {
            "model_id": model_id,
            "total_features": len(summary),
            "features": features,
        }

    # No filters — use pre-computed summary CSV
    csv_path = _summary_csv(model_id)
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No SHAP summary found for model '{model_id}'. "
                   f"Run backtest with --shap-select first.",
        )
    df = pd.read_csv(csv_path)
    features = df.head(top_n).to_dict(orient="records")
    return {
        "model_id": model_id,
        "total_features": len(df),
        "features": features,
    }


# ---------------------------------------------------------------------------
# GET /forecast/shap/{model_id}/timeframes
# ---------------------------------------------------------------------------


@router.get("/forecast/shap/{model_id}/timeframes")
async def shap_timeframes(model_id: str) -> dict:
    """List all timeframes for which SHAP outputs exist for a model."""
    shap_d = _shap_dir(model_id)
    if not shap_d.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No SHAP outputs found for model '{model_id}'.",
        )
    timeframe_files = sorted(shap_d.glob("shap_timeframe_*.csv"))
    timeframes = []
    for f in timeframe_files:
        try:
            df = pd.read_csv(f, usecols=["timeframe", "cutoff_date"])
            idx = int(df["timeframe"].iloc[0])
            cutoff = df["cutoff_date"].iloc[0]
            label = chr(ord("A") + idx)
            timeframes.append({
                "index": idx,
                "label": label,
                "cutoff_date": str(cutoff),
            })
        except Exception:
            continue
    return {"model_id": model_id, "timeframes": timeframes}


# ---------------------------------------------------------------------------
# GET /forecast/shap/{model_id}/timeframe/{idx}
# ---------------------------------------------------------------------------


@router.get("/forecast/shap/{model_id}/timeframe/{idx}")
async def shap_timeframe_detail(
    model_id: str,
    idx: int,
    top_n: int = Query(default=15, ge=1, le=200),
    cluster: str = Query(default="all", max_length=60),
    item: str | None = Query(default=None),
    location: str | None = Query(default=None),
    brand: str | None = Query(default=None),
    category: str | None = Query(default=None),
    market: str | None = Query(default=None),
) -> dict:
    """Per-timeframe SHAP feature importance detail.

    Returns features sorted by mean_abs_shap descending (rank 1 = most important).
    Use cluster="all" for pooled (default), or a specific cluster label for per-cluster SHAP.
    When global filter params are provided, resolves matching DFU clusters
    and re-aggregates SHAP from per-cluster data.
    """
    csv_path = _timeframe_csv(model_id, idx)
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No SHAP data for model '{model_id}' timeframe {idx}.",
        )
    df = pd.read_csv(csv_path)

    # Resolve global filters to clusters (takes precedence over cluster param)
    filter_clusters = _resolve_filter_clusters(item, location, brand, category, market)
    if filter_clusters is not None and "cluster" in df.columns:
        df = _aggregate_cluster_shap(df, filter_clusters)
    elif "cluster" in df.columns:
        # Manual cluster selection (existing behavior)
        filtered = df[df["cluster"] == cluster]
        if filtered.empty and cluster != "all":
            raise HTTPException(
                status_code=404,
                detail=f"No SHAP data for cluster '{cluster}' in model '{model_id}' timeframe {idx}.",
            )
        if not filtered.empty:
            df = filtered

    df = df.sort_values("rank").reset_index(drop=True)
    label = chr(ord("A") + idx) if 0 <= idx <= 25 else str(idx)
    cutoff_date = str(df["cutoff_date"].iloc[0]) if not df.empty else ""
    features = df.head(top_n).to_dict(orient="records")

    # List available clusters from the full CSV
    full_df = pd.read_csv(csv_path)
    available_clusters = sorted(full_df["cluster"].unique().tolist()) if "cluster" in full_df.columns else ["all"]

    return {
        "model_id": model_id,
        "timeframe_idx": idx,
        "label": label,
        "cutoff_date": cutoff_date,
        "total_features": len(df),
        "cluster": cluster,
        "available_clusters": available_clusters,
        "features": features,
    }


@router.get("/forecast/shap/{model_id}/clusters")
async def shap_clusters(model_id: str) -> dict:
    """List available cluster labels for SHAP data of a model.

    Scans all timeframe CSVs for cluster column values.
    Returns ["all"] if no per-cluster data exists.
    """
    shap_d = _shap_dir(model_id)
    if not shap_d.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No SHAP outputs found for model '{model_id}'.",
        )
    clusters_set: set[str] = set()
    for f in shap_d.glob("shap_timeframe_*.csv"):
        try:
            df = pd.read_csv(f, usecols=["cluster"])
            clusters_set.update(df["cluster"].unique().tolist())
        except (ValueError, KeyError):
            clusters_set.add("all")
    if not clusters_set:
        clusters_set.add("all")
    return {"model_id": model_id, "clusters": sorted(clusters_set)}


# ---------------------------------------------------------------------------
# GET /forecast/shap/{model_id}/dfu  — per-DFU on-demand SHAP
# ---------------------------------------------------------------------------


def _build_cat_encoders_from_distinct(conn) -> dict[str, dict]:
    """Build int label encoders from sorted unique values in dim_sku.

    Replicates build_cat_encoders() in generate_production_forecasts.py so
    inference uses the same encoding that was applied during training.

    NOTE: Training used pandas Categorical codes from the training grid, which
    contains only DFUs with sales. Inference builds from ALL dim_sku rows.
    If dim_sku has brands/regions with no historical sales, their insertion into
    the sorted list shifts codes for other values. The correct long-term fix is to
    persist encoders in the pkl artifact. For now, we match generate_production_forecasts.py
    which also uses all dim_sku rows — so SHAP and production inference are consistent.
    """
    with conn.cursor() as cur:
        # Query each column independently to get ALL unique values per column
        # (not just combinations) — same as build_cat_encoders() in generate_production_forecasts.py
        cur.execute(
            "SELECT ml_cluster, region, brand, abc_vol FROM dim_sku"
        )
        rows = cur.fetchall()

    if not rows:
        return {}

    df = pd.DataFrame(rows, columns=["ml_cluster", "region", "brand", "abc_vol"])
    encoders: dict[str, dict] = {}
    for col in CAT_FEATURES:
        if col in df.columns:
            cats = sorted(df[col].fillna("__unknown__").astype(str).unique())
            encoders[col] = {v: i for i, v in enumerate(cats)}
    return encoders


def _detect_model_framework(model, model_id: str) -> str:
    """Detect underlying ML framework from model object type.

    Uses the object's module name so 'champion', 'ceiling', and other
    derived model_ids are handled correctly (not just string prefix checks).

    Returns: 'catboost', 'lgbm', or 'xgboost'
    """
    module = type(model).__module__.split(".")[0].lower()
    if module == "catboost" or model_id.startswith("catboost"):
        return "catboost"
    if module == "lightgbm" or model_id.startswith("lgbm"):
        return "lgbm"
    # xgboost, sklearn wrappers, and anything else
    return "xgboost"


def _extract_model_feature_names(model, model_id: str) -> list[str] | None:
    """Extract the actual feature names the model was trained on.

    Returns None if the framework doesn't expose feature names or if the
    returned value is not a valid list of strings.
    """
    framework = _detect_model_framework(model, model_id)
    try:
        names = None
        if framework == "catboost":
            names = getattr(model, "feature_names_", None)
        elif framework == "lgbm":
            names = getattr(model, "feature_name_", None)
            if names is None and hasattr(model, "booster_"):
                booster = getattr(model, "booster_", None)
                if booster is not None and hasattr(booster, "feature_name"):
                    names = booster.feature_name()
        else:
            # XGBoost: get_booster().feature_names
            booster = model.get_booster() if hasattr(model, "get_booster") else model
            names = getattr(booster, "feature_names", None)
        # Validate: must be a real list of strings (not a MagicMock or other proxy)
        if names is not None and isinstance(names, (list, tuple)) and len(names) > 0 and isinstance(names[0], str):
            return list(names)
        return None
    except Exception:
        return None


def _compute_shap_full(model, X: pd.DataFrame, model_id: str, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Compute SIGNED SHAP values + per-row base values.

    Returns:
        shap_vals: shape (n_rows, n_features) — signed per-feature contributions
        base_vals: shape (n_rows,) — expected value (base prediction) per row
    """
    framework = _detect_model_framework(model, model_id)

    if framework == "catboost":
        import catboost as cb
        # Use column NAMES (not integer indices) so Pool is robust to column reordering
        # caused by SHAP feature selection. Exclude ml_cluster (trained as numeric,
        # not categorical; constant within cluster so irrelevant for SHAP).
        cat_cols = [c for c in CAT_FEATURES if c in X.columns and c != "ml_cluster"]
        pool = cb.Pool(X, cat_features=cat_cols)
        full = model.get_feature_importance(data=pool, type="ShapValues")
        return full[:, :-1], full[:, -1]
    elif framework == "lgbm":
        # Pass as numpy array to bypass LightGBM's categorical_feature dtype validation
        # (which triggers "train and valid dataset categorical_feature do not match"
        # when the input has int64 cols that were pd.Categorical during training).
        # The integer-encoded values match training codes so splits are still correct.
        full = model.predict(X.to_numpy(), pred_contrib=True)
        return full[:, :-1], full[:, -1]
    else:
        # XGBoost: use native pred_contribs to bypass shap library issues.
        # XGBRegressor has get_booster(); native Booster already has predict().
        # pred_contribs=True returns (n_samples, n_features + 1) where the
        # last column is the per-row base value (same convention as LGBM).
        import xgboost as xgb
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        dmatrix = xgb.DMatrix(X)
        full = booster.predict(dmatrix, pred_contribs=True)
        return full[:, :-1], full[:, -1]


@router.get("/forecast/shap/{model_id}/dfu")
async def shap_dfu(
    model_id: str,
    item_id: str = Query(..., description="Item number (item_id)"),
    loc: str = Query(..., description="Location code"),
    top_n: int = Query(default=10, ge=1, le=30),
    lookback_months: int = Query(default=48, ge=12, le=60),
) -> dict:
    """Compute per-DFU per-month SHAP feature contributions on demand.

    Loads the persisted pkl model for this DFU's cluster (requires F1.1
    make forecast-generate to have been run), reconstructs the feature matrix
    from historical sales + future production forecast data, and computes
    signed SHAP values for each month.

    Falls back to 404 if model artifacts are not available (F1.1 not run).
    """
    # Validate model_id to prevent path traversal attacks
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", model_id):
        raise HTTPException(status_code=400, detail="Invalid model_id")

    # -- Step 1: check model dir exists --
    model_dir = _MODELS_DIR / model_id
    if not model_dir.exists() or not any(model_dir.glob("cluster_*.pkl")):
        raise HTTPException(
            status_code=404,
            detail=(
                f"No model artifacts found for '{model_id}'. "
                f"Run 'make forecast-generate' to persist model weights."
            ),
        )

    with get_conn() as conn:
        # -- Step 2: look up DFU cluster --
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.ml_cluster, d.execution_lag, d.total_lt,
                       d.brand, d.region, d.abc_vol,
                       d.customer_group,
                       i.bpc, i.item_proof, i.case_weight
                FROM dim_sku d
                LEFT JOIN dim_item i ON i.item_id = d.item_id
                WHERE d.item_id = %s AND d.loc = %s
                ORDER BY d.customer_group ASC
                LIMIT 1
                """,
                (item_id, loc),
            )
            dfu_row = cur.fetchone()

        if dfu_row is None:
            raise HTTPException(
                status_code=404,
                detail=f"DFU not found: item_id={item_id!r}, loc={loc!r}",
            )

        ml_cluster, execution_lag, total_lt, brand, region, abc_vol, customer_group, bpc, item_proof, case_weight = dfu_row

        # -- Step 3: load pkl for this cluster --
        cluster_str = str(ml_cluster) if ml_cluster is not None else "__unknown__"
        pkl_path = model_dir / f"cluster_{cluster_str}.pkl"
        if not pkl_path.exists():
            # Try integer form
            try:
                int_cluster = int(cluster_str)
                pkl_path = model_dir / f"cluster_{int_cluster}.pkl"
            except (ValueError, TypeError):
                pass
        if not pkl_path.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No pkl artifact for cluster '{cluster_str}' in model '{model_id}'. "
                    f"Re-run 'make forecast-generate' to rebuild model registry."
                ),
            )

        with open(pkl_path, "rb") as f:
            artifact = pickle.load(f)

        model = artifact["model"]
        feature_cols: list[str] = artifact["feature_cols"]
        # Per-cluster models exclude ml_cluster from features (mirrors train_fn)
        effective_feature_cols = [c for c in feature_cols if c != "ml_cluster"]

        # If SHAP feature selection (Feature 42) was active, the model was retrained
        # on a subset of features but the pkl's feature_cols may still list the full
        # pre-selection set.  Extract the model's actual trained feature names to avoid
        # a feature_names mismatch error at SHAP computation time.
        model_feature_names = _extract_model_feature_names(model, model_id)
        if model_feature_names is not None:
            model_feature_set = set(model_feature_names)
            if model_feature_set != set(effective_feature_cols):
                # Model was trained on fewer features; use the model's own list
                effective_feature_cols = [c for c in model_feature_names if c != "ml_cluster"]

        # -- Step 4: load historical sales --
        # Filter by customer_group to match backtest training data exactly.
        # The backtest (backtest_framework.py) joins dim_sku ON item_id + customer_group + loc,
        # so training used only the DFU's specific customer group — NOT a sum of all groups.
        # Summing across all customer groups would give 2-3× larger qty, making all lag
        # features different from training and causing SHAP values to be "way off".
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT startdate, SUM(COALESCE(qty, 0)) AS qty
                FROM fact_sales_monthly
                WHERE item_id = %s AND customer_group = %s AND loc = %s
                GROUP BY startdate
                ORDER BY startdate DESC
                LIMIT %s
                """,
                (item_id, customer_group, loc, lookback_months),
            )
            sales_rows = list(reversed(cur.fetchall()))

        # -- Step 5: build categorical encoders --
        cat_encoders = _build_cat_encoders_from_distinct(conn)

        # -- Step 6: load future production forecasts (if any) --
        # Filter to latest plan_version to avoid mixing forecasts from different runs.
        # Config keeps last 3 plan_versions; without this filter, dict overwrite is non-deterministic.
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT forecast_month, forecast_qty, model_id
                FROM fact_production_forecast
                WHERE item_id = %s AND loc = %s
                  AND plan_version = (
                      SELECT MAX(plan_version)
                      FROM fact_production_forecast
                      WHERE item_id = %s AND loc = %s
                  )
                ORDER BY forecast_month
                """,
                (item_id, loc, item_id, loc),
            )
            future_rows = cur.fetchall()

    # -- Step 7: build feature matrix --
    # Historical sales series
    if len(sales_rows) < max(LAG_RANGE):
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient sales history for DFU ({len(sales_rows)} months; need ≥{max(LAG_RANGE)}).",
        )

    # Fill calendar gaps with qty=0 so lag indices align with calendar months.
    # fact_sales_monthly only stores months that have sales records; missing months
    # would compress the series and shift all lag/rolling indices vs training, which
    # uses a full cartesian grid with missing months filled to 0.
    import datetime as _dt
    raw_date_qty: dict = {r[0]: float(r[1]) for r in sales_rows}
    if raw_date_qty:
        # Sort dates so first_month/last_month are order-independent of SQL result set
        sorted_months = sorted(raw_date_qty.keys())
        first_month = sorted_months[0]
        last_month = sorted_months[-1]
        # Walk the calendar month-by-month and fill missing entries with 0
        cur_month = first_month
        filled_dates = []
        filled_qtys = []
        while cur_month <= last_month:
            filled_dates.append(cur_month)
            filled_qtys.append(raw_date_qty.get(cur_month, 0.0))
            # Advance one calendar month
            year, month = cur_month.year, cur_month.month
            if month == 12:
                cur_month = _dt.date(year + 1, 1, 1)
            else:
                cur_month = _dt.date(year, month + 1, 1)
        date_series = filled_dates
        qty_series = filled_qtys
    else:
        qty_series = []
        date_series = []

    attrs = {
        "ml_cluster": cluster_str,
        "execution_lag": float(execution_lag or 0),
        "total_lt": float(total_lt or 14),
        "brand": str(brand or "__unknown__"),
        "region": str(region or "__unknown__"),
        "abc_vol": str(abc_vol or "__unknown__"),
        "bpc": float(bpc or 0),
        "item_proof": float(item_proof or 0),
        "case_weight": float(case_weight or 0),
    }

    future_map = {r[0]: float(r[1] or 0) for r in future_rows}
    # Detect if future lags come from a different model than the one being analysed.
    # fact_production_forecast stores only the champion model per DFU, so non-champion
    # model SHAP for future months uses the champion's quantities as lag source.
    future_lag_model_id: str | None = future_rows[0][2] if future_rows else None
    future_months_sorted = sorted(future_map.keys())

    def _build_row(month_date, qty_hist: list[float]) -> dict:
        n = len(qty_hist)
        row: dict = {}
        for lag_n in LAG_RANGE:
            idx = n - lag_n
            row[f"qty_lag_{lag_n}"] = qty_hist[idx] if idx >= 0 else 0.0
        for w in ROLLING_WINDOWS:
            window_vals = qty_hist[max(0, n - w):n]
            if window_vals:
                row[f"rolling_mean_{w}m"] = float(np.mean(window_vals))
                # ddof=1 matches pandas rolling.std() used during backtest training
                row[f"rolling_std_{w}m"] = float(np.std(window_vals, ddof=1)) if len(window_vals) > 1 else 0.0
            else:
                row[f"rolling_mean_{w}m"] = 0.0
                row[f"rolling_std_{w}m"] = 0.0
        try:
            month_num = month_date.month
        except AttributeError:
            import pandas as _pd
            month_num = _pd.Timestamp(month_date).month
        row["month"] = month_num
        row["quarter"] = (month_num - 1) // 3 + 1
        row["month_sin"] = float(np.sin(2 * np.pi * month_num / 12))
        row["month_cos"] = float(np.cos(2 * np.pi * month_num / 12))
        for col in CAT_FEATURES:
            row[col] = attrs.get(col, "__unknown__")
        row["execution_lag"] = attrs["execution_lag"]
        row["total_lt"] = attrs["total_lt"]
        row["bpc"] = attrs["bpc"]
        row["item_proof"] = attrs["item_proof"]
        row["case_weight"] = attrs["case_weight"]
        # Derived features matching common/feature_engineering.py
        lag1 = row.get("qty_lag_1", 0.0)
        lag2 = row.get("qty_lag_2", 0.0)
        row["mom_growth"] = max(-2.0, min(2.0, (lag1 - lag2) / (abs(lag2) + 1.0)))
        rm3 = row.get("rolling_mean_3m", 0.0)
        rm6 = row.get("rolling_mean_6m", 0.0)
        rs3 = row.get("rolling_std_3m", 0.0)
        row["demand_accel"] = rm3 - rm6
        row["volatility_ratio"] = rs3 / (abs(rm3) + 1.0)
        row["is_quarter_end"] = 1 if month_num in (3, 6, 9, 12) else 0
        row["is_year_end"] = 1 if month_num == 12 else 0
        import calendar as _cal
        row["days_in_month"] = float(_cal.monthrange(month_date.year if hasattr(month_date, "year") else 2025, month_num)[1])
        return row

    all_rows = []
    months_meta = []

    # Historical months (skip the first max_lag months — insufficient lag history)
    # Use qty_series[:i] (not [:i+1]) so lag_1 = previous month's actual, matching
    # the backtest which predicts month i from data available BEFORE month i.
    max_lag = max(LAG_RANGE)
    for i in range(max_lag, len(date_series)):
        row = _build_row(date_series[i], qty_series[:i])
        all_rows.append(row)
        months_meta.append({"month": str(date_series[i])[:10], "is_future": False})

    # Future months — append using forecast values as lag source
    future_qty_series = list(qty_series)
    for fm in future_months_sorted:
        row = _build_row(fm, future_qty_series)
        all_rows.append(row)
        months_meta.append({"month": str(fm)[:10], "is_future": True})
        # Write-back: treat this forecast as lag-1 for next month
        future_qty_series.append(future_map[fm])

    if not all_rows:
        raise HTTPException(status_code=422, detail="No data rows could be built for SHAP computation.")

    X = pd.DataFrame(all_rows)

    # Select only feature columns known to model
    avail = [c for c in effective_feature_cols if c in X.columns]

    # Build X_model with the correct categorical representation for each framework:
    #
    # CatBoost: was trained on raw STRING category labels (pandas Categorical → string
    #   labels extracted by CatBoost). Pass strings so CatBoost hashes "NE"/"A"/etc.
    #   the same way it did during training. Integer-encoding would hash "3"/"0" instead.
    #
    # LightGBM / XGBoost: was trained on integer category codes (pandas Categorical
    #   codes, alphabetically sorted → 0-based int). Encode via cat_encoders which
    #   uses the same sorted-unique mapping.
    framework = _detect_model_framework(model, model_id)
    if framework == "catboost":
        X_model = X[avail].copy()
        # Coerce only non-categorical numeric columns (leave cat cols as strings)
        non_cat = [c for c in avail if c not in CAT_FEATURES]
        X_model[non_cat] = X_model[non_cat].apply(pd.to_numeric, errors="coerce").fillna(0)
    else:
        # Encode cat features to integers, then coerce everything to float64
        X_model = X[avail].copy()
        for col in CAT_FEATURES:
            if col in X_model.columns and col in cat_encoders:
                X_model[col] = X_model[col].map(cat_encoders[col]).fillna(0).astype(int)
        X_model = X_model.apply(pd.to_numeric, errors="coerce").fillna(0)

    # -- Step 8: compute SHAP --
    try:
        shap_values, base_values = _compute_shap_full(model, X_model, model_id, avail)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SHAP computation failed: {exc}") from exc

    # shap_values shape: (n_rows, n_avail_features)
    # Select top_n features by mean absolute SHAP across all rows
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][: top_n]
    top_features = [avail[i] for i in top_indices]
    top_indices_set = set(top_indices.tolist())

    # Build per-month response
    points = []
    for i, meta in enumerate(months_meta):
        feat_contribs = [
            {"name": top_features[j], "value": round(float(shap_values[i, top_indices[j]]), 6)}
            for j in range(len(top_features))
        ]
        # Sum of contributions from features NOT in top_n
        other_shap = float(np.sum([
            shap_values[i, k] for k in range(shap_values.shape[1]) if k not in top_indices_set
        ]))
        points.append({
            "month": meta["month"],
            "is_future": meta["is_future"],
            "base_value": round(float(base_values[i]), 6),
            "other_shap": round(other_shap, 6),
            "features": feat_contribs,
        })

    return {
        "item_id": item_id,
        "loc": loc,
        "model_id": model_id,
        "cluster_id": cluster_str,
        "top_n": len(top_features),
        "computed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        # future_lag_model_id: model whose stored forecasts were used as lag source for
        # future months. Differs from model_id when the requested model is not the
        # production champion — the SHAP interpretation for future months is approximate.
        "future_lag_model_id": future_lag_model_id,
        "points": points,
    }
