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

import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from psycopg import Error as PsycopgError
from psycopg import sql

from api.core import get_conn
from common.core.constants import CAT_FEATURES, LAG_RANGE, ROLLING_WINDOWS
from common.core.paths import PROJECT_ROOT
from common.core.planning_date import get_planning_date
from common.core.utils import load_forecast_pipeline_config
from common.ml.feature_engineering import compute_ts_profile_from_values
from common.ml.tree_artifact_lineage import ProductionTreeArtifactLineage
from common.ml.tree_artifacts import (
    LoadedTreeArtifactSet,
    build_production_tree_model_config_payload,
    build_tree_artifact_spec,
    load_active_tree_artifact_set,
)
from common.services.cluster_lineage import load_promoted_cluster_population
from common.services.forecast_population import resolve_forecast_sales_table
from common.services.sales_lineage import load_completed_sales_lineage

router = APIRouter(tags=["shap"])
logger = logging.getLogger(__name__)

# Root of the model-scoped backtest output directories
_BACKTEST_DATA_DIR = Path(os.environ.get("BACKTEST_DATA_DIR", "data/backtest"))

_SHAP_MODEL_ID = "lgbm_cluster"


def _validate_shap_model_id(model_id: str) -> None:
    if model_id != _SHAP_MODEL_ID:
        raise HTTPException(
            status_code=404,
            detail=f"SHAP is available only for '{_SHAP_MODEL_ID}'.",
        )


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

    sql = (
        "SELECT DISTINCT ca.ml_cluster::TEXT FROM dim_sku d "
        "LEFT JOIN current_sku_cluster_assignment ca ON ca.sku_ck = d.sku_ck"
    )
    if need_item_join:
        sql += " LEFT JOIN dim_item i ON i.item_id = d.item_id"
    if need_loc_join:
        sql += " LEFT JOIN dim_location l ON l.location_id = d.loc"
    sql += f" WHERE {' AND '.join(conditions)} AND ca.ml_cluster IS NOT NULL"

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
    """List the canonical LightGBM model when its SHAP output exists."""
    if not _summary_csv(_SHAP_MODEL_ID).exists():
        return {"models": []}
    return {"models": [_SHAP_MODEL_ID]}


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
    _validate_shap_model_id(model_id)
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
        summary = summary.sort_values(
            "mean_abs_shap_across_timeframes", ascending=False
        ).reset_index(drop=True)
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
    _validate_shap_model_id(model_id)
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
            timeframes.append(
                {
                    "index": idx,
                    "label": label,
                    "cutoff_date": str(cutoff),
                }
            )
        except (ValueError, TypeError, KeyError, IndexError):
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
    _validate_shap_model_id(model_id)
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
        if cluster == "all":
            # Prefer a saved pooled view when present. Per-cluster backtests do
            # not currently write one, so aggregate their repeated feature rows
            # before ranking to keep one meaningful bar per feature.
            pooled = df[df["cluster"] == "all"]
            if not pooled.empty:
                df = pooled
            else:
                cluster_labels = [str(value) for value in df["cluster"].dropna().unique()]
                df = _aggregate_cluster_shap(df, cluster_labels)
        else:
            filtered = df[df["cluster"] == cluster]
            if filtered.empty:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"No SHAP data for cluster '{cluster}' in model "
                        f"'{model_id}' timeframe {idx}."
                    ),
                )
            df = filtered

    df = df.sort_values("rank").reset_index(drop=True)
    label = chr(ord("A") + idx) if 0 <= idx <= 25 else str(idx)
    cutoff_date = str(df["cutoff_date"].iloc[0]) if not df.empty else ""
    features = df.head(top_n).to_dict(orient="records")

    # List available clusters from the full CSV
    full_df = pd.read_csv(csv_path)
    available_clusters = (
        sorted(full_df["cluster"].unique().tolist()) if "cluster" in full_df.columns else ["all"]
    )

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
    _validate_shap_model_id(model_id)
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


def _latest_closed_month() -> date:
    planning_month = pd.Timestamp(get_planning_date()).normalize().replace(day=1)
    return (planning_month - pd.DateOffset(months=1)).date()


def _model_registry_base_path(config: dict) -> Path:
    production = config.get("production_forecast")
    if not isinstance(production, dict):
        raise RuntimeError("Forecast configuration is missing production_forecast")
    registry = production.get("model_registry")
    if not isinstance(registry, dict):
        raise RuntimeError("Forecast configuration is missing model_registry")
    raw_path = registry.get("base_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise RuntimeError("Forecast model_registry.base_path must be configured")
    base_path = Path(raw_path)
    return base_path if base_path.is_absolute() else PROJECT_ROOT / base_path


def _load_active_lgbm_artifact_set(conn) -> LoadedTreeArtifactSet:
    """Load the exact active LightGBM set under the production lineage contract."""
    config = load_forecast_pipeline_config()
    clustering = config.get("clustering")
    if not isinstance(clustering, dict):
        raise RuntimeError("Forecast configuration is missing clustering settings")
    clustering_enabled = clustering.get("enabled")
    if not isinstance(clustering_enabled, bool):
        raise RuntimeError("clustering.enabled must be explicitly true or false")

    sales_lineage = load_completed_sales_lineage(conn)
    promoted_clusters = load_promoted_cluster_population(conn) if clustering_enabled else None
    lineage = ProductionTreeArtifactLineage(
        source_sales_batch_id=sales_lineage.batch_id,
        data_checksum=sales_lineage.source_hash,
        history_end=_latest_closed_month(),
        cluster_experiment_id=(promoted_clusters.experiment_id if promoted_clusters else None),
        cluster_assignment_count=(
            promoted_clusters.assignment_count if promoted_clusters else None
        ),
        cluster_assignment_checksum=(
            promoted_clusters.assignment_checksum if promoted_clusters else None
        ),
    )
    cluster_strategy = "per_cluster" if clustering_enabled else "global"
    cluster_labels = (
        promoted_clusters.cluster_labels if promoted_clusters else frozenset({"global"})
    )
    model_config = build_production_tree_model_config_payload(
        config,
        model_id=_SHAP_MODEL_ID,
        project_root=PROJECT_ROOT,
    )
    expected_spec = build_tree_artifact_spec(
        model_id=_SHAP_MODEL_ID,
        model_config=model_config,
        lineage=lineage,
        cluster_strategy=cluster_strategy,
        cluster_labels=cluster_labels,
    )
    return load_active_tree_artifact_set(
        model_id=_SHAP_MODEL_ID,
        expected_spec=expected_spec,
        base_dir=_model_registry_base_path(config),
    )


def _resolve_dfu_context(
    conn,
    *,
    item_id: str,
    loc: str,
    customer_group: str | None,
) -> tuple:
    """Resolve one exact training-grain DFU; never choose a group implicitly."""
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH item_location_dfus AS (
                SELECT d.sku_ck, ca.ml_cluster, d.execution_lag, d.total_lt,
                       d.brand, d.region, d.abc_vol, d.customer_group,
                       i.bpc, i.item_proof, i.case_weight,
                       COUNT(*) OVER () AS item_location_dfu_count
                FROM dim_sku d
                LEFT JOIN current_sku_cluster_assignment ca
                       ON ca.sku_ck = d.sku_ck
                LEFT JOIN dim_item i ON i.item_id = d.item_id
                WHERE d.item_id = %s AND d.loc = %s
            )
            SELECT *
            FROM item_location_dfus
            WHERE (%s::TEXT IS NULL OR customer_group = %s)
            ORDER BY customer_group, sku_ck
            """,
            (item_id, loc, customer_group, customer_group),
        )
        rows = cur.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=(
                f"DFU not found: item_id={item_id!r}, loc={loc!r}, "
                f"customer_group={customer_group!r}"
            ),
        )
    if len(rows) > 1 and customer_group is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "This item/location has multiple customer_group values; provide "
                "customer_group to select one exact DFU."
            ),
        )
    if len(rows) > 1:
        raise HTTPException(
            status_code=409,
            detail="The requested DFU resolves to multiple dimension rows.",
        )
    return rows[0]


def _resolve_artifact_for_dfu(
    loaded: LoadedTreeArtifactSet,
    ml_cluster: object,
) -> tuple[str, dict]:
    strategy = loaded.ref.metadata.get("cluster_strategy")
    if strategy == "global":
        cluster_label = "global"
    elif strategy == "per_cluster":
        if ml_cluster is None or not str(ml_cluster).strip():
            raise RuntimeError("The DFU has no promoted cluster assignment")
        cluster_label = str(ml_cluster).strip()
    else:
        raise RuntimeError("The active LightGBM artifact has an invalid cluster strategy")

    artifact = loaded.artifacts.get(cluster_label)
    if not isinstance(artifact, dict):
        raise RuntimeError(
            f"The active LightGBM artifact has no exact model for cluster {cluster_label!r}"
        )
    return cluster_label, artifact


def _artifact_history_lookback(loaded: LoadedTreeArtifactSet) -> int:
    model_config = loaded.ref.metadata.get("model_config")
    feature_history = (
        model_config.get("feature_history") if isinstance(model_config, dict) else None
    )
    lookback = feature_history.get("lookback_months") if isinstance(feature_history, dict) else None
    if not isinstance(lookback, int) or isinstance(lookback, bool) or lookback <= 0:
        raise RuntimeError("The active LightGBM artifact has no valid history lookback")
    return lookback


def _load_shap_sales_history(
    conn,
    *,
    item_id: str,
    customer_group: str,
    loc: str,
    lookback_months: int,
) -> tuple[list[tuple[date, float]], date, date]:
    """Load the canonical source on the same latest-closed window as generation."""
    history_end = _latest_closed_month()
    history_start = (pd.Timestamp(history_end) - pd.DateOffset(months=lookback_months - 1)).date()
    with conn.cursor() as cur:
        sales_table = resolve_forecast_sales_table(cur)
        table = sql.Identifier(sales_table)
        cur.execute(
            sql.SQL(
                """SELECT MAX(startdate)
                   FROM {}
                   WHERE type = 1 AND qty IS NOT NULL AND startdate <= %s"""
            ).format(table),
            (history_end,),
        )
        latest_row = cur.fetchone()
        latest_month = latest_row[0] if latest_row else None
        normalized_latest = (
            pd.Timestamp(latest_month).normalize().date() if latest_month is not None else None
        )
        if normalized_latest != history_end:
            raise RuntimeError("The canonical sales source is not latest-closed complete")

        cur.execute(
            sql.SQL(
                """SELECT startdate, SUM(qty) AS qty
                   FROM {}
                   WHERE item_id = %s
                     AND customer_group = %s
                     AND loc = %s
                     AND type = 1
                     AND qty IS NOT NULL
                     AND startdate >= %s
                     AND startdate <= %s
                   GROUP BY startdate
                   ORDER BY startdate"""
            ).format(table),
            (item_id, customer_group, loc, history_start, history_end),
        )
        rows = [(row[0], float(row[1])) for row in cur.fetchall()]
    return rows, history_start, history_end


def _load_future_forecast_rows(
    conn,
    *,
    item_id: str,
    loc: str,
    history_end: date,
) -> list[tuple[date, float, str]]:
    """Load the active aggregate plan used only as an approximate future lag source."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT forecast_month, forecast_qty, model_id
               FROM fact_production_forecast
               WHERE item_id = %s AND loc = %s
                 AND forecast_month > %s
                 AND plan_version = (
                     SELECT MAX(plan_version)
                     FROM fact_production_forecast
                     WHERE item_id = %s AND loc = %s
                 )
               ORDER BY forecast_month""",
            (item_id, loc, history_end, item_id, loc),
        )
        return [(row[0], float(row[1] or 0), str(row[2])) for row in cur.fetchall()]


def _extract_model_feature_names(model, model_id: str) -> list[str] | None:
    """Extract the feature names from a LightGBM artifact.

    Returns None if the framework doesn't expose feature names or if the
    returned value is not a valid list of strings.
    """
    if model_id != _SHAP_MODEL_ID:
        return None
    try:
        names = getattr(model, "feature_name_", None)
        if names is None and hasattr(model, "booster_"):
            booster = getattr(model, "booster_", None)
            if booster is not None and hasattr(booster, "feature_name"):
                names = booster.feature_name()
        # Validate: must be a real list of strings (not a MagicMock or other proxy)
        if (
            names is not None
            and isinstance(names, (list, tuple))
            and len(names) > 0
            and isinstance(names[0], str)
        ):
            return list(names)
        return None
    except (AttributeError, TypeError):
        return None


def _compute_shap_full(
    model, X: pd.DataFrame, model_id: str, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute signed LightGBM SHAP values and per-row base values.

    Returns:
        shap_vals: shape (n_rows, n_features) — signed per-feature contributions
        base_vals: shape (n_rows,) — expected value (base prediction) per row
    """
    if model_id != _SHAP_MODEL_ID:
        raise ValueError(f"SHAP is available only for {_SHAP_MODEL_ID}")
    # Pass as numpy to bypass LightGBM categorical dtype validation. The
    # integer-encoded category values match the production artifact encoders.
    full = model.predict(X.to_numpy(), pred_contrib=True)
    return full[:, :-1], full[:, -1]


@router.get("/forecast/shap/{model_id}/dfu")
async def shap_dfu(
    model_id: str,
    item_id: str = Query(..., description="Item number (item_id)"),
    loc: str = Query(..., description="Location code"),
    customer_group: str | None = Query(
        default=None,
        description="Customer group required when item/location is ambiguous",
    ),
    top_n: int = Query(default=10, ge=1, le=30),
    lookback_months: int = Query(default=48, ge=12, le=60),
) -> dict:
    """Explain one exact DFU with its lineage-valid active production model."""
    _validate_shap_model_id(model_id)

    with get_conn() as conn:
        try:
            dfu_row = _resolve_dfu_context(
                conn,
                item_id=item_id,
                loc=loc,
                customer_group=customer_group,
            )
        except PsycopgError:
            logger.exception("Resolving the requested DFU for SHAP failed")
            raise HTTPException(
                status_code=500,
                detail="Failed to resolve the requested DFU",
            ) from None
        (
            _sku_ck,
            ml_cluster,
            execution_lag,
            total_lt,
            brand,
            region,
            abc_vol,
            resolved_customer_group,
            bpc,
            item_proof,
            case_weight,
            item_location_dfu_count,
        ) = dfu_row

        try:
            loaded_set = _load_active_lgbm_artifact_set(conn)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No active LightGBM production artifact is published. Run "
                    "'make train-production MODEL=lgbm_cluster'."
                ),
            ) from None
        except (RuntimeError, ValueError):
            logger.exception("Active LightGBM production artifact is stale or invalid")
            raise HTTPException(
                status_code=409,
                detail=(
                    "The active LightGBM production artifact is stale or invalid. "
                    "Run 'make train-production MODEL=lgbm_cluster'."
                ),
            ) from None

        try:
            cluster_str, artifact = _resolve_artifact_for_dfu(
                loaded_set,
                ml_cluster,
            )
            artifact_lookback_months = _artifact_history_lookback(loaded_set)
        except RuntimeError:
            logger.exception("Resolving the exact LightGBM cluster artifact failed")
            raise HTTPException(
                status_code=409,
                detail=(
                    "The active LightGBM production artifact does not match this "
                    "DFU's promoted cluster assignment. Retrain the production model."
                ),
            ) from None

        model = artifact.get("model")
        feature_cols = artifact.get("feature_cols")
        if not callable(getattr(model, "predict", None)) or not isinstance(feature_cols, list):
            raise HTTPException(
                status_code=409,
                detail="The active LightGBM production artifact is invalid.",
            )
        if (
            not feature_cols
            or any(not isinstance(name, str) for name in feature_cols)
            or len(set(feature_cols)) != len(feature_cols)
        ):
            raise HTTPException(
                status_code=409,
                detail="The active LightGBM production artifact has invalid features.",
            )
        if "ml_cluster" in feature_cols:
            raise HTTPException(
                status_code=409,
                detail="The active LightGBM artifact improperly uses cluster metadata as a feature.",
            )
        model_feature_names = _extract_model_feature_names(model, model_id)
        if model_feature_names is not None and model_feature_names != feature_cols:
            raise HTTPException(
                status_code=409,
                detail="The active LightGBM artifact feature contract is inconsistent.",
            )

        try:
            sales_rows, history_start, history_end = _load_shap_sales_history(
                conn,
                item_id=item_id,
                customer_group=str(resolved_customer_group),
                loc=loc,
                lookback_months=artifact_lookback_months,
            )
        except RuntimeError:
            logger.exception("Loading latest-closed canonical SHAP sales history failed")
            raise HTTPException(
                status_code=409,
                detail=(
                    "The canonical sales source is not ready through the latest "
                    "closed month. Complete the sales load before requesting SHAP."
                ),
            ) from None
        if not sales_rows:
            raise HTTPException(
                status_code=422,
                detail="The requested DFU has no canonical sales history to explain.",
            )
        # The published plan is aggregated to item/location. It is a valid lag
        # source only when that item/location contains this one customer group;
        # otherwise attributing the aggregate to one constituent DFU is false.
        future_rows = (
            _load_future_forecast_rows(
                conn,
                item_id=item_id,
                loc=loc,
                history_end=history_end,
            )
            if int(item_location_dfu_count) == 1
            else []
        )

    # Match production's shared calendar: every DFU is completed from the
    # configured lookback boundary through the latest closed month, including
    # post-sale zero months for intermittent demand.
    raw_date_qty = {pd.Timestamp(row[0]).date(): float(row[1]) for row in sales_rows}
    date_series = [
        timestamp.date() for timestamp in pd.date_range(history_start, history_end, freq="MS")
    ]
    qty_series = [raw_date_qty.get(month, 0.0) for month in date_series]

    attrs = {
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
    ts_profile = compute_ts_profile_from_values(qty_series)

    def _build_row(month_date, qty_hist: list[float]) -> dict:
        n = len(qty_hist)
        row: dict = {}
        for lag_n in LAG_RANGE:
            idx = n - lag_n
            row[f"qty_lag_{lag_n}"] = qty_hist[idx] if idx >= 0 else 0.0
        for w in ROLLING_WINDOWS:
            window_vals = qty_hist[max(0, n - w) : n]
            if window_vals:
                row[f"rolling_mean_{w}m"] = float(np.mean(window_vals))
                # ddof=1 matches pandas rolling.std() used during backtest training
                row[f"rolling_std_{w}m"] = (
                    float(np.std(window_vals, ddof=1)) if len(window_vals) > 1 else 0.0
                )
            else:
                row[f"rolling_mean_{w}m"] = 0.0
                row[f"rolling_std_{w}m"] = 0.0
        month_timestamp = pd.Timestamp(month_date)
        month_num = month_timestamp.month
        row["month"] = month_num
        row["quarter"] = (month_num - 1) // 3 + 1
        # Fourier seasonal terms (replaces legacy month_sin/cos)
        for period in [12, 6, 4, 3]:
            angle = 2.0 * np.pi * month_num / period
            row[f"fourier_sin_{period}"] = float(np.sin(angle))
            row[f"fourier_cos_{period}"] = float(np.cos(angle))
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
        lag12 = row.get("qty_lag_12", 0.0)
        row["mom_growth"] = max(-2.0, min(2.0, (lag1 - lag2) / (abs(lag2) + 1.0)))
        rm3 = row.get("rolling_mean_3m", 0.0)
        rm6 = row.get("rolling_mean_6m", 0.0)
        rm12 = row.get("rolling_mean_12m", 0.0)
        rs3 = row.get("rolling_std_3m", 0.0)
        row["demand_accel"] = rm3 - rm6
        row["volatility_ratio"] = rs3 / (abs(rm3) + 1.0)
        row["lag_ratio_yoy"] = max(-10.0, min(10.0, lag1 / (abs(lag12) + 1.0)))
        row["lag_ratio_mom"] = max(-10.0, min(10.0, lag1 / (abs(lag2) + 1.0)))
        row["lag_ratio_3v12"] = max(-10.0, min(10.0, rm3 / (abs(rm12) + 1.0)))
        row["n_zero_last_6m"] = sum(
            1.0 for lag in range(1, 7) if row.get(f"qty_lag_{lag}", 0.0) == 0.0
        )
        row.update(ts_profile)
        row["is_quarter_end"] = 1 if month_num in (3, 6, 9, 12) else 0
        row["is_year_end"] = 1 if month_num == 12 else 0
        row["days_in_month"] = float(month_timestamp.days_in_month)
        return row

    all_rows = []
    months_meta = []

    # Historical months (skip the first max_lag months — insufficient lag history)
    # Use qty_series[:i] (not [:i+1]) so lag_1 = previous month's actual, matching
    # the backtest which predicts month i from data available BEFORE month i.
    max_lag = max(LAG_RANGE)
    first_display_index = max(max_lag, len(date_series) - lookback_months)
    for i in range(first_display_index, len(date_series)):
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
        raise HTTPException(
            status_code=422, detail="No data rows could be built for SHAP computation."
        )

    X = pd.DataFrame(all_rows)

    # Production inference zero-fills only genuinely unavailable numeric model
    # features. Categorical codes must come from the immutable fitted artifact;
    # rebuilding against today's dimension universe changes LightGBM split paths.
    for feature_name in feature_cols:
        if feature_name not in X.columns:
            X[feature_name] = 0.0
    avail = list(feature_cols)
    X_model = X[avail].copy()
    categorical_features = [name for name in avail if name in CAT_FEATURES]
    raw_encoders = artifact.get("categorical_encoders")
    if categorical_features and not isinstance(raw_encoders, dict):
        raise HTTPException(
            status_code=409,
            detail="The active LightGBM artifact is missing its categorical encoders.",
        )
    for column in categorical_features:
        mapping = raw_encoders.get(column) if isinstance(raw_encoders, dict) else None
        if not isinstance(mapping, dict):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"The active LightGBM artifact is missing the categorical encoder for {column}."
                ),
            )
        values = X_model[column].fillna("__unknown__").astype(str)
        if not set(values).issubset(mapping):
            raise HTTPException(
                status_code=409,
                detail=(
                    "The active LightGBM categorical encoder is incompatible with "
                    f"this DFU's {column}. Retrain the production model."
                ),
            )
        X_model[column] = values.map(mapping).astype(int)
    X_model = X_model.apply(pd.to_numeric, errors="coerce").fillna(0)

    # -- Step 8: compute SHAP --
    try:
        shap_values, base_values = _compute_shap_full(model, X_model, model_id, avail)
    except (ValueError, RuntimeError, MemoryError):
        logger.exception("SHAP computation failed for model %s", model_id)
        raise HTTPException(status_code=500, detail="SHAP computation failed") from None

    expected_shape = (len(X_model), len(avail))
    if shap_values.shape != expected_shape or base_values.shape != (len(X_model),):
        logger.error(
            "SHAP result shape mismatch for %s: values=%s base=%s expected=%s",
            model_id,
            shap_values.shape,
            base_values.shape,
            expected_shape,
        )
        raise HTTPException(status_code=500, detail="SHAP computation failed")

    # shap_values shape: (n_rows, n_avail_features)
    # Select top_n features by mean absolute SHAP across all rows
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n]
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
        other_shap = float(
            np.sum(
                [shap_values[i, k] for k in range(shap_values.shape[1]) if k not in top_indices_set]
            )
        )
        points.append(
            {
                "month": meta["month"],
                "is_future": meta["is_future"],
                "base_value": round(float(base_values[i]), 6),
                "other_shap": round(other_shap, 6),
                "features": feat_contribs,
            }
        )

    return {
        "item_id": item_id,
        "customer_group": str(resolved_customer_group),
        "loc": loc,
        "model_id": model_id,
        "cluster_id": cluster_str,
        "artifact_set_id": loaded_set.ref.artifact_set_id,
        "history_end": history_end.isoformat(),
        "top_n": len(top_features),
        "computed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        # future_lag_model_id: model whose stored forecasts were used as lag source for
        # future months. Differs from model_id when the requested model is not the
        # production champion — the SHAP interpretation for future months is approximate.
        "future_lag_model_id": future_lag_model_id,
        "points": points,
    }
