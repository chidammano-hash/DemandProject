"""Champion / Model Competition endpoints (feature 15).

Uses shared strategy module for leak-free per-DFU per-month selection.
All strategies enforce strict causality: selection for month T uses only
data from months < T.
"""
from __future__ import annotations

import io
import json
from datetime import UTC
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_api_key
from api.core import get_conn
from common.ml.champion import STRATEGY_REGISTRY as _STRAT_REG
from common.core.utils import get_competing_model_ids, load_forecast_pipeline_config, reset_config

router = APIRouter(tags=["competition"])

from common.core.paths import PROJECT_ROOT as _PROJECT_ROOT
_PIPELINE_CONFIG_PATH = _PROJECT_ROOT / "config" / "forecasting" / "forecast_pipeline_config.yaml"
_CHAMPION_SUMMARY_PATH = _PROJECT_ROOT / "data" / "champion" / "champion_summary.json"

_VALID_STRATEGIES = set(_STRAT_REG.keys())


class CompetitionConfigUpdate(BaseModel):
    metric: str = "wape"
    lag: str = "execution"
    min_dfu_rows: int = 3
    champion_model_id: str = "champion"
    models: list[str]
    strategy: str = "expanding"
    strategy_params: dict[str, Any] = {}


def _load_monthly_errors(
    models: list[str], lag_mode: str,
) -> pd.DataFrame:
    """Load per-DFU per-month per-model forecast errors as DataFrame."""
    placeholders = ",".join(["%s"] * len(models))
    params: list[Any] = list(models)

    if lag_mode == "execution":
        lag_cond = "lag::text = execution_lag::text"
    else:
        lag_cond = "lag = %s"
        params.append(int(lag_mode))

    sql = f"""
        SELECT item_id, customer_group, loc, startdate, model_id,
               basefcst_pref, tothist_dmd,
               ABS(basefcst_pref - tothist_dmd) AS abs_err
        FROM fact_external_forecast_monthly
        WHERE model_id IN ({placeholders})
          AND {lag_cond}
          AND basefcst_pref IS NOT NULL
          AND tothist_dmd IS NOT NULL
        ORDER BY item_id, customer_group, loc, model_id, startdate
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    df["startdate"] = pd.to_datetime(df["startdate"])
    for col in ["basefcst_pref", "tothist_dmd", "abs_err"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@router.get("/competition/config")
def get_competition_config():
    """Return current model competition config + available models in DB."""
    pipeline_cfg = load_forecast_pipeline_config()
    champion = pipeline_cfg.get("champion", {})

    # Build config dict matching the legacy response shape
    cfg = {
        "strategy": champion.get("strategy", "expanding"),
        "strategy_params": champion.get("strategy_params", {}),
        "metric": champion.get("metric", "wape"),
        "lag": champion.get("lag", "execution"),
        "min_dfu_rows": champion.get("min_dfu_rows", 3),
        "champion_model_id": champion.get("champion_model_id", "champion"),
        "fallback_model_id": champion.get("fallback_model_id"),
        "models": get_competing_model_ids(),
        "meta_learner": champion.get("meta_learner", {}),
    }

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT model_id FROM fact_external_forecast_monthly ORDER BY 1"
        )
        available = [r[0] for r in cur.fetchall() if r[0]]

    return {"config": cfg, "available_models": available}


@router.put("/competition/config", dependencies=[Depends(require_api_key)])
def update_competition_config(body: CompetitionConfigUpdate):
    """Update model competition config (writes to forecast_pipeline_config.yaml champion section)."""
    import yaml

    if body.metric not in ("wape", "accuracy_pct"):
        raise HTTPException(422, "metric must be 'wape' or 'accuracy_pct'")
    valid_lags = {"execution", "0", "1", "2", "3", "4"}
    if body.lag not in valid_lags:
        raise HTTPException(422, f"lag must be one of: {sorted(valid_lags)}")
    if len(body.models) < 2:
        raise HTTPException(422, "At least 2 models required for competition")
    if body.strategy not in _VALID_STRATEGIES:
        raise HTTPException(422, f"strategy must be one of: {sorted(_VALID_STRATEGIES)}")

    # Read existing pipeline config
    with _PIPELINE_CONFIG_PATH.open() as f:
        pipeline_cfg = yaml.safe_load(f) or {}

    # Update champion section
    champion = pipeline_cfg.setdefault("champion", {})
    champion["metric"] = body.metric
    champion["lag"] = body.lag
    champion["min_dfu_rows"] = body.min_dfu_rows
    champion["champion_model_id"] = body.champion_model_id
    champion["strategy"] = body.strategy
    champion["strategy_params"] = body.strategy_params

    with _PIPELINE_CONFIG_PATH.open("w") as f:
        yaml.dump(pipeline_cfg, f, default_flow_style=False, sort_keys=False)

    reset_config("forecast_pipeline_config.yaml")

    return {"status": "ok", "config": champion}


def _insert_pick_winners(
    winners_df: pd.DataFrame, target_model_id: str, table_suffix: str = "winners",
) -> int:
    """Bulk-insert pick-one winner rows into fact_external_forecast_monthly.

    Creates a temp table, COPYs winner mappings, then joins to copy full
    forecast rows under *target_model_id*.  Returns the number of rows inserted.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM fact_external_forecast_monthly WHERE model_id = %s",
            (target_model_id,),
        )
        cur.execute(f"""
            CREATE TEMP TABLE _{table_suffix} (
                item_id TEXT NOT NULL,
                customer_group TEXT NOT NULL,
                loc TEXT NOT NULL,
                startdate DATE NOT NULL,
                winning_model_id TEXT NOT NULL
            ) ON COMMIT DROP
        """)

        buf = io.StringIO()
        for _, r in winners_df.iterrows():
            buf.write(
                f"{r['item_id']}\t{r['customer_group']}\t{r['loc']}\t"
                f"{r['startdate'].date()}\t{r['model_id']}\n"
            )
        buf.seek(0)
        with cur.copy(f"COPY _{table_suffix} FROM STDIN") as copy:
            copy.write(buf.read())

        cur.execute(
            f"""
            INSERT INTO fact_external_forecast_monthly
                (forecast_ck, item_id, customer_group, loc, fcstdate, startdate,
                 lag, execution_lag, basefcst_pref, tothist_dmd, model_id)
            SELECT
                f.forecast_ck, f.item_id, f.customer_group, f.loc, f.fcstdate,
                f.startdate, f.lag, f.execution_lag, f.basefcst_pref,
                f.tothist_dmd, %s
            FROM fact_external_forecast_monthly f
            INNER JOIN _{table_suffix} w
                ON f.item_id = w.item_id
               AND f.customer_group = w.customer_group
               AND f.loc = w.loc
               AND f.startdate = w.startdate
               AND f.model_id = w.winning_model_id
            """,
            (target_model_id,),
        )
        inserted = cur.rowcount
        conn.commit()
    return inserted


def _insert_ensemble_winners(
    winners_df: pd.DataFrame, target_model_id: str, models: list[str],
) -> int:
    """Bulk-insert ensemble (blended) winner rows into fact_external_forecast_monthly.

    Copies blended basefcst_pref values while borrowing metadata from any
    competing model's row for the same DFU-month.  Returns inserted count.
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM fact_external_forecast_monthly WHERE model_id = %s",
            (target_model_id,),
        )
        cur.execute("""
            CREATE TEMP TABLE _champion_ensemble (
                item_id TEXT NOT NULL,
                customer_group TEXT NOT NULL,
                loc TEXT NOT NULL,
                startdate DATE NOT NULL,
                basefcst_pref DOUBLE PRECISION NOT NULL,
                tothist_dmd DOUBLE PRECISION NOT NULL
            ) ON COMMIT DROP
        """)

        buf = io.StringIO()
        for _, r in winners_df.iterrows():
            buf.write(
                f"{r['item_id']}\t{r['customer_group']}\t{r['loc']}\t"
                f"{r['startdate'].date()}\t{r['basefcst_pref']}\t"
                f"{r['tothist_dmd']}\n"
            )
        buf.seek(0)
        with cur.copy("COPY _champion_ensemble FROM STDIN") as copy:
            copy.write(buf.read())

        placeholders = ",".join(["%s"] * len(models))
        cur.execute(
            f"""
            INSERT INTO fact_external_forecast_monthly
                (forecast_ck, item_id, customer_group, loc, fcstdate, startdate,
                 lag, execution_lag, basefcst_pref, tothist_dmd, model_id)
            SELECT DISTINCT ON (e.item_id, e.customer_group, e.loc, e.startdate)
                f.forecast_ck, f.item_id, f.customer_group, f.loc, f.fcstdate,
                f.startdate, f.lag, f.execution_lag,
                e.basefcst_pref, e.tothist_dmd,
                %s
            FROM _champion_ensemble e
            INNER JOIN fact_external_forecast_monthly f
                ON f.item_id = e.item_id
               AND f.customer_group = e.customer_group
               AND f.loc = e.loc
               AND f.startdate = e.startdate
               AND f.model_id IN ({placeholders})
            ORDER BY e.item_id, e.customer_group, e.loc, e.startdate, f.model_id
            """,
            [target_model_id, *models],
        )
        inserted = cur.rowcount
        conn.commit()
    return inserted


def _count_model_wins(df: pd.DataFrame) -> dict[str, int]:
    """Count per-model wins from a winners DataFrame, sorted descending."""
    wins: dict[str, int] = {}
    for mid in df["model_id"]:
        wins[mid] = wins.get(mid, 0) + 1
    return dict(sorted(wins.items(), key=lambda x: -x[1]))


def _enqueue_forecast_view_refresh() -> str | None:
    """Submit a background job that refreshes forecast-dependent MVs.

    The actual REFRESH was previously executed synchronously inside the
    request handler. At production scale this exceeded the 30s
    ``statement_timeout`` and held ACCESS EXCLUSIVE locks that blocked other
    readers. Moving the work to ``job_registry`` runs it on the APScheduler
    pool with ``REFRESH ... CONCURRENTLY``, so the API returns immediately and
    callers can poll ``GET /jobs/{id}`` for completion.

    Returns the job_id string when the job was submitted, or ``None`` when the
    scheduler is unavailable (e.g. unit-test environments without
    APScheduler) — in which case the refresh is silently skipped rather than
    blocking the request.
    """
    try:
        from common.services.job_registry import JobManager

        return JobManager().submit_job(
            job_type="refresh_forecast_views",
            params={},
            label="Forecast MV refresh (post-competition)",
            triggered_by="competition_run",
        )
    except Exception:
        # Scheduler not available (tests) or DB-side failure — degrade
        # gracefully. The next regular MV refresh job will pick up the data.
        logger = __import__("logging").getLogger(__name__)
        logger.exception("Failed to enqueue forecast view refresh job")
        return None


@router.post("/competition/run", dependencies=[Depends(require_api_key)])
def run_competition():
    """Execute champion model selection using configured strategy.

    Uses shared strategy module for leak-free per-DFU per-month selection.
    All strategies enforce strict causality: selection for month T uses only
    data from months < T.
    """
    from datetime import datetime

    from common.ml.champion import (
        STRATEGY_REGISTRY,
        compute_ceiling,
        compute_strategy_accuracy,
    )

    pipeline_cfg = load_forecast_pipeline_config()
    cfg = pipeline_cfg.get("champion", {})

    models = get_competing_model_ids()
    lag_mode = str(cfg.get("lag", "execution"))
    min_rows = int(cfg.get("min_dfu_rows", 3))
    champion_id = cfg.get("champion_model_id", "champion")
    strategy_name = cfg.get("strategy", "expanding")
    strategy_params = cfg.get("strategy_params", {})

    if len(models) < 2:
        raise HTTPException(422, "At least 2 models required for competition")

    strategy_fn = STRATEGY_REGISTRY.get(strategy_name)
    if strategy_fn is None:
        raise HTTPException(422, f"Unknown strategy: {strategy_name}")

    monthly_errors = _load_monthly_errors(models, lag_mode)
    if monthly_errors.empty:
        raise HTTPException(404, "No forecast data found for configured models")

    strat_kwargs: dict[str, Any] = {
        "min_prior_months": min_rows,
        **strategy_params,
    }
    winners_df = strategy_fn(monthly_errors, **strat_kwargs)

    if winners_df.empty:
        raise HTTPException(404, "No qualifying DFUs found with current config")

    champion_acc = compute_strategy_accuracy(winners_df)

    # Insert champion rows (ensemble vs pick-one)
    if strategy_name == "ensemble":
        inserted = _insert_ensemble_winners(winners_df, champion_id, models)
    else:
        inserted = _insert_pick_winners(winners_df, champion_id, "champion_winners")

    # Compute and insert ceiling (oracle best-per-DFU-month)
    ceiling_id = cfg.get("ceiling_model_id", "ceiling")
    ceiling_df = compute_ceiling(monthly_errors)
    ceiling_acc = compute_strategy_accuracy(ceiling_df)
    ceiling_inserted = 0
    if not ceiling_df.empty:
        ceiling_inserted = _insert_pick_winners(ceiling_df, ceiling_id, "ceiling_winners")

    mv_refresh_job_id = _enqueue_forecast_view_refresh()

    # Build summary
    n_unique_dfus = (
        winners_df[["item_id", "customer_group", "loc"]]
        .drop_duplicates()
        .shape[0]
    )

    summary: dict[str, Any] = {
        "config": {
            "metric": cfg.get("metric", "wape"),
            "lag": lag_mode,
            "min_dfu_rows": min_rows,
            "champion_model_id": champion_id,
            "models": models,
            "strategy": strategy_name,
        },
        "total_dfus": n_unique_dfus,
        "total_dfu_months": len(winners_df),
        "total_champion_rows": inserted,
        "model_wins": _count_model_wins(winners_df),
        "overall_champion_wape": champion_acc.get("wape"),
        "overall_champion_accuracy_pct": champion_acc.get("accuracy_pct"),
        "run_ts": datetime.now(UTC).isoformat(),
        "mv_refresh_job_id": mv_refresh_job_id,
    }

    if not ceiling_df.empty:
        summary["total_ceiling_rows"] = ceiling_inserted
        summary["ceiling_model_wins"] = _count_model_wins(ceiling_df)
        summary["overall_ceiling_wape"] = ceiling_acc.get("wape")
        summary["overall_ceiling_accuracy_pct"] = ceiling_acc.get("accuracy_pct")

    _CHAMPION_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CHAMPION_SUMMARY_PATH.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


@router.get("/competition/summary")
def get_competition_summary():
    """Return the last champion selection summary, if available."""
    if not _CHAMPION_SUMMARY_PATH.exists():
        return {"status": "not_run", "summary": None}
    with _CHAMPION_SUMMARY_PATH.open() as f:
        return {"status": "ok", "summary": json.load(f)}
