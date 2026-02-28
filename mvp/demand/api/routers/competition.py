"""Champion / Model Competition endpoints (feature 15).

Uses shared strategy module for leak-free per-DFU per-month selection.
All strategies enforce strict causality: selection for month T uses only
data from months < T.
"""
from __future__ import annotations

from typing import Any
import io
import json
import os

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.core import get_conn
from api.auth import require_api_key

router = APIRouter()

_COMPETITION_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "model_competition.yaml",
)

_CHAMPION_SUMMARY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "champion", "champion_summary.json",
)

_VALID_STRATEGIES = {"expanding", "rolling", "decay", "ensemble", "meta_learner"}


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
        SELECT dmdunit, dmdgroup, loc, startdate, model_id,
               basefcst_pref, tothist_dmd,
               ABS(basefcst_pref - tothist_dmd) AS abs_err
        FROM fact_external_forecast_monthly
        WHERE model_id IN ({placeholders})
          AND {lag_cond}
          AND basefcst_pref IS NOT NULL
          AND tothist_dmd IS NOT NULL
        ORDER BY dmdunit, dmdgroup, loc, model_id, startdate
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
    import yaml

    if not os.path.exists(_COMPETITION_CONFIG_PATH):
        raise HTTPException(404, "Competition config not found")
    with open(_COMPETITION_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("competition", {})

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT model_id FROM fact_external_forecast_monthly ORDER BY 1"
        )
        available = [r[0] for r in cur.fetchall() if r[0]]

    return {"config": cfg, "available_models": available}


@router.put("/competition/config", dependencies=[Depends(require_api_key)])
def update_competition_config(body: CompetitionConfigUpdate):
    """Update model competition config (writes YAML to disk)."""
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

    cfg = {
        "competition": {
            "name": "default",
            "metric": body.metric,
            "lag": body.lag,
            "min_dfu_rows": body.min_dfu_rows,
            "champion_model_id": body.champion_model_id,
            "models": body.models,
            "strategy": body.strategy,
            "strategy_params": body.strategy_params,
        }
    }
    os.makedirs(os.path.dirname(_COMPETITION_CONFIG_PATH), exist_ok=True)
    with open(_COMPETITION_CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return {"status": "ok", "config": cfg["competition"]}


@router.post("/competition/run", dependencies=[Depends(require_api_key)])
def run_competition():
    """Execute champion model selection using configured strategy.

    Uses shared strategy module for leak-free per-DFU per-month selection.
    All strategies enforce strict causality: selection for month T uses only
    data from months < T.
    """
    import yaml
    from datetime import datetime, timezone
    from common.champion_strategies import (
        STRATEGY_REGISTRY,
        compute_ceiling,
        compute_strategy_accuracy,
    )

    if not os.path.exists(_COMPETITION_CONFIG_PATH):
        raise HTTPException(404, "Competition config not found")
    with open(_COMPETITION_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("competition", {})

    models = cfg.get("models", [])
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

    # Load monthly errors as DataFrame
    monthly_errors = _load_monthly_errors(models, lag_mode)
    if monthly_errors.empty:
        raise HTTPException(404, "No forecast data found for configured models")

    # Run strategy — all strategies enforce strict causality
    strat_kwargs: dict[str, Any] = {
        "min_prior_months": min_rows,
        **strategy_params,
    }
    winners_df = strategy_fn(monthly_errors, **strat_kwargs)

    if winners_df.empty:
        raise HTTPException(404, "No qualifying DFUs found with current config")

    is_ensemble = strategy_name == "ensemble"
    champion_acc = compute_strategy_accuracy(winners_df)

    # Bulk insert champion rows
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM fact_external_forecast_monthly WHERE model_id = %s",
            (champion_id,),
        )

        if is_ensemble:
            # Ensemble: insert blended forecasts directly
            cur.execute("""
                CREATE TEMP TABLE _champion_ensemble (
                    dmdunit TEXT NOT NULL,
                    dmdgroup TEXT NOT NULL,
                    loc TEXT NOT NULL,
                    startdate DATE NOT NULL,
                    basefcst_pref DOUBLE PRECISION NOT NULL,
                    tothist_dmd DOUBLE PRECISION NOT NULL
                ) ON COMMIT DROP
            """)

            buf = io.StringIO()
            for _, r in winners_df.iterrows():
                buf.write(
                    f"{r['dmdunit']}\t{r['dmdgroup']}\t{r['loc']}\t"
                    f"{r['startdate'].date()}\t{r['basefcst_pref']}\t"
                    f"{r['tothist_dmd']}\n"
                )
            buf.seek(0)
            with cur.copy("COPY _champion_ensemble FROM STDIN") as copy:
                copy.write(buf.read())

            # For ensemble: copy metadata (forecast_ck, lag, etc.) from any
            # competing model's row for the same DFU-month, then override
            # basefcst_pref with the blended value.
            cur.execute(f"""
                INSERT INTO fact_external_forecast_monthly
                    (forecast_ck, dmdunit, dmdgroup, loc, fcstdate, startdate,
                     lag, execution_lag, basefcst_pref, tothist_dmd, model_id)
                SELECT DISTINCT ON (e.dmdunit, e.dmdgroup, e.loc, e.startdate)
                    f.forecast_ck, f.dmdunit, f.dmdgroup, f.loc, f.fcstdate,
                    f.startdate, f.lag, f.execution_lag,
                    e.basefcst_pref, e.tothist_dmd,
                    %s
                FROM _champion_ensemble e
                INNER JOIN fact_external_forecast_monthly f
                    ON f.dmdunit = e.dmdunit
                   AND f.dmdgroup = e.dmdgroup
                   AND f.loc = e.loc
                   AND f.startdate = e.startdate
                   AND f.model_id IN ({",".join(["%s"] * len(models))})
                ORDER BY e.dmdunit, e.dmdgroup, e.loc, e.startdate, f.model_id
            """, [champion_id] + models)
        else:
            # Pick-one: copy winning model's rows per DFU per month
            cur.execute("""
                CREATE TEMP TABLE _champion_winners (
                    dmdunit TEXT NOT NULL,
                    dmdgroup TEXT NOT NULL,
                    loc TEXT NOT NULL,
                    startdate DATE NOT NULL,
                    winning_model_id TEXT NOT NULL
                ) ON COMMIT DROP
            """)

            buf = io.StringIO()
            for _, r in winners_df.iterrows():
                buf.write(
                    f"{r['dmdunit']}\t{r['dmdgroup']}\t{r['loc']}\t"
                    f"{r['startdate'].date()}\t{r['model_id']}\n"
                )
            buf.seek(0)
            with cur.copy("COPY _champion_winners FROM STDIN") as copy:
                copy.write(buf.read())

            cur.execute(
                """
                INSERT INTO fact_external_forecast_monthly
                    (forecast_ck, dmdunit, dmdgroup, loc, fcstdate, startdate,
                     lag, execution_lag, basefcst_pref, tothist_dmd, model_id)
                SELECT
                    f.forecast_ck, f.dmdunit, f.dmdgroup, f.loc, f.fcstdate,
                    f.startdate, f.lag, f.execution_lag, f.basefcst_pref,
                    f.tothist_dmd, %s
                FROM fact_external_forecast_monthly f
                INNER JOIN _champion_winners w
                    ON f.dmdunit = w.dmdunit
                   AND f.dmdgroup = w.dmdgroup
                   AND f.loc = w.loc
                   AND f.startdate = w.startdate
                   AND f.model_id = w.winning_model_id
                """,
                (champion_id,),
            )
        inserted = cur.rowcount
        conn.commit()

    # Compute ceiling (oracle)
    ceiling_id = cfg.get("ceiling_model_id", "ceiling")
    ceiling_df = compute_ceiling(monthly_errors)
    ceiling_acc = compute_strategy_accuracy(ceiling_df)
    ceiling_inserted = 0

    if not ceiling_df.empty:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM fact_external_forecast_monthly WHERE model_id = %s",
                (ceiling_id,),
            )
            cur.execute("""
                CREATE TEMP TABLE _ceiling_winners (
                    dmdunit TEXT NOT NULL,
                    dmdgroup TEXT NOT NULL,
                    loc TEXT NOT NULL,
                    startdate DATE NOT NULL,
                    winning_model_id TEXT NOT NULL
                ) ON COMMIT DROP
            """)

            buf2 = io.StringIO()
            for _, r in ceiling_df.iterrows():
                buf2.write(
                    f"{r['dmdunit']}\t{r['dmdgroup']}\t{r['loc']}\t"
                    f"{r['startdate'].date()}\t{r['model_id']}\n"
                )
            buf2.seek(0)
            with cur.copy("COPY _ceiling_winners FROM STDIN") as copy:
                copy.write(buf2.read())

            cur.execute(
                """
                INSERT INTO fact_external_forecast_monthly
                    (forecast_ck, dmdunit, dmdgroup, loc, fcstdate, startdate,
                     lag, execution_lag, basefcst_pref, tothist_dmd, model_id)
                SELECT
                    f.forecast_ck, f.dmdunit, f.dmdgroup, f.loc, f.fcstdate,
                    f.startdate, f.lag, f.execution_lag, f.basefcst_pref,
                    f.tothist_dmd, %s
                FROM fact_external_forecast_monthly f
                INNER JOIN _ceiling_winners w
                    ON f.dmdunit = w.dmdunit
                   AND f.dmdgroup = w.dmdgroup
                   AND f.loc = w.loc
                   AND f.startdate = w.startdate
                   AND f.model_id = w.winning_model_id
                """,
                (ceiling_id,),
            )
            ceiling_inserted = cur.rowcount
            conn.commit()

    # Refresh materialized views
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '512MB'")
        cur.execute("REFRESH MATERIALIZED VIEW agg_forecast_monthly")
        cur.execute("REFRESH MATERIALIZED VIEW agg_accuracy_by_dim")
        cur.execute("REFRESH MATERIALIZED VIEW agg_dfu_coverage")
        conn.commit()

    # Build summary
    model_wins: dict[str, int] = {}
    for mid in winners_df["model_id"]:
        model_wins[mid] = model_wins.get(mid, 0) + 1

    n_unique_dfus = winners_df[["dmdunit", "dmdgroup", "loc"]].drop_duplicates().shape[0]

    from datetime import datetime, timezone

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
        "model_wins": dict(sorted(model_wins.items(), key=lambda x: -x[1])),
        "overall_champion_wape": champion_acc.get("wape"),
        "overall_champion_accuracy_pct": champion_acc.get("accuracy_pct"),
        "run_ts": datetime.now(timezone.utc).isoformat(),
    }

    # Ceiling metrics
    if not ceiling_df.empty:
        ceil_wins: dict[str, int] = {}
        for mid in ceiling_df["model_id"]:
            ceil_wins[mid] = ceil_wins.get(mid, 0) + 1

        summary["total_ceiling_rows"] = ceiling_inserted
        summary["ceiling_model_wins"] = dict(sorted(ceil_wins.items(), key=lambda x: -x[1]))
        summary["overall_ceiling_wape"] = ceiling_acc.get("wape")
        summary["overall_ceiling_accuracy_pct"] = ceiling_acc.get("accuracy_pct")

    # Save summary to disk
    summary_dir = os.path.dirname(_CHAMPION_SUMMARY_PATH)
    os.makedirs(summary_dir, exist_ok=True)
    with open(_CHAMPION_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


@router.get("/competition/summary")
def get_competition_summary():
    """Return the last champion selection summary, if available."""
    if not os.path.exists(_CHAMPION_SUMMARY_PATH):
        return {"status": "not_run", "summary": None}
    with open(_CHAMPION_SUMMARY_PATH) as f:
        return {"status": "ok", "summary": json.load(f)}
