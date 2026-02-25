"""Champion / Model Competition endpoints (feature 15)."""
from __future__ import annotations

from typing import Any
import io
import json
import os

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


class CompetitionConfigUpdate(BaseModel):
    metric: str = "wape"
    lag: str = "execution"
    min_dfu_rows: int = 3
    champion_model_id: str = "champion"
    models: list[str]


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

    cfg = {
        "competition": {
            "name": "default",
            "metric": body.metric,
            "lag": body.lag,
            "min_dfu_rows": body.min_dfu_rows,
            "champion_model_id": body.champion_model_id,
            "models": body.models,
        }
    }
    os.makedirs(os.path.dirname(_COMPETITION_CONFIG_PATH), exist_ok=True)
    with open(_COMPETITION_CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return {"status": "ok", "config": cfg["competition"]}


@router.post("/competition/run", dependencies=[Depends(require_api_key)])
def run_competition():
    """Execute champion model selection and return summary."""
    import yaml
    from datetime import datetime, timezone

    if not os.path.exists(_COMPETITION_CONFIG_PATH):
        raise HTTPException(404, "Competition config not found")
    with open(_COMPETITION_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("competition", {})

    models = cfg.get("models", [])
    metric = cfg.get("metric", "wape")
    lag_mode = str(cfg.get("lag", "execution"))
    min_rows = int(cfg.get("min_dfu_rows", 3))
    champion_id = cfg.get("champion_model_id", "champion")

    if len(models) < 2:
        raise HTTPException(422, "At least 2 models required for competition")

    placeholders = ",".join(["%s"] * len(models))
    params: list[Any] = list(models)

    if lag_mode == "execution":
        lag_cond = "lag::text = execution_lag::text"
    else:
        lag_cond = "lag = %s"
        params.append(int(lag_mode))

    winner_sql = f"""
    WITH dfu_model_wape AS (
        SELECT
            dmdunit, dmdgroup, loc, model_id,
            SUM(ABS(basefcst_pref - tothist_dmd))
                / NULLIF(ABS(SUM(tothist_dmd)), 0) AS wape,
            COUNT(*) AS n_rows
        FROM fact_external_forecast_monthly
        WHERE model_id IN ({placeholders})
          AND {lag_cond}
          AND basefcst_pref IS NOT NULL
          AND tothist_dmd IS NOT NULL
        GROUP BY dmdunit, dmdgroup, loc, model_id
        HAVING COUNT(*) >= %s
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY dmdunit, dmdgroup, loc
                ORDER BY wape ASC NULLS LAST
            ) AS rn
        FROM dfu_model_wape
        WHERE wape IS NOT NULL
    )
    SELECT dmdunit, dmdgroup, loc, model_id, wape, n_rows
    FROM ranked
    WHERE rn = 1
    ORDER BY dmdunit, dmdgroup, loc
    """
    params.append(min_rows)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(winner_sql, params)
        winners = cur.fetchall()

    if not winners:
        raise HTTPException(404, "No qualifying DFUs found with current config")

    # Bulk insert champion rows
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM fact_external_forecast_monthly WHERE model_id = %s",
            (champion_id,),
        )

        cur.execute("""
            CREATE TEMP TABLE _champion_winners (
                dmdunit TEXT NOT NULL,
                dmdgroup TEXT NOT NULL,
                loc TEXT NOT NULL,
                winning_model_id TEXT NOT NULL
            ) ON COMMIT DROP
        """)

        buf = io.StringIO()
        for dmdunit, dmdgroup, loc, model_id, _wape, _n in winners:
            buf.write(f"{dmdunit}\t{dmdgroup}\t{loc}\t{model_id}\n")
        buf.seek(0)
        with cur.copy("COPY _champion_winners FROM STDIN") as copy:
            copy.write(buf.read())

        cur.execute(
            """
            INSERT INTO fact_external_forecast_monthly
                (forecast_ck, dmdunit, dmdgroup, loc, fcstdate, startdate,
                 lag, execution_lag, basefcst_pref, tothist_dmd, model_id)
            SELECT
                f.forecast_ck, f.dmdunit, f.dmdgroup, f.loc, f.fcstdate, f.startdate,
                f.lag, f.execution_lag, f.basefcst_pref, f.tothist_dmd,
                %s
            FROM fact_external_forecast_monthly f
            INNER JOIN _champion_winners w
                ON f.dmdunit = w.dmdunit
               AND f.dmdgroup = w.dmdgroup
               AND f.loc = w.loc
               AND f.model_id = w.winning_model_id
            """,
            (champion_id,),
        )
        inserted = cur.rowcount
        conn.commit()

    # Compute ceiling (oracle)
    ceiling_id = cfg.get("ceiling_model_id", "ceiling")
    ceil_placeholders = ",".join(["%s"] * len(models))
    ceil_params: list[Any] = list(models)
    if lag_mode == "execution":
        ceil_lag_cond = "lag::text = execution_lag::text"
    else:
        ceil_lag_cond = "lag = %s"
        ceil_params.append(int(lag_mode))

    ceiling_sql = f"""
    WITH monthly_ranked AS (
        SELECT
            dmdunit, dmdgroup, loc, startdate, model_id,
            ABS(basefcst_pref - tothist_dmd) AS abs_err,
            basefcst_pref, tothist_dmd,
            ROW_NUMBER() OVER (
                PARTITION BY dmdunit, dmdgroup, loc, startdate
                ORDER BY ABS(basefcst_pref - tothist_dmd) ASC NULLS LAST
            ) AS rn
        FROM fact_external_forecast_monthly
        WHERE model_id IN ({ceil_placeholders})
          AND {ceil_lag_cond}
          AND basefcst_pref IS NOT NULL
          AND tothist_dmd IS NOT NULL
    )
    SELECT dmdunit, dmdgroup, loc, startdate, model_id,
           abs_err, basefcst_pref, tothist_dmd
    FROM monthly_ranked
    WHERE rn = 1
    ORDER BY dmdunit, dmdgroup, loc, startdate
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(ceiling_sql, ceil_params)
        ceiling_rows = cur.fetchall()

    # Bulk insert ceiling rows
    ceiling_inserted = 0
    if ceiling_rows:
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
            for dmdunit, dmdgroup, loc, startdate, model_id, *_ in ceiling_rows:
                buf2.write(f"{dmdunit}\t{dmdgroup}\t{loc}\t{startdate}\t{model_id}\n")
            buf2.seek(0)
            with cur.copy("COPY _ceiling_winners FROM STDIN") as copy:
                copy.write(buf2.read())

            cur.execute(
                """
                INSERT INTO fact_external_forecast_monthly
                    (forecast_ck, dmdunit, dmdgroup, loc, fcstdate, startdate,
                     lag, execution_lag, basefcst_pref, tothist_dmd, model_id)
                SELECT
                    f.forecast_ck, f.dmdunit, f.dmdgroup, f.loc, f.fcstdate, f.startdate,
                    f.lag, f.execution_lag, f.basefcst_pref, f.tothist_dmd,
                    %s
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

    # Build summary (champion)
    model_wins: dict[str, int] = {}
    total_wape_num = 0.0
    total_wape_denom = 0.0
    for _u, _g, _l, mid, wape, n_rows in winners:
        model_wins[mid] = model_wins.get(mid, 0) + 1
        if wape is not None:
            total_wape_num += float(wape) * float(n_rows)
            total_wape_denom += float(n_rows)

    overall_wape = (total_wape_num / total_wape_denom * 100) if total_wape_denom else None
    overall_acc = (100.0 - overall_wape) if overall_wape is not None else None

    from datetime import datetime, timezone

    summary: dict[str, Any] = {
        "config": {
            "metric": metric,
            "lag": lag_mode,
            "min_dfu_rows": min_rows,
            "champion_model_id": champion_id,
            "models": models,
        },
        "total_dfus": len(winners),
        "total_champion_rows": inserted,
        "model_wins": dict(sorted(model_wins.items(), key=lambda x: -x[1])),
        "overall_champion_wape": round(overall_wape, 4) if overall_wape is not None else None,
        "overall_champion_accuracy_pct": round(overall_acc, 4) if overall_acc is not None else None,
        "run_ts": datetime.now(timezone.utc).isoformat(),
    }

    # Ceiling metrics
    if ceiling_rows:
        ceil_wins: dict[str, int] = {}
        ceil_abs_err_sum = 0.0
        ceil_actual_sum = 0.0
        for _u, _g, _l, _sd, mid, abs_err, _fcst, actual in ceiling_rows:
            ceil_wins[mid] = ceil_wins.get(mid, 0) + 1
            ceil_abs_err_sum += float(abs_err)
            ceil_actual_sum += abs(float(actual))

        ceil_wape = (ceil_abs_err_sum / ceil_actual_sum * 100) if ceil_actual_sum else None
        ceil_acc = (100.0 - ceil_wape) if ceil_wape is not None else None

        summary["total_ceiling_rows"] = ceiling_inserted
        summary["ceiling_model_wins"] = dict(sorted(ceil_wins.items(), key=lambda x: -x[1]))
        summary["overall_ceiling_wape"] = round(ceil_wape, 4) if ceil_wape is not None else None
        summary["overall_ceiling_accuracy_pct"] = round(ceil_acc, 4) if ceil_acc is not None else None

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
