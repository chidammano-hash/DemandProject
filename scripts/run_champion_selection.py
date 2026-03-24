"""
Champion model selection: rolling/expanding window best-model per DFU per month.

For each DFU (item_id + customer_group + loc) at each month, evaluates competing
models using cumulative WAPE from **prior months only** (before-the-fact)
and selects the best model for that month.  The selected model's forecast
rows are copied into fact_external_forecast_monthly with
model_id = '<champion_model_id>' (default: 'champion').

The ceiling (oracle) picks the best model per DFU per month using that
month's actual error — the theoretical upper bound with perfect foresight.

Usage:
    python scripts/run_champion_selection.py [--config config/model_competition.yaml]
"""

import argparse
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.champion_strategies import STRATEGY_REGISTRY
from common.db import get_db_params
from common.services.perf_profiler import profiled_section


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "metric": "wape",
    "lag": "execution",
    "min_dfu_rows": 3,
    "champion_model_id": "champion",
    "fallback_model_id": "lgbm_cluster",
    "models": [],
    "strategy": "expanding",
    "strategy_params": {},
}

_VALID_STRATEGIES = {"expanding", "rolling", "decay", "ensemble", "meta_learner"}


def load_config(config_path: Path) -> dict[str, Any]:
    """Read and validate the competition config YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = raw.get("competition", {})
    for k, v in _DEFAULTS.items():
        cfg.setdefault(k, v)
    # Validate
    if cfg["metric"] not in ("wape", "accuracy_pct"):
        raise ValueError(f"Invalid metric '{cfg['metric']}'; must be wape or accuracy_pct")
    valid_lags = {"execution", "0", "1", "2", "3", "4"}
    if str(cfg["lag"]) not in valid_lags:
        raise ValueError(f"Invalid lag '{cfg['lag']}'; must be one of {sorted(valid_lags)}")
    if len(cfg["models"]) < 2:
        raise ValueError("At least 2 models required for competition")
    if cfg["strategy"] not in _VALID_STRATEGIES:
        raise ValueError(
            f"Invalid strategy '{cfg['strategy']}'; "
            f"must be one of {sorted(_VALID_STRATEGIES)}"
        )
    return cfg


# ---------------------------------------------------------------------------
# DataFrame-based data loading (for strategy registry)
# ---------------------------------------------------------------------------

def load_monthly_errors_df(
    db: dict[str, Any],
    models: list[str],
    lag_mode: str,
) -> pd.DataFrame:
    """Load per-DFU per-month per-model errors as a DataFrame.

    Includes fcstdate and execution_lag so strategies can compute the
    correct causal prior window (startdate < fcstdate = T - exec_lag).
    """
    placeholders = ",".join(["%s"] * len(models))
    params: list[Any] = list(models)

    if lag_mode == "execution":
        lag_cond = "lag::text = execution_lag::text"
    else:
        lag_cond = "lag = %s"
        params.append(int(lag_mode))

    sql = f"""
        SELECT item_id, customer_group, loc, startdate, fcstdate, execution_lag,
               model_id, basefcst_pref, tothist_dmd,
               ABS(basefcst_pref - tothist_dmd) AS abs_err
        FROM fact_external_forecast_monthly
        WHERE model_id IN ({placeholders})
          AND {lag_cond}
          AND basefcst_pref IS NOT NULL
          AND tothist_dmd IS NOT NULL
        ORDER BY item_id, customer_group, loc, model_id, startdate
    """
    with psycopg.connect(**db) as conn:
        df = pd.read_sql(sql, conn, params=params)
    df["startdate"] = pd.to_datetime(df["startdate"])
    df["fcstdate"] = pd.to_datetime(df["fcstdate"])
    for col in ["basefcst_pref", "tothist_dmd", "abs_err"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["execution_lag"] = pd.to_numeric(df["execution_lag"], errors="coerce").fillna(0).astype(int)
    return df


def load_dfu_features(db: dict[str, Any]) -> pd.DataFrame:
    """Load DFU static features for meta-learner strategy."""
    sql = """
        SELECT item_id, customer_group, loc,
               ml_cluster, abc_vol, execution_lag, total_lt,
               brand, region,
               seasonality_profile, seasonality_strength,
               is_yearly_seasonal, peak_month, trough_month,
               peak_trough_ratio
        FROM dim_sku
    """
    with psycopg.connect(**db) as conn:
        df = pd.read_sql(sql, conn)
    for col in ["ml_cluster", "abc_vol", "brand", "region", "seasonality_profile"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


# ---------------------------------------------------------------------------
# Champion selection logic — rolling/expanding window (before-the-fact)
# ---------------------------------------------------------------------------

def compute_champion_winners(
    cur: psycopg.Cursor,
    models: list[str],
    lag_mode: str,
    min_rows: int,
) -> list[tuple[str, str, str, str, str, float, float, float]]:
    """Return per-DFU per-month champion winners using expanding window.

    For each (item_id, customer_group, loc, startdate), picks the model with
    lowest cumulative WAPE computed from all **prior** months only
    (before-the-fact selection).  Requires at least ``min_rows`` prior
    months of history before a model can be selected.

    Returns (item_id, customer_group, loc, startdate, model_id, prior_wape,
             basefcst_pref, tothist_dmd).
    """
    placeholders = ",".join(["%s"] * len(models))

    # Lag filter condition
    if lag_mode == "execution":
        lag_cond = "lag::text = execution_lag::text"
        params: list[Any] = list(models)
    else:
        lag_cond = "lag = %s"
        params = list(models) + [int(lag_mode)]

    # fcstdate = startdate - execution_lag months = the calendar date the
    # forecast was issued.  Prior data for month T = rows where
    # startdate < fcstdate(T), i.e. actuals that existed at issuance time.
    sql = f"""
    WITH monthly_errors AS (
        SELECT
            item_id, customer_group, loc, startdate, model_id,
            basefcst_pref, tothist_dmd, execution_lag,
            fcstdate,
            ABS(basefcst_pref - tothist_dmd) AS abs_err
        FROM fact_external_forecast_monthly
        WHERE model_id IN ({placeholders})
          AND {lag_cond}
          AND basefcst_pref IS NOT NULL
          AND tothist_dmd IS NOT NULL
    ),
    cumulative AS (
        -- Self-join: for each row A at issuance date fcstdate(A),
        -- sum errors from rows B whose startdate < fcstdate(A).
        -- This enforces exec-lag causality: excludes the last exec_lag
        -- months that weren't available when the forecast was issued.
        SELECT
            a.item_id, a.customer_group, a.loc, a.startdate, a.model_id,
            a.basefcst_pref, a.tothist_dmd,
            SUM(b.abs_err)     AS cum_abs_err,
            SUM(b.tothist_dmd) AS cum_actual,
            COUNT(b.startdate) AS prior_months
        FROM monthly_errors a
        LEFT JOIN monthly_errors b
            ON  a.item_id  = b.item_id
            AND a.customer_group = b.customer_group
            AND a.loc      = b.loc
            AND a.model_id = b.model_id
            AND b.startdate < a.fcstdate   -- causal cutoff: prior to issuance
        GROUP BY
            a.item_id, a.customer_group, a.loc, a.startdate, a.model_id,
            a.basefcst_pref, a.tothist_dmd
    ),
    with_wape AS (
        SELECT *,
            cum_abs_err / NULLIF(ABS(cum_actual), 0) AS prior_wape
        FROM cumulative
        WHERE prior_months >= %s
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY item_id, customer_group, loc, startdate
                ORDER BY prior_wape ASC NULLS LAST
            ) AS rn
        FROM with_wape
        WHERE prior_wape IS NOT NULL
    )
    SELECT item_id, customer_group, loc, startdate, model_id,
           prior_wape, basefcst_pref, tothist_dmd
    FROM ranked
    WHERE rn = 1
    ORDER BY item_id, customer_group, loc, startdate
    """
    params.append(min_rows)
    cur.execute(sql, params)
    return cur.fetchall()


# ---------------------------------------------------------------------------
# Ceiling selection logic — oracle / perfect foresight (after-the-fact)
# ---------------------------------------------------------------------------

def compute_ceiling_winners(
    cur: psycopg.Cursor,
    models: list[str],
    lag_mode: str,
) -> list[tuple[str, str, str, str, str, float, float, float]]:
    """Return per-DFU per-month oracle winners.

    For each (item_id, customer_group, loc, startdate), picks the model with
    lowest absolute error — the theoretical best if you always knew which
    model would be most accurate for every single month.

    Returns (item_id, customer_group, loc, startdate, model_id, abs_err,
             basefcst_pref, tothist_dmd).
    """
    placeholders = ",".join(["%s"] * len(models))

    if lag_mode == "execution":
        lag_cond = "lag::text = execution_lag::text"
        params: list[Any] = list(models)
    else:
        lag_cond = "lag = %s"
        params = list(models) + [int(lag_mode)]

    sql = f"""
    WITH monthly_ranked AS (
        SELECT
            item_id, customer_group, loc, startdate, model_id,
            ABS(basefcst_pref - tothist_dmd) AS abs_err,
            basefcst_pref, tothist_dmd,
            ROW_NUMBER() OVER (
                PARTITION BY item_id, customer_group, loc, startdate
                ORDER BY ABS(basefcst_pref - tothist_dmd) ASC NULLS LAST
            ) AS rn
        FROM fact_external_forecast_monthly
        WHERE model_id IN ({placeholders})
          AND {lag_cond}
          AND basefcst_pref IS NOT NULL
          AND tothist_dmd IS NOT NULL
    )
    SELECT item_id, customer_group, loc, startdate, model_id,
           abs_err, basefcst_pref, tothist_dmd
    FROM monthly_ranked
    WHERE rn = 1
    ORDER BY item_id, customer_group, loc, startdate
    """
    cur.execute(sql, params)
    return cur.fetchall()


def insert_ceiling_forecasts(
    cur: psycopg.Cursor,
    ceiling_rows: list[tuple[str, str, str, str, str, float, float, float]],
    ceiling_model_id: str,
) -> int:
    """Bulk-insert ceiling (oracle) forecast rows.

    Uses temp table + INSERT ... SELECT.  The ceiling picks the best model
    per DFU per month, so the temp table key includes startdate.
    Returns number of rows inserted.
    """
    if not ceiling_rows:
        return 0

    # 1. Delete existing ceiling rows
    cur.execute(
        "DELETE FROM fact_external_forecast_monthly WHERE model_id = %s",
        (ceiling_model_id,),
    )
    deleted = cur.rowcount
    print(f"  Deleted {deleted:,} existing ceiling rows")

    # 2. Create temp table with DFU+month -> winning model mapping
    cur.execute("""
        CREATE TEMP TABLE _ceiling_winners (
            item_id TEXT NOT NULL,
            customer_group TEXT NOT NULL,
            loc TEXT NOT NULL,
            startdate DATE NOT NULL,
            winning_model_id TEXT NOT NULL
        ) ON COMMIT DROP
    """)

    # 3. COPY ceiling winners into temp table
    buf = io.StringIO()
    for item_id, customer_group, loc, startdate, model_id, *_ in ceiling_rows:
        buf.write(f"{item_id}\t{customer_group}\t{loc}\t{startdate}\t{model_id}\n")
    buf.seek(0)
    with cur.copy("COPY _ceiling_winners FROM STDIN") as copy:
        copy.write(buf.read())

    # 4. Bulk INSERT ... SELECT ceiling rows
    cur.execute(
        """
        INSERT INTO fact_external_forecast_monthly
            (forecast_ck, item_id, customer_group, loc, fcstdate, startdate,
             lag, execution_lag, basefcst_pref, tothist_dmd, model_id)
        SELECT
            f.forecast_ck, f.item_id, f.customer_group, f.loc, f.fcstdate, f.startdate,
            f.lag, f.execution_lag, f.basefcst_pref, f.tothist_dmd,
            %s
        FROM fact_external_forecast_monthly f
        INNER JOIN _ceiling_winners w
            ON f.item_id = w.item_id
           AND f.customer_group = w.customer_group
           AND f.loc = w.loc
           AND f.startdate = w.startdate
           AND f.model_id = w.winning_model_id
        """,
        (ceiling_model_id,),
    )
    inserted = cur.rowcount
    return inserted


def insert_champion_forecasts(
    cur: psycopg.Cursor,
    winners: list[tuple[str, str, str, str, str, float, float, float]],
    champion_model_id: str,
) -> int:
    """Bulk-insert champion forecast rows using rolling window winners.

    Uses temp table + INSERT ... SELECT.  The champion picks the best model
    per DFU per month (before-the-fact), so the temp table key includes
    startdate.  Returns number of rows inserted.
    """
    if not winners:
        return 0

    # 1. Delete existing champion rows
    cur.execute(
        "DELETE FROM fact_external_forecast_monthly WHERE model_id = %s",
        (champion_model_id,),
    )
    deleted = cur.rowcount
    print(f"  Deleted {deleted:,} existing champion rows")

    # 2. Create temp table with DFU+month -> winning model mapping
    cur.execute("""
        CREATE TEMP TABLE _champion_winners (
            item_id TEXT NOT NULL,
            customer_group TEXT NOT NULL,
            loc TEXT NOT NULL,
            startdate DATE NOT NULL,
            winning_model_id TEXT NOT NULL
        ) ON COMMIT DROP
    """)

    # 3. COPY champion winners into temp table
    buf = io.StringIO()
    for item_id, customer_group, loc, startdate, model_id, *_ in winners:
        buf.write(f"{item_id}\t{customer_group}\t{loc}\t{startdate}\t{model_id}\n")
    buf.seek(0)
    with cur.copy("COPY _champion_winners FROM STDIN") as copy:
        copy.write(buf.read())

    # 4. Bulk INSERT ... SELECT champion rows
    # source_model_id records which underlying algorithm won (e.g. lgbm_cluster)
    # so production forecast can load the right .pkl artifact per DFU.
    cur.execute(
        """
        INSERT INTO fact_external_forecast_monthly
            (forecast_ck, item_id, customer_group, loc, fcstdate, startdate,
             lag, execution_lag, basefcst_pref, tothist_dmd, model_id, source_model_id)
        SELECT
            f.forecast_ck, f.item_id, f.customer_group, f.loc, f.fcstdate, f.startdate,
            f.lag, f.execution_lag, f.basefcst_pref, f.tothist_dmd,
            %s, w.winning_model_id
        FROM fact_external_forecast_monthly f
        INNER JOIN _champion_winners w
            ON f.item_id = w.item_id
           AND f.customer_group = w.customer_group
           AND f.loc = w.loc
           AND f.startdate = w.startdate
           AND f.model_id = w.winning_model_id
        """,
        (champion_model_id,),
    )
    inserted = cur.rowcount
    return inserted


def insert_ensemble_forecasts(
    cur: psycopg.Cursor,
    winners_df: pd.DataFrame,
    champion_model_id: str,
) -> int:
    """Bulk-insert ensemble (blended) forecast rows.

    The ensemble strategy produces synthetic blended forecast values.
    We find a reference model row for each DFU-month and override
    basefcst_pref with the blended value.
    """
    if winners_df.empty:
        return 0

    # 1. Delete existing champion rows
    cur.execute(
        "DELETE FROM fact_external_forecast_monthly WHERE model_id = %s",
        (champion_model_id,),
    )
    deleted = cur.rowcount
    print(f"  Deleted {deleted:,} existing champion rows")

    # 2. Create temp table with blended forecast values
    cur.execute("""
        CREATE TEMP TABLE _ensemble_winners (
            item_id TEXT NOT NULL,
            customer_group TEXT NOT NULL,
            loc TEXT NOT NULL,
            startdate DATE NOT NULL,
            blended_fcst DOUBLE PRECISION NOT NULL
        ) ON COMMIT DROP
    """)

    # 3. COPY ensemble data
    buf = io.StringIO()
    for _, row in winners_df.iterrows():
        buf.write(
            f"{row['item_id']}\t{row['customer_group']}\t{row['loc']}\t"
            f"{row['startdate'].date()}\t{row['basefcst_pref']}\n"
        )
    buf.seek(0)
    with cur.copy("COPY _ensemble_winners FROM STDIN") as copy:
        copy.write(buf.read())

    # 4. Insert using a reference model row for metadata, override basefcst_pref
    cur.execute(
        """
        INSERT INTO fact_external_forecast_monthly
            (forecast_ck, item_id, customer_group, loc, fcstdate, startdate,
             lag, execution_lag, basefcst_pref, tothist_dmd, model_id)
        SELECT DISTINCT ON (f.item_id, f.customer_group, f.loc, f.startdate)
            f.forecast_ck, f.item_id, f.customer_group, f.loc, f.fcstdate, f.startdate,
            f.lag, f.execution_lag, w.blended_fcst, f.tothist_dmd,
            %s
        FROM fact_external_forecast_monthly f
        INNER JOIN _ensemble_winners w
            ON f.item_id = w.item_id
           AND f.customer_group = w.customer_group
           AND f.loc = w.loc
           AND f.startdate = w.startdate
        WHERE f.model_id != %s AND f.model_id != 'ceiling'
        ORDER BY f.item_id, f.customer_group, f.loc, f.startdate, f.model_id
        """,
        (champion_model_id, champion_model_id),
    )
    inserted = cur.rowcount
    return inserted


def insert_fallback_champions(
    cur: psycopg.Cursor,
    lag_mode: str,
    champion_model_id: str,
    fallback_model_id: str,
) -> int:
    """Insert champion rows using the fallback model for DFU-months without a champion.

    During the warm-up period — the first (exec_lag + min_dfu_rows) months per DFU —
    no strategy can pick a winner because there is insufficient prior history.
    This function fills those gaps with the fallback model's forecast (default:
    lgbm_cluster), so every DFU-month that has a backtest row always has a champion.

    The insert is idempotent: only fills DFU-months where no champion row already
    exists (NOT EXISTS sub-select + ON CONFLICT DO NOTHING).

    Returns the number of fallback rows inserted.
    """
    if lag_mode == "execution":
        lag_cond = "lag::text = execution_lag::text"
        params: list[Any] = [champion_model_id, fallback_model_id, champion_model_id]
    else:
        lag_cond = "lag = %s"
        params = [champion_model_id, fallback_model_id, int(lag_mode), champion_model_id]

    sql = f"""
    INSERT INTO fact_external_forecast_monthly
        (forecast_ck, item_id, customer_group, loc, fcstdate, startdate,
         lag, execution_lag, basefcst_pref, tothist_dmd, model_id)
    SELECT
        f.forecast_ck, f.item_id, f.customer_group, f.loc, f.fcstdate, f.startdate,
        f.lag, f.execution_lag, f.basefcst_pref, f.tothist_dmd,
        %s
    FROM fact_external_forecast_monthly f
    WHERE f.model_id = %s
      AND {lag_cond}
      AND f.basefcst_pref IS NOT NULL
      AND f.tothist_dmd IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM fact_external_forecast_monthly c
          WHERE c.model_id = %s
            AND c.item_id    = f.item_id
            AND c.customer_group   = f.customer_group
            AND c.loc        = f.loc
            AND c.startdate  = f.startdate
      )
    ON CONFLICT (forecast_ck, model_id) DO NOTHING
    """
    cur.execute(sql, params)
    return cur.rowcount


def generate_summary(
    winners: list[tuple[str, str, str, str, str, float, float, float]],
    champion_model_id: str,
    total_rows: int,
    config: dict[str, Any],
    ceiling_rows: list[tuple] | None = None,
    ceiling_inserted: int = 0,
    fallback_inserted: int = 0,
) -> dict[str, Any]:
    """Produce a summary dict from the winner list + optional ceiling data."""
    model_wins: dict[str, int] = {}
    champ_abs_err_sum = 0.0
    champ_actual_sum = 0.0
    unique_dfus: set[tuple[str, str, str]] = set()

    for item_id, customer_group, loc, _startdate, model_id, _prior_wape, basefcst_pref, tothist_dmd in winners:
        model_wins[model_id] = model_wins.get(model_id, 0) + 1
        champ_abs_err_sum += abs(float(basefcst_pref) - float(tothist_dmd))
        champ_actual_sum += float(tothist_dmd)
        unique_dfus.add((item_id, customer_group, loc))

    overall_wape = (champ_abs_err_sum / abs(champ_actual_sum) * 100) if champ_actual_sum else None
    overall_acc = (100.0 - overall_wape) if overall_wape is not None else None

    # Sort model wins descending
    sorted_wins = dict(sorted(model_wins.items(), key=lambda x: -x[1]))

    summary: dict[str, Any] = {
        "config": {
            "metric": config.get("metric"),
            "lag": config.get("lag"),
            "min_dfu_rows": config.get("min_dfu_rows"),
            "champion_model_id": champion_model_id,
            "fallback_model_id": config.get("fallback_model_id"),
            "models": config.get("models", []),
        },
        "total_dfus": len(unique_dfus),
        "total_dfu_months": len(winners),
        "total_champion_rows": total_rows,
        "fallback_rows_inserted": fallback_inserted,
        "model_wins": sorted_wins,
        "overall_champion_wape": round(overall_wape, 4) if overall_wape is not None else None,
        "overall_champion_accuracy_pct": round(overall_acc, 4) if overall_acc is not None else None,
        "run_ts": datetime.now(timezone.utc).isoformat(),
    }

    # Ceiling (oracle) metrics — per-DFU per-month best model
    if ceiling_rows:
        ceil_wins: dict[str, int] = {}
        ceil_abs_err_sum = 0.0
        ceil_actual_sum = 0.0
        for _u, _g, _l, _sd, mid, abs_err, _fcst, actual in ceiling_rows:
            ceil_wins[mid] = ceil_wins.get(mid, 0) + 1
            ceil_abs_err_sum += float(abs_err)
            ceil_actual_sum += float(actual)

        ceil_wape = (ceil_abs_err_sum / abs(ceil_actual_sum) * 100) if ceil_actual_sum else None
        ceil_acc = (100.0 - ceil_wape) if ceil_wape is not None else None
        sorted_ceil = dict(sorted(ceil_wins.items(), key=lambda x: -x[1]))

        summary["total_ceiling_rows"] = ceiling_inserted
        summary["ceiling_model_wins"] = sorted_ceil
        summary["overall_ceiling_wape"] = round(ceil_wape, 4) if ceil_wape is not None else None
        summary["overall_ceiling_accuracy_pct"] = round(ceil_acc, 4) if ceil_acc is not None else None

    return summary


# ---------------------------------------------------------------------------
# Materialized view refresh
# ---------------------------------------------------------------------------

def refresh_views(db_params: dict[str, Any]) -> None:
    """Refresh accuracy materialized views."""
    print("Refreshing materialized views...")
    t0 = time.time()
    with psycopg.connect(**db_params) as conn, conn.cursor() as cur:
        cur.execute("SET maintenance_work_mem = '512MB'")
        cur.execute("REFRESH MATERIALIZED VIEW agg_forecast_monthly")
        print(f"  agg_forecast_monthly ({time.time() - t0:.1f}s)")
        t1 = time.time()
        cur.execute("REFRESH MATERIALIZED VIEW agg_accuracy_by_dim")
        print(f"  agg_accuracy_by_dim ({time.time() - t1:.1f}s)")
        t2 = time.time()
        cur.execute("REFRESH MATERIALIZED VIEW agg_dfu_coverage")
        print(f"  agg_dfu_coverage ({time.time() - t2:.1f}s)")
        conn.commit()
    print(f"  All views refreshed in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Champion model selection")
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_competition.yaml",
        help="Path to competition config YAML",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Selection strategy (overrides config): "
             "expanding, rolling, decay, ensemble, meta_learner",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    db = get_db_params()

    config_path = ROOT / args.config
    cfg = load_config(config_path)

    models = cfg["models"]
    lag_mode = str(cfg["lag"])
    min_rows = int(cfg["min_dfu_rows"])
    champion_id = cfg["champion_model_id"]
    fallback_id = cfg.get("fallback_model_id", "lgbm_cluster")
    strategy_name = args.strategy or cfg.get("strategy", "expanding")
    strategy_params = cfg.get("strategy_params", {})

    print(f"Champion Selection ({strategy_name}) — {len(models)} competing models")
    print(f"  Lag: {lag_mode}  |  Min prior months: {min_rows}")
    print(f"  Models: {', '.join(models)}")
    print(f"  Champion model_id: '{champion_id}'")
    print()

    # Step 1: Run strategy to compute champion winners
    t0 = time.time()
    strategy_fn = STRATEGY_REGISTRY.get(strategy_name)

    with profiled_section("compute_champion_winners"):
        if strategy_fn:
            # Use DataFrame-based strategy from registry
            print(f"Loading monthly errors for strategy '{strategy_name}'...")
            monthly_errors_df = load_monthly_errors_df(db, models, lag_mode)
            print(f"  {len(monthly_errors_df):,} rows loaded")

            # Build kwargs
            strat_kwargs = {
                "min_prior_months": min_rows,
                **strategy_params,
            }
            if strategy_name == "meta_learner":
                strat_kwargs["dfu_features"] = load_dfu_features(db)
                strat_kwargs.setdefault(
                    "meta_model_path",
                    str(ROOT / "data" / "champion" / "meta_learner.joblib"),
                )

            print(f"Computing champion winners ({strategy_name}, prior months only)...")
            winners_df = strategy_fn(monthly_errors_df, **strat_kwargs)

            # Convert to tuple list for existing insert/summary logic
            winners = list(winners_df.itertuples(index=False, name=None))
            is_ensemble = strategy_name == "ensemble"
        else:
            # Fallback to SQL-based expanding window
            print("Computing per-DFU per-month champion (expanding window, prior months only)...")
            with psycopg.connect(**db) as conn, conn.cursor() as cur:
                winners = compute_champion_winners(cur, models, lag_mode, min_rows)
            winners_df = pd.DataFrame()
            is_ensemble = False

    elapsed = time.time() - t0
    unique_dfus = len({(w[0], w[1], w[2]) for w in winners})
    print(f"  {len(winners):,} DFU-month selections across {unique_dfus:,} DFUs ({elapsed:.1f}s)")

    if not winners:
        print("No qualifying DFU-months found. Aborting.")
        sys.exit(1)

    # Print top model wins
    wins_count: dict[str, int] = {}
    for w in winners:
        mid = w[4]
        wins_count[mid] = wins_count.get(mid, 0) + 1
    for mid, cnt in sorted(wins_count.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / len(winners)
        print(f"    {mid:<25s} {cnt:>6,} DFU-months ({pct:.1f}%)")
    print()

    # Step 2: Insert champion rows
    print("Inserting champion forecast rows...")
    t1 = time.time()
    with profiled_section("insert_champion_rows"):
        with psycopg.connect(**db) as conn, conn.cursor() as cur:
            if is_ensemble and not winners_df.empty:
                inserted = insert_ensemble_forecasts(cur, winners_df, champion_id)
            else:
                inserted = insert_champion_forecasts(cur, winners, champion_id)
            conn.commit()
    print(f"  Inserted {inserted:,} champion rows ({time.time() - t1:.1f}s)")

    # Step 2b: Fill warm-up gaps with fallback model
    fallback_inserted = 0
    if fallback_id:
        print(f"Filling warm-up gaps with fallback model '{fallback_id}'...")
        t1b = time.time()
        with profiled_section("insert_fallback_rows"):
            with psycopg.connect(**db) as conn, conn.cursor() as cur:
                fallback_inserted = insert_fallback_champions(
                    cur, lag_mode, champion_id, fallback_id,
                )
                conn.commit()
        print(f"  Inserted {fallback_inserted:,} fallback rows ({time.time() - t1b:.1f}s)")
    print()

    # Step 3: Compute ceiling (oracle) — best model per DFU per month
    ceiling_id = cfg.get("ceiling_model_id", "ceiling")
    print("Computing ceiling (oracle) — best model per DFU per month (after-the-fact)...")
    t2 = time.time()
    with profiled_section("compute_ceiling_winners"):
        with psycopg.connect(**db) as conn, conn.cursor() as cur:
            ceiling_rows = compute_ceiling_winners(cur, models, lag_mode)
    print(f"  {len(ceiling_rows):,} DFU-month rows evaluated ({time.time() - t2:.1f}s)")

    # Print ceiling model wins
    if ceiling_rows:
        ceil_count: dict[str, int] = {}
        for _, _, _, _, mid, *_ in ceiling_rows:
            ceil_count[mid] = ceil_count.get(mid, 0) + 1
        for mid, cnt in sorted(ceil_count.items(), key=lambda x: -x[1]):
            pct = 100.0 * cnt / len(ceiling_rows)
            print(f"    {mid:<25s} {cnt:>6,} months ({pct:.1f}%)")
        print()

    # Step 4: Insert ceiling rows
    print("Inserting ceiling forecast rows...")
    t3 = time.time()
    with profiled_section("insert_ceiling_rows"):
        with psycopg.connect(**db) as conn, conn.cursor() as cur:
            ceiling_inserted = insert_ceiling_forecasts(cur, ceiling_rows, ceiling_id)
            conn.commit()
    print(f"  Inserted {ceiling_inserted:,} ceiling rows ({time.time() - t3:.1f}s)")
    print()

    # Step 5: Refresh materialized views
    with profiled_section("refresh_materialized_views"):
        refresh_views(db)
    print()

    # Step 6: Save summary JSON
    with profiled_section("save_summary"):
        summary = generate_summary(
            winners, champion_id, inserted, cfg,
            ceiling_rows=ceiling_rows, ceiling_inserted=ceiling_inserted,
            fallback_inserted=fallback_inserted,
        )
        summary_dir = ROOT / "data" / "champion"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "champion_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")
    print(f"  Champion: {unique_dfus:,} DFUs, {len(winners):,} DFU-months")
    print(f"  Fallback rows (warm-up gaps): {fallback_inserted:,}")
    print(f"  Overall champion accuracy: {summary['overall_champion_accuracy_pct']}%")
    print(f"  Overall champion WAPE: {summary['overall_champion_wape']}%")
    if summary.get("overall_ceiling_accuracy_pct") is not None:
        print(f"  Overall ceiling accuracy: {summary['overall_ceiling_accuracy_pct']}%")
        print(f"  Overall ceiling WAPE: {summary['overall_ceiling_wape']}%")
        gap = summary["overall_ceiling_accuracy_pct"] - summary["overall_champion_accuracy_pct"]
        print(f"  Gap to ceiling: {gap:.2f} pp")
    print("\nDone.")


if __name__ == "__main__":
    main()
