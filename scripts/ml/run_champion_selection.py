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
    python scripts/run_champion_selection.py [--config config/forecasting/forecast_pipeline_config.yaml]
"""

import argparse
import ast
import io
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
import yaml
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ml.champion import (
    STRATEGY_REGISTRY,
    compute_strategy_accuracy,
)
from common.core.constants import FORECAST_QTY_COL
from common.core.db import get_db_params
from common.core.mv_refresh import refresh_for_tables  # noqa: E402 — after sys.path bootstrap
from common.core.sql_helpers import read_sql_chunked  # noqa: E402 — after sys.path bootstrap
from common.services.perf_profiler import profiled_section
from common.core.utils import get_competing_model_ids, load_forecast_pipeline_config


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


def _load_config_from_pipeline() -> dict[str, Any] | None:
    """Try to load champion config from forecast_pipeline_config.yaml.

    Returns a merged config dict compatible with the old competition format,
    or None if the pipeline config is unavailable.
    """
    try:
        pipeline_cfg = load_forecast_pipeline_config()
    except FileNotFoundError:
        return None

    champion = pipeline_cfg.get("champion", {})
    if not champion:
        return None

    # Derive competing model list from the algorithm roster
    models = get_competing_model_ids()

    cfg: dict[str, Any] = {
        "models": models,
        "strategy": champion.get("strategy", "expanding"),
        "strategy_params": champion.get("strategy_params", {}),
        "metric": champion.get("metric", "accuracy_pct"),
        "lag": champion.get("lag", "execution"),
        "min_dfu_rows": champion.get("min_dfu_rows", 3),
        "min_sku_rows": champion.get("min_sku_rows", 3),
        "champion_model_id": champion.get("champion_model_id", "champion"),
        "fallback_model_id": champion.get("fallback_model_id", "seasonal_naive"),
        "meta_learner": champion.get("meta_learner", {}),
    }
    return cfg


def load_config(config_path: Path) -> dict[str, Any]:
    """Read and validate the competition config YAML.

    Loads champion settings from forecast_pipeline_config.yaml.
    """
    pipeline_cfg = _load_config_from_pipeline()
    if pipeline_cfg is None:
        raise FileNotFoundError(
            "forecast_pipeline_config.yaml not found or missing 'champion' section"
        )
    cfg = pipeline_cfg

    for k, v in _DEFAULTS.items():
        cfg.setdefault(k, v)
    # Validate
    # Gen-4 Stream G (AI-2 P1): probabilistic metrics are accepted but
    # currently fall back to WAPE when quantile data is absent. Full
    # CRPS/pinball selection requires per-DFU quantile rows in
    # fact_candidate_forecast.
    if cfg["metric"] not in ("wape", "accuracy_pct", "crps", "pinball"):
        raise ValueError(
            f"Invalid metric '{cfg['metric']}'; must be wape, accuracy_pct, crps, or pinball"
        )
    if cfg["metric"] in ("crps", "pinball"):
        # TODO(gen-4): Wire compute_crps/compute_pinball_loss once candidate
        # quantile columns (forecast_qty_p10/p50/p90) are consistently populated
        # by the champion FM for every DFU. For now we log & fall back so the
        # pipeline doesn't fail silently on partially-quantiled data.
        import logging as _log
        _log.getLogger(__name__).warning(
            "champion.metric=%s requested; falling back to WAPE because per-DFU "
            "quantile data is not yet guaranteed. See common.ml.crps.",
            cfg["metric"],
        )
        cfg["metric"] = "wape"
    valid_lags = {"execution", "0", "1", "2", "3", "4"}
    if str(cfg["lag"]) not in valid_lags:
        raise ValueError(f"Invalid lag '{cfg['lag']}'; must be one of {sorted(valid_lags)}")
    if len(cfg["models"]) < 2:
        raise ValueError("At least 2 models required for competition")
    if cfg["strategy"] not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Invalid strategy '{cfg['strategy']}'; "
            f"must be one of {sorted(STRATEGY_REGISTRY.keys())}"
        )
    return cfg


# ---------------------------------------------------------------------------
# Pre-champion roster-coverage guard
# ---------------------------------------------------------------------------

def assert_competing_models_covered(
    cur: psycopg.Cursor,
    models: list[str],
) -> None:
    """Fail loud if any competing model has no rows in fact_external_forecast_monthly.

    The champion competition can only select among models that were actually
    backtested AND loaded into ``fact_external_forecast_monthly``. If a
    ``compete: true`` model (per ``get_competing_model_ids()``) was never produced
    by ``backtest-all`` / ``backtest-load-all`` — e.g. the cust_enriched trees
    before they were chained in — it silently has zero candidate rows and drops
    out of the competition. Picking the champion from a partial field is a quiet
    data-coverage defect, so we assert full coverage up front and raise on the gap
    rather than ship a champion chosen from an incomplete roster.

    Raises:
        RuntimeError: if one or more competing models have zero candidate rows.
    """
    if not models:
        return

    placeholders = ",".join(["%s"] * len(models))
    sql = f"""
        SELECT model_id, COUNT(*) AS n
        FROM fact_external_forecast_monthly
        WHERE model_id IN ({placeholders})
        GROUP BY model_id
    """
    cur.execute(sql, list(models))
    counts = {model_id: int(n) for model_id, n in cur.fetchall()}

    missing = sorted(m for m in models if counts.get(m, 0) == 0)
    if missing:
        logger.error(
            "Champion roster coverage gap: %d competing model(s) have zero rows in "
            "fact_external_forecast_monthly: %s. These were never backtested+loaded, "
            "so the champion would be selected from a partial field. Run their "
            "backtest+load targets (see Makefile backtest-all / backtest-load-all) "
            "before champion selection.",
            len(missing),
            ", ".join(missing),
        )
        raise RuntimeError(
            "Competing models missing from fact_external_forecast_monthly "
            f"(zero candidate rows): {', '.join(missing)}. "
            "Champion selection requires every compete:true model to be "
            "backtested and loaded first."
        )


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

    Lag modes:
        "execution"  — only rows where lag == execution_lag (one per DFU-month).
                        Queries fact_external_forecast_monthly.
        "all"        — all 5 lag rows (0-4) per DFU-month, for cross-horizon
                        analysis.  Queries backtest_lag_archive.
        "0".."4"     — single specific lag for all DFUs.
                        Queries fact_external_forecast_monthly.
    """
    placeholders = ",".join(["%s"] * len(models))
    params: list[Any] = list(models)

    if lag_mode == "all":
        # All lags from archive table (5 rows per DFU-month)
        sql = f"""
            SELECT item_id, customer_group, loc, startdate, fcstdate,
                   execution_lag, lag,
                   model_id, basefcst_pref, tothist_dmd,
                   ABS(basefcst_pref - tothist_dmd) AS abs_err
            FROM backtest_lag_archive
            WHERE model_id IN ({placeholders})
              AND basefcst_pref IS NOT NULL
              AND tothist_dmd IS NOT NULL
            ORDER BY item_id, customer_group, loc, model_id, lag, startdate
        """
    else:
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
        # Fact-table scan (fact_external_forecast_monthly / backtest_lag_archive) —
        # stream in chunks to bound peak memory at scale (drop-in for pd.read_sql).
        df = read_sql_chunked(conn, sql, params=params)
    df["startdate"] = pd.to_datetime(df["startdate"])
    df["fcstdate"] = pd.to_datetime(df["fcstdate"])
    for col in [FORECAST_QTY_COL, "tothist_dmd", "abs_err"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["execution_lag"] = pd.to_numeric(df["execution_lag"], errors="coerce").fillna(0).astype(int)
    if "lag" in df.columns:
        df["lag"] = pd.to_numeric(df["lag"], errors="coerce").fillna(0).astype(int)
    return df


def load_dfu_features(db: dict[str, Any]) -> pd.DataFrame:
    """Load DFU static features for meta-learner strategy."""
    sql = """
        SELECT d.item_id, d.customer_group, d.loc,
               ca.ml_cluster, d.abc_vol, d.execution_lag, d.total_lt,
               d.brand, d.region,
               d.seasonality_profile, d.seasonality_strength,
               d.is_yearly_seasonal, d.peak_month, d.trough_month,
               d.peak_trough_ratio
        FROM dim_sku d
        LEFT JOIN current_sku_cluster_assignment ca
               ON ca.sku_ck = d.sku_ck
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

    # 3. COPY ceiling winners into temp table. write_row lets psycopg3 escape each
    #    value; a manual tab-delimited buffer desyncs the stream if any
    #    item_id/customer_group/loc contains a tab/newline/backslash.
    with cur.copy(
        "COPY _ceiling_winners "
        "(item_id, customer_group, loc, startdate, winning_model_id) FROM STDIN"
    ) as copy:
        for item_id, customer_group, loc, startdate, model_id, *_ in ceiling_rows:
            copy.write_row((item_id, customer_group, loc, startdate, model_id))

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
    champion_experiment_id: int | None = None,
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

    # 3. COPY champion winners into temp table (write_row escapes each value;
    #    see ceiling-winners note above re: tab/newline/backslash desync).
    with cur.copy(
        "COPY _champion_winners "
        "(item_id, customer_group, loc, startdate, winning_model_id) FROM STDIN"
    ) as copy:
        for item_id, customer_group, loc, startdate, model_id, *_ in winners:
            copy.write_row((item_id, customer_group, loc, startdate, model_id))

    # 4. Bulk INSERT ... SELECT champion rows
    # source_model_id records which underlying algorithm won (e.g. lgbm_cluster)
    # so production forecast can load the right .pkl artifact per DFU.
    cur.execute(
        """
        INSERT INTO fact_external_forecast_monthly
            (forecast_ck, item_id, customer_group, loc, fcstdate, startdate,
             lag, execution_lag, basefcst_pref, tothist_dmd, model_id,
             source_model_id, champion_experiment_id)
        SELECT
            f.forecast_ck, f.item_id, f.customer_group, f.loc, f.fcstdate, f.startdate,
            f.lag, f.execution_lag, f.basefcst_pref, f.tothist_dmd,
            %s, w.winning_model_id, %s
        FROM fact_external_forecast_monthly f
        INNER JOIN _champion_winners w
            ON f.item_id = w.item_id
           AND f.customer_group = w.customer_group
           AND f.loc = w.loc
           AND f.startdate = w.startdate
           AND f.model_id = w.winning_model_id
        """,
        (champion_model_id, champion_experiment_id),
    )
    inserted = cur.rowcount
    return inserted


def insert_ensemble_forecasts(
    cur: psycopg.Cursor,
    winners_df: pd.DataFrame,
    champion_model_id: str,
    champion_experiment_id: int | None = None,
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

    # 2. Create temp table with blended forecast values + the per-DFU winning
    #    sub-model/strategy label (so source_model_id is recorded even for
    #    router/blend strategies, not just single-model winners — lets the UI
    #    label the champion line "champion (<model>)").
    cur.execute("""
        CREATE TEMP TABLE _ensemble_winners (
            item_id TEXT NOT NULL,
            customer_group TEXT NOT NULL,
            loc TEXT NOT NULL,
            startdate DATE NOT NULL,
            blended_fcst DOUBLE PRECISION NOT NULL,
            source_model_id TEXT,
            source_mix JSONB
        ) ON COMMIT DROP
    """)

    # 3. COPY ensemble data. source_mix is JSON (no tabs/newlines from
    #    json.dumps default separators) so tab-delimited COPY is safe; '\N'
    #    encodes SQL NULL for rows without a mix.
    buf = io.StringIO()
    for _, row in winners_df.iterrows():
        mix = row.get("source_mix")
        mix_cell = json.dumps(mix) if isinstance(mix, list) and mix else r"\N"
        buf.write(
            f"{row['item_id']}\t{row['customer_group']}\t{row['loc']}\t"
            f"{row['startdate'].date()}\t{row['basefcst_pref']}\t{row['model_id']}\t{mix_cell}\n"
        )
    buf.seek(0)
    with cur.copy("COPY _ensemble_winners FROM STDIN") as copy:
        copy.write(buf.read())

    # 4. Insert using a reference model row for metadata, override basefcst_pref
    cur.execute(
        """
        INSERT INTO fact_external_forecast_monthly
            (forecast_ck, item_id, customer_group, loc, fcstdate, startdate,
             lag, execution_lag, basefcst_pref, tothist_dmd, model_id,
             source_model_id, source_mix, champion_experiment_id)
        SELECT DISTINCT ON (f.item_id, f.customer_group, f.loc, f.startdate)
            f.forecast_ck, f.item_id, f.customer_group, f.loc, f.fcstdate, f.startdate,
            f.lag, f.execution_lag, w.blended_fcst, f.tothist_dmd,
            %s, w.source_model_id, w.source_mix, %s
        FROM fact_external_forecast_monthly f
        INNER JOIN _ensemble_winners w
            ON f.item_id = w.item_id
           AND f.customer_group = w.customer_group
           AND f.loc = w.loc
           AND f.startdate = w.startdate
        WHERE f.model_id != %s AND f.model_id != 'ceiling'
        ORDER BY f.item_id, f.customer_group, f.loc, f.startdate, f.model_id
        """,
        (champion_model_id, champion_experiment_id, champion_model_id),
    )
    inserted = cur.rowcount
    return inserted


def insert_fallback_champions(
    cur: psycopg.Cursor,
    lag_mode: str,
    champion_model_id: str,
    fallback_model_id: str,
    champion_experiment_id: int | None = None,
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
    # source_model_id records which model actually produced the value. Fallback
    # rows copy the fallback model's forecast verbatim, so they carry
    # source_model_id = fallback_model_id — NOT NULL. Leaving it NULL made
    # get_champion_assignments treat these as "legacy pre-column" rows and the
    # Item-Analysis UI mislabel them, masking the true source. NOTE: this corrects
    # the source ATTRIBUTION only — it does not by itself change which model
    # generates the go-forward production forecast, because the consumer's
    # _resolve_artifact still substitutes the production fallback for pkl-less
    # statistical sources (seasonal_naive/rolling_mean). That production-routing
    # gap is tracked separately (BACKLOG F-11).
    if lag_mode == "execution":
        lag_cond = "lag::text = execution_lag::text"
        params: list[Any] = [
            champion_model_id,
            fallback_model_id,
            champion_experiment_id,
            fallback_model_id,
            champion_model_id,
        ]
    else:
        lag_cond = "lag = %s"
        params = [
            champion_model_id,
            fallback_model_id,
            champion_experiment_id,
            fallback_model_id,
            int(lag_mode),
            champion_model_id,
        ]

    sql = f"""
    INSERT INTO fact_external_forecast_monthly
        (forecast_ck, item_id, customer_group, loc, fcstdate, startdate,
         lag, execution_lag, basefcst_pref, tothist_dmd, model_id,
         source_model_id, champion_experiment_id)
    SELECT
        f.forecast_ck, f.item_id, f.customer_group, f.loc, f.fcstdate, f.startdate,
        f.lag, f.execution_lag, f.basefcst_pref, f.tothist_dmd,
        %s, %s, %s
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


def _load_dfu_attributes(db: dict[str, Any]) -> pd.DataFrame:
    """Load DFU segment attributes (ml_cluster, abc_vol)."""
    sql = """
        SELECT d.item_id, d.customer_group, d.loc, ca.ml_cluster, d.abc_vol
        FROM dim_sku d
        LEFT JOIN current_sku_cluster_assignment ca
               ON ca.sku_ck = d.sku_ck
    """
    with psycopg.connect(**db) as conn:
        df = pd.read_sql(sql, conn)
    return df


def _compute_segment_accuracy(
    winners_df: pd.DataFrame,
    segment_col: str,
) -> list[dict[str, Any]]:
    """Compute WAPE and accuracy per unique value of ``segment_col``.

    Returns a list of dicts, one per segment value, sorted by accuracy
    descending.  Each dict contains:
        segment, wape, accuracy_pct, n_dfu_months, n_dfus
    """
    if winners_df.empty or segment_col not in winners_df.columns:
        return []

    results: list[dict[str, Any]] = []
    for seg_val, seg_df in winners_df.groupby(segment_col, sort=False, dropna=False):
        acc = compute_strategy_accuracy(seg_df)
        unique_dfus = seg_df[["item_id", "customer_group", "loc"]].drop_duplicates()
        results.append({
            "segment": str(seg_val) if pd.notna(seg_val) else "unknown",
            "wape": acc["wape"],
            "accuracy_pct": acc["accuracy_pct"],
            "n_dfu_months": acc["n_dfu_months"],
            "n_dfus": len(unique_dfus),
        })

    results.sort(key=lambda d: d["accuracy_pct"] if d["accuracy_pct"] is not None else -1, reverse=True)
    return results


def generate_summary(
    winners: list[tuple[str, str, str, str, str, float, float, float]],
    champion_model_id: str,
    total_rows: int,
    config: dict[str, Any],
    ceiling_rows: list[tuple] | None = None,
    ceiling_inserted: int = 0,
    fallback_inserted: int = 0,
    per_segment_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a summary dict from the winner list + optional ceiling data."""
    model_wins: dict[str, int] = {}
    champ_abs_err_sum = 0.0
    champ_actual_sum = 0.0
    unique_dfus: set[tuple[str, str, str]] = set()

    for item_id, customer_group, loc, _startdate, model_id, _prior_wape, basefcst_pref, tothist_dmd, *_ in winners:
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

    # Per-segment accuracy breakdowns (cluster, ABC class)
    if per_segment_analysis:
        for key, value in per_segment_analysis.items():
            summary[key] = value

    return summary


# ---------------------------------------------------------------------------
# Materialized view refresh
# ---------------------------------------------------------------------------

def refresh_views(db_params: dict[str, Any]) -> None:
    """Refresh every MV reading fact_external_forecast_monthly.

    The champion run just rewrote champion/ceiling rows there; the central
    dependency map (common/core/mv_refresh.py) covers the full dependent set,
    including agg_dfu_naive_scale and the previously skipped lag-archive MVs.
    """
    refresh_for_tables(["fact_external_forecast_monthly"], db_params=db_params)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_cached_winners(
    csv_path: Path,
    competing_models: list[str],
) -> tuple[pd.DataFrame, list[tuple], bool]:
    """Load cached winners from a CSV file saved by a champion experiment.

    *competing_models* is the list of model IDs from the config — used to
    detect whether the cached winners are from an ensemble strategy (winner
    model_ids not in the competing set means blended/synthetic forecasts).

    Returns (winners_df, winners_tuples, is_ensemble).
    """
    print(f"Loading cached winners from {csv_path}...")
    winners_df = pd.read_csv(csv_path, dtype={"item_id": str, "customer_group": str, "loc": str})
    winners_df["startdate"] = pd.to_datetime(winners_df["startdate"])
    for col in [FORECAST_QTY_COL, "tothist_dmd", "prior_wape"]:
        if col in winners_df.columns:
            winners_df[col] = pd.to_numeric(winners_df[col], errors="coerce")
    # source_mix round-trips through CSV as a JSON string — parse back to a list.
    if "source_mix" in winners_df.columns:
        winners_df["source_mix"] = winners_df["source_mix"].apply(_parse_source_mix)
    winners = list(winners_df.itertuples(index=False, name=None))
    # Detect ensemble: if winner model_ids contain values not in the
    # competing model list, the strategy produced blended forecasts.
    winner_model_ids = set(winners_df["model_id"].unique())
    is_ensemble = not winner_model_ids.issubset(set(competing_models))
    print(f"  {len(winners):,} cached DFU-month winners loaded")
    return winners_df, winners, is_ensemble


def _parse_source_mix(value: Any) -> list[dict[str, Any]] | None:
    """Parse canonical JSON and legacy pandas/Python literal CSV values safely."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError("Invalid source_mix in cached champion winners") from exc
    if not isinstance(parsed, list) or not all(isinstance(entry, dict) for entry in parsed):
        raise ValueError("source_mix must be a list of model-weight objects")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Champion model selection")
    parser.add_argument(
        "--config",
        type=str,
        default="config/forecasting/forecast_pipeline_config.yaml",
        help="Path to competition config YAML",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Selection strategy (overrides config): "
             "expanding, rolling, decay, ensemble, meta_learner",
    )
    parser.add_argument(
        "--load-winners-from",
        type=str,
        default=None,
        help="Path to a cached winners CSV (skips computation, loads directly)",
    )
    parser.add_argument(
        "--champion-experiment-id",
        type=int,
        default=None,
        help="Stamp canonical champion rows with the explicit results-promotion experiment",
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

    # Step 0: Roster-coverage guard. Every compete:true model must have candidate
    # rows in fact_external_forecast_monthly or the competition runs on a partial
    # field (the durable guard against backtest-all / config roster drift). Skip
    # when loading pre-computed experiment winners — that path doesn't query the
    # live candidate set, and the experiment may intentionally use a subset.
    if not args.load_winners_from:
        print("Verifying competing-model coverage in fact_external_forecast_monthly...")
        with psycopg.connect(**db) as conn, conn.cursor() as cur:
            assert_competing_models_covered(cur, models)
        print(f"  All {len(models)} competing models present.")
        print()

    # Step 1: Load cached winners or run strategy to compute them
    t0 = time.time()

    if args.load_winners_from:
        # Fast path: load pre-computed winners from experiment CSV
        cached_path = Path(args.load_winners_from)
        if not cached_path.is_absolute():
            cached_path = ROOT / cached_path
        if not cached_path.exists():
            print(f"ERROR: Cached winners file not found: {cached_path}")
            print("Falling back to full computation...")
            args.load_winners_from = None  # fall through to computation below

    if args.load_winners_from:
        with profiled_section("load_cached_winners"):
            cached_path = Path(args.load_winners_from)
            if not cached_path.is_absolute():
                cached_path = ROOT / cached_path
            winners_df, winners, is_ensemble = _load_cached_winners(cached_path, models)
    else:
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
                if strategy_name in ("meta_learner", "per_cluster"):
                    strat_kwargs["dfu_features"] = load_dfu_features(db)
                if strategy_name == "meta_learner":
                    strat_kwargs.setdefault(
                        "meta_model_path",
                        str(ROOT / "data" / "champion" / "meta_learner.joblib"),
                    )

                print(f"Computing champion winners ({strategy_name}, prior months only)...")
                winners_df = strategy_fn(monthly_errors_df, **strat_kwargs)

                # Convert to tuple list for existing insert/summary logic
                winners = list(winners_df.itertuples(index=False, name=None))
                # Detect ensemble: if winner model_ids are synthetic (not in the
                # competing model list), the strategy produced blended forecasts
                # that need the ensemble insert path.
                winner_model_ids = set(winners_df["model_id"].unique())
                is_ensemble = not winner_model_ids.issubset(set(models))
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
                inserted = insert_ensemble_forecasts(
                    cur,
                    winners_df,
                    champion_id,
                    args.champion_experiment_id,
                )
            else:
                inserted = insert_champion_forecasts(
                    cur,
                    winners,
                    champion_id,
                    args.champion_experiment_id,
                )
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
                    cur,
                    lag_mode,
                    champion_id,
                    fallback_id,
                    args.champion_experiment_id,
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

    # Step 6: Per-segment accuracy breakdowns
    per_segment_analysis: dict[str, Any] = {}
    with profiled_section("per_segment_accuracy"):
        print("Computing per-segment accuracy breakdowns...")
        try:
            dfu_attrs = _load_dfu_attributes(db)
            print(f"  Loaded {len(dfu_attrs):,} DFU attribute rows from dim_sku")

            # Build champion winners DataFrame for segment analysis
            champ_df = pd.DataFrame(
                winners,
                columns=[
                    "item_id", "customer_group", "loc", "startdate",
                    "model_id", "prior_wape", FORECAST_QTY_COL, "tothist_dmd",
                    "source_mix",
                ],
            )
            champ_merged = champ_df.merge(
                dfu_attrs, on=["item_id", "customer_group", "loc"], how="left",
            )

            # Build ceiling winners DataFrame for segment analysis
            ceil_df = pd.DataFrame(
                ceiling_rows or [],
                columns=[
                    "item_id", "customer_group", "loc", "startdate",
                    "model_id", "abs_err", FORECAST_QTY_COL, "tothist_dmd",
                ],
            )
            ceil_merged = ceil_df.merge(
                dfu_attrs, on=["item_id", "customer_group", "loc"], how="left",
            ) if not ceil_df.empty else ceil_df

            for seg_col, seg_key in [
                ("ml_cluster", "per_cluster_analysis"),
                ("abc_vol", "per_abc_analysis"),
            ]:
                champ_seg = _compute_segment_accuracy(champ_merged, seg_col)
                ceil_seg = _compute_segment_accuracy(ceil_merged, seg_col)

                # Build a lookup for ceiling accuracy by segment
                ceil_lookup = {d["segment"]: d for d in ceil_seg}

                # Enrich champion segments with ceiling and gap
                for entry in champ_seg:
                    seg = entry["segment"]
                    ceil_entry = ceil_lookup.get(seg)
                    if ceil_entry and ceil_entry["accuracy_pct"] is not None:
                        entry["ceiling_wape"] = ceil_entry["wape"]
                        entry["ceiling_accuracy_pct"] = ceil_entry["accuracy_pct"]
                        if entry["accuracy_pct"] is not None:
                            entry["gap_to_ceiling_pp"] = round(
                                ceil_entry["accuracy_pct"] - entry["accuracy_pct"], 4,
                            )
                        else:
                            entry["gap_to_ceiling_pp"] = None
                    else:
                        entry["ceiling_wape"] = None
                        entry["ceiling_accuracy_pct"] = None
                        entry["gap_to_ceiling_pp"] = None

                per_segment_analysis[seg_key] = champ_seg
                print(f"  {seg_key}: {len(champ_seg)} segments")

        except psycopg.Error as exc:
            print(f"  WARNING: Could not load DFU attributes for segment analysis: {exc}")
    print()

    # Step 7: Save summary JSON
    with profiled_section("save_summary"):
        summary = generate_summary(
            winners, champion_id, inserted, cfg,
            ceiling_rows=ceiling_rows, ceiling_inserted=ceiling_inserted,
            fallback_inserted=fallback_inserted,
            per_segment_analysis=per_segment_analysis,
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

    # Print per-segment summaries
    for seg_key in ["per_cluster_analysis", "per_abc_analysis"]:
        seg_data = summary.get(seg_key)
        if seg_data:
            label = "Cluster" if "cluster" in seg_key else "ABC class"
            print(f"\n  Per-{label} accuracy:")
            for entry in seg_data:
                gap_str = f", gap={entry['gap_to_ceiling_pp']:.1f}pp" if entry.get("gap_to_ceiling_pp") is not None else ""
                print(
                    f"    {entry['segment']:<12s}  acc={entry['accuracy_pct']:.1f}%  "
                    f"wape={entry['wape']:.1f}%  "
                    f"DFUs={entry['n_dfus']:,}  months={entry['n_dfu_months']:,}"
                    f"{gap_str}"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
