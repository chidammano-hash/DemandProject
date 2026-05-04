"""
Champion experiment runner — runs a champion selection strategy experiment
and stores per-lag and per-month breakdowns in the database.

Designed to be called as a subprocess by the job engine:
    python scripts/run_champion_experiment.py --experiment-id <int>

The experiment config (strategy, strategy_params, models, metric, lag_mode,
min_sku_rows) is read from the champion_experiment DB row.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.champion_strategies import (
    STRATEGY_REGISTRY,
    compute_ceiling,
    compute_strategy_accuracy,
)
from common.db import get_db_params
from scripts.ml.run_champion_selection import load_dfu_features, load_monthly_errors_df

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _connect(db: dict[str, Any]) -> psycopg.Connection:
    return psycopg.connect(**db, autocommit=True)


def _load_experiment(conn: psycopg.Connection, experiment_id: int) -> dict[str, Any]:
    """Load experiment config from DB row."""
    row = conn.execute(
        """
        SELECT experiment_id, strategy, strategy_params, meta_learner_params,
               models, metric, lag_mode, min_sku_rows, status
        FROM champion_experiment
        WHERE experiment_id = %s
        """,
        (experiment_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Champion experiment {experiment_id} not found")
    cols = [
        "experiment_id", "strategy", "strategy_params", "meta_learner_params",
        "models", "metric", "lag_mode", "min_sku_rows", "status",
    ]
    d = dict(zip(cols, row))
    # Parse JSONB strings if needed
    for key in ("strategy_params", "meta_learner_params", "models"):
        if isinstance(d[key], str):
            d[key] = json.loads(d[key])
    return d


def _set_running(conn: psycopg.Connection, experiment_id: int) -> None:
    conn.execute(
        "UPDATE champion_experiment SET status = 'running', started_at = NOW() "
        "WHERE experiment_id = %s",
        (experiment_id,),
    )


def _set_completed(
    conn: psycopg.Connection,
    experiment_id: int,
    champion_accuracy: float | None,
    ceiling_accuracy: float | None,
    gap_bps: float | None,
    n_champions: int,
    n_dfu_months: int,
    model_distribution: dict[str, float],
    runtime_seconds: float,
) -> None:
    conn.execute(
        """
        UPDATE champion_experiment
        SET status = 'completed', completed_at = NOW(),
            champion_accuracy = %s, ceiling_accuracy = %s, gap_bps = %s,
            n_champions = %s, n_dfu_months = %s,
            model_distribution = %s, runtime_seconds = %s
        WHERE experiment_id = %s
        """,
        (
            champion_accuracy, ceiling_accuracy, gap_bps,
            n_champions, n_dfu_months,
            json.dumps(model_distribution), round(runtime_seconds, 1),
            experiment_id,
        ),
    )


def _set_failed(conn: psycopg.Connection, experiment_id: int, error: str, runtime: float) -> None:
    conn.execute(
        """
        UPDATE champion_experiment
        SET status = 'failed', completed_at = NOW(),
            notes = COALESCE(notes, '') || %s,
            runtime_seconds = %s
        WHERE experiment_id = %s
        """,
        (f"\n\nERROR:\n{error}", round(runtime, 1), experiment_id),
    )


def _insert_lag_rows(
    conn: psycopg.Connection,
    experiment_id: int,
    lag_rows: list[dict[str, Any]],
) -> None:
    """Insert per-execution-lag breakdown rows (batch)."""
    if not lag_rows:
        return
    values: list[tuple[Any, ...]] = [
        (
            experiment_id, r["exec_lag"], r["champion_accuracy"],
            r["ceiling_accuracy"], r["gap_bps"], r["n_dfu_months"],
            json.dumps(r["model_distribution"]),
        )
        for r in lag_rows
    ]
    placeholders = ", ".join(["(%s, %s, %s, %s, %s, %s, %s)"] * len(values))
    flat = [v for tup in values for v in tup]
    conn.execute(
        f"""
        INSERT INTO champion_experiment_lag
            (experiment_id, exec_lag, champion_accuracy, ceiling_accuracy,
             gap_bps, n_dfu_months, model_distribution)
        VALUES {placeholders}
        ON CONFLICT (experiment_id, exec_lag) DO UPDATE SET
            champion_accuracy = EXCLUDED.champion_accuracy,
            ceiling_accuracy = EXCLUDED.ceiling_accuracy,
            gap_bps = EXCLUDED.gap_bps,
            n_dfu_months = EXCLUDED.n_dfu_months,
            model_distribution = EXCLUDED.model_distribution
        """,
        flat,
    )


def _insert_month_rows(
    conn: psycopg.Connection,
    experiment_id: int,
    month_rows: list[dict[str, Any]],
) -> None:
    """Insert per-month breakdown rows (batch)."""
    if not month_rows:
        return
    values: list[tuple[Any, ...]] = [
        (
            experiment_id, str(r["month_start"]), r["champion_accuracy"],
            r["ceiling_accuracy"], r["gap_bps"], r["n_champions"],
            json.dumps(r["model_distribution"]),
        )
        for r in month_rows
    ]
    placeholders = ", ".join(["(%s, %s, %s, %s, %s, %s, %s)"] * len(values))
    flat = [v for tup in values for v in tup]
    conn.execute(
        f"""
        INSERT INTO champion_experiment_month
            (experiment_id, month_start, champion_accuracy, ceiling_accuracy,
             gap_bps, n_champions, model_distribution)
        VALUES {placeholders}
        ON CONFLICT (experiment_id, month_start) DO UPDATE SET
            champion_accuracy = EXCLUDED.champion_accuracy,
            ceiling_accuracy = EXCLUDED.ceiling_accuracy,
            gap_bps = EXCLUDED.gap_bps,
            n_champions = EXCLUDED.n_champions,
            model_distribution = EXCLUDED.model_distribution
        """,
        flat,
    )


# ---------------------------------------------------------------------------
# Model distribution helper
# ---------------------------------------------------------------------------

def _model_distribution(df: pd.DataFrame) -> dict[str, float]:
    """Compute model win % from a winners DataFrame."""
    if df.empty:
        return {}
    counts = df["model_id"].value_counts()
    total = counts.sum()
    return {mid: round(100.0 * cnt / total, 2) for mid, cnt in counts.items()}


# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------

def run_experiment(experiment_id: int) -> None:
    """Run a single champion selection experiment."""
    load_dotenv(ROOT / ".env")
    db = get_db_params()
    conn: psycopg.Connection | None = None
    t0 = time.time()

    try:
        conn = _connect(db)
        # 1. Load config
        exp = _load_experiment(conn, experiment_id)
        logger.info(
            "Champion experiment #%d: strategy=%s, models=%s, metric=%s, lag=%s",
            experiment_id, exp["strategy"], exp["models"], exp["metric"], exp["lag_mode"],
        )

        # 2. Set running
        _set_running(conn, experiment_id)

        strategy = exp["strategy"]
        strategy_params = exp["strategy_params"] or {}
        meta_learner_params = exp["meta_learner_params"] or {}
        models = exp["models"]
        lag_mode = str(exp["lag_mode"])
        min_sku_rows = int(exp["min_sku_rows"])

        # 3. Load data
        logger.info("Loading monthly errors for %d models...", len(models))
        monthly_errors_df = load_monthly_errors_df(db, models, lag_mode)
        logger.info("  %d rows loaded", len(monthly_errors_df))

        if monthly_errors_df.empty:
            raise ValueError("No monthly error data found for the configured models/lag")

        # Load DFU features for meta_learner
        dfu_features = None
        if strategy == "meta_learner":
            logger.info("Loading DFU features for meta-learner...")
            dfu_features = load_dfu_features(db)
            logger.info("  %d DFU feature rows loaded", len(dfu_features))

        # 4. Run strategy
        strategy_fn = STRATEGY_REGISTRY.get(strategy)
        if strategy_fn is None:
            raise ValueError(f"Unknown strategy: {strategy}")

        strat_kwargs: dict[str, Any] = {
            "min_prior_months": min_sku_rows,
            **strategy_params,
        }
        if strategy == "meta_learner":
            strat_kwargs["dfu_features"] = dfu_features
            strat_kwargs.update(meta_learner_params)
            strat_kwargs.setdefault(
                "meta_model_path",
                str(ROOT / "data" / "champion" / "meta_learner.joblib"),
            )

        logger.info("Running %s strategy...", strategy)
        winners_df = strategy_fn(monthly_errors_df, **strat_kwargs)
        logger.info("  %d DFU-month winners", len(winners_df))

        # 4b. Cache winners CSV for fast results loading (Stage 2)
        winners_dir = ROOT / "data" / "champion"
        winners_dir.mkdir(parents=True, exist_ok=True)
        winners_path = winners_dir / f"experiment_{experiment_id}_winners.csv"
        winners_df.to_csv(winners_path, index=False)
        logger.info("  Cached winners to %s", winners_path)

        # 5. Compute overall champion accuracy
        champ_stats = compute_strategy_accuracy(winners_df)
        champion_accuracy = champ_stats["accuracy_pct"]
        logger.info("  Champion accuracy: %s%%", champion_accuracy)

        # 6. Compute ceiling (oracle)
        ceiling_df = compute_ceiling(monthly_errors_df)
        ceiling_stats = compute_strategy_accuracy(ceiling_df)
        ceiling_accuracy = ceiling_stats["accuracy_pct"]
        logger.info("  Ceiling accuracy: %s%%", ceiling_accuracy)

        # Gap in basis points
        gap_bps = None
        if champion_accuracy is not None and ceiling_accuracy is not None:
            gap_bps = round((ceiling_accuracy - champion_accuracy) * 100, 2)
            logger.info("  Gap to ceiling: %s bps", gap_bps)

        # 7. Model distribution
        overall_dist = _model_distribution(winners_df)
        n_champions = winners_df[["item_id", "customer_group", "loc"]].drop_duplicates().shape[0]
        n_dfu_months = len(winners_df)

        # 8. Per-lag breakdown (all DFUs at each forecast horizon 0-4)
        #
        # Load all-lags data from backtest_lag_archive so that each lag
        # slice contains ALL DFUs at that forecast horizon — not just the
        # DFUs whose execution_lag matches.  This aligns with how
        # lgbm_tuning_lag is populated in seed_production_baselines.py.
        logger.info("Computing per-forecast-lag breakdown...")
        lag_rows: list[dict[str, Any]] = []
        try:
            all_lags_df = load_monthly_errors_df(db, models, lag_mode="all")
            logger.info("  %d all-lags rows loaded", len(all_lags_df))
        except Exception:
            logger.warning("Failed to load all-lags data; falling back to empty", exc_info=True)
            all_lags_df = pd.DataFrame()

        if not all_lags_df.empty and "lag" in all_lags_df.columns:
            for lag_val, lag_group in all_lags_df.groupby("lag"):
                try:
                    lag_winners = strategy_fn(lag_group, **strat_kwargs)
                    lag_champ = compute_strategy_accuracy(lag_winners)
                    lag_ceiling = compute_ceiling(lag_group)
                    lag_ceil_stats = compute_strategy_accuracy(lag_ceiling)
                    lag_gap = None
                    if lag_champ["accuracy_pct"] is not None and lag_ceil_stats["accuracy_pct"] is not None:
                        lag_gap = round(
                            (lag_ceil_stats["accuracy_pct"] - lag_champ["accuracy_pct"]) * 100, 2
                        )
                    lag_rows.append({
                        "exec_lag": int(lag_val),
                        "champion_accuracy": lag_champ["accuracy_pct"],
                        "ceiling_accuracy": lag_ceil_stats["accuracy_pct"],
                        "gap_bps": lag_gap,
                        "n_dfu_months": lag_champ["n_dfu_months"],
                        "model_distribution": _model_distribution(lag_winners),
                    })
                except (ValueError, KeyError) as exc:
                    logger.warning("Skipping lag %s: %s", lag_val, exc)
        _insert_lag_rows(conn, experiment_id, lag_rows)
        logger.info("  Inserted %d lag breakdown rows", len(lag_rows))

        # 9. Per-month breakdown
        logger.info("Computing per-month breakdown...")
        month_rows: list[dict[str, Any]] = []
        if not winners_df.empty and "startdate" in winners_df.columns:
            for month_val, month_group in winners_df.groupby(
                winners_df["startdate"].dt.to_period("M")
            ):
                m_stats = compute_strategy_accuracy(month_group)
                # Ceiling for same month
                month_mask = ceiling_df["startdate"].dt.to_period("M") == month_val
                m_ceil = compute_strategy_accuracy(ceiling_df[month_mask])
                m_gap = None
                if m_stats["accuracy_pct"] is not None and m_ceil["accuracy_pct"] is not None:
                    m_gap = round((m_ceil["accuracy_pct"] - m_stats["accuracy_pct"]) * 100, 2)
                month_rows.append({
                    "month_start": month_val.to_timestamp().date(),
                    "champion_accuracy": m_stats["accuracy_pct"],
                    "ceiling_accuracy": m_ceil["accuracy_pct"],
                    "gap_bps": m_gap,
                    "n_champions": len(month_group),
                    "model_distribution": _model_distribution(month_group),
                })
        _insert_month_rows(conn, experiment_id, month_rows)
        logger.info("  Inserted %d month breakdown rows", len(month_rows))

        # 10. Mark completed
        runtime = time.time() - t0
        _set_completed(
            conn, experiment_id,
            champion_accuracy, ceiling_accuracy, gap_bps,
            n_champions, n_dfu_months, overall_dist, runtime,
        )
        logger.info(
            "Champion experiment #%d completed in %.1fs — accuracy=%.2f%%, ceiling=%.2f%%",
            experiment_id, runtime,
            champion_accuracy or 0, ceiling_accuracy or 0,
        )

    except Exception as exc:
        runtime = time.time() - t0
        logger.exception("Champion experiment #%d failed: %s", experiment_id, exc)
        if conn is not None:
            try:
                _set_failed(conn, experiment_id, traceback.format_exc(), runtime)
            except psycopg.Error as db_exc:
                logger.error("Failed to update experiment status to failed: %s", db_exc)
        sys.exit(1)
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a champion selection experiment")
    parser.add_argument(
        "--experiment-id",
        type=int,
        required=True,
        help="ID of the champion_experiment row to execute",
    )
    args = parser.parse_args()
    run_experiment(args.experiment_id)


if __name__ == "__main__":
    main()
