"""
Champion model selection: pick the best-performing model per DFU.

For each DFU (dmdunit + dmdgroup + loc), evaluates competing models at
the configured lag/metric and selects the winner.  All forecast rows from
the winning model are copied into fact_external_forecast_monthly with
model_id = '<champion_model_id>' (default: 'champion').

The champion composite automatically appears in all accuracy comparison
views because it reuses the existing model_id mechanism.

Usage:
    python scripts/run_champion_selection.py [--config config/model_competition.yaml]
"""

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_db_params() -> dict[str, Any]:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5440")),
        "dbname": os.getenv("POSTGRES_DB", "demand_mvp"),
        "user": os.getenv("POSTGRES_USER", "demand"),
        "password": os.getenv("POSTGRES_PASSWORD", "demand"),
    }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "metric": "wape",
    "lag": "execution",
    "min_dfu_rows": 3,
    "champion_model_id": "champion",
    "models": [],
}


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
    return cfg


# ---------------------------------------------------------------------------
# Champion selection logic
# ---------------------------------------------------------------------------

def compute_dfu_winners(
    cur: psycopg.Cursor,
    models: list[str],
    lag_mode: str,
    min_rows: int,
    metric: str,
) -> list[tuple[str, str, str, str, float, int]]:
    """Return (dmdunit, dmdgroup, loc, winning_model_id, wape, n_rows) per DFU."""

    placeholders = ",".join(["%s"] * len(models))

    # Lag filter condition
    if lag_mode == "execution":
        lag_cond = "lag::text = execution_lag::text"
        params: list[Any] = list(models)
    else:
        lag_cond = "lag = %s"
        params = list(models) + [int(lag_mode)]

    sql = f"""
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
    cur.execute(sql, params)
    return cur.fetchall()


def insert_champion_forecasts(
    cur: psycopg.Cursor,
    winners: list[tuple[str, str, str, str, float, int]],
    champion_model_id: str,
) -> int:
    """Bulk-insert champion forecast rows.

    Uses temp table + INSERT ... SELECT for efficiency.
    Returns number of rows inserted.
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

    # 2. Create temp table with DFU → winning model mapping
    cur.execute("""
        CREATE TEMP TABLE _champion_winners (
            dmdunit TEXT NOT NULL,
            dmdgroup TEXT NOT NULL,
            loc TEXT NOT NULL,
            winning_model_id TEXT NOT NULL
        ) ON COMMIT DROP
    """)

    # 3. COPY winners into temp table
    buf = io.StringIO()
    for dmdunit, dmdgroup, loc, model_id, _wape, _n in winners:
        buf.write(f"{dmdunit}\t{dmdgroup}\t{loc}\t{model_id}\n")
    buf.seek(0)
    with cur.copy("COPY _champion_winners FROM STDIN") as copy:
        copy.write(buf.read())

    # 4. Bulk INSERT ... SELECT champion rows
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
        (champion_model_id,),
    )
    inserted = cur.rowcount
    return inserted


def generate_summary(
    winners: list[tuple[str, str, str, str, float, int]],
    champion_model_id: str,
    total_rows: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Produce a summary dict from the winner list."""
    model_wins: dict[str, int] = {}
    total_wape_num = 0.0
    total_wape_denom = 0.0
    for _dmdunit, _dmdgroup, _loc, model_id, wape, n_rows in winners:
        model_wins[model_id] = model_wins.get(model_id, 0) + 1
        if wape is not None:
            total_wape_num += wape * n_rows
            total_wape_denom += n_rows

    overall_wape = (total_wape_num / total_wape_denom * 100) if total_wape_denom else None
    overall_acc = (100.0 - overall_wape) if overall_wape is not None else None

    # Sort model wins descending
    sorted_wins = dict(sorted(model_wins.items(), key=lambda x: -x[1]))

    return {
        "config": {
            "metric": config.get("metric"),
            "lag": config.get("lag"),
            "min_dfu_rows": config.get("min_dfu_rows"),
            "champion_model_id": champion_model_id,
            "models": config.get("models", []),
        },
        "total_dfus": len(winners),
        "total_champion_rows": total_rows,
        "model_wins": sorted_wins,
        "overall_champion_wape": round(overall_wape, 4) if overall_wape is not None else None,
        "overall_champion_accuracy_pct": round(overall_acc, 4) if overall_acc is not None else None,
        "run_ts": datetime.now(timezone.utc).isoformat(),
    }


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
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    db = _get_db_params()

    config_path = ROOT / args.config
    cfg = load_config(config_path)

    models = cfg["models"]
    metric = cfg["metric"]
    lag_mode = str(cfg["lag"])
    min_rows = int(cfg["min_dfu_rows"])
    champion_id = cfg["champion_model_id"]

    print(f"Champion Selection — {len(models)} competing models")
    print(f"  Metric: {metric}  |  Lag: {lag_mode}  |  Min rows: {min_rows}")
    print(f"  Models: {', '.join(models)}")
    print(f"  Champion model_id: '{champion_id}'")
    print()

    # Step 1: Compute DFU-level winners
    t0 = time.time()
    print("Computing per-DFU WAPE for each model...")
    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        winners = compute_dfu_winners(cur, models, lag_mode, min_rows, metric)
    print(f"  {len(winners):,} DFUs evaluated ({time.time() - t0:.1f}s)")

    if not winners:
        print("No qualifying DFUs found. Aborting.")
        sys.exit(1)

    # Print top model wins
    wins_count: dict[str, int] = {}
    for _, _, _, mid, _, _ in winners:
        wins_count[mid] = wins_count.get(mid, 0) + 1
    for mid, cnt in sorted(wins_count.items(), key=lambda x: -x[1]):
        pct = 100.0 * cnt / len(winners)
        print(f"    {mid:<25s} {cnt:>6,} DFUs ({pct:.1f}%)")
    print()

    # Step 2: Insert champion rows
    print("Inserting champion forecast rows...")
    t1 = time.time()
    with psycopg.connect(**db) as conn, conn.cursor() as cur:
        inserted = insert_champion_forecasts(cur, winners, champion_id)
        conn.commit()
    print(f"  Inserted {inserted:,} champion rows ({time.time() - t1:.1f}s)")
    print()

    # Step 3: Refresh materialized views
    refresh_views(db)
    print()

    # Step 4: Save summary JSON
    summary = generate_summary(winners, champion_id, inserted, cfg)
    summary_dir = ROOT / "data" / "champion"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "champion_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    print(f"  Overall champion accuracy: {summary['overall_champion_accuracy_pct']}%")
    print(f"  Overall champion WAPE: {summary['overall_champion_wape']}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
