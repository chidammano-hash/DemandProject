"""Multi-Algorithm Inventory Comparison.

Computes safety stock, EOQ, and reorder point targets for every forecast
algorithm present in fact_production_forecast_staging, then writes the results
to fact_inventory_algorithm_comparison for side-by-side analysis.

For each model_id the script:
1. Aggregates forecasts: AVG and STDDEV of forecast_qty per item_id + loc.
2. Joins dim_sku for abc_vol and dim_item_lead_time_profile for lead times.
3. Looks up service_level / z_score from config by ABC class.
4. Computes SS (combined variability), EOQ (Wilson), and ROP.
5. DELETE-then-INSERT per model_id.

Usage:
    uv run python scripts/compare_inventory_algorithms.py
    uv run python scripts/compare_inventory_algorithms.py --models lgbm_cluster,nbeats
    uv run python scripts/compare_inventory_algorithms.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Any

import psycopg

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.db import get_db_params  # noqa: E402
from common.scripts_base import setup_logging  # noqa: E402
from common.services.perf_profiler import profiled_section  # noqa: E402
from common.utils import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DAYS_PER_MONTH: float = 30.44


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_ss_config() -> dict[str, Any]:
    """Load safety stock config (includes shared_constants via _includes)."""
    cfg = load_config("safety_stock_config")
    return cfg["safety_stock"]


def _load_eoq_config() -> dict[str, Any]:
    """Load EOQ config (includes shared_constants via _includes)."""
    return load_config("eoq_config")


def _get_service_level(abc_vol: str | None, service_levels: dict[str, float]) -> float:
    """Return configured service level for a given ABC class."""
    if abc_vol and abc_vol.upper() in service_levels:
        return float(service_levels[abc_vol.upper()])
    return float(service_levels.get("default", 0.95))


def _get_z_score(service_level: float, z_table: dict[str, float]) -> float:
    """Look up the Z-score for a given service level from the config z_table.

    Uses string key lookup to avoid float comparison fragility (YAML keys
    like 0.98 may not round-trip exactly through float conversion).
    Falls back to the closest key, then to 1.645.
    """
    sl_str = str(service_level)
    if sl_str in z_table:
        return float(z_table[sl_str])
    if z_table:
        closest = min(z_table.keys(), key=lambda k: abs(float(k) - service_level))
        return float(z_table[closest])
    return 1.645


# ---------------------------------------------------------------------------
# Pure formula functions
# ---------------------------------------------------------------------------

def _compute_ss_combined(
    z: float,
    sigma_d_daily: float,
    lt_mean_days: float,
    avg_daily_demand: float,
    lt_std_days: float,
) -> float:
    """SS_combined = Z * sqrt(LT_mean * sigma_D^2 + D_avg^2 * lt_std^2)."""
    demand_var = lt_mean_days * (sigma_d_daily ** 2)
    lt_var = (avg_daily_demand ** 2) * (lt_std_days ** 2)
    return z * math.sqrt(demand_var + lt_var)


def _compute_ss_demand_only(
    z: float,
    sigma_d_daily: float,
    lt_mean_days: float,
) -> float:
    """SS_demand = Z * sqrt(LT_mean * sigma_D^2)."""
    variance_term = lt_mean_days * (sigma_d_daily ** 2)
    return z * math.sqrt(variance_term) if variance_term > 0 else 0.0


def _compute_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_pct: float,
    unit_cost: float,
) -> float | None:
    """Wilson EOQ = sqrt(2 * D * S / (H * C)). Returns None if inputs invalid."""
    denominator = holding_cost_pct * unit_cost
    if denominator <= 0 or annual_demand <= 0:
        return None
    return math.sqrt(2 * annual_demand * ordering_cost / denominator)


def _compute_effective_eoq(
    eoq: float,
    moq: float,
    max_months_supply: float,
    demand_mean_monthly: float,
) -> float:
    """Apply MOQ floor and months-supply cap to raw EOQ."""
    cap = max_months_supply * demand_mean_monthly if demand_mean_monthly > 0 else float("inf")
    effective = max(eoq, moq)
    if cap > 0:
        effective = min(effective, cap)
    return effective


# ---------------------------------------------------------------------------
# DB data loaders
# ---------------------------------------------------------------------------

def _load_models(cur: Any, model_filter: list[str] | None = None) -> list[str]:
    """Get distinct model_ids from staging table."""
    cur.execute("SELECT DISTINCT model_id FROM fact_production_forecast_staging ORDER BY model_id")
    all_models = [row[0] for row in cur.fetchall()]
    if model_filter:
        all_models = [m for m in all_models if m in model_filter]
    return all_models


def _load_forecast_stats(cur: Any, model_id: str) -> list[dict]:
    """Aggregate forecast_qty per item_id + loc for a given model_id."""
    sql = """
        SELECT item_id, loc,
               AVG(forecast_qty)::numeric(15,4)    AS avg_qty,
               STDDEV(forecast_qty)::numeric(15,4)  AS std_qty
        FROM fact_production_forecast_staging
        WHERE model_id = %s
        GROUP BY item_id, loc
    """
    cur.execute(sql, [model_id])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


def _load_abc_map(cur: Any) -> dict[tuple[str, str], str | None]:
    """Load abc_vol from dim_sku keyed by (item_id, loc)."""
    try:
        cur.execute("SAVEPOINT abc_load")
        cur.execute("SELECT item_id, loc, abc_vol FROM dim_sku")
        result: dict[tuple[str, str], str | None] = {}
        for row in cur.fetchall():
            result[(row[0], row[1])] = row[2]
        return result
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT abc_load")
        log.warning("dim_sku not found — abc_vol defaults to None")
        return {}


def _load_lead_time_map(cur: Any) -> dict[tuple[str, str], dict]:
    """Load lead time profiles from dim_item_lead_time_profile."""
    try:
        cur.execute("SAVEPOINT lt_load")
        cur.execute("""
            SELECT item_id, loc, lt_mean_days, lt_std_days
            FROM dim_item_lead_time_profile
        """)
        result: dict[tuple[str, str], dict] = {}
        for row in cur.fetchall():
            result[(row[0], row[1])] = {
                "lt_mean_days": float(row[2]) if row[2] is not None else None,
                "lt_std_days": float(row[3]) if row[3] is not None else None,
            }
        return result
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT lt_load")
        log.warning("dim_item_lead_time_profile not found — using default LT values")
        return {}


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def run(
    models: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Compute SS, EOQ, ROP for each model_id in staging and write results.

    Args:
        models:  Optional list of model_ids to process (default: all).
        dry_run: If True, compute but do not write to DB.

    Returns:
        dict with per_model stats and totals.
    """
    # -- Load configs -----------------------------------------------------------
    with profiled_section("load_config"):
        ss_cfg = _load_ss_config()
        eoq_cfg = _load_eoq_config()

        service_levels: dict[str, float] = ss_cfg["service_levels"]
        z_table: dict[str, float] = {
            str(k): float(v) for k, v in ss_cfg["z_table"].items()
        }
        lt_std_fallback_pct: float = float(ss_cfg.get("lt_std_fallback_pct", 0.20))

        # EOQ cost defaults
        costs = eoq_cfg["costs"]
        ordering_cost: float = float(costs["default_ordering_cost"])
        holding_cost_pct: float = float(costs["default_holding_cost_pct"])
        unit_cost: float = float(costs["default_unit_cost"])
        moq: float = float(costs["default_moq"])
        max_eoq_months: float = float(eoq_cfg["constraints"]["max_eoq_months_supply"])

    log.info(
        "Algorithm Inventory Comparison (models=%s, dry_run=%s)",
        models or "all",
        dry_run,
    )

    conn = psycopg.connect(**get_db_params())
    conn.autocommit = False

    with conn.cursor() as cur:
        # -- Load reference data once -------------------------------------------
        with profiled_section("load_reference_data"):
            available_models = _load_models(cur, models)
            if not available_models:
                log.warning("No models found in fact_production_forecast_staging")
                conn.close()
                return {"models_processed": 0, "total_rows": 0}

            log.info("Models to process: %s", available_models)

            # Validate user-requested models exist in staging
            if models:
                requested = set(models)
                available = set(available_models)
                missing = requested - available
                if missing:
                    log.warning(
                        "Models not found in staging: %s (available: %s)",
                        missing, available,
                    )

            abc_map = _load_abc_map(cur)
            lt_map = _load_lead_time_map(cur)
            log.info("  %d ABC entries, %d LT profiles loaded", len(abc_map), len(lt_map))

        # -- Bulk-delete all target model rows before re-inserting ---------------
        if not dry_run:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM fact_inventory_algorithm_comparison WHERE model_id = ANY(%s)",
                    [available_models],
                )
                conn.commit()
                log.info("Bulk-deleted prior rows for %d models", len(available_models))

    # -- Process each model -----------------------------------------------------
    model_stats: dict[str, dict] = {}
    total_rows = 0

    for model_id in available_models:
        with profiled_section(f"process_{model_id}"):
            with conn.cursor() as cur:
                forecast_rows = _load_forecast_stats(cur, model_id)

            log.info("  Model %s: %d DFUs", model_id, len(forecast_rows))

            insert_rows: list[tuple] = []
            skipped = 0

            for fr in forecast_rows:
                item_id: str = fr["item_id"]
                loc: str = fr["loc"]
                avg_qty = float(fr["avg_qty"]) if fr["avg_qty"] is not None else 0.0
                std_qty = float(fr["std_qty"]) if fr["std_qty"] is not None else 0.0

                # -- CV -------------------------------------------------------
                cv = std_qty / avg_qty if avg_qty > 0 else None

                # -- ABC from dim_sku -----------------------------------------
                abc_vol = abc_map.get((item_id, loc))

                # -- Lead time ------------------------------------------------
                lt_rec = lt_map.get((item_id, loc), {})
                lt_mean_raw = lt_rec.get("lt_mean_days")
                lt_std_raw = lt_rec.get("lt_std_days")

                lt_mean_days = float(lt_mean_raw) if lt_mean_raw is not None else 14.0
                if lt_mean_days <= 0:
                    skipped += 1
                    continue

                lt_std_days = (
                    float(lt_std_raw) if lt_std_raw is not None
                    else lt_mean_days * lt_std_fallback_pct
                )

                # -- Service level & Z ----------------------------------------
                sl = _get_service_level(abc_vol, service_levels)
                z = _get_z_score(sl, z_table)

                # -- Daily demand metrics from forecast -----------------------
                avg_daily = avg_qty / DAYS_PER_MONTH
                sigma_d_daily = std_qty / math.sqrt(DAYS_PER_MONTH) if std_qty > 0 else 0.0

                # -- SS -------------------------------------------------------
                ss_combined = _compute_ss_combined(
                    z, sigma_d_daily, lt_mean_days, avg_daily, lt_std_days,
                )
                ss_demand_only = _compute_ss_demand_only(
                    z, sigma_d_daily, lt_mean_days,
                )

                # -- EOQ ------------------------------------------------------
                annual_demand = avg_qty * 12.0
                raw_eoq = _compute_eoq(
                    annual_demand, ordering_cost, holding_cost_pct, unit_cost,
                )
                if raw_eoq is not None:
                    eff_eoq = _compute_effective_eoq(raw_eoq, moq, max_eoq_months, avg_qty)
                else:
                    raw_eoq = 0.0
                    eff_eoq = moq

                # -- ROP & Cycle stock ----------------------------------------
                rop = avg_daily * lt_mean_days + ss_combined
                cycle_stock = eff_eoq / 2.0

                insert_rows.append((
                    model_id, item_id, loc,
                    avg_qty, std_qty, cv,
                    ss_combined, ss_demand_only,
                    raw_eoq, eff_eoq,
                    rop, cycle_stock,
                    abc_vol, sl,
                ))

            model_stats[model_id] = {
                "dfus": len(forecast_rows),
                "computed": len(insert_rows),
                "skipped": skipped,
            }

            # -- Write to DB per model ----------------------------------------
            if not dry_run and insert_rows:
                with profiled_section(f"write_{model_id}"):
                    with conn.cursor() as cur:
                        insert_sql = """
                            INSERT INTO fact_inventory_algorithm_comparison (
                                model_id, item_id, loc,
                                forecast_avg_monthly, forecast_std_monthly, forecast_cv,
                                ss_combined, ss_demand_only,
                                eoq, effective_eoq,
                                reorder_point, cycle_stock,
                                abc_vol, service_level
                            ) VALUES (
                                %s, %s, %s,
                                %s, %s, %s,
                                %s, %s,
                                %s, %s,
                                %s, %s,
                                %s, %s
                            )
                        """
                        cur.executemany(insert_sql, insert_rows)
                    conn.commit()
                    log.info(
                        "  %s: inserted %d rows",
                        model_id,
                        len(insert_rows),
                    )

            total_rows += len(insert_rows)

    conn.close()

    # -- Summary ---------------------------------------------------------------
    log.info("=== Algorithm Comparison Summary ===")
    for mid, stats in model_stats.items():
        log.info(
            "  %-25s dfus=%d  computed=%d  skipped=%d",
            mid, stats["dfus"], stats["computed"], stats["skipped"],
        )
    log.info("Total rows written: %d%s", total_rows, " (DRY RUN)" if dry_run else "")

    return {
        "models_processed": len(available_models),
        "total_rows": total_rows,
        "per_model": model_stats,
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare inventory targets (SS, EOQ, ROP) across forecast algorithms",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of model_ids to compare (default: all in staging)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute but do not write to DB",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model_list = args.models.split(",") if args.models else None
    result = run(models=model_list, dry_run=args.dry_run)
    log.info("Result: %s", result)
    sys.exit(0)
