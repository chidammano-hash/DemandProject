"""IPfeature4: EOQ & Cycle Stock Calculator.

Reads demand stats from dim_sku (populated by IPfeature1) and computes
EOQ, cycle stock, order frequency, and annual cost metrics per item-location.
Upserts results into fact_eoq_targets.

Usage:
    uv run python scripts/compute_eoq.py
    uv run python scripts/compute_eoq.py --config config/inventory/eoq_config.yaml
    uv run python scripts/compute_eoq.py --dry-run
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.core.db import get_db_params
from common.services.perf_profiler import profiled_section
from common.core.utils import load_config as _load_config


# ---------------------------------------------------------------------------
# Pure computation helpers (importable by unit tests)
# ---------------------------------------------------------------------------

def compute_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_pct: float,
    unit_cost: float,
) -> float | None:
    """Classic Wilson EOQ formula.

    EOQ = sqrt(2 × D × S / (H × C))

    Returns None if inputs prevent a valid calculation (e.g. zero denominator).
    """
    denominator = holding_cost_pct * unit_cost
    if denominator <= 0 or annual_demand <= 0:
        return None
    return math.sqrt(2 * annual_demand * ordering_cost / denominator)


def compute_effective_eoq(
    eoq: float,
    moq: float,
    max_eoq_months_supply: float,
    demand_mean_monthly: float,
) -> float:
    """Apply MOQ floor and months-supply cap to the raw EOQ.

    effective_eoq = clamp(max(eoq, moq), lo=moq, hi=cap)
    """
    cap = max_eoq_months_supply * demand_mean_monthly if demand_mean_monthly > 0 else float("inf")
    effective = max(eoq, moq)
    if cap > 0:
        effective = min(effective, cap)
    return effective


def compute_eoq_metrics(
    demand_mean_monthly: float,
    config: dict,
    ordering_cost: float | None = None,
    holding_cost_pct: float | None = None,
    unit_cost: float | None = None,
    moq: float | None = None,
) -> dict[str, Any] | None:
    """Compute full EOQ metric set for one DFU.

    Uses config defaults for any cost parameter not explicitly provided.
    Returns None when annual_demand is below the configured minimum.
    """
    costs = config["costs"]
    constraints = config["constraints"]

    s = ordering_cost if ordering_cost is not None else costs["default_ordering_cost"]
    h = holding_cost_pct if holding_cost_pct is not None else costs["default_holding_cost_pct"]
    c = unit_cost if unit_cost is not None else costs["default_unit_cost"]
    q_min = moq if moq is not None else costs["default_moq"]

    annual_demand = demand_mean_monthly * 12.0
    if annual_demand < constraints["min_annual_demand"]:
        return None

    raw_eoq = compute_eoq(annual_demand, s, h, c)
    if raw_eoq is None:
        return None

    eff_eoq = compute_effective_eoq(
        raw_eoq,
        q_min,
        constraints["max_eoq_months_supply"],
        demand_mean_monthly,
    )

    cycle_stock = eff_eoq / 2.0
    order_freq = annual_demand / eff_eoq if eff_eoq > 0 else 0.0
    annual_holding = h * c * cycle_stock
    annual_order = s * annual_demand / eff_eoq if eff_eoq > 0 else 0.0
    total_cost = annual_holding + annual_order

    return {
        "annual_demand": annual_demand,
        "ordering_cost": s,
        "holding_cost_pct": h,
        "unit_cost": c,
        "moq": q_min,
        "eoq": raw_eoq,
        "effective_eoq": eff_eoq,
        "eoq_cycle_stock": cycle_stock,
        "order_frequency": order_freq,
        "annual_holding_cost": annual_holding,
        "annual_order_cost": annual_order,
        "total_annual_cost": total_cost,
    }


def sensitivity_curve(
    avg_demand_monthly: float,
    config: dict,
) -> list[dict[str, float]]:
    """Compute EOQ and total cost as ordering_cost varies.

    Returns a list of {ordering_cost, eoq, total_annual_cost} dicts.
    """
    sens = config["sensitivity"]
    costs = config["costs"]
    h = costs["default_holding_cost_pct"]
    c = costs["default_unit_cost"]
    annual_demand = avg_demand_monthly * 12.0

    s_min = sens["ordering_cost_min"]
    s_max = sens["ordering_cost_max"]
    steps = max(2, int(sens["ordering_cost_steps"]))

    step_size = (s_max - s_min) / (steps - 1)
    result: list[dict[str, float]] = []

    for i in range(steps):
        s = s_min + i * step_size
        raw_eoq = compute_eoq(annual_demand, s, h, c)
        if raw_eoq is None:
            continue
        eff_eoq = compute_effective_eoq(
            raw_eoq,
            costs["default_moq"],
            config["constraints"]["max_eoq_months_supply"],
            avg_demand_monthly,
        )
        annual_holding = h * c * (eff_eoq / 2.0)
        annual_order = s * annual_demand / eff_eoq if eff_eoq > 0 else 0.0
        result.append({
            "ordering_cost": round(s, 2),
            "eoq": round(raw_eoq, 4),
            "effective_eoq": round(eff_eoq, 4),
            "total_annual_cost": round(annual_holding + annual_order, 4),
        })

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(config: dict, dry_run: bool = False) -> dict[str, int]:
    """Load demand stats from dim_sku, compute EOQ, upsert fact_eoq_targets.

    Returns {processed, updated, skipped}.
    """
    import psycopg

    with profiled_section("load_config"):
        batch_size = config["batch"]["batch_size"]
        conn_params = get_db_params()

    load_sql = """
        SELECT
            item_id  AS item_id,
            loc,
            abc_vol,
            demand_mean
        FROM dim_sku
        WHERE demand_mean IS NOT NULL
          AND demand_mean > 0
        ORDER BY item_id, loc
    """

    upsert_sql = """
        INSERT INTO fact_eoq_targets (
            item_id, loc, abc_vol,
            demand_mean_monthly, annual_demand,
            ordering_cost, holding_cost_pct, unit_cost, moq,
            eoq, effective_eoq, eoq_cycle_stock, order_frequency,
            annual_holding_cost, annual_order_cost, total_annual_cost,
            computed_at
        ) VALUES (
            %(item_id)s, %(loc)s, %(abc_vol)s,
            %(demand_mean_monthly)s, %(annual_demand)s,
            %(ordering_cost)s, %(holding_cost_pct)s, %(unit_cost)s, %(moq)s,
            %(eoq)s, %(effective_eoq)s, %(eoq_cycle_stock)s, %(order_frequency)s,
            %(annual_holding_cost)s, %(annual_order_cost)s, %(total_annual_cost)s,
            %(computed_at)s
        )
        ON CONFLICT (item_id, loc) DO UPDATE SET
            abc_vol              = EXCLUDED.abc_vol,
            demand_mean_monthly  = EXCLUDED.demand_mean_monthly,
            annual_demand        = EXCLUDED.annual_demand,
            ordering_cost        = EXCLUDED.ordering_cost,
            holding_cost_pct     = EXCLUDED.holding_cost_pct,
            unit_cost            = EXCLUDED.unit_cost,
            moq                  = EXCLUDED.moq,
            eoq                  = EXCLUDED.eoq,
            effective_eoq        = EXCLUDED.effective_eoq,
            eoq_cycle_stock      = EXCLUDED.eoq_cycle_stock,
            order_frequency      = EXCLUDED.order_frequency,
            annual_holding_cost  = EXCLUDED.annual_holding_cost,
            annual_order_cost    = EXCLUDED.annual_order_cost,
            total_annual_cost    = EXCLUDED.total_annual_cost,
            computed_at          = EXCLUDED.computed_at
    """

    processed = 0
    updated = 0
    skipped = 0
    now = datetime.now(timezone.utc)

    with psycopg.connect(**conn_params) as conn:
        with profiled_section("load_dfu_data"):
            with conn.cursor() as cur:
                cur.execute(load_sql)
                rows = cur.fetchall()

        with profiled_section("compute_eoq"):
            all_rows: list[dict] = []
            for item_id, loc, abc_vol, demand_mean in rows:
                processed += 1
                demand_mean_f = float(demand_mean)
                metrics = compute_eoq_metrics(demand_mean_f, config)
                if metrics is None:
                    skipped += 1
                    continue

                row = {
                    "item_id": item_id,
                    "loc": loc,
                    "abc_vol": abc_vol,
                    "demand_mean_monthly": demand_mean_f,
                    **metrics,
                    "computed_at": now,
                }
                all_rows.append(row)

        with profiled_section("batch_upsert"):
            if all_rows and not dry_run:
                for i in range(0, len(all_rows), batch_size):
                    batch = all_rows[i : i + batch_size]
                    with conn.cursor() as cur:
                        cur.executemany(upsert_sql, batch)
                    updated += len(batch)
                conn.commit()

    return {"processed": processed, "updated": updated, "skipped": skipped}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute EOQ & cycle stock targets.")
    parser.add_argument(
        "--config",
        default="config/inventory/eoq_config.yaml",
        help="Path to YAML config (default: config/inventory/eoq_config.yaml)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip DB writes")
    args = parser.parse_args()

    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    cfg = _load_config(cfg_name)

    summary = run(cfg, dry_run=args.dry_run)
    print(
        f"[eoq] processed={summary['processed']} "
        f"updated={summary['updated']} skipped={summary['skipped']}"
        + (" (DRY RUN)" if args.dry_run else "")
    )
