"""
run_supply_chain_scenario.py — F4.4 Supply Chain Disruption Scenario Planning

Simulates supply chain disruption scenarios (supplier delays, capacity constraints,
demand shocks) and computes their impact on inventory position, service levels,
and financial exposure.

Usage:
    uv run python scripts/run_supply_chain_scenario.py --scenario-id 1
    uv run python scripts/run_supply_chain_scenario.py --action list
    uv run python scripts/run_supply_chain_scenario.py --action create \
        --scenario-name "China port delay" --disruption-type supplier_delay \
        --impact-pct 50 --duration-weeks 4
    uv run python scripts/run_supply_chain_scenario.py --scenario-id 1 --dry-run

Config: config/supply_scenario_config.yaml
"""

from __future__ import annotations

import argparse
import yaml
import psycopg
from typing import Optional

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.db import get_db_params
from common.planning_date import get_planning_date
from common.services.perf_profiler import profiled_section

CONFIG_PATH = "config/supply_scenario_config.yaml"

DISRUPTION_TYPES = {
    "supplier_delay",       # Supplier lead time extended by N weeks
    "capacity_constraint",  # Supplier can only ship X% of ordered quantity
    "demand_shock",         # Unexpected demand surge (e.g., competitor exit)
    "transport_disruption", # Freight unavailable for N weeks
    "quality_hold",         # % of incoming stock placed on hold
}


def load_config(path: str = CONFIG_PATH) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f).get("supply_scenario", {})
    except FileNotFoundError:
        return {
            "simulation_horizon_weeks": 13,
            "service_level_target": 0.95,
            "stockout_cost_per_unit": 10.0,
            "excess_holding_cost_pct": 0.25,
        }


def compute_adjusted_lead_time(
    base_lt_days: float,
    disruption_type: str,
    impact_pct: float,
    duration_weeks: int,
) -> tuple[float, float]:
    """
    Compute adjusted lead time under disruption.

    For supplier_delay / transport_disruption:
        adjusted_lt = base_lt + (duration_weeks × 7) × (impact_pct / 100)

    For other disruption types, lead time is unchanged.

    Args:
        base_lt_days: Normal lead time in days
        disruption_type: Type of disruption
        impact_pct: Severity (0-100%)
        duration_weeks: How many weeks the disruption lasts

    Returns:
        (adjusted_lt_days, lt_increase_days)

    Examples:
        compute_adjusted_lead_time(10, 'supplier_delay', 50, 4)
        → (24.0, 14.0)  # 10 + (4×7 × 0.50) = 10 + 14 = 24 days
    """
    if disruption_type in {"supplier_delay", "transport_disruption"}:
        increase = (duration_weeks * 7.0) * (impact_pct / 100.0)
        adjusted = base_lt_days + increase
        return round(adjusted, 1), round(increase, 1)
    return base_lt_days, 0.0


def compute_available_supply(
    ordered_qty: float,
    disruption_type: str,
    impact_pct: float,
) -> tuple[float, float]:
    """
    Compute available supply quantity under disruption.

    For capacity_constraint / quality_hold:
        available = ordered_qty × (1 - impact_pct / 100)

    Args:
        ordered_qty: Normal ordered quantity
        disruption_type: Type of disruption
        impact_pct: Severity (0-100%)

    Returns:
        (available_qty, shortfall_qty)

    Examples:
        compute_available_supply(100, 'capacity_constraint', 40)
        → (60.0, 40.0)  # 40% capacity constraint leaves 60 units
    """
    if disruption_type in {"capacity_constraint", "quality_hold"}:
        available = ordered_qty * (1.0 - impact_pct / 100.0)
        shortfall = ordered_qty - available
        return round(max(0.0, available), 2), round(max(0.0, shortfall), 2)
    return ordered_qty, 0.0


def compute_stockout_days(
    on_hand: float,
    daily_demand: float,
    adjusted_lt_days: float,
    available_supply: float,
) -> float:
    """
    Estimate number of stockout days during the disruption.

    Depletion time = (on_hand + available_supply) / daily_demand
    Stockout days = max(0, adjusted_lt_days - depletion_time)

    Args:
        on_hand: Current on-hand inventory
        daily_demand: Average daily demand
        adjusted_lt_days: Lead time under disruption
        available_supply: Supply available during disruption

    Returns:
        Estimated stockout days (0 if no stockout expected)
    """
    if daily_demand <= 0:
        return 0.0
    depletion_time = (on_hand + available_supply) / daily_demand
    return max(0.0, round(adjusted_lt_days - depletion_time, 1))


def compute_scenario_financial_impact(
    stockout_days: float,
    daily_demand: float,
    unit_cost: float,
    stockout_cost_per_unit: float,
    excess_qty: float = 0.0,
    holding_cost_pct: float = 0.25,
) -> dict:
    """
    Compute financial impact of a supply chain disruption.

    stockout_value = stockout_days × daily_demand × stockout_cost_per_unit
    holding_cost   = excess_qty × unit_cost × (holding_cost_pct / 52) per week

    Returns:
        dict with stockout_units, stockout_cost, holding_cost, total_impact
    """
    stockout_units = stockout_days * daily_demand
    stockout_cost = stockout_units * stockout_cost_per_unit
    holding_cost = excess_qty * unit_cost * (holding_cost_pct / 52.0)

    return {
        "stockout_units": round(stockout_units, 1),
        "stockout_cost": round(stockout_cost, 2),
        "holding_cost": round(holding_cost, 2),
        "total_impact": round(stockout_cost + holding_cost, 2),
    }


def fetch_scenario(conn: psycopg.Connection, scenario_id: int) -> Optional[dict]:
    """Fetch a scenario from fact_supply_scenarios."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT scenario_id, scenario_name, scenario_type,
                   shock_parameters, affected_items, affected_locations,
                   affected_suppliers, horizon_months, status, created_by
            FROM fact_supply_scenarios
            WHERE scenario_id = %s
            """,
            (scenario_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    import json as _json
    shock_params = row[3] if isinstance(row[3], dict) else (_json.loads(row[3]) if row[3] else {})
    return {
        "scenario_id": row[0],
        "scenario_name": row[1],
        "disruption_type": row[2] or "supplier_delay",
        "impact_pct": float(shock_params.get("impact_pct", 50)),
        "duration_weeks": int(shock_params.get("duration_weeks", 4)),
        "affected_supplier_id": row[6],
        "affected_item_id": row[4],
        "affected_loc": row[5],
        "status": row[8],
        "created_by": row[9],
    }


def fetch_affected_dfus(
    conn: psycopg.Connection,
    scenario: dict,
) -> list[dict]:
    """Fetch DFUs affected by the scenario."""
    conditions = []
    params: list = []

    if scenario.get("affected_item_id"):
        conditions.append("inv.item_id = %s")
        params.append(scenario["affected_item_id"])
    if scenario.get("affected_loc"):
        conditions.append("inv.loc = %s")
        params.append(scenario["affected_loc"])

    where = " AND ".join(conditions) if conditions else "TRUE"

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT DISTINCT ON (inv.item_id, inv.loc)
                inv.item_id, inv.loc, inv.qty_on_hand,
                COALESCE(inv.mtd_sales, 0) / GREATEST(EXTRACT(DAY FROM inv.snapshot_date), 1) AS avg_daily_sales,
                COALESCE(ic.unit_cost, 0) AS unit_cost,
                COALESCE(lt.mean_lt_days, 14) AS base_lt_days
            FROM fact_inventory_snapshot inv
            LEFT JOIN dim_item_cost ic
                ON ic.item_id = inv.item_id AND ic.loc = inv.loc
                AND ic.effective_to IS NULL
            LEFT JOIN dim_lead_time_profile lt
                ON lt.item_category = inv.item_id AND lt.loc = inv.loc
            WHERE {where}
            ORDER BY inv.item_id, inv.loc, inv.snapshot_date DESC
            """,
            params,
        )
        rows = cur.fetchall()
    return [
        {
            "item_id": r[0],
            "loc": r[1],
            "on_hand": float(r[2]) if r[2] else 0.0,
            "daily_demand": float(r[3]) if r[3] else 0.0,
            "unit_cost": float(r[4]),
            "base_lt_days": float(r[5]),
        }
        for r in rows
    ]


def upsert_scenario_results(conn: psycopg.Connection, rows: list[dict]) -> int:
    sql = """
        INSERT INTO fact_scenario_results (
            scenario_id, item_id, loc, plan_month,
            baseline_qty, scenario_qty, impact_qty, impact_pct,
            stockout_risk_days, excess_risk_qty, mitigation_option
        ) VALUES (
            %(scenario_id)s, %(item_id)s, %(loc)s, %(run_date)s,
            %(normal_supply_qty)s, %(available_supply_qty)s, %(supply_shortfall_qty)s,
            %(impact_pct)s, %(stockout_days_estimated)s, 0, %(mitigation)s
        )
        ON CONFLICT (scenario_id, item_id, loc, plan_month)
        DO UPDATE SET
            baseline_qty            = EXCLUDED.baseline_qty,
            scenario_qty            = EXCLUDED.scenario_qty,
            impact_qty              = EXCLUDED.impact_qty,
            impact_pct              = EXCLUDED.impact_pct,
            stockout_risk_days      = EXCLUDED.stockout_risk_days,
            computed_at             = NOW()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def create_scenario(
    conn: psycopg.Connection,
    scenario_name: str,
    disruption_type: str,
    impact_pct: float,
    duration_weeks: int,
    created_by: str = "system",
    affected_supplier_id: Optional[str] = None,
    affected_item_id: Optional[str] = None,
    affected_loc: Optional[str] = None,
) -> int:
    """Create a new supply chain scenario. Returns scenario_id."""
    import json as _json
    shock_params = _json.dumps({
        "impact_pct": impact_pct,
        "duration_weeks": duration_weeks,
    })
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO fact_supply_scenarios (
                scenario_name, scenario_type, shock_parameters,
                affected_items, affected_locations, affected_suppliers,
                status, created_by
            ) VALUES (%s, %s, %s::jsonb, %s, %s, %s, 'draft', %s)
            RETURNING scenario_id
            """,
            (
                scenario_name, disruption_type, shock_params,
                affected_item_id, affected_loc, affected_supplier_id,
                created_by,
            ),
        )
        row = cur.fetchone()
    return row[0]


def run(
    action: str = "run",
    scenario_id: Optional[int] = None,
    scenario_name: Optional[str] = None,
    disruption_type: Optional[str] = None,
    impact_pct: float = 50.0,
    duration_weeks: int = 4,
    created_by: str = "system",
    dry_run: bool = False,
) -> dict:
    """Main entry point."""
    with profiled_section("load_config"):
        cfg = load_config()
        stockout_cost_per_unit = cfg.get("stockout_cost_per_unit", 10.0)
        holding_cost_pct = cfg.get("excess_holding_cost_pct", 0.25)

    with psycopg.connect(**get_db_params()) as conn:
        if action == "create":
            if not scenario_name or not disruption_type:
                raise ValueError("--scenario-name and --disruption-type required for 'create'")

            if dry_run:
                return {
                    "action": "create",
                    "scenario_name": scenario_name,
                    "disruption_type": disruption_type,
                    "impact_pct": impact_pct,
                    "duration_weeks": duration_weeks,
                    "dry_run": True,
                }

            new_id = create_scenario(
                conn, scenario_name, disruption_type, impact_pct,
                duration_weeks, created_by,
            )
            conn.commit()
            return {"action": "create", "scenario_id": new_id, "status": "draft"}

        elif action == "list":
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT scenario_id, scenario_name, scenario_type,
                           shock_parameters, status
                    FROM fact_supply_scenarios
                    ORDER BY scenario_id DESC
                    LIMIT 20
                    """
                )
                rows = cur.fetchall()
            return {
                "scenarios": [
                    {
                        "scenario_id": r[0], "scenario_name": r[1],
                        "scenario_type": r[2], "shock_parameters": r[3],
                        "status": r[4],
                    }
                    for r in rows
                ]
            }

        elif action == "run":
            if not scenario_id:
                raise ValueError("--scenario-id required for 'run' action")

            with profiled_section("load_scenario_data"):
                scenario = fetch_scenario(conn, scenario_id)
                if not scenario:
                    raise ValueError(f"Scenario {scenario_id} not found")

                dfus = fetch_affected_dfus(conn, scenario)

            rows_to_write: list[dict] = []
            total_impact = 0.0
            today = get_planning_date()

            with profiled_section("run_simulation"):
                for dfu in dfus:
                    adj_lt, lt_increase = compute_adjusted_lead_time(
                        dfu["base_lt_days"], scenario["disruption_type"],
                        scenario["impact_pct"], scenario["duration_weeks"],
                    )
                    # Assume normal supply = 30-day average demand × some factor
                    normal_supply = dfu["daily_demand"] * 30.0
                    avail_supply, shortfall = compute_available_supply(
                        normal_supply, scenario["disruption_type"], scenario["impact_pct"]
                    )
                    stockout_days = compute_stockout_days(
                        dfu["on_hand"], dfu["daily_demand"], adj_lt, avail_supply
                    )
                    fin = compute_scenario_financial_impact(
                        stockout_days, dfu["daily_demand"], dfu["unit_cost"],
                        stockout_cost_per_unit, 0.0, holding_cost_pct,
                    )
                    total_impact += fin["total_impact"]

                    impact_pct_val = round(shortfall / normal_supply * 100.0, 2) if normal_supply > 0 else 0.0
                    rows_to_write.append({
                        "scenario_id": scenario_id,
                        "item_id": dfu["item_id"],
                        "loc": dfu["loc"],
                        "run_date": today,
                        "normal_supply_qty": round(normal_supply, 2),
                        "available_supply_qty": avail_supply,
                        "supply_shortfall_qty": shortfall,
                        "impact_pct": impact_pct_val,
                        "stockout_days_estimated": stockout_days,
                        "mitigation": "expedite" if stockout_days > 0 else "none",
                    })

            with profiled_section("write_results"):
                rows_written = 0
                if not dry_run and rows_to_write:
                    rows_written = upsert_scenario_results(conn, rows_to_write)
                    # Update scenario status to 'completed'
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE fact_supply_scenarios SET status = 'completed', last_run_at = NOW() WHERE scenario_id = %s",
                            (scenario_id,),
                        )
                    conn.commit()

            return {
                "scenario_id": scenario_id,
                "dfus_analyzed": len(rows_to_write),
                "rows_written": rows_written,
                "total_financial_impact_usd": round(total_impact, 2),
                "dry_run": dry_run,
            }
        else:
            raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run supply chain scenario analysis")
    parser.add_argument("--action", choices=["create", "list", "run"], default="run")
    parser.add_argument("--scenario-id", type=int)
    parser.add_argument("--scenario-name")
    parser.add_argument("--disruption-type", choices=list(DISRUPTION_TYPES))
    parser.add_argument("--impact-pct", type=float, default=50.0,
                        help="Disruption severity 0-100%")
    parser.add_argument("--duration-weeks", type=int, default=4)
    parser.add_argument("--created-by", default="system")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(
        action=args.action,
        scenario_id=args.scenario_id,
        scenario_name=args.scenario_name,
        disruption_type=args.disruption_type,
        impact_pct=args.impact_pct,
        duration_weeks=args.duration_weeks,
        created_by=args.created_by,
        dry_run=args.dry_run,
    )
    print(result)
