"""
run_sop_cycle.py — F4.2 Sales & Operations Planning (S&OP) Module

Creates and advances S&OP cycles through the stage machine:
  demand_review → supply_review → pre_sop → executive_sop → approved → closed

Also populates demand review data and supply constraints, and
generates the approved plan snapshot when executive_sop is reached.

Usage:
    uv run python scripts/run_sop_cycle.py --action create --cycle-month 2026-05-01
    uv run python scripts/run_sop_cycle.py --action advance --cycle-id 1
    uv run python scripts/run_sop_cycle.py --action populate-demand --cycle-id 1
    uv run python scripts/run_sop_cycle.py --dry-run --action create --cycle-month 2026-05-01

Config: config/sop_config.yaml
"""

from __future__ import annotations

import argparse
import yaml
import psycopg
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Optional

from common.db import get_db_params

CONFIG_PATH = "config/sop_config.yaml"

STAGE_ORDER = [
    "demand_review",
    "supply_review",
    "pre_sop",
    "executive_sop",
    "approved",
    "closed",
]


def load_config(path: str = CONFIG_PATH) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f).get("sop", {})
    except FileNotFoundError:
        return {
            "planning_horizon_months": 12,
            "demand_review_day": 5,
            "supply_review_day": 10,
            "pre_sop_day": 15,
            "executive_sop_day": 20,
        }


def next_stage(current: str) -> str:
    """
    Return the next stage in the S&OP stage machine.

    Raises ValueError if current is the final stage.

    Examples:
        next_stage('demand_review') → 'supply_review'
        next_stage('executive_sop') → 'approved'
        next_stage('closed') → raises ValueError
    """
    idx = STAGE_ORDER.index(current)
    if idx >= len(STAGE_ORDER) - 1:
        raise ValueError(f"Stage '{current}' is already the final stage")
    return STAGE_ORDER[idx + 1]


def is_terminal_stage(stage: str) -> bool:
    """Return True if this stage is the final stage (closed)."""
    return stage == "closed"


def compute_cycle_dates(cycle_month: date, cfg: dict) -> dict:
    """
    Compute scheduled stage dates for a cycle month.

    Args:
        cycle_month: First day of the planning month
        cfg: sop config dict

    Returns:
        Dict with scheduled_{stage}_date keys
    """
    base_month = cycle_month - relativedelta(months=1)  # Stage dates are in prior month
    return {
        "demand_review_date": base_month.replace(day=cfg.get("demand_review_day", 5)),
        "supply_review_date": base_month.replace(day=cfg.get("supply_review_day", 10)),
        "pre_sop_date": base_month.replace(day=cfg.get("pre_sop_day", 15)),
        "executive_sop_date": base_month.replace(day=cfg.get("executive_sop_day", 20)),
    }


def create_sop_cycle(
    conn: psycopg.Connection,
    cycle_month: date,
    cfg: dict,
    run_by: str = "system",
) -> int:
    """
    Create a new S&OP cycle record.

    Args:
        conn: DB connection
        cycle_month: First day of the planning month
        cfg: sop config
        run_by: User creating the cycle

    Returns:
        cycle_id of the created record
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO fact_sop_cycles (
                cycle_month, status, run_by
            ) VALUES (%s, 'demand_review', %s)
            RETURNING cycle_id
            """,
            (cycle_month, run_by),
        )
        row = cur.fetchone()
    return row[0]


def advance_sop_cycle(
    conn: psycopg.Connection,
    cycle_id: int,
    performed_by: str,
) -> str:
    """
    Advance an S&OP cycle to the next stage.

    Args:
        conn: DB connection
        cycle_id: ID of the cycle to advance
        performed_by: User advancing the stage

    Returns:
        The new stage name

    Raises:
        ValueError: If cycle_id not found or stage is already terminal
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM fact_sop_cycles WHERE cycle_id = %s",
            (cycle_id,),
        )
        row = cur.fetchone()

    if not row:
        raise ValueError(f"S&OP cycle {cycle_id} not found")

    current = row[0]
    new_stage = next_stage(current)

    # Build update — set the stage-specific timestamp for the completed stage
    update_fields = [
        "status = %s",
        "facilitated_by = %s",
        "updated_at = NOW()",
    ]
    params: list = [new_stage, performed_by]

    # Stage-specific timestamp columns (schema: demand_review_at, supply_review_at, etc.)
    _stage_ts_map = {
        "demand_review": "demand_review_at",
        "supply_review": "supply_review_at",
        "pre_sop": "pre_sop_at",
        "executive_sop": "executive_sop_at",
    }
    if current in _stage_ts_map:
        ts_col = _stage_ts_map[current]
        update_fields.append(f"{ts_col} = NOW()")

    params.append(cycle_id)

    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE fact_sop_cycles SET {', '.join(update_fields)} WHERE cycle_id = %s",
            params,
        )

    return new_stage


def populate_demand_review(
    conn: psycopg.Connection,
    cycle_id: int,
    cycle_month: date,
    horizon_months: int = 12,
) -> int:
    """
    Populate fact_sop_demand_review from the current champion forecast.

    Reads fact_external_forecast_monthly (model_id='champion') for the
    planning horizon and aggregates by item category (via dim_item join).

    Returns:
        Number of rows inserted
    """
    horizon_end = cycle_month + relativedelta(months=horizon_months)
    sql = """
        INSERT INTO fact_sop_demand_review (
            cycle_id, item_category,
            statistical_demand_qty, consensus_demand_qty
        )
        SELECT
            %s,
            COALESCE(i.category, 'Unknown'),
            SUM(f.basefcst_pref),
            SUM(f.basefcst_pref)   -- start with statistical as consensus
        FROM fact_external_forecast_monthly f
        LEFT JOIN dim_item i ON i.item_no = f.dmdunit
        WHERE f.model_id = 'champion'
          AND f.startdate >= %s
          AND f.startdate < %s
        GROUP BY COALESCE(i.category, 'Unknown')
        ON CONFLICT (cycle_id, item_category)
        DO UPDATE SET
            statistical_demand_qty = EXCLUDED.statistical_demand_qty,
            consensus_demand_qty = EXCLUDED.consensus_demand_qty
    """
    with conn.cursor() as cur:
        cur.execute(sql, (cycle_id, cycle_month, horizon_end))
        return cur.rowcount


def populate_supply_constraints(
    conn: psycopg.Connection,
    cycle_id: int,
) -> int:
    """
    Populate supply constraints from current replenishment exceptions.

    Returns:
        Number of constraint rows inserted
    """
    sql = """
        INSERT INTO fact_sop_supply_constraints (
            cycle_id, constraint_type, supplier_id,
            impact_qty, impact_period, mitigation_status
        )
        SELECT
            %s,
            e.exception_type,
            NULL,
            e.recommended_order_qty,
            e.created_at::DATE,
            'open'
        FROM fact_replenishment_exceptions e
        WHERE e.status NOT IN ('rejected', 'closed')
        ON CONFLICT DO NOTHING
    """
    with conn.cursor() as cur:
        cur.execute(sql, (cycle_id,))
        return cur.rowcount


def generate_approved_plan_snapshot(
    conn: psycopg.Connection,
    cycle_id: int,
    cycle_month: date,
    approved_by: str,
) -> int:
    """
    Generate the approved plan snapshot from consensus demand.

    Called when a cycle is advanced to 'approved' stage.
    Creates fact_sop_approved_plan rows from the champion forecast
    for the planning horizon, using consensus demand quantities.

    Returns:
        Number of rows in the approved plan
    """
    cfg = load_config()
    horizon_months = cfg.get("planning_horizon_months", 12)
    horizon_end = cycle_month + relativedelta(months=horizon_months)

    # Update cycle with approved info
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE fact_sop_cycles
            SET approved_by = %s,
                approved_plan_version = %s,
                approved_at = NOW(),
                updated_at = NOW()
            WHERE cycle_id = %s
            """,
            (approved_by, f"sop_{cycle_month.isoformat()}_approved", cycle_id),
        )

    # Insert approved plan rows from champion forecast
    sql = """
        INSERT INTO fact_sop_approved_plan (
            cycle_id, item_no, loc, plan_month,
            approved_qty, statistical_qty, source, locked
        )
        SELECT
            %s,
            f.dmdunit,
            f.loc,
            f.startdate,
            f.basefcst_pref,
            f.basefcst_pref,
            'consensus',
            TRUE
        FROM fact_external_forecast_monthly f
        WHERE f.model_id = 'champion'
          AND f.startdate >= %s
          AND f.startdate < %s
        ON CONFLICT (cycle_id, item_no, loc, plan_month)
        DO UPDATE SET
            approved_qty = EXCLUDED.approved_qty,
            statistical_qty = EXCLUDED.statistical_qty
    """
    with conn.cursor() as cur:
        cur.execute(sql, (cycle_id, cycle_month, horizon_end))
        return cur.rowcount


def run(
    action: str,
    cycle_id: Optional[int] = None,
    cycle_month_str: Optional[str] = None,
    performed_by: str = "system",
    dry_run: bool = False,
) -> dict:
    """
    Main entry point.

    Actions:
        create           — Create a new S&OP cycle for the given month
        advance          — Advance an existing cycle to the next stage
        populate-demand  — Populate demand review data for a cycle
    """
    cfg = load_config()

    with psycopg.connect(**get_db_params()) as conn:
        if action == "create":
            if not cycle_month_str:
                raise ValueError("--cycle-month required for 'create' action")
            cycle_month = date.fromisoformat(cycle_month_str)

            if dry_run:
                dates = compute_cycle_dates(cycle_month, cfg)
                return {"action": "create", "cycle_month": str(cycle_month),
                        "scheduled_dates": {k: str(v) for k, v in dates.items()},
                        "dry_run": True}

            new_cycle_id = create_sop_cycle(conn, cycle_month, cfg, performed_by)
            conn.commit()
            return {"action": "create", "cycle_id": new_cycle_id, "cycle_month": str(cycle_month),
                    "stage": "demand_review"}

        elif action == "advance":
            if not cycle_id:
                raise ValueError("--cycle-id required for 'advance' action")

            if dry_run:
                with conn.cursor() as cur:
                    cur.execute("SELECT status FROM fact_sop_cycles WHERE cycle_id = %s", (cycle_id,))
                    row = cur.fetchone()
                if not row:
                    return {"error": f"Cycle {cycle_id} not found"}
                new_stage = next_stage(row[0])
                return {"action": "advance", "cycle_id": cycle_id, "new_stage": new_stage, "dry_run": True}

            new_stage = advance_sop_cycle(conn, cycle_id, performed_by)

            # If advancing to 'approved', generate approved plan snapshot
            if new_stage == "approved":
                with conn.cursor() as cur:
                    cur.execute("SELECT cycle_month FROM fact_sop_cycles WHERE cycle_id = %s", (cycle_id,))
                    row = cur.fetchone()
                if row:
                    n = generate_approved_plan_snapshot(conn, cycle_id, row[0], performed_by)

            conn.commit()
            return {"action": "advance", "cycle_id": cycle_id, "new_stage": new_stage}

        elif action == "populate-demand":
            if not cycle_id:
                raise ValueError("--cycle-id required for 'populate-demand' action")

            with conn.cursor() as cur:
                cur.execute("SELECT cycle_month FROM fact_sop_cycles WHERE cycle_id = %s",
                            (cycle_id,))
                row = cur.fetchone()

            if not row:
                return {"error": f"Cycle {cycle_id} not found"}

            cycle_month = row[0]
            horizon = cfg.get("planning_horizon_months", 12)

            if dry_run:
                return {"action": "populate-demand", "cycle_id": cycle_id,
                        "cycle_month": str(cycle_month), "dry_run": True}

            n_demand = populate_demand_review(conn, cycle_id, cycle_month, horizon)
            n_supply = populate_supply_constraints(conn, cycle_id)
            conn.commit()
            return {"action": "populate-demand", "cycle_id": cycle_id,
                    "demand_rows": n_demand, "supply_constraints": n_supply}

        else:
            raise ValueError(f"Unknown action: {action}. Use create, advance, or populate-demand")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S&OP cycle operations")
    parser.add_argument("--action", required=True,
                        choices=["create", "advance", "populate-demand"])
    parser.add_argument("--cycle-id", type=int)
    parser.add_argument("--cycle-month", help="YYYY-MM-DD (first of month)")
    parser.add_argument("--performed-by", default="system")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(
        action=args.action,
        cycle_id=args.cycle_id,
        cycle_month_str=args.cycle_month,
        performed_by=args.performed_by,
        dry_run=args.dry_run,
    )
    print(result)
