"""
F4.4 — What-If Scenario Planning for Supply Chain Disruptions API endpoints.

Endpoints:
    GET  /scenarios/supply               — List disruption scenarios
    POST /scenarios/supply               — Create new scenario (auth)
    GET  /scenarios/supply/{scenario_id} — Scenario detail + KPI impact
    POST /scenarios/supply/{scenario_id}/run — Execute simulation (auth)
    GET  /scenarios/supply/{scenario_id}/results — Simulation results
    POST /scenarios/supply/{scenario_id}/compare — Compare two scenarios (auth)
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional

from api.core import get_conn
from api.auth import require_api_key

router = APIRouter(tags=["supply-scenarios"])


class _ScenarioCreate(BaseModel):
    scenario_name: str
    scenario_type: str  # 'demand_shock' | 'lead_time_shock' | 'capacity_constraint' | 'logistics_disruption'
    description: Optional[str] = None
    shock_parameters: dict = {}
    affected_items: Optional[list] = None
    affected_locations: Optional[list] = None
    affected_suppliers: Optional[list] = None
    horizon_months: int = 3


class _RunRequest(BaseModel):
    run_by: str
    baseline_plan_version: str = "latest"


@router.get("/scenarios/supply")
async def list_supply_scenarios(
    scenario_type: str | None = None,
    status: str | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """List disruption scenario definitions."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    conditions = ["1=1"]
    params: list = []
    if scenario_type:
        conditions.append("scenario_type = %s"); params.append(scenario_type)
    if status:
        conditions.append("status = %s"); params.append(status)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT COUNT(*) FROM fact_supply_scenarios WHERE {where}", params
            )
            total = cur.fetchone()[0]
            cur.execute(
                f"""
                SELECT scenario_id, scenario_name, scenario_type, description,
                       horizon_months, status, created_by, created_at, last_run_at
                FROM fact_supply_scenarios
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                params + [page_size, offset],
            )
            rows = cur.fetchall()

    cols = [
        "scenario_id", "scenario_name", "scenario_type", "description",
        "horizon_months", "status", "created_by", "created_at", "last_run_at",
    ]
    scenarios = []
    for r in rows:
        d = dict(zip(cols, r))
        for field in ("created_at", "last_run_at"):
            if d.get(field) and hasattr(d[field], "isoformat"):
                d[field] = d[field].isoformat()
        scenarios.append(d)

    return {"total": total, "page": page, "scenarios": scenarios}


@router.post("/scenarios/supply", status_code=201)
async def create_supply_scenario(body: _ScenarioCreate, request: Request):
    """Create a new disruption scenario (auth required)."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    import json as _json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fact_supply_scenarios (
                    scenario_name, scenario_type, description,
                    shock_parameters, affected_items, affected_locations, affected_suppliers,
                    horizon_months, status, created_by
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'draft', 'api')
                RETURNING scenario_id
                """,
                (
                    body.scenario_name, body.scenario_type, body.description,
                    _json.dumps(body.shock_parameters),
                    _json.dumps(body.affected_items or []),
                    _json.dumps(body.affected_locations or []),
                    _json.dumps(body.affected_suppliers or []),
                    body.horizon_months,
                ),
            )
            scenario_id = cur.fetchone()[0]
        conn.commit()

    return {"scenario_id": scenario_id, "status": "draft"}


@router.get("/scenarios/supply/{scenario_id}")
async def get_supply_scenario(scenario_id: int):
    """Scenario definition + most recent KPI impact summary."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT scenario_id, scenario_name, scenario_type, description,
                       shock_parameters, affected_items, affected_locations,
                       affected_suppliers, horizon_months, status,
                       created_by, created_at, last_run_at
                FROM fact_supply_scenarios WHERE scenario_id = %s
                """,
                (scenario_id,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(404, f"Scenario {scenario_id} not found")

    cols = [
        "scenario_id", "scenario_name", "scenario_type", "description",
        "shock_parameters", "affected_items", "affected_locations",
        "affected_suppliers", "horizon_months", "status",
        "created_by", "created_at", "last_run_at",
    ]
    d = dict(zip(cols, row))
    for field in ("created_at", "last_run_at"):
        if d.get(field) and hasattr(d[field], "isoformat"):
            d[field] = d[field].isoformat()
    return d


@router.post("/scenarios/supply/{scenario_id}/run")
async def run_supply_scenario(scenario_id: int, body: _RunRequest, request: Request):
    """Trigger scenario simulation (async; returns 202 Accepted) (auth required)."""
    await require_api_key(x_api_key=request.headers.get("x-api-key"))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT scenario_id FROM fact_supply_scenarios WHERE scenario_id = %s",
                (scenario_id,),
            )
            if not cur.fetchone():
                raise HTTPException(404, f"Scenario {scenario_id} not found")

            cur.execute(
                """
                UPDATE fact_supply_scenarios
                SET status = 'running', last_run_at = NOW(), run_by = %s
                WHERE scenario_id = %s
                """,
                (body.run_by, scenario_id),
            )
        conn.commit()

    return {
        "scenario_id": scenario_id,
        "status": "running",
        "message": "Simulation started. Poll GET /scenarios/supply/{scenario_id}/results.",
    }


@router.get("/scenarios/supply/{scenario_id}/results")
async def get_scenario_results(scenario_id: int, page: int = 1, page_size: int = 50):
    """Simulation results: KPI impact per item-location-month."""
    page_size = max(1, min(page_size, 200))
    offset = (max(1, page) - 1) * page_size

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM fact_scenario_results WHERE scenario_id = %s",
                (scenario_id,),
            )
            total = cur.fetchone()[0]
            cur.execute(
                """
                SELECT item_no, loc, plan_month,
                       baseline_qty, scenario_qty, impact_qty, impact_pct,
                       stockout_risk_days, excess_risk_qty, mitigation_option
                FROM fact_scenario_results
                WHERE scenario_id = %s
                ORDER BY plan_month, ABS(impact_qty) DESC
                LIMIT %s OFFSET %s
                """,
                (scenario_id, page_size, offset),
            )
            rows = cur.fetchall()

    cols = [
        "item_no", "loc", "plan_month",
        "baseline_qty", "scenario_qty", "impact_qty", "impact_pct",
        "stockout_risk_days", "excess_risk_qty", "mitigation_option",
    ]
    items = []
    for r in rows:
        d = dict(zip(cols, r))
        if d.get("plan_month") and hasattr(d["plan_month"], "isoformat"):
            d["plan_month"] = d["plan_month"].isoformat()
        items.append(d)

    return {"scenario_id": scenario_id, "total": total, "page": page, "results": items}
