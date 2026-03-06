"""Inventory Planning — IPfeature10: Monte Carlo Simulation endpoints."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response as FastAPIResponse

from api.auth import require_api_key
from api.core import get_conn, set_cache

router = APIRouter()


def _f(v: Any) -> float | None:
    return float(v) if v is not None else None


@router.post("/inv-planning/simulation/run")
def run_simulation(
    item_no: str,
    loc: str,
    n_simulations: int = 10000,
    target_csl: Optional[float] = None,
    _: None = Depends(require_api_key),
) -> dict:
    """Trigger a Monte Carlo safety stock simulation for an item-location."""
    import uuid as _uuid
    sim_run_id = str(_uuid.uuid4())
    # Run synchronously for simplicity (large N is still fast enough ~30s for 10k)
    try:
        from scripts.run_ss_simulation import run as _sim_run
        result = _sim_run(
            item_no=item_no,
            loc=loc,
            n_simulations=n_simulations,
            target_csl=target_csl,
        )
        return {"sim_run_id": result["sim_run_id"], "status": "completed"}
    except Exception as exc:
        return {"sim_run_id": sim_run_id, "status": "failed", "error": str(exc)}


@router.get("/inv-planning/simulation/results")
def get_simulation_results(
    response: FastAPIResponse,
    item: str = Query(..., max_length=120),
    location: str = Query(..., max_length=120),
) -> dict:
    """Latest simulation results for an item-location."""
    set_cache(response, max_age=300)
    import json as _json

    sql = """
        SELECT sim_run_id, item_no, loc, simulation_date, n_simulations,
               demand_distribution, demand_mean, demand_std,
               lt_distribution, lt_mean_days, lt_std_days,
               results_by_ss_level,
               target_csl, recommended_ss, recommended_ss_days,
               analytical_ss, sim_vs_analytical_pct
        FROM fact_ss_simulation_results
        WHERE item_no = %s AND loc = %s
        ORDER BY simulation_date DESC, load_ts DESC
        LIMIT 1
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [item, location])
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="No simulation results found")

    curve = _json.loads(row[11]) if isinstance(row[11], str) else (row[11] or [])
    return {
        "sim_run_id":           row[0],
        "item_no":              row[1],
        "loc":                  row[2],
        "simulation_date":      str(row[3]),
        "n_simulations":        int(row[4]),
        "demand_distribution":  row[5],
        "demand_mean":          _f(row[6]),
        "demand_std":           _f(row[7]),
        "lt_distribution":      row[8],
        "lt_mean_days":         _f(row[9]),
        "lt_std_days":          _f(row[10]),
        "service_level_curve":  curve,
        "target_csl":           _f(row[12]),
        "recommended_ss":       _f(row[13]),
        "recommended_ss_days":  _f(row[14]),
        "analytical_ss":        _f(row[15]),
        "sim_vs_analytical_pct":_f(row[16]),
    }


@router.get("/inv-planning/simulation/compare")
def get_simulation_compare(
    response: FastAPIResponse,
    item: str = Query(..., max_length=120),
    location: str = Query(..., max_length=120),
) -> dict:
    """Compare simulated vs analytical SS for an item-location."""
    set_cache(response, max_age=300)
    import json as _json

    sql = """
        SELECT recommended_ss, analytical_ss, sim_vs_analytical_pct,
               results_by_ss_level, target_csl
        FROM fact_ss_simulation_results
        WHERE item_no = %s AND loc = %s
        ORDER BY simulation_date DESC
        LIMIT 1
    """
    eom_sql = """
        SELECT eom_qty_on_hand FROM agg_inventory_monthly
        WHERE item_no = %s AND loc = %s
        ORDER BY month_start DESC LIMIT 1
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [item, location])
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="No simulation results found")

            cur.execute(eom_sql, [item, location])
            eom_row = cur.fetchone()
            current_on_hand = _f(eom_row[0]) if eom_row else None

    curve = _json.loads(row[3]) if isinstance(row[3], str) else (row[3] or [])
    # Find current CSL from curve
    current_csl = None
    if current_on_hand is not None and curve:
        for pt in curve:
            if pt["ss_qty"] >= current_on_hand:
                current_csl = pt["csl"]
                break

    return {
        "item_no":              item,
        "loc":                  location,
        "analytical_ss":        _f(row[1]),
        "simulated_ss":         _f(row[0]),
        "difference_pct":       _f(row[2]),
        "service_level_curve":  curve,
        "current_on_hand":      current_on_hand,
        "current_csl":          current_csl,
    }


@router.get("/inv-planning/simulation/{sim_run_id}/status")
def get_simulation_status(
    sim_run_id: str,
) -> dict:
    """Get status of a simulation run."""
    sql = """
        SELECT item_no, loc, simulation_date, load_ts
        FROM fact_ss_simulation_results
        WHERE sim_run_id = %s
        LIMIT 1
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [sim_run_id])
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Simulation run not found")

    return {
        "sim_run_id":    sim_run_id,
        "status":        "completed",
        "progress_pct":  100,
        "item_no":       row[0],
        "loc":           row[1],
        "started_at":    None,
        "completed_at":  str(row[3]) if row[3] else None,
        "error":         None,
    }
