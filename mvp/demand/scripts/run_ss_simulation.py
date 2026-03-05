"""
IPfeature10 — Safety Stock Monte Carlo Simulation

Runs Monte Carlo simulation for a single item-location pair, computing
a service level curve and recommended safety stock.

Usage:
    uv run python scripts/run_ss_simulation.py --item ITEM_NO --loc LOC [--n-simulations N]
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg
from common.db import get_db_params  # noqa: E402

CONFIG_PATH = Path(__file__).parent.parent / "config" / "simulation_config.yaml"


def compute_service_level_curve(
    demand_obs: list[float],
    lt_obs: list[float],
    n_simulations: int,
    ss_levels: list[float],
    random_seed: int = 42,
) -> list[dict]:
    """Run Monte Carlo simulation to build service level curve."""
    rng = np.random.default_rng(random_seed)
    demand_pool = np.array(demand_obs, dtype=float)
    lt_pool = np.array(lt_obs, dtype=float)

    results = []
    for ss_qty in ss_levels:
        lt_sim = rng.choice(lt_pool, size=n_simulations, replace=True)
        stockouts = 0
        for lt in lt_sim:
            lt_int = max(1, int(round(lt)))
            demand_during_lt = float(np.sum(rng.choice(demand_pool, size=lt_int, replace=True)))
            if demand_during_lt > ss_qty:
                stockouts += 1
        csl = 1.0 - (stockouts / n_simulations)
        results.append({"ss_qty": float(ss_qty), "csl": round(csl, 4)})

    return results


def find_recommended_ss(curve: list[dict], target_csl: float) -> float | None:
    """Find minimum SS level achieving target CSL."""
    for point in curve:
        if point["csl"] >= target_csl:
            return point["ss_qty"]
    return None


def run(
    item_no: str,
    loc: str,
    n_simulations: int | None = None,
    target_csl: float | None = None,
) -> dict:
    config = yaml.safe_load(open(CONFIG_PATH))
    sim_cfg = config.get("simulation", {})
    n_simulations = n_simulations or sim_cfg.get("n_simulations", 10000)
    random_seed = sim_cfg.get("random_seed", 42)
    ss_levels_to_test = sim_cfg.get("ss_levels_to_test", 20)

    sim_run_id = str(uuid.uuid4())
    sim_date = date.today()
    t0 = time.time()

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            # Load demand history (monthly qty → daily rate)
            cur.execute("""
                SELECT qty_shipped FROM fact_sales_monthly
                WHERE dmdunit = %s AND loc = %s AND type = 1
                ORDER BY startdate DESC
                LIMIT 24
            """, [item_no, loc])
            demand_rows = cur.fetchall()
            if not demand_rows:
                raise ValueError(f"No demand history found for {item_no}/{loc}")
            daily_demand_obs = [float(r[0]) / 30.44 for r in demand_rows if r[0] and r[0] > 0]

            # Load lead time profile
            cur.execute("""
                SELECT lt_mean_days, lt_std_days
                FROM dim_item_lead_time_profile
                WHERE item_no = %s AND loc = %s
                ORDER BY computed_at DESC LIMIT 1
            """, [item_no, loc])
            lt_row = cur.fetchone()
            if lt_row:
                lt_mean, lt_std = float(lt_row[0]), float(lt_row[1]) if lt_row[1] else 0.0
                lt_obs = list(np.random.normal(lt_mean, lt_std, 100).clip(1))
                lt_distribution = "empirical"
            else:
                lt_mean, lt_std = 14.0, 0.0
                lt_obs = [14.0] * 100
                lt_distribution = "constant"

            # Load analytical SS
            cur.execute("""
                SELECT ss_combined FROM fact_safety_stock_targets
                WHERE item_no = %s AND loc = %s AND policy_version = 'v1'
                LIMIT 1
            """, [item_no, loc])
            ss_row = cur.fetchone()
            analytical_ss = float(ss_row[0]) if ss_row and ss_row[0] else 0.0

            # Target CSL
            if target_csl is None:
                target_csl = 0.95  # default

    if not daily_demand_obs:
        raise ValueError(f"No positive demand observations for {item_no}/{loc}")

    demand_mean = float(np.mean(daily_demand_obs))
    demand_std = float(np.std(daily_demand_obs))

    # Build SS levels: 0 to 2×analytical_ss
    max_ss = max(2 * analytical_ss, demand_mean * float(lt_mean) * 2, 100.0)
    ss_levels = list(np.linspace(0, max_ss, ss_levels_to_test))

    curve = compute_service_level_curve(daily_demand_obs, lt_obs, n_simulations, ss_levels, random_seed)
    recommended_ss = find_recommended_ss(curve, target_csl)

    avg_daily_demand = demand_mean if demand_mean > 0 else 1.0
    recommended_ss_days = (recommended_ss / avg_daily_demand) if recommended_ss is not None else None

    sim_vs_analytical_pct = None
    if recommended_ss is not None and analytical_ss and analytical_ss > 0:
        sim_vs_analytical_pct = (recommended_ss - analytical_ss) / analytical_ss * 100.0

    duration = time.time() - t0

    import json
    insert_sql = """
        INSERT INTO fact_ss_simulation_results (
            sim_run_id, item_no, loc, simulation_date, n_simulations,
            demand_distribution, demand_mean, demand_std,
            lt_distribution, lt_mean_days, lt_std_days,
            results_by_ss_level,
            target_csl, recommended_ss, recommended_ss_days,
            analytical_ss, sim_vs_analytical_pct, run_duration_secs
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (sim_run_id, item_no, loc) DO UPDATE SET
            results_by_ss_level = EXCLUDED.results_by_ss_level,
            recommended_ss = EXCLUDED.recommended_ss,
            run_duration_secs = EXCLUDED.run_duration_secs
    """

    with psycopg.connect(**get_db_params()) as conn:
        with conn.cursor() as cur:
            cur.execute(insert_sql, [
                sim_run_id, item_no, loc, sim_date, n_simulations,
                "empirical", demand_mean, demand_std,
                lt_distribution, lt_mean, lt_std,
                json.dumps(curve),
                target_csl, recommended_ss, recommended_ss_days,
                analytical_ss, sim_vs_analytical_pct, round(duration, 2),
            ])
        conn.commit()

    print(f"Simulation complete for {item_no}/{loc}: sim_run_id={sim_run_id}")
    print(f"  Recommended SS: {recommended_ss:.1f} units ({recommended_ss_days:.1f} days)")
    print(f"  Analytical SS:  {analytical_ss:.1f} units")
    if sim_vs_analytical_pct:
        print(f"  Sim vs analytical: {sim_vs_analytical_pct:+.1f}%")
    print(f"  Duration: {duration:.1f}s")

    return {
        "sim_run_id": sim_run_id,
        "recommended_ss": recommended_ss,
        "analytical_ss": analytical_ss,
        "sim_vs_analytical_pct": sim_vs_analytical_pct,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monte Carlo safety stock simulation")
    parser.add_argument("--item", required=True, help="Item number")
    parser.add_argument("--loc", required=True, help="Location code")
    parser.add_argument("--n-simulations", type=int, help="Number of simulations (default: config)")
    parser.add_argument("--target-csl", type=float, help="Target cycle service level (0-1)")
    args = parser.parse_args()
    run(
        item_no=args.item,
        loc=args.loc,
        n_simulations=args.n_simulations,
        target_csl=args.target_csl,
    )
