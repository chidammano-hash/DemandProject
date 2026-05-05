"""
compute_echelon_targets.py — F3.5 Network / Multi-Echelon Planning

Computes echelon-level safety stock and reorder points using risk pooling.
Reads network topology from dim_echelon_network, aggregates downstream
demand variance, and writes results to fact_echelon_ss_targets and
fact_echelon_reorder_points.

Usage:
    uv run python scripts/compute_echelon_targets.py
    uv run python scripts/compute_echelon_targets.py --item-no 100320
    uv run python scripts/compute_echelon_targets.py --dry-run

Config: config/inventory/echelon_config.yaml
"""

from __future__ import annotations

import argparse
import math
import yaml
import psycopg
from typing import Optional

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.core.db import get_db_params
from common.core.planning_date import get_planning_date
from common.services.perf_profiler import profiled_section
from common.core.utils import load_config as _load_config


def load_config() -> dict:
    return _load_config("echelon_config.yaml").get("echelon", {})


def compute_pooled_sigma(downstream_sigmas: list[float]) -> float:
    """
    Pool demand standard deviations from independent downstream nodes.

    Pooled σ = sqrt(σ₁² + σ₂² + ... + σₙ²)

    This is strictly less than the naive sum, capturing risk pooling benefits.

    Args:
        downstream_sigmas: List of daily demand σ values from downstream nodes

    Returns:
        Pooled daily demand σ for the echelon node

    Examples:
        [15, 12, 18] → sqrt(225 + 144 + 324) = sqrt(693) ≈ 26.3
    """
    if not downstream_sigmas:
        return 0.0
    return math.sqrt(sum(s ** 2 for s in downstream_sigmas))


def compute_echelon_ss(
    mean_demand_daily: float,
    sigma_demand_daily: float,
    mean_lt_days: float,
    sigma_lt_days: float,
    z_score: float,
) -> float:
    """
    Compute echelon safety stock using the combined formula.

    SS = Z × sqrt(mean_LT × σ_demand² + mean_demand² × σ_LT²)

    This accounts for both demand variability and lead-time variability.

    Args:
        mean_demand_daily: Average daily demand at this echelon
        sigma_demand_daily: Std dev of daily demand (pooled for DC nodes)
        mean_lt_days: Mean supplier lead time in days
        sigma_lt_days: Std dev of supplier lead time
        z_score: Service level Z-score (e.g., 1.645 for 95%)

    Returns:
        Safety stock in units (floored at 0)

    Examples:
        mean_demand=45, σ_demand=26.3, mean_LT=10, σ_LT=2, Z=1.645
        SS = 1.645 × sqrt(10×26.3² + 45²×4) = 1.645 × sqrt(6917+8100) = 1.645×122.6 ≈ 202
    """
    variance = mean_lt_days * (sigma_demand_daily ** 2) + (mean_demand_daily ** 2) * (sigma_lt_days ** 2)
    return max(0.0, z_score * math.sqrt(variance))


def compute_echelon_rop(
    mean_demand_daily: float,
    mean_lt_days: float,
    echelon_ss: float,
) -> float:
    """
    Compute echelon reorder point.

    ROP = mean_demand × mean_LT + SS

    Args:
        mean_demand_daily: Average daily demand
        mean_lt_days: Mean lead time in days
        echelon_ss: Computed echelon safety stock

    Returns:
        Reorder point in units
    """
    return mean_demand_daily * mean_lt_days + echelon_ss


def compute_downstream_coverage_days(
    dc_on_hand: float,
    total_downstream_daily_demand: float,
) -> float:
    """
    How many days of downstream demand the DC can cover from current stock.

    Args:
        dc_on_hand: Current on-hand at the DC
        total_downstream_daily_demand: Sum of daily demands of all downstream nodes

    Returns:
        Coverage in days (0 if no downstream demand)
    """
    if total_downstream_daily_demand <= 0:
        return 0.0
    return dc_on_hand / total_downstream_daily_demand


def compute_cascade_risk_score(
    downstream_node_count: int,
    dc_on_hand: float,
    echelon_rop: float,
) -> tuple[float, str]:
    """
    Estimate cascade risk when DC is below ROP.

    Risk score = (ROP - on_hand) / ROP × downstream_node_count × 10
    Capped at 100.

    Args:
        downstream_node_count: Number of stores served by this DC
        dc_on_hand: Current DC on-hand
        echelon_rop: Echelon reorder point

    Returns:
        (risk_score 0-100, severity label)
    """
    if dc_on_hand >= echelon_rop:
        return 0.0, "ok"

    shortfall_pct = (echelon_rop - dc_on_hand) / echelon_rop
    raw_score = shortfall_pct * downstream_node_count * 10.0
    score = min(100.0, raw_score)

    if score >= 70:
        severity = "critical"
    elif score >= 40:
        severity = "high"
    elif score >= 20:
        severity = "medium"
    else:
        severity = "low"

    return round(score, 1), severity


def fetch_network_topology(
    conn: psycopg.Connection,
    item_id: Optional[str] = None,
) -> list[dict]:
    """Fetch echelon network topology from dim_echelon_network."""
    conditions = ["is_active = TRUE"]
    params: list = []
    where = " AND ".join(conditions)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT parent_loc, child_loc, echelon_level,
                   replenishment_lead_time_days
            FROM dim_echelon_network
            WHERE {where}
            ORDER BY parent_loc, child_loc
            """,
            params,
        )
        rows = cur.fetchall()
    return [
        {
            "item_id": None,
            "dc_loc": r[0],
            "store_loc": r[1],
            "echelon_level": r[2],
            "lt_days": float(r[3]) if r[3] else 10.0,
            "lt_std_days": 2.0,  # default std dev
        }
        for r in rows
    ]


def _batch_load_demand_stats(conn: psycopg.Connection) -> dict[tuple[str, str], dict]:
    """
    Batch-load all demand stats from dim_sku in a single query.

    Returns dict keyed by (item_id, loc) with mean_demand and sigma_demand.
    This replaces per-DC fetch_downstream_stats calls.
    """
    demand_stats: dict[tuple[str, str], dict] = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT item_id, loc,
                   COALESCE(demand_mean, 0) AS mean_demand,
                   COALESCE(demand_std, 0)  AS sigma_demand
            FROM dim_sku
        """)
        for row in cur.fetchall():
            demand_stats[(row[0], row[1])] = {
                "loc": row[1],
                "mean_demand": float(row[2]),
                "sigma_demand": float(row[3]),
            }
    return demand_stats


def _batch_load_on_hand(conn: psycopg.Connection) -> dict[tuple[str, str], float]:
    """
    Batch-load latest on-hand inventory for all (item_id, loc) in a single query.

    Returns dict keyed by (item_id, loc) with qty_on_hand float.
    This replaces per-DC fetch_dc_on_hand calls.
    """
    onhand_map: dict[tuple[str, str], float] = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (item_id, loc)
                   item_id, loc, qty_on_hand
            FROM fact_inventory_snapshot
            ORDER BY item_id, loc, snapshot_date DESC
        """)
        for row in cur.fetchall():
            onhand_map[(row[0], row[1])] = float(row[2]) if row[2] else 0.0
    return onhand_map


def fetch_downstream_stats(
    conn: psycopg.Connection,
    item_id: str,
    store_locs: list[str],
    *,
    batch_demand_stats: dict[tuple[str, str], dict] | None = None,
) -> list[dict]:
    """Fetch demand stats for downstream store locations.

    When batch_demand_stats is provided, uses dict lookups instead of SQL.
    """
    if not store_locs:
        return []

    if batch_demand_stats is not None:
        return [
            batch_demand_stats[(item_id, loc)]
            for loc in store_locs
            if (item_id, loc) in batch_demand_stats
        ]

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.loc,
                   COALESCE(d.avg_daily_demand, 0) AS mean_demand,
                   COALESCE(d.demand_std_dev, 0)   AS sigma_demand
            FROM dim_sku d
            WHERE d.item_id = %s AND d.loc = ANY(%s)
            """,
            (item_id, store_locs),
        )
        rows = cur.fetchall()
    return [
        {
            "loc": r[0],
            "mean_demand": float(r[1]),
            "sigma_demand": float(r[2]),
        }
        for r in rows
    ]


def fetch_dc_on_hand(
    conn: psycopg.Connection,
    item_id: str,
    dc_loc: str,
    *,
    batch_onhand: dict[tuple[str, str], float] | None = None,
) -> float:
    """Fetch current DC on-hand from latest inventory snapshot.

    When batch_onhand is provided, uses dict lookup instead of SQL.
    """
    if batch_onhand is not None:
        return batch_onhand.get((item_id, dc_loc), 0.0)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT qty_on_hand
            FROM fact_inventory_snapshot
            WHERE item_id = %s AND loc = %s
            ORDER BY snapshot_date DESC
            LIMIT 1
            """,
            (item_id, dc_loc),
        )
        row = cur.fetchone()
    return float(row[0]) if row and row[0] else 0.0


def upsert_echelon_ss_targets(conn: psycopg.Connection, rows: list[dict]) -> int:
    sql = """
        INSERT INTO fact_echelon_ss_targets (
            item_id, dc_loc, plan_date, downstream_node_count,
            total_downstream_daily_demand, pooled_sigma_demand,
            echelon_lt_days, echelon_lt_std_days, z_score,
            echelon_ss_units, echelon_rop_units, current_dc_on_hand,
            coverage_days, cascade_risk_score, cascade_risk_severity
        ) VALUES (
            %(item_id)s, %(dc_loc)s, %(plan_date)s, %(downstream_node_count)s,
            %(total_downstream_daily_demand)s, %(pooled_sigma_demand)s,
            %(echelon_lt_days)s, %(echelon_lt_std_days)s, %(z_score)s,
            %(echelon_ss_units)s, %(echelon_rop_units)s, %(current_dc_on_hand)s,
            %(coverage_days)s, %(cascade_risk_score)s, %(cascade_risk_severity)s
        )
        ON CONFLICT (item_id, dc_loc, plan_date)
        DO UPDATE SET
            downstream_node_count           = EXCLUDED.downstream_node_count,
            total_downstream_daily_demand   = EXCLUDED.total_downstream_daily_demand,
            pooled_sigma_demand             = EXCLUDED.pooled_sigma_demand,
            echelon_ss_units                = EXCLUDED.echelon_ss_units,
            echelon_rop_units               = EXCLUDED.echelon_rop_units,
            current_dc_on_hand              = EXCLUDED.current_dc_on_hand,
            coverage_days                   = EXCLUDED.coverage_days,
            cascade_risk_score              = EXCLUDED.cascade_risk_score,
            cascade_risk_severity           = EXCLUDED.cascade_risk_severity,
            computed_at                     = NOW()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def run(
    item_id: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Main entry point: compute and write echelon targets."""
    with profiled_section("load_config"):
        cfg = load_config()
        z_score = cfg.get("z_score_default", 1.645)
        today = get_planning_date()

    rows_to_write: list[dict] = []
    below_rop_count = 0

    with psycopg.connect(**get_db_params()) as conn:
        with profiled_section("load_data"):
            topology = fetch_network_topology(conn, item_id)

            # Group by (item_id, dc_loc)
            from collections import defaultdict
            dc_groups: dict = defaultdict(list)
            dc_lt: dict = {}
            for node in topology:
                key = (node["item_id"], node["dc_loc"])
                dc_groups[key].append(node)
                if key not in dc_lt:
                    dc_lt[key] = {"lt_days": node["lt_days"], "lt_std_days": node["lt_std_days"]}

            # Batch-load all demand stats and on-hand data upfront (2 queries instead of 2*N)
            batch_demand_stats = _batch_load_demand_stats(conn)
            batch_onhand = _batch_load_on_hand(conn)

        with profiled_section("compute_echelon_targets"):
            for (item, dc), nodes in dc_groups.items():
                store_locs = [n["store_loc"] for n in nodes if n["store_loc"]]
                downstream = fetch_downstream_stats(
                    conn, item, store_locs, batch_demand_stats=batch_demand_stats
                )

                if not downstream:
                    continue

                sigmas = [d["sigma_demand"] for d in downstream]
                means = [d["mean_demand"] for d in downstream]

                pooled_sigma = compute_pooled_sigma(sigmas)
                total_mean = sum(means)
                lt_days = dc_lt[(item, dc)]["lt_days"]
                lt_std = dc_lt[(item, dc)]["lt_std_days"]

                echelon_ss = compute_echelon_ss(total_mean, pooled_sigma, lt_days, lt_std, z_score)
                echelon_rop = compute_echelon_rop(total_mean, lt_days, echelon_ss)
                dc_on_hand = fetch_dc_on_hand(
                    conn, item, dc, batch_onhand=batch_onhand
                )
                coverage = compute_downstream_coverage_days(dc_on_hand, total_mean)
                risk_score, risk_severity = compute_cascade_risk_score(
                    len(downstream), dc_on_hand, echelon_rop
                )

                if dc_on_hand < echelon_rop:
                    below_rop_count += 1

                rows_to_write.append({
                    "item_id": item,
                    "dc_loc": dc,
                    "plan_date": today,
                    "downstream_node_count": len(downstream),
                    "total_downstream_daily_demand": round(total_mean, 2),
                    "pooled_sigma_demand": round(pooled_sigma, 2),
                    "echelon_lt_days": lt_days,
                    "echelon_lt_std_days": lt_std,
                    "z_score": z_score,
                    "echelon_ss_units": round(echelon_ss, 2),
                    "echelon_rop_units": round(echelon_rop, 2),
                    "current_dc_on_hand": round(dc_on_hand, 2),
                    "coverage_days": round(coverage, 1),
                    "cascade_risk_score": risk_score,
                    "cascade_risk_severity": risk_severity,
                })

        with profiled_section("write_results"):
            rows_written = 0
            if not dry_run and rows_to_write:
                rows_written = upsert_echelon_ss_targets(conn, rows_to_write)
                conn.commit()

    return {
        "dc_nodes_processed": len(rows_to_write),
        "rows_written": rows_written,
        "below_rop_count": below_rop_count,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute multi-echelon safety stock targets")
    parser.add_argument("--item-no", help="Process single item only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(args.item_no, args.dry_run)
    print(result)
