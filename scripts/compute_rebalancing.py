"""
Inventory Rebalancing — Cross-Location Transfer Optimization

Detects spatial imbalances (excess at location A, shortage at location B)
and computes cost-optimal transfer recommendations using a greedy solver
or optional LP solver (scipy).

Usage:
    uv run python scripts/compute_rebalancing.py [--solver greedy|lp] [--horizon 4] [--dry-run] [--item ITEM]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import uuid
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg

from common.db import get_db_params
from common.planning_date import get_planning_date
from common.scripts_base import add_common_args, setup_logging
from common.services.perf_profiler import profiled_section
from common.utils import load_config as _load_config


def load_config() -> dict:
    return _load_config("rebalancing_config.yaml")


def load_network(conn) -> list[dict]:
    """Load active transfer lanes from dim_transfer_lane."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT lane_id, source_loc, dest_loc, transfer_mode,
                   cost_per_unit, handling_cost, freight_cost, receiving_cost,
                   fixed_cost_per_shipment, transfer_lt_days,
                   min_transfer_qty, max_transfer_qty, batch_size,
                   max_shipments_per_week, max_receiving_units_per_period
            FROM dim_transfer_lane
            WHERE is_active = TRUE
              AND (effective_to IS NULL OR effective_to >= %s)
        """, [get_planning_date()])
        rows = cur.fetchall()

    return [
        {
            "lane_id": r[0], "source_loc": r[1], "dest_loc": r[2],
            "transfer_mode": r[3], "cost_per_unit": float(r[4]),
            "handling_cost": float(r[5] or 0), "freight_cost": float(r[6] or 0),
            "receiving_cost": float(r[7] or 0), "fixed_cost_per_shipment": float(r[8] or 0),
            "transfer_lt_days": int(r[9]), "min_transfer_qty": int(r[10] or 1),
            "max_transfer_qty": int(r[11]) if r[11] else None,
            "batch_size": int(r[12] or 1),
            "max_shipments_per_week": int(r[13] or 5),
            "max_receiving_units_per_period": int(r[14]) if r[14] else None,
        }
        for r in rows
    ]


def load_inventory_state(
    conn,
    item_filter: str | None = None,
    available_supply_cfg: dict | None = None,
) -> dict[tuple[str, str], dict]:
    """Load latest month inventory + safety stock targets per item-loc.

    Gen-4 SC-4: ``available_supply`` = on_hand + in_transit + open_pos. Each
    source can be toggled via rebalancing_config.yaml. When a dependent table
    is missing, that source contributes 0 (degrades gracefully).
    """
    cfg = available_supply_cfg or {}
    include_in_transit = bool(cfg.get("include_in_transit", True))
    include_open_pos = bool(cfg.get("include_open_pos", True))

    wheres = ["a.month_start = (SELECT MAX(month_start) FROM agg_inventory_monthly)"]
    params: list = []
    if item_filter:
        wheres.append("a.item_id = %s")
        params.append(item_filter)

    where_clause = " AND ".join(wheres)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT a.item_id, a.loc, a.eom_qty_on_hand, a.avg_daily_sls,
                   s.ss_combined, d.abc_vol,
                   COALESCE(s.reorder_point, 0) AS reorder_point
            FROM agg_inventory_monthly a
            LEFT JOIN fact_safety_stock_targets s
                ON a.item_id = s.item_id AND a.loc = s.loc
            LEFT JOIN dim_sku d
                ON a.item_id = d.item_id AND a.loc = d.loc
            WHERE {where_clause}
        """, params)
        rows = cur.fetchall()

    state = {}
    for r in rows:
        on_hand = float(r[2] or 0)
        daily_sales = float(r[3] or 0)
        ss = float(r[4]) if r[4] is not None else None
        dos = on_hand / daily_sales if daily_sales > 0 else None
        state[(r[0], r[1])] = {
            "item_id": r[0], "loc": r[1],
            "on_hand": on_hand, "daily_sales": daily_sales,
            "in_transit": 0.0, "open_po_qty": 0.0,
            "ss_target": ss, "dos": dos,
            "abc_vol": r[5], "reorder_point": float(r[6]),
        }

    # Gen-4 SC-4: augment with in-transit qty (best-effort)
    if include_in_transit:
        with conn.cursor() as cur:
            try:
                cur.execute("SAVEPOINT load_in_transit")
                cur.execute("""
                    SELECT item_id, dest_loc, COALESCE(SUM(qty_in_transit), 0)
                    FROM fact_inventory_in_transit
                    WHERE expected_arrival_date >= CURRENT_DATE
                    GROUP BY item_id, dest_loc
                """)
                for item_id, loc, qty in cur.fetchall():
                    key = (item_id, loc)
                    if key in state:
                        state[key]["in_transit"] = float(qty or 0)
            except Exception:
                cur.execute("ROLLBACK TO SAVEPOINT load_in_transit")

    # Gen-4 SC-4: augment with open PO qty (best-effort)
    if include_open_pos:
        with conn.cursor() as cur:
            try:
                cur.execute("SAVEPOINT load_open_pos")
                cur.execute("""
                    SELECT item_id, loc, COALESCE(SUM(order_qty - COALESCE(received_qty, 0)), 0)
                    FROM fact_purchase_orders
                    WHERE is_closed = FALSE
                    GROUP BY item_id, loc
                """)
                for item_id, loc, qty in cur.fetchall():
                    key = (item_id, loc)
                    if key in state:
                        state[key]["open_po_qty"] = float(qty or 0)
            except Exception:
                cur.execute("ROLLBACK TO SAVEPOINT load_open_pos")

    # Compute available_supply and projected_dos for each SKU
    for info in state.values():
        avail = info["on_hand"]
        if include_in_transit:
            avail += info["in_transit"]
        if include_open_pos:
            avail += info["open_po_qty"]
        info["available_supply"] = avail
        info["projected_dos"] = (
            avail / info["daily_sales"] if info["daily_sales"] > 0 else None
        )
    return state


def detect_imbalances(
    state: dict[tuple[str, str], dict],
    excess_pct: float,
    shortage_pct: float,
) -> dict[str, dict]:
    """Classify each item-loc as excess, shortage, or balanced.

    Returns dict keyed by item_id with lists of excess and shortage locations.
    Only items with BOTH excess AND shortage are included.
    """
    by_item: dict[str, dict] = {}
    for (item_id, loc), info in state.items():
        if info["ss_target"] is None or info["ss_target"] <= 0:
            continue
        ratio = info["on_hand"] / info["ss_target"]
        entry = by_item.setdefault(item_id, {"excess": [], "shortage": []})
        if ratio > excess_pct:
            entry["excess"].append(info)
        elif ratio < shortage_pct:
            entry["shortage"].append(info)

    # Keep only items with both excess and shortage
    return {k: v for k, v in by_item.items() if v["excess"] and v["shortage"]}


def build_transfer_candidates(
    imbalances: dict[str, dict],
    lanes: list[dict],
    config: dict,
) -> list[dict]:
    """Build all feasible (excess_loc → shortage_loc) transfer candidates."""
    lane_map: dict[tuple[str, str], dict] = {}
    for lane in lanes:
        lane_map[(lane["source_loc"], lane["dest_loc"])] = lane

    defaults = config.get("network", {})
    constraints = config.get("constraints", {})
    max_drawdown = constraints.get("max_source_drawdown_pct", 0.30)

    candidates = []
    for item_id, groups in imbalances.items():
        for src in groups["excess"]:
            for dst in groups["shortage"]:
                lane = lane_map.get((src["loc"], dst["loc"]))

                # Use lane-specific or default costs
                if lane:
                    cost = lane["cost_per_unit"]
                    lt_days = lane["transfer_lt_days"]
                    min_qty = lane["min_transfer_qty"]
                    max_qty = lane["max_transfer_qty"]
                    batch = lane["batch_size"]
                    fixed_cost = lane["fixed_cost_per_shipment"]
                    lane_id = lane["lane_id"]
                else:
                    cost = defaults.get("default_cost_per_unit", 0.50)
                    lt_days = defaults.get("default_transfer_lt_days", 3)
                    min_qty = defaults.get("default_min_transfer_qty", 10)
                    max_qty = None
                    batch = defaults.get("default_batch_size", 1)
                    fixed_cost = 0
                    lane_id = None

                excess_qty = src["on_hand"] - src["ss_target"]
                shortage_qty = dst["ss_target"] - dst["on_hand"]
                drawdown_limit = src["on_hand"] * max_drawdown

                raw_qty = min(excess_qty, shortage_qty, drawdown_limit)
                if max_qty is not None:
                    raw_qty = min(raw_qty, max_qty)

                # Round to batch size
                if batch > 1:
                    raw_qty = math.floor(raw_qty / batch) * batch

                if raw_qty < min_qty:
                    continue

                candidates.append({
                    "item_id": item_id,
                    "source_loc": src["loc"],
                    "dest_loc": dst["loc"],
                    "lane_id": lane_id,
                    "recommended_qty": raw_qty,
                    "source_on_hand": src["on_hand"],
                    "source_dos": src["dos"],
                    "source_ss_target": src["ss_target"],
                    "source_excess_qty": excess_qty,
                    "dest_on_hand": dst["on_hand"],
                    "dest_dos": dst["dos"],
                    "dest_ss_target": dst["ss_target"],
                    "dest_shortage_qty": shortage_qty,
                    "abc_class": src.get("abc_vol"),
                    "cost_per_unit": cost,
                    "fixed_cost": fixed_cost,
                    "transfer_lt_days": lt_days,
                    "dest_reorder_point": dst.get("reorder_point", 0),
                })

    return candidates


def compute_financials(
    candidates: list[dict],
    config: dict,
) -> list[dict]:
    """Compute transfer cost, carrying savings, stockout avoidance, ROI."""
    costs = config.get("costs", {})
    stockout_mult = costs.get("stockout_cost_multiplier", 5.0)
    carrying_pct = costs.get("carrying_cost_annual_pct", 0.25)
    horizon_days = config.get("optimization", {}).get("horizon_weeks", 4) * 7

    for c in candidates:
        qty = c["recommended_qty"]
        c["transfer_cost"] = qty * c["cost_per_unit"] + c["fixed_cost"]
        # Approximate unit cost = 1.0 (no unit_cost in current schema)
        unit_cost = 1.0
        c["carrying_cost_saved"] = (
            c["source_excess_qty"] * unit_cost * carrying_pct * (horizon_days / 365)
        )
        # Stockout probability proxy: how far below SS the dest is
        shortage_severity = min(1.0, c["dest_shortage_qty"] / max(c["dest_ss_target"], 1))
        c["stockout_cost_avoided"] = (
            min(qty, c["dest_shortage_qty"]) * unit_cost * stockout_mult * shortage_severity
        )
        c["net_benefit"] = (
            c["stockout_cost_avoided"] + c["carrying_cost_saved"] - c["transfer_cost"]
        )
        c["roi"] = c["net_benefit"] / max(c["transfer_cost"], 0.01)

    return candidates


def assign_urgency(candidates: list[dict]) -> list[dict]:
    """Assign urgency level based on destination DOS and ABC class."""
    for c in candidates:
        dos = c.get("dest_dos")
        abc = c.get("abc_class", "")
        if dos is not None and dos < 3 and abc == "A":
            c["urgency"] = "critical"
        elif dos is not None and dos < 7:
            c["urgency"] = "high"
        elif dos is not None and dos < 14:
            c["urgency"] = "medium"
        else:
            c["urgency"] = "low"
    return candidates


def greedy_solver(
    candidates: list[dict],
    config: dict,
) -> list[dict]:
    """Greedy solver: rank by priority, assign transfers greedily."""
    min_benefit = config.get("triggers", {}).get("min_benefit_per_transfer", 5.0)

    # Filter out unprofitable transfers
    viable = [c for c in candidates if c["net_benefit"] >= min_benefit]

    # Score: urgency weight + ROI
    urgency_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    for c in viable:
        shortage_severity = c["dest_shortage_qty"] / max(c["dest_ss_target"], 1)
        c["priority_score"] = (
            urgency_weights.get(c["urgency"], 1) * shortage_severity
            + max(c["roi"], 0)
        )

    viable.sort(key=lambda x: -x["priority_score"])

    # Track remaining excess/shortage per item-loc
    remaining_excess: dict[tuple[str, str], float] = {}
    remaining_shortage: dict[tuple[str, str], float] = {}
    for c in viable:
        key_src = (c["item_id"], c["source_loc"])
        key_dst = (c["item_id"], c["dest_loc"])
        remaining_excess.setdefault(key_src, c["source_excess_qty"])
        remaining_shortage.setdefault(key_dst, c["dest_shortage_qty"])

    selected = []
    for c in viable:
        key_src = (c["item_id"], c["source_loc"])
        key_dst = (c["item_id"], c["dest_loc"])

        avail_excess = remaining_excess.get(key_src, 0)
        avail_shortage = remaining_shortage.get(key_dst, 0)
        qty = min(c["recommended_qty"], avail_excess, avail_shortage)

        if qty <= 0:
            continue

        c["recommended_qty"] = qty
        # Recompute cost/benefit for adjusted qty
        c["transfer_cost"] = qty * c["cost_per_unit"] + c["fixed_cost"]
        remaining_excess[key_src] = avail_excess - qty
        remaining_shortage[key_dst] = avail_shortage - qty
        selected.append(c)

    return selected


def lp_solver(
    candidates: list[dict],
    config: dict,
) -> list[dict]:
    """LP solver using scipy.optimize.linprog for cost minimisation."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        print("scipy not available, falling back to greedy solver")
        return greedy_solver(candidates, config)

    min_benefit = config.get("triggers", {}).get("min_benefit_per_transfer", 5.0)
    viable = [c for c in candidates if c["net_benefit"] >= min_benefit]
    if not viable:
        return []

    n = len(viable)
    # Objective: minimize transfer cost (negative benefit = maximize benefit)
    # We minimize -net_benefit to maximize benefit
    c_obj = [-c["net_benefit"] for c in viable]

    # Upper bounds: each transfer <= recommended_qty
    bounds = [(0, c["recommended_qty"]) for c in viable]

    # Source constraints: sum of transfers from same source <= excess
    source_keys = list({(c["item_id"], c["source_loc"]) for c in viable})
    A_ub = []
    b_ub = []
    for sk in source_keys:
        row = [0.0] * n
        for i, c in enumerate(viable):
            if (c["item_id"], c["source_loc"]) == sk:
                row[i] = 1.0
        A_ub.append(row)
        b_ub.append(viable[[i for i, c in enumerate(viable) if (c["item_id"], c["source_loc"]) == sk][0]]["source_excess_qty"])

    # Dest constraints: sum of transfers to same dest <= shortage
    dest_keys = list({(c["item_id"], c["dest_loc"]) for c in viable})
    for dk in dest_keys:
        row = [0.0] * n
        for i, c in enumerate(viable):
            if (c["item_id"], c["dest_loc"]) == dk:
                row[i] = 1.0
        A_ub.append(row)
        b_ub.append(viable[[i for i, c in enumerate(viable) if (c["item_id"], c["dest_loc"]) == dk][0]]["dest_shortage_qty"])

    time_limit = config.get("optimization", {}).get("time_limit_seconds", 60)
    result = linprog(
        c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
        method="highs",
        options={"time_limit": time_limit},
    )

    if not result.success:
        print(f"LP solver failed: {result.message}, falling back to greedy")
        return greedy_solver(candidates, config)

    selected = []
    for i, c in enumerate(viable):
        qty = result.x[i]
        if qty > 0.5:  # threshold for rounding
            c["recommended_qty"] = round(qty)
            c["transfer_cost"] = c["recommended_qty"] * c["cost_per_unit"] + c["fixed_cost"]
            # Assign priority score for ordering
            urgency_weights = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            shortage_severity = c["dest_shortage_qty"] / max(c["dest_ss_target"], 1)
            c["priority_score"] = urgency_weights.get(c["urgency"], 1) * shortage_severity + max(c["roi"], 0)
            selected.append(c)

    return selected


def max_service_solver(
    candidates: list[dict],
    config: dict,
) -> list[dict]:
    """Gen-4 SC-4: max_service objective — fill largest shortage first.

    Ignores net_benefit threshold (pure service-level optimization). Sorts
    candidates by shortage severity * abc-priority and greedily assigns
    transfers until excess/shortage pools drain.
    """
    abc_weight = {"A": 3, "B": 2, "C": 1}

    for c in candidates:
        dst_ss = max(c.get("dest_ss_target", 0), 1)
        shortage_weight = c.get("dest_shortage_qty", 0) / dst_ss
        c["service_score"] = (
            shortage_weight
            * abc_weight.get(c.get("abc_class") or "C", 1)
        )

    candidates.sort(key=lambda x: -x["service_score"])

    remaining_excess: dict[tuple[str, str], float] = {}
    remaining_shortage: dict[tuple[str, str], float] = {}
    for c in candidates:
        key_src = (c["item_id"], c["source_loc"])
        key_dst = (c["item_id"], c["dest_loc"])
        remaining_excess.setdefault(key_src, c["source_excess_qty"])
        remaining_shortage.setdefault(key_dst, c["dest_shortage_qty"])

    selected = []
    for c in candidates:
        key_src = (c["item_id"], c["source_loc"])
        key_dst = (c["item_id"], c["dest_loc"])
        avail_excess = remaining_excess.get(key_src, 0)
        avail_shortage = remaining_shortage.get(key_dst, 0)
        qty = min(c["recommended_qty"], avail_excess, avail_shortage)
        if qty <= 0:
            continue
        c["recommended_qty"] = qty
        c["transfer_cost"] = qty * c["cost_per_unit"] + c["fixed_cost"]
        remaining_excess[key_src] = avail_excess - qty
        remaining_shortage[key_dst] = avail_shortage - qty
        selected.append(c)
    return selected


def equalize_dos_solver(
    candidates: list[dict],
    config: dict,
    state: dict[tuple[str, str], dict],
) -> list[dict]:
    """Gen-4 SC-4: equalize_dos objective — level DOS across network.

    For each item, compute network-mean DOS and flag locations >10% above
    (surplus) or <10% below (deficit). Only keep candidate transfers that
    move from surplus to deficit (by DOS).
    """
    obj_cfg = config.get("objectives", {}).get("equalize_dos", {})
    tol = float(obj_cfg.get("target_dos_tolerance_pct", 0.10))
    max_move = float(obj_cfg.get("max_move_per_item", 1000))

    # Per-item network mean
    from collections import defaultdict
    dos_by_item: dict[str, list[float]] = defaultdict(list)
    for (item_id, _), info in state.items():
        if info.get("dos") is not None:
            dos_by_item[item_id].append(float(info["dos"]))

    mean_dos: dict[str, float] = {
        item_id: sum(v) / len(v) for item_id, v in dos_by_item.items() if v
    }

    # Track qty moved per item to enforce max_move_per_item safety cap
    moved_per_item: dict[str, float] = defaultdict(float)

    selected = []
    for c in candidates:
        m = mean_dos.get(c["item_id"])
        if m is None or m <= 0:
            continue
        src_dos = c.get("source_dos") or 0
        dst_dos = c.get("dest_dos") or 0
        src_surplus = src_dos > m * (1 + tol)
        dst_deficit = dst_dos < m * (1 - tol)
        if not (src_surplus and dst_deficit):
            continue
        if moved_per_item[c["item_id"]] + c["recommended_qty"] > max_move:
            continue
        moved_per_item[c["item_id"]] += c["recommended_qty"]
        c["transfer_cost"] = c["recommended_qty"] * c["cost_per_unit"] + c["fixed_cost"]
        selected.append(c)
    return selected


def compute_network_balance(state: dict[tuple[str, str], dict]) -> float:
    """Compute network DOS coefficient of variation across all items."""
    from collections import defaultdict
    item_dos: dict[str, list[float]] = defaultdict(list)
    for (item_id, _), info in state.items():
        if info["dos"] is not None:
            item_dos[item_id].append(info["dos"])

    cvs = []
    for dos_vals in item_dos.values():
        if len(dos_vals) >= 2:
            mean = sum(dos_vals) / len(dos_vals)
            if mean > 0:
                variance = sum((d - mean) ** 2 for d in dos_vals) / len(dos_vals)
                cv = (variance ** 0.5) / mean
                cvs.append(cv)

    return sum(cvs) / len(cvs) if cvs else 0.0


def write_plan(
    conn, transfers: list[dict], config: dict,
    solver_method: str, runtime_ms: int,
    state: dict[tuple[str, str], dict],
) -> str:
    """Write plan header + transfer rows to DB. Returns plan_id."""
    plan_id = str(uuid.uuid4())
    computation_date = get_planning_date()
    horizon_weeks = config.get("optimization", {}).get("horizon_weeks", 4)
    objective = config.get("optimization", {}).get("objective", "min_cost")

    total_qty = sum(t["recommended_qty"] for t in transfers)
    total_cost = sum(t["transfer_cost"] for t in transfers)
    total_avoided = sum(t.get("stockout_cost_avoided", 0) for t in transfers)
    net_roi = (total_avoided - total_cost) / max(total_cost, 0.01)
    balance_before = compute_network_balance(state)
    items_rebalanced = len({t["item_id"] for t in transfers})
    lanes_used = len({(t["source_loc"], t["dest_loc"]) for t in transfers})

    plan_sql = """
        INSERT INTO fact_rebalancing_plan (
            plan_id, computation_date, horizon_weeks, solver_method, objective,
            total_transfer_qty, total_transfer_cost, total_avoided_stockout_value,
            net_roi, network_balance_before, items_rebalanced, lanes_used,
            status, solver_runtime_ms
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'draft',%s)
    """

    transfer_sql = """
        INSERT INTO fact_rebalancing_transfer (
            transfer_id, plan_id, item_id, source_loc, dest_loc, lane_id, transfer_mode,
            recommended_qty, source_on_hand, source_dos, source_ss_target, source_excess_qty,
            dest_on_hand, dest_dos, dest_ss_target, dest_shortage_qty,
            transfer_cost, carrying_cost_saved, stockout_cost_avoided, net_benefit, roi,
            planned_ship_date, expected_arrival_date, transfer_lt_days,
            priority_score, abc_class, urgency, status
        ) VALUES (%s,%s,%s,%s,%s,%s,'truck',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'recommended')
    """

    with conn.cursor() as cur:
        cur.execute(plan_sql, [
            plan_id, computation_date, horizon_weeks, solver_method, objective,
            total_qty, total_cost, total_avoided, net_roi, balance_before,
            items_rebalanced, lanes_used, runtime_ms,
        ])
        for t in transfers:
            ship_date = computation_date + timedelta(days=config.get("constraints", {}).get("frozen_period_days", 7))
            arrival_date = ship_date + timedelta(days=t.get("transfer_lt_days", 3))
            cur.execute(transfer_sql, [
                str(uuid.uuid4()), plan_id, t["item_id"],
                t["source_loc"], t["dest_loc"], t.get("lane_id"), t["recommended_qty"],
                t["source_on_hand"], t["source_dos"], t["source_ss_target"], t["source_excess_qty"],
                t["dest_on_hand"], t["dest_dos"], t["dest_ss_target"], t["dest_shortage_qty"],
                t["transfer_cost"], t.get("carrying_cost_saved", 0),
                t.get("stockout_cost_avoided", 0), t.get("net_benefit", 0), t.get("roi", 0),
                ship_date, arrival_date, t.get("transfer_lt_days", 3),
                t.get("priority_score", 0), t.get("abc_class"), t.get("urgency"),
            ])
    conn.commit()
    return plan_id


def run(
    solver: str = "greedy",
    horizon: int | None = None,
    dry_run: bool = False,
    item_filter: str | None = None,
    budget_cap: float | None = None,
) -> dict:
    config = load_config()
    if horizon is not None:
        config.setdefault("optimization", {})["horizon_weeks"] = horizon
    solver_method = solver or config.get("optimization", {}).get("solver", "greedy")

    start = time.time()

    with profiled_section("load_network_and_inventory"):
        with psycopg.connect(**get_db_params()) as conn:
            lanes = load_network(conn)
            # Gen-4 SC-4: pass available_supply config so loader folds in
            # in-transit + open POs alongside on-hand.
            state = load_inventory_state(
                conn, item_filter,
                available_supply_cfg=config.get("available_supply"),
            )

    if not state:
        print("No inventory data found.")
        return {"plan_id": None, "transfers": 0, "total_cost": 0}

    excess_pct = config.get("triggers", {}).get("excess_threshold_pct", 1.50)
    shortage_pct = config.get("triggers", {}).get("shortage_threshold_pct", 0.80)

    with profiled_section("detect_imbalances_and_solve"):
        imbalances = detect_imbalances(state, excess_pct, shortage_pct)
        print(f"Detected {len(imbalances)} items with spatial imbalances")

        if not imbalances:
            print("No imbalances detected — network is balanced.")
            return {"plan_id": None, "transfers": 0, "total_cost": 0}

        candidates = build_transfer_candidates(imbalances, lanes, config)
        print(f"Built {len(candidates)} transfer candidates")

        candidates = compute_financials(candidates, config)
        candidates = assign_urgency(candidates)

        # Gen-4 SC-4: honor `objective` knob
        objective = config.get("optimization", {}).get("objective", "min_cost")
        if objective == "max_service":
            selected = max_service_solver(candidates, config)
        elif objective == "equalize_dos":
            selected = equalize_dos_solver(candidates, config, state)
        elif solver_method == "lp":
            selected = lp_solver(candidates, config)
        else:
            selected = greedy_solver(candidates, config)

        if budget_cap is not None:
            # Sort by ROI desc, keep under budget
            selected.sort(key=lambda x: -x.get("roi", 0))
            capped = []
            cumulative = 0.0
            for t in selected:
                if cumulative + t["transfer_cost"] > budget_cap:
                    break
                capped.append(t)
                cumulative += t["transfer_cost"]
            selected = capped

    runtime_ms = int((time.time() - start) * 1000)
    total_cost = sum(t["transfer_cost"] for t in selected)
    total_qty = sum(t["recommended_qty"] for t in selected)

    print(f"Selected {len(selected)} transfers, total qty={total_qty:,.0f}, cost=${total_cost:,.2f}")

    if dry_run:
        print("[DRY RUN] No data written.")
        for t in selected[:10]:
            print(f"  {t['item_id']} {t['source_loc']}→{t['dest_loc']} qty={t['recommended_qty']:.0f} "
                  f"cost=${t['transfer_cost']:.2f} benefit=${t['net_benefit']:.2f} urgency={t['urgency']}")
        return {"plan_id": None, "transfers": len(selected), "total_cost": total_cost}

    with profiled_section("write_rebalancing_plan"):
        with psycopg.connect(**get_db_params()) as conn:
            plan_id = write_plan(conn, selected, config, solver_method, runtime_ms, state)

    print(f"Plan {plan_id} written with {len(selected)} transfers.")
    return {"plan_id": plan_id, "transfers": len(selected), "total_cost": total_cost}


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Compute inventory rebalancing plan")
    parser.add_argument("--solver", choices=["greedy", "lp"], default=None,
                        help="Solver method (default: from config)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Planning horizon in weeks (default: from config)")
    add_common_args(parser)
    parser.add_argument("--item", type=str, default=None,
                        help="Filter to a single item")
    parser.add_argument("--budget-cap", type=float, default=None,
                        help="Maximum transfer budget in dollars")
    args = parser.parse_args()
    run(
        solver=args.solver,
        horizon=args.horizon,
        dry_run=args.dry_run,
        item_filter=args.item,
        budget_cap=args.budget_cap,
    )
