"""
generate_planned_orders.py

Generates time-phased planned order recommendations for all active DFUs (or a single DFU).

Reads:
    fact_inventory_projection   — daily demand rates (no_order scenario)
    fact_open_purchase_orders   — confirmed inbound receipts by delivery date
    fact_safety_stock_targets   — safety stock and reorder point
    dim_replenishment_policy + fact_dfu_policy_assignment — MOQ, review_cycle_days
    dim_item_supplier           — lead time, unit cost, preferred supplier
    fact_inventory_snapshot     — current qty_on_hand

Writes:
    fact_planned_orders

Usage:
    uv run python scripts/generate_planned_orders.py [--dfu ITEM LOC] [--dry-run]
"""

import argparse
import math
import uuid
import yaml
from datetime import date, timedelta

import psycopg

from common.db import get_db_params

CONFIG_PATH = "config/order_recommendation_config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def round_to_moq(qty: float, moq: float, strategy: str = "ceil_to_moq") -> float:
    """
    Round a net requirement quantity up to the nearest MOQ multiple.

    Examples:
        round_to_moq(220, 100, 'ceil_to_moq') -> 300
        round_to_moq(200, 100, 'ceil_to_moq') -> 200  (exact multiple, no change)
        round_to_moq(  1, 100, 'ceil_to_moq') -> 100  (min = 1 MOQ)
        round_to_moq(  0, 100, 'ceil_to_moq') -> 100  (minimum 1 MOQ)
    """
    if moq <= 0:
        moq = 1.0
    if qty <= 0:
        return moq
    if strategy == "ceil_to_moq":
        return max(moq, math.ceil(qty / moq) * moq)
    elif strategy == "nearest_moq":
        return max(moq, round(qty / moq) * moq)
    else:  # floor_to_moq
        return max(moq, math.floor(qty / moq) * moq)


def compute_net_requirements(inputs: dict, config: dict) -> list:
    """
    Runs the time-phased net requirements calculation.

    Returns list of planned order dicts, one per reorder cycle within the horizon.

    Algorithm:
    1. Start from today's qty_on_hand
    2. For each future day (starting tomorrow), apply receipts then subtract demand
    3. When projected_position[d] <= reorder_point, that is trigger_date
    4. Compute lt_demand = sum of daily demand over [trigger_date, trigger_date + LT]
    5. Compute net_requirement = SS + lt_demand - projected_position[trigger_date]
    6. Round to MOQ
    7. Add planned receipt at trigger_date + LT to the simulation
    8. Continue until max_orders or end of horizon
    """
    horizon_days = config["recommendation"]["horizon_days"]
    max_orders = config["recommendation"]["max_orders_per_dfu"]
    moq_strategy = config["moq_handling"]["rounding_strategy"]

    qty = float(inputs["current_qty_on_hand"])
    ss = float(inputs["safety_stock"])
    rp = float(inputs["reorder_point"])
    lt = int(inputs["lead_time_days"])
    moq = float(inputs.get("moq") or 1.0)
    unit_cost = float(inputs.get("unit_cost") or 0.0)

    # Copy confirmed receipts so we can add planned orders to the simulation
    planned_receipts: dict = dict(inputs.get("confirmed_receipts_by_date") or {})
    demand_by_day: dict = dict(inputs.get("daily_demand_by_date") or {})

    orders = []
    today = date.today()

    # Simulate from tomorrow (day 1) — today's qty_on_hand is the starting state
    for i in range(1, horizon_days + 1):
        d = today + timedelta(days=i)
        daily_receipts = float(planned_receipts.get(d, 0.0))
        daily_demand = float(demand_by_day.get(d, 0.0))
        qty = max(0.0, qty + daily_receipts - daily_demand)

        if qty <= rp and len(orders) < max_orders:
            trigger_date = d
            order_by_date = trigger_date  # immediate (no placement lead time in MVP)

            # Sum forecast demand over lead time window
            lt_demand = sum(
                float(demand_by_day.get(trigger_date + timedelta(days=j), 0.0))
                for j in range(lt)
            )

            net_req = max(0.0, ss + lt_demand - qty)
            order_qty = round_to_moq(net_req, moq, moq_strategy)

            receipt_date = order_by_date + timedelta(days=lt)
            confirmed_inbound = sum(
                float(v) for v in (inputs.get("confirmed_receipts_by_date") or {}).values()
            )

            orders.append({
                "item_no": inputs["item_no"],
                "loc": inputs["loc"],
                "supplier_id": inputs.get("supplier_id"),
                "policy_id": inputs.get("policy_id"),
                "net_requirement_qty": round(net_req, 2),
                "recommended_qty": order_qty,
                "moq": moq,
                "unit_cost": unit_cost if unit_cost > 0 else None,
                "currency": "USD",
                "trigger_date": trigger_date,
                "trigger_reason": "projected_below_ss",
                "order_by_date": order_by_date,
                "expected_receipt_date": receipt_date,
                "lead_time_days": lt,
                "review_cycle_days": inputs.get("review_cycle_days"),
                "current_qty_on_hand": float(inputs["current_qty_on_hand"]),
                "safety_stock": ss,
                "reorder_point": rp,
                "confirmed_inbound_qty": round(confirmed_inbound, 2),
                "lt_forecast_demand": round(lt_demand, 2),
                "plan_version": inputs.get("plan_version"),
                "status": "proposed",
                "run_id": inputs["run_id"],
            })

            # Add the planned receipt into the simulation so the next cycle is correct
            planned_receipts[receipt_date] = (
                planned_receipts.get(receipt_date, 0.0) + order_qty
            )
            qty += order_qty  # account for planned receipt immediately

    return orders


def compute_confidence_score(inputs: dict, orders: list, config: dict) -> tuple:
    """
    Returns (score: float, reason: str).
    Score degrades based on data quality flags.
    """
    score = 1.0
    reasons = []
    penalties = config["recommendation"]["confidence"]

    if inputs.get("forecast_source") == "fallback_avg":
        score -= penalties["penalty_fallback_forecast"]
        reasons.append("using fallback demand average (no production forecast)")

    if not inputs.get("open_po_data_available", True):
        score -= penalties["penalty_no_open_po_data"]
        reasons.append("open PO delivery dates unavailable")

    today = date.today()
    for order in orders:
        if order["order_by_date"] < today:
            score -= penalties["penalty_past_due_order"]
            reasons.append("order already past due")
            break

    score = round(max(0.0, min(1.0, score)), 3)
    reason = "; ".join(reasons) if reasons else "all data sources available"
    return score, reason


def get_active_dfus_for_recommendation(conn) -> list:
    """Returns list of (item_no, loc) tuples that have inventory projection data."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT item_no, loc
            FROM fact_inventory_projection
            WHERE scenario = 'no_order'
            ORDER BY item_no, loc
        """)
        return cur.fetchall()


def get_dfu_inputs(item_no: str, loc: str, run_id: str, conn) -> dict:
    """
    Assembles all inputs needed for net requirement calculation for one DFU.
    """
    result: dict = {"item_no": item_no, "loc": loc, "run_id": run_id}

    with conn.cursor() as cur:
        # 1. Current inventory
        cur.execute("""
            SELECT qty_on_hand
            FROM fact_inventory_snapshot
            WHERE item_no = %s AND loc = %s
            ORDER BY snapshot_date DESC LIMIT 1
        """, (item_no, loc))
        row = cur.fetchone()
        result["current_qty_on_hand"] = float(row[0]) if row else 0.0

        # 2. Safety stock + reorder point
        cur.execute("""
            SELECT ss_combined, COALESCE(reorder_point, ss_combined)
            FROM fact_safety_stock_targets
            WHERE item_no = %s AND loc = %s
            ORDER BY computed_at DESC LIMIT 1
        """, (item_no, loc))
        row = cur.fetchone()
        if row:
            result["safety_stock"] = float(row[0] or 0)
            result["reorder_point"] = float(row[1] or 0)
        else:
            result["safety_stock"] = 0.0
            result["reorder_point"] = 0.0

        # 3. Policy, MOQ, lead time, unit cost, supplier
        cur.execute("""
            SELECT p.id, p.review_cycle_days, COALESCE(p.moq, 1),
                   COALESCE(s.lead_time_days, 14), s.price_per_unit, s.supplier_id
            FROM fact_dfu_policy_assignment pa
            JOIN dim_replenishment_policy p ON p.id = pa.policy_id
            LEFT JOIN dim_item_supplier s ON s.item_no = pa.item_no
                                         AND s.loc = pa.loc
                                         AND s.is_preferred = TRUE
            WHERE pa.item_no = %s AND pa.loc = %s
            LIMIT 1
        """, (item_no, loc))
        row = cur.fetchone()
        if row:
            result["policy_id"] = row[0]
            result["review_cycle_days"] = row[1]
            result["moq"] = float(row[2] or 1)
            result["lead_time_days"] = int(row[3] or 14)
            result["unit_cost"] = float(row[4]) if row[4] else 0.0
            result["supplier_id"] = row[5]
        else:
            result["policy_id"] = None
            result["review_cycle_days"] = None
            result["moq"] = 1.0
            result["lead_time_days"] = 14
            result["unit_cost"] = 0.0
            result["supplier_id"] = None

        # 4. Daily demand rates from inventory projection (no_order scenario)
        cur.execute("""
            SELECT projection_date, daily_demand_rate, forecast_source, plan_version
            FROM fact_inventory_projection
            WHERE item_no = %s AND loc = %s AND scenario = 'no_order'
            ORDER BY projection_date
        """, (item_no, loc))
        rows = cur.fetchall()
        demand_by_day = {}
        forecast_source = "fallback_avg"
        plan_version = None
        for r in rows:
            demand_by_day[r[0]] = float(r[1] or 0)
            forecast_source = r[2] or "fallback_avg"
            plan_version = r[3]
        result["daily_demand_by_date"] = demand_by_day
        result["forecast_source"] = forecast_source
        result["plan_version"] = plan_version

        # 5. Confirmed receipts from open POs
        cur.execute("""
            SELECT effective_delivery_date, SUM(open_qty) AS expected_qty
            FROM fact_open_purchase_orders
            WHERE item_no = %s AND loc = %s
              AND line_status NOT IN ('closed', 'cancelled')
              AND effective_delivery_date >= CURRENT_DATE
            GROUP BY effective_delivery_date
        """, (item_no, loc))
        rows = cur.fetchall()
        confirmed_receipts = {r[0]: float(r[1] or 0) for r in rows}
        result["confirmed_receipts_by_date"] = confirmed_receipts
        result["open_po_data_available"] = len(confirmed_receipts) > 0

    return result


def write_planned_orders(orders: list, dry_run: bool, conn) -> int:
    """Inserts planned orders into fact_planned_orders. Returns count written."""
    if not orders or dry_run:
        return len(orders)

    sql = """
        INSERT INTO fact_planned_orders (
            item_no, loc, supplier_id, policy_id,
            net_requirement_qty, recommended_qty, moq, unit_cost, currency,
            trigger_date, trigger_reason, order_by_date, expected_receipt_date,
            lead_time_days, review_cycle_days,
            current_qty_on_hand, safety_stock, reorder_point,
            confirmed_inbound_qty, lt_forecast_demand, plan_version,
            confidence_score, confidence_reason, status, run_id
        ) VALUES (
            %(item_no)s, %(loc)s, %(supplier_id)s, %(policy_id)s,
            %(net_requirement_qty)s, %(recommended_qty)s, %(moq)s, %(unit_cost)s, %(currency)s,
            %(trigger_date)s, %(trigger_reason)s, %(order_by_date)s, %(expected_receipt_date)s,
            %(lead_time_days)s, %(review_cycle_days)s,
            %(current_qty_on_hand)s, %(safety_stock)s, %(reorder_point)s,
            %(confirmed_inbound_qty)s, %(lt_forecast_demand)s, %(plan_version)s,
            %(confidence_score)s, %(confidence_reason)s, %(status)s, %(run_id)s
        )
        ON CONFLICT DO NOTHING
    """
    with conn.cursor() as cur:
        cur.executemany(sql, orders)
    conn.commit()
    return len(orders)


def main():
    parser = argparse.ArgumentParser(description="Generate planned order recommendations")
    parser.add_argument("--dfu", nargs=2, metavar=("ITEM", "LOC"),
                        help="Generate for a single DFU")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute without writing to DB")
    args = parser.parse_args()

    config = load_config()
    run_id = str(uuid.uuid4())
    total_written = 0

    with psycopg.connect(**get_db_params()) as conn:
        if args.dfu:
            dfus = [(args.dfu[0], args.dfu[1])]
        else:
            dfus = get_active_dfus_for_recommendation(conn)

        print(f"Generating planned orders for {len(dfus)} DFU(s) | run_id={run_id}")

        for item_no, loc in dfus:
            try:
                inputs = get_dfu_inputs(item_no, loc, run_id, conn)
                inputs["run_id"] = run_id
                orders = compute_net_requirements(inputs, config)

                confidence, confidence_reason = compute_confidence_score(
                    inputs, orders, config
                )
                for order in orders:
                    order["confidence_score"] = confidence
                    order["confidence_reason"] = confidence_reason

                written = write_planned_orders(orders, args.dry_run, conn)
                total_written += written

                if orders:
                    print(f"  {item_no} @ {loc}: {len(orders)} order(s) | "
                          f"confidence={confidence:.3f}")
            except Exception as e:
                print(f"  ERROR {item_no} @ {loc}: {e}")

    suffix = " (dry-run)" if args.dry_run else ""
    print(f"Done. Total orders generated: {total_written}{suffix}")


if __name__ == "__main__":
    main()
