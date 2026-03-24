"""
F2.3 — Consensus Forecasting & Planner Overrides

Generate the consensus demand plan by merging the statistical baseline
(fact_demand_plan P50) with approved planner overrides (fact_forecast_overrides).

Usage:
    uv run scripts/generate_consensus_plan.py \\
        --plan-version 2026-04-01_production \\
        --months-ahead 12 \\
        --dry-run

Config: config/consensus_config.yaml
Output: fact_consensus_plan
"""

import argparse
import sys
import os
from datetime import date

import psycopg
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common.db import get_db_params
from common.planning_date import get_planning_date
from common.services.perf_profiler import profiled_section

# ---------------------------------------------------------------------------
# Override type priority (lower = higher priority in conflict resolution)
# ---------------------------------------------------------------------------
OVERRIDE_PRIORITY = {
    "CAPACITY_LOCK": 1,
    "PROMO": 2,
    "LAUNCH": 2,
    "PHASE_OUT": 3,
    "MARKET_EVENT": 3,
    "MANUAL": 4,
}


# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------

def apply_override(
    statistical_qty: float,
    override_type: str,
    override_qty: float | None,
    override_multiplier: float | None,
    override_additive_qty: float,
    is_hard_override: bool,
) -> tuple[float, float]:
    """
    Apply a single override to the statistical baseline quantity.

    Args:
        statistical_qty: P50 from fact_demand_plan
        override_type: PROMO, LAUNCH, PHASE_OUT, MARKET_EVENT, CAPACITY_LOCK, MANUAL
        override_qty: Absolute quantity (for MANUAL, LAUNCH, CAPACITY_LOCK hard overrides)
        override_multiplier: Multiplicative factor (for PROMO, MARKET_EVENT, PHASE_OUT)
        override_additive_qty: Additive lift (added after multiplier)
        is_hard_override: If True, override_qty completely replaces statistical qty

    Returns:
        Tuple of (consensus_qty, override_delta_units)
    """
    if is_hard_override and override_qty is not None:
        consensus_qty = max(0.0, float(override_qty))
        delta = consensus_qty - statistical_qty
        return round(consensus_qty, 2), round(delta, 2)

    multiplier = float(override_multiplier) if override_multiplier is not None else 1.0
    additive = float(override_additive_qty) if override_additive_qty else 0.0
    consensus_qty = max(0.0, statistical_qty * multiplier + additive)
    delta = consensus_qty - statistical_qty
    return round(consensus_qty, 2), round(delta, 2)


def resolve_conflicts(overrides: list[dict]) -> dict:
    """
    Given multiple overrides for the same DFU-month, select the winning override.

    Priority:
      1. CAPACITY_LOCK type (always wins)
      2. Lowest type_priority value (PROMO=2, MANUAL=4, etc.)
      3. Lowest priority_rank number (1=highest)
      4. Most recent created_at as tiebreaker

    Returns the winning override dict.
    """
    if len(overrides) == 1:
        return overrides[0]

    # Annotate type_priority
    for o in overrides:
        o["_type_priority"] = OVERRIDE_PRIORITY.get(o["override_type"], 99)

    # Sort: type_priority ASC, priority_rank ASC, created_at DESC
    sorted_overrides = sorted(
        overrides,
        key=lambda o: (o["_type_priority"], o["priority_rank"], -o["created_at"].timestamp()
                       if hasattr(o["created_at"], "timestamp") else 0),
    )
    winner = sorted_overrides[0]

    n_conflicts = len(overrides) - 1
    print(
        f"[CONFLICT] {winner['item_id']}@{winner['loc']} "
        f"{winner['override_month']}: {n_conflicts} override(s) superseded "
        f"by {winner['override_type']} (override_id={winner['override_id']})"
    )
    return winner


# ---------------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------------

def load_statistical_baseline(plan_version: str, conn) -> list[dict]:
    """Load P10/P50/P90 from fact_demand_plan pivoted by plan_month."""
    sql = """
        SELECT
            item_id, loc, plan_month,
            MAX(CASE WHEN quantile = 0.10 THEN forecast_qty END) AS p10,
            MAX(CASE WHEN quantile = 0.50 THEN forecast_qty END) AS p50,
            MAX(CASE WHEN quantile = 0.90 THEN forecast_qty END) AS p90
        FROM fact_demand_plan
        WHERE plan_version = %s
        GROUP BY item_id, loc, plan_month
        ORDER BY item_id, loc, plan_month
    """
    with conn.cursor() as cur:
        cur.execute(sql, (plan_version,))
        rows = cur.fetchall()
    return [
        {"item_id": r[0], "loc": r[1], "plan_month": r[2],
         "p10": float(r[3]) if r[3] is not None else None,
         "p50": float(r[4]) if r[4] is not None else 0.0,
         "p90": float(r[5]) if r[5] is not None else None}
        for r in rows
    ]


def load_approved_overrides(plan_run_date: date, conn) -> list[dict]:
    """Load all approved, non-expired overrides valid as of plan_run_date."""
    sql = """
        SELECT
            override_id, item_id, loc, override_month,
            override_type, override_qty, override_multiplier,
            override_additive_qty, is_hard_override,
            priority_rank, created_at, approved_by
        FROM fact_forecast_overrides
        WHERE status = 'approved'
          AND valid_from <= %s
          AND valid_to   >= %s
        ORDER BY item_id, loc, override_month,
                 priority_rank ASC, created_at DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (plan_run_date, plan_run_date))
        rows = cur.fetchall()
    return [
        {
            "override_id": r[0],
            "item_id": r[1],
            "loc": r[2],
            "override_month": r[3],
            "override_type": r[4],
            "override_qty": float(r[5]) if r[5] is not None else None,
            "override_multiplier": float(r[6]) if r[6] is not None else None,
            "override_additive_qty": float(r[7]) if r[7] is not None else 0.0,
            "is_hard_override": bool(r[8]),
            "priority_rank": r[9],
            "created_at": r[10],
            "approved_by": r[11],
        }
        for r in rows
    ]


def expire_stale_overrides(plan_run_date: date, conn) -> int:
    """Set status='expired' for any approved overrides with valid_to < plan_run_date."""
    sql = """
        UPDATE fact_forecast_overrides
        SET status = 'expired'
        WHERE status = 'approved'
          AND expires_auto = TRUE
          AND valid_to < %s
        RETURNING override_id
    """
    with conn.cursor() as cur:
        cur.execute(sql, (plan_run_date,))
        return cur.rowcount


def write_consensus_rows(rows: list[dict], conn) -> int:
    """Upsert consensus plan rows into fact_consensus_plan."""
    if not rows:
        return 0
    sql = """
        INSERT INTO fact_consensus_plan
            (item_id, loc, plan_month, plan_version,
             statistical_qty, statistical_p10, statistical_p90,
             override_qty, consensus_qty, consensus_p10, consensus_p90,
             override_applied, override_id, override_type, override_multiplier,
             is_hard_override, overrider, approver, uplift_pct, generated_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
        ON CONFLICT (item_id, loc, plan_month, plan_version)
        DO UPDATE SET
            statistical_qty  = EXCLUDED.statistical_qty,
            override_qty     = EXCLUDED.override_qty,
            consensus_qty    = EXCLUDED.consensus_qty,
            override_applied = EXCLUDED.override_applied,
            override_id      = EXCLUDED.override_id,
            uplift_pct       = EXCLUDED.uplift_pct,
            generated_at     = NOW()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, [
            (
                r["item_id"], r["loc"], r["plan_month"], r["plan_version"],
                r["statistical_qty"], r["statistical_p10"], r["statistical_p90"],
                r["override_qty"], r["consensus_qty"], r["consensus_p10"], r["consensus_p90"],
                r["override_applied"], r.get("override_id"), r.get("override_type"),
                r.get("override_multiplier"), r.get("is_hard_override", False),
                r.get("overrider"), r.get("approver"), r.get("uplift_pct"),
            )
            for r in rows
        ])
    return len(rows)


# ---------------------------------------------------------------------------
# Main plan generation
# ---------------------------------------------------------------------------

def generate_consensus_plan(
    plan_version: str,
    plan_run_date: date | None = None,
    months_ahead: int = 12,
    dry_run: bool = False,
) -> dict:
    """
    Merge statistical baseline (F2.2) with approved planner overrides.

    Returns summary dict: total_rows, overridden_rows, total_override_impact_units
    """
    cfg = yaml.safe_load(open("config/consensus_config.yaml"))
    adjust_p = cfg["consensus_plan"]["consensus_plan_output"]["adjust_p10_p90_proportionally"]

    if plan_run_date is None:
        plan_run_date = get_planning_date()

    with psycopg.connect(**get_db_params()) as conn:
        # Step 1: expire stale overrides
        with profiled_section("expire_stale_overrides"):
            n_expired = expire_stale_overrides(plan_run_date, conn)
        if n_expired:
            print(f"Expired {n_expired} stale overrides.")

        # Step 2: load baseline and overrides
        with profiled_section("load_baseline_and_overrides"):
            baseline = load_statistical_baseline(plan_version, conn)
            overrides = load_approved_overrides(plan_run_date, conn)

    if not baseline:
        print(f"No statistical baseline found for plan_version={plan_version}.")
        return {"total_rows": 0, "overridden_rows": 0, "total_override_impact_units": 0.0}

    # Step 3: pre-sort ALL overrides once and build a resolved lookup
    # so we avoid re-sorting per DFU-month during conflict resolution
    from collections import defaultdict
    with profiled_section("resolve_override_conflicts"):
        override_by_dfu_month: dict = defaultdict(list)
        for o in overrides:
            o["_type_priority"] = OVERRIDE_PRIORITY.get(o["override_type"], 99)
            key = (o["item_id"], o["loc"], o["override_month"])
            override_by_dfu_month[key].append(o)

        # Pre-resolve conflicts: pick the winning override per key
        override_lookup: dict[tuple, dict] = {}
        for key, ov_list in override_by_dfu_month.items():
            if len(ov_list) == 1:
                override_lookup[key] = ov_list[0]
            else:
                # Sort once: type_priority ASC, priority_rank ASC, created_at DESC
                sorted_ovs = sorted(
                    ov_list,
                    key=lambda o: (
                        o["_type_priority"],
                        o["priority_rank"],
                        -o["created_at"].timestamp() if hasattr(o["created_at"], "timestamp") else 0,
                    ),
                )
                winner = sorted_ovs[0]
                n_conflicts = len(ov_list) - 1
                print(
                    f"[CONFLICT] {winner['item_id']}@{winner['loc']} "
                    f"{winner['override_month']}: {n_conflicts} override(s) superseded "
                    f"by {winner['override_type']} (override_id={winner['override_id']})"
                )
                override_lookup[key] = winner

    # Step 4: merge
    consensus_rows = []
    n_overridden = 0
    total_impact = 0.0

    with profiled_section("merge_baseline_overrides"):
        for b in baseline:
            key = (b["item_id"], b["loc"], b["plan_month"])
            ov = override_lookup.get(key)

            if ov:
                consensus_qty, delta = apply_override(
                    statistical_qty=b["p50"],
                    override_type=ov["override_type"],
                    override_qty=ov["override_qty"],
                    override_multiplier=ov["override_multiplier"],
                    override_additive_qty=ov["override_additive_qty"],
                    is_hard_override=ov["is_hard_override"],
                )

                # Scale P10/P90 proportionally if configured
                ratio = (consensus_qty / b["p50"]) if b["p50"] and b["p50"] > 0 else 1.0
                c_p10 = round(b["p10"] * ratio, 2) if b["p10"] is not None and adjust_p else b["p10"]
                c_p90 = round(b["p90"] * ratio, 2) if b["p90"] is not None and adjust_p else b["p90"]

                uplift_pct = round((delta / b["p50"]) * 100, 4) if b["p50"] else 0.0
                n_overridden += 1
                total_impact += abs(delta)

                consensus_rows.append({
                    "item_id": b["item_id"],
                    "loc": b["loc"],
                    "plan_month": b["plan_month"],
                    "plan_version": plan_version,
                    "statistical_qty": b["p50"],
                    "statistical_p10": b["p10"],
                    "statistical_p90": b["p90"],
                    "override_qty": delta,
                    "consensus_qty": consensus_qty,
                    "consensus_p10": c_p10,
                    "consensus_p90": c_p90,
                    "override_applied": True,
                    "override_id": ov["override_id"],
                    "override_type": ov["override_type"],
                    "override_multiplier": ov.get("override_multiplier"),
                    "is_hard_override": ov["is_hard_override"],
                    "overrider": ov.get("approved_by", ""),
                    "approver": ov.get("approved_by", ""),
                    "uplift_pct": uplift_pct,
                })
            else:
                consensus_rows.append({
                    "item_id": b["item_id"],
                    "loc": b["loc"],
                    "plan_month": b["plan_month"],
                    "plan_version": plan_version,
                    "statistical_qty": b["p50"],
                    "statistical_p10": b["p10"],
                    "statistical_p90": b["p90"],
                    "override_qty": 0.0,
                    "consensus_qty": b["p50"],
                    "consensus_p10": b["p10"],
                    "consensus_p90": b["p90"],
                    "override_applied": False,
                    "uplift_pct": 0.0,
                })

    if dry_run:
        print(f"[dry-run] Would write {len(consensus_rows)} consensus rows "
              f"({n_overridden} with overrides, impact={total_impact:.0f} units).")
    else:
        with profiled_section("write_consensus_rows"):
            with psycopg.connect(**get_db_params()) as conn:
                n_written = write_consensus_rows(consensus_rows, conn)
                conn.commit()
        print(f"Written {n_written} consensus rows ({n_overridden} overridden, "
              f"total impact={total_impact:.0f} units).")

    return {
        "total_rows": len(consensus_rows),
        "overridden_rows": n_overridden,
        "total_override_impact_units": round(total_impact, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate consensus demand plan.")
    parser.add_argument("--plan-version", required=True, help="Plan version string")
    parser.add_argument("--months-ahead", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    generate_consensus_plan(
        plan_version=args.plan_version,
        months_ahead=args.months_ahead,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
