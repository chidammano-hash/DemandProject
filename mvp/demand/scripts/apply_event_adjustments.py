"""
apply_event_adjustments.py — F4.3 Promotion & Event Planning

Applies approved event uplifts/reductions to statistical forecasts,
producing adjusted forecasts in fact_event_adjusted_forecast.
Also computes post-event performance metrics.

Usage:
    uv run python scripts/apply_event_adjustments.py
    uv run python scripts/apply_event_adjustments.py --event-id 42
    uv run python scripts/apply_event_adjustments.py --month 2026-05-01
    uv run python scripts/apply_event_adjustments.py --dry-run

Config: config/event_planning_config.yaml
"""

from __future__ import annotations

import argparse
import yaml
import psycopg
from datetime import date
from typing import Optional

from common.db import get_db_params

CONFIG_PATH = "config/event_planning_config.yaml"

EVENT_TYPES = {"PROMOTION", "HOLIDAY", "PRODUCT_LAUNCH", "PHASE_OUT", "DISRUPTION", "TRADE_SHOW"}


def load_config(path: str = CONFIG_PATH) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f).get("event_planning", {})
    except FileNotFoundError:
        return {
            "max_uplift_multiplier": 5.0,
            "min_uplift_multiplier": 0.0,
            "require_approval_above_impact_value": 5000.0,
            "post_event_lag_weeks": 2,
        }


def apply_event_uplift(
    base_qty: float,
    uplift_multiplier: float,
    additive_qty: float = 0.0,
    is_hard_override: bool = False,
    override_qty: Optional[float] = None,
    max_multiplier: float = 5.0,
) -> tuple[float, float]:
    """
    Apply event uplift to a base forecast quantity.

    Hard override: adjusted_qty = override_qty
    Soft override: adjusted_qty = base_qty × clamp(multiplier, 0, max) + additive

    Args:
        base_qty: Statistical forecast quantity
        uplift_multiplier: Multiplicative factor (e.g., 1.40 = +40%)
        additive_qty: Absolute additive quantity on top of multiplied base
        is_hard_override: If True, use override_qty instead of formula
        override_qty: Hard override quantity (only used if is_hard_override=True)
        max_multiplier: Maximum allowed multiplier (guard rail)

    Returns:
        (adjusted_qty, uplift_delta) — both rounded to 2 decimal places

    Examples:
        apply_event_uplift(450, 1.40) → (630.0, 180.0)
        apply_event_uplift(450, 1.0, additive_qty=50) → (500.0, 50.0)
        apply_event_uplift(450, 0, is_hard_override=True, override_qty=300) → (300.0, -150.0)
    """
    if is_hard_override and override_qty is not None:
        adjusted = max(0.0, override_qty)
        delta = adjusted - base_qty
        return round(adjusted, 2), round(delta, 2)

    clamped_mult = max(0.0, min(max_multiplier, uplift_multiplier))
    adjusted = max(0.0, base_qty * clamped_mult + additive_qty)
    delta = adjusted - base_qty
    return round(adjusted, 2), round(delta, 2)


def compute_event_impact_value(
    uplift_delta_units: float,
    avg_unit_cost: float,
) -> float:
    """
    Estimate financial impact of an event in dollars.

    impact_value = |uplift_delta_units| × avg_unit_cost

    Args:
        uplift_delta_units: Absolute change in units (can be negative for phase-outs)
        avg_unit_cost: Average unit cost in $

    Returns:
        Financial impact in $ (always positive — magnitude of change)
    """
    return round(abs(uplift_delta_units) * avg_unit_cost, 2)


def compute_post_event_accuracy(
    adjusted_qty: float,
    actual_qty: float,
) -> tuple[float, float]:
    """
    Compute post-event forecast accuracy.

    bias = (adjusted - actual) / actual   [0 if actual=0]
    abs_error_pct = |adjusted - actual| / actual × 100

    Returns:
        (bias_pct, abs_error_pct)
    """
    if actual_qty <= 0:
        return 0.0, 0.0
    bias = round((adjusted_qty - actual_qty) / actual_qty * 100, 2)
    abs_err = round(abs(adjusted_qty - actual_qty) / actual_qty * 100, 2)
    return bias, abs_err


def fetch_approved_events(
    conn: psycopg.Connection,
    event_id: Optional[int] = None,
    month: Optional[date] = None,
) -> list[dict]:
    """Fetch approved events from fact_event_calendar."""
    conditions = ["status = 'approved'"]
    params: list = []
    if event_id:
        conditions.append("event_id = %s")
        params.append(event_id)
    if month:
        conditions.append("event_start_date <= %s AND event_end_date >= %s")
        params.extend([month.replace(day=28), month])
    where = " AND ".join(conditions)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT event_id, event_name, event_type, item_no, loc,
                   event_start_date, event_end_date,
                   uplift_multiplier, additive_qty, is_hard_override, override_qty
            FROM fact_event_calendar
            WHERE {where}
            ORDER BY event_start_date, event_id
            """,
            params,
        )
        rows = cur.fetchall()
    return [
        {
            "event_id": r[0],
            "event_name": r[1],
            "event_type": r[2],
            "item_no": r[3],
            "loc": r[4],
            "start_date": r[5],
            "end_date": r[6],
            "uplift_multiplier": float(r[7]) if r[7] else 1.0,
            "additive_qty": float(r[8]) if r[8] else 0.0,
            "is_hard_override": bool(r[9]),
            "override_qty": float(r[10]) if r[10] else None,
        }
        for r in rows
    ]


def fetch_base_forecast(
    conn: psycopg.Connection,
    item_no: str,
    loc: str,
    start_date: date,
    end_date: date,
) -> list[dict]:
    """Fetch statistical forecast for the event period."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT startdate, basefcst_pref
            FROM fact_external_forecast_monthly
            WHERE dmdunit = %s AND loc = %s
              AND model_id = 'champion'
              AND startdate >= %s AND startdate <= %s
            ORDER BY startdate
            """,
            (item_no, loc, start_date, end_date),
        )
        rows = cur.fetchall()
    return [
        {"plan_month": r[0], "stat_qty": float(r[1]) if r[1] else 0.0}
        for r in rows
    ]


def upsert_adjusted_forecasts(conn: psycopg.Connection, rows: list[dict]) -> int:
    sql = """
        INSERT INTO fact_event_adjusted_forecast (
            event_id, item_no, loc, plan_month,
            base_forecast_qty, uplift_multiplier, additive_qty,
            adjusted_forecast_qty, uplift_delta_units, impact_value_usd,
            is_hard_override
        ) VALUES (
            %(event_id)s, %(item_no)s, %(loc)s, %(plan_month)s,
            %(base_forecast_qty)s, %(uplift_multiplier)s, %(additive_qty)s,
            %(adjusted_forecast_qty)s, %(uplift_delta_units)s, %(impact_value_usd)s,
            %(is_hard_override)s
        )
        ON CONFLICT (event_id, item_no, loc, plan_month)
        DO UPDATE SET
            base_forecast_qty       = EXCLUDED.base_forecast_qty,
            uplift_multiplier       = EXCLUDED.uplift_multiplier,
            adjusted_forecast_qty   = EXCLUDED.adjusted_forecast_qty,
            uplift_delta_units      = EXCLUDED.uplift_delta_units,
            impact_value_usd        = EXCLUDED.impact_value_usd,
            computed_at             = NOW()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def run(
    event_id: Optional[int] = None,
    month_str: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """Main entry point: apply event adjustments to forecasts."""
    cfg = load_config()
    max_multiplier = cfg.get("max_uplift_multiplier", 5.0)

    month: Optional[date] = date.fromisoformat(month_str) if month_str else None

    rows_to_write: list[dict] = []
    events_processed = 0

    with psycopg.connect(**get_db_params()) as conn:
        events = fetch_approved_events(conn, event_id, month)

        for event in events:
            events_processed += 1
            item = event["item_no"]
            loc = event["loc"]

            base_forecasts = fetch_base_forecast(
                conn, item, loc, event["start_date"], event["end_date"]
            )

            for forecast in base_forecasts:
                adjusted, delta = apply_event_uplift(
                    base_qty=forecast["stat_qty"],
                    uplift_multiplier=event["uplift_multiplier"],
                    additive_qty=event["additive_qty"],
                    is_hard_override=event["is_hard_override"],
                    override_qty=event["override_qty"],
                    max_multiplier=max_multiplier,
                )
                impact_value = compute_event_impact_value(delta, avg_unit_cost=0.0)

                rows_to_write.append({
                    "event_id": event["event_id"],
                    "item_no": item,
                    "loc": loc,
                    "plan_month": forecast["plan_month"],
                    "base_forecast_qty": forecast["stat_qty"],
                    "uplift_multiplier": event["uplift_multiplier"],
                    "additive_qty": event["additive_qty"],
                    "adjusted_forecast_qty": adjusted,
                    "uplift_delta_units": delta,
                    "impact_value_usd": impact_value,
                    "is_hard_override": event["is_hard_override"],
                })

        rows_written = 0
        if not dry_run and rows_to_write:
            rows_written = upsert_adjusted_forecasts(conn, rows_to_write)
            conn.commit()

    return {
        "events_processed": events_processed,
        "forecast_rows_computed": len(rows_to_write),
        "rows_written": rows_written,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply event adjustments to forecasts")
    parser.add_argument("--event-id", type=int, help="Process a specific event only")
    parser.add_argument("--month", help="Process events overlapping a given month (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(args.event_id, args.month, args.dry_run)
    print(result)
