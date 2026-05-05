#!/usr/bin/env python3
"""Compute forward-looking replenishment plan from production forecast CI bands.

Reads:
  - fact_production_forecast (CI bands + point forecasts)
  - fact_dfu_policy_assignment + dim_replenishment_policy (policy type per DFU)
  - fact_inventory_snapshot (lead time data — avg/std per item_id + loc)
  - fact_safety_stock_targets (historical SS for delta comparison)
  - agg_inventory_monthly (current on-hand position)
  - dim_sku (ABC class, demand_std fallback, ml_cluster)

Writes:
  - fact_replenishment_plan (upsert on plan_version + item_id + loc + plan_month)

Usage:
    uv run python scripts/compute_replenishment_plan.py
    uv run python scripts/compute_replenishment_plan.py --plan-version 2026-02
    uv run python scripts/compute_replenishment_plan.py --dry-run
    uv run python scripts/compute_replenishment_plan.py --item 100320 --loc 1401-BULK
    uv run python scripts/compute_replenishment_plan.py --config config/inventory/replenishment_plan_config.yaml
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.core.db import get_db_params
from common.services.perf_profiler import profiled_section
from common.core.utils import load_config as _load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = "config/inventory/replenishment_plan_config.yaml"
DAYS_PER_MONTH: float = 30.44


# ---------------------------------------------------------------------------
# Pure formula functions (unit-testable, no DB dependencies)
# ---------------------------------------------------------------------------


def compute_sigma_from_ci(
    lower: float | None,
    upper: float | None,
    ci_z_score: float,
) -> float | None:
    """Derive monthly demand sigma from P10/P90 CI bands.

    sigma = (upper - lower) / (2 * ci_z_score)

    Args:
        lower:        Lower CI bound (e.g. P10 forecast quantity).
        upper:        Upper CI bound (e.g. P90 forecast quantity).
        ci_z_score:   Z-score corresponding to the CI width (e.g. 1.282 for P10/P90).

    Returns:
        Estimated monthly demand standard deviation, or None when either bound
        is missing or the interval is degenerate (upper <= lower).
    """
    if lower is None or upper is None or upper <= lower:
        return None
    return (upper - lower) / (2.0 * ci_z_score)


def compute_forward_ss(
    z: float,
    sigma_monthly: float,
    lt_mean_days: float,
    lt_std_days: float,
    avg_daily_demand: float,
) -> dict[str, float]:
    """Compute forward-looking safety stock components.

    Uses the combined uncorrelated formula:
        SS_demand   = z * sigma_monthly * sqrt(lt_mean_days / DAYS_PER_MONTH)
        SS_lt       = avg_daily_demand * lt_std_days * z
        SS_combined = sqrt(SS_demand^2 + SS_lt^2)

    Args:
        z:                Z-score for the target service level.
        sigma_monthly:    Monthly demand standard deviation (units).
        lt_mean_days:     Mean lead time in days.
        lt_std_days:      Std dev of lead time in days.
        avg_daily_demand: Average daily demand (units/day).

    Returns:
        dict with keys: ss_demand_only, ss_lt_only, ss_combined (all in units).
    """
    ss_demand = z * sigma_monthly * math.sqrt(lt_mean_days / DAYS_PER_MONTH)
    ss_lt = avg_daily_demand * lt_std_days * z
    ss_combined = math.sqrt(ss_demand ** 2 + ss_lt ** 2)
    return {
        "ss_demand_only": round(ss_demand, 4),
        "ss_lt_only": round(ss_lt, 4),
        "ss_combined": round(ss_combined, 4),
    }


def apply_ss_guard_rails(
    ss: float,
    avg_daily_demand: float,
    min_ss_days: int,
    max_ss_days: int,
) -> float:
    """Clamp safety stock to [min_ss_days, max_ss_days] of average daily demand.

    Guard rails are only meaningful when avg_daily_demand > 0; for zero-demand
    items the raw SS value is returned unchanged (formula already returns 0).

    Args:
        ss:               Raw SS quantity to clamp.
        avg_daily_demand: Average daily demand (units/day).
        min_ss_days:      Minimum SS expressed as days of supply.
        max_ss_days:      Maximum SS expressed as days of supply.

    Returns:
        Clamped SS quantity in units.
    """
    if avg_daily_demand <= 0:
        return max(0.0, ss)
    min_ss = avg_daily_demand * min_ss_days
    max_ss = avg_daily_demand * max_ss_days
    return float(np.clip(ss, min_ss, max_ss))


def compute_forward_eoq(annual_demand: float, cfg: dict) -> dict[str, float]:
    """Compute EOQ from forecasted annual demand using config cost parameters.

    EOQ             = sqrt(2 * D * ordering_cost / (unit_cost * holding_cost_pct))
    effective_EOQ   = max(MOQ, min(EOQ, D * max_months / 12))
    cycle_stock     = effective_EOQ / 2

    Args:
        annual_demand: Annualised forecast demand (units/year).
        cfg:           Full replenishment_plan config dict (top-level key).

    Returns:
        dict with keys: eoq, effective_eoq, cycle_stock (all in units).
    """
    costs = cfg.get("costs", {})
    ordering_cost: float = float(costs.get("default_ordering_cost", 50.0))
    holding_cost_pct: float = float(costs.get("default_holding_cost_pct", 0.25))
    unit_cost: float = float(costs.get("default_unit_cost", 1.0))
    moq: float = float(costs.get("default_moq", 1))
    max_months: float = float(cfg.get("constraints", {}).get("max_eoq_months_supply", 6))
    min_demand: float = float(cfg.get("constraints", {}).get("min_annual_demand", 0.001))

    if annual_demand < min_demand:
        return {"eoq": 0.0, "effective_eoq": moq, "cycle_stock": moq / 2.0}

    holding_cost = unit_cost * holding_cost_pct
    eoq = math.sqrt(2.0 * annual_demand * ordering_cost / holding_cost)
    max_qty = annual_demand * max_months / 12.0
    effective_eoq = max(moq, min(eoq, max_qty))

    return {
        "eoq": round(eoq, 4),
        "effective_eoq": round(effective_eoq, 4),
        "cycle_stock": round(effective_eoq / 2.0, 4),
    }


def compute_policy_params(
    policy_type: str,
    ss: float,
    effective_eoq: float,
    avg_daily_demand: float,
    lt_mean_days: float,
    review_cycle_days: int | None,
) -> dict[str, float | bool | None]:
    """Compute policy-specific replenishment trigger parameters.

    Policy logic:
        continuous_rop:
            ROP       = avg_daily * lt_mean + ss
            order_qty = effective_eoq

        min_max  (s, S system):
            s (reorder_point)    = ROP
            S (order_up_to_level) = ROP + effective_eoq

        periodic_review:
            order_up_to = avg_daily * (review_cycle_days + lt_mean) + ss

        manual / jit:
            is_jit = True; all trigger quantities are None

        unknown type  → falls through to continuous_rop behaviour.

    Args:
        policy_type:       One of continuous_rop | min_max | periodic_review | manual | jit.
        ss:                Forward-looking safety stock quantity (units).
        effective_eoq:     Effective EOQ after MOQ + cap constraints (units).
        avg_daily_demand:  Average daily demand (units/day).
        lt_mean_days:      Mean lead time in days.
        review_cycle_days: Review interval for periodic_review policies (days).

    Returns:
        dict with keys: reorder_point, order_qty, order_up_to_level, is_jit.
    """
    if policy_type in ("manual", "jit"):
        return {
            "reorder_point": None,
            "order_qty": None,
            "order_up_to_level": None,
            "is_jit": True,
        }

    rop = avg_daily_demand * lt_mean_days + ss

    if policy_type == "continuous_rop":
        return {
            "reorder_point": round(rop, 4),
            "order_qty": round(effective_eoq, 4),
            "order_up_to_level": None,
            "is_jit": False,
        }

    if policy_type == "min_max":
        return {
            "reorder_point": round(rop, 4),         # s (lower trigger)
            "order_qty": None,
            "order_up_to_level": round(rop + effective_eoq, 4),  # S (upper target)
            "is_jit": False,
        }

    if policy_type == "periodic_review":
        review_days = float(review_cycle_days or 30)
        order_up_to = avg_daily_demand * (review_days + lt_mean_days) + ss
        return {
            "reorder_point": None,
            "order_qty": None,
            "order_up_to_level": round(order_up_to, 4),
            "is_jit": False,
        }

    # Unknown policy type — default to continuous_rop behaviour
    log.warning("Unknown policy_type '%s' — falling back to continuous_rop.", policy_type)
    return {
        "reorder_point": round(rop, 4),
        "order_qty": round(effective_eoq, 4),
        "order_up_to_level": None,
        "is_jit": False,
    }


# ---------------------------------------------------------------------------
# Database loading helpers
# ---------------------------------------------------------------------------


def get_latest_plan_version(conn: psycopg.Connection) -> str | None:
    """Return the most recent plan_version from fact_production_forecast."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT plan_version FROM fact_production_forecast"
            " ORDER BY plan_version DESC LIMIT 1"
        )
        row = cur.fetchone()
    return row[0] if row else None


def load_production_forecasts(
    conn: psycopg.Connection,
    plan_version: str,
    item_id: str | None = None,
    loc: str | None = None,
) -> pd.DataFrame:
    """Load per-DFU-month point forecasts + CI bands for the given plan_version."""
    sql = """
        SELECT
            item_id, loc, plan_version,
            forecast_month, forecast_qty,
            forecast_qty_lower, forecast_qty_upper,
            horizon_months, model_id
        FROM fact_production_forecast
        WHERE plan_version = %s
    """
    params: list[Any] = [plan_version]
    if item_id:
        sql += " AND item_id = %s"
        params.append(item_id)
    if loc:
        sql += " AND loc = %s"
        params.append(loc)
    sql += " ORDER BY item_id, loc, forecast_month"

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    cols = [
        "item_id", "loc", "plan_version", "forecast_month",
        "forecast_qty", "forecast_qty_lower", "forecast_qty_upper",
        "horizon_months", "model_id",
    ]
    return pd.DataFrame(rows, columns=cols)


def load_policy_assignments(conn: psycopg.Connection) -> pd.DataFrame:
    """Load active policy assignments with policy details per DFU."""
    sql = """
        SELECT
            pa.item_id,
            pa.loc,
            pa.policy_id,
            rp.policy_type,
            rp.review_cycle_days
        FROM fact_dfu_policy_assignment pa
        JOIN dim_replenishment_policy rp ON pa.policy_id = rp.policy_id
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    cols = ["item_id", "loc", "policy_id", "policy_type", "review_cycle_days"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def load_dfu_attrs(conn: psycopg.Connection) -> pd.DataFrame:
    """Load DFU attributes: ABC class, demand_std fallback, ml_cluster."""
    sql = """
        SELECT
            item_id                         AS item_id,
            loc,
            abc_vol,
            COALESCE(demand_std, 0.0)       AS demand_std,
            ml_cluster
        FROM dim_sku
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    cols = ["item_id", "loc", "abc_vol", "demand_std", "ml_cluster"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def load_lead_times(conn: psycopg.Connection) -> pd.DataFrame:
    """Load per-DFU lead time statistics from fact_inventory_snapshot.

    Derives lt_mean_days and lt_std_days by aggregating non-zero lead time
    observations from the snapshot table.  Falls back gracefully if the table
    is empty or unavailable — the caller applies config defaults.
    """
    sql = """
        SELECT
            item_id,
            loc,
            AVG(lead_time_days)    AS lt_mean_days,
            STDDEV(lead_time_days) AS lt_std_days
        FROM fact_inventory_snapshot
        WHERE lead_time_days IS NOT NULL
          AND lead_time_days > 0
        GROUP BY item_id, loc
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    except psycopg.Error:
        log.warning("Could not load lead times from fact_inventory_snapshot — config defaults will apply.")
        rows = []

    cols = ["item_id", "loc", "lt_mean_days", "lt_std_days"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def load_historical_ss(conn: psycopg.Connection) -> pd.DataFrame:
    """Load latest historical safety stock targets for delta comparison."""
    sql = """
        SELECT item_id, loc, ss_combined
        FROM fact_safety_stock_targets
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    except psycopg.Error:
        log.warning("fact_safety_stock_targets not available — ss_delta columns will be NULL.")
        rows = []

    cols = ["item_id", "loc", "historical_ss"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def load_inventory_position(conn: psycopg.Connection) -> pd.DataFrame:
    """Load the most recent EOM on-hand quantity per DFU from agg_inventory_monthly."""
    sql = """
        SELECT DISTINCT ON (item_id, loc)
            item_id,
            loc,
            eom_qty_on_hand AS current_qty_on_hand
        FROM agg_inventory_monthly
        ORDER BY item_id, loc, month_start DESC
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    except psycopg.Error:
        log.warning("agg_inventory_monthly not available — current_qty_on_hand will default to 0.")
        rows = []

    cols = ["item_id", "loc", "current_qty_on_hand"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_plan(
    config: dict,
    conn: psycopg.Connection,
    plan_version: str,
    item_id: str | None = None,
    loc: str | None = None,
    dry_run: bool = False,
) -> int:
    """Compute forward-looking replenishment plan rows and upsert to DB.

    Args:
        config:       Full parsed YAML config (top-level key: replenishment_plan).
        conn:         Live psycopg connection (used for reads only — writes open a new conn).
        plan_version: Production forecast plan_version to base the plan on.
        item_id:      Optional: restrict computation to a single item.
        loc:          Optional: restrict computation to a single location.
        dry_run:      If True, compute but skip all DB writes.

    Returns:
        Number of rows computed (or that would have been written in dry-run mode).
    """
    cfg: dict = config["replenishment_plan"]

    # --- Config parameters --------------------------------------------------
    ci_z_score: float = float(cfg.get("ci_z_score", 1.282))  # P10/P90 → 80% CI
    raw_z_table: dict = cfg.get("z_table", {0.95: 1.645})
    z_table: dict[float, float] = {float(k): float(v) for k, v in raw_z_table.items()}
    service_levels: dict[str, float] = cfg.get(
        "service_levels", {"A": 0.98, "B": 0.95, "C": 0.90, "default": 0.95}
    )
    min_ss_days: int = int(cfg.get("min_ss_days", 3))
    max_ss_days: int = int(cfg.get("max_ss_days", 120))
    lt_default: float = float(cfg.get("lt_default_days", 14))
    lt_std_pct: float = float(cfg.get("lt_std_fallback_pct", 0.20))
    eoq_annualization_months: int = int(cfg.get("eoq_annualization_months", 12))
    horizon_months: int = int(cfg.get("horizon_months", 12))
    batch_size: int = int(cfg.get("batch_size", 1000))
    default_policy: str = cfg.get("default_policy_type", "continuous_rop")
    fallback_to_historical: bool = bool(cfg.get("fallback_to_historical", True))

    # --- Load source data ---------------------------------------------------
    with profiled_section("load_source_data"):
        log.info("Loading production forecasts (plan_version=%s)…", plan_version)
        forecasts = load_production_forecasts(conn, plan_version, item_id, loc)
        if forecasts.empty:
            log.warning("No forecast rows found for plan_version=%s — nothing to compute.", plan_version)
            return 0
        log.info("  %d forecast rows loaded.", len(forecasts))

        log.info("Loading policy assignments…")
        policies = load_policy_assignments(conn)
        log.info("  %d policy assignments loaded.", len(policies))

        log.info("Loading DFU attributes…")
        dfu_attrs = load_dfu_attrs(conn)
        log.info("  %d DFU attribute rows loaded.", len(dfu_attrs))

        log.info("Loading lead times from fact_inventory_snapshot…")
        lead_times = load_lead_times(conn)
        log.info("  %d lead time records loaded.", len(lead_times))

        log.info("Loading historical safety stock targets…")
        hist_ss = load_historical_ss(conn)
        log.info("  %d historical SS records loaded.", len(hist_ss))

        log.info("Loading inventory positions from agg_inventory_monthly…")
        inv_pos = load_inventory_position(conn)
        log.info("  %d inventory position records loaded.", len(inv_pos))

    # --- Build lookup dicts for O(1) access ---------------------------------
    with profiled_section("compute_plan_rows"):
        policy_map: dict[tuple, Any] = {
        (r.item_id, r.loc): r for r in policies.itertuples(index=False)
    }
    dfu_map: dict[tuple, Any] = {
        (r.item_id, r.loc): r for r in dfu_attrs.itertuples(index=False)
    }
    lt_map: dict[tuple, Any] = {
        (r.item_id, r.loc): r for r in lead_times.itertuples(index=False)
    }
    hist_ss_map: dict[tuple, float] = {
        (r.item_id, r.loc): float(r.historical_ss)
        for r in hist_ss.itertuples(index=False)
        if r.historical_ss is not None
    }
    inv_map: dict[tuple, float] = {
        (r.item_id, r.loc): float(r.current_qty_on_hand)
        for r in inv_pos.itertuples(index=False)
        if r.current_qty_on_hand is not None
    }

    log.info("Computing replenishment plan rows…")
    rows_to_write: list[dict] = []
    computed_at = datetime.now(timezone.utc)

    for (item, loc_val), grp in forecasts.groupby(["item_id", "loc"], sort=False):
        grp = grp.sort_values("forecast_month").head(horizon_months)

        # -- DFU attributes --------------------------------------------------
        dfu = dfu_map.get((item, loc_val))
        abc: str | None = dfu.abc_vol if dfu is not None else None
        demand_std_hist: float = float(dfu.demand_std) if dfu is not None else 0.0

        # -- Policy ----------------------------------------------------------
        pol = policy_map.get((item, loc_val))
        pol_id: str = pol.policy_id if pol is not None else "unassigned"
        pol_type: str = pol.policy_type if pol is not None else default_policy
        review_days: int | None = int(pol.review_cycle_days) if (pol is not None and pd.notna(pol.review_cycle_days)) else None

        # -- Lead time -------------------------------------------------------
        lt = lt_map.get((item, loc_val))
        lt_mean: float = float(lt.lt_mean_days) if (lt is not None and lt.lt_mean_days) else lt_default
        lt_std: float = float(lt.lt_std_days) if (lt is not None and lt.lt_std_days) else lt_mean * lt_std_pct

        # -- Service level + Z-score -----------------------------------------
        svc: float = float(service_levels.get(abc or "", service_levels.get("default", 0.95)))
        z: float = float(
            min(z_table.items(), key=lambda kv: abs(kv[0] - svc))[1]
        )

        # -- Annualised demand for EOQ (scale up when fewer than 12 months available) --
        n_months_available: int = len(grp)
        sum_forecast: float = float(grp["forecast_qty"].fillna(0).sum())
        if n_months_available > 0 and n_months_available < eoq_annualization_months:
            annual_demand = sum_forecast * (12.0 / n_months_available)
        else:
            annual_demand = sum_forecast

        # -- EOQ (computed once per DFU, constant across months) -------------
        eoq_result = compute_forward_eoq(annual_demand, cfg)

        # -- Historical SS + inventory position (DFU-level) ------------------
        hist_ss_val: float | None = hist_ss_map.get((item, loc_val))
        current_oh: float = inv_map.get((item, loc_val), 0.0)

        # -- Per-month rows ---------------------------------------------------
        for frow in grp.itertuples(index=False):
            fqty: float = float(frow.forecast_qty) if frow.forecast_qty is not None else 0.0
            flower_raw = frow.forecast_qty_lower
            fupper_raw = frow.forecast_qty_upper
            flower: float | None = float(flower_raw) if flower_raw is not None else None
            fupper: float | None = float(fupper_raw) if fupper_raw is not None else None
            horizon_h: int = int(frow.horizon_months) if frow.horizon_months is not None else 1

            # Demand variability from CI bands
            sigma: float | None = compute_sigma_from_ci(flower, fupper, ci_z_score)
            sigma_method: str
            if sigma is not None:
                sigma_method = "ci_spread"
            elif fallback_to_historical and demand_std_hist > 0:
                sigma = float(demand_std_hist)
                sigma_method = "historical_fallback"
            else:
                sigma = 0.0
                sigma_method = "zero"

            avg_daily: float = fqty / DAYS_PER_MONTH if fqty > 0 else 0.0
            sigma_daily: float = (sigma / math.sqrt(DAYS_PER_MONTH)) if (sigma and sigma > 0) else 0.0

            # Safety stock
            if avg_daily > 0 or (sigma and sigma > 0):
                ss_result = compute_forward_ss(z, sigma or 0.0, lt_mean, lt_std, avg_daily)
                ss_combined_raw: float = ss_result["ss_combined"]
                ss_combined: float = apply_ss_guard_rails(
                    ss_combined_raw, avg_daily, min_ss_days, max_ss_days
                )
            else:
                ss_result = {"ss_demand_only": 0.0, "ss_lt_only": 0.0, "ss_combined": 0.0}
                ss_combined = 0.0

            # Policy-specific trigger parameters
            pol_params = compute_policy_params(
                pol_type,
                ss_combined,
                eoq_result["effective_eoq"],
                avg_daily,
                lt_mean,
                review_days,
            )

            # Delta vs historical SS
            ss_delta: float | None = None
            ss_delta_pct: float | None = None
            if hist_ss_val is not None:
                ss_delta = round(ss_combined - hist_ss_val, 4)
                if hist_ss_val != 0.0:
                    ss_delta_pct = round(ss_delta / hist_ss_val * 100.0, 4)

            # Current position vs forward SS
            ss_gap: float | None = round(current_oh - ss_combined, 4)
            is_below_ss: bool = bool(current_oh < ss_combined)

            rows_to_write.append({
                "plan_version":            plan_version,
                "item_id":                 item,
                "loc":                     loc_val,
                "plan_month":              frow.forecast_month,
                "horizon_months":          horizon_h,
                "policy_id":               pol_id,
                "policy_type":             pol_type,
                "abc_vol":                 abc,
                "review_cycle_days":       review_days,
                "forecast_qty":            round(fqty, 4) if fqty else None,
                "forecast_qty_lower":      round(flower, 4) if flower is not None else None,
                "forecast_qty_upper":      round(fupper, 4) if fupper is not None else None,
                "forecast_annual_demand":  round(annual_demand, 4),
                "sigma_demand_monthly":    round(sigma, 4) if sigma else None,
                "sigma_demand_daily":      round(sigma_daily, 4) if sigma_daily else None,
                "avg_daily_demand":        round(avg_daily, 4),
                "sigma_method":            sigma_method,
                "lt_mean_days":            round(lt_mean, 2),
                "lt_std_days":             round(lt_std, 2),
                "service_level_target":    svc,
                "z_score":                 z,
                "ss_demand_only":          ss_result["ss_demand_only"],
                "ss_lt_only":              ss_result["ss_lt_only"],
                "ss_combined":             round(ss_combined, 4),
                "eoq":                     eoq_result["eoq"],
                "effective_eoq":           eoq_result["effective_eoq"],
                "cycle_stock":             eoq_result["cycle_stock"],
                "reorder_point":           pol_params["reorder_point"],
                "order_qty":               pol_params["order_qty"],
                "order_up_to_level":       pol_params["order_up_to_level"],
                "is_jit":                  pol_params["is_jit"],
                "historical_ss":           hist_ss_val,
                "ss_delta":                ss_delta,
                "ss_delta_pct":            ss_delta_pct,
                "current_qty_on_hand":     round(current_oh, 4),
                "ss_gap":                  ss_gap,
                "is_below_ss":             is_below_ss,
                "computed_at":             computed_at,
            })

        log.info("Computed %d plan rows across %d DFUs.", len(rows_to_write), forecasts.groupby(["item_id", "loc"]).ngroups)

    if dry_run:
        log.info("[DRY RUN] Would write %d rows to fact_replenishment_plan.", len(rows_to_write))
        if rows_to_write:
            sample = rows_to_write[0]
            log.info(
                "  Sample: item=%s loc=%s plan_month=%s ss=%.2f eoq=%.2f rop=%s",
                sample["item_id"], sample["loc"], sample["plan_month"],
                sample["ss_combined"], sample["effective_eoq"],
                sample["reorder_point"],
            )
        return len(rows_to_write)

    # --- Upsert in batches --------------------------------------------------
    with profiled_section("write_replenishment_plan"):
        upsert_sql = """
        INSERT INTO fact_replenishment_plan (
            plan_version, item_id, loc, plan_month, horizon_months,
            policy_id, policy_type, abc_vol, review_cycle_days,
            forecast_qty, forecast_qty_lower, forecast_qty_upper,
            forecast_annual_demand,
            sigma_demand_monthly, sigma_demand_daily,
            avg_daily_demand, sigma_method,
            lt_mean_days, lt_std_days,
            service_level_target, z_score,
            ss_demand_only, ss_lt_only, ss_combined,
            eoq, effective_eoq, cycle_stock,
            reorder_point, order_qty, order_up_to_level, is_jit,
            historical_ss, ss_delta, ss_delta_pct,
            current_qty_on_hand, ss_gap, is_below_ss,
            computed_at
        ) VALUES (
            %(plan_version)s, %(item_id)s, %(loc)s, %(plan_month)s, %(horizon_months)s,
            %(policy_id)s, %(policy_type)s, %(abc_vol)s, %(review_cycle_days)s,
            %(forecast_qty)s, %(forecast_qty_lower)s, %(forecast_qty_upper)s,
            %(forecast_annual_demand)s,
            %(sigma_demand_monthly)s, %(sigma_demand_daily)s,
            %(avg_daily_demand)s, %(sigma_method)s,
            %(lt_mean_days)s, %(lt_std_days)s,
            %(service_level_target)s, %(z_score)s,
            %(ss_demand_only)s, %(ss_lt_only)s, %(ss_combined)s,
            %(eoq)s, %(effective_eoq)s, %(cycle_stock)s,
            %(reorder_point)s, %(order_qty)s, %(order_up_to_level)s, %(is_jit)s,
            %(historical_ss)s, %(ss_delta)s, %(ss_delta_pct)s,
            %(current_qty_on_hand)s, %(ss_gap)s, %(is_below_ss)s,
            %(computed_at)s
        )
        ON CONFLICT (plan_version, item_id, loc, plan_month) DO UPDATE SET
            horizon_months        = EXCLUDED.horizon_months,
            policy_id             = EXCLUDED.policy_id,
            policy_type           = EXCLUDED.policy_type,
            abc_vol               = EXCLUDED.abc_vol,
            review_cycle_days     = EXCLUDED.review_cycle_days,
            forecast_qty          = EXCLUDED.forecast_qty,
            forecast_qty_lower    = EXCLUDED.forecast_qty_lower,
            forecast_qty_upper    = EXCLUDED.forecast_qty_upper,
            forecast_annual_demand = EXCLUDED.forecast_annual_demand,
            sigma_demand_monthly  = EXCLUDED.sigma_demand_monthly,
            sigma_demand_daily    = EXCLUDED.sigma_demand_daily,
            avg_daily_demand      = EXCLUDED.avg_daily_demand,
            sigma_method          = EXCLUDED.sigma_method,
            lt_mean_days          = EXCLUDED.lt_mean_days,
            lt_std_days           = EXCLUDED.lt_std_days,
            service_level_target  = EXCLUDED.service_level_target,
            z_score               = EXCLUDED.z_score,
            ss_demand_only        = EXCLUDED.ss_demand_only,
            ss_lt_only            = EXCLUDED.ss_lt_only,
            ss_combined           = EXCLUDED.ss_combined,
            eoq                   = EXCLUDED.eoq,
            effective_eoq         = EXCLUDED.effective_eoq,
            cycle_stock           = EXCLUDED.cycle_stock,
            reorder_point         = EXCLUDED.reorder_point,
            order_qty             = EXCLUDED.order_qty,
            order_up_to_level     = EXCLUDED.order_up_to_level,
            is_jit                = EXCLUDED.is_jit,
            historical_ss         = EXCLUDED.historical_ss,
            ss_delta              = EXCLUDED.ss_delta,
            ss_delta_pct          = EXCLUDED.ss_delta_pct,
            current_qty_on_hand   = EXCLUDED.current_qty_on_hand,
            ss_gap                = EXCLUDED.ss_gap,
            is_below_ss           = EXCLUDED.is_below_ss,
            computed_at           = EXCLUDED.computed_at
    """

        total_written = 0
        with psycopg.connect(**get_db_params()) as write_conn:
            with write_conn.cursor() as cur:
                for i in range(0, len(rows_to_write), batch_size):
                    batch = rows_to_write[i : i + batch_size]
                    cur.executemany(upsert_sql, batch)
                    total_written += len(batch)
                    log.info("  Batch %d–%d written.", i + 1, i + len(batch))
            write_conn.commit()

        log.info("Upserted %d rows into fact_replenishment_plan (plan_version=%s).", total_written, plan_version)
    return total_written


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute forward-looking replenishment plan from production forecast CI bands."
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help=f"Path to replenishment_plan_config.yaml (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--plan-version",
        default=None,
        help="Plan version string (e.g. 2026-02). Defaults to the latest version in fact_production_forecast.",
    )
    parser.add_argument(
        "--item",
        default=None,
        help="Restrict computation to a single item_id (useful for testing).",
    )
    parser.add_argument(
        "--loc",
        default=None,
        help="Restrict computation to a single location (useful for testing).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute plan but do not write to the database.",
    )
    return parser.parse_args()


def run(dry_run: bool = False) -> int:
    """Entry point for profiler and CLI usage."""
    config = _load_config(Path(CONFIG_PATH).stem)

    with psycopg.connect(**get_db_params()) as conn:
        plan_version = get_latest_plan_version(conn)
        if not plan_version:
            log.error(
                "No plan_version found in fact_production_forecast. "
                "Run 'make forecast-generate' first."
            )
            return 0
        log.info("Using plan_version: %s", plan_version)

        n = compute_plan(
            config=config,
            conn=conn,
            plan_version=plan_version,
            dry_run=dry_run,
        )

    log.info("Done. %d rows processed.", n)
    return n


if __name__ == "__main__":
    args = _parse_args()

    config = _load_config(Path(args.config).stem)

    with psycopg.connect(**get_db_params()) as conn:
        plan_version = args.plan_version or get_latest_plan_version(conn)
        if not plan_version:
            log.error(
                "No plan_version found in fact_production_forecast. "
                "Run 'make forecast-generate' first."
            )
            sys.exit(1)
        log.info("Using plan_version: %s", plan_version)

        n = compute_plan(
            config=config,
            conn=conn,
            plan_version=plan_version,
            item_id=args.item,
            loc=args.loc,
            dry_run=args.dry_run,
        )

    log.info("Done. %d rows processed.", n)
