"""Safety Stock Engine — IPfeature3.

Computes per-SKU-location safety stock targets using the combined
demand + lead time variability formula and upserts results into
fact_safety_stock_targets.

Formula:
    sigma_D_daily  = demand_std_monthly / sqrt(30.44)
    D_avg_daily    = demand_mean_monthly / 30.44

    SS_demand  = Z * sqrt(LT_mean_days * sigma_D_daily^2)
    SS_lt      = Z * D_avg_daily * lt_std_days
    SS_combined = Z * sqrt(LT_mean_days * sigma_D_daily^2 + D_avg_daily^2 * lt_std_days^2)

    ROP = D_avg_daily * LT_mean_days + SS_combined

Guard rails:
    if SS_combined < min_ss_days * D_avg_daily  →  SS_combined = min_ss_days * D_avg_daily
    if SS_combined > max_ss_days * D_avg_daily  →  SS_combined = max_ss_days * D_avg_daily

Usage:
    uv run python scripts/compute_safety_stock.py
    uv run python scripts/compute_safety_stock.py --dry-run
    uv run python scripts/compute_safety_stock.py --policy-version v2
    uv run python scripts/compute_safety_stock.py --config config/safety_stock_config.yaml
"""
from __future__ import annotations

import argparse
import datetime
import logging
import math
import sys
from pathlib import Path
from typing import Any

import psycopg
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.db import get_db_params

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
DAYS_PER_MONTH: float = 30.44


# ---------------------------------------------------------------------------
# Pure formula functions (unit-testable, no DB dependencies)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Individual SS formula functions (expected by unit tests in test_safety_stock.py)
# ---------------------------------------------------------------------------

def compute_ss_demand(
    z_score: float,
    sigma_demand: float,
    lt_mean_days: float,
) -> float:
    """Demand variability component of safety stock.

    SS_demand = Z * sqrt(LT_mean_days * sigma_demand^2)

    Args:
        z_score:       Z-score for the target service level.
        sigma_demand:  Daily demand standard deviation (units/day).
        lt_mean_days:  Mean lead time in days.

    Returns:
        SS_demand in units.
    """
    variance_term = lt_mean_days * (sigma_demand ** 2)
    return z_score * math.sqrt(variance_term) if variance_term > 0 else 0.0


def compute_ss_lt(
    z_score: float,
    avg_daily_demand: float,
    lt_std_days: float,
) -> float:
    """Lead time variability component of safety stock.

    SS_lt = Z * avg_daily_demand * lt_std_days

    Args:
        z_score:           Z-score for the target service level.
        avg_daily_demand:  Mean daily demand (units/day).
        lt_std_days:       Standard deviation of lead time in days.

    Returns:
        SS_lt in units.
    """
    return z_score * avg_daily_demand * lt_std_days


def compute_ss_combined(
    z_score: float,
    sigma_demand: float,
    lt_mean_days: float,
    avg_daily_demand: float,
    lt_std_days: float,
) -> float:
    """Combined (uncorrelated) safety stock using the full formula.

    SS_combined = Z * sqrt(LT_mean * sigma_D^2 + D_avg^2 * lt_std^2)

    Args:
        z_score:           Z-score for the target service level.
        sigma_demand:      Daily demand standard deviation (units/day).
        lt_mean_days:      Mean lead time in days.
        avg_daily_demand:  Mean daily demand (units/day).
        lt_std_days:       Standard deviation of lead time in days.

    Returns:
        SS_combined in units.
    """
    demand_var = lt_mean_days * (sigma_demand ** 2)
    lt_var = (avg_daily_demand ** 2) * (lt_std_days ** 2)
    return z_score * math.sqrt(demand_var + lt_var)


def compute_reorder_point(
    ss_combined: float,
    avg_daily_demand: float,
    lt_mean_days: float,
) -> float:
    """Reorder point = cycle demand during lead time + safety stock.

    ROP = avg_daily_demand * lt_mean_days + ss_combined

    Args:
        ss_combined:       Safety stock quantity.
        avg_daily_demand:  Mean daily demand (units/day).
        lt_mean_days:      Mean lead time in days.

    Returns:
        Reorder point in units.
    """
    return avg_daily_demand * lt_mean_days + ss_combined


def compute_ss_coverage(
    current_on_hand: float,
    ss_combined: float,
) -> float | None:
    """Safety stock coverage ratio.

    ss_coverage = current_on_hand / ss_combined

    Returns None when ss_combined = 0 (no safety stock required).
    """
    if ss_combined == 0.0:
        return None
    return current_on_hand / ss_combined


def get_z_score(service_level: float, z_table: dict[float, float]) -> float:
    """Look up the Z-score for a given service level from the config z_table.

    Tries an exact match first, then the closest key.
    Falls back to 1.645 (95%) if nothing is found.
    """
    # Exact match
    if service_level in z_table:
        return float(z_table[service_level])
    # Closest key
    if z_table:
        closest = min(z_table.keys(), key=lambda k: abs(k - service_level))
        return float(z_table[closest])
    return 1.645  # hard fallback


def get_service_level(abc_vol: str | None, service_levels: dict[str, float]) -> float:
    """Return configured service level for a given ABC class.

    Falls back to 'default' key when abc_vol is None or unrecognized.
    """
    if abc_vol and abc_vol.upper() in service_levels:
        return float(service_levels[abc_vol.upper()])
    return float(service_levels.get("default", 0.95))


def compute_ss_components(
    z: float,
    demand_mean_monthly: float,
    demand_std_monthly: float,
    lt_mean_days: float,
    lt_std_days: float,
) -> dict[str, float | None]:
    """Compute safety stock components.

    Args:
        z:                    Z-score for the target service level.
        demand_mean_monthly:  Mean monthly demand (units).
        demand_std_monthly:   Std dev of monthly demand (units).
        lt_mean_days:         Mean lead time in days.
        lt_std_days:          Std dev of lead time in days.

    Returns dict with:
        avg_daily_demand, sigma_d_daily,
        ss_demand_only, ss_lt_only, ss_combined, ss_method
    """
    avg_daily = demand_mean_monthly / DAYS_PER_MONTH if DAYS_PER_MONTH else 0.0
    sigma_d_daily = demand_std_monthly / math.sqrt(DAYS_PER_MONTH) if demand_std_monthly else 0.0

    demand_variance_term = lt_mean_days * (sigma_d_daily ** 2)
    lt_variance_term = (avg_daily ** 2) * (lt_std_days ** 2)

    ss_demand = z * math.sqrt(demand_variance_term) if demand_variance_term >= 0 else 0.0
    ss_lt = z * avg_daily * lt_std_days if lt_std_days else 0.0
    ss_combined = z * math.sqrt(demand_variance_term + lt_variance_term)

    # Determine method label
    if demand_mean_monthly == 0.0 and demand_std_monthly == 0.0:
        ss_method = "demand_only"  # zero-demand; formula returns 0
    elif lt_std_days == 0.0:
        ss_method = "demand_only"
    else:
        ss_method = "combined"

    return {
        "avg_daily_demand": avg_daily,
        "sigma_d_daily": sigma_d_daily,
        "ss_demand_only": ss_demand,
        "ss_lt_only": ss_lt,
        "ss_combined": ss_combined,
        "ss_method": ss_method,
    }


def apply_guard_rails(
    ss_combined: float = 0.0,
    avg_daily_demand: float = 0.0,
    min_ss_days: float = 3.0,
    max_ss_days: float = 120.0,
    # Aliases used by unit tests (ss_days, min_days, max_days)
    ss_days: float | None = None,
    min_days: float | None = None,
    max_days: float | None = None,
) -> float:
    """Clamp safety stock between min_ss_days and max_ss_days of supply.

    Can be called either with qty (ss_combined) or days (ss_days).
    When ss_days is provided it is converted to qty via ss_days * avg_daily_demand.
    The return value is always a quantity (units).

    For zero-demand items, guard rails produce 0 (no stock needed).
    """
    # Resolve aliases
    _min = min_days if min_days is not None else min_ss_days
    _max = max_days if max_days is not None else max_ss_days
    _ss_qty: float
    if ss_days is not None:
        _ss_qty = ss_days * avg_daily_demand
    else:
        _ss_qty = ss_combined

    if avg_daily_demand <= 0.0:
        return _ss_qty  # guard rails only meaningful when demand > 0
    min_qty = _min * avg_daily_demand
    max_qty = _max * avg_daily_demand
    return max(min_qty, min(max_qty, _ss_qty))


def compute_position_metrics(
    ss_combined: float,
    avg_daily_demand: float,
    lt_mean_days: float,
    current_qty_on_hand: float,
) -> dict[str, float | bool | None]:
    """Derive ROP, coverage, gap, and is_below_ss from SS output.

    Args:
        ss_combined:         Recommended safety stock quantity.
        avg_daily_demand:    Average daily demand (units/day).
        lt_mean_days:        Mean lead time in days.
        current_qty_on_hand: Latest on-hand quantity.

    Returns dict with reorder_point, target_dos_min, ss_coverage, ss_gap, is_below_ss.
    """
    reorder_point = avg_daily_demand * lt_mean_days + ss_combined

    target_dos_min: float | None
    if avg_daily_demand > 0:
        target_dos_min = ss_combined / avg_daily_demand
        current_dos: float | None = current_qty_on_hand / avg_daily_demand
    else:
        target_dos_min = None
        current_dos = None

    ss_coverage: float | None
    if ss_combined > 0:
        ss_coverage = current_qty_on_hand / ss_combined
    else:
        ss_coverage = None  # no SS required; avoid divide-by-zero

    ss_gap = current_qty_on_hand - ss_combined
    is_below_ss = current_qty_on_hand < ss_combined

    return {
        "reorder_point": reorder_point,
        "target_dos_min": target_dos_min,
        "current_dos": current_dos,
        "ss_coverage": ss_coverage,
        "ss_gap": ss_gap,
        "is_below_ss": is_below_ss,
    }


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _load_dfu_data(cur) -> list[dict]:
    """Load DFU demand stats and ABC classification from dim_dfu."""
    sql = """
        SELECT
            dmdunit                     AS item_no,
            loc,
            abc_vol,
            demand_mean,
            demand_std,
            demand_cv
        FROM dim_dfu
        WHERE demand_mean IS NOT NULL
           OR demand_std  IS NOT NULL
        ORDER BY dmdunit, loc
    """
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _load_lead_time_data(cur) -> dict[tuple[str, str], dict]:
    """Load lead time profiles from dim_item_lead_time_profile (IPfeature2).

    Returns dict keyed by (item_no, loc).
    Falls back to empty dict if table does not exist — script applies defaults.
    """
    try:
        sql = """
            SELECT
                item_no,
                loc,
                lt_mean_days,
                lt_std_days
            FROM dim_item_lead_time_profile
        """
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        result: dict[tuple[str, str], dict] = {}
        for row in cur.fetchall():
            rec = dict(zip(cols, row))
            result[(rec["item_no"], rec["loc"])] = rec
        return result
    except Exception:
        log.warning("dim_item_lead_time_profile not found — using default LT values")
        return {}


def _load_inventory_data(cur) -> dict[tuple[str, str], float]:
    """Load latest EOM on-hand per item_no + loc from agg_inventory_monthly."""
    try:
        sql = """
            SELECT DISTINCT ON (item_no, loc)
                item_no,
                loc,
                eom_qty_on_hand
            FROM agg_inventory_monthly
            ORDER BY item_no, loc, month_start DESC
        """
        cur.execute(sql)
        result: dict[tuple[str, str], float] = {}
        for row in cur.fetchall():
            item_no, loc, qty = row[0], row[1], row[2]
            result[(item_no, loc)] = float(qty) if qty is not None else 0.0
        return result
    except Exception:
        log.warning("agg_inventory_monthly not found — current_qty_on_hand defaults to 0")
        return {}


# ---------------------------------------------------------------------------
# Batch upsert
# ---------------------------------------------------------------------------

UPSERT_SQL = """
    INSERT INTO fact_safety_stock_targets (
        ss_ck,
        item_no, loc, policy_version, effective_date,
        service_level_target, z_score,
        demand_mean_monthly, demand_std_monthly,
        lead_time_mean_days, lead_time_std_days,
        abc_vol,
        ss_demand_only, ss_lt_only, ss_combined, ss_method,
        avg_daily_demand, demand_cv,
        lt_mean_days, lt_std_days,
        reorder_point, target_min_qty, target_dos_min,
        current_qty_on_hand, current_dos,
        ss_coverage, ss_gap, is_below_ss,
        computed_at, load_ts, modified_ts
    ) VALUES (
        %s,
        %s, %s, %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
        %s,
        %s, %s, %s, %s,
        %s, %s,
        %s, %s,
        %s, %s, %s,
        %s, %s,
        %s, %s, %s,
        %s, %s, %s
    )
    ON CONFLICT (item_no, loc, policy_version) DO UPDATE SET
        ss_ck                = EXCLUDED.ss_ck,
        effective_date       = EXCLUDED.effective_date,
        service_level_target = EXCLUDED.service_level_target,
        z_score              = EXCLUDED.z_score,
        demand_mean_monthly  = EXCLUDED.demand_mean_monthly,
        demand_std_monthly   = EXCLUDED.demand_std_monthly,
        lead_time_mean_days  = EXCLUDED.lead_time_mean_days,
        lead_time_std_days   = EXCLUDED.lead_time_std_days,
        abc_vol              = EXCLUDED.abc_vol,
        ss_demand_only       = EXCLUDED.ss_demand_only,
        ss_lt_only           = EXCLUDED.ss_lt_only,
        ss_combined          = EXCLUDED.ss_combined,
        ss_method            = EXCLUDED.ss_method,
        avg_daily_demand     = EXCLUDED.avg_daily_demand,
        demand_cv            = EXCLUDED.demand_cv,
        lt_mean_days         = EXCLUDED.lt_mean_days,
        lt_std_days          = EXCLUDED.lt_std_days,
        reorder_point        = EXCLUDED.reorder_point,
        target_min_qty       = EXCLUDED.target_min_qty,
        target_dos_min       = EXCLUDED.target_dos_min,
        current_qty_on_hand  = EXCLUDED.current_qty_on_hand,
        current_dos          = EXCLUDED.current_dos,
        ss_coverage          = EXCLUDED.ss_coverage,
        ss_gap               = EXCLUDED.ss_gap,
        is_below_ss          = EXCLUDED.is_below_ss,
        computed_at          = EXCLUDED.computed_at,
        modified_ts          = EXCLUDED.modified_ts
"""


def _batch_upsert(cur, rows: list[tuple], batch_size: int) -> int:
    """Insert rows in batches using executemany. Returns count of rows processed."""
    total = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        cur.executemany(UPSERT_SQL, chunk)
        total += len(chunk)
    return total


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def run(
    config_path: str = "config/safety_stock_config.yaml",
    policy_version: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Compute safety stock targets for all DFUs and upsert into the DB.

    Args:
        config_path:    Path to safety_stock_config.yaml.
        policy_version: Override the policy_version from config.
        dry_run:        If True, compute but do not write to DB.

    Returns:
        dict with inserted_count, skipped_count, zero_demand_count.
    """
    # -- Load config ----------------------------------------------------------
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    ss_cfg = cfg["safety_stock"]

    pv = policy_version or ss_cfg.get("policy_version", "v1")
    service_levels: dict[str, float] = ss_cfg["service_levels"]
    z_table: dict[float, float] = {float(k): float(v) for k, v in ss_cfg["z_table"].items()}
    min_ss_days: float = float(ss_cfg.get("min_ss_days", 3))
    max_ss_days: float = float(ss_cfg.get("max_ss_days", 120))
    lt_std_fallback_pct: float = float(ss_cfg.get("lt_std_fallback_pct", 0.20))
    batch_size: int = int(ss_cfg.get("batch_size", 1000))

    log.info("Safety Stock Engine — IPfeature3 (policy_version=%s, dry_run=%s)", pv, dry_run)

    # -- Query source data ----------------------------------------------------
    conn = psycopg.connect(**get_db_params())
    conn.autocommit = False

    with conn.cursor() as cur:
        log.info("Loading DFU demand data from dim_dfu …")
        dfu_rows = _load_dfu_data(cur)
        log.info("  %d DFUs loaded", len(dfu_rows))

        log.info("Loading lead time profiles from dim_item_lead_time_profile …")
        lt_map = _load_lead_time_data(cur)
        log.info("  %d LT profiles loaded", len(lt_map))

        log.info("Loading latest on-hand from agg_inventory_monthly …")
        inv_map = _load_inventory_data(cur)
        log.info("  %d inventory records loaded", len(inv_map))

    # -- Compute SS per DFU --------------------------------------------------
    now = datetime.datetime.now(datetime.timezone.utc)
    today = now.date()

    upsert_rows: list[tuple] = []
    skipped_lt_zero = 0
    zero_demand_count = 0
    skipped_count = 0

    for dfu in dfu_rows:
        item_no: str = dfu["item_no"]
        loc: str = dfu["loc"]
        abc_vol: str | None = dfu.get("abc_vol")
        demand_mean: float = float(dfu.get("demand_mean") or 0.0)
        demand_std: float = float(dfu.get("demand_std") or 0.0)
        demand_cv_raw = dfu.get("demand_cv")
        demand_cv: float | None = float(demand_cv_raw) if demand_cv_raw is not None else None

        # -- Lead time from profile table or fallback -----------------------
        lt_rec = lt_map.get((item_no, loc), {})
        lt_mean_raw = lt_rec.get("lt_mean_days")
        lt_std_raw = lt_rec.get("lt_std_days")

        if lt_mean_raw is None:
            lt_mean_days = 14.0          # no profile → assume 14-day lead time
        else:
            lt_mean_days = float(lt_mean_raw)

        if lt_mean_days <= 0:
            log.warning("Skipping %s/%s: lt_mean_days=0 (invalid)", item_no, loc)
            skipped_lt_zero += 1
            skipped_count += 1
            continue

        if lt_std_raw is None:
            lt_std_days = lt_mean_days * lt_std_fallback_pct
        else:
            lt_std_days = float(lt_std_raw)

        # -- Service level & Z-score ----------------------------------------
        sl = get_service_level(abc_vol, service_levels)
        z = get_z_score(sl, z_table)

        # -- SS formula -------------------------------------------------------
        result = compute_ss_components(
            z=z,
            demand_mean_monthly=demand_mean,
            demand_std_monthly=demand_std,
            lt_mean_days=lt_mean_days,
            lt_std_days=lt_std_days,
        )

        avg_daily: float = result["avg_daily_demand"]
        ss_combined: float = result["ss_combined"]
        ss_method: str = result["ss_method"]

        if demand_mean == 0.0 and demand_std == 0.0:
            zero_demand_count += 1

        # -- Guard rails ------------------------------------------------------
        ss_combined = apply_guard_rails(ss_combined, avg_daily, min_ss_days, max_ss_days)

        # -- Current inventory position ---------------------------------------
        current_qty = inv_map.get((item_no, loc), 0.0)

        pos = compute_position_metrics(
            ss_combined=ss_combined,
            avg_daily_demand=avg_daily,
            lt_mean_days=lt_mean_days,
            current_qty_on_hand=current_qty,
        )

        # -- Build composite key & row tuple ----------------------------------
        ss_ck = f"{item_no}_{loc}_{pv}"

        row: tuple = (
            ss_ck,
            item_no, loc, pv, today,
            sl, z,
            demand_mean, demand_std,
            lt_mean_days, lt_std_days,
            abc_vol,
            result["ss_demand_only"],
            result["ss_lt_only"],
            ss_combined,
            ss_method,
            avg_daily,
            demand_cv,
            lt_mean_days,
            lt_std_days,
            pos["reorder_point"],
            ss_combined,             # target_min_qty = ss_combined
            pos["target_dos_min"],
            current_qty,
            pos["current_dos"],
            pos["ss_coverage"],
            pos["ss_gap"],
            pos["is_below_ss"],
            now, now, now,           # computed_at, load_ts, modified_ts
        )
        upsert_rows.append(row)

    log.info(
        "Computation complete: %d rows to upsert, %d zero-demand, %d skipped",
        len(upsert_rows),
        zero_demand_count,
        skipped_count,
    )

    if dry_run:
        log.info("[DRY RUN] No data written to DB.")
        # Print a sample
        if upsert_rows:
            sample = upsert_rows[0]
            log.info(
                "Sample row: item=%s loc=%s ss_combined=%.2f rop=%.2f is_below=%s",
                sample[1], sample[2], sample[14], sample[20], sample[27],
            )
        conn.close()
        return {
            "inserted_count": 0,
            "skipped_count": skipped_count,
            "zero_demand_count": zero_demand_count,
            "dry_run": True,
        }

    # -- Upsert into DB -------------------------------------------------------
    with conn.cursor() as cur:
        inserted = _batch_upsert(cur, upsert_rows, batch_size)

    conn.commit()
    conn.close()

    log.info("Upserted %d rows into fact_safety_stock_targets (policy_version=%s)", inserted, pv)

    return {
        "inserted_count": inserted,
        "skipped_count": skipped_count,
        "zero_demand_count": zero_demand_count,
        "dry_run": False,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-SKU-location safety stock targets (IPfeature3)"
    )
    parser.add_argument(
        "--config",
        default="config/safety_stock_config.yaml",
        help="Path to safety_stock_config.yaml (default: config/safety_stock_config.yaml)",
    )
    parser.add_argument(
        "--policy-version",
        default=None,
        help="Override policy_version from config (e.g. v2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute SS but do not write to DB",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run(
        config_path=args.config,
        policy_version=args.policy_version,
        dry_run=args.dry_run,
    )
    log.info("Result: %s", result)
    sys.exit(0)
