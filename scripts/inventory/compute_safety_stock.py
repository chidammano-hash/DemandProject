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

Guard rails (ABC-specific):
    Bounds are differentiated by ABC class (A=tighter, C=wider).
    if SS_combined < min_ss_days[abc] * D_avg_daily  →  SS_combined = min_ss * D_avg_daily
    if SS_combined > max_ss_days[abc] * D_avg_daily  →  SS_combined = max_ss * D_avg_daily
    Zero-demand items get an absolute floor (zero_demand_min_units).

Outlier detection:
    MAD (Median Absolute Deviation) flags demand months with extreme values.
    Items with >20% outlier months are marked volatile and get a service level boost.

Usage:
    uv run python scripts/compute_safety_stock.py
    uv run python scripts/compute_safety_stock.py --dry-run
    uv run python scripts/compute_safety_stock.py --policy-version v2
    uv run python scripts/compute_safety_stock.py --config config/inventory/safety_stock_config.yaml
    uv run python scripts/compute_safety_stock.py --forecast-source production
    uv run python scripts/compute_safety_stock.py --forecast-source staging --model-id lgbm_cluster
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import psycopg

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.core.db import get_db_params
from common.core.planning_date import get_planning_date
from common.inventory.safety_stock import (
    apply_guard_rails,
    apply_seasonal_adjustment,
    compute_position_metrics,
    compute_seasonal_factors,
    compute_ss_components,
    detect_outliers,
    get_service_level,
    get_z_score,
)
from common.scripts_base import setup_logging
from common.services.perf_profiler import profiled_section
from common.core.utils import load_config as _load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging()
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _load_dfu_data(cur) -> list[dict]:
    """Load DFU demand stats, ABC classification, and intermittency from dim_sku."""
    sql = """
        SELECT
            item_id                     AS item_id,
            loc,
            abc_vol,
            demand_mean,
            demand_std,
            demand_cv,
            intermittency_ratio
        FROM dim_sku
        WHERE demand_mean IS NOT NULL
           OR demand_std  IS NOT NULL
        ORDER BY item_id, loc
    """
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _load_lead_time_data(cur) -> dict[tuple[str, str], dict]:
    """Load lead time profiles from dim_item_lead_time_profile (IPfeature2).

    Returns dict keyed by (item_id, loc).
    Falls back to empty dict if table does not exist — script applies defaults.
    """
    try:
        cur.execute("SAVEPOINT lt_load")
        sql = """
            SELECT
                item_id,
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
            result[(rec["item_id"], rec["loc"])] = rec
        return result
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT lt_load")
        log.warning("dim_item_lead_time_profile not found — using default LT values")
        return {}


def _load_inventory_data(cur) -> dict[tuple[str, str], float]:
    """Load latest EOM on-hand per item_id + loc from agg_inventory_monthly."""
    try:
        cur.execute("SAVEPOINT inv_load")
        sql = """
            SELECT DISTINCT ON (item_id, loc)
                item_id,
                loc,
                eom_qty_on_hand
            FROM agg_inventory_monthly
            ORDER BY item_id, loc, month_start DESC
        """
        cur.execute(sql)
        result: dict[tuple[str, str], float] = {}
        for row in cur.fetchall():
            item_id, loc, qty = row[0], row[1], row[2]
            result[(item_id, loc)] = float(qty) if qty is not None else 0.0
        return result
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT inv_load")
        log.warning("agg_inventory_monthly not found — current_qty_on_hand defaults to 0")
        return {}


def _load_demand_history(cur) -> dict[tuple[str, str], list[float]]:
    """Load monthly demand history per (item_id, loc) from agg_sales_monthly.

    Returns dict keyed by (item_id, loc) with a list of monthly qty values
    ordered by month_start (ascending). Used for MAD-based outlier detection.
    Falls back to empty dict if table does not exist.
    """
    try:
        cur.execute("SAVEPOINT demand_hist_load")
        sql = """
            SELECT item_id, loc, qty
            FROM agg_sales_monthly
            ORDER BY item_id, loc, month_start
        """
        cur.execute(sql)
        result: dict[tuple[str, str], list[float]] = {}
        for row in cur.fetchall():
            item_id, loc, qty = str(row[0]), str(row[1]), row[2]
            val = float(qty) if qty is not None else 0.0
            key = (item_id, loc)
            if key not in result:
                result[key] = []
            result[key].append(val)
        return result
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT demand_hist_load")
        log.warning("agg_sales_monthly not found — outlier detection disabled")
        return {}


def _load_demand_history_with_dates(
    cur,
) -> dict[tuple[str, str], list[tuple[date, float]]]:
    """Load monthly demand history with dates per (item_id, loc) from agg_sales_monthly.

    Returns dict keyed by (item_id, loc) with a list of (month_start_date, qty) tuples
    ordered by month_start (ascending). Used for computing seasonal demand factors.
    Falls back to empty dict if table does not exist.
    """
    try:
        cur.execute("SAVEPOINT demand_hist_dates_load")
        sql = """
            SELECT item_id, loc, month_start, qty
            FROM agg_sales_monthly
            ORDER BY item_id, loc, month_start
        """
        cur.execute(sql)
        result: dict[tuple[str, str], list[tuple[date, float]]] = {}
        for row in cur.fetchall():
            item_id, loc, month_start, qty = str(row[0]), str(row[1]), row[2], row[3]
            val = float(qty) if qty is not None else 0.0
            key = (item_id, loc)
            if key not in result:
                result[key] = []
            # month_start may be date or datetime — ensure it's a date
            if isinstance(month_start, datetime.datetime):
                month_start = month_start.date()
            result[key].append((month_start, val))
        return result
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT demand_hist_dates_load")
        log.warning("agg_sales_monthly not found — seasonal adjustment disabled")
        return {}


def _load_forecast_demand_stats(
    cur,
    source: str,
    model_id: str | None,
    ci_z: float = 1.282,
) -> dict[tuple[str, str], dict[str, float]]:
    """Load demand mean/std from forecast CI bands.

    For each (item_id, loc):
      demand_mean = AVG(forecast_qty) across all horizon months
      demand_std  = AVG((forecast_qty_upper - forecast_qty_lower) / (2 * ci_z))

    When CI bands are NULL for a DFU, that DFU is excluded (caller falls back
    to historical stats from dim_sku).

    Source behavior:
      - source='production': reads from fact_production_forecast using the
        latest plan_version (all rows, no model_id filter).
      - source='staging': reads from fact_production_forecast_staging filtered
        by the given model_id. Requires model_id to be set.

    Args:
        cur:       Database cursor.
        source:    'production' (fact_production_forecast) or
                   'staging' (fact_production_forecast_staging).
        model_id:  Required when source='staging'; filters by model_id.
        ci_z:      Z-score assumed for the CI bands (default 1.282 for P10/P90).

    Returns:
        dict keyed by (item_id, loc) with keys demand_mean, demand_std, demand_cv.
        Returns empty dict if the target table does not exist or the query fails.
    """
    if source == "production":
        sql = """
            WITH active_promotion AS (
                SELECT id
                FROM model_promotion_log
                WHERE is_active = TRUE
            )
            SELECT forecast.item_id,
                   forecast.loc,
                   AVG(forecast.forecast_qty) AS demand_mean,
                   AVG(
                       CASE
                           WHEN forecast.forecast_qty_upper IS NOT NULL
                            AND forecast.forecast_qty_lower IS NOT NULL
                            AND %s > 0
                           THEN (forecast.forecast_qty_upper - forecast.forecast_qty_lower)
                                / (2.0 * %s)
                           ELSE NULL
                       END
                   ) AS demand_std
            FROM fact_production_forecast forecast
            JOIN active_promotion
              ON active_promotion.id = forecast.promotion_log_id
            GROUP BY forecast.item_id, forecast.loc
            HAVING AVG(forecast.forecast_qty) IS NOT NULL
        """
        query_params = (ci_z, ci_z)
    elif source == "staging":
        if not model_id:
            log.warning("--forecast-source=staging requires --model-id; falling back to historical")
            return {}
        sql = """
            WITH selected_run AS (
                SELECT run_id
                FROM forecast_generation_run
                WHERE generation_purpose = 'release_candidate'
                  AND run_status IN ('ready', 'promoted')
                  AND requested_model_id = %s
                ORDER BY completed_at DESC NULLS LAST, created_at DESC, run_id
                LIMIT 1
            )
            SELECT staging.item_id,
                   staging.loc,
                   AVG(staging.forecast_qty) AS demand_mean,
                   AVG(
                       CASE
                           WHEN staging.forecast_qty_upper IS NOT NULL
                            AND staging.forecast_qty_lower IS NOT NULL
                            AND %s > 0
                           THEN (staging.forecast_qty_upper - staging.forecast_qty_lower)
                                / (2.0 * %s)
                           ELSE NULL
                       END
                   ) AS demand_std
            FROM fact_production_forecast_staging staging
            JOIN selected_run ON selected_run.run_id = staging.run_id
            WHERE staging.generation_purpose = 'release_candidate'
              AND staging.candidate_model_id = %s
            GROUP BY staging.item_id, staging.loc
            HAVING AVG(staging.forecast_qty) IS NOT NULL
        """
        query_params = (model_id, ci_z, ci_z, model_id)
    else:
        return {}

    try:
        cur.execute("SAVEPOINT fcst_load")
        cur.execute(sql, query_params)
        result: dict[tuple[str, str], dict[str, float]] = {}
        for row in cur.fetchall():
            item_id, loc, mean_val, std_val = row
            mean_f = float(mean_val) if mean_val is not None else 0.0
            std_f = float(std_val) if std_val is not None else 0.0
            cv_f = (std_f / mean_f) if mean_f > 0 else 0.0
            # Skip DFUs with NULL std (CI bands were NULL for all rows)
            if std_val is None:
                continue
            result[(str(item_id), str(loc))] = {
                "demand_mean": mean_f,
                "demand_std": std_f,
                "demand_cv": cv_f,
            }
        return result
    except psycopg.Error:
        cur.execute("ROLLBACK TO SAVEPOINT fcst_load")
        log.warning(
            "%s not found or query failed — falling back to historical demand stats",
            source,
        )
        return {}


# ---------------------------------------------------------------------------
# Batch upsert
# ---------------------------------------------------------------------------

UPSERT_SQL = """
    INSERT INTO fact_safety_stock_targets (
        ss_ck,
        item_id, loc, policy_version, effective_date,
        service_level_target, z_score,
        demand_mean_monthly, demand_std_monthly,
        lead_time_mean_days, lead_time_std_days,
        abc_vol, xyz_class, abc_xyz_segment,
        ss_demand_only, ss_lt_only, ss_combined, ss_method,
        avg_daily_demand, demand_cv,
        lt_mean_days, lt_std_days,
        reorder_point, target_min_qty, target_dos_min,
        current_qty_on_hand, current_dos,
        ss_coverage, ss_gap, is_below_ss,
        forecast_source, forecast_model_id,
        has_demand_outliers, outlier_pct,
        guard_rail_applied, guard_rail_min, guard_rail_max,
        seasonal_factor, ss_seasonal, is_seasonal_adjusted,
        sl_adjustment_reason,
        computed_at, load_ts, modified_ts
    ) VALUES (
        %s,
        %s, %s, %s, %s,
        %s, %s,
        %s, %s,
        %s, %s,
        %s, %s, %s,
        %s, %s, %s, %s,
        %s, %s,
        %s, %s,
        %s, %s, %s,
        %s, %s,
        %s, %s, %s,
        %s, %s,
        %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s,
        %s, %s, %s
    )
    ON CONFLICT (item_id, loc, policy_version) DO UPDATE SET
        ss_ck                  = EXCLUDED.ss_ck,
        effective_date         = EXCLUDED.effective_date,
        service_level_target   = EXCLUDED.service_level_target,
        z_score                = EXCLUDED.z_score,
        demand_mean_monthly    = EXCLUDED.demand_mean_monthly,
        demand_std_monthly     = EXCLUDED.demand_std_monthly,
        lead_time_mean_days    = EXCLUDED.lead_time_mean_days,
        lead_time_std_days     = EXCLUDED.lead_time_std_days,
        abc_vol                = EXCLUDED.abc_vol,
        xyz_class              = EXCLUDED.xyz_class,
        abc_xyz_segment        = EXCLUDED.abc_xyz_segment,
        ss_demand_only         = EXCLUDED.ss_demand_only,
        ss_lt_only             = EXCLUDED.ss_lt_only,
        ss_combined            = EXCLUDED.ss_combined,
        ss_method              = EXCLUDED.ss_method,
        avg_daily_demand       = EXCLUDED.avg_daily_demand,
        demand_cv              = EXCLUDED.demand_cv,
        lt_mean_days           = EXCLUDED.lt_mean_days,
        lt_std_days            = EXCLUDED.lt_std_days,
        reorder_point          = EXCLUDED.reorder_point,
        target_min_qty         = EXCLUDED.target_min_qty,
        target_dos_min         = EXCLUDED.target_dos_min,
        current_qty_on_hand    = EXCLUDED.current_qty_on_hand,
        current_dos            = EXCLUDED.current_dos,
        ss_coverage            = EXCLUDED.ss_coverage,
        ss_gap                 = EXCLUDED.ss_gap,
        is_below_ss            = EXCLUDED.is_below_ss,
        forecast_source        = EXCLUDED.forecast_source,
        forecast_model_id      = EXCLUDED.forecast_model_id,
        has_demand_outliers    = EXCLUDED.has_demand_outliers,
        outlier_pct            = EXCLUDED.outlier_pct,
        guard_rail_applied     = EXCLUDED.guard_rail_applied,
        guard_rail_min         = EXCLUDED.guard_rail_min,
        guard_rail_max         = EXCLUDED.guard_rail_max,
        seasonal_factor        = EXCLUDED.seasonal_factor,
        ss_seasonal            = EXCLUDED.ss_seasonal,
        is_seasonal_adjusted   = EXCLUDED.is_seasonal_adjusted,
        sl_adjustment_reason   = EXCLUDED.sl_adjustment_reason,
        computed_at            = EXCLUDED.computed_at,
        modified_ts            = EXCLUDED.modified_ts
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
    config_path: str = "config/inventory/safety_stock_config.yaml",
    policy_version: str | None = None,
    dry_run: bool = False,
    forecast_source: str = "historical",
    model_id: str | None = None,
) -> dict[str, Any]:
    """Compute safety stock targets for all DFUs and upsert into the DB.

    Args:
        config_path:      Path to safety_stock_config.yaml.
        policy_version:   Override the policy_version from config.
        dry_run:          If True, compute but do not write to DB.
        forecast_source:  Demand stats source: 'historical' (dim_sku default),
                          'production' (promoted forecast CI bands),
                          'staging' (staging forecast CI bands).
        model_id:         When forecast_source='staging', which model_id to use.

    Returns:
        dict with inserted_count, skipped_count, zero_demand_count.
    """
    # -- Load config (via load_config for _includes support) -----------------
    with profiled_section("load_config"):
        cfg_name = Path(config_path).stem
        cfg = _load_config(cfg_name)
        ss_cfg = cfg["safety_stock"]

        pv = policy_version or ss_cfg.get("policy_version", "v1")
        # YAML-level SL targets (fallback). DB overrides from
        # fact_service_level_targets are applied inside the transaction below.
        service_levels: dict[str, float] = dict(ss_cfg["service_levels"])
        z_table: dict[str, float] = {str(k): float(v) for k, v in ss_cfg["z_table"].items()}
        min_ss_days: float = float(ss_cfg.get("min_ss_days", 3))
        max_ss_days: float = float(ss_cfg.get("max_ss_days", 120))
        lt_std_fallback_pct: float = float(ss_cfg.get("lt_std_fallback_pct", 0.20))
        batch_size: int = int(ss_cfg.get("batch_size", 1000))

        # ABC x XYZ service level matrix (optional — falls back to ABC-only)
        raw_matrix = ss_cfg.get("service_level_matrix")
        service_level_matrix: dict[str, float] | None = (
            {str(k): float(v) for k, v in raw_matrix.items()} if raw_matrix else None
        )
        xyz_thresholds: dict[str, float] | None = ss_cfg.get("xyz_thresholds")

        # ABC-specific guard rails (optional — falls back to global min/max)
        guard_rails_config: dict | None = ss_cfg.get("guard_rails")

        # Outlier detection config (optional — disabled when absent)
        outlier_cfg: dict = {}
        if guard_rails_config:
            outlier_cfg = guard_rails_config.get("outlier_detection", {})
        outlier_enabled: bool = bool(outlier_cfg.get("enabled", False))
        outlier_method: str = str(outlier_cfg.get("method", "mad"))
        outlier_threshold: float = float(outlier_cfg.get("threshold", 3.0))
        outlier_max_pct: float = float(outlier_cfg.get("max_outlier_pct", 0.20))
        volatile_sl_boost: float = float(outlier_cfg.get("volatile_sl_boost", 0.02))

        # Seasonal adjustment config (optional — disabled when absent)
        seasonal_cfg: dict = ss_cfg.get("seasonal_adjustment", {})
        seasonal_enabled: bool = bool(seasonal_cfg.get("enabled", False))
        seasonal_min_history: int = int(seasonal_cfg.get("min_history_months", 24))
        seasonal_dampening: float = float(seasonal_cfg.get("dampening", 0.5))

        # Dynamic service level adjustments (optional — no adjustments when absent)
        sl_adjustments: dict[str, float] | None = ss_cfg.get("service_level_adjustments")

    log.info(
        "Safety Stock Engine — IPfeature3 (policy_version=%s, dry_run=%s, forecast_source=%s)",
        pv, dry_run, forecast_source,
    )

    # -- Query source data ----------------------------------------------------
    conn = psycopg.connect(**get_db_params())
    conn.autocommit = False

    with conn.cursor() as cur:
        # Priority #1 (gen-4 roadmap): DB `fact_service_level_targets`
        # overrides YAML-level service_levels so SS, fill-rate, and S&OP
        # share the same numbers.
        with profiled_section("load_sl_targets"):
            from common.core.service_levels import load_sl_targets_by_abc
            db_sl_targets = load_sl_targets_by_abc(cursor=cur)
            service_levels.update(db_sl_targets)
        log.info("  SL targets resolved: %s", service_levels)

        log.info("Loading DFU demand data from dim_sku …")
        with profiled_section("load_dfu_data"):
            dfu_rows = _load_dfu_data(cur)
        log.info("  %d DFUs loaded", len(dfu_rows))

        log.info("Loading lead time profiles from dim_item_lead_time_profile …")
        with profiled_section("load_lead_times"):
            lt_map = _load_lead_time_data(cur)
        log.info("  %d LT profiles loaded", len(lt_map))

        log.info("Loading latest on-hand from agg_inventory_monthly …")
        with profiled_section("load_inventory"):
            inv_map = _load_inventory_data(cur)
        log.info("  %d inventory records loaded", len(inv_map))

        # -- Optionally load forecast-derived demand stats --------------------
        fcst_stats: dict[tuple[str, str], dict[str, float]] = {}
        if forecast_source != "historical":
            log.info(
                "Loading forecast demand stats from %s (model_id=%s) …",
                forecast_source, model_id,
            )
            with profiled_section("load_forecast_demand_stats"):
                fcst_stats = _load_forecast_demand_stats(
                    cur, source=forecast_source, model_id=model_id,
                )
            log.info("  %d DFUs with forecast-derived demand stats", len(fcst_stats))

        # -- Optionally load demand history for outlier detection ---------------
        demand_hist: dict[tuple[str, str], list[float]] = {}
        if outlier_enabled:
            log.info("Loading monthly demand history for outlier detection …")
            with profiled_section("load_demand_history"):
                demand_hist = _load_demand_history(cur)
            log.info("  %d DFUs with demand history loaded", len(demand_hist))

        # -- Optionally load demand history with dates for seasonal adjustment --
        demand_hist_dates: dict[tuple[str, str], list[tuple[date, float]]] = {}
        if seasonal_enabled:
            log.info("Loading monthly demand history with dates for seasonal adjustment …")
            with profiled_section("load_demand_history_dates"):
                demand_hist_dates = _load_demand_history_with_dates(cur)
            log.info("  %d DFUs with dated demand history loaded", len(demand_hist_dates))

    # -- Compute SS per DFU --------------------------------------------------
    now = datetime.datetime.now(datetime.UTC)
    today = now.date()
    planning_month = get_planning_date().month

    upsert_rows: list[tuple] = []
    skipped_lt_zero = 0
    zero_demand_count = 0
    skipped_count = 0
    fcst_used_count = 0
    volatile_count = 0
    seasonal_count = 0
    sl_adjusted_count = 0

    with profiled_section("compute_safety_stock"):
        for dfu in dfu_rows:
            item_id: str = dfu["item_id"]
            loc: str = dfu["loc"]
            abc_vol: str | None = dfu.get("abc_vol")

            # -- Demand stats: prefer forecast-derived when available -----------
            fcst_rec = fcst_stats.get((item_id, loc))
            if forecast_source != "historical" and fcst_rec is not None:
                demand_mean = fcst_rec["demand_mean"]
                demand_std = fcst_rec["demand_std"]
                demand_cv = fcst_rec["demand_cv"]
                row_forecast_source = forecast_source
                row_forecast_model_id = model_id
                fcst_used_count += 1
            else:
                demand_mean = float(dfu.get("demand_mean") or 0.0)
                demand_std = float(dfu.get("demand_std") or 0.0)
                demand_cv_raw = dfu.get("demand_cv")
                demand_cv = float(demand_cv_raw) if demand_cv_raw is not None else None
                row_forecast_source = "historical"
                row_forecast_model_id = None

            # -- Lead time from profile table or fallback -----------------------
            lt_rec = lt_map.get((item_id, loc), {})
            lt_mean_raw = lt_rec.get("lt_mean_days")
            lt_std_raw = lt_rec.get("lt_std_days")

            if lt_mean_raw is None:
                lt_mean_days = 14.0          # no profile → assume 14-day lead time
            else:
                lt_mean_days = float(lt_mean_raw)

            if lt_mean_days <= 0:
                log.warning("Skipping %s/%s: lt_mean_days=0 (invalid)", item_id, loc)
                skipped_lt_zero += 1
                skipped_count += 1
                continue

            if lt_std_raw is None:
                lt_std_days = lt_mean_days * lt_std_fallback_pct
            else:
                lt_std_days = float(lt_std_raw)

            lt_std_days = max(lt_std_days, 0.1)  # Prevent zero in SS formula

            # -- Outlier detection on demand history --------------------------------
            outlier_pct: float = 0.0
            has_demand_outliers: bool = False
            is_volatile: bool = False

            if outlier_enabled:
                hist = demand_hist.get((item_id, loc), [])
                if hist:
                    outlier_pct, is_volatile = detect_outliers(
                        hist,
                        method=outlier_method,
                        threshold=outlier_threshold,
                        max_outlier_pct=outlier_max_pct,
                    )
                    has_demand_outliers = outlier_pct > 0.0
                    if is_volatile:
                        volatile_count += 1

            # -- Seasonal factor (compute early — needed for SL adjustments) -----
            seasonal_factor_val: float | None = None
            ss_seasonal_val: float | None = None
            is_seasonal_adjusted: bool = False
            is_peak_season: bool = False
            is_trough_season: bool = False

            if seasonal_enabled:
                hist_dates = demand_hist_dates.get((item_id, loc), [])
                seasonal_factors = compute_seasonal_factors(
                    hist_dates,
                    min_history_months=seasonal_min_history,
                )
                factor = seasonal_factors.get(planning_month, 1.0)
                # Determine peak/trough from seasonal factor for SL adjustments
                is_peak_season = factor > 1.15
                is_trough_season = factor < 0.85

            # -- Intermittency ratio from dim_sku ----------------------------------
            intermittency_raw = dfu.get("intermittency_ratio")
            intermittency_ratio = float(intermittency_raw) if intermittency_raw is not None else 0.0

            # -- Service level & Z-score ----------------------------------------
            sl, xyz_class, abc_xyz_segment, sl_adjustment_reason = get_service_level(
                abc_vol, service_levels,
                demand_cv=demand_cv,
                service_level_matrix=service_level_matrix,
                xyz_thresholds=xyz_thresholds,
                is_peak_season=is_peak_season,
                is_trough_season=is_trough_season,
                intermittency_ratio=intermittency_ratio,
                adjustments=sl_adjustments,
            )

            if sl_adjustment_reason is not None:
                sl_adjusted_count += 1

            # Volatile items get a service level boost (capped at 0.999)
            if is_volatile and volatile_sl_boost > 0:
                sl = min(sl + volatile_sl_boost, 0.999)

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

            # -- Guard rails (ABC-specific) ----------------------------------------
            ss_combined, guard_rail_applied, guard_rail_min, guard_rail_max = (
                apply_guard_rails(
                    ss_combined, avg_daily, min_ss_days, max_ss_days,
                    abc_vol=abc_vol,
                    guard_rails_config=guard_rails_config,
                )
            )

            # -- Seasonal SS adjustment (optional) ---------------------------------
            if seasonal_enabled:
                factor = seasonal_factors.get(planning_month, 1.0)
                # Only apply if the factor deviates from 1.0 (i.e. real seasonality)
                has_real_seasonality = any(
                    abs(v - 1.0) > 1e-6 for v in seasonal_factors.values()
                )
                if has_real_seasonality:
                    seasonal_factor_val = round(factor, 4)
                    ss_seasonal_val = round(
                        apply_seasonal_adjustment(
                            ss_combined, factor, dampening=seasonal_dampening,
                        ),
                        4,
                    )
                    is_seasonal_adjusted = True
                    seasonal_count += 1

            # -- Current inventory position ---------------------------------------
            current_qty = inv_map.get((item_id, loc), 0.0)

            pos = compute_position_metrics(
                ss_combined=ss_combined,
                avg_daily_demand=avg_daily,
                lt_mean_days=lt_mean_days,
                current_qty_on_hand=current_qty,
            )

            # -- Build composite key & row tuple ----------------------------------
            ss_ck = f"{item_id}_{loc}_{pv}"

            row: tuple = (
                ss_ck,
                item_id, loc, pv, today,
                sl, z,
                demand_mean, demand_std,
                lt_mean_days, lt_std_days,
                abc_vol, xyz_class, abc_xyz_segment,
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
                row_forecast_source,
                row_forecast_model_id,
                has_demand_outliers,
                outlier_pct if outlier_pct > 0 else None,
                guard_rail_applied,
                round(guard_rail_min, 2) if guard_rail_min else None,
                round(guard_rail_max, 2) if guard_rail_max else None,
                seasonal_factor_val,
                ss_seasonal_val,
                is_seasonal_adjusted,
                sl_adjustment_reason,
                now, now, now,           # computed_at, load_ts, modified_ts
            )
            upsert_rows.append(row)

    log.info(
        "Computation complete: %d rows to upsert, %d zero-demand, %d skipped, "
        "%d forecast-derived, %d volatile, %d seasonal-adjusted, %d sl-adjusted",
        len(upsert_rows),
        zero_demand_count,
        skipped_count,
        fcst_used_count,
        volatile_count,
        seasonal_count,
        sl_adjusted_count,
    )

    if dry_run:
        log.info("[DRY RUN] No data written to DB.")
        # Print a sample
        if upsert_rows:
            sample = upsert_rows[0]
            log.info(
                "Sample row: item=%s loc=%s abc_xyz=%s ss_combined=%.2f rop=%.2f is_below=%s",
                sample[1], sample[2], sample[13], sample[16], sample[22], sample[29],
            )
        conn.close()
        return {
            "inserted_count": 0,
            "skipped_count": skipped_count,
            "zero_demand_count": zero_demand_count,
            "volatile_count": volatile_count,
            "seasonal_count": seasonal_count,
            "sl_adjusted_count": sl_adjusted_count,
            "dry_run": True,
        }

    # -- Upsert into DB -------------------------------------------------------
    with profiled_section("batch_upsert"):
        with conn.cursor() as cur:
            inserted = _batch_upsert(cur, upsert_rows, batch_size)

        conn.commit()
    conn.close()

    log.info("Upserted %d rows into fact_safety_stock_targets (policy_version=%s)", inserted, pv)

    return {
        "inserted_count": inserted,
        "skipped_count": skipped_count,
        "zero_demand_count": zero_demand_count,
        "volatile_count": volatile_count,
        "seasonal_count": seasonal_count,
        "sl_adjusted_count": sl_adjusted_count,
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
        default="config/inventory/safety_stock_config.yaml",
        help="Path to safety_stock_config.yaml (default: config/inventory/safety_stock_config.yaml)",
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
    parser.add_argument(
        "--forecast-source",
        choices=["historical", "production", "staging"],
        default="historical",
        help="Demand stats source: historical (dim_sku), production (promoted forecast), staging (staging forecast)",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="When --forecast-source=staging, which model_id to use",
    )
    args = parser.parse_args()
    if args.forecast_source == "staging" and not args.model_id:
        parser.error("--model-id is required when --forecast-source=staging")
    return args


if __name__ == "__main__":
    args = _parse_args()
    result = run(
        config_path=args.config,
        policy_version=args.policy_version,
        dry_run=args.dry_run,
        forecast_source=args.forecast_source,
        model_id=args.model_id,
    )
    log.info("Result: %s", result)
    sys.exit(0)
