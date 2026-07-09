"""Central materialized-view refresh service.

Single source of truth mapping every materialized view to the relations
(fact/dim tables and upstream MVs) it reads. Any code path that writes a fact
or dimension table refreshes the dependent MVs through this module — never a
hand-picked inline list. Hand-picked lists are how MVs silently went stale:
``agg_dfu_naive_scale`` (the MASE denominator) had no automated refresher at
all, and ``agg_accuracy_by_dfu`` was skipped by three of its writers.

``MV_SOURCES`` is verified against the sql/ DDL by
``tests/unit/test_mv_refresh.py``: every ``CREATE MATERIALIZED VIEW`` in sql/
must appear here, and the dict order must be a valid dependency order (an
MV's upstream MVs come before it). Add new MVs to this map in the same
change as their DDL.

Typical usage after committing a write::

    from common.core.mv_refresh import refresh_for_tables
    refresh_for_tables(["fact_external_forecast_monthly", "backtest_lag_archive"])

Refreshes run on a dedicated autocommit connection so ``REFRESH MATERIALIZED
VIEW CONCURRENTLY`` is legal (it cannot run inside a transaction block); a
non-populated or index-less MV falls back to a plain ``REFRESH``.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event
from typing import Callable, Iterable

import psycopg
from psycopg import sql as psql

from common.core.db import get_db_params

logger = logging.getLogger(__name__)

# MV name -> relations (tables and upstream MVs) its query reads.
# Dict order IS the refresh order: upstream MVs are declared before dependents.
MV_SOURCES: dict[str, frozenset[str]] = {
    # ── Tier 1: aggregates directly over fact/dim tables ────────────────────
    "agg_sales_monthly": frozenset({"fact_sales_monthly"}),
    "agg_sales_weekly": frozenset({"fact_sales_monthly"}),
    "agg_forecast_monthly": frozenset({"fact_external_forecast_monthly"}),
    "agg_inventory_monthly": frozenset({"fact_inventory_snapshot"}),
    "agg_accuracy_by_dim": frozenset(
        {"fact_external_forecast_monthly", "dim_sku", "sku_cluster_assignment", "cluster_experiment"}
    ),
    "agg_accuracy_by_dfu": frozenset(
        {"fact_external_forecast_monthly", "dim_sku", "sku_cluster_assignment", "cluster_experiment"}
    ),
    "agg_dfu_coverage": frozenset(
        {"fact_external_forecast_monthly", "dim_sku", "sku_cluster_assignment", "cluster_experiment"}
    ),
    "agg_accuracy_lag_archive": frozenset(
        {"backtest_lag_archive", "dim_sku", "sku_cluster_assignment", "cluster_experiment"}
    ),
    "agg_dfu_coverage_lag_archive": frozenset(
        {"backtest_lag_archive", "dim_sku", "sku_cluster_assignment", "cluster_experiment"}
    ),
    "agg_dfu_naive_scale": frozenset({"fact_sales_monthly", "fact_external_forecast_monthly"}),
    "agg_accuracy_snapshot": frozenset({"fact_forecast_snapshot", "fact_sales_monthly"}),
    "mv_fill_rate_monthly": frozenset({"fact_sales_monthly", "dim_sku"}),
    "mv_intramonth_stockout": frozenset({"fact_inventory_snapshot", "dim_sku"}),
    # mv_supplier_performance was retired by sql/143 — mv_supplier_po_performance
    # is the only supplier MV.
    "mv_supplier_po_performance": frozenset({"fact_purchase_orders"}),
    "mv_po_lead_time_analysis": frozenset({"fact_purchase_orders"}),
    "mv_integrated_planning_targets": frozenset(
        {"fact_safety_stock_targets", "fact_eoq_targets"}
    ),
    "mv_inventory_projection_summary": frozenset({"fact_inventory_projection"}),
    "mv_dq_dashboard": frozenset({"fact_dq_check_results"}),
    "mv_sensing_overrides_active": frozenset({"fact_blended_demand_plan"}),
    "mv_customer_activity_monthly": frozenset({"fact_customer_demand_monthly", "dim_customer"}),
    "mv_customer_filter_options": frozenset({"dim_customer"}),
    "mv_ca_segment_trends": frozenset({"fact_customer_demand_monthly", "dim_customer"}),
    "mv_ca_demand_at_risk": frozenset({"fact_customer_demand_monthly"}),
    "mv_ca_order_patterns": frozenset({"fact_customer_demand_monthly", "dim_customer"}),
    "mv_ca_item_state": frozenset(
        {"fact_customer_demand_monthly", "dim_customer", "dim_item"}
    ),
    # ── Tier 2: MVs reading tier-1 MVs ──────────────────────────────────────
    "mv_inventory_forecast_monthly": frozenset(
        {"agg_inventory_monthly", "dim_sku", "fact_external_forecast_monthly"}
    ),
    "mv_network_balance": frozenset({"agg_inventory_monthly", "fact_safety_stock_targets"}),
    "mv_fairness_audit": frozenset(
        {"agg_sales_monthly", "fact_production_forecast", "dim_sku", "dim_location"}
    ),
    # ── Tier 3 ───────────────────────────────────────────────────────────────
    "mv_inventory_health_score": frozenset(
        {
            "agg_inventory_monthly",
            "mv_inventory_forecast_monthly",
            "dim_sku",
            "fact_safety_stock_targets",
        }
    ),
    # ── Tier 4 ───────────────────────────────────────────────────────────────
    "mv_control_tower_kpis": frozenset(
        {
            "mv_inventory_health_score",
            "mv_fill_rate_monthly",
            "mv_intramonth_stockout",
            "fact_demand_signals",
            "fact_replenishment_exceptions",
        }
    ),
}

# MVs whose full refresh is expensive enough (fact_inventory_snapshot scan,
# 10-30 min) that dimension-only writers may explicitly opt out; the
# scheduled refresh_all_mvs safety net still covers them.
HEAVY_MVS: frozenset[str] = frozenset({"mv_intramonth_stockout"})


def all_mvs() -> list[str]:
    """All known MVs in dependency (refresh) order."""
    return list(MV_SOURCES)


def mvs_for_tables(tables: Iterable[str], *, include_heavy: bool = True) -> list[str]:
    """MVs that (directly or transitively) read any of *tables*, in refresh order.

    A single pass in map order suffices because ``MV_SOURCES`` is
    topologically ordered: by the time a dependent MV is considered, its
    upstream MVs have already been marked dirty.
    """
    dirty: set[str] = {str(t) for t in tables}
    ordered: list[str] = []
    for mv, sources in MV_SOURCES.items():
        if sources & dirty:
            dirty.add(mv)
            if not include_heavy and mv in HEAVY_MVS:
                logger.info("Skipping heavy MV %s (include_heavy=False)", mv)
                continue
            ordered.append(mv)
    return ordered


def refresh_tiers(mvs: Iterable[str]) -> list[list[str]]:
    """Group *mvs* into dependency levels for parallel refresh.

    MVs within one level have no dependency on each other; each level only
    depends on earlier levels.
    """
    requested = [mv for mv in MV_SOURCES if mv in set(mvs)]
    level: dict[str, int] = {}
    for mv in MV_SOURCES:  # full-map pass so levels are stable across subsets
        upstream = [s for s in MV_SOURCES[mv] if s in MV_SOURCES]
        level[mv] = 1 + max((level[u] for u in upstream), default=-1)
    tiers: dict[int, list[str]] = {}
    for mv in requested:
        tiers.setdefault(level[mv], []).append(mv)
    return [tiers[k] for k in sorted(tiers)]


def _populated_map(cur, mvs: list[str]) -> dict[str, bool | None]:
    """Return {mv: True/False if it exists (populated?), None if missing}."""
    cur.execute(
        "SELECT matviewname, ispopulated FROM pg_matviews "
        "WHERE schemaname = 'public' AND matviewname = ANY(%s)",
        (mvs,),
    )
    found = {row[0]: bool(row[1]) for row in cur.fetchall()}
    return {mv: found.get(mv) for mv in mvs}


def _refresh_one(cur, mv: str, populated: bool) -> None:
    """Refresh a single MV: CONCURRENTLY when possible, plain otherwise."""
    if populated:
        try:
            cur.execute(
                psql.SQL("REFRESH MATERIALIZED VIEW CONCURRENTLY {}").format(
                    psql.Identifier(mv)
                )
            )
            return
        except psycopg.Error as exc:
            # No unique index / other CONCURRENTLY restriction — fall through.
            logger.warning("CONCURRENTLY refresh of %s failed (%s); retrying plain", mv, exc)
    cur.execute(psql.SQL("REFRESH MATERIALIZED VIEW {}").format(psql.Identifier(mv)))


def refresh_materialized_views(
    mvs: Iterable[str],
    *,
    db_params: dict | None = None,
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
) -> dict[str, list[str]]:
    """Refresh *mvs* sequentially on a dedicated autocommit connection.

    Returns ``{"refreshed": [...], "failed": [...], "missing": [...]}``.
    Failures are logged and skipped so one broken MV does not block the rest.
    """
    mv_list = list(mvs)
    result: dict[str, list[str]] = {"refreshed": [], "failed": [], "missing": []}
    if not mv_list:
        return result

    with psycopg.connect(**(db_params or get_db_params()), autocommit=True) as conn:
        with conn.cursor() as cur:
            try:
                # Session knob matching the historical writer paths.
                cur.execute("SET maintenance_work_mem = '512MB'")
            except psycopg.Error:
                logger.exception("Failed to set maintenance_work_mem")
            populated = _populated_map(cur, mv_list)
            for idx, mv in enumerate(mv_list, start=1):
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("Job cancelled by user")
                if progress_cb:
                    pct = int(5 + (idx - 1) / len(mv_list) * 90)
                    progress_cb(pct=pct, msg=f"Refreshing {mv}")
                if populated[mv] is None:
                    logger.info("MV %s does not exist — skipping", mv)
                    result["missing"].append(mv)
                    continue
                try:
                    _refresh_one(cur, mv, bool(populated[mv]))
                    result["refreshed"].append(mv)
                except psycopg.Error:
                    logger.exception("Failed to refresh %s", mv)
                    result["failed"].append(mv)

    if progress_cb:
        progress_cb(pct=100, msg=f"Refreshed {len(result['refreshed'])}/{len(mv_list)} views")
    logger.info(
        "MV refresh done: %d refreshed, %d failed, %d missing",
        len(result["refreshed"]), len(result["failed"]), len(result["missing"]),
    )
    return result


def refresh_materialized_views_parallel(
    mvs: Iterable[str],
    *,
    max_workers: int = 3,
    db_params: dict | None = None,
) -> dict[str, list[str]]:
    """Refresh *mvs* tier-by-tier, parallel within a tier.

    Each worker uses its own autocommit connection. Tiers guarantee an MV
    never refreshes before the MVs it reads.
    """
    result: dict[str, list[str]] = {"refreshed": [], "failed": [], "missing": []}
    params = db_params or get_db_params()

    def _one(mv: str) -> tuple[str, str]:
        try:
            with psycopg.connect(**params, autocommit=True) as conn, conn.cursor() as cur:
                populated = _populated_map(cur, [mv])[mv]
                if populated is None:
                    return mv, "missing"
                _refresh_one(cur, mv, bool(populated))
                return mv, "refreshed"
        except psycopg.Error:
            logger.exception("Failed to refresh %s", mv)
            return mv, "failed"

    for tier in refresh_tiers(mvs):
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_one, mv) for mv in tier]
            for future in as_completed(futures):
                mv, status = future.result()
                result[status].append(mv)
                logger.info("  %s: %s", mv, status)
    return result


def refresh_for_tables(
    tables: Iterable[str],
    *,
    include_heavy: bool = True,
    db_params: dict | None = None,
    progress_cb: Callable | None = None,
    cancel_event: Event | None = None,
) -> dict[str, list[str]]:
    """Refresh every MV that depends on any of *tables* (the standard entry point)."""
    mvs = mvs_for_tables(tables, include_heavy=include_heavy)
    if not mvs:
        logger.info("No MVs depend on tables %s", sorted(set(tables)))
        return {"refreshed": [], "failed": [], "missing": []}
    logger.info("Refreshing %d MV(s) for tables %s", len(mvs), sorted(set(tables)))
    return refresh_materialized_views(
        mvs, db_params=db_params, progress_cb=progress_cb, cancel_event=cancel_event
    )
