-- 117: Add targeted partial + composite indexes for hot query paths
-- These replace the broad, unused indexes that were dropped by the
-- drop_unused_indexes.py maintenance script.
--
-- Strategy: fewer, more targeted indexes that cover actual query patterns
-- found in the API routers and scripts.

-- ==========================================================================
-- 1. PARTIAL INDEXES — filter on common WHERE clause predicates
-- ==========================================================================

-- Replenishment exceptions: action feed queries filter on open + severity
-- Used by ~35 queries across insights, action feed, release planned orders
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_repl_exc_open_severity
ON fact_replenishment_exceptions (item_id, loc, severity)
WHERE status = 'open';

-- Demand signals: alert feed queries filter on urgent priority
-- Used by ~20 queries in insights and demand signal endpoints
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_demand_signals_urgent
ON fact_demand_signals (item_id, loc, signal_date)
WHERE alert_priority = 'urgent';

-- Forecast overrides: approval workflow queries filter pending status
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_override_pending
ON fact_forecast_overrides (item_id, loc, override_month)
WHERE status = 'pending_approval';

-- Planned orders: release workflow queries filter proposed status
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_planned_orders_proposed
ON fact_planned_orders (item_id, loc, order_by_date)
WHERE status = 'proposed';

-- Inventory health: control tower queries filter critical/at-risk tiers
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_health_critical_atrisk
ON mv_inventory_health_score (item_id, loc)
WHERE health_tier IN ('critical', 'at_risk');

-- Exception queue: active exceptions for planner dashboard
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_exception_queue_active
ON exception_queue (severity DESC, generated_at DESC)
WHERE status IN ('open', 'investigating');

-- ==========================================================================
-- 2. COMPOSITE COVERING INDEXES — satisfy common multi-column lookups
-- ==========================================================================

-- Safety stock targets: frequently joined with replenishment plan
-- Covers the most common lookup pattern (item + loc + latest policy)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ss_targets_item_loc_version
ON fact_safety_stock_targets (item_id, loc, policy_version DESC);

-- Inventory projection: timeline queries need item+loc+scenario+date
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_inv_proj_item_loc_scenario
ON fact_inventory_projection (item_id, loc, scenario, projection_date);

-- Bias corrections: accuracy review queries need item+loc+month
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_bias_corr_item_loc_month
ON fact_bias_corrections (item_id, loc, plan_month);

-- Service level performance: SLA review queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_svc_perf_item_loc_month
ON fact_service_level_performance (item_id, loc, perf_month);
