-- 119: Add unique indexes on materialized views to enable CONCURRENTLY refresh
--
-- Without a unique index, REFRESH MATERIALIZED VIEW CONCURRENTLY fails.
-- With it, the refresh builds a new version in the background and swaps
-- atomically — zero read downtime for dashboards.

-- ==========================================================================
-- Control Tower KPIs (refreshed every 5 min for real-time dashboard)
-- ==========================================================================
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_control_tower_kpis_at
ON mv_control_tower_kpis (computed_at);

-- ==========================================================================
-- Inventory Health Score (refreshed every 30 min)
-- ==========================================================================
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_inv_health_item_loc
ON mv_inventory_health_score (item_id, loc);

-- ==========================================================================
-- Fill Rate Monthly (refreshed daily)
-- ==========================================================================
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_fill_rate_item_loc_month
ON mv_fill_rate_monthly (item_id, loc, month_start);

-- ==========================================================================
-- Intramonth Stockout (refreshed daily, currently takes 10-30 min)
-- ==========================================================================
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_intramonth_item_loc_month
ON mv_intramonth_stockout (item_id, loc, month_start);

-- ==========================================================================
-- Supplier Performance (refreshed daily)
-- ==========================================================================
-- Already has unique on supplier_no in the MV definition, verify:
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_supplier_perf_supplier
ON mv_supplier_performance (supplier_no);

-- ==========================================================================
-- Supplier PO Performance (refreshed daily)
-- ==========================================================================
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_supplier_po_perf
ON mv_supplier_po_performance (supplier_id);

-- ==========================================================================
-- DQ Dashboard (refreshed after every load)
-- ==========================================================================
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_dq_dashboard_domain_date
ON mv_dq_dashboard (domain, run_date);

-- ==========================================================================
-- Inventory Forecast Monthly (refreshed after forecast loads)
-- ==========================================================================
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_inv_fcst_item_loc_month_model
ON mv_inventory_forecast_monthly (item_id, loc, month_start, model_id);

-- ==========================================================================
-- Demand Decomposition (refreshed after forecast loads)
-- ==========================================================================
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_demand_decomp_item_loc_month
ON mv_demand_decomposition (item_id, loc, month);

-- ==========================================================================
-- Sensing Overrides Active (refreshed frequently)
-- ==========================================================================
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_sensing_overrides_item_loc
ON mv_sensing_overrides_active (item_id, loc);

-- ==========================================================================
-- Accuracy MVs (refreshed after backtest loads)
-- ==========================================================================
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_agg_accuracy_dim
ON agg_accuracy_by_dim (model_id, lag, month_start, cluster_assignment, ml_cluster,
                        supplier_desc, abc_vol, region, brand_desc, seasonality_profile);

CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_agg_accuracy_lag_archive
ON agg_accuracy_lag_archive (model_id, lag, timeframe, month_start, cluster_assignment,
                             ml_cluster, supplier_desc, abc_vol, region, brand_desc,
                             seasonality_profile);

-- ==========================================================================
-- Sales & Forecast Aggregates (refreshed after data loads)
-- ==========================================================================
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_agg_sales_item_loc_month
ON agg_sales_monthly (item_id, loc, month_start);

CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_agg_forecast_item_loc_month_model
ON agg_forecast_monthly (item_id, loc, month_start, model_id);

-- ==========================================================================
-- Inventory Monthly (refreshed after snapshot loads)
-- ==========================================================================
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_agg_inv_item_loc_month
ON agg_inventory_monthly (item_id, loc, month_start);

-- ==========================================================================
-- DFU Coverage MVs (refreshed after backtest loads)
-- ==========================================================================
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_dfu_coverage_model_lag_dfu
ON agg_dfu_coverage (model_id, lag, item_id, customer_group, loc);

CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_dfu_coverage_lag_archive
ON agg_dfu_coverage_lag_archive (model_id, lag, item_id, customer_group, loc);

-- ==========================================================================
-- Inventory Projection Summary (refreshed after projection runs)
-- ==========================================================================
-- Already has uq_mv_proj_summary, verify it exists
-- CREATE UNIQUE INDEX ... ON mv_inventory_projection_summary (...)
-- Skipping — already defined in 045_create_inventory_projection.sql

-- ==========================================================================
-- Rebalancing Analysis (refreshed after rebalancing runs)
-- Skipped — mv_rebalancing_analysis may not exist yet (stub feature)
-- ==========================================================================
-- CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_rebalancing_analysis
-- ON mv_rebalancing_analysis (from_loc, to_loc, item_id)
-- WHERE from_loc IS NOT NULL AND to_loc IS NOT NULL;

-- ==========================================================================
-- Network Balance (refreshed after inventory loads)
-- Skipped — mv_network_balance may not exist yet (stub feature)
-- ==========================================================================
-- CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_network_balance_item
-- ON mv_network_balance (item_id);
