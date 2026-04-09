-- Issue #9: Intelligent Guard Rails — ABC-specific bounds + outlier detection
-- Adds columns to fact_safety_stock_targets for tracking guard rail application
-- and demand outlier detection results.

ALTER TABLE fact_safety_stock_targets
    ADD COLUMN IF NOT EXISTS has_demand_outliers BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS outlier_pct NUMERIC(5,4),
    ADD COLUMN IF NOT EXISTS guard_rail_applied BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS guard_rail_min NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS guard_rail_max NUMERIC(12,2);

-- Index for filtering outlier/volatile items in exception queues
CREATE INDEX IF NOT EXISTS idx_ss_targets_outliers
    ON fact_safety_stock_targets (has_demand_outliers)
    WHERE has_demand_outliers = TRUE;

-- Index for filtering guard-rail-clamped items
CREATE INDEX IF NOT EXISTS idx_ss_targets_guard_rail
    ON fact_safety_stock_targets (guard_rail_applied)
    WHERE guard_rail_applied = TRUE;
