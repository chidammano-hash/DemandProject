-- 134_add_seasonal_ss_cols.sql
-- Adds seasonal safety stock adaptation columns to fact_safety_stock_targets.
-- Supports Issue #10: monthly seasonal demand profiles that scale SS up during
-- peak months and down during trough months.
--
-- seasonal_factor:       Monthly demand ratio (>1.0 = peak, <1.0 = trough, 1.0 = average)
-- ss_seasonal:           Seasonally adjusted safety stock quantity
-- is_seasonal_adjusted:  Whether seasonal adjustment was applied for this row

ALTER TABLE fact_safety_stock_targets
    ADD COLUMN IF NOT EXISTS seasonal_factor NUMERIC(6,4);       -- e.g. 1.35 (Dec peak), 0.72 (Jun trough)

ALTER TABLE fact_safety_stock_targets
    ADD COLUMN IF NOT EXISTS ss_seasonal NUMERIC(15,4);          -- ss_combined * sqrt(seasonal_factor) blended with dampening

ALTER TABLE fact_safety_stock_targets
    ADD COLUMN IF NOT EXISTS is_seasonal_adjusted BOOLEAN DEFAULT FALSE;  -- TRUE when seasonal adjustment was applied
