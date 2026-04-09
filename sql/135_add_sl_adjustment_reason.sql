-- IPfeature11: Dynamic ABC-Class Service Level Policies
-- Adds sl_adjustment_reason column to track which dynamic adjustments
-- were applied to the base service level (e.g. seasonal peak boost,
-- intermittent demand relaxation). NULL when no adjustments applied.

ALTER TABLE fact_safety_stock_targets
    ADD COLUMN IF NOT EXISTS sl_adjustment_reason TEXT;
