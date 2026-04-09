-- 129_add_xyz_to_safety_stock.sql
-- Adds XYZ classification and ABC-XYZ segment columns to fact_safety_stock_targets.
-- Supports the 9-cell ABC x XYZ service level matrix (Issue #1).
--
-- xyz_class:      'X' (stable, CV < 0.3), 'Y' (moderate, 0.3-0.8), 'Z' (volatile, > 0.8)
-- abc_xyz_segment: Combined segment e.g. 'AX', 'BY', 'CZ' for matrix service level lookup

ALTER TABLE fact_safety_stock_targets
    ADD COLUMN IF NOT EXISTS xyz_class VARCHAR(2);       -- X, Y, or Z (NULL when demand_cv unavailable)

ALTER TABLE fact_safety_stock_targets
    ADD COLUMN IF NOT EXISTS abc_xyz_segment VARCHAR(5);  -- e.g. AX, BY, CZ (NULL when either class unavailable)

-- Index for ABC-XYZ segment filtering (analytics, dashboards)
CREATE INDEX IF NOT EXISTS idx_ss_targets_abc_xyz_segment
    ON fact_safety_stock_targets (abc_xyz_segment);
