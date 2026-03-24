-- 095_create_dq_corrections.sql
-- Audit trail for DQ auto-fix corrections.
-- Stores original vs corrected values so the UI can show before/after comparisons.

CREATE TABLE IF NOT EXISTS fact_dq_corrections (
    correction_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    domain          TEXT        NOT NULL,       -- e.g. 'sales', 'inventory'
    table_name      TEXT        NOT NULL,       -- e.g. 'fact_sales_monthly'
    item_id         TEXT,                       -- item key (nullable for non-item tables)
    loc             TEXT,                       -- location key
    period          DATE,                       -- startdate or snapshot_date
    column_name     TEXT        NOT NULL,       -- e.g. 'qty', 'qty_on_hand'
    old_value       DOUBLE PRECISION,           -- original value before correction
    new_value       DOUBLE PRECISION,           -- corrected value
    fix_type        TEXT        NOT NULL,       -- 'range', 'outliers', 'completeness', 'lead_time'
    fix_strategy    TEXT,                       -- 'iqr_per_sku', 'iqr_global', 'zscore', 'clamp', 'median_impute'
    threshold       DOUBLE PRECISION,           -- IQR multiplier or Z-score threshold used
    lower_bound     DOUBLE PRECISION,           -- computed lower bound for this group
    upper_bound     DOUBLE PRECISION,           -- computed upper bound for this group
    applied_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for querying corrections by item/loc and by period
CREATE INDEX IF NOT EXISTS idx_dq_corrections_item_loc
    ON fact_dq_corrections (item_id, loc);
CREATE INDEX IF NOT EXISTS idx_dq_corrections_table_col
    ON fact_dq_corrections (table_name, column_name);
CREATE INDEX IF NOT EXISTS idx_dq_corrections_period
    ON fact_dq_corrections (period);
CREATE INDEX IF NOT EXISTS idx_dq_corrections_applied_at
    ON fact_dq_corrections (applied_at);
