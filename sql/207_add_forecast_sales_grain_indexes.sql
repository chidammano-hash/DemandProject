-- Production forecasting reads bounded history at the canonical DFU grain.
-- These indexes support both the all-history first-observation lookup and the
-- closed-window quantity scan without dropping customer_group from the key.

CREATE INDEX IF NOT EXISTS idx_fact_sales_monthly_dfu_date_nonnull
    ON fact_sales_monthly (item_id, customer_group, loc, startdate)
    WHERE type = 1 AND qty IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_fact_sales_original_dfu_date_nonnull
    ON fact_sales_monthly_original (item_id, customer_group, loc, startdate)
    WHERE type = 1 AND qty IS NOT NULL;
