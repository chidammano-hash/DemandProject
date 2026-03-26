-- 099_add_fk_constraints.sql
-- Add foreign key constraints from fact tables to dimension tables.
-- Prerequisites: unique indexes on dim_item.item_id and dim_location.location_id.
-- Uses NOT VALID to avoid full table scans / exclusive locks on creation,
-- followed by VALIDATE CONSTRAINT to verify existing rows in the background.

-- ============================================================
-- Step 1: Ensure referenced columns have UNIQUE indexes
-- (required for FK targets; idempotent via IF NOT EXISTS)
-- ============================================================

CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_item_item_id
    ON dim_item (item_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_location_location_id
    ON dim_location (location_id);

-- ============================================================
-- Step 2: FK from fact_sales_monthly.item_id -> dim_item.item_id
-- ============================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_fact_sales_monthly_item_id'
          AND conrelid = 'fact_sales_monthly'::regclass
    ) THEN
        ALTER TABLE fact_sales_monthly
            ADD CONSTRAINT fk_fact_sales_monthly_item_id
            FOREIGN KEY (item_id) REFERENCES dim_item (item_id)
            NOT VALID;
    END IF;
END $$;

ALTER TABLE fact_sales_monthly
    VALIDATE CONSTRAINT fk_fact_sales_monthly_item_id;

-- ============================================================
-- Step 3: FK from fact_sales_monthly.loc -> dim_location.location_id
-- ============================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_fact_sales_monthly_loc'
          AND conrelid = 'fact_sales_monthly'::regclass
    ) THEN
        ALTER TABLE fact_sales_monthly
            ADD CONSTRAINT fk_fact_sales_monthly_loc
            FOREIGN KEY (loc) REFERENCES dim_location (location_id)
            NOT VALID;
    END IF;
END $$;

ALTER TABLE fact_sales_monthly
    VALIDATE CONSTRAINT fk_fact_sales_monthly_loc;

-- ============================================================
-- Step 4: FK from fact_external_forecast_monthly.item_id -> dim_item.item_id
-- ============================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_fact_external_forecast_monthly_item_id'
          AND conrelid = 'fact_external_forecast_monthly'::regclass
    ) THEN
        ALTER TABLE fact_external_forecast_monthly
            ADD CONSTRAINT fk_fact_external_forecast_monthly_item_id
            FOREIGN KEY (item_id) REFERENCES dim_item (item_id)
            NOT VALID;
    END IF;
END $$;

ALTER TABLE fact_external_forecast_monthly
    VALIDATE CONSTRAINT fk_fact_external_forecast_monthly_item_id;

-- ============================================================
-- Step 5: FK from fact_external_forecast_monthly.loc -> dim_location.location_id
-- ============================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_fact_external_forecast_monthly_loc'
          AND conrelid = 'fact_external_forecast_monthly'::regclass
    ) THEN
        ALTER TABLE fact_external_forecast_monthly
            ADD CONSTRAINT fk_fact_external_forecast_monthly_loc
            FOREIGN KEY (loc) REFERENCES dim_location (location_id)
            NOT VALID;
    END IF;
END $$;

ALTER TABLE fact_external_forecast_monthly
    VALIDATE CONSTRAINT fk_fact_external_forecast_monthly_loc;
