-- Original (uncorrected) sales table for accuracy computation.
-- Identical schema to fact_sales_monthly but NEVER receives DQ corrections.
-- Backtest framework and accuracy views read from this table.

CREATE TABLE IF NOT EXISTS fact_sales_monthly_original (
    sales_sk        BIGSERIAL PRIMARY KEY,
    sales_ck        TEXT UNIQUE NOT NULL,
    item_id         TEXT NOT NULL,
    customer_group        TEXT NOT NULL,
    loc             TEXT NOT NULL,
    startdate       DATE NOT NULL,
    type            INTEGER NOT NULL,
    qty_shipped     NUMERIC(18,4),
    qty_ordered     NUMERIC(18,4),
    qty             NUMERIC(18,4),
    file_dt         DATE,
    load_ts         TIMESTAMPTZ DEFAULT NOW(),
    modified_ts     TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT chk_fact_sales_orig_type_1 CHECK (type = 1),
    CONSTRAINT chk_fact_sales_orig_month_start CHECK (
        startdate = date_trunc('month', startdate)::date
    )
);

CREATE INDEX IF NOT EXISTS idx_fact_sales_orig_item_loc_date
    ON fact_sales_monthly_original (item_id, loc, startdate);
CREATE INDEX IF NOT EXISTS idx_fact_sales_orig_startdate
    ON fact_sales_monthly_original (startdate);
