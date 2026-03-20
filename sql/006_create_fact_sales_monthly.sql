CREATE TABLE IF NOT EXISTS fact_sales_monthly (
  sales_sk BIGSERIAL PRIMARY KEY,
  sales_ck TEXT UNIQUE NOT NULL,
  dmdunit TEXT NOT NULL,
  dmdgroup TEXT NOT NULL,
  loc TEXT NOT NULL,
  startdate DATE NOT NULL,
  type INTEGER NOT NULL,
  qty_shipped NUMERIC(18,4),
  qty_ordered NUMERIC(18,4),
  qty NUMERIC(18,4),
  file_dt DATE,
  load_ts TIMESTAMPTZ DEFAULT NOW(),
  modified_ts TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT chk_fact_sales_monthly_type_1 CHECK (type = 1),
  CONSTRAINT chk_fact_sales_monthly_month_start CHECK (
    startdate = date_trunc('month', startdate)::date
  )
);

DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'fact_sales_monthly' AND column_name = 'u_qty_shipped'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'fact_sales_monthly' AND column_name = 'qty_shipped'
  ) THEN
    ALTER TABLE fact_sales_monthly RENAME COLUMN u_qty_shipped TO qty_shipped;
  END IF;

  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'fact_sales_monthly' AND column_name = 'u_qty_ordered'
  ) AND NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'fact_sales_monthly' AND column_name = 'qty_ordered'
  ) THEN
    ALTER TABLE fact_sales_monthly RENAME COLUMN u_qty_ordered TO qty_ordered;
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'chk_fact_sales_monthly_type_1'
      AND conrelid = 'fact_sales_monthly'::regclass
  ) THEN
    ALTER TABLE fact_sales_monthly
      ADD CONSTRAINT chk_fact_sales_monthly_type_1 CHECK (type = 1);
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'chk_fact_sales_monthly_month_start'
      AND conrelid = 'fact_sales_monthly'::regclass
  ) THEN
    ALTER TABLE fact_sales_monthly
      ADD CONSTRAINT chk_fact_sales_monthly_month_start CHECK (
        startdate = date_trunc('month', startdate)::date
      );
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_fact_sales_monthly_item ON fact_sales_monthly (dmdunit);
CREATE INDEX IF NOT EXISTS idx_fact_sales_monthly_loc ON fact_sales_monthly (loc);
CREATE INDEX IF NOT EXISTS idx_fact_sales_monthly_month ON fact_sales_monthly (startdate);
CREATE INDEX IF NOT EXISTS idx_fact_sales_monthly_type ON fact_sales_monthly (type);
