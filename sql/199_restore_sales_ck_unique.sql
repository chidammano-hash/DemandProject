-- 199: Restore the UNIQUE constraint on fact_sales_monthly.sales_ck.
--
-- Reverses the first half of migration 172, which dropped
-- fact_sales_monthly_sales_ck_key because pg_stat_user_indexes reported
-- idx_scan = 0. That reading was misleading for exactly the reason 172 itself
-- documents when it decides to KEEP uq_forecast_ck_model: ON CONFLICT arbiter
-- lookups do not increment idx_scan. The delta loader
-- (scripts/etl/load.py::_safe_upsert -> _resolve_conflict_target) discovers
-- its ON CONFLICT target from the table's non-primary unique indexes, so with
-- the constraint gone every `--mode delta` sales load fails with
-- "no usable unique constraint on fact_sales_monthly for ON CONFLICT".
--
-- Defensive dedupe first: while the constraint was absent, duplicate sales_ck
-- rows could have been inserted (none observed on 2026-07-08, 305,585 rows
-- clean). Keep the newest row per sales_ck (highest sales_sk).
--
-- ALTER TABLE ADD CONSTRAINT briefly takes AccessExclusive; on a hot database
-- run during a maintenance window.

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_index ix
    JOIN pg_attribute a ON a.attrelid = ix.indrelid AND a.attnum = ANY(ix.indkey)
    WHERE ix.indrelid = 'fact_sales_monthly'::regclass
      AND ix.indisunique
      AND NOT ix.indisprimary
      AND a.attname = 'sales_ck'
  ) THEN
    DELETE FROM fact_sales_monthly f
    USING fact_sales_monthly d
    WHERE f.sales_ck = d.sales_ck
      AND f.sales_sk < d.sales_sk;

    ALTER TABLE fact_sales_monthly
      ADD CONSTRAINT uq_fact_sales_monthly_sales_ck UNIQUE (sales_ck);
  END IF;
END $$;
