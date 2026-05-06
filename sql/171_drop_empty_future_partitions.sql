-- 171: Drop empty future partitions of fact_customer_demand_monthly.
--
-- Background:
--   sql/110_create_fact_customer_demand_monthly.sql provisions partitions
--   2024-01 .. 2026-12 up front. As of 2026-05, partitions 2026-04 .. 2026-12
--   are empty (no demand data has landed for those future months yet) but the
--   query planner still considers them on every plan, paying constraint-
--   exclusion overhead and slightly larger plan sizes for no benefit.
--
--   Dropping the empty future partitions reduces planner work and clears the
--   way for the auto-create-partitions cron (sql/173 / scripts/db/auto_create_partitions.py)
--   to recreate them on a rolling window basis.
--
-- Safety:
--   Each DROP is guarded by IF EXISTS, so this migration is idempotent. If
--   any of these partitions has actually been backfilled in the meantime the
--   DROP will fail because data exists -- in that case skip the corresponding
--   line. Verify emptiness first if running on a non-dev environment:
--
--     SELECT relname,
--            (SELECT count(*) FROM ONLY pg_class.relname::regclass) AS rows
--     FROM pg_class
--     WHERE relname LIKE 'fact_customer_demand_monthly_2026_%';
--
-- Recreation:
--   The `make auto-create-partitions` target (added in this batch) runs
--   `scripts/db/auto_create_partitions.py`, which recreates the next 12
--   monthly partitions idempotently on every invocation. Schedule it monthly
--   via cron, or run it manually before any large customer-demand backfill.

DROP TABLE IF EXISTS fact_customer_demand_monthly_2026_04;
DROP TABLE IF EXISTS fact_customer_demand_monthly_2026_05;
DROP TABLE IF EXISTS fact_customer_demand_monthly_2026_06;
DROP TABLE IF EXISTS fact_customer_demand_monthly_2026_07;
DROP TABLE IF EXISTS fact_customer_demand_monthly_2026_08;
DROP TABLE IF EXISTS fact_customer_demand_monthly_2026_09;
DROP TABLE IF EXISTS fact_customer_demand_monthly_2026_10;
DROP TABLE IF EXISTS fact_customer_demand_monthly_2026_11;
DROP TABLE IF EXISTS fact_customer_demand_monthly_2026_12;
