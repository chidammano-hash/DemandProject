CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Core fact indexes for exact item/location/date filters and sort paths.
CREATE INDEX IF NOT EXISTS idx_fact_sales_item_loc_startdate
  ON fact_sales_monthly (dmdunit, loc, startdate);
CREATE INDEX IF NOT EXISTS idx_fact_sales_startdate
  ON fact_sales_monthly (startdate);
CREATE INDEX IF NOT EXISTS idx_fact_forecast_item_loc_startdate_fcstdate
  ON fact_external_forecast_monthly (dmdunit, loc, startdate, fcstdate);
CREATE INDEX IF NOT EXISTS idx_fact_forecast_fcstdate
  ON fact_external_forecast_monthly (fcstdate);

-- Trigram indexes for ILIKE-heavy search fields used by the API/UI.
-- Dimension tables:
CREATE INDEX IF NOT EXISTS idx_dim_item_item_desc_trgm
  ON dim_item USING gin (item_desc gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_dim_item_brand_name_trgm
  ON dim_item USING gin (brand_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_dim_location_site_desc_trgm
  ON dim_location USING gin (site_desc gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_dim_customer_customer_name_trgm
  ON dim_customer USING gin (customer_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_dim_dfu_brand_desc_trgm
  ON dim_dfu USING gin (brand_desc gin_trgm_ops);

-- Fact table text columns — enables fast ILIKE substring filters on large tables.
CREATE INDEX IF NOT EXISTS idx_fact_forecast_model_id_trgm
  ON fact_external_forecast_monthly USING gin (model_id gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fact_forecast_dmdunit_trgm
  ON fact_external_forecast_monthly USING gin (dmdunit gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fact_forecast_loc_trgm
  ON fact_external_forecast_monthly USING gin (loc gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fact_forecast_dmdgroup_trgm
  ON fact_external_forecast_monthly USING gin (dmdgroup gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fact_sales_dmdunit_trgm
  ON fact_sales_monthly USING gin (dmdunit gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fact_sales_loc_trgm
  ON fact_sales_monthly USING gin (loc gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fact_sales_dmdgroup_trgm
  ON fact_sales_monthly USING gin (dmdgroup gin_trgm_ops);

-- Monthly aggregates for fast trend analytics on fact domains.
CREATE MATERIALIZED VIEW IF NOT EXISTS agg_sales_monthly AS
SELECT
  date_trunc('month', startdate)::date AS month_start,
  dmdunit,
  loc,
  count(*)::bigint AS row_count,
  coalesce(sum(qty_shipped), 0)::double precision AS qty_shipped,
  coalesce(sum(qty_ordered), 0)::double precision AS qty_ordered,
  coalesce(sum(qty), 0)::double precision AS qty
FROM fact_sales_monthly
GROUP BY 1, 2, 3
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_agg_sales_monthly_pk
  ON agg_sales_monthly (dmdunit, loc, month_start);
CREATE INDEX IF NOT EXISTS idx_agg_sales_monthly_month
  ON agg_sales_monthly (month_start);

CREATE MATERIALIZED VIEW IF NOT EXISTS agg_forecast_monthly AS
SELECT
  date_trunc('month', startdate)::date AS month_start,
  dmdunit,
  loc,
  model_id,
  count(*)::bigint AS row_count,
  coalesce(sum(basefcst_pref), 0)::double precision AS basefcst_pref,
  coalesce(sum(tothist_dmd), 0)::double precision AS tothist_dmd
FROM fact_external_forecast_monthly
GROUP BY 1, 2, 3, 4
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_agg_forecast_monthly_pk
  ON agg_forecast_monthly (dmdunit, loc, month_start, model_id);
CREATE INDEX IF NOT EXISTS idx_agg_forecast_monthly_month
  ON agg_forecast_monthly (month_start);
CREATE INDEX IF NOT EXISTS idx_agg_forecast_monthly_model
  ON agg_forecast_monthly (model_id);
