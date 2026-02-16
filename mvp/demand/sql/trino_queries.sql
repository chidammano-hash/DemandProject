-- Item checks
SELECT count(*) AS item_cnt FROM iceberg.silver.dim_item;
SELECT * FROM iceberg.silver.dim_item LIMIT 20;

-- Location checks
SELECT count(*) AS location_cnt FROM iceberg.silver.dim_location;
SELECT * FROM iceberg.silver.dim_location LIMIT 20;

-- Customer checks
SELECT count(*) AS customer_cnt FROM iceberg.silver.dim_customer;
SELECT * FROM iceberg.silver.dim_customer LIMIT 20;

-- Time checks
SELECT count(*) AS time_cnt FROM iceberg.silver.dim_time;
SELECT * FROM iceberg.silver.dim_time LIMIT 20;

-- DFU checks
SELECT count(*) AS dfu_cnt FROM iceberg.silver.dim_dfu;
SELECT * FROM iceberg.silver.dim_dfu LIMIT 20;

-- Sales checks
SELECT count(*) AS sales_cnt FROM iceberg.silver.fact_sales_monthly;
SELECT * FROM iceberg.silver.fact_sales_monthly LIMIT 20;

-- Forecast checks
SELECT count(*) AS forecast_cnt FROM iceberg.silver.fact_external_forecast_monthly;
SELECT * FROM iceberg.silver.fact_external_forecast_monthly LIMIT 20;
