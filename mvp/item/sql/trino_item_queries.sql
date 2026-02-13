-- Run from Trino CLI or UI
SHOW CATALOGS;
SHOW SCHEMAS FROM iceberg;
SHOW TABLES FROM iceberg.silver;

SELECT item_no, item_desc, brand_name, category, class, sub_class, country
FROM iceberg.silver.dim_item
LIMIT 100;
