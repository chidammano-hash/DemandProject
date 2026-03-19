-- Bronze Layer: immutable raw data tables (all TEXT columns, append-only).
-- One table per domain. Each row tagged with load_batch_id + source_row_num.

-- ── item ──
CREATE TABLE IF NOT EXISTS bronze_item (
    _bronze_id      BIGSERIAL PRIMARY KEY,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _source_row_num BIGINT,
    _ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    item_no TEXT, item_desc TEXT, item_status TEXT, brand_name TEXT,
    category TEXT, class TEXT, sub_class TEXT, country TEXT,
    scm_rtd_flag TEXT, size TEXT, case_weight TEXT, cpl TEXT, cpp TEXT,
    lpp TEXT, case_weight_uom TEXT, bpc TEXT, bottle_pack TEXT,
    pack_case TEXT, item_proof TEXT, upc TEXT, national_service_model TEXT,
    supplier_no TEXT, supplier_name TEXT, item_is_deleted TEXT, producer_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_bronze_item_batch ON bronze_item (_load_batch_id);

-- ── location ──
CREATE TABLE IF NOT EXISTS bronze_location (
    _bronze_id      BIGSERIAL PRIMARY KEY,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _source_row_num BIGINT,
    _ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    location_id TEXT, site_id TEXT, site_desc TEXT, state_id TEXT,
    primary_demand_location TEXT
);
CREATE INDEX IF NOT EXISTS idx_bronze_location_batch ON bronze_location (_load_batch_id);

-- ── customer ──
CREATE TABLE IF NOT EXISTS bronze_customer (
    _bronze_id      BIGSERIAL PRIMARY KEY,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _source_row_num BIGINT,
    _ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    site TEXT, customer_no TEXT, customer_name TEXT, city TEXT,
    state TEXT, zip TEXT, premise_code TEXT, status TEXT,
    license_name TEXT, store_type_desc TEXT, chain_type_desc TEXT,
    state_chain_name TEXT, corp_chain_name TEXT, rpt_channel_desc TEXT,
    rpt_sub_channel_desc TEXT, rpt_ship_type_desc TEXT,
    customer_acct_type_desc TEXT, delivery_freq_code TEXT
);
CREATE INDEX IF NOT EXISTS idx_bronze_customer_batch ON bronze_customer (_load_batch_id);

-- ── time ──
CREATE TABLE IF NOT EXISTS bronze_time (
    _bronze_id      BIGSERIAL PRIMARY KEY,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _source_row_num BIGINT,
    _ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    date_key TEXT, day_name TEXT, day_of_week TEXT, day_of_month TEXT,
    day_of_year TEXT, iso_week_year TEXT, iso_week TEXT,
    week_start_date TEXT, week_end_date TEXT, month_number TEXT,
    month_name TEXT, month_start_date TEXT, month_end_date TEXT,
    quarter_number TEXT, quarter_label TEXT, quarter_start_date TEXT,
    quarter_end_date TEXT, year_number TEXT, year_start_date TEXT,
    year_end_date TEXT, week_bucket TEXT, month_bucket TEXT,
    quarter_bucket TEXT, year_bucket TEXT
);
CREATE INDEX IF NOT EXISTS idx_bronze_time_batch ON bronze_time (_load_batch_id);

-- ── dfu ──
CREATE TABLE IF NOT EXISTS bronze_dfu (
    _bronze_id      BIGSERIAL PRIMARY KEY,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _source_row_num BIGINT,
    _ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    dmdunit TEXT, dmdgroup TEXT, loc TEXT, brand TEXT, abc_vol TEXT,
    brand_desc TEXT, ded_div_sw TEXT, execution_lag TEXT, otc_status TEXT,
    premise TEXT, prod_subgrp_desc TEXT, region TEXT, service_lvl_grp TEXT,
    size TEXT, state_plan TEXT, supergroup TEXT, supplier_desc TEXT,
    total_lt TEXT, vintage TEXT, sales_div TEXT, purge_sw TEXT,
    alcoh_pct TEXT, bot_type_desc TEXT, brand_size TEXT, cnty TEXT,
    dom_imp_opt TEXT, grape_vrty_desc TEXT, material TEXT,
    prod_cat_desc TEXT, producer_desc TEXT, proof TEXT,
    subclass_desc TEXT, prod_class_desc TEXT, file_dt TEXT,
    histstart TEXT, cluster_assignment TEXT, ml_cluster TEXT,
    sop_ref TEXT, seasonality_profile TEXT, seasonality_strength TEXT,
    is_yearly_seasonal TEXT, peak_month TEXT, trough_month TEXT,
    peak_trough_ratio TEXT,
    demand_mean TEXT, demand_std TEXT, demand_cv TEXT, demand_mad TEXT,
    demand_p50 TEXT, demand_p90 TEXT, demand_skewness TEXT,
    demand_kurtosis TEXT, zero_demand_months TEXT, total_demand_months TEXT,
    intermittency_ratio TEXT, variability_class TEXT, demand_profile_ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_bronze_dfu_batch ON bronze_dfu (_load_batch_id);

-- ── sales ──
CREATE TABLE IF NOT EXISTS bronze_sales (
    _bronze_id      BIGSERIAL PRIMARY KEY,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _source_row_num BIGINT,
    _ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    dmdunit TEXT, dmdgroup TEXT, loc TEXT, startdate TEXT, type TEXT,
    qty_shipped TEXT, qty_ordered TEXT, qty TEXT, file_dt TEXT
);
CREATE INDEX IF NOT EXISTS idx_bronze_sales_batch ON bronze_sales (_load_batch_id);

-- ── forecast ──
CREATE TABLE IF NOT EXISTS bronze_forecast (
    _bronze_id      BIGSERIAL PRIMARY KEY,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _source_row_num BIGINT,
    _ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    dmdunit TEXT, dmdgroup TEXT, loc TEXT, fcstdate TEXT, startdate TEXT,
    lag TEXT, execution_lag TEXT, basefcst_pref TEXT, tothist_dmd TEXT,
    model_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_bronze_forecast_batch ON bronze_forecast (_load_batch_id);

-- ── inventory ──
CREATE TABLE IF NOT EXISTS bronze_inventory (
    _bronze_id      BIGSERIAL PRIMARY KEY,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _source_row_num BIGINT,
    _ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    item_no TEXT, loc TEXT, snapshot_date TEXT, lead_time_days TEXT,
    qty_on_hand TEXT, qty_on_hand_on_order TEXT, qty_on_order TEXT,
    mtd_sales TEXT
);
CREATE INDEX IF NOT EXISTS idx_bronze_inventory_batch ON bronze_inventory (_load_batch_id);
