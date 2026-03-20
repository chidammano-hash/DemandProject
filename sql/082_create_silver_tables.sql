-- Silver Layer: typed, validated tables with DQ status tracking.
-- Same typed columns as gold tables + lineage/DQ metadata columns.
--
-- DDL1: UNIQUE constraints on *_ck fields enforce dedup at DB level
-- (Python-side DISTINCT ON still runs, but this is a safety net).
-- DDL2 NOTE: quarantine uses `quarantined_at`, lineage uses `created_at` —
-- naming inconsistency documented here for awareness.

-- ── item ──
CREATE TABLE IF NOT EXISTS silver_item (
    _silver_id      BIGSERIAL PRIMARY KEY,
    _bronze_id      BIGINT,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _dq_status      TEXT NOT NULL DEFAULT 'pending',
    _promoted_at    TIMESTAMPTZ,
    item_ck         TEXT NOT NULL,
    item_no TEXT NOT NULL, item_desc TEXT, item_status TEXT, brand_name TEXT,
    category TEXT, class TEXT, sub_class TEXT, country TEXT,
    scm_rtd_flag TEXT, size TEXT, case_weight NUMERIC, cpl INTEGER, cpp INTEGER,
    lpp INTEGER, case_weight_uom TEXT, bpc INTEGER, bottle_pack INTEGER,
    pack_case INTEGER, item_proof NUMERIC, upc TEXT, national_service_model TEXT,
    supplier_no TEXT, supplier_name TEXT, item_is_deleted TEXT, producer_name TEXT
);
CREATE INDEX IF NOT EXISTS idx_silver_item_batch ON silver_item (_load_batch_id);
CREATE INDEX IF NOT EXISTS idx_silver_item_dq ON silver_item (_dq_status);
CREATE INDEX IF NOT EXISTS idx_silver_item_ck ON silver_item (item_ck);
CREATE UNIQUE INDEX IF NOT EXISTS uq_silver_item_ck_batch ON silver_item (item_ck, _load_batch_id);

-- ── location ──
CREATE TABLE IF NOT EXISTS silver_location (
    _silver_id      BIGSERIAL PRIMARY KEY,
    _bronze_id      BIGINT,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _dq_status      TEXT NOT NULL DEFAULT 'pending',
    _promoted_at    TIMESTAMPTZ,
    location_ck     TEXT NOT NULL,
    location_id TEXT NOT NULL, site_id TEXT, site_desc TEXT, state_id TEXT,
    primary_demand_location TEXT
);
CREATE INDEX IF NOT EXISTS idx_silver_location_batch ON silver_location (_load_batch_id);
CREATE INDEX IF NOT EXISTS idx_silver_location_dq ON silver_location (_dq_status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_silver_location_ck_batch ON silver_location (location_ck, _load_batch_id);

-- ── customer ──
CREATE TABLE IF NOT EXISTS silver_customer (
    _silver_id      BIGSERIAL PRIMARY KEY,
    _bronze_id      BIGINT,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _dq_status      TEXT NOT NULL DEFAULT 'pending',
    _promoted_at    TIMESTAMPTZ,
    customer_ck     TEXT NOT NULL,
    site TEXT NOT NULL, customer_no TEXT NOT NULL, customer_name TEXT, city TEXT,
    state TEXT, zip TEXT, premise_code TEXT, status TEXT,
    license_name TEXT, store_type_desc TEXT, chain_type_desc TEXT,
    state_chain_name TEXT, corp_chain_name TEXT, rpt_channel_desc TEXT,
    rpt_sub_channel_desc TEXT, rpt_ship_type_desc TEXT,
    customer_acct_type_desc TEXT, delivery_freq_code TEXT
);
CREATE INDEX IF NOT EXISTS idx_silver_customer_batch ON silver_customer (_load_batch_id);
CREATE INDEX IF NOT EXISTS idx_silver_customer_dq ON silver_customer (_dq_status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_silver_customer_ck_batch ON silver_customer (customer_ck, _load_batch_id);

-- ── time ──
CREATE TABLE IF NOT EXISTS silver_time (
    _silver_id      BIGSERIAL PRIMARY KEY,
    _bronze_id      BIGINT,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _dq_status      TEXT NOT NULL DEFAULT 'pending',
    _promoted_at    TIMESTAMPTZ,
    time_ck         TEXT NOT NULL,
    date_key DATE NOT NULL, day_name TEXT, day_of_week INTEGER, day_of_month INTEGER,
    day_of_year INTEGER, iso_week_year INTEGER, iso_week INTEGER,
    week_start_date DATE, week_end_date DATE, month_number INTEGER,
    month_name TEXT, month_start_date DATE, month_end_date DATE,
    quarter_number INTEGER, quarter_label TEXT, quarter_start_date DATE,
    quarter_end_date DATE, year_number INTEGER, year_start_date DATE,
    year_end_date DATE, week_bucket TEXT, month_bucket TEXT,
    quarter_bucket TEXT, year_bucket TEXT
);
CREATE INDEX IF NOT EXISTS idx_silver_time_batch ON silver_time (_load_batch_id);
CREATE INDEX IF NOT EXISTS idx_silver_time_dq ON silver_time (_dq_status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_silver_time_ck_batch ON silver_time (time_ck, _load_batch_id);

-- ── dfu ──
CREATE TABLE IF NOT EXISTS silver_dfu (
    _silver_id      BIGSERIAL PRIMARY KEY,
    _bronze_id      BIGINT,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _dq_status      TEXT NOT NULL DEFAULT 'pending',
    _promoted_at    TIMESTAMPTZ,
    dfu_ck          TEXT NOT NULL,
    dmdunit TEXT NOT NULL, dmdgroup TEXT NOT NULL, loc TEXT NOT NULL,
    brand TEXT, abc_vol TEXT, brand_desc TEXT, ded_div_sw INTEGER,
    execution_lag INTEGER, otc_status TEXT, premise TEXT,
    prod_subgrp_desc TEXT, region TEXT, service_lvl_grp TEXT, size TEXT,
    state_plan TEXT, supergroup TEXT, supplier_desc TEXT, total_lt INTEGER,
    vintage INTEGER, sales_div TEXT, purge_sw INTEGER, alcoh_pct NUMERIC,
    bot_type_desc TEXT, brand_size TEXT, cnty TEXT, dom_imp_opt TEXT,
    grape_vrty_desc TEXT, material TEXT, prod_cat_desc TEXT,
    producer_desc TEXT, proof NUMERIC, subclass_desc TEXT,
    prod_class_desc TEXT, file_dt TEXT, histstart TEXT,
    cluster_assignment TEXT, ml_cluster TEXT, sop_ref TEXT,
    seasonality_profile TEXT, seasonality_strength NUMERIC,
    is_yearly_seasonal TEXT, peak_month INTEGER, trough_month INTEGER,
    peak_trough_ratio NUMERIC,
    demand_mean NUMERIC, demand_std NUMERIC, demand_cv NUMERIC,
    demand_mad NUMERIC, demand_p50 NUMERIC, demand_p90 NUMERIC,
    demand_skewness NUMERIC, demand_kurtosis NUMERIC,
    zero_demand_months INTEGER, total_demand_months INTEGER,
    intermittency_ratio NUMERIC, variability_class TEXT, demand_profile_ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_silver_dfu_batch ON silver_dfu (_load_batch_id);
CREATE INDEX IF NOT EXISTS idx_silver_dfu_dq ON silver_dfu (_dq_status);
CREATE INDEX IF NOT EXISTS idx_silver_dfu_ck ON silver_dfu (dfu_ck);
CREATE UNIQUE INDEX IF NOT EXISTS uq_silver_dfu_ck_batch ON silver_dfu (dfu_ck, _load_batch_id);

-- ── sales ──
CREATE TABLE IF NOT EXISTS silver_sales (
    _silver_id      BIGSERIAL PRIMARY KEY,
    _bronze_id      BIGINT,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _dq_status      TEXT NOT NULL DEFAULT 'pending',
    _promoted_at    TIMESTAMPTZ,
    _is_corrected   BOOLEAN NOT NULL DEFAULT FALSE,
    sales_ck        TEXT NOT NULL,
    dmdunit TEXT NOT NULL, dmdgroup TEXT NOT NULL, loc TEXT NOT NULL,
    startdate DATE NOT NULL, type INTEGER NOT NULL,
    qty_shipped NUMERIC(18,4), qty_ordered NUMERIC(18,4), qty NUMERIC(18,4),
    file_dt DATE,
    -- Snapshot of original values before DQ fixes (populated when _is_corrected = TRUE)
    _orig_qty_shipped NUMERIC(18,4),
    _orig_qty_ordered NUMERIC(18,4),
    _orig_qty         NUMERIC(18,4)
);
CREATE INDEX IF NOT EXISTS idx_silver_sales_batch ON silver_sales (_load_batch_id);
CREATE INDEX IF NOT EXISTS idx_silver_sales_dq ON silver_sales (_dq_status);
CREATE INDEX IF NOT EXISTS idx_silver_sales_ck ON silver_sales (sales_ck);
CREATE UNIQUE INDEX IF NOT EXISTS uq_silver_sales_ck_batch ON silver_sales (sales_ck, _load_batch_id);

-- ── forecast ──
CREATE TABLE IF NOT EXISTS silver_forecast (
    _silver_id      BIGSERIAL PRIMARY KEY,
    _bronze_id      BIGINT,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _dq_status      TEXT NOT NULL DEFAULT 'pending',
    _promoted_at    TIMESTAMPTZ,
    forecast_ck     TEXT NOT NULL,
    dmdunit TEXT NOT NULL, dmdgroup TEXT NOT NULL, loc TEXT NOT NULL,
    fcstdate DATE NOT NULL, startdate DATE NOT NULL,
    lag INTEGER, execution_lag INTEGER,
    basefcst_pref NUMERIC(18,4), tothist_dmd NUMERIC(18,4),
    model_id TEXT NOT NULL DEFAULT 'external'
);
CREATE INDEX IF NOT EXISTS idx_silver_forecast_batch ON silver_forecast (_load_batch_id);
CREATE INDEX IF NOT EXISTS idx_silver_forecast_dq ON silver_forecast (_dq_status);
CREATE INDEX IF NOT EXISTS idx_silver_forecast_ck ON silver_forecast (forecast_ck);
CREATE UNIQUE INDEX IF NOT EXISTS uq_silver_forecast_ck_batch ON silver_forecast (forecast_ck, _load_batch_id);

-- ── inventory ──
CREATE TABLE IF NOT EXISTS silver_inventory (
    _silver_id      BIGSERIAL PRIMARY KEY,
    _bronze_id      BIGINT,
    _load_batch_id  BIGINT NOT NULL REFERENCES audit_load_batch(batch_id),
    _dq_status      TEXT NOT NULL DEFAULT 'pending',
    _promoted_at    TIMESTAMPTZ,
    inventory_ck    TEXT NOT NULL,
    item_no TEXT NOT NULL, loc TEXT NOT NULL, snapshot_date DATE NOT NULL,
    lead_time_days NUMERIC, qty_on_hand NUMERIC, qty_on_hand_on_order NUMERIC,
    qty_on_order NUMERIC, mtd_sales NUMERIC
);
CREATE INDEX IF NOT EXISTS idx_silver_inventory_batch ON silver_inventory (_load_batch_id);
CREATE INDEX IF NOT EXISTS idx_silver_inventory_dq ON silver_inventory (_dq_status);
CREATE INDEX IF NOT EXISTS idx_silver_inventory_ck ON silver_inventory (inventory_ck);
CREATE UNIQUE INDEX IF NOT EXISTS uq_silver_inventory_ck_batch ON silver_inventory (inventory_ck, _load_batch_id);
