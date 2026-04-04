-- Customer-derived features aggregated to item × location × month grain.
-- Used by customer-enriched tree models and bolt hierarchical backtest.
-- Source: fact_customer_demand_monthly + dim_customer.

CREATE TABLE IF NOT EXISTS customer_features_monthly (
    item_id         TEXT NOT NULL,
    loc             TEXT NOT NULL,
    startdate       DATE NOT NULL,
    -- Concentration (6)
    n_active_cust           REAL DEFAULT 0,
    n_active_cust_6m        REAL DEFAULT 0,
    hhi_demand              REAL DEFAULT 0,
    top1_cust_share         REAL DEFAULT 0,
    top3_cust_share         REAL DEFAULT 0,
    cust_gini               REAL DEFAULT 0,
    -- Dynamics (5)
    new_cust_demand_share   REAL DEFAULT 0,
    churned_cust_demand_share REAL DEFAULT 0,
    cust_count_mom          REAL DEFAULT 0,
    cust_retention_rate     REAL DEFAULT 0,
    cust_tenure_mean        REAL DEFAULT 0,
    -- True Demand (7)
    true_demand_ratio       REAL DEFAULT 0,
    oos_rate                REAL DEFAULT 0,
    oos_cust_pct            REAL DEFAULT 0,
    demand_sales_gap_3m     REAL DEFAULT 0,
    oos_trend               REAL DEFAULT 0,
    demand_qty_lag1         REAL DEFAULT 0,
    demand_qty_lag3_mean    REAL DEFAULT 0,
    -- Channel Mix (4)
    channel_entropy         REAL DEFAULT 0,
    dominant_channel_share  REAL DEFAULT 0,
    channel_mix_shift       REAL DEFAULT 0,
    on_premise_share        REAL DEFAULT 0,
    -- Cross-Customer (3)
    cust_demand_cv_mean     REAL DEFAULT 0,
    cust_demand_sync        REAL DEFAULT 0,
    max_cust_share_delta    REAL DEFAULT 0,
    -- Customer Attribute Mix (7)
    store_type_entropy      REAL DEFAULT 0,
    dominant_store_type_share REAL DEFAULT 0,
    chain_ratio             REAL DEFAULT 0,
    top_chain_share         REAL DEFAULT 0,
    sub_channel_entropy     REAL DEFAULT 0,
    active_cust_pct         REAL DEFAULT 0,
    avg_delivery_freq       REAL DEFAULT 0,
    on_premise_acct_share   REAL DEFAULT 0,
    premise_diversity       REAL DEFAULT 0,
    -- Metadata
    load_ts TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT pk_cust_feat_monthly PRIMARY KEY (item_id, loc, startdate)
);

CREATE INDEX IF NOT EXISTS idx_cust_feat_item_loc
    ON customer_features_monthly (item_id, loc);
CREATE INDEX IF NOT EXISTS idx_cust_feat_startdate
    ON customer_features_monthly (startdate);
