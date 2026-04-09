-- 127_create_inventory_algorithm_comparison.sql
-- Multi-Algorithm Inventory Comparison — stores SS, EOQ, ROP computed from
-- each forecast algorithm's output in fact_production_forecast_staging.
-- Enables side-by-side comparison of how different forecast models affect
-- safety stock, EOQ, reorder point, and cycle stock targets.

CREATE TABLE IF NOT EXISTS fact_inventory_algorithm_comparison (
    id                      BIGSERIAL PRIMARY KEY,
    model_id                VARCHAR(100)    NOT NULL,   -- forecast algorithm (e.g. lgbm_cluster, nbeats)
    item_id                 VARCHAR(50)     NOT NULL,   -- DFU item
    loc                     VARCHAR(50)     NOT NULL,   -- DFU location
    forecast_avg_monthly    NUMERIC(15,4),              -- avg forecast_qty across horizon
    forecast_std_monthly    NUMERIC(15,4),              -- std derived from CI or forecast variance
    forecast_cv             NUMERIC(10,6),              -- coefficient of variation
    ss_combined             NUMERIC(15,4),              -- safety stock (combined variability)
    ss_demand_only          NUMERIC(15,4),              -- SS from demand variability only
    eoq                     NUMERIC(15,4),              -- economic order quantity
    effective_eoq           NUMERIC(15,4),              -- max(eoq, moq)
    reorder_point           NUMERIC(15,4),              -- avg_daily * lt_mean + ss_combined
    cycle_stock             NUMERIC(15,4),              -- effective_eoq / 2
    abc_vol                 VARCHAR(10),                -- ABC classification
    service_level           NUMERIC(6,4),               -- target service level used
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_algo_comp UNIQUE (model_id, item_id, loc)
);

CREATE INDEX IF NOT EXISTS idx_algo_comp_model ON fact_inventory_algorithm_comparison (model_id);
CREATE INDEX IF NOT EXISTS idx_algo_comp_item_loc ON fact_inventory_algorithm_comparison (item_id, loc);
