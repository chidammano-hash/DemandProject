-- F1.2 — Forward Inventory Projection
-- Day-by-day forward simulation of inventory position combining:
--   • current on-hand (fact_inventory_snapshot)
--   • ML demand forecast (fact_production_forecast)
--   • confirmed inbound POs (fact_open_purchase_orders)

-- ---------------------------------------------------------------------------
-- fact_inventory_projection
-- Grain: (projection_run_id, item_no, loc, scenario, projection_date)
-- Retention: only the most recent run per DFU is kept
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_inventory_projection (
    id                      BIGSERIAL PRIMARY KEY,
    projection_run_id       UUID            NOT NULL,
    item_no                 VARCHAR(50)     NOT NULL,
    loc                     VARCHAR(50)     NOT NULL,
    projection_date         DATE            NOT NULL,
    scenario                VARCHAR(30)     NOT NULL,   -- 'no_order', 'with_open_po', 'with_planned_orders'
    projected_qty           NUMERIC(12, 2)  NOT NULL,
    projected_dos           NUMERIC(8, 2),
    forecast_qty_consumed   NUMERIC(12, 2)  NOT NULL,
    receipts_expected       NUMERIC(12, 2)  NOT NULL DEFAULT 0,
    reorder_triggered       BOOLEAN         NOT NULL DEFAULT FALSE,
    stockout_risk           BOOLEAN         NOT NULL DEFAULT FALSE,
    excess_risk             BOOLEAN         NOT NULL DEFAULT FALSE,
    daily_demand_rate       NUMERIC(10, 4)  NOT NULL,
    forecast_source         VARCHAR(30)     NOT NULL DEFAULT 'production_forecast',
    plan_version            VARCHAR(30),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inv_proj_item_loc_scenario
    ON fact_inventory_projection (item_no, loc, scenario, projection_date);

CREATE INDEX IF NOT EXISTS idx_inv_proj_run_id
    ON fact_inventory_projection (projection_run_id);

CREATE INDEX IF NOT EXISTS idx_inv_proj_stockout
    ON fact_inventory_projection (item_no, loc, scenario, projection_date)
    WHERE stockout_risk = TRUE;

-- ---------------------------------------------------------------------------
-- mv_inventory_projection_summary
-- Derived summary: key dates per DFU per scenario per run
-- ---------------------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_inventory_projection_summary AS
SELECT
    p.item_no,
    p.loc,
    p.scenario,
    p.projection_run_id,
    p.plan_version,
    p.forecast_source,
    MIN(CASE WHEN p.reorder_triggered THEN p.projection_date END)   AS reorder_trigger_date,
    MIN(CASE WHEN p.stockout_risk     THEN p.projection_date END)   AS stockout_date,
    MIN(CASE WHEN p.excess_risk       THEN p.projection_date END)   AS excess_date,
    (MIN(CASE WHEN p.stockout_risk THEN p.projection_date END) - CURRENT_DATE)
                                                                    AS days_until_stockout,
    MAX(p.projected_qty)                                            AS max_projected_qty,
    MIN(p.projected_qty)                                            AS min_projected_qty,
    MAX(p.created_at)                                               AS last_computed_at
FROM fact_inventory_projection p
GROUP BY p.item_no, p.loc, p.scenario, p.projection_run_id, p.plan_version, p.forecast_source;

CREATE UNIQUE INDEX IF NOT EXISTS uq_mv_proj_summary
    ON mv_inventory_projection_summary (item_no, loc, scenario, projection_run_id);
