-- 128_create_inventory_backtest.sql
-- Inventory Planning Backtest — simulates historical inventory outcomes per forecast algorithm.
-- Answers: "If we had used algorithm X for inventory planning, what would fill rate / stockouts have been?"

CREATE TABLE IF NOT EXISTS fact_inventory_backtest (
    id                      BIGSERIAL PRIMARY KEY,
    model_id                VARCHAR(100)    NOT NULL,   -- forecast algorithm evaluated
    item_id                 VARCHAR(50)     NOT NULL,   -- DFU item identity
    loc                     VARCHAR(50)     NOT NULL,   -- DFU location identity
    eval_month              DATE            NOT NULL,   -- historical month being evaluated
    -- Forecast vs actual
    forecast_qty            NUMERIC(12,2),              -- what the algorithm predicted
    actual_demand           NUMERIC(12,2),              -- what actually happened (sales)
    forecast_error          NUMERIC(12,2),              -- forecast - actual
    abs_error               NUMERIC(12,2),              -- |forecast - actual|
    -- Simulated inventory targets (what SS/ROP would have been)
    simulated_ss            NUMERIC(15,4),              -- safety stock computed from this forecast
    simulated_rop           NUMERIC(15,4),              -- reorder point computed from this forecast
    -- Actual inventory position
    actual_eom_on_hand      NUMERIC(12,2),              -- actual end-of-month on-hand
    actual_monthly_sales    NUMERIC(12,2),              -- actual monthly sales
    -- Simulated outcomes
    would_have_stocked_out  BOOLEAN DEFAULT FALSE,      -- would this SS have prevented stockout?
    simulated_fill_rate     NUMERIC(6,4),               -- 0..1 fill rate with this SS level
    excess_inventory        NUMERIC(12,2),              -- on_hand + SS - demand (positive = excess)
    -- Attributes (denormalized)
    abc_vol                 VARCHAR(10),                -- ABC class at eval time
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_inv_bt UNIQUE (model_id, item_id, loc, eval_month)
);

CREATE INDEX IF NOT EXISTS idx_inv_bt_model ON fact_inventory_backtest (model_id);
CREATE INDEX IF NOT EXISTS idx_inv_bt_month ON fact_inventory_backtest (eval_month);
CREATE INDEX IF NOT EXISTS idx_inv_bt_item_loc ON fact_inventory_backtest (item_id, loc);
CREATE INDEX IF NOT EXISTS idx_inv_bt_stockout
    ON fact_inventory_backtest (model_id, eval_month) WHERE would_have_stocked_out = TRUE;
