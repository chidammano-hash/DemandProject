-- Gen-4 Roadmap SC-2: Per-SKU holding/ordering cost override.
--
-- Adds nullable columns to dim_sku allowing planners to override the global
-- $50 ordering cost and 25% holding cost with SKU-level values.
-- EOQ and replenishment scripts read these when non-null, else fall back to
-- shared_constants.yaml globals.

ALTER TABLE dim_sku
    ADD COLUMN IF NOT EXISTS holding_cost_pct  NUMERIC(7, 4),   -- annual holding cost as fraction of unit_cost (e.g. 0.25 = 25%)
    ADD COLUMN IF NOT EXISTS ordering_cost     NUMERIC(12, 4);  -- fixed $ cost per order (overrides global $50 default)

COMMENT ON COLUMN dim_sku.holding_cost_pct IS
    'Gen-4 SC-2: per-SKU override for annual holding cost rate; NULL falls back to shared_constants.carrying_cost_annual_pct';
COMMENT ON COLUMN dim_sku.ordering_cost IS
    'Gen-4 SC-2: per-SKU override for fixed ordering cost per PO; NULL falls back to global default';

-- Partial index for planners looking up overridden SKUs
CREATE INDEX IF NOT EXISTS idx_dim_sku_cost_override
    ON dim_sku (item_id, loc)
    WHERE holding_cost_pct IS NOT NULL OR ordering_cost IS NOT NULL;
