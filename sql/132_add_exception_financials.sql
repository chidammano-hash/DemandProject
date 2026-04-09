-- Issue #7: Add quantified financial impact columns to fact_replenishment_exceptions
-- Enables planners to see dollar impact per exception:
--   - loss_of_sales_7d / loss_of_sales_30d: potential lost revenue for stockout/below_ss/below_rop
--   - monthly_holding_cost: monthly carrying cost for excess exceptions
--   - financial_impact_total: primary metric for ranking (lost sales or holding cost)

ALTER TABLE fact_replenishment_exceptions
    ADD COLUMN IF NOT EXISTS unit_cost            NUMERIC(12,2),   -- item unit cost from fact_eoq_targets (or $10 default)
    ADD COLUMN IF NOT EXISTS unit_margin           NUMERIC(12,2),   -- unit_cost * 0.30 (assumed 30% margin)
    ADD COLUMN IF NOT EXISTS daily_demand_rate     NUMERIC(10,4),   -- demand_mean_monthly / 30.44
    ADD COLUMN IF NOT EXISTS loss_of_sales_7d      NUMERIC(12,2),   -- potential lost sales if not acted on in 7 days
    ADD COLUMN IF NOT EXISTS loss_of_sales_30d     NUMERIC(12,2),   -- potential lost sales if not acted on in 30 days
    ADD COLUMN IF NOT EXISTS monthly_holding_cost  NUMERIC(12,2),   -- for excess exceptions: monthly carrying cost
    ADD COLUMN IF NOT EXISTS financial_impact_total NUMERIC(12,2);  -- primary financial metric for ranking

-- Index for sorting/filtering by financial impact
CREATE INDEX IF NOT EXISTS idx_exceptions_financial_impact
    ON fact_replenishment_exceptions (financial_impact_total DESC NULLS LAST)
    WHERE status = 'open';
