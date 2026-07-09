-- 201: Remove deprecated dim_sku.ml_cluster.
--
-- Promoted ML cluster labels are durable lifecycle state and now live only in
-- sku_cluster_assignment/current_sku_cluster_assignment. This migration first
-- backfills from the legacy dim_sku.ml_cluster column when it exists, then drops
-- the physical column and rebuilds the canonical dim_sku DFU covering index
-- without that deprecated INCLUDE column.

DROP INDEX CONCURRENTLY IF EXISTS idx_dim_sku_ml_cluster;
DROP INDEX CONCURRENTLY IF EXISTS idx_dim_sku_dfu_triple;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'dim_sku'
          AND column_name = 'ml_cluster'
    ) THEN
        EXECUTE $backfill$
            INSERT INTO sku_cluster_assignment (
                experiment_id,
                sku_ck,
                item_id,
                customer_group,
                loc,
                cluster_label
            )
            SELECT
                e.experiment_id,
                d.sku_ck,
                d.item_id,
                d.customer_group,
                d.loc,
                d.ml_cluster
            FROM dim_sku d
            JOIN LATERAL (
                SELECT experiment_id
                FROM cluster_experiment
                WHERE is_promoted IS TRUE
                ORDER BY promoted_at DESC NULLS LAST, experiment_id DESC
                LIMIT 1
            ) e ON TRUE
            WHERE d.ml_cluster IS NOT NULL
            ON CONFLICT (experiment_id, sku_ck) DO UPDATE
            SET item_id = EXCLUDED.item_id,
                customer_group = EXCLUDED.customer_group,
                loc = EXCLUDED.loc,
                cluster_label = EXCLUDED.cluster_label,
                modified_ts = NOW()
            WHERE sku_cluster_assignment.item_id IS DISTINCT FROM EXCLUDED.item_id
               OR sku_cluster_assignment.customer_group IS DISTINCT FROM EXCLUDED.customer_group
               OR sku_cluster_assignment.loc IS DISTINCT FROM EXCLUDED.loc
               OR sku_cluster_assignment.cluster_label IS DISTINCT FROM EXCLUDED.cluster_label
        $backfill$;
    END IF;
END $$;

ALTER TABLE dim_sku DROP COLUMN IF EXISTS ml_cluster;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dim_sku_dfu_triple
ON dim_sku (item_id, customer_group, loc)
INCLUDE (abc_vol, region, brand_desc, seasonality_profile);
