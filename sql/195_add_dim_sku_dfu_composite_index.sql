-- 195: dim_sku composite covering index for the canonical 3-key DFU join
-- ============================================================================
-- Rationale (P0-5, docs/architecture/SCALING_ASSESSMENT_50x.md):
--   The CLAUDE.md-canonical join grain is dim_sku.(item_id, customer_group, loc)
--   — customer_group is NOT unique per (item_id, loc), so accuracy / FVA /
--   budget endpoints MUST join on all three keys (see
--   api/routers/forecasting/accuracy.py lines ~228-229, 340-341, 545-546:
--   `JOIN dim_sku d ON f.item_id = d.item_id AND f.customer_group = d.customer_group
--    AND f.loc = d.loc`).
--
--   Before this index dim_sku had only single-column indexes (item_id, loc),
--   so the planner either hash-joined the whole dimension or did a single-col
--   index scan + filter on the other two keys. At 50x (≈16M dim rows) that
--   degrades the accuracy/FVA endpoints AND the MV refreshes that join the
--   forecast facts to dim_sku.
--
--   INCLUDE columns are the dim_sku attributes those same endpoints project /
--   filter on after the join (see accuracy.py GROUP_FIELDS + segment filters:
--   abc_vol, region, brand_desc, seasonality_profile). Promoted ML cluster labels
--   live in sku_cluster_assignment/current_sku_cluster_assignment and are indexed
--   there. Carrying dim_sku attributes in the index leaf makes it COVERING for the
--   common segment-filtered accuracy query — the heap fetch is avoided
--   (index-only scan eligible).
--
-- CONCURRENTLY: the migration runner (Makefile `db-apply-sql`) pipes each file
--   through `psql ... < file` with NO --single-transaction / -1 flag and no
--   BEGIN/COMMIT in this file, so every statement auto-commits on its own.
--   That is the required environment for CREATE INDEX CONCURRENTLY (it cannot
--   run inside an explicit transaction block). This mirrors sql/117. Building
--   CONCURRENTLY also avoids an ACCESS EXCLUSIVE lock on the live dim_sku
--   table during the build. IF NOT EXISTS keeps the migration idempotent.
-- ============================================================================

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dim_sku_dfu_triple
ON dim_sku (item_id, customer_group, loc)
INCLUDE (abc_vol, region, brand_desc, seasonality_profile);
