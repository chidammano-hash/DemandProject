-- Adds inserted/updated/deleted counts to integration_job so the UI can show
-- "X new + Y updated + Z deleted" instead of an opaque upsert-processed total.
-- Deltas/upserts will always show deleted=0 (upsert never deletes); File-slice
-- and Onetime modes can show non-zero deletes.
ALTER TABLE integration_job
    ADD COLUMN IF NOT EXISTS rows_inserted INTEGER,
    ADD COLUMN IF NOT EXISTS rows_updated INTEGER,
    ADD COLUMN IF NOT EXISTS rows_deleted INTEGER;
