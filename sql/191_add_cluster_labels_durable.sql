-- Durable per-SKU cluster assignments.
--
-- Before this, a scenario's per-SKU cluster_labels.csv lived only under the
-- working scenario dir (/tmp/clustering_scenarios/<id>/). When /tmp was cleared
-- (reboot, OS cleanup), a completed experiment could no longer be re-promoted —
-- promote_scenario raised FileNotFoundError. Storing the labels (gzip-compressed)
-- on the experiment row makes every completed experiment self-contained and
-- re-promotable from the database alone, surviving reboots and `make clean-artifacts`.
ALTER TABLE cluster_experiment
    ADD COLUMN IF NOT EXISTS cluster_labels_gz BYTEA;

COMMENT ON COLUMN cluster_experiment.cluster_labels_gz IS
    'gzip-compressed cluster_labels.csv (sku_ck,cluster_label) — durable per-SKU '
    'assignments so a completed experiment can be re-promoted without the working '
    '/tmp artifacts. Populated on scenario completion; read by promote_scenario as '
    'a fallback when the working file is gone.';
