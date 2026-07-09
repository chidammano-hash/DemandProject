-- 198: Champion → cluster generation lineage.
--
-- champion_experiment previously carried no link to the cluster generation its
-- backtest data was computed under. Production generation joins the promoted
-- champion's winners CSV with the CURRENT promoted SKU cluster assignments and the current
-- data/models/*.pkl artifacts, with no way to detect that clustering had been
-- re-promoted since the champion ran (winners routing and per-cluster models
-- could silently belong to different generations).
--
-- This column records the promoted cluster_experiment at champion-experiment
-- creation time. scripts/forecasting/generate_production_forecasts.py refuses
-- to generate (unless --allow-cluster-mismatch) when the promoted champion's
-- cluster generation no longer matches the currently promoted one.

ALTER TABLE champion_experiment
    ADD COLUMN IF NOT EXISTS cluster_experiment_id INTEGER
        REFERENCES cluster_experiment(experiment_id) ON DELETE SET NULL;

COMMENT ON COLUMN champion_experiment.cluster_experiment_id IS
    'Promoted cluster_experiment at the time this champion experiment was created '
    '(generation lineage; NULL = legacy row created before sql/198).';
