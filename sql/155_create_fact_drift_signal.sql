-- 155_create_fact_drift_signal.sql
-- Gen-4 Stream G — Governance / AI-9: Drift detection signal table.
--
-- Every drift detector run (PSI on input features, rolling-WAPE on
-- predictions, etc.) appends one row here per (model_id, metric). Auto-retrain
-- triggers subscribe to `threshold_breached = TRUE` rows via notification
-- or nightly scan.

CREATE TABLE IF NOT EXISTS fact_drift_signal (
    id                  BIGSERIAL PRIMARY KEY,

    -- Model under observation (matches fact_candidate_forecast.model_id)
    model_id            VARCHAR(100)    NOT NULL,

    -- Metric name: 'psi_<feature>', 'rolling_wape', 'coverage', etc.
    metric              VARCHAR(80)     NOT NULL,

    -- Observed scalar value for the metric window
    value               NUMERIC(16, 6)  NOT NULL,

    -- Baseline value (population reference, e.g. training-time PSI baseline)
    baseline            NUMERIC(16, 6),

    -- Threshold used for breach decision (domain-specific; PSI 0.2 standard)
    threshold           NUMERIC(16, 6),

    -- TRUE when the detector considers this observation a drift breach
    threshold_breached  BOOLEAN         NOT NULL DEFAULT FALSE,

    -- Observation window description (free text, e.g. '2026-01 vs 2026-02')
    window_label        TEXT,

    -- Free-form detector metadata (bins, sample sizes, extra stats)
    details             JSONB,

    -- When this signal was written
    ts                  TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Primary query: latest signals per model for a metric.
CREATE INDEX IF NOT EXISTS idx_drift_signal_model_metric_ts
    ON fact_drift_signal (model_id, metric, ts DESC);

-- Filter breaches across all models for the auto-retrain sweep.
CREATE INDEX IF NOT EXISTS idx_drift_signal_breached
    ON fact_drift_signal (threshold_breached, ts DESC)
    WHERE threshold_breached;
