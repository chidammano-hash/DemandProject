-- Tracks backtest runs for all models (tunable and non-tunable).
-- Provides version history so users can pick which run to load into DB.
CREATE TABLE IF NOT EXISTS backtest_run (
    id SERIAL PRIMARY KEY,
    model_id TEXT NOT NULL,
    job_id TEXT,
    status TEXT NOT NULL DEFAULT 'queued',
    accuracy_pct NUMERIC,
    wape NUMERIC,
    bias NUMERIC,
    n_predictions INTEGER,
    n_dfus INTEGER,
    n_timeframes INTEGER,
    metadata JSONB,
    is_loaded_to_db BOOLEAN NOT NULL DEFAULT FALSE,
    loaded_at TIMESTAMPTZ,
    load_job_id TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_backtest_run_model_id ON backtest_run (model_id);
CREATE INDEX IF NOT EXISTS idx_backtest_run_status ON backtest_run (status);
