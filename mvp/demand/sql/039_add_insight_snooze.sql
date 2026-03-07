-- PL-012: Add snooze support to ai_insights
-- Allows planners to defer an insight without accepting or resolving it.

ALTER TABLE ai_insights
    ADD COLUMN IF NOT EXISTS snoozed_until TIMESTAMPTZ;

-- Expand the status CHECK to include 'snoozed'
ALTER TABLE ai_insights
    DROP CONSTRAINT IF EXISTS ai_insights_status_check;

ALTER TABLE ai_insights
    ADD CONSTRAINT ai_insights_status_check
    CHECK (status IN ('open', 'acknowledged', 'resolved', 'snoozed'));

-- Auto-wake: index to efficiently find expired snooze records
CREATE INDEX IF NOT EXISTS idx_ai_insights_snoozed_until
    ON ai_insights (snoozed_until)
    WHERE status = 'snoozed';
