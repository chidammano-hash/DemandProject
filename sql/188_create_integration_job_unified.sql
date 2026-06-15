-- US17b — Unified read surface for the integration UI.
--
-- The ingestion stack has two job ledgers:
--   * integration_job (sql/090) — single/chain ETL loads via IntegrationRunner.
--   * job_history     (sql/020) — JobManager jobs, incl. the etl_pipeline (US16)
--                                 and load_domain (US17c) ingestion job types.
--
-- This view merges both into one stream shaped like the integration_job `Job`
-- API model so GET /integration/jobs can show every ingestion run regardless of
-- which backend ran it — without losing legacy history. Read-only; submission
-- still writes to whichever backend owns the job (cutover is US17c/d).
--
-- Status vocabulary: the two ledgers diverge only on terminal success
-- (integration 'success' vs job_history 'completed'). The CASE below maps
-- job_history 'completed' -> 'success' to match the Python single source of
-- truth in common/services/job_shape.py (_JH_TO_INTEGRATION). Keep them in sync.
--
-- Cleanup: this is a VIEW (no rows of its own) — nothing to TRUNCATE. The base
-- tables integration_job / job_history are handled by their own cleanup entries.
-- Drop with `DROP VIEW IF EXISTS integration_job_unified;` if rebuilding.

CREATE OR REPLACE VIEW integration_job_unified AS
    -- (a) legacy IntegrationRunner rows (id cast to text to UNION with job_id)
    SELECT
        ij.id::text          AS id,
        ij.domain            AS domain,
        ij.mode              AS mode,
        ij.slice             AS slice,
        ij.file_path         AS file_path,
        ij.status            AS status,
        ij.rows_loaded       AS rows_loaded,
        ij.rows_inserted     AS rows_inserted,
        ij.rows_updated      AS rows_updated,
        ij.rows_deleted      AS rows_deleted,
        ij.error_message     AS error_message,
        ij.started_at        AS started_at,
        ij.completed_at      AS completed_at,
        ij.duration_ms       AS duration_ms,
        COALESCE(ij.triggered_by, 'api') AS triggered_by
    FROM integration_job ij

    UNION ALL

    -- (b) JobManager ingestion jobs, normalized to the integration Job shape
    SELECT
        jh.job_id            AS id,
        COALESCE(
            jh.params->>'domain',
            NULLIF(
                array_to_string(
                    ARRAY(SELECT jsonb_array_elements_text(jh.params->'domains')),
                    ','
                ),
                ''
            ),
            'pipeline'
        )                    AS domain,
        COALESCE(jh.params->>'mode', '')  AS mode,
        jh.params->>'slice'  AS slice,
        jh.params->>'file'   AS file_path,
        CASE WHEN jh.status = 'completed' THEN 'success' ELSE jh.status END AS status,
        COALESCE(
            (jh.result->>'rows_loaded')::int,
            (jh.result->>'loaded')::int,
            0
        )                    AS rows_loaded,
        (jh.result->>'rows_inserted')::int AS rows_inserted,
        (jh.result->>'rows_updated')::int  AS rows_updated,
        (jh.result->>'rows_deleted')::int  AS rows_deleted,
        jh.error             AS error_message,
        COALESCE(jh.started_at, jh.submitted_at) AS started_at,
        jh.completed_at      AS completed_at,
        CASE
            WHEN jh.completed_at IS NOT NULL AND jh.started_at IS NOT NULL
            THEN (EXTRACT(EPOCH FROM (jh.completed_at - jh.started_at)) * 1000)::int
        END                  AS duration_ms,
        COALESCE(jh.triggered_by, 'api') AS triggered_by
    FROM job_history jh
    WHERE jh.job_type IN ('etl_pipeline', 'load_domain');

COMMENT ON VIEW integration_job_unified IS
    'US17b: merged read surface over integration_job + job_history (ETL job '
    'types), normalized to the integration Job shape with completed->success.';
