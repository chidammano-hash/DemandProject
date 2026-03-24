-- Performance Profiling Schema
-- Stores profiling run results, section metrics, query metrics, and suggestions
-- for historical trend analysis and regression detection.

-- ── Run-level summary ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS perf_run (
    run_id          SERIAL PRIMARY KEY,
    script_name     TEXT NOT NULL,
    mode            TEXT NOT NULL DEFAULT 'script',  -- script | api | pipeline | report
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    total_wall_s    NUMERIC(10,3),
    total_cpu_s     NUMERIC(10,3),
    peak_memory_mb  NUMERIC(10,1),
    total_queries   INTEGER DEFAULT 0,
    total_query_ms  NUMERIC(12,3) DEFAULT 0,
    suggestion_count INTEGER DEFAULT 0,
    report_json     JSONB,                           -- full PerfReport.to_dict()
    metadata        JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_perf_run_script   ON perf_run (script_name);
CREATE INDEX IF NOT EXISTS idx_perf_run_started  ON perf_run (started_at DESC);

-- ── Section-level metrics ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS perf_section (
    section_id      SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES perf_run(run_id) ON DELETE CASCADE,
    parent_id       INTEGER REFERENCES perf_section(section_id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    wall_time_s     NUMERIC(10,3),
    cpu_time_s      NUMERIC(10,3),
    memory_peak_mb  NUMERIC(10,1),
    memory_delta_mb NUMERIC(10,1),
    query_count     INTEGER DEFAULT 0,
    metadata        JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_perf_section_run  ON perf_section (run_id);
CREATE INDEX IF NOT EXISTS idx_perf_section_name ON perf_section (name);

-- ── Query-level metrics ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS perf_query (
    query_id        SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES perf_run(run_id) ON DELETE CASCADE,
    section_id      INTEGER REFERENCES perf_section(section_id) ON DELETE CASCADE,
    sql_preview     TEXT,
    duration_ms     NUMERIC(12,3),
    rows_affected   INTEGER,
    is_executemany  BOOLEAN DEFAULT false,
    captured_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_perf_query_run     ON perf_query (run_id);
CREATE INDEX IF NOT EXISTS idx_perf_query_slow    ON perf_query (duration_ms DESC);

-- ── Suggestions ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS perf_suggestion (
    suggestion_id   SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES perf_run(run_id) ON DELETE CASCADE,
    severity        TEXT NOT NULL,     -- critical | warning | info
    category        TEXT NOT NULL,     -- query | memory | cpu | pattern
    message         TEXT NOT NULL,
    section_name    TEXT,
    evidence        JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_perf_suggestion_run ON perf_suggestion (run_id);
CREATE INDEX IF NOT EXISTS idx_perf_suggestion_sev ON perf_suggestion (severity);

-- ── Trend view: avg wall time per script over last 30 runs ───────────────────
CREATE OR REPLACE VIEW v_perf_trend AS
SELECT
    script_name,
    COUNT(*)                          AS run_count,
    ROUND(AVG(total_wall_s), 2)       AS avg_wall_s,
    ROUND(AVG(total_cpu_s), 2)        AS avg_cpu_s,
    ROUND(AVG(peak_memory_mb), 1)     AS avg_peak_mb,
    ROUND(MIN(total_wall_s), 2)       AS best_wall_s,
    ROUND(MAX(total_wall_s), 2)       AS worst_wall_s,
    ROUND(AVG(total_queries), 0)      AS avg_queries,
    ROUND(AVG(suggestion_count), 1)   AS avg_suggestions,
    MAX(started_at)                   AS last_run
FROM perf_run
WHERE started_at > now() - INTERVAL '90 days'
GROUP BY script_name
ORDER BY avg_wall_s DESC;

-- ── Regression detection view: compare latest run vs 10-run avg ──────────────
CREATE OR REPLACE VIEW v_perf_regression AS
WITH ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY script_name ORDER BY started_at DESC) AS rn
    FROM perf_run
),
latest AS (
    SELECT * FROM ranked WHERE rn = 1
),
baseline AS (
    SELECT script_name,
        AVG(total_wall_s)    AS baseline_wall_s,
        AVG(peak_memory_mb)  AS baseline_memory_mb
    FROM ranked
    WHERE rn BETWEEN 2 AND 11
    GROUP BY script_name
)
SELECT
    l.script_name,
    l.total_wall_s              AS latest_wall_s,
    b.baseline_wall_s,
    ROUND(((l.total_wall_s - b.baseline_wall_s) / NULLIF(b.baseline_wall_s, 0)) * 100, 1)
                                AS wall_pct_change,
    l.peak_memory_mb            AS latest_memory_mb,
    b.baseline_memory_mb,
    ROUND(((l.peak_memory_mb - b.baseline_memory_mb) / NULLIF(b.baseline_memory_mb, 0)) * 100, 1)
                                AS memory_pct_change,
    l.suggestion_count,
    l.started_at                AS latest_run
FROM latest l
JOIN baseline b ON b.script_name = l.script_name
ORDER BY wall_pct_change DESC NULLS LAST;
