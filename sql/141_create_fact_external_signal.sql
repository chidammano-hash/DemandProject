-- Gen-4 Roadmap Cross-cutting #3: Causal elasticity input store.
--
-- External signals (macro indicators, weather, competitor price, promo
-- calendars, marketing spend) land here so `scripts/ml/fit_elasticity.py`
-- can fit elasticities on demand.
--
-- Partitioned RANGE by event_ts (monthly). This migration creates:
--   * the partitioned parent
--   * a default partition for anything outside monthly bounds
--   * one concrete monthly partition as a worked example
-- Monthly partitions are created on demand by the signal-load pipeline
-- (add Make target once ingestion is wired).

CREATE TABLE IF NOT EXISTS fact_external_signal (
    id              BIGSERIAL       NOT NULL,
    signal_source   VARCHAR(64)     NOT NULL,       -- e.g. 'noaa_weather', 'google_trends', 'competitor_pricing'
    signal_kind     VARCHAR(64)     NOT NULL,       -- e.g. 'price_index', 'temp_anomaly', 'promo_flag'
    item_id         VARCHAR(50),                    -- NULL if market-wide
    loc             VARCHAR(50),                    -- NULL if nation-wide
    event_ts        TIMESTAMPTZ     NOT NULL,       -- partition key
    value           JSONB           NOT NULL DEFAULT '{}'::jsonb,  -- {numeric, unit, ...}
    loaded_at       TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, event_ts)
) PARTITION BY RANGE (event_ts);

CREATE INDEX IF NOT EXISTS idx_ext_signal_item_ts
    ON fact_external_signal (item_id, event_ts DESC);

CREATE INDEX IF NOT EXISTS idx_ext_signal_source_kind
    ON fact_external_signal (signal_source, signal_kind, event_ts DESC);

-- Default partition for dates outside explicit monthly partitions.
CREATE TABLE IF NOT EXISTS fact_external_signal_default
    PARTITION OF fact_external_signal DEFAULT;

-- One example monthly partition (2026-01). The ingestion pipeline
-- (to be added) should call `create_monthly_partition(event_ts)` before
-- inserting rows for a new month.
CREATE TABLE IF NOT EXISTS fact_external_signal_2026_01
    PARTITION OF fact_external_signal
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

-- -----------------------------------------------------------------------
-- Learned elasticity output table.
-- `scripts/ml/fit_elasticity.py` INSERTs into this.

CREATE TABLE IF NOT EXISTS fact_causal_elasticity (
    id              BIGSERIAL       PRIMARY KEY,
    item_id         VARCHAR(50),                    -- NULL = aggregate / category-level
    loc             VARCHAR(50),
    feature         VARCHAR(128)    NOT NULL,       -- e.g. 'price_log', 'promo_flag', 'temp_anomaly'
    coef            DOUBLE PRECISION NOT NULL,      -- elasticity / uplift coefficient
    p_value         DOUBLE PRECISION,
    std_err         DOUBLE PRECISION,
    n_obs           INTEGER,
    method          VARCHAR(64)     NOT NULL DEFAULT 'linear_regression',  -- 'linear_regression' | 'dml_econml' | ...
    run_id          VARCHAR(64),
    computed_at     TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_causal_elasticity_item_feature
    ON fact_causal_elasticity (item_id, feature, computed_at DESC);

CREATE INDEX IF NOT EXISTS idx_causal_elasticity_feature
    ON fact_causal_elasticity (feature, computed_at DESC);

-- -----------------------------------------------------------------------
-- `dim_event.uplift_pct` deprecation path.
-- The legacy column (sql/057 fact_event_calendar.uplift_pct, sql/048
-- consensus_plan.uplift_pct) stays writable. A view prefers learned
-- elasticity when available, falling back to the manual uplift.
--
-- Consumers should read v_event_uplift_effective instead of
-- fact_event_calendar.uplift_pct directly.

CREATE OR REPLACE VIEW v_event_uplift_effective AS
SELECT
    e.event_id,
    e.event_type,
    e.event_name,
    e.event_start,
    e.event_end,
    e.uplift_pct                                        AS manual_uplift_pct,
    -- Latest learned elasticity for `promo_flag` on this item_id (if any).
    -- NULL when no learned value exists yet.
    ce.coef                                             AS learned_uplift_pct,
    COALESCE(ce.coef, e.uplift_pct)                     AS effective_uplift_pct,
    CASE WHEN ce.coef IS NOT NULL THEN 'learned' ELSE 'manual' END AS source
FROM fact_event_calendar e
LEFT JOIN LATERAL (
    SELECT coef
    FROM fact_causal_elasticity c
    WHERE c.feature = 'promo_flag'
      AND (c.item_id IS NULL OR c.item_id = ANY(
              SELECT jsonb_array_elements_text(COALESCE(e.target_items, '[]'::jsonb))
          ))
    ORDER BY c.computed_at DESC
    LIMIT 1
) ce ON TRUE;

COMMENT ON VIEW v_event_uplift_effective IS
    'Effective promo uplift: prefers learned elasticity (fact_causal_elasticity) '
    'over manual fact_event_calendar.uplift_pct. Read this in downstream '
    'forecast adjusters instead of fact_event_calendar.uplift_pct.';
