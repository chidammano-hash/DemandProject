-- 066_create_collaboration.sql
-- Spec 08-05: Collaboration — annotations and shared views

BEGIN;

-- Annotations (threaded comments on any resource)
CREATE TABLE IF NOT EXISTS fact_annotation (
    annotation_id   BIGSERIAL       PRIMARY KEY,
    user_id         UUID,
    resource_type   TEXT            NOT NULL,
    resource_id     TEXT            NOT NULL,
    parent_id       BIGINT          REFERENCES fact_annotation(annotation_id),
    body            TEXT            NOT NULL,
    mentions        JSONB,
    is_resolved     BOOLEAN         DEFAULT FALSE,
    created_at      TIMESTAMPTZ     DEFAULT now(),
    updated_at      TIMESTAMPTZ     DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_annotation_resource
    ON fact_annotation (resource_type, resource_id);
CREATE INDEX IF NOT EXISTS idx_annotation_user
    ON fact_annotation (user_id);
CREATE INDEX IF NOT EXISTS idx_annotation_parent
    ON fact_annotation (parent_id);

-- Shared views (saved filter/layout snapshots)
CREATE TABLE IF NOT EXISTS fact_shared_view (
    view_id         BIGSERIAL       PRIMARY KEY,
    user_id         UUID,
    title           TEXT            NOT NULL,
    tab             TEXT,
    filters         JSONB,
    layout          JSONB,
    is_public       BOOLEAN         DEFAULT FALSE,
    created_at      TIMESTAMPTZ     DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_shared_view_user
    ON fact_shared_view (user_id);
CREATE INDEX IF NOT EXISTS idx_shared_view_public
    ON fact_shared_view (is_public) WHERE is_public = TRUE;

COMMIT;
