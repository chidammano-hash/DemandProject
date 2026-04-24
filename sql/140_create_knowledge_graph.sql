-- Gen-4 Roadmap Cross-cutting #9: Knowledge graph companion to RAG.
--
-- Stores typed entities and relationships extracted from domain data
-- (items, DFUs, suppliers, exceptions, past decisions) so an agent can
-- traverse relationships instead of re-running vector search.
--
-- Companion Python: common/ai/rag.py (shared client, search helpers)

CREATE TABLE IF NOT EXISTS kg_node (
    id              BIGSERIAL       PRIMARY KEY,
    kind            VARCHAR(64)     NOT NULL,       -- e.g. 'item', 'dfu', 'supplier', 'event', 'decision'
    canonical_name  TEXT            NOT NULL,       -- normalized display name (used for fuzzy match)
    props           JSONB           NOT NULL DEFAULT '{}'::jsonb,  -- arbitrary properties
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_kg_node_kind_name UNIQUE (kind, canonical_name)
);

CREATE INDEX IF NOT EXISTS idx_kg_node_kind
    ON kg_node (kind);

CREATE INDEX IF NOT EXISTS idx_kg_node_props
    ON kg_node USING gin (props);

-- -----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS kg_edge (
    id              BIGSERIAL       PRIMARY KEY,
    src_id          BIGINT          NOT NULL REFERENCES kg_node(id) ON DELETE CASCADE,
    dst_id          BIGINT          NOT NULL REFERENCES kg_node(id) ON DELETE CASCADE,
    kind            VARCHAR(64)     NOT NULL,       -- e.g. 'supplied_by', 'substitutes_for', 'caused_by'
    props           JSONB           NOT NULL DEFAULT '{}'::jsonb,
    weight          REAL            NOT NULL DEFAULT 1.0,  -- edge strength (0-1 recommended)
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_kg_edge UNIQUE (src_id, dst_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_kg_edge_src
    ON kg_edge (src_id, kind);

CREATE INDEX IF NOT EXISTS idx_kg_edge_dst
    ON kg_edge (dst_id, kind);

CREATE INDEX IF NOT EXISTS idx_kg_edge_kind
    ON kg_edge (kind);

COMMENT ON TABLE kg_node IS 'Knowledge graph nodes. Partner of rag_chunk.';
COMMENT ON TABLE kg_edge IS 'Knowledge graph edges between kg_node rows.';
