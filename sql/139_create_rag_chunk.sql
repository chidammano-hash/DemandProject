-- Gen-4 Roadmap Cross-cutting #9: pgvector-backed RAG chunks.
--
-- Replaces the legacy `chat_embeddings` table (sql/009) with a richer
-- schema supporting:
--   * mixed-source retrieval (docs, specs, runbooks, past decisions)
--   * hybrid search (vector cosine + BM25 via tsvector)
--   * JSONB metadata for filters (domain, doc_type, version)
--
-- Companion Python: common/ai/rag.py
--
-- Migration path for chat_embeddings (sql/009):
--   Kept in place for now. Rewriters should insert into rag_chunk and
--   gradually cut over reads. A future migration (>150) will drop
--   chat_embeddings once all writers have moved over.

CREATE EXTENSION IF NOT EXISTS vector;

-- Note: if the `vector` extension is unavailable (e.g. local dev without
-- pgvector), fall back to REAL[] by replacing the `embedding` column below
-- with `embedding REAL[]` and dropping the HNSW index.

CREATE TABLE IF NOT EXISTS rag_chunk (
    id              BIGSERIAL       PRIMARY KEY,
    doc_id          TEXT            NOT NULL,                 -- logical document id (spec path, runbook name, etc.)
    chunk_index     INTEGER         NOT NULL,                 -- ordinal within the parent doc
    source          TEXT            NOT NULL,                 -- origin: 'spec' | 'runbook' | 'chat' | 'decision' | ...
    text            TEXT            NOT NULL,                 -- raw chunk content
    embedding       VECTOR(1536),                             -- dense embedding (OpenAI 1536-dim default)
    metadata        JSONB           NOT NULL DEFAULT '{}'::jsonb,  -- {domain, doc_type, version, ...}
    ts_vector       TSVECTOR,                                 -- lexical search target (BM25-style via ts_rank)
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_rag_chunk_doc UNIQUE (doc_id, chunk_index)
);

-- HNSW is the fastest ANN index for high-recall cosine queries.
-- Falls back gracefully if the installed pgvector version is < 0.5.0
-- (then operators should switch to IVFFlat manually).
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS idx_rag_chunk_embedding_hnsw
        ON rag_chunk USING hnsw (embedding vector_cosine_ops);
EXCEPTION WHEN undefined_object THEN
    -- HNSW op family missing on older pgvector; ivfflat as fallback.
    CREATE INDEX IF NOT EXISTS idx_rag_chunk_embedding_ivf
        ON rag_chunk USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
END
$$;

CREATE INDEX IF NOT EXISTS idx_rag_chunk_tsv
    ON rag_chunk USING gin (ts_vector);

CREATE INDEX IF NOT EXISTS idx_rag_chunk_metadata
    ON rag_chunk USING gin (metadata);

CREATE INDEX IF NOT EXISTS idx_rag_chunk_source
    ON rag_chunk (source);

-- Keep ts_vector in sync with `text` automatically.
CREATE OR REPLACE FUNCTION rag_chunk_tsv_update() RETURNS TRIGGER AS $$
BEGIN
    NEW.ts_vector := to_tsvector('english', COALESCE(NEW.text, ''));
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_rag_chunk_tsv ON rag_chunk;
CREATE TRIGGER trg_rag_chunk_tsv
    BEFORE INSERT OR UPDATE OF text ON rag_chunk
    FOR EACH ROW EXECUTE FUNCTION rag_chunk_tsv_update();

COMMENT ON TABLE rag_chunk IS
    'Hybrid RAG store: dense embeddings + BM25 tsvector + JSONB metadata. '
    'Supersedes chat_embeddings (sql/009); see common/ai/rag.py.';
