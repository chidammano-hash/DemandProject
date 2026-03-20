CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chat_embeddings (
    id BIGSERIAL PRIMARY KEY,
    domain_name TEXT NOT NULL,
    content_type TEXT NOT NULL,
    source_text TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_embeddings_domain
  ON chat_embeddings (domain_name);
CREATE INDEX IF NOT EXISTS idx_chat_embeddings_type
  ON chat_embeddings (content_type);
