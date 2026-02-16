# Feature 12: Chatbot / Natural Language Queries (Phase 2)

## Objective
Allow users to ask questions in plain English ("What's the accuracy for item 100320 in Q4?") and receive answers with data from the Demand Studio database.

## Architecture

```
User Question
      ↓
OpenAI text-embedding-3-small → embed question
      ↓
pgvector cosine similarity search → top-10 relevant schema chunks
      ↓
GPT-4o prompt (system: schema + context + rules, user: question)
      ↓
Parse response → extract answer + SQL
      ↓
Validate SQL (SELECT only) → execute read-only with timeout
      ↓
Return answer + SQL + result data
```

## Components

### 1. pgvector Embeddings Table (`chat_embeddings`)
- Stores schema metadata as vector embeddings (1536 dimensions)
- Content types: `table_desc`, `column_desc`, `example_query`, `relationship`
- Populated by `scripts/generate_embeddings.py` from `domain_specs.py`
- Used for semantic search to find relevant context for user questions

### 2. API Endpoint (`POST /chat`)
- **Input:** `{ question: string, domain?: string }`
- **Output:** `{ answer: string, sql?: string, data?: row[], row_count?: int, domains_used: string[] }`
- Embeds user question, retrieves context via pgvector, calls GPT-4o, optionally executes SQL

### 3. Frontend Chat Panel
- Collapsible card below the main analytics/explorer grid
- Message history with user/assistant bubbles
- Shows generated SQL and result tables inline
- Sends current domain as context hint

## Safety Guardrails
- **SELECT only:** SQL must start with SELECT (after stripping whitespace/comments)
- **Read-only transaction:** `SET TRANSACTION READ ONLY`
- **Statement timeout:** `SET LOCAL statement_timeout = '5000'` (5 seconds)
- **Row limit:** Results capped at 500 rows
- **Graceful errors:** SQL execution failures return the SQL + error message, not HTTP 500

## LLM Configuration
- **Provider:** OpenAI
- **Chat model:** `gpt-4o`
- **Embedding model:** `text-embedding-3-small` (1536 dimensions)
- **API key:** `OPENAI_API_KEY` environment variable

## Schema Context Strategy
The system prompt includes:
1. Full compact schema summary (all 7 tables, columns, types)
2. Top-10 pgvector-retrieved context chunks (semantic match to question)
3. Business rules: accuracy formula, bias formula, sales TYPE=1 filter, lag 0-4
4. Instruction to generate PostgreSQL-compatible SELECT with LIMIT 500

## Files Created/Modified
- `sql/009_create_chat_embeddings.sql` — pgvector extension + embeddings table
- `scripts/generate_embeddings.py` — schema → embeddings pipeline
- `api/main.py` — POST /chat endpoint
- `frontend/src/App.tsx` — chat panel UI
- `docker-compose.yml` — pgvector/pgvector:pg16 image
- `pyproject.toml` — openai dependency
- `.env.example` — OPENAI_API_KEY
- `Makefile` — generate-embeddings, db-apply-chat targets
