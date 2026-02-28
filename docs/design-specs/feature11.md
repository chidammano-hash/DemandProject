# Feature 11: Chatbot / Natural Language Queries

## Objective
Allow users to ask questions in plain English ("What's the accuracy for item 100320 in Q4?") and receive answers with data from the Demand Studio database.

## Architecture

```
User Question
      |
OpenAI text-embedding-3-small -> embed question
      |
pgvector cosine similarity search -> top-10 relevant schema chunks
      |
GPT-4o prompt (system: schema + context + rules, user: question)
      |
Parse response -> extract answer + SQL
      |
Validate SQL (SELECT only) -> execute read-only with timeout
      |
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
- `scripts/generate_embeddings.py` — schema -> embeddings pipeline
- `api/main.py` — POST /chat endpoint
- `frontend/src/App.tsx` — chat panel UI
- `docker-compose.yml` — pgvector/pgvector:pg16 image
- `pyproject.toml` — openai dependency
- `.env.example` — OPENAI_API_KEY
- `Makefile` — generate-embeddings, db-apply-chat targets

## Makefile Targets
```makefile
db-apply-chat:          # Apply pgvector + embeddings table DDL
generate-embeddings:    # Generate and store schema embeddings (requires OPENAI_API_KEY)
```

## Dependencies
- All prior features (reads from all tables for schema context)
- OpenAI API (`OPENAI_API_KEY`)
- pgvector extension

---

## Implementation Corrections

### Actual Response Schema
```json
{"answer": "string", "sql": "string|null", "data": "[{...}]|null", "columns": "string[]", "row_count": "int|null", "error": "string (on failure only)"}
```
Note: `domains_used` field documented in original spec is NOT implemented.

### Authentication
- Requires `X-API-Key` header when `API_KEY` env var is set (via `require_api_key` dependency from `api/auth.py`)

### LLM Configuration
- `response_format={"type": "json_object"}` for structured output
- `temperature=0.1`, `max_tokens=2000`

### Router Module
- Also implemented in `api/routers/chat.py`

### Frontend
- Extracted to `frontend/src/tabs/ChatPanel.tsx` (not in `App.tsx`)
- Uses TanStack Query `useMutation`
- Branded as "Chat with Planthium"
- SQL shown in collapsible `<details>` section
- Inline data table limited to 10 rows

### Embeddings Optimization
- IVFFlat index: `idx_chat_embeddings_vector` with `lists = max(1, chunk_count / 10)`

### SQL Safety
- Forbidden keywords: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `TRUNCATE`, `GRANT`, `REVOKE`, `COPY`

### TypeScript Types
- `ChatMessage` type in `frontend/src/types/index.ts`
- `sendChatMessage(question, domain)` in `api/queries.ts`
