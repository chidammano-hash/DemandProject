<!-- SOURCE: feature11.md (Chatbot / NL Queries) -->
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


---

## Examples

### Example: Natural language query via chatbot

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the forecast accuracy for wine items in California for the last 3 months?"}' \
  | jq '{sql, answer, row_count}'
# {
#   "sql": "SELECT ROUND(100.0 - 100.0 * SUM(ABS(basefcst_pref - tothist_dmd)) / ...",
#   "answer": "Wine items in CA show 91.3% accuracy at lag 2 over the last 3 months.",
#   "row_count": 1
# }
```

### Example: pgvector schema retrieval (context for NL→SQL)

```sql
-- Embeddings stored in chat_embeddings table
SELECT table_name, column_name, embedding <=> $1 AS distance
FROM chat_embeddings
ORDER BY embedding <=> $1 LIMIT 5;
-- Returns most relevant schema context for the user's question
```

### Example: Frontend chat panel (TypeScript)

```typescript
const { mutateAsync: sendMessage } = useMutation({
  mutationFn: (question: string) =>
    fetch('/chat', { method: 'POST', body: JSON.stringify({ question }) })
      .then(r => r.json()),
})
// Usage: await sendMessage("Show top 10 DFUs by forecast error")
```


---

## Additional Examples

#### Example — End-to-end architecture flow

```
1. User types: "Top 10 DFUs by forecast error last 3 months"
2. Backend embeds question using text-embedding-3-small (1536-dim vector)
3. pgvector cosine similarity retrieves top-10 schema chunks from chat_embeddings:
     - fact_external_forecast_monthly.tothist_dmd  (distance=0.12)
     - fact_external_forecast_monthly.basefcst_pref (distance=0.14)
     - dim_dfu.dmdunit / dmdgroup / loc             (distance=0.18)
4. GPT-4o generates SQL:
     SELECT dmdunit, SUM(ABS(basefcst_pref - tothist_dmd)) AS total_error
     FROM fact_external_forecast_monthly
     WHERE startdate >= NOW() - INTERVAL '3 months'
       AND model_id = 'external' AND lag = 0
     GROUP BY dmdunit ORDER BY total_error DESC LIMIT 10;
5. Backend validates: starts with SELECT, no forbidden keywords
6. Executes in read-only transaction with 5s timeout, 500-row cap
7. Returns: answer (narrative), sql (the query), data (10 rows), columns, row_count
```

#### Example — Safety guardrails in action

```bash
# Attempt a forbidden write query via the chat endpoint
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Delete all forecast records"}' \
  | jq '{answer, error}'
# {"answer": "I can only run read-only queries on this database.",
#  "error": "SQL safety check failed: forbidden keyword DELETE"}

# SQL timeout guard — a slow aggregation is cut off at 5 seconds
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Count every row in every table"}' \
  | jq '{answer, error}'
# {"answer": "Query timed out. Try a more specific question.",
#  "error": "canceling statement due to statement timeout"}
```

#### Example — Schema context strategy (system prompt excerpt)

```
System prompt sent to GPT-4o:
---
Tables available:
  fact_external_forecast_monthly(forecast_ck, dmdunit, dmdgroup, loc, fcstdate,
    startdate, lag, execution_lag, basefcst_pref, tothist_dmd, model_id, ...)
  fact_sales_monthly(...)
  dim_dfu(dmdunit, dmdgroup, loc, ml_cluster, execution_lag, ...)

Business rules:
  - Forecast accuracy = 100 - (100 * SUM(ABS(F-A)) / ABS(SUM(A)))
  - Bias = (SUM(F) / SUM(A)) - 1
  - Sales rows: TYPE=1 only
  - lag 0 = same-month forecast; lag 4 = 4-month-ahead forecast
  - Always add LIMIT 500

Relevant schema context (from pgvector retrieval):
  [top-10 semantically similar chunks from chat_embeddings]

Generate a PostgreSQL SELECT query. Return JSON: {answer, sql}
---
```

#### Example — One-time setup and embedding generation

```bash
# Apply pgvector DDL (one-time)
make db-apply-chat
# Creates: pgvector extension, chat_embeddings table,
#          IVFFlat index idx_chat_embeddings_vector

# Generate and store schema embeddings
OPENAI_API_KEY=sk-... make generate-embeddings
# Reads domain_specs.py → generates ~150 chunks (table_desc, column_desc,
#   example_query, relationship) → upserts into chat_embeddings
# Elapsed: ~10 seconds, ~$0.001 in API cost

# Verify embeddings stored
docker exec demand-mvp-postgres psql -U demand -d demand_mvp \
  -c "SELECT content_type, COUNT(*) FROM chat_embeddings GROUP BY 1;"
# content_type   | count
# table_desc     |     8
# column_desc    |   112
# example_query  |    24
# relationship   |     6
```


---

<!-- SOURCE: feature18.md (Market Intelligence) -->
# Feature 18 — Market Intelligence

## Overview

AI-powered market intelligence tab that combines web search results with LLM-generated narrative to provide demand planners with contextual market insights for any product + location pair.

## Problem

Demand planners need external context beyond internal sales/forecast data to understand what's driving demand. Manually searching for product news, market trends, and demographic information is time-consuming and inconsistent across planners.

## Architecture

### Market Intelligence Flow

```
User selects Item + Location
          ↓
  POST /market-intelligence
          ↓
  ┌───────────────────────┐
  │ 1. Lookup item metadata│
  │    from dim_item       │
  │ 2. Lookup location     │
  │    from dim_location   │
  └───────────┬───────────┘
              ↓
  ┌───────────────────────┐
  │ 3. Build search query  │
  │    from item_desc +    │
  │    brand + category    │
  │ 4. Google Custom Search│
  │    → 5 web results     │
  └───────────┬───────────┘
              ↓
  ┌───────────────────────┐
  │ 5. Build system prompt │
  │    with product info + │
  │    search results +    │
  │    state demographics  │
  │ 6. Call GPT-4o         │
  │    → narrative briefing│
  └───────────┬───────────┘
              ↓
  React UI renders:
    - Context badges
    - Search result cards
    - Narrative briefing
```

## API Endpoint

### `POST /market-intelligence`

**Request:**
```json
{
  "item_no": "100320",
  "location_id": "1401-BULK"
}
```

**Response:**
```json
{
  "item_no": "100320",
  "location_id": "1401-BULK",
  "item_desc": "SMIRNOFF VODKA 80 1.75L",
  "brand_name": "SMIRNOFF",
  "category": "VODKA",
  "state_id": "CA",
  "site_desc": "California Distribution Center",
  "search_results": [
    {
      "title": "Vodka Market Trends 2025",
      "link": "https://example.com/...",
      "snippet": "The global vodka market..."
    }
  ],
  "narrative": "## Market Overview\n...",
  "generated_at": "2025-01-15T10:30:00Z"
}
```

**Error Codes:**
- `404` — Item or location not found
- `422` — Missing required fields
- `503` — Google API or OpenAI not configured

## External Services

| Service | Purpose | Config |
|---------|---------|--------|
| Google Custom Search API | Web search for product news/trends | `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` |
| OpenAI GPT-4o | Narrative synthesis + demographic context | `OPENAI_API_KEY` (existing) |

## Frontend

### Tab: "Mi" (Market Intelligence)
- Sky-blue color scheme (`bg-sky-100`)
- Chemistry-element button matching existing tab pattern

### UI Components
1. **Item selector** — datalist typeahead querying `/domains/item/suggest`
2. **Location selector** — datalist typeahead querying `/domains/location/suggest`
3. **Generate Briefing button** — triggers POST, shows loading spinner
4. **Context badges** — item_desc, brand, category, state, site
5. **Search result cards** — grid of clickable cards with title/snippet/link
6. **Narrative card** — sky-blue accent card with markdown-formatted GPT-4o response

### Loading State
- Chemistry-themed "Mi" element with pulse-glow animation (matches existing pattern)
- Descriptive text: "Searching the web and generating market briefing..."

### Error Handling
- Red-bordered error card for API failures
- Graceful degradation: if Google search fails, LLM still generates narrative from its own knowledge

## LLM Prompt Design

The system prompt instructs GPT-4o to act as a market intelligence analyst and:
1. Summarize market trends from web search results
2. Provide demographic/economic context for the target US state
3. Identify demand drivers and risks
4. Output 3-5 paragraph structured markdown narrative with section headers

Temperature: 0.4 (balanced creativity/consistency), max_tokens: 2000.

## Dependencies

- No new Python packages (uses `urllib.request` from stdlib for Google API)
- No new npm packages
- Reuses existing OpenAI integration (`_get_openai()` singleton)

## Configuration

Add to `.env`:
```
GOOGLE_API_KEY=<your-key>
GOOGLE_CSE_ID=<your-custom-search-engine-id>
```

## Files Modified

| File | Change |
|------|--------|
| `mvp/demand/api/main.py` | `POST /market-intelligence` endpoint + helpers |
| `mvp/demand/frontend/src/App.tsx` | Intel tab button, panel UI, state, effects |
| `mvp/demand/frontend/vite.config.ts` | `/market-intelligence` proxy route |
| `mvp/demand/.env.example` | `GOOGLE_API_KEY`, `GOOGLE_CX` |

---

## Implementation Corrections

### Environment Variable
- Actual env var is `GOOGLE_CX` (not `GOOGLE_CSE_ID` as spec states above)

### LLM Model
- Uses `gpt-4o-mini` (not `gpt-4o`)
- `temperature=0.7` (not 0.4)
- `max_tokens=1500` (not 2000)

### Sales Context
- Gathers recent 12-month sales data from `agg_sales_monthly` and includes in LLM prompt

### OpenAI Web Search Fallback
- When Google search not configured, falls back to OpenAI `responses.create()` with `web_search_preview` tool
- Extracts URL citations from response and adds to `search_results`

### Error Codes
- HTTP 502 for AI generation failures (not in original spec)

### Authentication
- Router uses `require_api_key` dependency

### File Locations
- Router: `api/routers/intel.py`
- Tab: `frontend/src/tabs/MarketIntelTab.tsx` (not `App.tsx`)
- Uses TanStack Query `useMutation` and `useQuery`
- Global filter integration: syncs item/location from `GlobalFilterContext`
- Retry button on error card

### Key Files
| File | Purpose |
|------|---------|
| `mvp/demand/api/routers/intel.py` | Router version of endpoint |
| `mvp/demand/frontend/src/tabs/MarketIntelTab.tsx` | Extracted tab component |
| `mvp/demand/tests/api/test_intel.py` | Backend API tests |
| `mvp/demand/frontend/src/tabs/__tests__/MarketIntelTab.test.tsx` | Frontend smoke test |


---

## Examples

### Example: Market intelligence request

```bash
curl -s -X POST http://localhost:8000/market-intelligence \
  -H "Content-Type: application/json" \
  -d '{"item_no": "100320", "location_id": "1401-BULK"}' \
  | jq '{item_desc, narrative_preview: .narrative[0:120]}'
# {
#   "item_desc": "CABERNET SAUV 750ML",
#   "narrative_preview": "California wine demand remains resilient heading into Q1 2026..."
# }
```

### Example: Required environment variables

```bash
# .env
OPENAI_API_KEY=sk-...        # Required for GPT-4o narrative synthesis
GOOGLE_API_KEY=AIza...       # Optional: Google Custom Search API
GOOGLE_CSE_ID=abc123...      # Optional: Google Custom Search Engine ID
```

### Example: Sample narrative response structure

```json
{
  "item_no": "100320",
  "item_desc": "CABERNET SAUV 750ML",
  "brand": "COASTAL RIDGE",
  "location_id": "1401-BULK",
  "state": "CA",
  "search_results": [
    {"title": "CA Wine Sales Outlook 2026", "snippet": "..."}
  ],
  "narrative": "**Market Overview**: Coastal Ridge Cabernet demand in CA remains strong...",
  "generated_at": "2026-02-28T14:30:00Z"
}
```

### Example: Error handling — Google API unavailable, fallback to OpenAI web search

```python
# api/routers/intel.py — graceful fallback when Google CSE not configured
try:
    search_results = google_search(query, api_key=GOOGLE_API_KEY, cx=GOOGLE_CX)
except Exception as exc:
    # Fallback: use OpenAI web_search_preview tool instead of Google
    logger.warning(f"Google search failed ({exc}), falling back to OpenAI web search")
    response = openai_client.responses.create(
        model="gpt-4o-mini",
        tools=[{"type": "web_search_preview"}],
        input=f"Search for: {query}"
    )
    # Extract URL citations from response
    search_results = [
        {"title": ann.title, "link": ann.url, "snippet": ""}
        for ann in response.output[0].content
        if hasattr(ann, "url")
    ]

# If BOTH Google and OpenAI web search fail → HTTP 503
if not search_results and not openai_available:
    raise HTTPException(503, detail="No search provider configured (GOOGLE_API_KEY or OPENAI_API_KEY required)")
```

### Example: Verify env config and test the endpoint

```bash
# Check which search provider is active
grep -E "GOOGLE_API_KEY|GOOGLE_CX|OPENAI_API_KEY" mvp/demand/.env

# Test market intelligence with curl (requires API_KEY if set)
curl -s -X POST http://localhost:8000/market-intelligence \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -d '{"item_no": "100320", "location_id": "1401-BULK"}' | jq '.narrative | .[0:200]'
# "## Market Overview\nCabernet Sauvignon from California remains..."

# 503 response if neither search provider configured:
# {"detail": "No search provider configured"}
```
