# Chatbot & Market Intelligence

Two AI-powered features: (1) a natural-language-to-SQL chatbot that lets planners query the database conversationally, and (2) a market intelligence briefing that combines web search results with GPT-4o narrative synthesis for any item-location pair.

| Field | Value |
|---|---|
| Status | Implemented |
| Spec | 06-ai-platform/02-chatbot-market-intel |
| Frontend | `ChatPanel.tsx` (chatbot), Market Intel tab |
| Backend | `api/main.py` (`POST /chat`, `POST /market-intelligence`) |
| Config | Requires `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` in `.env` |
| SQL | `sql/011_create_embeddings.sql` |

---

## Problem

Planners frequently need ad-hoc data answers: "Which items had WAPE above 50% last quarter?" or "Show me the top 10 locations by stockout count." Writing SQL is not realistic for business users. Separately, planners making item-level decisions lack external market context -- competitive moves, regulatory changes, or seasonal trends that affect demand but are not captured in internal data.

---

## Solution

**Chatbot:** A `POST /chat` endpoint translates natural language questions into SQL using GPT-4o. It retrieves relevant table schema context via pgvector (vector similarity search on pre-embedded schema descriptions), constructs a SQL query, executes it read-only with a 5-second timeout and 500-row limit, and returns both the query and results.

**Market Intelligence:** A `POST /market-intelligence` endpoint accepts an item number and location, looks up item metadata (description, brand, category) and location state from dimension tables, runs a Google Custom Search API query, and sends the search results to GPT-4o for narrative synthesis into a structured market briefing.

---

## How It Works

### Chatbot Flow

1. User types a natural language question in the chat panel.
2. The endpoint retrieves the top-K most relevant schema embeddings via pgvector cosine similarity.
3. GPT-4o receives the question + schema context and generates a SQL query.
4. The query is executed against PostgreSQL in a read-only transaction with a 5-second statement timeout and 500-row result limit.
5. Results are returned as a JSON array alongside the generated SQL for transparency.

### Market Intelligence Flow

1. User selects an item and location in the Market Intel tab.
2. The endpoint looks up `dim_item` (description, brand, category) and `dim_location` (state).
3. A search query is constructed from the item description + brand + category + location state.
4. Google Custom Search API returns top web results (news, industry reports, competitor activity).
5. GPT-4o synthesizes the search snippets into a structured narrative briefing with sections: market trends, competitive landscape, regulatory factors, and demand implications.

### Safety Guardrails

| Guardrail | Purpose |
|---|---|
| Read-only transaction | Prevents any data modification via chat |
| 5-second statement timeout | Kills runaway queries |
| 500-row result limit | Prevents massive result sets |
| Schema embedding retrieval | Only exposes relevant tables, not the full schema |
| API key requirement | Both endpoints require `OPENAI_API_KEY` |

---

## Data Model

| Table | Purpose | Key Columns |
|---|---|---|
| `schema_embeddings` | Pre-embedded schema descriptions for pgvector retrieval | `id`, `table_name`, `description`, `embedding` (vector) |

No additional tables are created for market intelligence -- it is a stateless pass-through.

---

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | Natural language question to SQL query + results |
| POST | `/market-intelligence` | Item-location market briefing from web search + GPT-4o |

### Chat Request/Response

| Field | Type | Description |
|---|---|---|
| `question` (request) | string | Natural language question |
| `sql` (response) | string | Generated SQL query |
| `rows` (response) | array | Query result rows |
| `error` (response) | string or null | Error message if query failed |

### Market Intelligence Request/Response

| Field | Type | Description |
|---|---|---|
| `item_id` (request) | string | Item number |
| `loc` (request) | string | Location code |
| `briefing` (response) | string | GPT-4o narrative synthesis |
| `sources` (response) | array | Web search result URLs |

---

## Pipeline

| Step | Command | What It Does |
|---|---|---|
| Schema | `make db-apply-chat` | Creates pgvector extension + `schema_embeddings` table |
| Embeddings | `make generate-embeddings` | Generates and stores schema embeddings (requires `OPENAI_API_KEY`) |

Market intelligence has no pipeline -- it is a stateless API call.

---

## Configuration

Both features are configured via environment variables in `.env`:

| Variable | Required By | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Both | GPT-4o API access |
| `GOOGLE_API_KEY` | Market Intel only | Google Custom Search API |
| `GOOGLE_CSE_ID` | Market Intel only | Google Custom Search Engine ID |

---

## Dependencies

| Dependency | Reason |
|---|---|
| `openai` Python package | GPT-4o API client |
| `pgvector` PostgreSQL extension | Vector similarity search for schema retrieval |
| Google Custom Search API | Web search for market intelligence |
| `dim_item`, `dim_location` | Item metadata and location state for search query construction |

---

## See Also

- `06-ai-platform/01-ai-planning-agent.md` -- deeper AI analysis using Claude tool_use (complements chatbot's quick answers)
- `07-user-experience/01-data-explorer.md` -- structured data browsing alternative to free-form chat queries
