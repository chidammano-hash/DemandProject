# Market Intelligence

> AI-powered market briefing that combines Google web search results with GPT-4o narrative synthesis for any item-location pair.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Market Intel |
| **Key Files** | `frontend/src/tabs/MarketIntelTab.tsx`, `api/routers/intel.py` |

---

## Problem

Planners making item-level decisions lack external market context — competitive moves, regulatory changes, or seasonal trends that affect demand but are not captured in internal data.

---

## Solution

A `POST /market-intelligence` endpoint accepts an item number and location, looks up item metadata (description, brand, category) and location state from dimension tables, runs a Google Custom Search API query, and sends the search results to GPT-4o for narrative synthesis into a structured market briefing.

---

## How It Works

1. User selects an item and location in the Market Intel tab.
2. The endpoint looks up `dim_item` (description, brand, category) and `dim_location` (state).
3. A search query is constructed from the item description + brand + category + location state.
4. Google Custom Search API returns top web results (news, industry reports, competitor activity).
5. GPT-4o synthesizes the search snippets into a structured narrative briefing with sections: market trends, competitive landscape, regulatory factors, and demand implications.

---

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/market-intelligence` | Item-location market briefing from web search + GPT-4o |

### Request/Response

| Field | Type | Description |
|---|---|---|
| `item_id` (request) | string | Item number |
| `loc` (request) | string | Location code |
| `briefing` (response) | string | GPT-4o narrative synthesis |
| `sources` (response) | array | Web search result URLs |

Market intelligence has no pipeline — it is a stateless API call.

---

## Configuration

Configured via environment variables in `.env`:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | GPT-4o API access |
| `GOOGLE_API_KEY` | Google Custom Search API |
| `GOOGLE_CSE_ID` | Google Custom Search Engine ID |

---

## Dependencies

| Dependency | Reason |
|---|---|
| `openai` Python package | GPT-4o API client |
| Google Custom Search API | Web search for market intelligence |
| `dim_item`, `dim_location` | Item metadata and location state for search query construction |

---

## See Also

- `06-ai-platform/01-ai-planning-agent.md` — deeper AI analysis using Claude tool_use
- `07-user-experience/01-data-explorer.md` — structured data browsing for ad-hoc queries
