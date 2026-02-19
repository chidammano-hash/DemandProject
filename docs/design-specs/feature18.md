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
| `mvp/demand/.env.example` | `GOOGLE_API_KEY`, `GOOGLE_CSE_ID` |
