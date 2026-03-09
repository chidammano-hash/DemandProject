# IPAIfeature1 — AI Planning Agent

## EPIC
AIPlanning

## Status
Planned

## Priority
P1 — Must Have (Foundation)

## Effort
L (Large)

## Expert Perspectives
- **AI/ML Systems Architect** (lead) — tool-use agent design, LLM orchestration, agentic loops
- **Supply Chain Planning Expert** — causal chain reasoning, exception prioritization, financial impact quantification
- **Inventory Optimization Expert** — stockout/excess thresholds, policy alignment signals
- **Software Architect** — async execution, DB schema, API design, frontend integration

---

## Problem Statement

The platform has five mature, siloed stages: demand forecasting (LGBM/CatBoost/XGBoost + champion selection) → inventory analytics (DOS, stockout/excess flags) → EOQ targets → replenishment policies + job automation. A planner today navigates each tab manually and mentally connects the dots.

**The gap**: Nobody knows that DFU X has 42% champion WAPE → which inflates its inventory target → the continuous_rop policy is then over-ordering based on a biased forecast → causing 200+ DOS excess worth $85K working capital. That causal chain spans 4 screens and requires deep domain knowledge to trace.

**What this is NOT**: A chatbot. No Q&A, no free-form prompts, no back-and-forth conversation.

**What this IS**: An automated exception work-queue. The AI agent scans the full portfolio on a schedule, traces causal chains across all data layers, and surfaces ranked insight cards with specific actions: *"Here are the 12 DFUs that need your attention this week, diagnosed and prioritized by financial impact."*

---

## Architecture Overview

```
config/ai_planner_config.yaml        ← thresholds, model, schedule, severity rules
         │
common/ai_planner.py                 ← Claude tool_use agent core; 10 tools (9 read + 1 write)
         │
sql/026_create_ai_insights.sql       ← ai_insights + ai_planning_memos tables
         │
scripts/generate_ai_insights.py      ← CLI batch job (portfolio scan or single DFU)
         │
api/routers/ai_planner.py            ← 5 FastAPI endpoints under /ai-planner/*
         │
frontend/src/tabs/AIPlannerTab.tsx   ← "AI Planner" tab: insight cards + planning memo
```

---

## Database Schema

### New File: `sql/026_create_ai_insights.sql`

#### Table: `ai_insights`

| Column | Type | Description |
|--------|------|-------------|
| insight_id | SERIAL PK | |
| insight_type | VARCHAR(80) | `stockout_risk`, `excess_inventory`, `forecast_bias`, `policy_gap`, `champion_degradation` |
| severity | VARCHAR(20) CHECK | `critical` / `high` / `medium` / `low` |
| item_no, loc | TEXT NOT NULL | DFU identity |
| abc_vol, cluster_assignment | TEXT | DFU attributes at insight creation time |
| summary | TEXT NOT NULL | 1-sentence description of the exception |
| recommendation | TEXT NOT NULL | Specific executable action for planner |
| reasoning | TEXT | AI chain-of-thought: how it connected the signals across layers |
| financial_impact_estimate | NUMERIC(15,2) | USD — positive = cost/risk |
| dos, total_lt_days | NUMERIC/INT | Snapshot at insight creation |
| champion_wape, forecast_bias_pct | NUMERIC(8,4) | Snapshot at insight creation |
| current_policy_id | TEXT | Active replenishment policy at insight time |
| eoq_effective | NUMERIC(15,4) | Effective EOQ at insight time |
| status | VARCHAR(20) CHECK | `open` → `acknowledged` → `resolved` |
| acknowledged_at, resolved_at | TIMESTAMPTZ | Status transition timestamps |
| model_version | TEXT | e.g. `claude-sonnet-4-6` |
| scan_run_id | TEXT | Groups all insights from one scan run |
| created_at, updated_at | TIMESTAMPTZ | |

**Indexes**:
- `(status, severity, created_at DESC)` — primary UI list query
- `(item_no, loc)` — DFU drill-in
- `(scan_run_id, created_at DESC)` — scan grouping

#### Table: `ai_planning_memos`

| Column | Type | Description |
|--------|------|-------------|
| memo_id | SERIAL PK | |
| period | DATE NOT NULL | Month-start, e.g. `2026-03-01` |
| scope | VARCHAR(20) CHECK | `portfolio` or `dfu` |
| item_no, loc | TEXT | NULL for portfolio scope |
| narrative_text | TEXT NOT NULL | Full AI narrative (rendered as markdown in UI) |
| content_json | JSONB NOT NULL | Structured data: totals, top exceptions, coverage stats |
| model_version | TEXT | |
| created_at | TIMESTAMPTZ | |

**Indexes**:
- `(period DESC, scope)` — memo list query
- `(item_no, loc, period DESC) WHERE scope = 'dfu'` — DFU-level memos

---

## Configuration

### New File: `config/ai_planner_config.yaml`

```yaml
model: "claude-sonnet-4-6"
max_tokens: 4096
temperature: 0.2                     # Low — deterministic structured output

portfolio_scan_limit: 100            # Top N DFUs to analyze per portfolio run
forecast_lookback_months: 6

insight_thresholds:
  stockout_dos_multiplier: 1.5       # DOS < total_lt * 1.5 → stockout risk
  excess_dos_days: 180               # DOS > 180 → excess
  bias_threshold_pct: 20.0           # |bias| > 20% for N consecutive months
  bias_persistence_months: 3
  champion_wape_critical: 50.0
  champion_wape_high: 35.0
  eoq_deviation_pct: 50.0

default_unit_cost: 10.0              # Fallback when dim_item has no unit_cost
carrying_cost_rate: 0.25             # Annual carrying cost fraction (for excess $ estimate)
stockout_cost_multiplier: 3.0        # Stockout cost = daily_demand * unit_cost * multiplier

scheduled: true
cron: "0 6 * * 1"                    # Every Monday at 6am UTC
```

---

## AI Agent Core

### New File: `common/ai_planner.py`

Uses `anthropic.Anthropic` with `tool_use` content blocks. The agentic loop continues until `stop_reason == "end_turn"`. All tools call PostgreSQL directly via the shared pool — no HTTP round-trips to self.

#### Class Design

```python
class AIPlannerAgent:
    def __init__(self, pool, config: dict):
        self.client = anthropic.Anthropic()
        self.pool = pool
        self.config = config

    async def run_dfu_analysis(self, item_no: str, loc: str, scan_run_id: str) -> list[dict]:
        """Analyze a single DFU; returns list of insight dicts created."""

    async def run_portfolio_scan(self, scan_run_id: str) -> dict:
        """Scan top N exceptions; returns summary with counts by severity/type."""

    async def generate_portfolio_memo(self, period: date, scan_run_id: str) -> dict:
        """Generate weekly planning memo; writes to ai_planning_memos; returns content."""

    def _dispatch_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Route a tool_use block to the corresponding Python handler."""

    def _run_agentic_loop(self, messages: list, task_context: str) -> str:
        """Core loop: call Claude → dispatch tools → repeat until end_turn."""
```

#### Tool Registry (10 Tools)

| # | Tool Name | SQL Layer | Key Output |
|---|-----------|-----------|------------|
| 1 | `get_dfu_full_context` | dim_dfu + latest agg_inventory_monthly + fact_eoq_targets + policy tables | ABC, DOS, EOQ, policy, lead time in one row |
| 2 | `get_forecast_performance` | fact_external_forecast_monthly WHERE model_id IN ('champion','external') AND lag=0 | Per-month forecast, actual, pct_error for last N months |
| 3 | `get_portfolio_exceptions` | dim_dfu + latest agg_inventory_monthly + champion WAPE CTE + policy assignment | Ranked list with stockout_risk / excess / policy_gap / high_wape flags |
| 4 | `compute_bias_trend` | fact_external_forecast_monthly + LAG window functions | monthly_bias_pct, bias_3m_avg, bias_6m_avg |
| 5 | `get_inventory_trend` | agg_inventory_monthly | DOS trajectory over last N months |
| 6 | `get_eoq_context` | fact_eoq_targets + derived columns | eoq_months_supply, cycle_stock_value, deviation from current ordering pattern |
| 7 | `get_similar_dfus` | dim_dfu (same cluster + ABC class) + latest inventory + champion WAPE | Peer benchmark for the target DFU |
| 8 | `check_stockout_history` | mv_inventory_forecast_monthly | stockout_months/6, excess_months/6, avg DOS at stockout |
| 9 | `get_portfolio_health_summary` | Aggregation across all 4 layers | total_dfus, avg_dos, stockout_count, avg_wape, policy_coverage, open_insights |
| 10 | `create_insight` | INSERT INTO ai_insights | Returns insight_id (the only write-capable tool) |

#### Tool Input Schemas

**Tool 3 — `get_portfolio_exceptions`**:
```json
{
  "limit": {"type": "integer", "default": 50},
  "abc_vol_filter": {"type": "string", "description": "Optional: A, B, or C"}
}
```

**Tool 10 — `create_insight`**:
```json
{
  "insight_type": {"type": "string", "enum": ["stockout_risk","excess_inventory","forecast_bias","policy_gap","champion_degradation"]},
  "severity": {"type": "string", "enum": ["critical","high","medium","low"]},
  "item_no": {"type": "string"},
  "loc": {"type": "string"},
  "abc_vol": {"type": "string"},
  "cluster_assignment": {"type": "string"},
  "summary": {"type": "string", "description": "One sentence: what is wrong"},
  "recommendation": {"type": "string", "description": "Specific executable action"},
  "reasoning": {"type": "string", "description": "Chain-of-thought across layers"},
  "financial_impact_estimate": {"type": "number"},
  "dos": {"type": "number"},
  "total_lt_days": {"type": "integer"},
  "champion_wape": {"type": "number"},
  "forecast_bias_pct": {"type": "number"},
  "current_policy_id": {"type": "string"},
  "eoq_effective": {"type": "number"},
  "scan_run_id": {"type": "string"}
}
```

#### System Prompt (Key Directives)

The system prompt embeds:
1. **Full workflow context**: The 5 data layers (forecasting → champion → inventory → EOQ → policy) and how each feeds the next
2. **Business rules**: A-class → continuous_rop, lumpy/intermittent → manual, B → periodic_review, C → min_max
3. **Severity classification rules** from config
4. **Instruction**: "Always call tools to get real data before reasoning. Do not guess or infer values. Trace the causal chain across layers: forecast accuracy → inventory consequence → policy effectiveness → financial impact."
5. **Instruction**: "Use `create_insight` for every exception you identify. Be specific — name the action (change policy type from X to Y, reduce EOQ by ~N%, trigger emergency reorder for Z units)."
6. **Instruction**: "Quantify financial impact where possible using: excess = qty_excess × unit_cost × carrying_cost_rate; stockout = daily_demand × unit_cost × stockout_cost_multiplier × days_at_risk."

---

## Insight Types

### 5 Insight Types Generated

| Type | Trigger Condition | Severity | Example Summary |
|------|------------------|----------|-----------------|
| `stockout_risk` | DOS < total_lt × 1.5 AND trend declining | critical/high | "Item 100320 at DC1 has only 18 DOS against 14-day lead time with declining trend for 3 months" |
| `excess_inventory` | DOS > 180 days | high (A-class) / medium (B/C) | "Item 200450 at DC3 has 245 DOS — $62K excess working capital at 25% carrying cost" |
| `forecast_bias` | Rolling bias > 20% for 3+ consecutive months | high/medium | "Champion consistently over-forecasts Item 300120 by +31% — inflating inventory targets by ~$8K/month" |
| `policy_gap` | No policy_id assigned in fact_dfu_policy_assignment | medium | "37 DFUs at DC2 have no replenishment policy — reorder timing is undefined" |
| `champion_degradation` | champion WAPE > 35% | high (>50% = critical) | "Champion WAPE for seasonal cluster at 48% this quarter — forecast quality insufficient for replenishment decisions" |

### Causal Chain Reasoning (the key differentiator)

The AI doesn't just flag the symptom — it traces the chain. Example for `excess_inventory + forecast_bias` combined:

> "Item 300120 at DC2 has 210 DOS [Excess]. Root cause: champion model (lgbm_cluster) has over-forecast by +28% for the last 4 months [Forecast Bias], leading to systematic over-ordering under the continuous_rop policy [Policy Execution]. The effective EOQ of 180 units amplifies each order cycle by ~50 units beyond actual need. Recommendation: (1) Switch champion strategy from 'expanding' to 'rolling_3m' to faster adapt to the recent demand regime. (2) Reduce effective EOQ floor to 100 units pending forecast stabilization. (3) Pause next replenishment cycle — current on-hand covers 7 months at actual demand rate. Financial impact: ~$52K working capital freed if DOS reduced to 90 days."

---

## API Endpoints

### New Router: `api/routers/ai_planner.py`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/ai-planner/analyze` | required | Synchronous DFU analysis. Body: `{item_no: str, loc: str}`. Returns list of insights created. ~5-15s. |
| `POST` | `/ai-planner/portfolio-scan` | required | Trigger portfolio scan. Returns HTTP 202 immediately. Scan runs in background thread. |
| `GET` | `/ai-planner/insights` | none | Paginated insights. Query params: `severity`, `status`, `insight_type`, `page`, `page_size` (default 20). |
| `PUT` | `/ai-planner/insights/{insight_id}/status` | required | Update status. Body: `{status: "acknowledged" \| "resolved"}`. |
| `GET` | `/ai-planner/memos` | none | List planning memos. Query params: `scope` (portfolio/dfu), `item_no`, `loc`, `limit` (default 10). |

**Mount in `api/main.py`** (before `domains.py`):
```python
from api.routers import ai_planner
app.include_router(ai_planner.router, prefix="", tags=["ai-planner"])
```

**Add to `frontend/vite.config.ts`**:
```typescript
"/ai-planner": { target: "http://127.0.0.1:8000", changeOrigin: true }
```

---

## Batch Script

### New File: `scripts/generate_ai_insights.py`

```bash
# Full portfolio scan
uv run python scripts/generate_ai_insights.py --portfolio

# Single DFU analysis
uv run python scripts/generate_ai_insights.py --item 100320 --loc 1401-BULK

# Preview without writing to DB
uv run python scripts/generate_ai_insights.py --portfolio --dry-run
```

Reads `config/ai_planner_config.yaml`. Uses `common/ai_planner.py`. Writes to `ai_insights` + `ai_planning_memos`. Prints scan summary.

---

## Job Scheduler Integration

Add to `common/job_registry.py` under a new `ai` group:

```python
"generate_ai_insights": JobType(
    job_type="generate_ai_insights",
    group="ai",
    display_name="AI Insight Generation",
    description="Scan portfolio for planning exceptions and generate AI insights",
    script="scripts/generate_ai_insights.py",
    args=["--portfolio"],
)
```

---

## Frontend: AI Planner Tab

### New File: `frontend/src/tabs/AIPlannerTab.tsx`

**NOT a chat interface.** A structured exception work-queue with 3 sections.

#### Section 1: Portfolio Health Bar (always-visible, compact)
4 KPI chips derived from insights + latest memo:
- **Open Insights** — total open count
- **Critical** — critical-severity count (red badge)
- **Avg DOS** — portfolio average days of supply
- **Champion WAPE** — average champion accuracy

#### Section 2: Insight Cards (main content)

**Filter bar**:
- Severity: All / Critical / High / Medium / Low
- Type: All / Stockout Risk / Excess / Forecast Bias / Policy Gap / Degradation
- Status: Open / Acknowledged / Resolved

**"Generate Now" button** → `POST /ai-planner/portfolio-scan` → spinner: "Scanning portfolio…" → auto-refreshes list on completion

**Card design** (sorted: severity DESC then financial_impact DESC):
```
┌─────────────────────────────────────────────────────────────────┐
│ [● CRITICAL]  [stockout_risk]          Item 100320 @ 1401-BULK  │
│               [A-class] [high_volume_steady cluster]            │
│                                                                  │
│  "Only 18 DOS against 14-day lead time with declining DOS       │
│   trend for 3 months and 41% champion WAPE."                    │
│                                                                  │
│  → Trigger emergency reorder; switch champion to ensemble       │
│    strategy for high-volume items.                              │
│                                                                  │
│  [DOS: 18d] [WAPE: 41%] [Policy: continuous_rop]  [$8,500 risk] │
│                                                                  │
│  ▸ View reasoning                                               │
│                                                    [Acknowledge] [Resolve]
└─────────────────────────────────────────────────────────────────┘
```

Severity badge colors:
- `critical` → red background
- `high` → orange
- `medium` → yellow
- `low` → gray

#### Section 3: Planning Memo (collapsible bottom panel)
- Latest portfolio memo rendered as markdown
- Period + model_version badge
- "View previous memos" accordion

### TypeScript Types: `frontend/src/types/ai_planner.ts`

```typescript
type InsightSeverity = 'critical' | 'high' | 'medium' | 'low';
type InsightStatus   = 'open' | 'acknowledged' | 'resolved';
type InsightType     = 'stockout_risk' | 'excess_inventory' | 'forecast_bias'
                     | 'policy_gap' | 'champion_degradation';

interface AiInsight {
  insight_id: number;
  insight_type: InsightType;
  severity: InsightSeverity;
  item_no: string;
  loc: string;
  abc_vol: string | null;
  cluster_assignment: string | null;
  summary: string;
  recommendation: string;
  reasoning: string | null;
  financial_impact_estimate: number | null;
  dos: number | null;
  champion_wape: number | null;
  current_policy_id: string | null;
  status: InsightStatus;
  created_at: string;
}

interface AiPlanningMemo {
  memo_id: number;
  period: string;           // ISO date string
  scope: 'portfolio' | 'dfu';
  item_no: string | null;
  loc: string | null;
  narrative_text: string;
  content_json: Record<string, unknown>;
  model_version: string;
  created_at: string;
}

interface AiInsightsResponse {
  insights: AiInsight[];
  total: number;
  page: number;
  page_size: number;
}
```

### TanStack Query Keys (`frontend/src/api/queries.ts` additions):
```typescript
export const aiInsightsQuery = (filters: AiInsightFilters) =>
  queryOptions({ queryKey: ['ai-insights', filters], queryFn: () => fetchAiInsights(filters), refetchInterval: 30_000 });

export const aiMemosQuery = (scope: string) =>
  queryOptions({ queryKey: ['ai-memos', scope], queryFn: () => fetchAiMemos(scope) });
```

---

## App Integration

### Files to Modify

| File | Change |
|------|--------|
| `frontend/src/App.tsx` | Add `const AIPlannerTab = lazy(() => import("./tabs/AIPlannerTab"))` + case `"aiPlanner"` |
| `frontend/src/components/AppSidebar.tsx` | Add 13th nav item: `{ id: "aiPlanner", label: "AI Planner", icon: Brain, section: "Intelligence" }` |
| `frontend/src/hooks/useUrlState.ts` | Add `"aiPlanner"` to tabs list |
| `frontend/vite.config.ts` | Add `"/ai-planner"` proxy entry |
| `api/main.py` | Mount `ai_planner` router before `domains` router |
| `common/job_registry.py` | Add `generate_ai_insights` job type under `ai` group |
| `Makefile` | Add `ai-insights-schema`, `ai-insights-scan`, `ai-insights-dfu`, `ai-insights-all` targets |

---

## Makefile Targets

```makefile
ai-insights-schema:
	uv run psql $(PSQL_ARGS) -f sql/026_create_ai_insights.sql

ai-insights-scan:
	uv run python scripts/generate_ai_insights.py --portfolio

ai-insights-dfu:
	uv run python scripts/generate_ai_insights.py --item $(ITEM) --loc $(LOC)

ai-insights-all: ai-insights-schema ai-insights-scan
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `sql/026_create_ai_insights.sql` | DDL for ai_insights + ai_planning_memos |
| `config/ai_planner_config.yaml` | Model, thresholds, severity rules, schedule |
| `common/ai_planner.py` | AI agent core + 10-tool registry |
| `api/routers/ai_planner.py` | 5 FastAPI endpoints |
| `scripts/generate_ai_insights.py` | CLI batch generation script |
| `frontend/src/tabs/AIPlannerTab.tsx` | AI Planner tab component |
| `frontend/src/types/ai_planner.ts` | TypeScript types |
| `tests/unit/test_ai_planner.py` | ~15-20 unit tests |
| `tests/api/test_ai_planner_api.py` | ~10 API tests |
| `frontend/src/tabs/__tests__/AIPlannerTab.test.tsx` | ~6 frontend smoke tests |

---

## Test Plan

### Backend Unit Tests (`tests/unit/test_ai_planner.py`)
- Each tool function in isolation with mocked DB pool (psycopg fetchall mock)
- `_dispatch_tool` routing to correct handler
- Severity classification logic (thresholds from config)
- `run_dfu_analysis` with mocked `anthropic.Anthropic` client
- `generate_portfolio_memo` structure validation
- Dry-run mode: no DB writes, correct log output

### Backend API Tests (`tests/api/test_ai_planner_api.py`)
Pattern: `httpx.AsyncClient(transport=ASGITransport(app), base_url="http://test")`

| Test | Assertion |
|------|-----------|
| `test_get_insights_empty` | 200, empty list |
| `test_get_insights_filtered_by_severity` | Only matching severity returned |
| `test_get_insights_filtered_by_status` | Only matching status returned |
| `test_update_insight_status_acknowledge` | 200, status = "acknowledged" |
| `test_update_insight_status_resolve` | 200, status = "resolved" |
| `test_update_insight_status_invalid` | 422 |
| `test_update_insight_not_found` | 404 |
| `test_get_memos_empty` | 200, empty list |
| `test_analyze_dfu_missing_body` | 422 |
| `test_portfolio_scan_returns_202` | 202 |

### Frontend Tests (`frontend/src/tabs/__tests__/AIPlannerTab.test.tsx`)
- Renders without crash (empty mock data)
- "Generate Now" button present
- Insight cards render with mock insights array
- Severity badge has correct color class per severity
- Acknowledge button triggers mutation
- Empty state message when no insights

---

## End-to-End Example

**DFU: Item 100320 @ 1401-BULK | A-class | cluster=high_volume_steady**

1. Agent calls `get_dfu_full_context` → DOS=18, total_lt=14, policy=continuous_rop, champion_wape=41%, EOQ=250
2. Agent calls `get_forecast_performance` → champion over-forecasts +28% for last 4 months
3. Agent calls `compute_bias_trend` → bias_3m_avg=+26%, direction=persistent_over
4. Agent calls `check_stockout_history` → stockout_months=1/6, excess_months=0/6
5. Agent calls `get_similar_dfus` → peers in same cluster avg DOS=42 → this DFU undersupplied vs peers
6. Agent reasons: *Despite systematic over-forecast, DOS is critically low — actual demand is exceeding even the inflated forecast. This is a demand surprise scenario, not an overstock risk.*
7. Agent calls `create_insight(insight_type="stockout_risk", severity="high", summary="Item 100320 at 1401-BULK has 18 DOS vs 14-day lead time; actual demand is exceeding even the over-inflated champion forecast (+28% bias) indicating a demand spike beyond model capacity", recommendation="(1) Trigger emergency reorder of ~300 units. (2) Switch champion strategy from 'expanding' to 'ensemble' to capture demand spikes. (3) Review if recent event (promo, new customer) explains the demand step-change.", financial_impact_estimate=8500, ...)`
8. Insight written to DB → appears in `GET /ai-planner/insights`
9. Planner opens AI Planner tab → sees critical card → reads reasoning → clicks Acknowledge → places reorder

---

## Dependency

Requires `anthropic` Python SDK (already used by existing chat endpoint in `api/routers/chat.py`). No new infrastructure dependency.

---

## Verification

1. `make ai-insights-schema` → `make check-db` confirms `ai_insights` and `ai_planning_memos` tables exist
2. `uv run python scripts/generate_ai_insights.py --portfolio --dry-run` logs exceptions without writing
3. `curl http://localhost:8000/ai-planner/insights` → 200 with empty list
4. `POST /ai-planner/analyze` with known item/loc → returns `insights: [...]`
5. Navigate to "AI Planner" tab in UI → insight cards render (or empty state message)
6. `make test-all` → all tests pass
7. `POST /jobs/run` with `job_type=generate_ai_insights` → 202 → job appears in JobsTab history
