# Feature 40 — Demand Planner Storyboard: Exception-Based Value Workflow

**Status:** Design
**Priority:** Critical — Highest value-per-effort feature remaining
**Dependencies:** Feature 15 (champion selection), Feature 7 (clustering), Feature 34 (inventory), Feature 39 (job scheduler)

---

## 1. The Problem Steve Jobs Identified

> "You've built a Swiss Army knife when you needed a scalpel."

Demand Studio has 39 features, 10 tabs, 6 forecasting algorithms, and a chemistry-themed loading animation. But it doesn't answer the one question every demand planner asks on Monday morning:

**"What needs my attention today, and what should I do about it?"**

The product shows data. It doesn't have a point of view.

---

## 2. The Shift: From Feature Dump to Value Flow

### Before (Current State)
```
User opens app → 10 tabs → Clicks around → Finds data → Interprets it → Decides → Does nothing in the tool
```

### After (Storyboard)
```
User opens app → Sees 23 exceptions ranked by $ impact → Clicks top exception
→ Sees root cause (bias, stockout risk, model drift) → Reviews recommended action
→ Accepts / Overrides / Escalates → Decision logged → Moves to next exception
```

Three screens. One flow. **See → Understand → Act.**

---

## 3. Design Philosophy

| Principle | Application |
|-----------|-------------|
| **Show impact in dollars, not percentages** | "Item X is over-forecasted by 12,000 units → $340K excess inventory risk" beats "WAPE: 42%" |
| **Exceptions, not dashboards** | Don't show 5,000 items. Show the 30 that matter. |
| **Recommend, don't just display** | "Switch to CatBoost model (would reduce WAPE from 42% to 18%)" is an action. A bar chart is not. |
| **Invisible machinery** | Clustering, backtesting, champion selection run automatically. The planner sees insights, not algorithms. |
| **Close the loop** | Every exception resolves to: Accept forecast, Override value, Escalate to manager, Snooze for 1 week. Decisions are logged and auditable. |

---

## 4. Architecture

### 4.1 Storyboard Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PLANNER STORYBOARD                              │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐ │
│  │  SCREEN 1     │    │  SCREEN 2         │    │  SCREEN 3          │ │
│  │  Exception    │ →  │  Root Cause &     │ →  │  Action &          │ │
│  │  Queue        │    │  Investigation    │    │  Resolution        │ │
│  │              │    │                   │    │                    │ │
│  │  "What needs  │    │  "Why is this     │    │  "What should I    │ │
│  │   attention?" │    │   happening?"     │    │   do about it?"    │ │
│  └──────────────┘    └──────────────────┘    └───────────────────┘ │
│         ↑                                              │            │
│         └──────────── Loop back to next item ──────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

```
Nightly automated pipeline (invisible to planner):
  ┌────────────────────────────────────┐
  │ 1. Refresh materialized views      │
  │ 2. Run champion selection          │
  │ 3. Compute exception scores        │
  │ 4. Rank by business impact ($)     │
  │ 5. Generate recommended actions    │
  │ 6. Store in exception_queue table  │
  └────────────────┬───────────────────┘
                   ↓
  Planner opens app → Exception Queue ready
```

### 4.3 Backend Schema

#### `exception_queue` table

```sql
CREATE TABLE exception_queue (
    exception_id    TEXT PRIMARY KEY,           -- ex_20260227_abc12345
    created_date    DATE NOT NULL,              -- date exception was generated
    dfu_ck          TEXT NOT NULL,              -- DFU composite key
    item_no         TEXT NOT NULL,
    location        TEXT NOT NULL,
    exception_type  TEXT NOT NULL,              -- forecast_bias | stockout_risk | accuracy_drop | excess_risk | model_drift | new_item
    severity        TEXT NOT NULL,              -- critical | high | medium | low
    impact_dollars  NUMERIC(12,2),             -- estimated $ impact (revenue at risk or excess cost)
    impact_units    INTEGER,                   -- units affected
    headline        TEXT NOT NULL,             -- human-readable 1-liner
    root_cause      JSONB NOT NULL,            -- structured root cause data
    recommended_action JSONB NOT NULL,         -- structured recommendation
    status          TEXT NOT NULL DEFAULT 'open',  -- open | accepted | overridden | escalated | snoozed | dismissed
    resolved_by     TEXT,                      -- planner username
    resolved_at     TIMESTAMPTZ,
    resolution_note TEXT,                      -- planner's note on decision
    override_value  NUMERIC(12,2),            -- if planner overrides the forecast
    snooze_until    DATE,                      -- if snoozed
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_exception_queue_date ON exception_queue(created_date);
CREATE INDEX idx_exception_queue_status ON exception_queue(status);
CREATE INDEX idx_exception_queue_severity ON exception_queue(severity);
CREATE INDEX idx_exception_queue_dfu ON exception_queue(dfu_ck);
CREATE INDEX idx_exception_queue_impact ON exception_queue(impact_dollars DESC NULLS LAST);
```

#### `planner_decisions` table (audit log)

```sql
CREATE TABLE planner_decisions (
    decision_id     TEXT PRIMARY KEY,          -- dec_20260227_abc12345
    exception_id    TEXT NOT NULL REFERENCES exception_queue(exception_id),
    dfu_ck          TEXT NOT NULL,
    decision_type   TEXT NOT NULL,             -- accept | override | escalate | snooze | dismiss
    previous_value  NUMERIC(12,2),            -- forecast value before decision
    new_value       NUMERIC(12,2),            -- forecast value after decision (if override)
    rationale       TEXT,                      -- planner's reason
    decided_by      TEXT NOT NULL,
    decided_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_planner_decisions_exception ON planner_decisions(exception_id);
CREATE INDEX idx_planner_decisions_dfu ON planner_decisions(dfu_ck);
CREATE INDEX idx_planner_decisions_date ON planner_decisions(decided_at);
```

---

## 5. Exception Types & Detection Rules

### 5.1 Exception Type Definitions

| Type | Trigger Rule | Severity Logic | Impact Calculation |
|------|-------------|----------------|-------------------|
| **forecast_bias** | Bias > ±20% over trailing 3 months | Critical if bias > ±40%, High if > ±20% | `abs(bias) × avg_monthly_units × unit_cost` |
| **stockout_risk** | Days of Supply < Lead Time × 1.5 | Critical if DOS < LT, High if DOS < LT × 1.5 | `daily_sales × stockout_days × unit_revenue` |
| **accuracy_drop** | WAPE increased > 10pp vs prior 3-month avg | Critical if drop > 20pp, High if > 10pp | `wape_delta × avg_monthly_revenue` |
| **excess_risk** | DOS > 90 days AND monthly sales declining | Critical if DOS > 180, High if > 90 | `excess_units × unit_cost × carrying_cost_rate` |
| **model_drift** | Champion model changed for >30% of recent months | High always | Estimated accuracy improvement in $ |
| **new_item** | DFU has < 3 months history, no champion model | Medium always | `avg_category_revenue` (proxy) |

### 5.2 Exception Scoring Formula

Each exception receives a **priority score** combining severity and dollar impact:

```python
priority_score = severity_weight × log10(max(impact_dollars, 1))

# severity_weight:
#   critical = 4.0
#   high     = 3.0
#   medium   = 2.0
#   low      = 1.0
```

Exceptions are presented **sorted by priority_score descending** — highest-impact critical issues first.

---

## 6. Screen 1: Exception Queue

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  DEMAND PLANNER STORYBOARD           Mon, Feb 27 2026           │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ 23 Open  │ │ $2.4M    │ │ 7        │ │ 85%      │           │
│  │ Exceptions│ │ At Risk  │ │ Critical │ │ Resolved │           │
│  │          │ │          │ │          │ │ This Week│           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│                                                                  │
│  Filter: [All Types ▼] [All Severity ▼] [My Items ▼]           │
│                                                                  │
│  ┌─ CRITICAL ─────────────────────────────────────────────────┐ │
│  │ ⚠ Stockout Risk: Item 100320 @ Loc 1401                    │ │
│  │   DOS: 8 days | Lead Time: 21 days | $142K revenue at risk  │ │
│  │   Recommended: Expedite PO + raise safety stock             │ │
│  │                                        [Investigate →]      │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ ⚠ Forecast Bias: Brand "Acme" Northeast region              │ │
│  │   +38% over-forecast for 4 consecutive months | $89K excess │ │
│  │   Recommended: Switch to CatBoost model (18% vs 42% WAPE)  │ │
│  │                                        [Investigate →]      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ HIGH ─────────────────────────────────────────────────────┐ │
│  │ ↗ Accuracy Drop: Cluster "seasonal_high_volume"             │ │
│  │   WAPE rose 14pp (from 22% to 36%) | 340 DFUs affected     │ │
│  │   Recommended: Review seasonal profile, retrain models      │ │
│  │                                        [Investigate →]      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### KPI Cards

| Card | Source | Meaning |
|------|--------|---------|
| **Open Exceptions** | `COUNT(*) WHERE status='open'` | Work remaining |
| **$ At Risk** | `SUM(impact_dollars) WHERE status='open'` | Business impact of unresolved items |
| **Critical Count** | `COUNT(*) WHERE severity='critical' AND status='open'` | Urgent items |
| **Resolved This Week** | `COUNT(*) WHERE status IN ('accepted','overridden','escalated') AND resolved_at > week_start` / total generated | Planner throughput |

### Exception Card Anatomy

Each card shows:
1. **Severity icon** — color-coded (red/amber/yellow/blue)
2. **Headline** — one-liner generated from structured data (e.g., "Stockout Risk: Item 100320 @ Loc 1401")
3. **Key metrics** — 2-3 numbers that justify the severity
4. **Recommended action** — one sentence, actionable
5. **[Investigate]** button — navigates to Screen 2

---

## 7. Screen 2: Root Cause Investigation

Clicking an exception card opens a focused investigation view with context from across the system — no tab-hopping required.

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  ← Back to Queue    STOCKOUT RISK: Item 100320 @ Loc 1401      │
│                                                         CRITICAL │
│                                                                  │
│  ┌─ CONTEXT ──────────────────────────────────────────────────┐ │
│  │ Item: Widget A (Brand: Acme, Category: Fasteners)          │ │
│  │ Location: Dallas Warehouse (Region: South)                  │ │
│  │ Cluster: high_volume_steady | Seasonality: non_seasonal     │ │
│  │ Champion Model: lgbm_cluster (WAPE: 18.2%)                 │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ WHY THIS IS FLAGGED ──────────────────────────────────────┐ │
│  │                                                             │ │
│  │  Days of Supply     Lead Time      Gap                      │ │
│  │  ████ 8 days        ████████ 21d   -13 days                │ │
│  │                                                             │ │
│  │  Trailing 3-Month Trend:                                    │ │
│  │  [Mini sparkline: DOS declining from 32 → 18 → 8]          │ │
│  │                                                             │ │
│  │  Root Cause: Actual demand exceeded forecast by 24% in      │ │
│  │  Jan and Feb. Champion model under-forecasted due to        │ │
│  │  regime change (new customer onboarded Q4 2025).            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ FORECAST vs ACTUAL (12-month) ────────────────────────────┐ │
│  │  [ECharts line chart: actual, champion forecast, +/- band]  │ │
│  │  [Shaded region showing under-forecast months]              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌─ MODEL COMPARISON (this DFU) ──────────────────────────────┐ │
│  │  Model            WAPE    Bias     Last 3mo    Recommend?   │ │
│  │  lgbm_cluster     18.2%   -24%     ↗ worse     current     │ │
│  │  catboost_global   12.1%   -8%      → stable    ★ switch    │ │
│  │  xgboost_cluster   15.5%   -15%     ↘ improving              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│                                  [Take Action →]               │
└─────────────────────────────────────────────────────────────────┘
```

### Data Sources (Composited from Existing Tables)

| Section | Source |
|---------|--------|
| Context | `dim_item` + `dim_location` + `dim_dfu` (cluster, seasonality) |
| Inventory metrics | `agg_inventory_monthly` (DOS, lead time) |
| Forecast vs Actual | `fact_external_forecast_monthly` (champion + actuals) |
| Model comparison | `backtest_lag_archive` (per-model per-DFU WAPE) |
| Root cause narrative | Generated by exception engine from structured signals |

---

## 8. Screen 3: Action & Resolution

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  ← Back to Investigation    RESOLVE: Item 100320 @ Loc 1401    │
│                                                                  │
│  ┌─ RECOMMENDED ACTION ──────────────────────────────────────┐  │
│  │                                                            │  │
│  │  Based on the analysis:                                    │  │
│  │                                                            │  │
│  │  1. Switch champion model from lgbm_cluster to             │  │
│  │     catboost_global (estimated WAPE improvement: 6pp)      │  │
│  │                                                            │  │
│  │  2. Increase next-month forecast from 8,200 to 10,200      │  │
│  │     units (+24% adjustment for new customer volume)        │  │
│  │                                                            │  │
│  │  3. Flag for safety stock review (DOS < LT)                │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ YOUR DECISION ───────────────────────────────────────────┐  │
│  │                                                            │  │
│  │  ( ) Accept recommendation as-is                           │  │
│  │  ( ) Override forecast value: [________] units             │  │
│  │      Reason: [________________________________]            │  │
│  │  ( ) Escalate to manager                                   │  │
│  │      Note: [________________________________]              │  │
│  │  ( ) Snooze for: [1 week ▼]                               │  │
│  │  ( ) Dismiss (not actionable)                              │  │
│  │      Reason: [________________________________]            │  │
│  │                                                            │  │
│  │                          [Submit Decision]                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌─ DECISION HISTORY (this DFU) ─────────────────────────────┐  │
│  │  Feb 20: Accepted forecast (auto) — J. Smith              │  │
│  │  Jan 15: Overridden to 9,500 units — J. Smith             │  │
│  │          "Holiday promo expected to spike demand"           │  │
│  │  Dec 10: Escalated to manager — J. Smith                   │  │
│  │          "New customer PO not in system yet"                │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Decision Types

| Decision | Effect | Audit Trail |
|----------|--------|-------------|
| **Accept** | Mark exception resolved. No forecast change. | Logged with timestamp + planner. |
| **Override** | Write new forecast value to `fact_external_forecast_monthly` as `model_id='planner_override'`. Mark resolved. | Logged with old value, new value, and rationale. |
| **Escalate** | Mark as escalated. Appears in manager's queue with planner's note. | Logged with note. |
| **Snooze** | Hide from queue until snooze date. Re-surfaces automatically. | Logged with snooze duration. |
| **Dismiss** | Mark as not actionable. Removed from queue. | Logged with reason (prevents repeated generation). |

---

## 9. Exception Generation Engine

### `scripts/generate_exceptions.py`

Runs nightly (or on-demand via job scheduler). Computes all exception types, scores them, and writes to `exception_queue`.

```python
"""
Exception generation pipeline:
1. Load latest inventory, forecast, accuracy, and DFU data
2. Apply detection rules for each exception type
3. Score and rank by business impact
4. Generate human-readable headlines and root cause structures
5. Insert into exception_queue (idempotent: DELETE + INSERT for today's date)
"""

def generate_exceptions(pool, config):
    today = date.today()

    # Clear today's unresolved exceptions (re-generate fresh)
    delete_open_exceptions(pool, today)

    exceptions = []

    # 1. Stockout risk: DOS < LT threshold
    exceptions += detect_stockout_risk(pool, config)

    # 2. Forecast bias: sustained over/under-forecast
    exceptions += detect_forecast_bias(pool, config)

    # 3. Accuracy drops: WAPE degradation vs trailing avg
    exceptions += detect_accuracy_drops(pool, config)

    # 4. Excess inventory risk: high DOS + declining sales
    exceptions += detect_excess_risk(pool, config)

    # 5. Model drift: champion model instability
    exceptions += detect_model_drift(pool, config)

    # 6. New items: insufficient history for forecasting
    exceptions += detect_new_items(pool, config)

    # Score and rank
    scored = score_exceptions(exceptions)

    # Persist (skip items that were dismissed or snoozed)
    insert_exceptions(pool, scored, today)
```

### Configuration: `config/exception_config.yaml`

```yaml
exceptions:
  # Stockout risk
  stockout:
    dos_lt_ratio: 1.5          # flag when DOS < LT × this ratio
    critical_ratio: 1.0        # critical when DOS < LT × this ratio
    min_daily_sales: 1.0       # skip items with trivial sales

  # Forecast bias
  bias:
    threshold_pct: 20          # flag when abs(bias) > this %
    critical_pct: 40           # critical when abs(bias) > this %
    trailing_months: 3         # evaluate over this window
    min_actual_units: 100      # skip low-volume items

  # Accuracy drop
  accuracy_drop:
    wape_increase_pp: 10       # flag when WAPE increased by this many pp
    critical_increase_pp: 20   # critical threshold
    baseline_months: 3         # compare against this trailing window

  # Excess inventory
  excess:
    dos_threshold: 90          # flag when DOS > this
    critical_dos: 180          # critical threshold
    carrying_cost_rate: 0.25   # annual carrying cost as % of unit cost

  # Model drift
  model_drift:
    change_threshold_pct: 30   # flag when champion changed > this % of recent months
    lookback_months: 6         # window for evaluating champion stability

  # New items
  new_item:
    min_history_months: 3      # flag items with fewer months than this

  # Scoring
  scoring:
    severity_weights:
      critical: 4.0
      high: 3.0
      medium: 2.0
      low: 1.0

  # Generation
  max_exceptions_per_day: 200  # cap to prevent overwhelming the planner
  auto_dismiss_after_days: 30  # auto-dismiss unresolved exceptions older than this
```

---

## 10. API Endpoints

### Exception Queue Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/storyboard/exceptions` | GET | List open exceptions (paginated, filterable by type/severity) |
| `/storyboard/exceptions/{id}` | GET | Get exception detail with root cause and recommendation |
| `/storyboard/exceptions/{id}/resolve` | POST | Submit a planner decision (accept/override/escalate/snooze/dismiss) |
| `/storyboard/exceptions/kpis` | GET | Summary KPIs (open count, $ at risk, critical count, resolution rate) |
| `/storyboard/exceptions/generate` | POST | Trigger exception generation (on-demand) |
| `/storyboard/decisions` | GET | Audit log of planner decisions (paginated, filterable by DFU/planner/date) |
| `/storyboard/investigation/{id}` | GET | Composite investigation data (DFU context, inventory, forecasts, model comparison) |

### Example: GET /storyboard/exceptions

```json
{
  "exceptions": [
    {
      "exception_id": "ex_20260227_abc12345",
      "exception_type": "stockout_risk",
      "severity": "critical",
      "headline": "Stockout Risk: Item 100320 @ Loc 1401-BULK",
      "impact_dollars": 142000,
      "impact_units": 4200,
      "item_no": "100320",
      "location": "1401-BULK",
      "priority_score": 20.6,
      "recommended_action": {
        "summary": "Expedite PO + raise safety stock",
        "details": [
          "Switch champion model to catboost_global (WAPE: 12.1% vs 18.2%)",
          "Increase next-month forecast by 24% (new customer volume)",
          "Flag for safety stock review (DOS 8d < LT 21d)"
        ]
      },
      "created_date": "2026-02-27",
      "status": "open"
    }
  ],
  "total": 23,
  "page": 1,
  "page_size": 20
}
```

### Example: POST /storyboard/exceptions/{id}/resolve

```json
{
  "decision": "override",
  "override_value": 10200,
  "rationale": "New customer PO confirmed for 2,000 units/month starting March",
  "decided_by": "j.smith"
}
```

---

## 11. Frontend Implementation

### 11.1 New Tab: Storyboard (replaces Overview as default landing)

The Storyboard tab becomes the **default landing page** — the first thing a planner sees. The existing Overview/Dashboard remains accessible but is no longer the entry point.

**Sidebar placement:** First item in sidebar, section "Planning."

### 11.2 Component Architecture

```
StoryboardTab.tsx
├── ExceptionKpiCards.tsx          -- 4 KPI summary cards
├── ExceptionFilters.tsx           -- type/severity/assignment filter bar
├── ExceptionQueue.tsx             -- scrollable exception card list
│   └── ExceptionCard.tsx          -- individual exception card
├── InvestigationPanel.tsx         -- slide-over or full-page investigation view
│   ├── DfuContextCard.tsx         -- item/location/cluster/seasonality context
│   ├── InventoryGauge.tsx         -- DOS vs Lead Time visual gauge
│   ├── ForecastActualChart.tsx    -- 12-month forecast vs actual (ECharts)
│   ├── ModelComparisonTable.tsx   -- per-model WAPE/bias for this DFU
│   └── RootCauseNarrative.tsx     -- generated explanation text
└── ResolutionForm.tsx             -- decision form (accept/override/escalate/snooze/dismiss)
    └── DecisionHistory.tsx        -- past decisions for this DFU
```

### 11.3 State Management

Minimal state — the storyboard is server-driven:

| State | Source | Pattern |
|-------|--------|---------|
| Exception list | `GET /storyboard/exceptions` | TanStack Query, staleTime: 30s |
| KPIs | `GET /storyboard/exceptions/kpis` | TanStack Query, staleTime: 30s |
| Investigation data | `GET /storyboard/investigation/{id}` | TanStack Query, on-demand |
| Selected exception | Local `useState` (single string ID) | — |
| Filter state | URL params via `useUrlState` | type, severity, assignment |

### 11.4 Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `j` / `k` | Navigate up/down in exception queue |
| `Enter` | Open investigation for selected exception |
| `Esc` | Back to queue from investigation |
| `a` | Accept recommendation |
| `o` | Open override form |
| `e` | Escalate |
| `s` | Snooze |

---

## 12. Integration with Existing Systems

### 12.1 Job Scheduler Integration

Exception generation registers as a job type in `common/job_registry.py`:

```python
JOB_TYPE_REGISTRY["exception_generation"] = {
    "group": "exception",
    "label": "Exception Generation",
    "callable": run_exception_generation,
}
```

Scheduled nightly at 2:00 AM (after backtest/champion jobs complete). Can also be triggered manually via "Refresh Exceptions" button.

### 12.2 Champion Selection Integration

When a planner accepts a model-switch recommendation:
1. The exception engine logs the decision
2. A `model_override` record is created for that DFU
3. Next champion selection run respects the override (planner's model wins for that DFU until revoked)

### 12.3 Inventory Integration

Stockout and excess risk exceptions pull directly from `agg_inventory_monthly`:
- `eom_on_hand`, `avg_daily_sales` → DOS computation
- `latest_lead_time` → LT comparison
- `monthly_sales` trend → excess detection (declining sales + high DOS)

### 12.4 Clustering Integration

Exception headlines include cluster context:
- "Cluster: high_volume_steady" helps planners recognize patterns
- Accuracy drops that affect an entire cluster are grouped into a single exception with affected DFU count

### 12.5 Global Filters

The storyboard respects global filters (brand, category, location, market, channel). A regional planner filtering to "South" only sees exceptions for their items.

---

## 13. What Gets Hidden (Steve Jobs' "Cut List")

The storyboard doesn't add a 12th tab. It **absorbs and hides** existing complexity:

| Currently Visible | In Storyboard | Where It Goes |
|-------------------|---------------|---------------|
| 6 model names (LGBM, CatBoost, etc.) | "Champion Model" / "Recommended Model" | Model names only visible in investigation drill-down |
| Clustering tab What-If scenarios | Automatic clustering, insights surfaced as exceptions | Runs in background, planner sees "cluster accuracy dropped" |
| Jobs tab | Invisible | Notifications only; scheduling happens in settings |
| Benchmarking panel | Removed | Engineering-only concern |
| Market Intelligence tab | Retained but secondary | Accessible from sidebar, not in main flow |
| 10 sidebar tabs | 6 tabs: Storyboard, Explorer, Accuracy, Inventory, Analysis, Settings | Chat embedded as floating widget; Jobs/Clusters/MarketIntel under "Advanced" |

---

## 14. Metrics That Prove Value

### Planner Productivity KPIs (tracked in `planner_decisions`)

| Metric | Definition | Target |
|--------|-----------|--------|
| **Exceptions resolved / week** | Decisions submitted per planner per week | > 80% of generated exceptions |
| **Time to resolution** | Median time from exception creation to decision | < 4 hours for critical, < 2 days for high |
| **Override rate** | % of exceptions where planner overrides vs accepts | 15-25% (too high = bad models, too low = rubber-stamping) |
| **$ impact resolved** | Sum of `impact_dollars` for resolved exceptions | Track weekly trend |
| **Forecast accuracy lift** | WAPE improvement from planner overrides vs unmodified champion | Positive = planners add value |

---

## 15. Phased Rollout

### Phase 1: Exception Queue + Investigation (4 weeks)

- `exception_queue` table + `planner_decisions` table DDL
- Exception generation engine (4 exception types: stockout, bias, accuracy drop, excess)
- Storyboard tab with KPI cards + exception list
- Investigation panel (DFU context, forecast chart, model comparison)
- `GET /storyboard/exceptions` + `GET /storyboard/investigation/{id}` endpoints

### Phase 2: Action Framework (2 weeks)

- Resolution form (accept/override/escalate/snooze/dismiss)
- `POST /storyboard/exceptions/{id}/resolve` endpoint
- Decision history display
- `planner_decisions` audit log + `GET /storyboard/decisions` endpoint
- Override writes to `fact_external_forecast_monthly` as `model_id='planner_override'`

### Phase 3: Automation + Polish (2 weeks)

- Job scheduler integration (nightly exception generation)
- Model drift + new item exception types
- Keyboard shortcuts (j/k/Enter/Esc/a/o/e/s)
- Sidebar reorganization (Storyboard as default, "Advanced" section for power-user tabs)
- Exception generation config UI in Settings

### Phase 4: Value Tracking (1 week)

- Planner productivity KPIs dashboard (embedded in Storyboard)
- Weekly digest: "You resolved 45 exceptions worth $1.2M this week"
- Override accuracy tracking: did planner overrides improve or hurt forecast accuracy?

---

## 16. Key Files (Planned)

| File | Purpose |
|------|---------|
| `sql/022_create_exception_queue.sql` | DDL for `exception_queue` + `planner_decisions` tables |
| `config/exception_config.yaml` | Exception detection thresholds and scoring |
| `common/exception_engine.py` | Exception detection rules, scoring, headline generation |
| `scripts/generate_exceptions.py` | Nightly exception generation script |
| `api/routers/storyboard.py` | 7 REST API endpoints for storyboard |
| `frontend/src/tabs/StoryboardTab.tsx` | Main storyboard tab component |
| `frontend/src/components/ExceptionCard.tsx` | Individual exception card |
| `frontend/src/components/InvestigationPanel.tsx` | Root cause investigation view |
| `frontend/src/components/ResolutionForm.tsx` | Planner decision form |
| `tests/unit/test_exception_engine.py` | Exception detection rule tests |
| `tests/api/test_storyboard.py` | Storyboard API endpoint tests |
| `frontend/src/tabs/__tests__/StoryboardTab.test.tsx` | Storyboard tab component tests |

---

## 17. Design Rationale

| Decision | Why |
|----------|-----|
| Dollars, not percentages, as headline metric | Steve Jobs: "The product doesn't have a point of view." Dollars ARE the point of view — they tell you what matters. |
| Exception queue, not dashboard | UX Expert: "Where's the user journey?" The queue IS the journey — work top to bottom. |
| Three screens (Queue → Investigate → Act) | Steve Jobs: "Three screens, not ten tabs." |
| Composite investigation from existing tables | Supply Chain Expert: data is already there; what's missing is the operational presentation. |
| Decision audit log | Supply Chain Expert: "Where do planners document why they changed a number?" |
| Nightly generation + on-demand refresh | Keeps queue fresh without real-time complexity. Job scheduler already exists. |
| Override as `model_id='planner_override'` | Reuses existing fact table pattern — zero downstream changes (same as champion/ceiling approach). |
| Keyboard shortcuts for triage | UX Expert: "No keyboard shortcuts for common actions." j/k/Enter is the vim-style triage flow planners will use 100x/day. |
| Severity-weighted $ scoring | Ensures critical stockout on a $500K item ranks above medium bias on a $2K item. |
| Max 200 exceptions/day cap | Prevents alert fatigue. Force-ranks to show only what truly matters. |
| Phased rollout starting with read-only | De-risks the write path (overrides). Phase 1 delivers value even without action framework. |

---

## 18. CLI Commands (Planned)

```bash
# Exception pipeline
make exception-schema         # Apply DDL for exception_queue + planner_decisions tables
make exception-generate       # Generate today's exceptions from current data
make exception-stats          # Print exception summary (open/resolved/by-type)
make storyboard-all           # schema + generate (full pipeline)
```

---

## 19. Vite Proxy Addition

Add `/storyboard` prefix to `frontend/vite.config.ts`:

```typescript
'/storyboard': {
  target: 'http://127.0.0.1:8000',
  changeOrigin: true,
},
```

---

## 20. Success Criteria

The feature is successful when:

1. A demand planner can open the app and within 30 seconds understand what needs attention today
2. The planner resolves exceptions without visiting any other tab for 80%+ of cases
3. Decision audit trail enables manager review of all forecast adjustments
4. Planner overrides measurably improve forecast accuracy (tracked via `planner_override` model_id)
5. The app has a point of view: it tells you what's wrong, why, and what to do — not just "here's some data"


---

## Examples

### Example: Exception queue — critical items needing planner action

```bash
curl -s "http://localhost:8000/storyboard/exceptions?severity=critical&limit=5" | jq '.exceptions[0]'
# {
#   "id": "ex_20260227_100320_1401BULK",
#   "item_no": "100320", "loc": "1401-BULK",
#   "item_desc": "CABERNET SAUV 750ML",
#   "severity": "critical",
#   "reason": "Stockout risk: 12 days of supply (threshold: 14)",
#   "recommended_action": "Expedite PO 98421 or request emergency transfer",
#   "dos": 12.4, "woc": 1.8, "qty_on_hand": 980
# }
```

### Example: Resolve an exception

```bash
curl -s -X POST "http://localhost:8000/storyboard/exceptions/ex_20260227_100320_1401BULK/resolve" \
  -H "Content-Type: application/json" \
  -d '{
    "decision": "expedite",
    "rationale": "Customer PO 98421 confirmed — expediting from supplier",
    "decided_by": "j.smith"
  }'
# {"resolved": true, "exception_id": "ex_20260227_100320_1401BULK", "decision": "expedite"}
```

### Example: Exception scoring formula

```python
# Exceptions scored by severity = combination of DOS risk + forecast deviation
def score_exception(dos, woc, forecast_error_pct, bias_direction):
    stockout_score = max(0, (14 - dos) / 14)  # 0-1, peaks at DOS=0
    excess_score   = max(0, (dos - 90) / 90)  # 0-1, peaks at DOS=180+
    accuracy_score = abs(forecast_error_pct) / 100
    return stockout_score * 0.6 + excess_score * 0.2 + accuracy_score * 0.2
```

### Example: Exception generation scheduled job

```bash
# Register daily exception generation at 6AM (after nightly forecast refresh)
curl -s -X POST http://localhost:8000/jobs/schedule \
  -d '{"job_type": "generate_exceptions", "cron": "0 6 * * *", "label": "Daily Exception Refresh"}'
```
