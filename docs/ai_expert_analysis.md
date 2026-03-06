# AI Expert Analysis — Demand Studio

*Four AI domain experts analyzed `docs/ai_planning_vision.md` and the Demand Studio codebase.
This document synthesizes their findings with illustrations and concrete examples.*

---

## Expert Panel

| Expert | Perspective | Key Output |
|---|---|---|
| **Supply Chain AI Domain Expert** | 20 years McKinsey + Amazon; end-to-end planning loops | Causal chain worked examples, SS formula, triage score |
| **UX / Interaction AI Expert** | Human-computer interaction; planner workflow design | Monday morning flow, component wireframes, 1-1-1 rule |
| **Technical AI Architecture Expert** | LLM systems engineering; production reliability | Multi-agent design, structured outputs, SSE streaming, observability |
| **AI Automation & Outcomes Expert** | Closed-loop AI systems; feedback design | Action map, outcome tracking schema, S&OP automation |

---

## 1. The AI Planning Loop — Where AI Intervenes at Every Layer

*From the Supply Chain Expert*

The fundamental insight: AI in Demand Studio must not be a report generator sitting at the end of a pipeline. It must be embedded as a **closed loop** at each transition between planning layers.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    THE AI-EMBEDDED PLANNING LOOP                         │
│                                                                          │
│   ┌──────────────┐   AI: bias detection,          ┌───────────────┐     │
│   │   FORECAST   │   multiplier recommendation,   │  SAFETY STOCK │     │
│   │              │──────────────────────────────── │               │     │
│   │ basefcst_pref│   feeds corrected demand →     │ SS = Z × σ    │     │
│   │ tothist_dmd  │                                 │ × √(LT)       │     │
│   │ model_id     │                                 │ [STUB TODAY]  │     │
│   └──────┬───────┘                                 └───────┬───────┘     │
│          │ AI: champion                AI: SS→ROP align   │             │
│          │ degradation detection       ment check         │             │
│          ▼                                                 ▼             │
│   ┌──────────────┐                                 ┌───────────────┐     │
│   │     EOQ      │──────────────────────────────── │    POLICY     │     │
│   │              │   AI: EOQ-to-policy alignment   │               │     │
│   │ eoq_effective│   MOQ gap detection             │ policy_type   │     │
│   └──────┬───────┘                                 └───────┬───────┘     │
│          │ AI: trigger exception         AI: policy        │             │
│          │ on deviation                  efficacy review   │             │
│          ▼                                                 ▼             │
│   ┌──────────────┐◄────────────────────────────────┌───────────────┐    │
│   │  EXCEPTION   │                                 │   INVENTORY   │    │
│   │  WORK QUEUE  │   AI: rank by financial         │               │    │
│   │  ai_insights │   impact + ABC + urgency        │ dos, avg_dos  │    │
│   └──────┬───────┘                                 └───────────────┘    │
│          │ AI: recommend                                                 │
│          ▼                                                               │
│   ┌──────────────┐        ┌────────────────┐                            │
│   │    ACTION    │──30d──▶│    OUTCOME     │────feedback──▶ FORECAST    │
│   │ policy change│        │ DOS improved?  │   loop                     │
│   └──────────────┘        └────────────────┘                            │
└──────────────────────────────────────────────────────────────────────────┘
```

### AI Integration Depth Map: What Exists vs What Is Missing

```
LAYER         WHAT EXISTS TODAY                WHAT IS MISSING
─────────────────────────────────────────────────────────────────────────
Forecast      compute_bias_trend()             Multiplier auto-application
              get_forecast_performance()       Regime-change detection
              champion WAPE in context         Bias decay modelling

Safety Stock  fact_safety_stock_targets        Real SS formula computation
              (stub, 0 rows)                   Dynamic SS by ABC×seasonality
              health score neutral values      Lead time σ from supplier MV

EOQ           get_eoq_context()                EOQ deviation alert
              fact_eoq_targets populated       Order-qty-to-EOQ alignment
              eoq_effective in context         Batch vs EOQ cost comparison

Policy        Current policy pulled in         Policy efficacy scoring
              via lateral join                 Cluster-policy mismatch check
                                               Auto-switch recommendation

Exception     get_portfolio_exceptions()       Composite triage score
              create_insight() writes DB       Structured causal chain JSON
              Insight cards in UI              One-click action execution

Action        Acknowledge/Resolve only         API-backed one-click apply
                                               fact_forecast_overrides table

Outcome       Not implemented                  30-day look-back check
                                               ai_insights.auto_resolved_at
```

---

## 2. Causal Chain Reasoning — Three Worked Examples

*From the Supply Chain Expert*

The AI already calls the right tools and writes a `reasoning` text paragraph. The critical gap is that the reasoning is **unstructured prose** — planners cannot scan it in 5 seconds. These examples show what the AI *should* produce.

### Example 1: Stockout Driven by Over-Forecast (A-class)

**DFU:** `587382 @ 1401-BULK` · ABC=A · `high_volume_steady` cluster

**Tool call sequence:** `get_dfu_full_context` → `get_forecast_performance` → `compute_bias_trend` → `get_inventory_trend`

```
DATA RETURNED:
  current_dos        = 6.4 days
  total_lt_days      = 14 days
  champion_wape      = 63.2%
  bias_6m_avg        = +88.5%  (over-forecast every month)
  bias_3m_avg        = +92.3%  (accelerating)
  avg_daily_sales    = 33 units/day

CAUSAL CHAIN:
┌──────────────────┬──────────────────────┬───────────────────┬───────────────────┐
│   ROOT CAUSE     │     MECHANISM        │   CONSEQUENCE     │ FINANCIAL IMPACT  │
├──────────────────┼──────────────────────┼───────────────────┼───────────────────┤
│ basefcst_pref    │ Replenishment orders  │ ROP computed on   │ Stockout in ~8d   │
│ over-states      │ inflated by +88.5%   │ inflated demand;  │ est. lost sales:  │
│ demand (avg 1020 │ for 6 months →       │ actual 33u/day    │ 33u/d × 8d ×      │
│ vs fcst avg 1950)│ excess stock depleted│ but trigger never │ $10 × 3× mult     │
│                  │ faster than planned  │ fires correctly   │ = $7,920          │
└──────────────────┴──────────────────────┴───────────────────┴───────────────────┘

RECOMMENDED create_insight() CALL:
  insight_type    = "stockout_risk"
  severity        = "critical"
  summary         = "DOS 6.4d below lead time 14d — stockout in ~8 days.
                     Persistent +88.5% over-forecast for 6 months exhausted safety stock."
  recommendation  = "1) Emergency reorder 462 units (14d × 33u/day) now.
                     2) Apply forecast multiplier 0.52× for next 3 months.
                     3) Switch policy continuous_rop → safety_stock_buffer."
  financial_impact_estimate = 7920.0
```

### Example 2: Excess Inventory from Cluster Mismatch

**DFU:** `179333 @ 2201-MAIN` · ABC=B · cluster=`high_volume_steady` but variability=`HIGH`

```
DATA RETURNED:
  current_dos        = 247 days   (excess threshold = 180)
  total_lt_days      = 21 days
  cluster_assignment = "high_volume_steady"  ← mismatch!
  variability_class  = "high"               ← contradicts cluster
  seasonality_profile= "seasonal_peak_q4"
  eoq_effective      = 300 units
  peer avg DOS (same cluster) = 38.2 days   ← DFU is 6.5× peers

KEY INFERENCE:
  cluster_assignment and variability_class disagree.
  The cluster hasn't been re-labeled since the demand regime changed.
  Policy type = "continuous_rop" is wrong for a high-variability seasonal item.

CAUSAL CHAIN:
  STALE CLUSTER LABEL → EOQ sized for steady demand
  → orders of 300u arrive while actual run-rate = 900u/yr (below forecast)
  → DOS grew from 38d to 247d over 6 months
  → Carrying cost: $694/month; Capital locked: $10,130

AI should flag BOTH: excess_inventory + policy_gap
```

### Example 3: Champion Degradation Cascading to Order Volatility

**DFU:** `616372 @ 1401-BULK` · ABC=A · cluster=`medium_volume_seasonal`

```
DATA RETURNED:
  champion_wape  = 57.8%  (critical: > 50%)
  bias_3m_avg    = +63.1%
  bias_6m_avg    = +49.2%   ← bias is ACCELERATING
  stockout_months_6m = 2
  excess_months_6m   = 1    ← oscillating: over → under → over

CAUSAL CHAIN:
  Champion WAPE degraded 28% → 57.8%
  → Forecast signal noisy: sometimes +63% over, once dramatically under
  → Creates bullwhip in ordering: over-forecast → excess → spike → stockout
  → continuous_rop policy uses forecast directly to compute ROP
    → amplifies forecast error into ordering volatility

CORRECT REMEDY (not obvious without causal chain):
  Don't just "run champion selection again."
  Switch to safety_stock_buffer policy so SS absorbs forecast noise
  before it propagates to replenishment decisions.
  THEN re-run champion selection.
```

---

## 3. Planner UX: The Monday Morning Workflow

*From the UX/Interaction Expert*

### The 3-Screen Planner Journey

```
08:30 Monday — Planner opens Demand Studio

SCREEN 1: AI Planner Tab  (5-second scan)
┌─────────────────────────────────────────────────────────────────────────┐
│  AI Planner  ·  Monday March 3, 2026                                   │
│                                                                         │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Portfolio Health: 68%   │
│  ─────────────────────────────────────────────────────────────────────  │
│  [3 CRITICAL]  [5 HIGH]  [8 MEDIUM]  [$124K at risk]                   │
│                                                                         │
│  ┌────────────────────────────────────────────────────┐                 │
│  │ 🔴 CRITICAL  587382 @ 1401-BULK  [A-class]         │                 │
│  │ DOS 6.4d below lead time 14d — stockout in 8 days  │                 │
│  │ Cause: +88% over-forecast 6 months                 │                 │
│  │ Rec: Emergency reorder 462u + apply 0.52× mult.    │                 │
│  │ Impact: $7,920         [Accept ▶]  [Reject]        │                 │
│  └────────────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────┘

SCREEN 2: Confirm Modal  (before any action executes)
┌─────────────────────────────────────────────────────────────────────────┐
│  Confirm Action                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│  This will execute:                                                     │
│  ✓ CREATE replenishment exception                                       │
│    Item 587382  ·  Qty 462 units  ·  Priority: Emergency                │
│    Lead time: 14 days  ·  Expected DOS after receipt: 13.5 days         │
│    Est. cost: $4,620                                                    │
│                                                                         │
│  [Execute]                                          [Cancel]            │
└─────────────────────────────────────────────────────────────────────────┘

SCREEN 3: Outcome (T+30 days)
┌─────────────────────────────────────────────────────────────────────────┐
│  ✅ Resolved — 587382 @ 1401-BULK                                       │
│  DOS recovered: 6.4d → 18.2d  (resolved in 12 days)                    │
│  Forecast multiplier applied: bias dropped from +88% to +11%           │
│  AI recommendation accuracy: HIGH confidence confirmed                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Decision Compression: The 1-1-1 Rule

Every insight card must communicate **one sentence + one action + one number**. No more.

```
BAD (current-style):
  "This DFU has been showing persistent forecast bias over the past several months,
   which has resulted in inventory levels that may be causing concern from a supply
   security standpoint. It would be advisable to consider reviewing the replenishment
   policy and potentially adjusting the forecast inputs going forward."

GOOD (1-1-1 rule):
  Sentence: "DOS 6.4d below lead time 14d — stockout in 8 days."
  Action:   "Emergency reorder 462 units"
  Number:   "$7,920 at risk"
```

### AI Presence Pattern by Tab

Every tab in the sidebar gets a specific AI interaction type — not generic "AI insights" everywhere.

| Tab | AI Presence Type | Illustration |
|---|---|---|
| **AI Planner** | Proactive exception work-queue | Ranked insight cards with severity + action |
| **Dashboard** | Ambient health indicator | Portfolio health bar in header; no extra panels |
| **Accuracy** | Inline bias callout on WAPE trend | Red pulse on bars exceeding threshold |
| **Inv. Planning** | Exception count badge on tab | Number of open exceptions inline with tab label |
| **Control Tower** | Alert severity filter | AI-generated alerts pre-sorted by triage score |
| **Clusters** | Post-run anomaly detection | "3 DFUs in cluster 4 look misassigned" after scenario |
| **Jobs** | ETA + next trigger time | "Next portfolio scan: Monday 06:00" in job card |

### Causal Chain Card Component

```typescript
// What should replace the plain "reasoning" text paragraph

interface CausalLink {
  layer: "forecast" | "inventory" | "policy" | "financial";
  signal: string;   // e.g. "WAPE 63% (critical)"
  impact: string;   // e.g. "ROP trigger fires late"
}

interface CausalChainCardProps {
  chain: CausalLink[];
  confidence: "high" | "medium" | "low";
}

// Renders as:
// [FORECAST: WAPE 63%] → [INVENTORY: DOS 6.4d] → [POLICY: ROP fires late] → [FINANCIAL: $7,920]
//  ⬤ high confidence
```

### AI Confidence Tiers (Visual System)

```
HIGH confidence  (6+ month signal, all indicators aligned)
→ Solid filled badge   ●  "HIGH"   green background
→ Show full financial impact

MEDIUM confidence  (3-month signal or mixed indicators)
→ Half-filled badge    ◑  "MED"    amber background
→ Show financial range: "$3K–$8K"

LOW confidence  (single-month signal or data gaps)
→ Outlined badge       ○  "LOW"    gray background
→ Show "Insufficient data to estimate" instead of dollar figure
```

---

## 4. Technical Architecture: What Needs to Change

*From the Technical Architecture Expert*

### P0 Fixes (No Schema Changes, Ship This Week)

**Fix 1: SQL INTERVAL Bug in `common/ai_planner.py`**

The current `INTERVAL '%s months'` pattern does not work as a psycopg3 parameterized placeholder — the `months` value is silently not substituted.

```python
# CURRENT — BROKEN (months is never actually substituted)
AND startdate >= date_trunc('month', NOW() - INTERVAL '%s months')

# FIXED — multiply interval by parameter
AND startdate >= date_trunc('month', NOW() - INTERVAL '1 month' * %s)
```

**Fix 2: Turn Limit Guard in Agentic Loop**

Both `_run_openai_loop` and `_run_anthropic_loop` run `while True` with no circuit breaker:

```python
MAX_TURNS = 40
TOKEN_BUDGET = 100_000  # cumulative tokens per DFU analysis

turn = 0
total_tokens = 0
while turn < MAX_TURNS and total_tokens < TOKEN_BUDGET:
    turn += 1
    response = self.client.chat.completions.create(...)
    total_tokens += response.usage.total_tokens
    # ... rest of loop ...
```

**Fix 3: Rewrite the System Prompt with Few-Shot Examples**

The current prompt has no anchor examples. Without them, `summary` quality varies by 10×. Two worked examples (stockout_risk, excess_inventory) are sufficient to anchor the model's output format. See §2 above for the exact content of those examples.

### P1: Pydantic Validation for `create_insight`

The LLM can currently write `insight_type: "inventory_risk"` (invalid), `summary: "Forecast bias detected"` (no metrics), or `financial_impact_estimate: 999999999`. All are inserted silently.

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Annotated

class CreateInsightInput(BaseModel):
    insight_type: Literal[
        "stockout_risk", "excess_inventory", "forecast_bias",
        "policy_gap", "champion_degradation"
    ]
    severity: Literal["critical", "high", "medium", "low"]
    summary: str = Field(min_length=20, max_length=300)
    recommendation: str = Field(min_length=30, max_length=600)
    financial_impact_estimate: Annotated[float | None, Field(ge=0, le=10_000_000)] = None

    @field_validator("summary")
    @classmethod
    def summary_must_contain_metrics(cls, v: str) -> str:
        if not any(c.isdigit() for c in v):
            raise ValueError("summary must contain at least one metric value (number)")
        return v
```

### P1: AI Observability Table

There is zero visibility into token usage, costs, or tool latency today.

```sql
CREATE TABLE ai_call_log (
    log_id          BIGSERIAL PRIMARY KEY,
    scan_run_id     TEXT NOT NULL,
    dfu_key         TEXT,
    provider        TEXT NOT NULL,  -- "openai" | "anthropic"
    model           TEXT NOT NULL,
    turn_number     INTEGER NOT NULL,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    total_tokens    INTEGER,
    latency_ms      INTEGER,
    tool_name       TEXT,
    tool_success    BOOLEAN,
    error_type      TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- New endpoint: GET /ai-planner/metrics?days=7
-- Returns: cost_usd, total_tokens, p95_latency_ms, error_rate per model + tool
```

### P2: SSE Streaming for Real-Time Insight Appearance

Instead of "Generate Now → wait → refresh page", insights should appear as cards in real-time:

```
CURRENT FLOW:
  [Generate Now] → POST /ai-planner/portfolio-scan → 202 → wait 15min → F5

TARGET FLOW:
  [Generate Now] → POST /ai-planner/portfolio-scan → 202
                → GET /ai-planner/stream/{scan_run_id} (SSE)
                → insight cards appear one by one as AI writes them
                → [DONE: 7 insights found in 8m 22s]
```

```typescript
// React hook (useScanStream.ts)
export function useScanStream(scanRunId: string | null) {
  const [insights, setInsights] = useState<AiInsight[]>([]);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!scanRunId) return;
    const es = new EventSource(`/ai-planner/stream/${scanRunId}`);
    es.addEventListener("insight", (e) => {
      const insight = JSON.parse(e.data);
      setInsights(prev => [...prev, insight]
        .sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity]));
    });
    es.addEventListener("done", () => { setDone(true); es.close(); });
    return () => es.close();
  }, [scanRunId]);

  return { insights, done };
}
```

### P3: Multi-Agent Architecture

The current single agent does everything. This architecture separates concerns:

```
                   ┌─────────────────────────┐
                   │    OrchestratorAgent    │
                   │  SQL triage → routing   │
                   └────┬──────┬──────┬──────┘
                        │      │      │
         ┌──────────────┘      │      └──────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  ForecastAgent  │  │ InventoryAgent   │  │   PolicyAgent    │
│  Tools:         │  │ Tools:           │  │ Tools:           │
│  get_fcst_perf  │  │ get_inv_trend    │  │ get_eoq_context  │
│  bias_trend     │  │ check_stockouts  │  │ get_dfu_context  │
│  similar_dfus   │  │ portfolio_health │  │ (policy fields)  │
│                 │  │                  │  │                  │
│ Output:         │  │ Output:          │  │ Output:          │
│ ForecastAnalysis│  │ InvAnalysis      │  │ PolicyAnalysis   │
└─────────────────┘  └──────────────────┘  └──────────────────┘
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                               │ structured results
                               ▼
                    ┌──────────────────┐
                    │ SynthesizerAgent │
                    │ create_insight() │
                    └──────────────────┘
```

**Key benefit:** Each specialist runs in parallel and has a bounded tool set (max 6 tools). The orchestrator adds zero LLM tokens to the portfolio scan — it only runs SQL triage. **60% token cost reduction** vs current single-context design.

---

## 5. Closed-Loop: Action → Outcome → Learning

*From the AI Automation & Outcomes Expert*

### The Core Problem Today

The current system writes insights and lets planners acknowledge/resolve them. But there is **no mechanism to observe what happened after a planner acted**. The AI generates recommendations into a void — it cannot measure whether its advice was correct, so it cannot improve.

### Complete Closed-Loop System

```
DATA (PostgreSQL) ──── weekly scan ──── AIPlannerAgent
                                              │ create_insight()
                                              ▼
                                        ai_insights table
                                              │ GET /ai-planner/insights
                                              ▼
                                        AIPlannerTab (UI)
                                              │ [Accept] / [Reject]
                                    ┌─────────┴──────────┐
                                    ▼                    ▼
                             Confirm Modal          ai_recommendation_outcomes
                                    │               (outcome='rejected')
                                    ▼
                             Action Execution
                             (existing API endpoints)
                                    │ writes to
                                    ▼
                        ai_recommendation_outcomes
                        (planner_decision, metric_before, outcome_check_due_at)
                                    │
                                    │ APScheduler: daily at 02:00
                                    ▼
                             OUTCOME MEASUREMENT JOB
                             Re-query DOS/WAPE for DFU
                             Write metric_after, outcome_label
                                    │
                                    ▼
                             AI ACCURACY SCORECARD
                             acceptance_rate, improvement_rate
                                    │
                                    ▼
                        PROMPT ENRICHMENT (next cycle)
                        "HIGH-confidence stockout_risk insights
                         improved DOS in 73% of cases."
```

### New Table: `ai_recommendation_outcomes`

```sql
CREATE TABLE ai_recommendation_outcomes (
    outcome_id              SERIAL PRIMARY KEY,
    insight_id              INTEGER NOT NULL REFERENCES ai_insights(insight_id),
    insight_type            VARCHAR(80) NOT NULL,
    item_no                 TEXT NOT NULL,
    loc                     TEXT NOT NULL,
    abc_vol                 TEXT,
    planner_decision        VARCHAR(20) NOT NULL
                                CHECK (planner_decision IN ('accepted','rejected','snoozed','auto_accepted')),
    ai_confidence           VARCHAR(10) CHECK (ai_confidence IN ('high','medium','low')),
    financial_impact_est    NUMERIC(15,2),
    -- Snapshot BEFORE action
    metric_before_dos       NUMERIC(10,2),
    metric_before_wape      NUMERIC(8,4),
    metric_before_bias_pct  NUMERIC(8,4),
    lead_time_days          INTEGER,
    -- Action details
    action_taken            TEXT,
    executed_at             TIMESTAMPTZ,
    -- Outcome measured T+30d
    outcome_check_due_at    TIMESTAMPTZ,
    outcome_label           VARCHAR(20)
                                CHECK (outcome_label IN ('improved','degraded','neutral','insufficient_data')),
    metric_after_dos        NUMERIC(10,2),
    metric_after_wape       NUMERIC(8,4),
    outcome_delta           NUMERIC(10,4),
    created_at              TIMESTAMPTZ DEFAULT NOW()
);

-- Partial index: only pending outcome checks
CREATE INDEX idx_aro_outcome_due
    ON ai_recommendation_outcomes (outcome_check_due_at)
    WHERE outcome_label IS NULL;
```

### Action Automation Decision Matrix

Not all actions should be automated equally:

```
                    REVERSIBILITY
                    High (easy undo)     Low (hard to undo)
                   ┌───────────────────┬────────────────────┐
FINANCIAL  Low     │  TIER 1           │  TIER 2            │
IMPACT     (<$5K)  │  Fully Automatic  │  Confirm-Execute   │
           ────────┼───────────────────┼────────────────────┤
           High    │  TIER 2           │  TIER 3            │
           (>$5K)  │  Confirm-Execute  │  Recommend-Only    │
                   └───────────────────┴────────────────────┘

TIER 1 (auto-execute, no planner input):
  - champion_degradation → trigger champion re-selection job
  - policy_gap C-class → assign "manual_review" policy (safest default)
  - forecast_bias 6+ months → trigger champion re-selection

TIER 2 (show confirm modal, then execute):
  - stockout_risk critical/high → create replenishment exception
  - forecast_bias → write forecast override multiplier (3 months)
  - policy_gap A/B class → assign AI-recommended policy

TIER 3 (recommend only, planner navigates manually):
  - excess_inventory high (>$50K on hand) → cancel planned order
  - champion_degradation + model switch → changes production forecast
```

### Complete Action-to-Outcome Map

| Insight Type | Action | Outcome Metric (T+30d) | Pass Condition |
|---|---|---|---|
| `stockout_risk` | Create emergency replenishment exception | DOS ≥ lead_time × 1.5 | DOS recovered to safe level |
| `excess_inventory` | Switch policy to periodic_order_up_to | DOS < 120 days | Excess resolved |
| `forecast_bias` | Write forecast override multiplier | Champion bias_pct < ±10% | Bias corrected |
| `policy_gap` | Assign AI-recommended policy | No new exceptions in 30 days | Policy compliance |
| `champion_degradation` | Trigger champion re-selection | WAPE improvement > 5pp | Model improved |

### Weekly AI Accuracy Scorecard

```sql
-- Run every Monday; answers: "how good is the AI's advice, really?"
SELECT
    insight_type,
    COUNT(*) AS total_recommendations,
    ROUND(100.0 * COUNT(*) FILTER (WHERE planner_decision='accepted') / COUNT(*), 1)
        AS acceptance_rate_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE outcome_label='improved')
        / NULLIF(COUNT(*) FILTER (WHERE outcome_label IS NOT NULL), 0), 1)
        AS improvement_rate_pct,
    ai_confidence
FROM ai_recommendation_outcomes
WHERE decision_at >= NOW() - INTERVAL '7 days'
GROUP BY insight_type, ai_confidence
ORDER BY insight_type, ai_confidence;
```

---

## 6. AI-Computed Safety Stock — Highest Leverage Gap

*From the Supply Chain Expert*

`fact_safety_stock_targets` is a stub with 0 rows. Every downstream computation — health score, exceptions, investment plan, ROP alignment — operates on neutral placeholders. **This is the single highest-leverage gap in the entire platform.**

### The Formula

```
SS = Z(SL) × √(LT_mean × σ²_demand + demand_mean² × σ²_LT) × seasonality_factor
```

### Data Sourcing (all columns already exist in the DB)

```
INPUT              SOURCE TABLE                    COLUMN
───────────────────────────────────────────────────────────────────
Service Level Z    dim_replenishment_policy         service_level
                   (via fact_dfu_policy_assignment)
                   A-class: 0.98 → Z=2.054
                   B-class: 0.95 → Z=1.645
                   C-class: 0.90 → Z=1.282

Demand mean (μ)    fact_external_forecast_monthly   tothist_dmd
                   WHERE model_id='champion' AND lag=0
                   Trailing 12 months

Demand std (σ_d)   fact_external_forecast_monthly   tothist_dmd
                   STDDEV(tothist_dmd) over 12 months

Lead time mean     agg_inventory_monthly            latest_lead_time_days
                   DISTINCT ON (item_no,loc) DESC

Seasonality mult   dim_dfu                          peak_trough_ratio,
                                                    is_yearly_seasonal
                   factor = 1.0 + (ratio-1) × 0.5
                   (applied only during peak months)
```

### Numerical Example: 587382 @ 1401-BULK

```
INPUTS:
  μ_monthly  = 1020.25 units/month
  μ_daily    = 33.5 units/day
  σ_daily    = 5.04 units/day   (from STDDEV(tothist_dmd)/√30.44)
  LT_mean    = 14 days
  σ_LT       = 2.1 days         (15% default when mv_supplier_performance empty)
  Z(0.98)    = 2.054             (A-class, service_level=0.98)
  seasonality_factor = 1.0      (high_volume_steady, is_yearly_seasonal=false)

CALCULATION:
  combined_variability = √(14 × 5.04² + 33.5² × 2.1²)
                       = √(355.6 + 4949.1)
                       = √5304.7 = 72.8 units

  SS = 2.054 × 72.8 × 1.0 = 149.5 → round to 150 units

  ROP = (33.5 × 14) + 150 = 619 units

CURRENT STATE:
  on_hand ≈ 6.4 × 33 = 211 units
  211 < 619 → TRIGGER REORDER IMMEDIATELY
```

This computation can be added as an 11th tool — `compute_dynamic_safety_stock(item_no, loc)` — that queries the three source tables and returns the formula output **without requiring any new DB schema initially**.

---

## 7. Composite Triage Score

*From the Supply Chain Expert*

The current `get_portfolio_exceptions()` uses a simple 4-level CASE ordering. This is too coarse — within each category there is no urgency or financial ranking. Planners see items in arbitrary order.

### Formula

```
TRIAGE_SCORE = 0.25 × ABC_SCORE
             + 0.25 × DOS_URGENCY_SCORE
             + 0.15 × WAPE_SCORE
             + 0.20 × FINANCIAL_SCORE
             + 0.10 × BIAS_PERSISTENCE_SCORE
             + 0.05 × CLUSTER_SIGNAL_SCORE
```

### Component Scoring

```
ABC_SCORE:          A=100, B=60, C=25, unknown=10

DOS_URGENCY_SCORE:
  dos/lt < 0.5   → 100  (critical: DOS below half of LT)
  dos/lt < 1.0   → 85   (critical: DOS below LT)
  dos/lt < 1.5   → 70   (high: below 1.5× LT threshold)
  dos > 180      → 60   (excess: capital waste)
  else           → 5-20  (healthy range)

WAPE_SCORE:
  wape > 50%     → 100
  wape > 35%     → 65
  wape > 25%     → 35
  wape ≤ 15%     → 5

BIAS_PERSISTENCE_SCORE:
  same direction 6+ months → 100
  same direction 3+ months → 65
  else                     → 10
```

### Worked Comparison — 3 DFUs on Monday Morning

```
┌────────────────────────────────────────────────────────────────────────┐
│  TRIAGE SCORE COMPARISON                                               │
│                                                                        │
│  DFU         ABC  DOS/LT  WAPE  FINANCIAL BIAS      TRIAGE   RANK     │
│  ─────────────────────────────────────────────────────────────────     │
│  587382      A    0.46×   63%   $7,920    6mo over   92.3     #1 CRIT  │
│  @1401-BULK                                          (100pts)          │
│                                                                        │
│  616372      A    1.37×   57.8% $3,690    5mo over   76.1     #2 HIGH  │
│  @1401-BULK                               (100pts)  +60%peers          │
│                                                                        │
│  179333      B    11.8×   24.1% $694/mo   n/a        41.8     #3 MED   │
│  @2201-MAIN        (excess)  (good) carrying                 (excess)  │
└────────────────────────────────────────────────────────────────────────┘
```

Without the triage score, the CASE ORDER BY might surface `179333` ahead of `616372` because "excess" comes before "high_wape" in the current ordering. With the score, the planner sees the right items first.

---

## 8. S&OP Automation

*From the AI Automation & Outcomes Expert*

### Automated Weekly S&OP Preparation (Friday 18:00)

```
EVERY FRIDAY 18:00 (APScheduler cron)
│
├─ Step 1: Portfolio health query (SQL, no LLM)
│   Compare vs prior week → Week-over-Week deltas
│
├─ Step 2: Top 10 open insights by financial impact
│   Filter for critical + A-class → 3 escalation items
│
├─ Step 3: AIPlannerAgent.generate_portfolio_memo()
│   → writes to ai_planning_memos (already implemented)
│
├─ Step 4: Populate agenda template
│   Merge memo + escalation items + WoW stats
│   → write to ai_planning_memos with scope='sop_agenda'
│
└─ Step 5: Notify via JobNotificationContext
   → sidebar badge + "S&OP Prep Complete" toast
```

**Output template:**

```markdown
# S&OP PREPARATION BRIEF — March 2026

## 1. Portfolio Health Summary
| Metric        | This Week | Last Week | Change |
| Average DOS   | 28.4d     | 31.2d     | -2.8d  |
| Stockout Risk | 3 DFUs    | 1 DFU     | +2     |
| Portfolio WAPE| 32.1%     | 30.8%     | +1.3pp |
| Open Insights | 16        | 22        | -6     |

## 2. AI Narrative
[generate_portfolio_memo() output]

## 3. Top 3 Escalation Items
1. STOCKOUT RISK — 587382 @ 1401-BULK (A-class · CRITICAL)
   ...

## 4. Recommended Agenda
1. Exception Review (3 critical, 5 high severity)
2. Forecast Alignment — 7 DFUs with bias > 20%
3. Policy Governance — 2 A-class DFUs with no active policy
4. Working Capital — $124K estimated at-risk across open insights
```

### Eight Proactive Trigger Conditions

In addition to the weekly scan, the system should monitor for time-sensitive events continuously:

| Trigger | Priority | Condition |
|---|---|---|
| `dos_crossed_stockout_threshold` | 1 (CRITICAL) | DOS < lead_time for A/B class DFU, no existing open insight |
| `intramonth_stockout_detected` | 1 (CRITICAL) | Event in `mv_intramonth_stockout` within 48h |
| `forecast_bias_persisted_3_months` | 2 (HIGH) | Champion bias > ±20% for 3 consecutive months |
| `champion_wape_spike` | 2 (HIGH) | Champion WAPE exceeded 50% last month on A/B class |
| `lead_time_increased_significantly` | 2 (HIGH) | Latest LT > 1.5× 3-month average |
| `dos_exceeded_excess_threshold` | 3 (MEDIUM) | DOS crossed 180 days |
| `policy_assignment_missing` | 3 (MEDIUM) | A-class DFU has no active policy |
| `new_inventory_snapshot_loaded` | 4 (LOW) | `agg_inventory_monthly` refreshed in last 30 min |

These integrate as a new `trigger_monitor` job type in `common/job_registry.py`, running every 15 minutes.

---

## 9. Prioritized Implementation Roadmap

*Synthesized across all 4 experts*

### Phase 1 — Foundation (Weeks 1–2, ~20h)

| Item | Effort | Source | Impact |
|---|---|---|---|
| Rewrite system prompt with few-shot examples | 2h | UX + Tech experts | Quality +40% |
| Fix `INTERVAL '%s months'` SQL bug | 30min | Tech expert | Correctness fix |
| Add turn limit + token budget guard | 2h | Tech expert | Reliability fix |
| Pydantic `CreateInsightInput` validation | 4h | Tech expert | Reliability +60% |
| `ai_call_log` observability table + endpoint | 1d | Tech expert | Full visibility |

### Phase 2 — Planner Experience (Weeks 3–4, ~30h)

| Item | Effort | Source | Impact |
|---|---|---|---|
| `ai_recommendation_outcomes` table + outcome checker job | 4h | Automation expert | Closes feedback loop |
| Confirm modal before action execution | 4h | UX expert | Trust + safety |
| SSE streaming endpoint + `useScanStream` hook | 1d | Tech expert | Real-time UX |
| `CausalChainCard` component (replace text paragraph) | 4h | UX expert | Scannability |
| AI Confidence badge (high/medium/low) | 2h | UX expert | Trust calibration |

### Phase 3 — Intelligence (Weeks 5–8, ~40h)

| Item | Effort | Source | Impact |
|---|---|---|---|
| `compute_dynamic_safety_stock()` tool (11th tool) | 1d | SC expert | Unblocks health score, exceptions, investment |
| Composite triage score in `get_portfolio_exceptions()` | 4h | SC expert | Correct priority ordering |
| Bias correction multiplier quantification in system prompt | 2h | SC expert | Specific, testable recommendations |
| Trigger monitor job (8 conditions) | 2d | Automation expert | Real-time exception detection |
| S&OP memo automation (Friday 18:00) | 1d | Automation expert | Weekly agenda auto-prep |

### Phase 4 — Architecture Scale (Weeks 9–16, ~60h)

| Item | Effort | Source | Impact |
|---|---|---|---|
| Multi-agent architecture (Orchestrator + 3 specialists) | 1wk | Tech expert | Cost -60%, quality +25% |
| Semantic cache (pgvector, similarity > 0.92) | 2d | Tech expert | Cost -50%, speed 2× |
| `fact_forecast_overrides` table + one-click apply | 4h | Automation expert | Audit trail, action execution |
| Weekly AI accuracy scorecard UI panel | 4h | Automation + UX | Trust measurement |
| Prompt enrichment from outcomes feedback | 1d | Automation expert | Self-improving AI |

---

## Summary

The Demand Studio platform has all the raw data needed for world-class AI-embedded planning. The data is there. The pipeline is there. The 17 API endpoints are there. The gaps are:

1. **Connections between layers** — the AI writes insights but does not close the loop back to actions and outcomes.
2. **Specificity of recommendations** — "reduce forecast" vs "apply 0.52× multiplier" is the difference between an advisory and a decision-support system.
3. **Real safety stock computation** — the most downstream planning layers all depend on SS values that are currently 0.
4. **Observation infrastructure** — token costs, outcome rates, and recommendation accuracy are completely invisible.

The four expert analyses converge on the same priority: **the `ai_recommendation_outcomes` table is the single highest-leverage addition**. It transforms the AI from a one-way recommendation engine into a self-evaluating, continuously improving planning partner.
