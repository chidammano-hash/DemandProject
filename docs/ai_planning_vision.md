# AI-Embedded Demand & Inventory Planning — Vision & Recommendations

**Perspective:** Supply chain domain expert · UI/UX architect · Storyboard designer · Minimalist

---

## 1. The Central Idea

> **A planner should arrive Monday morning, open one screen, and know exactly what to do — because the AI already did the thinking overnight.**

The current codebase has all the raw materials: multi-model forecasting, EOQ, policy management, exception queues, AI insights, safety stock simulation. But it is assembled as a *data platform*, not a *planning workflow*. The transformation is from "here is your data" to "here is your decision, here is why, here is the one-click action."

Three principles drive everything below:

1. **AI is the first reader, not the last** — The AI scans, ranks, traces causality, and pre-writes the recommendation. The planner approves or overrides — they do not discover.
2. **One screen, one job** — Each view has a single primary action. No tab has two competing focal points.
3. **Show less, mean more** — Every number on screen earns its place. If it does not drive a decision this week, it is behind a disclosure.

---

## 2. Current State Assessment

| Dimension | Current Grade | Notes |
|---|---|---|
| Data completeness | A | All 8 domains, 17 API routers, full forecast/inventory/EOQ/policy pipeline |
| AI integration | C+ | AI Planner is bolt-on tab, not woven into other views |
| Planner workflow | D | No Monday morning work queue, no action tracking, no outcome loop |
| UX minimalism | C | Dense tabs, multiple competing KPIs per view, inconsistent empty states |
| Supply chain depth | B | EOQ, policies, SS simulation exist but safety stock engine is a stub |
| Storyboard / narrative | F | No planner-facing exception workflow end-to-end |
| Explainability | C | AI reasoning exists but buried in collapsible; causality chain not visual |
| Mobile / responsive | D | Desktop-only at practice; sidebar collapses but tables don't reflow |

---

## 3. The Planner's Week — Reference Workflow

Before redesigning anything, understand the job to be done:

```
MONDAY
  07:00  AI overnight scan complete — portfolio exception digest ready
  08:00  Planner opens app → sees ranked work queue (3–7 priority items)
  08:30  Clicks top item → AI has pre-diagnosed: "Stockout risk in 8 days,
          caused by 40% over-forecast last 3 months pulling down safety stock"
  08:45  One-click: "Accept recommendation" → policy change queued
  09:00  Reviews remaining items, snoozes 2, escalates 1 to S&OP

WEDNESDAY
  Forecast review cycle: AI flags 12 DFUs with persistent bias > 20%
  Planner adjusts override multipliers for 4, marks 8 as "monitor"

FRIDAY
  S&OP prep: AI-generated portfolio narrative ready
  Planner edits narrative, exports to PowerPoint
```

Every screen recommendation below serves one or more moments in this workflow.

---

## 4. UX Architecture Redesign

### 4.1 Navigation: From Tabs to Workflow Modes

**Current:** 14 sidebar items across 5 arbitrary sections — planner must know which tab holds what.

**Recommended:** Three primary modes + one system mode:

```
┌─────────────────────────────────────────────────┐
│  ⚡ COMMAND CENTER    [default, Monday morning]  │
│  ├─ Exception Work Queue (AI-ranked)             │
│  ├─ Portfolio Health Pulse (4 KPIs max)          │
│  └─ AI Narrative Digest (this week's memo)       │
│                                                  │
│  📊 ANALYZE           [deep dives]               │
│  ├─ Forecast Accuracy                            │
│  ├─ DFU Deep Dive                                │
│  ├─ Inventory Health                             │
│  └─ Storyboard / S&OP Deck                      │
│                                                  │
│  ⚙️  CONFIGURE        [periodic, not daily]       │
│  ├─ Replenishment Policies                       │
│  ├─ Safety Stock Targets                         │
│  ├─ EOQ Parameters                               │
│  └─ Cluster Management                          │
│                                                  │
│  🔧 SYSTEM            [admin only]               │
│  ├─ Jobs & Automation                            │
│  ├─ Data Explorer                                │
│  └─ Market Intelligence                         │
└─────────────────────────────────────────────────┘
```

**Rationale:** Planners use Command Center daily, Analyze weekly, Configure monthly, System rarely. Hiding Configure and System reduces cognitive load by ~60% for the average session.

### 4.2 The Command Center — Primary Screen

Replace the current Dashboard tab with a true command center. Wireframe:

```
┌──────────────────────────────────────────────────────────────────┐
│  Good morning, Manohar.  Monday, March 5 · AI scan: 2 min ago    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐    │
│  │ 7 Open   │ │ 2 Crit.  │ │ DOS 42d  │ │ $128K at risk    │    │
│  │ Insights │ │ 🔴       │ │ ▼ -3d    │ │ this week        │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘    │
│                                                                  │
│  ── PRIORITY WORK QUEUE ─────────────────────── Sort: AI rank ▾  │
│                                                                  │
│  ① CRITICAL  587382 @ 1401-BULK · Stockout in ~8 days           │
│     Cause: 87% over-forecast → SS exhausted · Impact: $42K       │
│     [Review & Accept] [Override] [Snooze 3d]                     │
│                                                                  │
│  ② HIGH      616372 @ 2201-MAIN · Persistent bias +49%           │
│     Cause: Model degradation since Jan · WAPE 63%                │
│     [Review & Accept] [Override] [Snooze 3d]                     │
│                                                                  │
│  ③ HIGH      179333 @ 1401-BULK · DOS 6.4d vs LT 14d             │
│     Cause: Lead time increase, SS not recalculated               │
│     [Review & Accept] [Override] [Snooze 3d]                     │
│                                                                  │
│  + 4 more · View all                                             │
│                                                                  │
│  ── AI PLANNING DIGEST ────────────────── gpt-4o-mini · Mar 2026 │
│  "Portfolio is under pressure this week. Three A-class DFUs..."  │
│  Read full memo →                                                │
└──────────────────────────────────────────────────────────────────┘
```

**Key UX decisions:**
- Greeting with last-scan timestamp removes anxiety about data freshness
- Maximum 3 items visible without scroll; "view all" for rest
- Each work item has exactly 3 actions: Accept, Override, Snooze — no other options at this level
- Financial impact is dollar-first, not percentage-first (planners care about business impact)
- AI digest is collapsible — not forced reading

### 4.3 Exception Cards — Causal Chain Visualization

**Current:** Text-heavy collapsible reasoning block.

**Recommended:** Visual causal chain — a horizontal pipeline showing root cause → intermediate → consequence → financial impact:

```
┌─────────────────────────────────────────────────────────────────┐
│  🔴 CRITICAL  587382 — Widget A  ·  1401-BULK Distribution      │
│  Stockout Risk                                                  │
│                                                                 │
│  [Forecast]──→[Inventory]──→[Policy]──→[Consequence]            │
│   +87% bias    DOS 6.4d      ROP not     Stockout               │
│   3mo persist  vs LT 14d     triggered   in ~8 days             │
│                                                                 │
│  RECOMMENDATION                                                 │
│  Trigger emergency ROP: reorder 250 units (1× EOQ) today.      │
│  Adjust forecast multiplier: −0.4× for next 3 months.          │
│  Review SS formula: current 7d SS insufficient for LT 14d.     │
│                                                                 │
│  Financial: Stockout cost est. $42K · Reorder cost $1,200      │
│                                                                 │
│  [✓ Accept all recommendations]  [✎ Customize]  [— Snooze 3d]  │
└─────────────────────────────────────────────────────────────────┘
```

**Why this works:**
- Planner sees causality in 5 seconds without reading a paragraph
- The causal chain format mirrors how supply chain analysts actually think
- Three metrics per layer (no more) — progressive depth on click
- Single primary CTA ("Accept all") with secondary options

### 4.4 DFU Deep Dive — Single-Item Command Center

When a planner clicks any item number anywhere in the app, a **right-side drawer** slides in (not a new tab):

```
┌─────────────────────────────────┐
│  587382 — Widget A              │
│  1401-BULK · A-class · Cluster: │
│  high_volume_steady             │
│                                 │
│  [Forecast]  [Inventory]        │
│  [EOQ]       [Policy]           │
│                                 │
│  CURRENT STATUS ────────────── │
│  DOS:   6.4d  ⚠️ (LT: 14d)      │
│  WAPE:  63%   🔴                │
│  Bias:  +87%  3-month streak    │
│                                 │
│  AI SAYS ───────────────────── │
│  "Persistent over-forecast is   │
│  drawing down safety stock.     │
│  EOQ is correctly sized (250u). │
│  Issue: policy, not demand."    │
│                                 │
│  ACTIONS ─────────────────────  │
│  [ ] Trigger reorder now        │
│  [ ] Adjust forecast −40%       │
│  [ ] Change policy → ss_buffer  │
│  [ ] Flag for S&OP review       │
│                                 │
│  [Apply selected]               │
└─────────────────────────────────┘
```

**Principles:**
- Drawer pattern keeps context (user stays on their list, not navigated away)
- AI verdict in 2–3 sentences max — no essay
- Actionable checkboxes: planner selects what to apply, clicks once
- Tabs within drawer for deeper data (forecast chart, inventory trend, etc.)

---

## 5. AI Integration Depth Recommendations

### 5.1 AI Must Be Embedded, Not Siloed

**Current state:** AI Planner is tab #11 of 14. Planners must navigate there specifically.

**Target state:** AI presence on every screen where a decision is made.

| Location | Current | Recommended AI Enhancement |
|---|---|---|
| Dashboard | KPI numbers | Color-coded AI risk signal next to each KPI; hover for 1-sentence diagnosis |
| DFU Analysis | Charts | "AI says" annotation on forecast chart at divergence points |
| Accuracy tab | WAPE table | Inline AI flag: "Persistent bias — recommend model switch" per row |
| Inventory tab | DOS table | Red/amber/green AI risk tier per row; one-click to drill cause |
| Exception Queue | Manual thresholds | Replace with AI-ranked exceptions with stated reason per item |
| Policy Management | Static forms | AI recommends policy type based on DFU cluster + variability class |
| EOQ panel | Computed numbers | AI commentary: "EOQ is optimal but current order qty deviates 40%" |
| S&OP / Storyboard | Manual slide creation | AI drafts narrative; planner edits |

### 5.2 Proactive vs. Reactive Intelligence

Current AI is reactive: planner clicks "Generate Now" → AI scans → results appear.

**Recommend three intelligence modes:**

1. **Overnight Batch (already exists)** — Portfolio scan writes insights to DB. UI shows count badge on sidebar. Runs automatically.

2. **Contextual Inline (new)** — When planner opens a DFU, silently call AI to generate a 2-sentence assessment for *that specific DFU* using cached tool results. 500ms target. Cache 1 hour.

3. **S&OP Preparation Mode (new)** — Weekly deep scan: AI generates full narrative memo with top 10 exceptions, trend analysis, and recommended agenda items. Output formatted for export.

### 5.3 AI Action Execution (High Impact)

Today AI *recommends*, planners *manually execute*. Close the loop:

| Recommendation Type | Automated Action |
|---|---|
| "Change policy to ss_buffer" | `PUT /inv-planning/policy-assignments/{id}` with one-click confirm |
| "Trigger emergency reorder" | Create `fact_replenishment_exceptions` record with `recommended_order_qty` |
| "Adjust forecast −40%" | Write forecast override multiplier to new `fact_forecast_overrides` table |
| "Recalculate safety stock" | Trigger `run_ss_simulation` job for this DFU |
| "Escalate to S&OP" | Tag insight with `escalated_to_sop` flag; show in S&OP panel |

Each action needs: confirmation modal with preview → execution → outcome tracking (did the action help?).

### 5.4 Outcome Learning (Closing the Feedback Loop)

Current gap: AI recommends → planner acts → nothing measures whether it worked.

**Add outcome tracking:**
- When an insight is marked "resolved", record the resolution action taken
- 30/60/90 days later: auto-check whether the metric improved
- Feed outcome labels back to prompt engineering: "Previous recommendations that were accepted and improved DOS by >20% used this reasoning pattern..."
- Surface a simple accuracy score: "AI recommendations accepted: 73% · Outcomes improved: 61%"

This is the flywheel that makes the AI better over time without model retraining.

### 5.5 Multi-Model AI Strategy

**Current:** Single provider at a time (OpenAI or Anthropic), configured in YAML.

**Recommend tiered model strategy:**

| Task | Model | Rationale |
|---|---|---|
| Portfolio scan (100 DFUs) | `gpt-4o-mini` or `claude-haiku-4-5` | Low cost, high volume, structured output |
| Single DFU deep analysis | `gpt-4o` or `claude-sonnet-4-6` | Better reasoning for complex causality |
| S&OP narrative generation | `gpt-4o` or `claude-opus-4-6` | Best prose quality for executive consumption |
| Inline contextual hints | Cached previous results or `gpt-4o-mini` | Sub-500ms response required |

This can be expressed as a `task_model_overrides` section in `ai_planner_config.yaml`.

---

## 6. Supply Chain Domain Recommendations

### 6.1 Safety Stock Engine (Critical Gap — IPfeature3)

The current `fact_safety_stock_targets` is a stub. This is the foundation everything else depends on:

- **Formula:** `SS = Z × σ_demand × √(LT)` where Z is service-level Z-score, σ is demand standard deviation over lead time
- **Service level by segment:** A-class → 98%, B-class → 95%, C-class → 90% (configurable per policy)
- **Dynamic recalculation:** Triggered when forecast bias changes significantly, lead time changes, or policy is updated
- **Integration:** Safety stock feeds into: DOS alert thresholds, EOQ effective computation, health score, exception generation, investment planning

Without this, health scores use neutral placeholders and investment plans have no cost basis.

**Recommended implementation priority:** IPfeature3 before any other new feature.

### 6.2 Demand Sensing Enhancement (IPfeature9)

Current demand signals compute point-in-time velocity. Recommend adding:

- **Day-of-week seasonality correction** — retail DFUs spike Friday/Saturday; industrial DFUs flat
- **Trend-adjusted velocity** — 30-day velocity trending up/down vs 90-day baseline
- **Anomaly flag** — velocity > 2σ from recent baseline = demand signal alert
- **Short-horizon forecast override** — if 7-day velocity suggests demand significantly different from monthly forecast, auto-generate an override candidate

### 6.3 Forecast Override Management

**Currently missing entirely.** Demand planners override statistical forecasts constantly. Without capturing overrides:
- No audit trail
- No measurement of override accuracy vs statistical model
- AI cannot learn which human adjustments were correct

**Recommended `fact_forecast_overrides` table:**
```sql
forecast_override_id SERIAL PRIMARY KEY,
item_no TEXT, loc TEXT,
override_period DATE,
statistical_forecast NUMERIC,
override_value NUMERIC,
override_reason TEXT,
override_by TEXT,
accepted_by_ai BOOLEAN,
final_accuracy_wape NUMERIC,
created_at TIMESTAMPTZ
```

**UX:** Inline editable forecast cells in DFU Analysis tab. AI validates the override and warns if it contradicts its own analysis.

### 6.4 Supplier Lead Time Variability

Current `dim_replenishment_policy` uses fixed `review_cycle_days`. Reality: lead times vary. Recommend:

- Track lead time as a distribution (mean + std dev), not a point estimate
- Alert when realized lead time exceeds 1.5× the assumed value
- Feed lead time variability into safety stock formula
- Supplier scorecard: on-time delivery %, average lead time deviation — already partially built in IPfeature12 but not connected to planning parameters

### 6.5 S&OP Integration Layer

Supply chain planning is ultimately about the S&OP cycle. The platform needs S&OP-aware views:

- **Consensus forecast:** Statistical forecast + planner override + AI validation → final consensus
- **Volume scenario modeling:** What if demand is +10%, −15%? Show impact on inventory, service level, working capital
- **S&OP deck generation:** AI drafts the monthly business review narrative from current data
- **Action register:** Items escalated from exception queue → tracked to resolution across the S&OP cycle

### 6.6 Working Capital Optimization

Currently the platform tracks inventory value but doesn't connect it to working capital targets:

- **Cash-to-cash cycle time:** Days Inventory + Days Receivable − Days Payable
- **Inventory investment efficiency:** Service level achieved per dollar of inventory held (Efficient Frontier — IPfeature13, partially implemented)
- **Budget constraint optimization:** Given $X working capital budget, maximize service level across portfolio — connect efficient frontier to actionable allocation

---

## 7. UI/UX Component Recommendations

### 7.1 Replace Dense Tables with Ranked Lists

**Current pattern:** Every panel has a paginated table with 10–15 columns.

**Recommended pattern:** Two-tier display:
1. **Ranked list** (primary) — 5 columns max: item, location, key signal, AI tier (🔴🟠🟡🟢), action button
2. **Detail drawer** (on click) — all the data, organized by planning layer

This reduces initial cognitive load by ~70% while preserving full data access.

### 7.2 Semantic Color System

**Current:** Inconsistent color usage. Red sometimes means "error", sometimes "critical", sometimes just "low value."

**Recommended strict semantic palette:**

| Color | Meaning | Use case |
|---|---|---|
| Red `#ef4444` | Action required NOW | Stockout imminent, critical insight, overdue |
| Amber `#f59e0b` | Watch / at risk | High severity insight, declining trend |
| Yellow `#eab308` | Monitor | Medium risk, worth attention |
| Teal `#14b8a6` | AI-generated | Any AI annotation, insight, recommendation |
| Blue `#3b82f6` | Navigation / interactive | Buttons, links, selected state |
| Green `#22c55e` | Healthy / positive | DOS comfortable, accuracy good, resolved |
| Gray | Neutral / inactive | Historical, resolved, low priority |

AI-generated content should always use teal — a distinct "AI brand color" — so planners immediately recognize AI vs human data.

### 7.3 Progressive Disclosure Pattern

Apply consistently across all panels:

```
Level 0 (default): 1 KPI number + AI risk indicator
Level 1 (hover/expand): 3-5 supporting metrics + mini trend sparkline
Level 2 (click/drawer): Full detail, charts, history, AI reasoning
Level 3 (dedicated tab): Full analytical view (Accuracy, DFU Analysis, etc.)
```

Currently most panels jump straight to Level 2 or 3 by default.

### 7.4 Command Palette (⌘K)

Add a global command palette triggered by `Cmd+K` (macOS) / `Ctrl+K` (Windows):

```
⌘K
┌─────────────────────────────────────────┐
│  > _                                    │
│                                         │
│  RECENT                                 │
│  587382 @ 1401-BULK  (DFU analysis)     │
│  Portfolio scan  (2 min ago)            │
│                                         │
│  ACTIONS                                │
│  Run portfolio scan                     │
│  Generate S&OP memo                     │
│  Go to exception queue                  │
│                                         │
│  SEARCH                                 │
│  Type item number, location, or topic…  │
└─────────────────────────────────────────┘
```

Allows experienced planners to navigate entirely by keyboard. Especially useful for exception triage.

### 7.5 Bulk Actions on Exception Queue

**Current:** Each insight has individual Acknowledge/Resolve buttons.

**Recommended:** Checkbox multi-select + bulk action bar:
```
☑ 3 selected    [Acknowledge all]  [Snooze 3 days]  [Export]  [×]
```

Allows a planner to clear 10 low-priority insights in one action vs 10 individual clicks.

### 7.6 Trend Indicators on Every Metric

Numbers without direction are hard to interpret. Every KPI should show a trend arrow:

```
DOS  42.3 days  ▼ -3.2d  (7-day change)
WAPE 28.4%      ▲ +4.1%  (4-week change)
```

Color the arrow by direction × threshold (red if declining past threshold, green if improving).

### 7.7 AI Confidence Indicator

When the AI makes a recommendation, show a confidence tier:

```
🔵 HIGH CONFIDENCE  — based on 6+ months of consistent signal
🟡 MEDIUM            — 3-month pattern, could reverse
⚪ LOW               — single-month spike, monitor only
```

This teaches planners when to trust the AI and when to apply judgment. Builds trust faster than any amount of documentation.

### 7.8 Empty State Excellence

**Current empty states:** Generic "no data found" messages.

**Recommended:** Each empty state should be context-aware and action-driving:

| Location | Current | Recommended |
|---|---|---|
| AI Planner (no insights) | "No insights found. Click Generate Now." | "Portfolio looks healthy! Last scan: 2 min ago. Next scheduled: Monday 06:00." |
| Exception Queue (no exceptions) | Generic empty | "No exceptions match current thresholds. Adjust severity filters or run a manual scan." |
| Forecast accuracy (no model) | — | "No champion model selected for this DFU. Run champion selection to enable accuracy tracking." |
| Policy (unassigned DFU) | — | "No policy assigned. AI recommends: `continuous_rop` based on A-class + low variability profile." |

---

## 8. Storyboard: S&OP Exception Workflow

This is the planner's primary Monday morning journey, implemented as an integrated 4-screen flow.

### Screen 1: The Briefing
*Context: AI has finished overnight scan. Planner opens app.*

```
┌─────────────────────────────────────────────────────────────────┐
│  AI PLANNING BRIEF — Week of March 5, 2026                     │
│                                                                 │
│  I scanned 2,341 DFUs across 3 locations overnight.            │
│  Here's what needs your attention today:                        │
│                                                                 │
│  🔴  2 CRITICAL    Stockout within 7 days (A-class)             │
│  🟠  5 HIGH        Persistent forecast bias > 30%               │
│  🟡  8 MEDIUM      Excess inventory > 180 DOS                   │
│                                                                 │
│  Portfolio DOS: 42.3d  ▼  Service Risk: $172K   Working Cap:   │
│                            (est. lost sales)    $8.2M on hand  │
│                                                                 │
│  [Start Exception Review →]    [Skip to full list]             │
└─────────────────────────────────────────────────────────────────┘
```

### Screen 2: The Exception
*Context: Planner clicks through each AI-ranked exception one at a time.*

```
┌─────────────────────────────────────────────────────────────────┐
│  EXCEPTION 1 of 2 CRITICAL              ← Prev  Next →         │
│                                                                 │
│  WIDGET A · 587382  ·  1401-BULK DISTRIBUTION  ·  A-class      │
│                                                                 │
│  THE PROBLEM                                                    │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐   │
│  │Forecast │ →  │Inventory │ →  │ Policy  │ →  │  Risk    │   │
│  │+87% bias│    │DOS 6.4d  │    │ROP not  │    │Stockout  │   │
│  │3 months │    │LT: 14d   │    │triggered│    │in ~8 days│   │
│  └─────────┘    └──────────┘    └─────────┘    └──────────┘   │
│                                                                 │
│  WHY IT HAPPENED                                               │
│  Over-forecast pulled 250 units into safety stock. ROP         │
│  trigger (50 units) was not hit because on-hand appeared       │
│  healthy on paper. Real demand is 33 units/day vs forecasted   │
│  61 units/day.                                                  │
│                                                                 │
│  WHAT I RECOMMEND                                               │
│  ① Reorder 250 units NOW  (1× EOQ, $3,000 cost)                │
│  ② Adjust forecast: apply −0.4× multiplier for 3 months        │
│  ③ Change policy: ROP → Safety Stock Buffer (matches cluster)  │
│                                                                 │
│  [✓ Accept all 3]  [✎ Customize]  [— Snooze 48h]  [✗ Reject]  │
└─────────────────────────────────────────────────────────────────┘
```

### Screen 3: The Action Confirmation
*Context: Planner accepts. System shows exactly what will change.*

```
┌─────────────────────────────────────────────────────────────────┐
│  CONFIRM ACTIONS — 587382 @ 1401-BULK                          │
│                                                                 │
│  You're about to make the following changes:                    │
│                                                                 │
│  ✓  CREATE replenishment exception:                             │
│       Qty: 250 units  ·  Priority: Emergency  ·  Due: Today    │
│                                                                 │
│  ✓  ADD forecast override:                                      │
│       Multiplier: 0.6×  ·  Period: Mar–May 2026               │
│       Statistical: 61 units/day → Override: 37 units/day       │
│                                                                 │
│  ✓  CHANGE replenishment policy:                                │
│       From: continuous_rop  →  To: safety_stock_buffer         │
│       New SS target: 14 days (= 1× lead time)                  │
│                                                                 │
│  Expected outcome (AI projection):                              │
│  DOS after reorder: ~13.5 days  ·  Stockout risk: eliminated   │
│  Service level: 97% → 99% (vs SLA 98%)                         │
│                                                                 │
│  [Execute all changes]                    [Go back]             │
└─────────────────────────────────────────────────────────────────┘
```

### Screen 4: The Outcome (30 days later — auto-generated)
*Context: AI automatically reviews impact of accepted actions.*

```
┌─────────────────────────────────────────────────────────────────┐
│  OUTCOME REVIEW — 587382 @ 1401-BULK  ·  Actions taken Mar 5   │
│                                                                 │
│  30-day result:                                                 │
│                                                                 │
│  DOS:        6.4d  →  18.2d  ✅  (target: 14d)                  │
│  Forecast:   +87%  →  +12%   ✅  (multiplier working)          │
│  Policy:     ROP   →  SSB    ✅  (assigned Mar 5)               │
│  Stockout:   8 days out  →  0 events  ✅                        │
│                                                                 │
│  AI ASSESSMENT                                                  │
│  "Actions were effective. The emergency reorder prevented an    │
│   estimated $42K stockout. Consider making SS Buffer policy     │
│   permanent for this cluster segment."                          │
│                                                                 │
│  [Accept permanent policy change]  [Review in 30 days]         │
└─────────────────────────────────────────────────────────────────┘
```

This four-screen flow is **Feature 40 (Storyboard)** — it is currently not implemented and is the highest-impact UX gap.

---

## 9. Technical Architecture Recommendations

### 9.1 Event-Driven Insight Triggering

**Current:** Portfolio scan runs on schedule (cron Monday 06:00) or manually.

**Recommended:** Trigger-based scanning:

| Event | Trigger |
|---|---|
| Lead time data updated | Re-run safety stock calculation for affected DFUs |
| Champion model switches | Re-evaluate policy alignment for that DFU |
| Actual demand exceeds forecast by >50% | Immediate DFU-level AI analysis |
| DOS crosses stockout threshold | Critical insight generated in real time |
| New inventory snapshot loaded | Incremental exception scan for changed DFUs |

Implement as: `fact_change_events` table → APScheduler polls and dispatches micro-scans.

### 9.2 AI Insight Deduplication & Staleness

**Current:** Each scan run creates new insight rows. Risk of duplicate insights.

**Recommended:**
- Upsert on `(item_no, loc, insight_type, status='open')` — only one open insight per type per DFU
- Staleness flag: if insight is 7+ days old and underlying condition is unchanged, mark as `stale`
- Auto-close: if the triggering condition resolves (DOS recovers, bias drops), auto-resolve the insight and log `auto_resolved_at`
- Insight lifecycle: `open` → `acknowledged` → `in_progress` → `resolved` | `auto_resolved`

### 9.3 Forecast Override Table

Already discussed in §6.3. Schema:
```sql
CREATE TABLE fact_forecast_overrides (
    override_id      SERIAL PRIMARY KEY,
    item_no          TEXT NOT NULL,
    loc              TEXT NOT NULL,
    override_period  DATE NOT NULL,
    statistical_qty  NUMERIC(14,4),
    override_qty     NUMERIC(14,4) NOT NULL,
    override_reason  TEXT,
    override_source  TEXT DEFAULT 'manual',  -- 'manual' | 'ai_recommendation' | 'system'
    ai_insight_id    INTEGER REFERENCES ai_insights(insight_id),
    accepted_by      TEXT,
    outcome_wape     NUMERIC(8,4),           -- populated 30 days after period
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (item_no, loc, override_period)
);
```

### 9.4 Streaming AI Responses

**Current:** AI analysis blocks until complete (30–120 seconds for portfolio scan).

**Recommended:** Server-Sent Events (SSE) for real-time streaming:
- `GET /ai-planner/stream/{scan_run_id}` — streams insight cards as they are created
- Frontend renders each insight as the AI writes it
- Planner sees insights appear one-by-one instead of waiting for full batch
- Dramatically improves perceived performance

### 9.5 Semantic Caching

**Current:** Every AI call hits the LLM API. No caching.

**Recommended:**
- Cache tool results (DB query results) with 1-hour TTL — `get_dfu_full_context` for the same DFU is identical across multiple analyses
- Cache AI assessments for unchanged DFUs with 24-hour TTL
- Use pgvector (already installed for chatbot) to find semantically similar previous analyses — if a DFU looks like a previous analyzed DFU with the same outcome, use cached reasoning

### 9.6 WebSocket for Real-Time Collaboration

Multiple planners may work on the same exception queue simultaneously. Add:
- Presence indicators: "Maria is reviewing 587382"
- Lock-on-edit: when planner opens exception for action, others see it as "being reviewed"
- Conflict resolution: if two planners accept different recommendations, show merge dialog

### 9.7 Audit Trail

Every planning action should be immutable and auditable:
```sql
CREATE TABLE planning_audit_log (
    log_id          SERIAL PRIMARY KEY,
    entity_type     TEXT,  -- 'insight' | 'policy' | 'forecast_override' | 'exception'
    entity_id       INTEGER,
    action          TEXT,  -- 'created' | 'acknowledged' | 'resolved' | 'override_applied'
    performed_by    TEXT,
    performed_at    TIMESTAMPTZ DEFAULT NOW(),
    previous_value  JSONB,
    new_value       JSONB,
    ai_scan_run_id  TEXT
);
```

---

## 10. Supply Chain KPI Framework

### 10.1 The Five Metrics That Matter

Reduce the KPI surface to what planners actually use for decisions:

| KPI | Definition | Alert Threshold | Target |
|---|---|---|---|
| **DOS** | Days of supply (on-hand / avg daily demand) | < LT × 1.5 | Category-specific |
| **Service Level** | Fill rate (units shipped / units ordered) | < 95% for A-class | 98%+ A, 95% B, 90% C |
| **Forecast Bias** | Systematic over/under-forecast | > ±20% for 3 months | ±5% |
| **WAPE** | Champion model accuracy | > 35% high, > 50% critical | < 25% |
| **Inventory Turns** | Annual demand / avg inventory | < 4× for A-class | 8–12× |

Everything else (WOC, LT Coverage, cycle stock value, etc.) is Level 2 — accessible but not on the primary screen.

### 10.2 The Planning Scorecard

Add a weekly planner scorecard visible in the Command Center:

```
YOUR WEEK — March 1–7, 2026
────────────────────────────────────────────────────────
Exceptions reviewed:     12 / 15   (80%)
Recommendations accepted: 9 / 12   (75%)
Insights resolved:        7
Outstanding critical:     2  ⚠️
AI accuracy last month:   61% of accepted recs improved metrics
────────────────────────────────────────────────────────
```

This drives planner engagement and measures AI effectiveness simultaneously.

---

## 11. Minimalism Audit — What to Remove or Hide

The following currently occupy primary screen real estate but should be demoted:

| Item | Current | Recommended |
|---|---|---|
| Data Explorer tab | Primary nav item | Move to System section; planners don't use raw explorer daily |
| Market Intelligence tab | Primary nav item | Move to System section; a supporting tool, not primary |
| Chat (NL→SQL) | Primary nav item | Hide behind `⌘K` command palette; replace with context-aware AI hints |
| Cluster management | Primary nav item | Move to Configure section |
| Jobs tab | Primary nav item | Collapse to notification badge + modal (most planners never need full scheduler view) |
| SHAP feature importance | Prominent panel in Accuracy tab | Move to "Model details" drawer; planners care about WAPE, not SHAP |
| Benchmark tab (Postgres vs Trino) | System-level | Move to system admin only |
| All KPI cards showing 6+ metrics | Default view | Max 4 KPIs visible by default; rest behind "See all metrics" |

The goal: when a demand planner opens the app, they see ≤7 meaningful numbers, not 40.

---

## 12. Implementation Roadmap

### Phase 1: Foundation (Weeks 1–4) — Fix what's broken

1. **IPfeature3 — Safety Stock Engine** ← blocks everything
   - Real SS formula, not stub
   - Config-driven service levels by segment
   - Connect to health score, exceptions, investment plan
2. **AI Insight deduplication & auto-resolve** — prevent DB bloat, improve UX
3. **Forecast override table** — essential for audit trail and AI learning

### Phase 2: Command Center (Weeks 5–8) — The Monday morning experience

4. **Redesign Dashboard → Command Center**
   - Priority work queue with AI ranks
   - Causal chain cards (not text blocks)
   - Daily AI digest integration
5. **DFU drawer** — click any item number to get AI context in-place
6. **Bulk actions on exception queue**

### Phase 3: Storyboard Workflow (Weeks 9–12) — End-to-end exception handling

7. **Feature 40: S&OP Storyboard**
   - Screen 1: AI Briefing
   - Screen 2: Exception triage (causal chain)
   - Screen 3: Action confirmation
   - Screen 4: Outcome review (30-day look-back)
8. **AI action execution** — Accept recommendation → system executes change
9. **Outcome tracking** — close the feedback loop

### Phase 4: Intelligence Depth (Weeks 13–16) — AI woven everywhere

10. **Inline AI annotations** on Accuracy, DFU Analysis, Inventory tabs
11. **Contextual AI hints** on policy management, EOQ, safety stock panels
12. **S&OP narrative generation** with export to PowerPoint/PDF
13. **Streaming AI responses** via SSE
14. **Semantic caching** for repeated DFU analyses

### Phase 5: Missing Frontend Panels (Weeks 17–20) — Complete the platform

15. **IPfeature1 — Demand Variability** panel in Inv. Planning tab
16. **IPfeature2 — Lead Time Variability** panel
17. **IPfeature9 — Demand Sensing** panel
18. **IPfeature10 — Safety Stock Simulation** panel with Monte Carlo visualization
19. **IPfeature13 — Investment Optimization** panel (efficient frontier)
20. **Feature 32 — Seasonality filtering** in Accuracy and DFU Analysis tabs

### Phase 6: Polish & Production Readiness (Weeks 21–24)

21. **Command palette** (⌘K)
22. **Audit trail UI** — "history" drawer on any planning entity
23. **WebSocket collaboration** — presence + conflict resolution
24. **Mobile-responsive reflow** for tablet use in warehouse/S&OP meetings
25. **Accessibility audit** — WCAG 2.1 AA compliance
26. **E2E testing** — Playwright suite for critical planner workflows

---

## 13. Quick Wins (< 1 day each)

These can be done immediately without architectural changes:

1. **Add `::NUMERIC` cast everywhere `ROUND()` is used with `double precision`** — prevents recurring SQL errors ✅ (done)
2. **Move Data Explorer, Chat, Market Intel, Jobs to secondary navigation** — reduces primary nav from 14 to ~7 items
3. **Add trend arrows to all KPI cards** — `▼ -3.2d` next to DOS number
4. **Add AI confidence tier to insight cards** — HIGH / MEDIUM / LOW based on signal duration
5. **Replace "No insights found" empty state** with portfolio health confirmation message
6. **Add `⌘K` shortcut hint** in sidebar footer
7. **Limit portfolio health bar to 4 KPIs** (current has 4 — good; don't add more)
8. **Auto-dismiss scan success banner** after 5 seconds — current persists indefinitely
9. **Add last-scan timestamp** to AI Planner header ("Last scan: 2 min ago")
10. **Color-code exception severity in table rows** (red/amber/yellow background tint) in Exception Queue

---

## 14. Design Tokens to Establish

These should be codified in `frontend/src/constants/design-tokens.ts`:

```typescript
export const AI_COLOR = '#14b8a6';          // Teal — all AI-generated content
export const CRITICAL_COLOR = '#ef4444';    // Red — action required now
export const HIGH_COLOR = '#f59e0b';        // Amber — high risk
export const MEDIUM_COLOR = '#eab308';      // Yellow — medium risk
export const HEALTHY_COLOR = '#22c55e';     // Green — all good
export const NEUTRAL_COLOR = '#6b7280';     // Gray — inactive / historical

export const MAX_PRIMARY_KPIS = 4;          // Never show more than 4 on primary view
export const MAX_LIST_ITEMS_DEFAULT = 7;    // Work queue default visible items
export const AI_CACHE_TTL_MS = 3600_000;    // 1 hour AI result cache
export const INSIGHT_STALE_DAYS = 7;        // Auto-flag stale open insights
```

---

## Summary

| Priority | What | Impact |
|---|---|---|
| 🔴 Critical | Safety Stock Engine (IPfeature3) | Unblocks 6 downstream features |
| 🔴 Critical | Command Center (replace Dashboard) | Core daily planner experience |
| 🔴 Critical | S&OP Storyboard (Feature 40) | End-to-end exception workflow |
| 🟠 High | AI action execution (one-click apply) | Closes the AI→action loop |
| 🟠 High | Causal chain visualization | Replaces text walls with scannable insights |
| 🟠 High | Outcome tracking + feedback loop | Makes AI smarter over time |
| 🟡 Medium | Inline AI annotations across tabs | AI feels native, not siloed |
| 🟡 Medium | Missing frontend panels (IP1,2,9,10,13) | Completes the Inv. Planning EPIC |
| 🟡 Medium | Command palette (⌘K) | Power user productivity |
| 🟢 Low | Mobile responsiveness | Tablet/S&OP meeting use |
| 🟢 Low | WebSocket collaboration | Multi-planner workflows |
