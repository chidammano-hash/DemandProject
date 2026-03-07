# Demand Studio — UI/UX Improvement Plan

**Author:** UI/UX + Supply Chain Expert Review
**Codebase Snapshot:** March 2026 — 14 tabs, 30 API routers, 1089 backend + 374 frontend tests
**Reference:** `docs/ai_planning_vision.md` (strategic north star)

---

## Executive Summary

Demand Studio has excellent data depth — 8 domains, full forecasting/inventory/EOQ/policy pipelines, an AI exception engine, and real-time job orchestration. The gap is workflow assembly: the platform is built as a *data warehouse with a UI* rather than a *planning cockpit for a demand planner*. Every recommendation below translates that data depth into planner-facing decisions.

**Three guiding principles:**
1. **AI is the first reader** — scans, ranks, traces causality, writes the recommendation. Planner approves, overrides, or snoozes.
2. **One screen, one job** — no tab has two competing focal points.
3. **Show less, mean more** — every metric on screen earns its place by driving a decision this week.

**Current grades (honest self-assessment):**

| Dimension | Grade | Key gap |
|---|---|---|
| Data completeness | A | All 8 domains, full pipeline |
| AI integration | C+ | AI Planner is an isolated tab, not woven in |
| Planner workflow | D | No ranked work queue, no action tracking |
| UX minimalism | C | Dense tabs, 6–40 competing KPIs per view |
| Supply chain depth | B | EOQ/policies/SS exist but SS is a stub |
| Explainability | C | Causal chain exists but buried in collapsible |
| Mobile/responsive | D | Desktop-only in practice |

---

## Part 1: Navigation Restructure

### Current State
14 sidebar items across 5 sections. Planner must know which tab holds what. Information architecture is technology-driven (tabs mirror API routers), not workflow-driven.

### Recommended Navigation Architecture

Reorganize into **three workflow modes** + one system mode:

```
┌─────────────────────────────────────────────────────────┐
│  ⚡ COMMAND CENTER          [default — opens on login]   │
│     Exception Work Queue                                │
│     Portfolio Health Pulse                              │
│     AI Planning Digest                                  │
│                                                         │
│  📊 ANALYZE                [deep dives, weekly]          │
│     Forecast Accuracy                                   │
│     DFU Deep Dive                                       │
│     Inventory Health                                    │
│     Storyboard / S&OP Deck                              │
│                                                         │
│  ⚙️  CONFIGURE              [periodic, not daily]         │
│     Replenishment Policies                              │
│     Safety Stock Targets                                │
│     EOQ Parameters                                      │
│     Cluster Management                                  │
│                                                         │
│  🔧 SYSTEM                 [admin only]                  │
│     Jobs & Automation                                   │
│     Data Explorer                                       │
│     Market Intelligence                                 │
│     Chat (NL→SQL)                                       │
└─────────────────────────────────────────────────────────┘
```

**What changes:**
- Current 14 nav items collapse to 4 modes, each containing 3–4 sub-items
- Command Center is the default landing page (replaces static Dashboard)
- Configure and System are collapsed by default; planners expand them when needed
- Chat panel moves from always-mounted bottom strip to a `⌘K` command palette item
- Jobs tab collapses to a notification badge + slide-in drawer (most planners never need the full scheduler view)

**Implementation in `AppSidebar.tsx`:**
- Add `mode` state: `command | analyze | configure | system`
- Mode headers are clickable accordion triggers
- Active item within expanded mode shows left-pill indicator (current behavior preserved)
- Keyboard shortcut `M` cycles through modes; `1–9` still navigates items within active mode
- Mobile: bottom tab bar shows 4 mode icons only; items slide up in a sheet

---

## Part 2: Page-by-Page Improvements

### 2.1 Command Center (replaces Dashboard Tab)

**Current:** Static KPI cards + alerts + heatmap + top movers + trend chart. No direct action path.

**Goal:** Monday morning work queue — planner arrives, sees 3–7 ranked items, acts on them in order.

**Layout redesign:**

```
┌──────────────────────────────────────────────────────────────────┐
│  Good morning, Manohar. Monday, Mar 5 · AI scan: 2 min ago       │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │
│  │ 7 Open   │ │ 2 Crit.  │ │ DOS 42d  │ │ $128K at risk      │  │
│  │ Insights │ │    🔴    │ │  ▼ −3d   │ │ this week          │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────────┘  │
│                                                                  │
│  ── PRIORITY WORK QUEUE ──────────────────────── Sort: AI rank ▾ │
│                                                                  │
│  ① CRITICAL  587382 @ 1401-BULK · Stockout in ~8 days           │
│     [Forecast +87% bias] → [DOS 6.4d vs LT 14d] → [Stockout]   │
│     Impact: $42K   [Review & Accept] [Override] [Snooze 3d]     │
│                                                                  │
│  ② HIGH      616372 @ 2201-MAIN · Persistent bias +49%          │
│     [Model degraded since Jan] → [WAPE 63%] → [Forecast risk]  │
│     Impact: $18K   [Review & Accept] [Override] [Snooze 3d]     │
│                                                                  │
│  ③ HIGH      179333 @ 1401-BULK · DOS 6.4d vs LT 14d            │
│     [LT increased] → [SS not recalculated] → [Stockout risk]   │
│     Impact: $11K   [Review & Accept] [Override] [Snooze 3d]     │
│                                                                  │
│  + 4 more · View all in AI Planner                              │
│                                                                  │
│  ── AI PLANNING DIGEST ──────────────── claude-sonnet · Mar '26  │
│  "Portfolio under pressure this week. Three A-class DFUs..."    │
│  Read full memo →                                                │
└──────────────────────────────────────────────────────────────────┘
```

**Specific changes to `DashboardTab.tsx`:**
1. Replace static KPI row with greeting + last-scan timestamp header
2. Keep 4 KPIs maximum: Open Insights, Critical Count, Portfolio DOS, $ at Risk (financial-first)
3. Replace heatmap + top movers with the priority work queue (top 3 items visible without scroll)
4. Each queue item shows a mini causal chain (3 nodes, inline) — not a paragraph
5. Each queue item has exactly 3 actions: Review & Accept / Override / Snooze
6. AI digest section at bottom is collapsible; shows first 2 sentences + "Read full memo →" link
7. Remove: weekly trend chart from primary view (move to Analyze → Forecast Accuracy)
8. Remove: Top Movers widget from primary view (move to Analyze → DFU Deep Dive)

**New component needed:** `WorkQueueItem.tsx` — renders a single ranked insight with mini causal chain and 3 action buttons.

---

### 2.2 AI Planner Tab

**Current (1,038 lines):** Portfolio health bar, insight cards with causal chains, confidence badges, confirm modal, auto-accept modal, planning memo. Good foundation with these gaps:

1. Causal chain nodes are hardcoded text — don't update reactively when insight data changes
2. No bulk action on multiple insights (must click each individually)
3. "Generate Now" shows a spinner but no streaming updates — planner waits 30–120s with no feedback
4. Filter pills don't persist across page navigation
5. Planning memo panel has no "copy to clipboard" or export button
6. Confidence badge logic (HIGH/MED/LOW) derived at render time — should be stored in DB

**Improvements:**

1. **Bulk selection bar** — Checkbox on each insight card. When ≥1 checked, sticky bottom bar:
   ```
   ☑ 3 selected   [Acknowledge all]  [Snooze 3 days]  [Export PDF]  [×]
   ```

2. **Streaming insight generation** — Replace polling with SSE on `/ai-planner/stream/{scan_run_id}`. Cards appear one-by-one. Show progress: "Analyzing 47 of 2,341 DFUs..."

3. **Persistent filters** — Save severity/status/type filter state to URL params (`?severity=critical,high&status=open`)

4. **Memo export** — Add "Copy markdown" and "Export PDF" buttons to planning memo panel

5. **Snooze with reason** — Snooze prompts: "Snooze until [date] — Reason: [text]". Snoozed insights get a gray card with wake-up date shown.

6. **Auto-resolve indicator** — If underlying condition resolved (DOS recovered), card shows "Auto-resolved: condition cleared" with a green badge, distinct from manual resolve.

7. **DFU history drawer** — Clicking item number on any insight card opens a right-side drawer with full DFU context: DOS trend, forecast bias trend, policy history, past insights for this DFU.

---

### 2.3 Storyboard Tab

**Current (654 lines):** 3-zone layout — summary strip + exception queue + investigation panel.

**The problem:** This is effectively a second exception queue. It overlaps with AI Planner and InvPlanningTab Exception Queue. Planners are confused about which queue to use.

**Recommended redesign — make Storyboard the S&OP preparation workflow:**

```
STORYBOARD = Monthly S&OP Narrative Builder (not a 3rd exception queue)
```

**New layout:**

```
┌─────────────────────────────────────────────────────────────────┐
│  S&OP BRIEF — March 2026                        [Generate ▸]    │
│                                                                  │
│  NARRATIVE (AI-drafted, planner-editable)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Portfolio is under pressure in Q1. Three A-class DFUs... │   │
│  │ [Rich text editor — planner can edit AI narrative]       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  KEY METRICS                    EXCEPTION SUMMARY               │
│  DOS: 42.3d ▼ −3.2d            2 Critical  · 5 High            │
│  Service Level: 94.2%           Total at risk: $172K            │
│  WAPE: 28.4% ▲ +4.1%                                           │
│                                                                  │
│  TOP 5 ITEMS FOR S&OP DISCUSSION                                 │
│  [item · issue · AI recommendation · status]                    │
│                                                                  │
│  ACTION REGISTER (carried from exception reviews)                │
│  [DFU · Action taken · Date · Owner · Outcome due]              │
│                                                                  │
│  [Export to PowerPoint]  [Export to PDF]  [Copy link to share]  │
└─────────────────────────────────────────────────────────────────┘
```

What to keep: the exception decision tracking (action register) — preserve it. Eliminate the duplicate exception queue (already handled by AI Planner).

**Backend changes needed:**
- `POST /storyboard/generate-narrative` — AI generates monthly S&OP narrative
- `PUT /storyboard/narrative/{id}` — planner saves edits
- `GET /storyboard/action-register` — items escalated from AI Planner reviews

---

### 2.4 Control Tower Tab

**Current:** 5 zones — KPI strip (6 metrics), health distribution, exception queue, critical items, 6-month trend. Auto-refreshes every 2 minutes.

**Gaps:**
1. KPI strip has 6 metrics — exceeds the `MAX_PRIMARY_KPIS = 4` rule in design tokens
2. Exception queue here and in InvPlanningTab are functionally identical — duplicate confusion
3. Critical items horizontal scroll has no keyboard navigation, no item count
4. Trend chart uses Recharts; rest of analysis tabs use ECharts — inconsistent interaction

**Improvements:**
1. **Reduce KPI strip to 4:** Portfolio Health Score, Open Critical Exceptions, Service Level, $ at Risk. Move Fill Rate, Stockout Items, Below SS %, Urgent Signals to "Details" expand.
2. **Remove duplicate exception queue** — show count badge + link to AI Planner instead.
3. **Replace horizontal scroll with a ranked list** (max 5 items visible, "View all" to AI Planner).
4. **Convert Recharts trend chart to ECharts** for consistency.
5. **Drill-through on KPI cards** — clicking opens a drawer with trend detail, not tab navigation.

---

### 2.5 Inventory Planning Tab (13-panel hub)

**Current:** 13+ panels stacked in a single scrolling page. All panels mount simultaneously — ~30 API calls on tab open. No visual hierarchy between primary and secondary panels.

**Redesign — two-tier layout:**

**Tier 1 (always visible): 3 primary panels in CSS grid**
```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│  Exception Queue    │  Portfolio Health    │  Fill Rate          │
│  (top priority)     │  (4-component score) │  (service level)    │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

**Tier 2 (tabbed sub-navigation): Secondary panels — mount only when active**
```
[Policy] [EOQ] [Safety Stock] [ABC-XYZ] [Supplier] [Demand Signals]
[Simulation] [Investment] [Variability] [Lead Time] [Intramonth]
```

**Implementation in `InvPlanningTab.tsx`:**
```tsx
const [activePanel, setActivePanel] = useState("policy");

// Tier 1 — always mounted
<ExceptionQueuePanel />
<PortfolioHealthPanel />
<FillRatePanel />

// Tier 2 — mount only when active
{activePanel === "policy"   && <PolicyManagementPanel />}
{activePanel === "eoq"      && <EoqPanel />}
{activePanel === "safety"   && <SafetyStockPanel />}
// ...
```

This reduces initial API calls from ~30 to ~6 on tab open.

---

### 2.6 Accuracy Tab

**Current:** 4 panels (Slice Table, Trend Chart, Champion Panel, SHAP Panel). Dense but well-structured.

**Improvements:**
1. **Inline AI flags per row** — In the slice table, add an AI risk column: "Persistent bias — recommend model switch" for DFUs where bias > 20% for 3+ months. Reads from `ai_insights` filtered by `insight_type = 'forecast_bias'`.
2. **Move SHAP panel behind "Advanced" accordion** — SHAP is for model developers, not daily planners. Collapse by default.
3. **Champion vs Ceiling gap as progress bar** — Visual horizontal bar per row (green < 5pp, amber 5–15pp, red > 15pp) instead of just a number.
4. **Seasonality filter prominence** (Feature 32 — wired but buried) — Move to the main filter bar row.

---

### 2.7 DFU Analysis Tab

**Current:** 3 scope modes, dual Y-axis ECharts overlay chart, per-model KPI cards.

**Improvements:**
1. **"AI says" chart annotation** — When an AI insight exists for this DFU, render a vertical reference line at the month the bias started, with a tooltip: "AI flagged: +87% bias began Jan 2026. See insight →"
2. **Forecast override input** — Editable multiplier field above the chart (e.g., 0.6× for a date range). Calls `POST /forecast/overrides` (new endpoint). Override renders as a dashed line on the chart.
3. **Export chart** — "Save as PNG" button via ECharts native `toolbox.feature.saveAsImage`.

---

### 2.8 Inventory Tab

**Current:** KPI cards + 5-line trend chart + position table + item detail.

**Improvements:**
1. **AI risk tier column** in position table — 🔴🟠🟡🟢 per item-location. Derived from `ai_insights` for this DFU + DOS threshold check. One-click opens DFU drawer.
2. **DOS alert threshold line** on trend chart — Horizontal reference line at `LT × 1.5`. Area below the line is red-shaded.
3. **KPI strip is already 4 KPIs** (DOS, WOC, Turns, LT Coverage) — keep as-is ✅.

---

### 2.9 Explorer Tab

**Current:** Domain selector, field selector, type-aware filters, virtualized data table. Solid.

**Improvements:**
1. **Move to System section** in navigation — Admin/analyst tool, not daily planner use.
2. **"Save view" button** — Save current domain + filter + column config as a named view.
3. **Cross-domain join view** — "DFU" domain shows enriched rows: item + location + DFU metadata in one table.

---

### 2.10 Jobs Tab

**Current:** Full APScheduler dashboard — 5 sections, KPI cards, live monitoring, schedules, history.

**Most planners only care: "Did the overnight AI scan succeed?"**

**Recommended:**
1. **Collapse Jobs tab to a notification drawer** — triggered by clicking the active-job-count badge in the sidebar (already exists).
2. **Drawer shows:** Last 5 job runs, success/fail status, last-run timestamp, "Run now" button.
3. **Keep full JobsTab in System section** — for administrators and data scientists.

---

## Part 3: Design System Improvements

### 3.1 Strict Semantic Color System

Codify in `constants/design-tokens.ts` (extend existing file):

```typescript
// Severity — action urgency (already partially defined)
export const CRITICAL_COLOR  = '#ef4444';  // Red   — act now
export const HIGH_COLOR       = '#f59e0b';  // Amber — act this week
export const MEDIUM_COLOR     = '#eab308';  // Yellow — monitor
export const HEALTHY_COLOR    = '#22c55e';  // Green — no action needed
export const NEUTRAL_COLOR    = '#6b7280';  // Gray  — resolved / historical

// AI brand color — ONLY for AI-generated content
export const AI_COLOR         = '#14b8a6';  // Teal

// Interactive
export const PRIMARY_COLOR    = '#3b82f6';  // Blue — buttons, links, selected state
```

**Enforcement:** Add ESLint rule: no raw hex color strings outside `design-tokens.ts`. Audit all `className` strings in tab components.

---

### 3.2 Progressive Disclosure — Apply Consistently

Every panel should follow this pattern:

| Level | What shows | How triggered |
|---|---|---|
| **0 — Default** | 1 primary KPI + AI risk tier | On page load |
| **1 — Hover** | 3–5 supporting metrics + sparkline | Mouse hover / touch |
| **2 — Expand** | Full detail, secondary charts | Click "See more ▾" |
| **3 — Drawer** | AI reasoning, full time series, history | Click item number |

Currently most panels jump to Level 2 by default. This is the primary cause of the "dense tab" problem.

**Quick fixes:**
- Inventory position table: default 5 columns (item, loc, DOS, AI tier, action) — rest behind column chooser
- Accuracy slice table: default 4 columns (group, WAPE, bias, AI flag) — rest behind "Show columns"
- InvPlanningTab panels: default closed except Exception Queue and Portfolio Health

---

### 3.3 Trend Arrows on All KPIs

Every numeric KPI should show a delta vs prior period:

```tsx
// Add to KpiCard.tsx
<span className="text-sm">
  {delta > 0 ? '▲' : '▼'} {Math.abs(delta).toFixed(1)}{unit}
  <span className="text-xs ml-1 text-muted-foreground">vs last {period}</span>
</span>
```

Color: green if improving toward threshold direction, red if worsening, gray if flat.

---

### 3.4 Unify Chart Library

**Current mix:**
- ECharts: DfuAnalysis, ControlTower, Storyboard, ForecastTrendChart
- Recharts: DashboardTab (sparklines), AccuracyTab (trend chart)

**Standardize on ECharts** for all charts:
- ECharts handles the most complex charts (dual Y-axis, zoom, export, reference lines)
- Better canvas rendering performance for large datasets
- `EChartContainer.tsx` is already a theme-aware wrapper — reuse everywhere
- ECharts `toolbox` enables native save-as-image, data zoom — no extra work

**Migration:** Replace `<AreaChart>` / `<LineChart>` from Recharts in DashboardTab and AccuracyTab. Low risk — same data, different renderer.

---

### 3.5 Typography Hierarchy

Establish consistent scale across all tabs:

```
H2: 18px/600  — Panel headings
H3: 14px/600  — Section subheadings within panels
Body: 14px/400 — Standard content
KPI: 28px/700  — Primary metric number
Label: 12px/500 — KPI caption, badge text
```

Add as Tailwind component classes in `tailwind.config.ts`:
```js
theme.extend.fontSize = {
  'kpi': ['1.75rem', { lineHeight: '1', fontWeight: '700' }],
  'label': ['0.75rem', { lineHeight: '1.25', fontWeight: '500' }],
}
```

---

### 3.6 Empty State Excellence

Replace every generic "No data found" with context-aware, action-driving states:

| Location | Recommended empty state |
|---|---|
| Command Center (no insights) | "Portfolio looks healthy! Last scan: 2 min ago. Next: Monday 06:00." |
| AI Planner (no critical) | "No critical insights. 3 medium-risk items are being monitored." |
| Exception Queue (filtered empty) | "No exceptions match current filters. Try removing the severity filter." |
| Accuracy (no champion) | "No champion model for this DFU. [Run champion selection →]" |
| Policy (unassigned DFU) | "No policy assigned. AI recommends: `continuous_rop` (A-class + low variability). [Apply →]" |

---

## Part 4: AI Integration Improvements

### 4.1 AI Woven In, Not Siloed

AI presence on every screen where a decision is made:

| Location | Add |
|---|---|
| Command Center KPIs | AI risk chip next to each metric; hover for 1-sentence diagnosis |
| DFU Analysis chart | Vertical reference line at bias start date with AI annotation |
| Accuracy slice table | Inline flag: "Persistent bias — recommend model switch" per row |
| Inventory position table | 🔴🟠🟡🟢 AI risk tier column; one-click to causal chain |
| Policy panel | AI recommendation per unassigned DFU based on cluster + variability class |
| EOQ panel | "EOQ is optimal but current order qty deviates 40%" inline note |

---

### 4.2 DFU Drawer — Context Without Tab Switch

When a planner clicks any item number anywhere in the app, a right-side drawer slides in:

```
┌─────────────────────────────────┐
│  587382 — Widget A         [×]  │
│  1401-BULK · A-class            │
│  Cluster: high_volume_steady    │
│                                 │
│  CURRENT STATUS                 │
│  DOS:   6.4d  ⚠️ (LT: 14d)     │
│  WAPE:  63%   🔴                │
│  Bias:  +87%  3-month streak    │
│                                 │
│  AI SAYS                        │
│  "Over-forecast drawing down    │
│  SS. EOQ correctly sized.       │
│  Issue: policy, not demand."    │
│                                 │
│  ACTIONS                        │
│  [ ] Trigger reorder now        │
│  [ ] Adjust forecast −40%       │
│  [ ] Change policy → ss_buffer  │
│  [ ] Flag for S&OP              │
│                                 │
│  [Apply selected]               │
│                                 │
│  [Forecast] [Inventory] [EOQ]   │
│  [Policy]   [Past Insights]     │
└─────────────────────────────────┘
```

**Implementation:** New `DfuDrawer.tsx` + `useDfuDrawer()` hook exposing `openDrawer(itemNo, loc)`. Renders as a portal over the current page. Uses existing `/dfu`, `/inventory/item-detail`, `/ai-planner/insights` endpoints.

---

### 4.3 AI Action Execution — Close the Loop

| AI Recommendation | One-click action | Endpoint |
|---|---|---|
| "Change policy to ss_buffer" | Policy change | `PUT /inv-planning/policy-assignments/{id}` |
| "Trigger emergency reorder" | Create exception | `POST /inv-planning/exceptions/generate` |
| "Adjust forecast −40%" | Forecast override | `POST /forecast/overrides` (new) |
| "Recalculate safety stock" | Trigger job | `POST /jobs/run` with `job_type: ss_compute` |
| "Escalate to S&OP" | Tag insight | `PUT /ai-planner/insights/{id}/status` |

Each action: preview → confirm → execute → outcome tracking.

---

### 4.4 Streaming AI Responses

Replace polling with SSE:
- New endpoint: `GET /ai-planner/stream/{scan_run_id}` — streams `data: {insight}` events
- Frontend uses `EventSource` in `useAiScan()` hook
- Cards appear one-by-one as AI writes them
- Progress: "Analyzing 47 of 2,341 DFUs..."
- Eliminates the current 30–120 second blank wait

---

### 4.5 Contextual Inline AI (Fast Path)

When planner opens a DFU (drawer or DFU Analysis tab), silently trigger a lightweight assessment:
- Call `POST /ai-planner/analyze` for that DFU
- Cache result for 1 hour (same DFU data = same analysis)
- Render in the "AI SAYS" drawer section within 2–3 seconds
- Use `claude-haiku-4-5` for speed; full portfolio scan uses `claude-sonnet-4-6`

---

## Part 5: Supply Chain Domain Completions

### 5.1 Safety Stock Engine — Critical Foundation (IPfeature3)

`fact_safety_stock_targets` is a stub. Blocks health scores, exception generation, investment planning.

**Formula:** `SS = Z × σ_demand × √(LT)` where Z is service-level Z-score by ABC class.

**Connect to:** Health score (IPfeature6), exception generation (IPfeature7), investment planning (IPfeature13).

### 5.2 Forecast Override Management

Currently missing entirely. New table `fact_forecast_overrides`:

```sql
item_no          TEXT NOT NULL,
loc              TEXT NOT NULL,
override_period  DATE NOT NULL,
statistical_qty  NUMERIC(14,4),
override_qty     NUMERIC(14,4) NOT NULL,
override_reason  TEXT,
override_source  TEXT DEFAULT 'manual',  -- 'manual' | 'ai_recommendation' | 'system'
ai_insight_id    INTEGER REFERENCES ai_insights(insight_id),
outcome_wape     NUMERIC(8,4),           -- populated 30 days later
UNIQUE (item_no, loc, override_period)
```

**UX:** Inline editable forecast cells in DFU Analysis tab. AI validates: "Your override (0.6×) aligns with AI recommendation (0.63×). ✅"

### 5.3 AI Insight Lifecycle Management

1. Upsert on `(item_no, loc, insight_type, status='open')` — one open insight per type per DFU
2. Staleness: 7+ days old and condition unchanged → `stale` status badge
3. Auto-resolve: if DOS recovers above threshold, auto-close with `auto_resolved_at` timestamp
4. Lifecycle: `open → acknowledged → in_progress → resolved | auto_resolved | snoozed`

### 5.4 Working Capital Dashboard

Connect inventory investment to business KPIs:
- **Cash-to-cash cycle** — DOS + Days Receivable − Days Payable
- **Service level per dollar** — Fill rate achieved / Inventory value
- **Budget constraint display** — "At current trajectory, working capital will exceed $Xm target by [date]"

---

## Part 6: Technical Improvements

### 6.1 Command Palette (⌘K)

```
⌘K → modal overlay
┌──────────────────────────────────────┐
│  > _                                 │
│  RECENT                              │
│  587382 @ 1401-BULK  (DFU analysis) │
│  Portfolio scan  (2 min ago)         │
│  ACTIONS                             │
│  Run portfolio scan now              │
│  Generate S&OP memo                  │
│  Go to exception queue               │
│  SEARCH                              │
│  Type item, location, or command…   │
└──────────────────────────────────────┘
```

New `CommandPalette.tsx`. State in `CommandPaletteContext`. Listener: `Cmd+K` / `Ctrl+K`. Fuzzy search over nav items, recent DFUs (localStorage), quick actions. Replaces always-mounted Chat panel as the primary power-user interface.

---

### 6.2 Lazy Panel Loading in InvPlanningTab

```tsx
// Before: all 13 panels mount and query simultaneously
// After:
const [activePanel, setActivePanel] = useState("policy");

// Tier 1 — always mounted (3 panels)
<ExceptionQueuePanel />
<PortfolioHealthPanel />
<FillRatePanel />

// Tier 2 — mount only when active
{activePanel === "policy"  && <PolicyManagementPanel />}
{activePanel === "eoq"     && <EoqPanel />}
{activePanel === "safety"  && <SafetyStockPanel />}
// ... etc
```

Reduces initial API calls from ~30 to ~6 on tab open.

---

### 6.3 Query Optimization

| Tab | Current | Improvement |
|---|---|---|
| AccuracyTab | 6 parallel queries | Stagger: KPI + slice first, SHAP deferred 500ms |
| InvPlanningTab | ~30 parallel queries | Lazy panel loading (above) eliminates most |
| ControlTowerTab | 4 queries + 2-min refresh | Reduce to 10-min interval when browser tab is hidden |

---

### 6.4 Accessibility Fixes

Priority items:
1. Data tables: add `scope="col"` to all `<th>` elements in `DataTable.tsx`
2. Modal dialogs: add `role="dialog"` + `aria-modal="true"` + focus trap to `ConfirmModal`
3. ARIA live regions: add `role="status"` to job completion + scan progress notifications
4. ECharts: add `aria-label` describing chart type and data range to `EChartContainer.tsx`
5. DFU drawer: focus moves to drawer when opened; returns to trigger element on close

---

## Part 7: Quick Wins (< 1 day each)

| # | Change | File | Impact |
|---|---|---|---|
| 1 | Trend arrows (▲▼) on all KpiCard instances | `components/KpiCard.tsx` | Immediate readability |
| 2 | Move Explorer, Chat, Market Intel to System nav | `components/AppSidebar.tsx` | Primary nav 14 → 8 items |
| 3 | Lazy Tier 2 panels in InvPlanningTab | `tabs/InvPlanningTab.tsx` | Eliminates ~24 API calls on open |
| 4 | "Copy markdown" button on planning memo | `tabs/AIPlannerTab.tsx` | Speeds up S&OP prep |
| 5 | Persistent filters via URL params in AI Planner | `tabs/AIPlannerTab.tsx` | Survives page refresh |
| 6 | Reduce Control Tower KPI strip 6 → 4 metrics | `tabs/ControlTowerTab.tsx` | Meets MAX_PRIMARY_KPIS = 4 |
| 7 | Add `⌘K` hint in sidebar footer | `components/AppSidebar.tsx` | Discoverability |
| 8 | ECharts saveAsImage toolbox on DFU chart | `tabs/dfu-analysis/OverlayChartPanel.tsx` | One-line config add |
| 9 | Column chooser on inventory position table | `tabs/inventory/PositionTablePanel.tsx` | Default 5 cols, rest selectable |
| 10 | Context-aware empty states (all tabs) | Tab-specific empty branches | Reduces user confusion |
| 11 | Snooze with reason + date picker | `tabs/AIPlannerTab.tsx` | Richer audit trail |
| 12 | Bulk checkbox + action bar on AI Planner | `tabs/AIPlannerTab.tsx` | Exception triage 10× faster |
| 13 | DOS reference line on inventory trend chart | `tabs/inventory/TrendChartPanel.tsx` | Immediate visual alert |
| 14 | Auto-collapse SHAP panel by default | `tabs/AccuracyTab.tsx` | Reduces tab density |
| 15 | Move Storyboard to Analyze nav section | `components/AppSidebar.tsx` | Correct info architecture |

---

## Part 8: Implementation Roadmap

### Phase 1 — Fix Foundations (Weeks 1–4)
Fix what's broken before adding new features.

1. **Safety Stock Engine** (IPfeature3) — real SS formula, connect to health/exceptions/investment
2. **AI Insight deduplication + auto-resolve** — upsert logic, staleness flag, auto-close on condition clear
3. **Forecast override table** — DDL + API + DFU Analysis editable cells
4. **All 15 quick wins** from Part 7

---

### Phase 2 — Command Center (Weeks 5–8)
The Monday morning experience.

5. **Redesign Dashboard → Command Center** — priority work queue, mini causal chains, 4 KPIs
6. **DFU Drawer** — `DfuDrawer.tsx` + `useDfuDrawer()`, accessible from any item number
7. **Bulk actions on AI Planner exception queue** — checkbox multi-select + sticky action bar
8. **Navigation restructure** — 4-mode sidebar (Command / Analyze / Configure / System)
9. **Command Palette** (⌘K) — fuzzy search nav + actions + recent DFUs

---

### Phase 3 — S&OP Storyboard Workflow (Weeks 9–12)
End-to-end exception handling with outcome tracking.

10. **Redesign Storyboard → S&OP Narrative Builder** — AI-drafted memo, rich text editor, action register
11. **AI action execution** — one-click policy change, exception create, forecast override, job trigger
12. **Streaming AI responses** — SSE endpoint, `useAiScan()` with `EventSource`
13. **Outcome tracking UI** — 30-day look-back card on resolved insights, AI accuracy score in header

---

### Phase 4 — AI Woven In (Weeks 13–16)
AI presence on every decision-making screen.

14. **Inline AI annotations** — DFU Analysis chart reference lines, Accuracy table flags, Inventory risk tiers
15. **Contextual inline AI** — DFU drawer assessment (Haiku, cached 1hr)
16. **AI policy recommendations** — Policy panel shows AI recommendation per unassigned DFU
17. **Lazy panel loading** in InvPlanningTab

---

### Phase 5 — Design System Polish (Weeks 17–20)

18. **Unify chart library** — Replace Recharts with ECharts in DashboardTab and AccuracyTab
19. **Typography hierarchy** — Codify scale in `tailwind.config.ts`
20. **Color token enforcement** — ESLint rule + audit raw hex colors
21. **Progressive disclosure** — Apply Level 0–3 pattern across all 14 tabs
22. **Working Capital Dashboard** — Connect IPfeature13 efficient frontier to cash-cycle metrics

---

### Phase 6 — Production Readiness (Weeks 21–24)

23. **Mobile responsive reflow** — Tablet layout for S&OP meeting use
24. **Accessibility audit** — WCAG 2.1 AA: ARIA, focus management, screen reader
25. **E2E test suite** — Playwright: Monday morning exception triage flow
26. **Audit trail UI** — "History" drawer on any planning entity
27. **WebSocket collaboration** — Presence indicators, lock-on-edit for multi-planner teams

---

## Part 9: Component Architecture

### New Components to Build

| Component | Purpose | Priority |
|---|---|---|
| `WorkQueueItem.tsx` | Ranked insight card with mini causal chain + 3 actions | P0 |
| `DfuDrawer.tsx` | Right-side context drawer for any DFU | P0 |
| `CommandPalette.tsx` | ⌘K fuzzy search over nav + actions + DFUs | P1 |
| `StreamingScanProgress.tsx` | SSE progress indicator during AI portfolio scan | P1 |
| `SnoozeDialog.tsx` | Date picker + reason for snoozed insights | P1 |
| `BulkActionBar.tsx` | Sticky bottom bar when multiple insights selected | P1 |
| `ForecastOverrideCell.tsx` | Editable cell in DFU Analysis chart | P2 |
| `OutcomeReviewCard.tsx` | 30-day look-back result for resolved insight | P2 |
| `SopNarrativeEditor.tsx` | Rich text editor for AI-drafted S&OP memo | P2 |
| `AiAnnotation.tsx` | Chart overlay annotation at bias/divergence point | P2 |
| `PlanningScorecard.tsx` | Weekly planner performance metrics card | P3 |

### Components to Consolidate

| Duplication | Action |
|---|---|
| `KpiCard` defined inline in DashboardTab, ControlTowerTab, StoryboardTab | Enforce import from `components/KpiCard.tsx` (exists) |
| Severity color maps defined locally in multiple tabs | Centralize in `constants/design-tokens.ts` |
| Loading skeleton patterns duplicated across tabs | Create `components/PanelSkeleton.tsx` with KPI/table/chart slots |
| Empty state fragments duplicated | Create `components/EmptyState.tsx` with `message` + `action` props |

---

## Part 10: The 5 KPIs That Matter

Reduce the KPI surface to what planners actually use for decisions. Everything else is Level 2.

| KPI | Definition | Alert threshold | Target |
|---|---|---|---|
| **DOS** | On-hand / avg daily demand | < LT × 1.5 | Category-specific |
| **Service Level** | Units shipped / units ordered | < 95% for A-class | 98% A · 95% B · 90% C |
| **Forecast Bias** | (ΣForecast / ΣActual) − 1 | > ±20% for 3 months | ±5% |
| **WAPE** | Champion model accuracy | > 35% high · > 50% critical | < 25% |
| **Inventory Turns** | Annual demand / avg inventory | < 4× for A-class | 8–12× |

WOC, LT Coverage, cycle stock value, and all other metrics remain accessible behind "See all metrics" but do not occupy primary screen real estate.

---

## Summary Priority Matrix

| Priority | Item | Effort | Impact |
|---|---|---|---|
| 🔴 P0 Critical | Safety Stock Engine (foundation) | L | Unblocks 6 features |
| 🔴 P0 Critical | Navigation restructure (4 modes) | M | −60% cognitive load |
| 🔴 P0 Critical | Command Center (work queue) | M | Core daily experience |
| 🔴 P0 Critical | All 15 quick wins | S | Immediate improvement |
| 🟠 P1 High | DFU Drawer (context without tab switch) | M | Eliminates tab-switching friction |
| 🟠 P1 High | Bulk actions on AI Planner | S | Exception triage 10× faster |
| 🟠 P1 High | Lazy panel loading in InvPlanningTab | S | 5× faster tab open |
| 🟠 P1 High | Streaming AI responses (SSE) | M | Perceived performance 10× better |
| 🟡 P2 Medium | Storyboard → S&OP Narrative Builder | L | End-to-end planning workflow |
| 🟡 P2 Medium | AI action execution (one-click apply) | L | Closes the AI→action loop |
| 🟡 P2 Medium | Inline AI annotations across tabs | M | AI native, not siloed |
| 🟡 P2 Medium | Chart library unification (ECharts) | M | Consistent interaction model |
| 🟡 P2 Medium | Command Palette (⌘K) | M | Power user productivity |
| 🟢 P3 Low | Forecast override management | L | Audit trail + AI learning |
| 🟢 P3 Low | Mobile responsive reflow | L | Tablet S&OP meeting use |
| 🟢 P3 Low | WebSocket collaboration | XL | Multi-planner workflows |
| 🟢 P3 Low | Accessibility WCAG 2.1 AA | M | Compliance |

**Effort key:** S = < 1 day · M = 1–5 days · L = 1–3 weeks · XL = > 3 weeks

---

*Last updated: March 2026*
