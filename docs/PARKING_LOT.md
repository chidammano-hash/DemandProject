# Parking Lot — Known Issues & Deferred Work

Issues captured here are real, confirmed problems that are not yet fixed.
Each entry includes root cause, impact, and a recommended fix when known.

---

## Fixed Issues

---

### PL-001 — Multiple Backtests Overwrite the Same Output CSVs

**Status:** Fixed — 2026-02-28
**Priority:** High
**Date captured:** 2026-02-28
**Fixed in:** `common/backtest_framework.py`, `scripts/load_backtest_forecasts.py`, `Makefile`
**Affects:** `run_backtest.py`, `run_backtest_catboost.py`, `run_backtest_xgboost.py`

#### Problem

Every backtest script (LGBM, CatBoost, XGBoost — all strategies) wrote output to the **same two fixed file paths** regardless of model or strategy:

```
mvp/demand/data/backtest/backtest_predictions.csv
mvp/demand/data/backtest/backtest_predictions_all_lags.csv
```

Running a second backtest before loading the first silently overwrote both CSVs, losing the first model's predictions permanently.

#### Fix Applied (Option B — Model-scoped subdirectory)

`common/backtest_framework.py` → `save_backtest_output()` now writes each model into its own subdirectory:

```
data/backtest/lgbm_cluster/backtest_predictions.csv
data/backtest/lgbm_cluster/backtest_predictions_all_lags.csv
data/backtest/catboost_cluster/backtest_predictions.csv
data/backtest/catboost_cluster/backtest_predictions_all_lags.csv
data/backtest/xgboost_cluster/backtest_predictions.csv
data/backtest/xgboost_cluster/backtest_predictions_all_lags.csv
```

`scripts/load_backtest_forecasts.py` was refactored to support:
- `--model MODEL_ID` — load a single model from `data/backtest/<MODEL_ID>/`
- `--all` — discover and load all models under `data/backtest/*/`
- `--input PATH` — backward-compatible explicit path

Makefile targets updated:
- `make backtest-load MODEL=lgbm_cluster` — load one model
- `make backtest-load-all` — load all available models

You can now batch multiple backtests safely:

```bash
make backtest-lgbm-cluster
make backtest-catboost-cluster
make backtest-xgboost-cluster
make backtest-load-all           # loads all three in sequence
```

---

---

### PL-002 — Hyperparameter Tuning Uses Full History, Causing Data Leakage Into Backtests

**Status:** Fixed — 2026-02-28
**Priority:** High
**Date captured:** 2026-02-28
**Fixed in:** `common/tuning.py`, `common/backtest_framework.py`, `scripts/run_backtest*.py`, `config/hyperparameter_tuning.yaml`, `Makefile`
**Affects:** `scripts/tune_hyperparams.py`, `common/tuning.py`, all tree-based backtest scripts (`run_backtest.py`, `run_backtest_catboost.py`, `run_backtest_xgboost.py`)

#### Problem

The current tuning pipeline (`make tune-lgbm/catboost/xgboost`) runs Optuna over the **full sales history** to find optimal hyperparameters. Those tuned parameters are then passed via `--params-file` to backtest scripts that evaluate model accuracy across 10 expanding timeframes (A–J).

This introduces **temporal data leakage**: the tuner has already seen observations from future timeframes (e.g. timeframe J) when selecting parameters that are then applied to earlier timeframes (e.g. timeframe A). The backtest accuracy numbers are therefore optimistically biased — the model was implicitly tuned on the data it is being evaluated against.

#### Example of the leak

```
Tuning window:  Jan-2020 → Dec-2025  (all history)
Timeframe A:    Jan-2020 → Dec-2022  (train) | Jan-2023 (test)
Timeframe J:    Jan-2020 → Nov-2025  (train) | Dec-2025 (test)

Params tuned on full history reflect signal from Dec-2025 data,
then those same params are used to "predict" Dec-2025 in Timeframe J.
```

#### Root Cause

`tune_hyperparams.py` does not receive a `cutoff_date` argument. `common/tuning.py` CV splits operate on whatever data is passed in — it has no knowledge of the backtest timeframe being evaluated.

#### Recommended Fix

Tune hyperparameters **within each backtest timeframe** using only the training data available at that point:

1. For each timeframe T (A–J), derive the training cutoff date from `TIMEFRAMES[T]`.
2. Filter sales history to `date < cutoff_T` before calling the Optuna study.
3. Use the timeframe-specific best params to train the model for that timeframe's test window.
4. This can be implemented as a new `--tune-inline` flag in each backtest script, or by pre-computing a params file per timeframe before the backtest loop.

**Result:** 10 separate param sets (one per timeframe), each tuned on strictly causal data. Backtest accuracy reflects true out-of-sample performance with no future leakage.

#### Impact

- All backtest accuracy metrics produced with a shared `--params-file` are optimistically biased (degree unknown, likely moderate for stable datasets, high for datasets with trend shifts or regime changes).
- Champion selection downstream of the backtest inherits this bias.
- The current approach is still useful for *production scoring* (tune on all history, apply to future), but **must not be used to evaluate or compare backtest accuracy**.

#### Workaround (superseded)

Run backtests without `--params-file` (default hyperparameters have no leakage). Use `--params-file` only when generating production forecasts, not for accuracy benchmarking.

#### Fix Applied

**`common/tuning.py`** — Added `tune_for_timeframe(model_name, train_fold_fn, full_grid, feature_cols, cat_cols, cutoff_date, config, n_trials)` which:
- Filters `full_grid` to months `<= cutoff_date` before building any CV splits
- Runs a lightweight Optuna study (default 20 trials, 3 folds) strictly in-sample
- Returns `(best_params_dict, best_n_estimators)` — ready to merge into the timeframe's model params

Also moved `_train_lgbm_fold`, `_train_catboost_fold`, `_train_xgboost_fold` from `tune_hyperparams.py` into `common/tuning.py` as `train_lgbm_fold`, `train_catboost_fold`, `train_xgboost_fold` and exposed them in `TRAIN_FOLD_FNS` registry.

**`common/backtest_framework.py`** — `run_tree_backtest()` accepts a new optional `inline_tuner_fn` parameter:
```python
inline_tuner_fn: Callable[[full_grid, feature_cols, cat_cols, train_end], dict] | None = None
```
When provided, each timeframe calls the tuner before training and uses `effective_params` instead of the static `model_params`.

**`scripts/run_backtest*.py`** — All three scripts accept new CLI flags:
- `--tune-inline` — enable per-timeframe causal tuning
- `--tune-n-trials N` — override trial count per timeframe
- `--tune-config PATH` — override tuning YAML path
- Mutual exclusion enforced: `--params-file` and `--tune-inline` cannot be combined

**`config/hyperparameter_tuning.yaml`** — Added:
```yaml
inline_n_trials: 20    # trials per timeframe
inline_n_splits: 3     # CV folds per timeframe
```

**Makefile** — New targets:
```bash
make backtest-lgbm-cluster-tuned       # LGBM per-cluster with inline tuning
make backtest-catboost-cluster-tuned   # CatBoost per-cluster with inline tuning
make backtest-xgboost-cluster-tuned    # XGBoost per-cluster with inline tuning
```

**Performance note:** Each of the 10 timeframes runs 20 Optuna trials × 3 CV folds = 60 model fits per timeframe (600 total vs. 250 for global one-shot tuning). Expect ~2–3× longer runtime compared to an untuned backtest. The trade-off is genuine out-of-sample accuracy with no future leakage.

**Two-mode workflow:**
- **Production scoring** (tune once on all history, apply to future): `make tune-lgbm && make backtest-lgbm-cluster ARGS="--params-file data/tuning/best_params_lgbm.json"`
- **Honest backtesting** (per-timeframe causal tuning): `make backtest-lgbm-cluster-tuned`

---

## Open Issues

---

### PL-003 — "Storyboard" Tab Name is Misleading for Supply Chain Planners

**Status:** Fixed — 2026-03-06
**Priority:** P0 — Confusing to end users
**Date captured:** 2026-03-06
**Affects:** `frontend/src/tabs/StoryboardTab.tsx`, `AppSidebar.tsx`, `useUrlState.ts`

#### Problem

The tab is named "Storyboard" — a term from media production, not supply chain management. Every planner who opens this product will look for "Exceptions", "Alerts", or "Alert Triage" when they need to work through exception queues. The current name actively misdirects users and creates onboarding confusion.

#### Impact

- New planners skip this tab or take too long to find it
- The sidebar navigation reads: AI Planner → Control Tower → … → **Storyboard** — the odd-one-out is jarring
- Internal docs and training materials must explain the naming mismatch

#### Recommended Fix

Rename to **"Exceptions"** or **"Alert Triage"**. Update the tab key (`storyboard` → `exceptions`) in `useUrlState.ts`, `App.tsx` lazy import, `AppSidebar.tsx` nav item, keyboard shortcut help, and all API/query references.

---

### PL-004 — AI Planner and Storyboard Have Overlapping Scope Without Clear Distinction

**Status:** Fixed — 2026-03-06
**Priority:** P0 — Planners don't know which tab to use for exceptions
**Date captured:** 2026-03-06
**Affects:** `AIPlannerTab.tsx`, `StoryboardTab.tsx`

#### Problem

Both AI Planner and Storyboard surface "exceptions" — stockout risk, excess inventory, forecast bias — with similar severity badges and acknowledge/resolve workflows. A planner looking at an item-level alert doesn't know whether to go to AI Planner or Storyboard. The distinction (AI-generated insights vs. rule-based exceptions) is not surfaced anywhere in the UI.

**Concrete overlap:**
- Both show `stockout_risk` type events with severity levels
- Both have "Acknowledge" / "Resolve" action buttons
- Both have filter pills for severity and type
- Both cards look visually similar

#### Impact

- Duplicate triage work: planners may action the same item in both tabs
- Outcome data is split across `ai_recommendation_outcomes` and `fact_storyboard_exceptions` with no unified view
- No single "my exception queue" view for a planner starting their day

#### Recommended Fix

1. Add a clear explanation at the top of each tab: e.g. *"AI Planner shows ML-generated insights ranked by financial impact. Storyboard shows rule-based threshold breaches from your replenishment policies."*
2. Cross-link: AI Planner insight cards should show a "See also: 2 policy exceptions" link when a matching item has open storyboard exceptions.
3. Long-term: evaluate merging both queues into a single "Work Queue" view with source type filter.

---

### PL-005 — Global Filter Bar Appears on Tabs Where It Has No Effect

**Status:** Fixed — 2026-03-06
**Priority:** P1 — Wastes screen real estate and confuses users
**Date captured:** 2026-03-06
**Affects:** `App.tsx`, `GlobalFilterBar.tsx`

#### Problem

The Global Filter Bar (Brand, Category, Item, Location, Market, Channel, Time Grain) renders on **every tab** unconditionally, including tabs that don't consume those filters at all:

- **AI Planner** — insight queries use their own severity/status/type filters; global filters are ignored
- **Jobs** — the automation dashboard has no filtering concept
- **Chat** — the natural language query interface ignores global filters
- **Clusters** — clustering scenarios have their own parameter UI
- **Inv. Backtest** — backtest comparison has its own model selectors

This creates confusion: a planner sets "Item = 100320" in the global filter bar, then switches to AI Planner and assumes insights are filtered to that item — they are not.

#### Impact

- False sense of filtering: users believe the active filters apply everywhere
- 40px of vertical space consumed on tabs where it provides no value
- Mobile/condensed views are even more impacted

#### Recommended Fix

Pass `activeTab` into `GlobalFilterBar` (or control rendering in `App.tsx`) and hide the bar on tabs that don't consume global filters: `aiPlanner`, `jobs`, `chat`, `clusters`, `invBacktest`, `storyboard`.

Alternatively, show a subtle "Filters not applied on this tab" badge on affected tabs.

---

### PL-006 — Storyboard Exception Severity Rendered as Raw Decimal Instead of Label

**Status:** Fixed — 2026-03-06
**Priority:** P1 — Numeric severity is meaningless to planners
**Date captured:** 2026-03-06
**Affects:** `frontend/src/tabs/StoryboardTab.tsx`

#### Problem

The Storyboard exception cards render `severity_score` as a raw decimal (e.g. `0.73`) via a badge. Supply chain planners think in terms of **Critical / High / Medium / Low**, not 0–1 float scores. The decimal is an internal computation artifact that should never be exposed in the UI.

Additionally, the `exception_type` is shown in snake_case (`stockout_risk`, `excess_inventory`) without human-friendly labels.

#### Impact

- Planners can't prioritize by glancing at the badge — they must mentally decode each number
- Inconsistent with AI Planner which correctly shows `critical` / `high` / `medium` / `low` severity badges
- Snake_case exception types look like debug output, not a production interface

#### Recommended Fix

Map `severity_score` to a label tier in `StoryboardTab.tsx`:
```typescript
function severityLabel(score: number): "critical" | "high" | "medium" | "low" {
  if (score >= 0.75) return "critical";
  if (score >= 0.50) return "high";
  if (score >= 0.25) return "medium";
  return "low";
}
```

Map exception types to human labels: `stockout_risk → "Stockout Risk"`, `excess_inventory → "Excess Inventory"`, etc.

---

### PL-007 — "Generate Now" Button Provides No Feedback After Click

**Status:** Fixed — 2026-03-06
**Priority:** P1 — Planners assume the action silently failed
**Date captured:** 2026-03-06
**Affects:** `frontend/src/tabs/AIPlannerTab.tsx`

#### Problem

The "Generate Now" button triggers `POST /ai-planner/portfolio-scan` which returns HTTP 202 (accepted, runs in background). The button briefly shows "Starting…" during the async call, but once the 202 response arrives, the UI returns to its default state with no indication that a scan is in progress, how long it will take, or when it completes.

A planner who clicks "Generate Now" has no idea:
- Whether the scan started successfully
- How many items are being analyzed
- When new insights will appear
- Whether the scan completed without errors

#### Impact

- Planners click the button multiple times (each triggering a new background scan)
- Frustration when insights don't immediately update
- No connection between "I clicked this button 2 minutes ago" and "new insights just appeared"

#### Recommended Fix

1. After a 202 response, show a persistent progress banner: *"Portfolio scan in progress — insights will refresh automatically when complete."*
2. Integrate with the Jobs system: `POST /ai-planner/portfolio-scan` should register a job in `job_history` so planners can see it in the Jobs tab
3. Auto-refresh the insights query once via a timed interval (e.g., poll every 30s for 5 minutes after triggering a scan, then stop)

---

### PL-008 — Auto-Accept Bulk Action Has No Undo / Revert

**Status:** Fixed — 2026-03-06
**Priority:** P1 — Irreversible bulk state change with no safeguard
**Date captured:** 2026-03-06
**Affects:** `frontend/src/tabs/AIPlannerTab.tsx`, `api/routers/ai_planner.py`

#### Problem

`POST /ai-planner/auto-accept` bulk-updates N insight rows from `open` → `acknowledged` and writes N outcome records to `ai_recommendation_outcomes`. There is no undo, no bulk-revert endpoint, and no way to recover if the planner ran it with the wrong severity threshold.

The modal only has a single confirmation step (no "Are you sure? This will affect 47 insights.") after the preview — the planner goes from preview (dry_run=true) directly to execute with one click.

#### Impact

- A planner who previews "47 insights" and clicks "Auto-Accept" has permanently changed 47 records
- If they meant `critical` but selected `medium`, all medium insights are now acknowledged with no clear audit trail of *why* they were acknowledged
- `action_taken` is recorded as `null` for auto-accepted records, reducing feedback loop quality

#### Recommended Fix

1. After dry-run preview, add a confirmation dialog: *"This will accept 47 open insights. This cannot be undone. Proceed?"*
2. Store `action_taken = 'auto_accepted via bulk rule: severity >= {min_severity}'` in `ai_recommendation_outcomes`
3. Add a `POST /ai-planner/auto-accept/revert` endpoint that resets `auto_accepted` insights back to `open` within a configurable time window (e.g., 1 hour)

---

### PL-009 — Inv. Planning Tab Loads 14 Panels as a Single Scrollable Page

**Status:** Fixed — 2026-03-06
**Priority:** P1 — Overwhelming UX and slow initial load
**Date captured:** 2026-03-06
**Affects:** `frontend/src/tabs/InvPlanningTab.tsx`

#### Problem

The Inventory Planning tab renders all 14 panels (Exception Queue, Portfolio Health, EOQ, Policy Management, Fill Rate, ABC-XYZ, Supplier, Intramonth, Safety Stock, Demand Variability, Lead Time, Demand Signals, Simulation, Investment) on a single infinitely-scrolling page. Each panel fires its own API query on mount.

For a planner arriving at this tab:
- The initial load fires 14+ API calls simultaneously
- The page is 5–6 full screen heights tall with no quick way to reach a specific panel
- There's no visual hierarchy indicating which panels are "primary" vs. supporting context
- Panels that are interdependent (e.g. Safety Stock drives Exception Queue and Portfolio Health) are not grouped

#### Impact

- Slow first paint on the tab (14 concurrent requests)
- Planners who only need the Exception Queue must scroll past 3 other panels to reach it
- Cognitive overload — 14 separate data stories presented as one monolithic scroll
- No "focus mode" for planners who use only 2–3 panels regularly

#### Recommended Fix

1. Add sub-navigation (horizontal tab strip or left anchor list): *Exception Queue | Health | EOQ | Policy | Fill Rate | ABC-XYZ | Supplier | Signals | Simulation | Investment*
2. Lazy-render panels below the fold — only mount queries when the panel scrolls into view using `IntersectionObserver`
3. Allow panels to be collapsed/pinned by the planner; persist preference in `localStorage`
4. Consider a "Daily Planning" mode that shows only Exception Queue + Portfolio Health (the 2 panels planners use daily)

---

### PL-010 — Control Tower Has No Navigation Links to Detailed Views

**Status:** Fixed — 2026-03-06
**Priority:** P2 — Command center is a dead end
**Date captured:** 2026-03-06
**Affects:** `frontend/src/tabs/ControlTowerTab.tsx`

#### Problem

The Control Tower tab is designed as a command center (see-everything-at-a-glance), but its cards, charts, and exception lists have no navigation to the detailed analysis:

- The **Top-Critical Items** list has a "View Detail" button — it navigates nowhere (or navigates to a dead link)
- The **Exception alert strip** has no "View All" link to the Storyboard or Inv. Planning tab
- Clicking a tier in the **health distribution chart** does not filter the exception list
- The **trend chart** shows 6 months but has no drill-down to see the raw data behind a spike

The Control Tower currently functions as a read-only poster rather than an operational launch pad.

#### Impact

- Planners see a critical alert in the strip but must manually navigate to another tab and search for it
- The "View Detail" button creates confusion — it appears clickable but leads nowhere
- The tab provides situational awareness but no path to action

#### Recommended Fix

1. "View Detail" on top-critical items → set item_no+loc in global filters and navigate to Inventory or DFU Analysis tab
2. "View All Exceptions" link in the alert strip → navigate to Storyboard/Exceptions tab
3. Health tier chips → click to pre-filter the Inv. Planning exception queue
4. Trend chart points → click to see month-level breakdown

---

### PL-011 — AI Planner Confidence Badges Have No Tooltip or Explanation

**Status:** Fixed — 2026-03-06
**Priority:** P2 — Planners don't know what "High confidence" means
**Date captured:** 2026-03-06
**Affects:** `frontend/src/tabs/AIPlannerTab.tsx`

#### Problem

Insight cards show confidence tier badges ("High confidence", "Med confidence", "Low confidence") derived from WAPE thresholds. There is no tooltip, help text, or documentation in the UI explaining:
- What "confidence" means in this context (forecast accuracy quality)
- What WAPE thresholds define each tier
- How a planner should adjust their response based on confidence level

A planner seeing "Low confidence" on a critical stockout risk doesn't know whether to trust the insight or ignore it.

#### Impact

- Planners either ignore confidence badges entirely (treating all insights as equal) or over-interpret them
- Low-confidence critical insights may be auto-accepted when they should be reviewed manually
- High-confidence medium insights may be dismissed when they should be prioritized

#### Recommended Fix

Add a `title` tooltip on confidence badges:
- `"High confidence — forecast WAPE < 20%. This insight is based on reliable forecasts."`
- `"Medium confidence — forecast WAPE 20–40%. Review the AI reasoning before acting."`
- `"Low confidence — forecast WAPE > 40%. Consider this a signal, not a directive."`

Add a help icon (?) in the insights panel header linking to a brief explanation.

---

### PL-012 — No "Snooze" Action for AI Insights

**Status:** Fixed — 2026-03-06
**Priority:** P2 — Forces premature accept/resolve decisions
**Date captured:** 2026-03-06
**Affects:** `frontend/src/tabs/AIPlannerTab.tsx`, `api/routers/ai_planner.py`

#### Problem

The only actions a planner can take on an insight are **Accept** (acknowledges) and **Resolve** (marks done). There is no way to:
- Defer an insight to review tomorrow
- Mark it as "watching — no action needed yet"
- Flag it for another team member

In practice, planners often see a valid insight but need more information before deciding. The only alternative to accepting prematurely is leaving the insight as `open`, where it stays in the queue forever and gets harder to distinguish from new issues.

#### Impact

- The insights queue fills up with `open` insights that are seen-but-not-actionable, burying newly generated ones
- Planners develop workarounds (accept everything to clear the queue) which degrades the feedback loop
- Multi-day items (e.g., waiting for supplier confirmation before resolving a stockout) have no supported state

#### Recommended Fix

1. Add `snoozed_until: datetime` column to `ai_insights`
2. Add a "Snooze" button (clock icon) on insight cards: snooze options of 1 day, 3 days, 7 days, custom
3. Snoozed insights are hidden from the default view but visible with `status=snoozed` filter
4. Auto-wake snoozed insights when `snoozed_until` passes (set status back to `open` via a scheduled job or on next read)

---

### PL-013 — Dashboard Alerts Panel Has No Navigation Link to Source Tab

**Status:** Fixed — 2026-03-06
**Priority:** P2 — Alerts are informational-only; no path to action
**Date captured:** 2026-03-06
**Affects:** `frontend/src/tabs/DashboardTab.tsx`, `frontend/src/components/AlertPanel.tsx`

#### Problem

The Dashboard/Overview tab shows an alert panel with severity-coded alerts (e.g., "Forecast accuracy dropped below 70% for cluster HIGH_VOLUME"). These alerts are informational summaries but clicking them or any element in the panel does not navigate to the relevant tab where the planner can investigate.

A planner seeing "Critical: 12 items at stockout risk" wants to immediately jump to those items — the current UI requires them to manually switch tabs and search.

#### Impact

- Dashboard feels like a read-only report rather than a command center
- Duplicates the job of Control Tower without the operational utility
- Planners spend extra navigation time going from alert to the relevant analysis

#### Recommended Fix

Each alert should include a `source_tab` and optionally pre-populated filter params:
```typescript
interface Alert {
  severity: "critical" | "warning" | "info";
  message: string;
  source_tab: string;       // e.g. "controlTower"
  filter_params?: Record<string, string>;  // e.g. { item_no: "100320" }
}
```

Alert cards become clickable — click navigates to the source tab with filters pre-applied.

---

### PL-014 — Control Tower Trend Chart Window is Hardcoded to 6 Months

**Status:** Fixed — 2026-03-06
**Priority:** P2 — Planners can't adjust the time horizon
**Date captured:** 2026-03-06
**Affects:** `frontend/src/tabs/ControlTowerTab.tsx`, `api/routers/control_tower.py`

#### Problem

The Control Tower trend chart always shows the last 6 months. Planners reviewing annual trends (seasonal businesses, year-over-year comparisons) need at least 12–18 months. Planners doing week-over-week triage need 4–6 weeks. There is no control to change the window.

#### Recommended Fix

Add a compact window selector above the trend chart:
- `4W | 3M | 6M | 12M | YTD` toggle pills (default: 6M)
- Pass selected window as `months` query param to `GET /control-tower/trend?months=N`

---

### PL-015 — Accuracy Tab Champion Panel Requires Two Clicks to Save and Run

**Status:** Fixed — 2026-03-06
**Priority:** P3 — Minor workflow friction
**Date captured:** 2026-03-06
**Affects:** `frontend/src/tabs/AccuracyTab.tsx`

#### Problem

The Champion panel has separate "Save" and "Run" buttons. The intended workflow is always: change config → save → run. There is no scenario where a planner saves config changes but does not want to immediately run the selection. This creates unnecessary extra clicks.

#### Recommended Fix

Replace with a single **"Save & Run"** button. If there are cases where saving without running is valid, add a `⋮` overflow menu with a "Save only" option as the secondary action.

---

### PL-016 — Inventory Position Table Has No Color-Coding for Threshold Breaches

**Status:** Fixed — 2026-03-06
**Priority:** P3 — Planners must scan every row to find problem items
**Date captured:** 2026-03-06
**Affects:** `frontend/src/tabs/InventoryTab.tsx`

#### Problem

The Inventory Position table shows DOS, WOC, and on-hand values as plain numbers. There are no visual indicators for items that are near stockout (DOS < safety stock threshold) or in excess (DOS > maximum coverage target). Planners must mentally evaluate each number.

#### Impact

- For tables with hundreds of items, finding at-risk items requires scrolling or manual sorting
- The AI Planner and Control Tower already identify at-risk items — the Inventory tab doesn't reinforce this

#### Recommended Fix

Apply row-level background tinting based on DOS thresholds:
- `dos < 7 days` → red-50 background (critical stockout risk)
- `7 ≤ dos < 14 days` → amber-50 background (monitor)
- `dos > 180 days` → blue-50 background (excess)

The thresholds should be configurable or pulled from replenishment policy data.

---

### PL-017 — AI Planner Planning Memo Shows Only the Latest Entry

**Status:** Fixed — 2026-03-06
**Priority:** P3 — Historical context is lost
**Date captured:** 2026-03-06
**Affects:** `frontend/src/tabs/AIPlannerTab.tsx`, `api/routers/ai_planner.py`

#### Problem

The Planning Memo panel displays only the single most recent portfolio memo. Planning memos are narratives generated after each portfolio scan, summarizing the week's inventory health, risk trends, and recommended focus areas. Planners reviewing a week-over-week comparison need to see the previous memo alongside the current one.

There is no "Previous memos" history, no date navigation, and no way to see how the narrative changed over time.

#### Recommended Fix

1. Add a `<` `>` navigation chevron to cycle through the last 5 memos with the date shown
2. Or add a "View Memo History" expandable section below the current memo showing the last 4 memos in collapsed form
3. The `GET /ai-planner/memos` endpoint already supports `limit` and `scope` params — the data is available, the UI just doesn't surface it

---

### PL-018 — No Unified "My Work Queue" View for a Planner's Daily Triage

**Status:** Open
**Priority:** P2 — Core planner workflow is not supported end-to-end
**Date captured:** 2026-03-06
**Affects:** Overall application architecture

#### Problem

A supply chain planner starting their day needs to answer: *"What do I need to act on today?"* The answer is currently spread across:

- **AI Planner** tab → AI-generated insights (severity-ranked)
- **Storyboard** tab → Rule-based policy exceptions
- **Inv. Planning** tab → Exception queue (replenishment exceptions)
- **Control Tower** tab → High-level KPI alerts

There is no single consolidated view. The planner must visit 3–4 tabs and mentally merge the queues, many of which may refer to the same item.

#### Impact

- Planners with 100+ open items across tabs face a fragmented morning triage
- The same item (e.g., item 100320 at LOC 1401-BULK) may appear in all 4 tabs with no deduplication
- High-value AI Planner insights are invisible to planners who don't check that tab

#### Recommended Fix

Long-term: Add a **"My Queue"** or **"Today's Actions"** tab that aggregates:
- Open AI Insights (critical + high severity)
- Open Storyboard Exceptions (critical + high)
- Open Replenishment Exceptions (critical)
- De-duplicated by item_no + loc with a unified priority score

Short-term: Make the Control Tower the true landing page by wiring its alert strips and top-critical list to navigate to the relevant detail tabs (see PL-010).

---

*Add new issues below using the PL-NNN format.*
