# 03-11 — Forward-Looking Replenishment Plan

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P1 — Must Have (Forward Planning Foundation)

## Effort
L (Large)

## Expert Perspectives
- **Inventory Planning Expert** (lead) — forward SS formula, policy parameter derivation, ROP/EOQ integration
- **Demand Planning Expert** — CI band interpretation, forecast signal quality, horizon scaling
- **Statistical Analyst** — combined variance propagation, CI-to-sigma conversion, guard rail design
- **UI/UX Expert** — plan version selector, comparison chart, DFU drill-down design

---

## Problem Statement

The existing inventory planning stack (IPfeature3–IPfeature7) computes safety stock and replenishment parameters from **historical demand statistics** — backward-looking averages and standard deviations pulled from `fact_safety_stock_targets`. When the ML production forecast (F1.1) is available, planners have a richer forward signal: not only a point forecast of expected demand, but **confidence interval bands** that encode model uncertainty horizon-by-horizon.

Without a forward-looking replenishment plan:
- Safety stock is anchored to history even when the forecast shows a step-change in demand
- EOQ is computed from trailing annual demand rather than the 12-month forward volume
- Policy parameters (ROP, order-up-to) have no link to the production forecast
- Planners cannot see how next quarter's inventory requirements differ from today's historical targets

This feature closes that gap by computing per-DFU per-month replenishment parameters driven by the production forecast CI bands, and comparing them against the historical baseline from `fact_safety_stock_targets`.

---

## User Story

> As an inventory planner, I want forward-looking safety stock and replenishment parameters computed from the ML production forecast confidence interval bands for each SKU-location across a 12-month horizon, compared against historical baselines, so I can see where future demand changes require me to adjust my current inventory position before a shortage or excess materialises.

---

## Business Value

- Converts the production forecast (F1.1) from a reporting artefact into an **actionable replenishment signal**
- Surfaces items where forward SS diverges materially from historical SS — the highest-priority items for planner review
- Closes the loop between ML forecasting (02-*) and inventory planning (03-*) into a single cohesive forward plan
- Enables proactive order placement ahead of demand ramps, not reactive ordering after a stockout
- Feeds the AI Planning Agent (05-01) with a pre-computed forward plan for its causal chain analysis

---

## Key Formulas

### Safety Stock (Forward-Looking)

The core insight is that the production forecast CI bands encode month-specific demand uncertainty. For a 80% CI (P10–P90), the implied σ per month is:

```
# Step 1: Extract sigma from CI bands (80% CI assumed)
# P10 corresponds to Z = -1.282, P90 to Z = +1.282
sigma_demand_monthly = (forecast_qty_upper - forecast_qty_lower) / (2 × 1.282)

# Fallback when CI bands are NULL (CI not yet computed or model not available):
sigma_demand_monthly = dim_dfu.demand_std   (historical monthly std dev)

# Step 2: Derive average daily demand from point forecast
avg_daily_demand = forecast_qty / 30.44

# Step 3: Lead time statistics (from fact_lead_time_profile if available, else dim_dfu)
avg_lt_days  = COALESCE(lt_profile.avg_lead_time_days, dim_dfu.avg_lt_days, config.default_lt_days)
lt_std_days  = COALESCE(lt_profile.lt_std_days, dim_dfu.lt_std_days, config.default_lt_std_days)

# Step 4: Z-score from policy service level (via fact_dfu_policy_assignment → dim_replenishment_policy)
# Fallback: config.default_service_level → Z lookup table
z = z_score_for_service_level(policy.service_level)

# Step 5: Combined safety stock components
# SS_demand: uncertainty in demand over lead time (monthly sigma scaled to LT days)
SS_demand = z × sigma_demand_monthly × sqrt(avg_lt_days / 30.44)

# SS_lt: uncertainty in lead time itself (avg daily demand × LT variability)
SS_lt = avg_daily_demand × lt_std_days × z

# Combined (independent sources of variance):
SS_combined = sqrt(SS_demand² + SS_lt²)

# Step 6: Guard rails (configured in replenishment_plan_config.yaml)
min_ss_days = 3   → SS_combined_qty = max(SS_combined, avg_daily_demand × 3)
max_ss_days = 120 → SS_combined_qty = min(SS_combined_qty, avg_daily_demand × 120)
```

### EOQ (Forward-Looking)

```
# Annual demand from the 12-month forward forecast
D_annual = SUM(forecast_qty for next 12 months for this DFU)

# Pro-rate if fewer than 12 forecast months are available:
D_annual = SUM(available_months_forecast_qty) × (12 / count_of_available_months)

# Wilson EOQ formula (same parameters as IPfeature4, from eoq_config.yaml)
EOQ = sqrt(2 × D_annual × ordering_cost / (unit_cost × holding_cost_pct))

# Apply the same constraints as IPfeature4:
effective_EOQ = max(MOQ, EOQ)
effective_EOQ = min(effective_EOQ, D_annual × max_eoq_months_supply / 12)

# Cycle stock
cycle_stock = effective_EOQ / 2
```

### Policy-Specific Parameters

These are derived per DFU from `fact_dfu_policy_assignment` → `dim_replenishment_policy`, using the forward SS_combined and effective_EOQ computed above.

| Policy Type | Reorder Point (ROP) | Order Quantity | Order-Up-To Level |
|---|---|---|---|
| `continuous_rop` | `avg_daily_demand × avg_lt_days + SS_combined` | `effective_EOQ` | — |
| `min_max` | `avg_daily_demand × avg_lt_days + SS_combined` (= min level s) | — | `s + effective_EOQ` (= max level S) |
| `periodic_review` | — | — | `avg_daily_demand × (review_cycle_days + avg_lt_days) + SS_combined` |
| `manual` / `JIT` | — | — | — (`is_jit = TRUE`) |

DFUs with no policy assignment use a `continuous_rop` fallback with a default service level from config.

---

## Data Grain

`fact_replenishment_plan`: one row per **(plan_version, item_no, loc, plan_month)**

- **plan_version**: matches `fact_production_forecast.plan_version` (e.g. `"2026-03"`)
- **plan_month**: one of the 12 forward months in the production forecast horizon
- **horizon_months**: integer 1–12, counting from the plan run date

The table is a rolling 12-month forward plan. Each new production forecast run creates a new `plan_version` and inserts 12 rows per DFU. Old plan versions are retained for comparison (subject to the same retention policy as `fact_production_forecast`).

---

## Schema

### New Table: `fact_replenishment_plan`

```sql
CREATE TABLE IF NOT EXISTS fact_replenishment_plan (
    -- Keys
    plan_sk             BIGSERIAL PRIMARY KEY,
    plan_version        TEXT NOT NULL,
    item_no             TEXT NOT NULL,
    loc                 TEXT NOT NULL,
    plan_month          DATE NOT NULL,   -- first day of the forward month
    horizon_months      SMALLINT NOT NULL CHECK (horizon_months BETWEEN 1 AND 36),
    UNIQUE (plan_version, item_no, loc, plan_month),

    -- Policy context
    policy_id           TEXT,           -- from fact_dfu_policy_assignment (NULL = no assignment)
    policy_type         TEXT,           -- continuous_rop | periodic_review | min_max | manual
    review_cycle_days   INTEGER,        -- for periodic_review policy
    service_level       NUMERIC(6,4),   -- z-score source service level
    z_score             NUMERIC(8,4),   -- derived from service_level
    abc_vol             TEXT,           -- from dim_dfu.abc_vol
    xyz_class           TEXT,           -- from dim_dfu.xyz_class

    -- Demand inputs (from fact_production_forecast)
    forecast_qty            NUMERIC(18,4),  -- point forecast for plan_month
    forecast_qty_lower      NUMERIC(18,4),  -- P10 CI lower bound (NULL if unavailable)
    forecast_qty_upper      NUMERIC(18,4),  -- P90 CI upper bound (NULL if unavailable)
    forecast_annual_demand  NUMERIC(18,4),  -- sum of 12-month forecast (or pro-rated)
    ci_source               TEXT,           -- 'forecast_ci' | 'historical_fallback'

    -- Demand variability
    sigma_demand_monthly    NUMERIC(18,4),  -- derived from CI bands or dim_dfu.demand_std
    avg_daily_demand        NUMERIC(18,6),  -- forecast_qty / 30.44

    -- Lead time inputs
    avg_lt_days             NUMERIC(10,2),  -- from lead time profile or dim_dfu
    lt_std_days             NUMERIC(10,2),  -- lead time standard deviation in days
    lt_source               TEXT,           -- 'lt_profile' | 'dfu_attribute' | 'config_default'

    -- Safety stock outputs
    ss_demand               NUMERIC(18,4),  -- demand-driven SS component (units)
    ss_lt                   NUMERIC(18,4),  -- lead-time-driven SS component (units)
    ss_combined             NUMERIC(18,4),  -- sqrt(ss_demand² + ss_lt²), guard-railed
    ss_days                 NUMERIC(10,2),  -- ss_combined / avg_daily_demand

    -- Cycle stock outputs
    eoq                     NUMERIC(18,4),  -- Wilson formula result (unconstrained)
    effective_eoq           NUMERIC(18,4),  -- after MOQ floor and months-supply cap
    cycle_stock             NUMERIC(18,4),  -- effective_eoq / 2
    moq                     NUMERIC(18,4),  -- minimum order quantity applied

    -- Policy parameters
    reorder_point           NUMERIC(18,4),  -- avg_daily * lt + SS (continuous_rop, min_max)
    order_qty               NUMERIC(18,4),  -- effective_eoq for continuous_rop; NULL otherwise
    order_up_to_level       NUMERIC(18,4),  -- S for min_max; OUL for periodic_review
    is_jit                  BOOLEAN DEFAULT FALSE,  -- TRUE for manual/JIT policies

    -- Comparison vs historical (from fact_safety_stock_targets.policy_version='v1')
    historical_ss           NUMERIC(18,4),  -- ss_combined from fact_safety_stock_targets
    historical_eoq          NUMERIC(18,4),  -- effective_eoq from fact_safety_stock_targets
    ss_delta                NUMERIC(18,4),  -- ss_combined - historical_ss (positive = forward higher)
    ss_delta_pct            NUMERIC(10,4),  -- ss_delta / NULLIF(historical_ss, 0)
    eoq_delta               NUMERIC(18,4),  -- effective_eoq - historical_eoq

    -- Current inventory position (from agg_inventory_monthly, latest month)
    current_qty_on_hand     NUMERIC(18,4),  -- eom_qty_on_hand at plan run date
    ss_gap                  NUMERIC(18,4),  -- ss_combined - current_qty_on_hand
    is_below_ss             BOOLEAN,        -- current_qty_on_hand < ss_combined

    -- Audit
    computed_at             TIMESTAMPTZ DEFAULT NOW(),
    forecast_model_id       TEXT            -- model_id from fact_production_forecast
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_rep_plan_version_item_loc
    ON fact_replenishment_plan (plan_version, item_no, loc);
CREATE INDEX IF NOT EXISTS idx_rep_plan_month
    ON fact_replenishment_plan (plan_version, plan_month);
CREATE INDEX IF NOT EXISTS idx_rep_plan_below_ss
    ON fact_replenishment_plan (plan_version, is_below_ss)
    WHERE is_below_ss = TRUE;
CREATE INDEX IF NOT EXISTS idx_rep_plan_abc
    ON fact_replenishment_plan (plan_version, abc_vol, plan_month);
CREATE INDEX IF NOT EXISTS idx_rep_plan_item_loc_month
    ON fact_replenishment_plan (item_no, loc, plan_month);
```

**DDL file:** `mvp/demand/sql/041_create_replenishment_plan.sql`

---

## Config

### New Config: `mvp/demand/config/replenishment_plan_config.yaml`

```yaml
replenishment_plan:
  # Horizon
  forward_months: 12          # number of forward months to plan

  # CI band interpretation
  ci_confidence_level: 0.80   # 80% CI → P10/P90; Z = 1.282 for each tail
  ci_z_score: 1.282           # derived from ci_confidence_level

  # Lead time fallbacks (used when fact_lead_time_profile has no row for item+loc)
  default_lt_days: 21         # 3-week lead time default
  default_lt_std_days: 3      # 3-day std dev default

  # Service level fallback (used when DFU has no policy assignment)
  default_service_level: 0.95
  # Z-table (same as safety_stock_config.yaml)
  z_table:
    0.80: 0.842
    0.85: 1.036
    0.90: 1.282
    0.95: 1.645
    0.98: 2.054
    0.99: 2.326

  # Guard rails
  min_ss_days: 3              # never less than 3 days of demand
  max_ss_days: 120            # never more than 120 days (4 months)

  # EOQ parameters (mirrors eoq_config.yaml; override here for forward plan)
  ordering_cost: 50.00
  holding_cost_pct: 0.25
  moq_source: config          # 'config' | 'dim_item'
  default_moq: 1
  max_eoq_months_supply: 6

  # Retention: keep N plan versions in DB before purging
  max_plan_versions: 6

  # Comparison threshold for flagging material SS changes
  ss_delta_pct_flag_threshold: 0.20   # flag if forward SS differs from historical by ≥20%
```

---

## Backend Script

### `mvp/demand/scripts/compute_replenishment_plan.py`

```python
# Algorithm:
#
# 1. Load config from config/replenishment_plan_config.yaml
#
# 2. Identify the latest plan_version from fact_production_forecast
#    (or accept --plan-version CLI arg to recompute a specific version)
#
# 3. Load production forecast rows for the plan_version:
#    SELECT item_no, loc, plan_version, month_start AS plan_month,
#           forecast_qty, forecast_qty_lower, forecast_qty_upper,
#           ROW_NUMBER() OVER (PARTITION BY item_no, loc ORDER BY month_start) AS horizon_months
#    FROM fact_production_forecast
#    WHERE plan_version = %s
#    ORDER BY item_no, loc, month_start
#
# 4. For each (item_no, loc), aggregate forecast_annual_demand:
#    SUM(forecast_qty) — pro-rate if row count < 12
#
# 5. Load policy assignments:
#    SELECT fpa.item_no, fpa.loc, drp.policy_id, drp.policy_type,
#           drp.service_level, drp.review_cycle_days
#    FROM fact_dfu_policy_assignment fpa
#    JOIN dim_replenishment_policy drp USING (policy_id)
#
# 6. Load lead time profiles:
#    SELECT item_no, loc, avg_lead_time_days, lt_std_days
#    FROM fact_lead_time_profile   -- if exists; else use dim_dfu columns
#
# 7. Load historical SS baseline:
#    SELECT item_no, loc, ss_combined AS historical_ss, effective_eoq AS historical_eoq
#    FROM fact_safety_stock_targets WHERE policy_version = 'v1'
#
# 8. Load current inventory position:
#    SELECT DISTINCT ON (item_no, loc) item_no, loc, eom_qty_on_hand
#    FROM agg_inventory_monthly ORDER BY item_no, loc, month_start DESC
#
# 9. Load DFU attributes:
#    SELECT dmdunit AS item_no, loc, abc_vol, xyz_class, demand_std, avg_lt_days, lt_std_days
#    FROM dim_dfu
#
# 10. For each forecast row (plan_version, item_no, loc, plan_month):
#     a. Determine service level Z-score:
#        policy = policy_assignments.get((item_no, loc))
#        z = z_table[policy.service_level] if policy else z_table[default_service_level]
#
#     b. Compute sigma_demand_monthly:
#        if forecast_qty_upper is not None and forecast_qty_lower is not None:
#            sigma = (forecast_qty_upper - forecast_qty_lower) / (2 * cfg.ci_z_score)
#            ci_source = 'forecast_ci'
#        else:
#            sigma = dfu_attr.demand_std  (monthly historical std dev)
#            ci_source = 'historical_fallback'
#
#     c. avg_daily_demand = forecast_qty / 30.44
#
#     d. Lead time:
#        lt_row = lt_profiles.get((item_no, loc))
#        avg_lt = lt_row.avg_lead_time_days if lt_row else (dfu_attr.avg_lt_days or cfg.default_lt_days)
#        lt_std = lt_row.lt_std_days if lt_row else (dfu_attr.lt_std_days or cfg.default_lt_std_days)
#        lt_source = 'lt_profile' | 'dfu_attribute' | 'config_default'
#
#     e. SS components:
#        ss_demand = z * sigma * math.sqrt(avg_lt / 30.44)
#        ss_lt     = avg_daily_demand * lt_std * z
#        ss_raw    = math.sqrt(ss_demand**2 + ss_lt**2)
#
#     f. Guard rails:
#        ss_min = avg_daily_demand * cfg.min_ss_days
#        ss_max = avg_daily_demand * cfg.max_ss_days
#        ss_combined = max(ss_min, min(ss_raw, ss_max))
#        ss_days = ss_combined / avg_daily_demand if avg_daily_demand > 0 else None
#
#     g. EOQ (once per DFU using forecast_annual_demand, applied to all 12 plan months):
#        D_annual = forecast_annual_demand (computed in step 4)
#        eoq = math.sqrt(2 * D_annual * cfg.ordering_cost / (unit_cost * cfg.holding_cost_pct))
#        effective_eoq = max(cfg.moq, eoq)
#        effective_eoq = min(effective_eoq, D_annual * cfg.max_eoq_months_supply / 12)
#        cycle_stock = effective_eoq / 2
#
#     h. Policy parameters (per policy_type):
#        continuous_rop:  rop = avg_daily * avg_lt + ss_combined
#                         order_qty = effective_eoq
#        min_max:         rop = avg_daily * avg_lt + ss_combined   (= s)
#                         order_up_to = rop + effective_eoq         (= S)
#        periodic_review: order_up_to = avg_daily * (review_cycle_days + avg_lt) + ss_combined
#        manual/JIT:      is_jit = True; all params = None
#        fallback (no policy): treat as continuous_rop with default service level
#
#     i. Comparison:
#        hist = historical_ss.get((item_no, loc))
#        historical_ss_val = hist.historical_ss if hist else None
#        ss_delta = ss_combined - historical_ss_val if historical_ss_val else None
#        ss_delta_pct = ss_delta / historical_ss_val if historical_ss_val else None
#
#     j. Current position:
#        inv = current_inv.get((item_no, loc))
#        current_qty = inv.eom_qty_on_hand if inv else None
#        ss_gap = ss_combined - current_qty if current_qty is not None else None
#        is_below_ss = current_qty < ss_combined if current_qty is not None else None
#
# 11. Batch upsert to fact_replenishment_plan
#     ON CONFLICT (plan_version, item_no, loc, plan_month) DO UPDATE SET ...
#
# 12. Purge old plan versions: keep only the most recent max_plan_versions versions
#
# 13. Print summary: DFU count, plan_version, rows written, below_ss_count, fallback_ci_count
```

**CLI Usage:**
```bash
# Full portfolio for latest plan_version
uv run python scripts/compute_replenishment_plan.py

# Specific plan version
uv run python scripts/compute_replenishment_plan.py --plan-version 2026-03

# Single DFU
uv run python scripts/compute_replenishment_plan.py --item 100320 --loc 1401-BULK

# Preview without DB write
uv run python scripts/compute_replenishment_plan.py --dry-run

# Custom config
uv run python scripts/compute_replenishment_plan.py --config config/replenishment_plan_config.yaml
```

---

## API Endpoints

**New router:** `mvp/demand/api/routers/inv_planning_replenishment_plan.py`
**Mount prefix:** `/inv-planning/replenishment`
**All endpoints use `get_conn()` directly (not `Depends(_get_pool)`) — consistent with all other `inv_planning_*.py` routers.**

```
GET /inv-planning/replenishment/summary
  Query params:
    plan_version (default: latest)
    abc_vol (A | B | C)
    policy_type (continuous_rop | min_max | periodic_review | manual)
    horizon_months (1–12; if omitted, aggregates all months equally)
  Response: {
    plan_version: str,
    plan_run_date: str (ISO date),
    total_dfus: int,
    below_ss_count: int,
    below_ss_pct: float,
    avg_forward_ss: float,         -- avg ss_combined across all DFU-months
    avg_historical_ss: float,
    avg_ss_delta_pct: float,       -- avg of abs(ss_delta_pct)
    avg_forward_eoq: float,
    total_forward_demand: float,   -- sum of forecast_annual_demand (unique per DFU)
    by_policy_type: {
      continuous_rop: { count, avg_ss, avg_eoq, below_ss_count },
      min_max:        { count, avg_ss, avg_eoq, below_ss_count },
      periodic_review:{ count, avg_ss, below_ss_count },
      manual:         { count, below_ss_count }
    },
    by_abc: {
      A: { count, avg_ss, avg_eoq, below_ss_count, avg_ss_delta_pct },
      B: { ... },
      C: { ... }
    },
    ci_coverage_pct: float         -- % of DFU-months where CI was available (not fallback)
  }
  Cache: max-age=120s

GET /inv-planning/replenishment/detail
  Query params:
    plan_version (default: latest)
    plan_month (ISO date, e.g. 2026-04-01; omit for first horizon month)
    item, location
    abc_vol, policy_type, xyz_class
    is_below_ss (true | false)
    limit (default: 50, max: 500)
    offset (default: 0)
    sort_by (ss_combined | ss_delta_pct | effective_eoq | reorder_point | ss_gap | horizon_months)
    sort_dir (asc | desc; default: desc for ss_delta_pct)
  Response: {
    total: int,
    plan_version: str,
    rows: [
      {
        item_no, loc, plan_month, horizon_months,
        policy_type, abc_vol, xyz_class, service_level,
        forecast_qty, forecast_qty_lower, forecast_qty_upper,
        sigma_demand_monthly, avg_daily_demand, ci_source,
        avg_lt_days, lt_std_days, lt_source,
        ss_demand, ss_lt, ss_combined, ss_days,
        eoq, effective_eoq, cycle_stock, moq,
        reorder_point, order_qty, order_up_to_level, is_jit,
        historical_ss, ss_delta, ss_delta_pct,
        current_qty_on_hand, ss_gap, is_below_ss
      }
    ]
  }
  Cache: max-age=60s

GET /inv-planning/replenishment/comparison
  Query params:
    plan_version (default: latest)
    abc_vol, policy_type
    ss_delta_pct_min (float, e.g. 0.10 — only show DFUs where |ss_delta_pct| >= threshold)
  Response: {
    plan_version: str,
    total_dfus: int,
    increased_count: int,         -- forward SS > historical SS
    decreased_count: int,         -- forward SS < historical SS
    unchanged_count: int,         -- within ±5% band
    by_abc: [
      {
        abc_vol: str,
        avg_historical_ss: float,
        avg_forward_ss: float,
        avg_ss_delta: float,
        avg_ss_delta_pct: float,
        count: int
      }
    ],
    top_increases: [
      { item_no, loc, historical_ss, ss_combined, ss_delta, ss_delta_pct }
      -- top 20 DFUs where forward SS most exceeds historical SS
    ],
    top_decreases: [
      { item_no, loc, historical_ss, ss_combined, ss_delta, ss_delta_pct }
      -- top 20 DFUs where forward SS most undercuts historical SS
    ]
  }
  Cache: max-age=120s

GET /inv-planning/replenishment/dfu
  Query params:
    item (required)
    location (required)
    plan_version (default: latest)
  Response: {
    item_no: str,
    loc: str,
    plan_version: str,
    policy_type: str,
    abc_vol: str,
    historical_ss: float | null,
    historical_eoq: float | null,
    current_qty_on_hand: float | null,
    months: [
      {
        plan_month, horizon_months,
        forecast_qty, forecast_qty_lower, forecast_qty_upper,
        ss_combined, ss_days,
        reorder_point, order_up_to_level, order_qty,
        is_below_ss, ss_gap
      }
    ]
  }
  Cache: max-age=60s
```

---

## Frontend Panel

### Panel: `ReplenishmentPlanPanel.tsx`

**Location:** `mvp/demand/frontend/src/tabs/inv-planning/ReplenishmentPlanPanel.tsx`
**Tab:** Added as a new sub-tab labelled "Replen. Plan" in `InvPlanningTab.tsx`

#### Control Bar (top)
- **Plan Version selector** — dropdown populated from `/forecast/production/versions`; defaults to latest
- **Horizon filter** — "Month 1" through "Month 12" chip selector, or "All" (aggregated)
- **ABC class filter** — A / B / C / All toggle pills
- **Policy type filter** — dropdown (All | Continuous ROP | Min-Max | Periodic Review | Manual)

#### KPI Cards (row of 4)

| Card | Value | Source |
|---|---|---|
| Total DFUs | `total_dfus` formatted with commas | summary endpoint |
| Avg Forward SS | `avg_forward_ss` in units (2 dp) | summary endpoint |
| Avg Forward EOQ | `avg_forward_eoq` in units (2 dp) | summary endpoint |
| Below SS Count | `below_ss_count` with severity color (red if > 0) | summary endpoint |

#### CI Coverage Badge (inline with KPI row)
- Small indicator: "Forecast CI: 84% coverage" — shows what fraction of DFU-months used CI bands vs historical fallback
- Color: green ≥ 80%, yellow 50–79%, red < 50%

#### Comparison Section: "Forward SS vs Historical SS"

**Bar Chart (Recharts `BarChart`):**
- X-axis: ABC class (A, B, C)
- Two bars per group: `avg_historical_ss` (gray) vs `avg_forward_ss` (indigo)
- Y-axis: safety stock quantity
- Tooltip: avg_historical_ss, avg_forward_ss, avg_ss_delta_pct (formatted as %)
- Title: "Forward Safety Stock vs Historical Baseline by ABC Class"
- Data source: `comparison` endpoint `by_abc` array

**Top Changes Table (below chart):**
- Toggle: "Largest Increases" / "Largest Decreases" button pair
- Columns: Item, Location, Historical SS, Forward SS, Delta, Delta % (color-coded: red for increases >20%, green for decreases >20%)
- Shows top 20 rows from comparison endpoint

#### Detail Table

**Virtualized table (TanStack Table + useVirtualizer):**
- Default sort: `ss_delta_pct` descending (items most changed from historical at top)
- Columns: Item No, Location, Plan Month, Policy Type badge, ABC, Forecast Qty, Lower CI, Upper CI, Sigma, Forward SS, Hist. SS, Delta %, EOQ, ROP / OUL, Below SS (red chip)
- Row highlight: red-50 background when `is_below_ss = true`
- Pagination: server-side (offset/limit)
- Item/Location text filters wired to `item` and `location` query params
- Below SS toggle: "Show Below SS Only" checkbox

#### DFU Drill-Down (drawer or expandable row)

**Trigger:** Click any row in the detail table → opens a side drawer or expands inline

**Line Chart (Recharts `LineChart`):**
- X-axis: plan_month (formatted as "Apr 26", "May 26", …)
- Lines:
  - `forecast_qty` (blue solid) — point forecast
  - `forecast_qty_upper` / `forecast_qty_lower` (blue dashed, shaded area) — CI band
  - `ss_combined` (amber solid) — forward safety stock
  - `reorder_point` (red dashed) — for continuous_rop and min_max policies
  - `historical_ss` (gray horizontal reference line) — constant baseline
- Right Y-axis: `ss_combined` and `reorder_point` in same units
- Tooltip: all five values for hovered month

**DFU summary header:** item_no, loc, policy_type badge, abc_vol badge, current_qty_on_hand vs current ss_gap

---

## Makefile Targets

```makefile
replen-plan-schema:
	# Apply mvp/demand/sql/041_create_replenishment_plan.sql
	uv run python -c "
	import psycopg; from common.db import get_db_params
	conn = psycopg.connect(**get_db_params())
	conn.autocommit = True
	conn.execute(open('sql/041_create_replenishment_plan.sql').read())
	conn.close()
	print('replenishment plan schema applied')
	"

replen-plan-compute:
	uv run python scripts/compute_replenishment_plan.py

replen-plan-dfu:
	# Single DFU: make replen-plan-dfu ITEM=100320 LOC=1401-BULK
	uv run python scripts/compute_replenishment_plan.py --item $(ITEM) --loc $(LOC)

replen-plan-dry:
	uv run python scripts/compute_replenishment_plan.py --dry-run

replen-plan-all: replen-plan-schema replen-plan-compute
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `fact_production_forecast` | F1.1 | Source of `forecast_qty`, CI bands, `plan_version` |
| `fact_dfu_policy_assignment` | IPfeature5 | Policy type + service level per DFU |
| `dim_replenishment_policy` | IPfeature5 | Policy parameters (review_cycle_days, service_level) |
| `fact_safety_stock_targets` | IPfeature3 | Historical SS baseline for comparison |
| `agg_inventory_monthly` | Existing | Current on-hand position for `ss_gap`, `is_below_ss` |
| `dim_dfu` | Existing | `demand_std`, `avg_lt_days`, `abc_vol`, `xyz_class` fallback attributes |
| `fact_lead_time_profile` | IPfeature3 / 03-02 | Preferred source for `avg_lt_days`, `lt_std_days` |
| `config/eoq_config.yaml` | IPfeature4 | EOQ parameters (ordering_cost, holding_cost_pct, MOQ) |
| `config/replenishment_plan_config.yaml` | New | CI interpretation, guard rails, retention, Z-table |

**Pipeline ordering:** `forecast-prod-all` → `replen-plan-all` (replenishment plan must run after production forecast).

---

## Testing Requirements

### Backend Unit Tests: `mvp/demand/tests/unit/test_replenishment_plan.py`

Minimum 25 tests:

**Sigma extraction from CI bands:**
- P10=80, P90=120, ci_z=1.282 → sigma = (120-80)/(2×1.282) ≈ 15.6
- P10=P90=100 → sigma = 0 (degenerate CI, no uncertainty)
- NULL CI bands → fallback to demand_std from dim_dfu

**Forward SS formula:**
- avg_daily=10, avg_lt=21, sigma=15.6, z=1.645 → SS_demand = 1.645 × 15.6 × sqrt(21/30.44) ≈ 20.8
- avg_daily=10, lt_std=3, z=1.645 → SS_lt = 10 × 3 × 1.645 = 49.4
- SS_combined = sqrt(SS_demand² + SS_lt²): verified numerically
- Guard rail: avg_daily=5, min_ss_days=3 → ss_combined ≥ 15 always
- Guard rail: avg_daily=5, max_ss_days=120 → ss_combined ≤ 600 always

**Forward EOQ:**
- D_annual=1200, ordering_cost=50, unit_cost=10, holding_cost_pct=0.25 → EOQ ≈ 219.1
- Pro-rate: 9 months available, sum=900 → D_annual = 900 × (12/9) = 1200
- MOQ floor: EOQ=50, MOQ=100 → effective_EOQ=100
- Months-supply cap: EOQ=800, D_annual/12=100, max_months=6 → effective_EOQ=600

**Policy parameters:**
- continuous_rop: rop = 10 × 21 + 50 = 260; order_qty = effective_eoq
- min_max: rop = 260; order_up_to = 260 + effective_eoq
- periodic_review: OUL = 10 × (28 + 21) + 50 = 540
- manual: all None; is_jit = True

**Comparison logic:**
- historical_ss=100, ss_combined=130 → ss_delta=30, ss_delta_pct=0.30
- historical_ss=None → ss_delta=None, ss_delta_pct=None

**Current position:**
- current_qty=80, ss_combined=100 → ss_gap=20, is_below_ss=True
- current_qty=150, ss_combined=100 → ss_gap=-50, is_below_ss=False

**Edge cases:**
- avg_daily_demand=0: SS_demand=0, ss_days=None (no division by zero)
- forecast_qty=0: D_annual=0, effective_EOQ=MOQ (no demand → order minimum)
- unit_cost=0: fallback to config default unit_cost

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_replenishment_plan.py`

Minimum 15 tests:

- `GET /inv-planning/replenishment/summary` → 200 OK, has `by_policy_type` and `by_abc`
- `GET /inv-planning/replenishment/summary?plan_version=2026-03` → filters to version
- `GET /inv-planning/replenishment/summary?abc_vol=A` → by_abc contains only A key
- `GET /inv-planning/replenishment/detail` → 200 OK, rows have `ss_combined` and `effective_eoq`
- `GET /inv-planning/replenishment/detail?is_below_ss=true` → all rows have `is_below_ss=true`
- `GET /inv-planning/replenishment/detail?sort_by=ss_delta_pct&sort_dir=desc` → first row has highest delta
- Pagination: `limit=2&offset=0` → 2 rows; `limit=2&offset=2` → next 2
- `GET /inv-planning/replenishment/comparison` → 200 OK, has `by_abc` array and `top_increases`
- `GET /inv-planning/replenishment/comparison?ss_delta_pct_min=0.20` → only items with |delta| ≥ 20%
- `GET /inv-planning/replenishment/dfu?item=X&location=Y` → 200 OK, `months` has up to 12 entries
- `GET /inv-planning/replenishment/dfu` without item → 422 Unprocessable Entity
- `GET /inv-planning/replenishment/dfu?item=X&location=Y` with no data → 200 OK, `months=[]`
- Summary with empty table → returns zeros, not 500
- `GET /inv-planning/replenishment/detail?policy_type=manual` → all rows `is_jit=true`
- `ci_coverage_pct` in summary is between 0.0 and 1.0

### Frontend Tests: `mvp/demand/frontend/src/tabs/inv-planning/__tests__/ReplenishmentPlanPanel.test.tsx`

Minimum 8 tests:

- Panel renders 4 KPI cards with correct labels
- CI coverage badge renders with correct percentage text
- Comparison bar chart renders with ABC class groups
- Detail table renders columns including "Forward SS" and "Delta %"
- "Below SS Only" checkbox toggles `is_below_ss=true` filter param
- DFU drill-down: clicking a row shows a chart with `plan_month` on X-axis
- "Largest Increases" / "Largest Decreases" toggle changes table content
- Renders without error when API returns empty data (zero DFUs)

---

## Success Criteria

- [ ] `fact_replenishment_plan` populated for all DFUs that appear in `fact_production_forecast` after `make replen-plan-all`
- [ ] Forward SS formula verified: CI-derived sigma used when CI bands are present; historical `demand_std` used as fallback; guard rails enforced in all cases
- [ ] EOQ computed from forward `forecast_annual_demand`, not trailing historical demand
- [ ] All 4 policy types produce correct parameters: ROP for continuous_rop and min_max, order-up-to for periodic_review and min_max, `is_jit=TRUE` for manual
- [ ] `ss_delta_pct` correctly reflects the proportional change from `fact_safety_stock_targets` for every DFU that has a historical SS baseline
- [ ] Comparison view in the UI shows the forward vs historical bar chart by ABC class
- [ ] DFU drill-down line chart renders all 5 data series (forecast, CI band, forward SS, ROP/OUL, historical SS baseline)
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/041_create_replenishment_plan.sql` | Create — `fact_replenishment_plan` DDL + 5 indexes |
| `mvp/demand/config/replenishment_plan_config.yaml` | Create — CI interpretation, guard rails, EOQ params, retention |
| `mvp/demand/scripts/compute_replenishment_plan.py` | Create — full forward plan computation pipeline |
| `mvp/demand/api/routers/inv_planning_replenishment_plan.py` | Create — 4 REST endpoints |
| `mvp/demand/api/main.py` | Modify — `include_router(inv_planning_replenishment_plan.router)` |
| `mvp/demand/frontend/src/tabs/inv-planning/ReplenishmentPlanPanel.tsx` | Create — panel component |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Modify — add "Replen. Plan" sub-tab |
| `mvp/demand/frontend/src/api/queries/inv-planning-replenishment-plan.ts` | Create — TypeScript fetch functions and interfaces |
| `mvp/demand/frontend/src/api/queries/index.ts` | Modify — re-export new query module |
| `mvp/demand/frontend/vite.config.ts` | No change — `/inv-planning` proxy prefix already registered |
| `mvp/demand/Makefile` | Modify — add `replen-plan-schema`, `replen-plan-compute`, `replen-plan-dfu`, `replen-plan-dry`, `replen-plan-all` |
| `mvp/demand/tests/unit/test_replenishment_plan.py` | Create — 25+ unit tests |
| `mvp/demand/tests/api/test_inv_planning_replenishment_plan.py` | Create — 15+ API tests |
| `mvp/demand/frontend/src/tabs/inv-planning/__tests__/ReplenishmentPlanPanel.test.tsx` | Create — 8+ component tests |
| `docs/specs/03-inventory-planning/03-11-replenishment-plan.md` | Create (this file) |
| `docs/specs/01-data-platform/01-01-infrastructure.md` | Modify — add feature entries |
| `CLAUDE.md` | Modify — add key files, commands, conventions |
