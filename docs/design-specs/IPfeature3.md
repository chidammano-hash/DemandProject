# IPfeature3 — Safety Stock Engine

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P1 — Must Have (Foundation)

## Effort
L (Large)

## Expert Perspectives
- **Inventory Planning Expert** (lead) — SS formula selection, policy mapping, ROP
- **Statistical Analyst** — Z-score service level math, combined uncertainty formula
- **Demand Planning Expert** — service level target configuration by segment

---

## Problem Statement

The existing system shows stockouts and excess inventory **after they happen** (via `mv_inventory_forecast_monthly`). It has no concept of a safety stock **target** — there is nothing to compare the current on-hand position against, no way to know if current inventory is adequate or dangerously low.

Without safety stock targets:
- Planners cannot tell if today's inventory level is "fine" or "dangerously low"
- No alert can be generated until a stockout already occurs
- Excess inventory cannot be distinguished from adequate buffer stock

This feature computes the mathematically correct safety stock for every SKU-location using the combined demand + lead time variability formula.

---

## User Story

> As an inventory planner, I want science-based safety stock targets per SKU-location using the combined demand-and-lead-time variability formula with configurable service level targets by ABC segment, so I see exactly how much buffer stock is required and can immediately see which items are below target.

---

## Business Value

- The **core computational feature** of the entire InventoryPlanning EPIC
- Every downstream feature (IPfeature6 health scores, IPfeature7 exceptions, IPfeature13 investment optimization) consumes `fact_safety_stock_targets`
- Replaces gut-feel or fixed-percentage SS rules with statistically defensible numbers
- Directly quantifies portfolio-level SS gap in units

---

## Key Formulas

```
# Unit conversions (monthly → daily)
σ_D_daily  = demand_std_monthly / sqrt(30.44)
D_avg_daily = demand_mean_monthly / 30.44

# Safety stock components
SS_demand   = Z × sqrt(LT_mean_days × σ_D_daily²)
SS_lt       = Z × D_avg_daily × lt_std_days
SS_combined = Z × sqrt(LT_mean_days × σ_D_daily² + D_avg_daily² × lt_std_days²)

# Reorder point
ROP = D_avg_daily × LT_mean_days + SS_combined

# Target DOS
target_dos_min = SS_combined / D_avg_daily   (in days)
target_dos_max = (SS_combined + effective_eoq/2) / D_avg_daily  (added in IPfeature4)

# Position comparison
ss_coverage = current_qty_on_hand / SS_combined
ss_gap      = current_qty_on_hand - SS_combined   (negative = shortfall)
is_below_ss = current_qty_on_hand < SS_combined
```

**Z-score lookup table (standard normal):**
| Service Level | Z |
|---|---|
| 85% | 1.036 |
| 90% | 1.282 |
| 95% | 1.645 |
| 97% | 1.881 |
| 98% | 2.054 |
| 99% | 2.326 |

---

## Data Requirements

### New DDL: `mvp/demand/sql/024_create_safety_stock_targets.sql`

New table `fact_safety_stock_targets`:

```sql
CREATE TABLE IF NOT EXISTS fact_safety_stock_targets (
    ss_sk                BIGSERIAL PRIMARY KEY,
    ss_ck                TEXT UNIQUE NOT NULL,    -- item_no || '_' || loc || '_' || policy_version
    item_no              TEXT NOT NULL,
    loc                  TEXT NOT NULL,
    policy_version       TEXT NOT NULL DEFAULT 'v1',
    effective_date       DATE NOT NULL,
    -- Inputs recorded for auditability
    service_level_target  NUMERIC(6,4),
    z_score               NUMERIC(8,4),
    demand_mean_monthly   NUMERIC(15,4),
    demand_std_monthly    NUMERIC(15,4),
    lead_time_mean_days   NUMERIC(10,2),
    lead_time_std_days    NUMERIC(10,2),
    -- Safety stock outputs
    ss_demand_only        NUMERIC(15,4),   -- demand uncertainty only
    ss_lt_only            NUMERIC(15,4),   -- LT uncertainty only
    ss_combined           NUMERIC(15,4),   -- recommended (combined formula)
    ss_method             TEXT NOT NULL,   -- 'combined' | 'demand_only'
    -- Derived targets
    avg_daily_demand      NUMERIC(15,4),
    reorder_point         NUMERIC(15,4),
    target_min_qty        NUMERIC(15,4),   -- = ss_combined
    target_max_qty        NUMERIC(15,4),   -- updated by IPfeature4
    target_dos_min        NUMERIC(10,2),   -- SS in days
    target_dos_max        NUMERIC(10,2),   -- updated by IPfeature4
    -- Current position comparison (refreshed on each run)
    current_qty_on_hand   NUMERIC(15,4),
    current_dos           NUMERIC(10,2),
    ss_coverage           NUMERIC(10,4),   -- current_qty / ss_combined
    ss_gap                NUMERIC(15,4),   -- current_qty - ss_combined
    is_below_ss           BOOLEAN,
    load_ts               TIMESTAMPTZ DEFAULT NOW(),
    modified_ts           TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ss_targets_pk
    ON fact_safety_stock_targets (item_no, loc, policy_version);
CREATE INDEX IF NOT EXISTS idx_ss_targets_item    ON fact_safety_stock_targets (item_no);
CREATE INDEX IF NOT EXISTS idx_ss_targets_loc     ON fact_safety_stock_targets (loc);
CREATE INDEX IF NOT EXISTS idx_ss_targets_below   ON fact_safety_stock_targets (is_below_ss)
    WHERE is_below_ss = TRUE;
```

### New Config: `mvp/demand/config/safety_stock_config.yaml`

```yaml
safety_stock:
  default_method: combined        # combined | demand_only
  policy_version: v1

  # Service levels by ABC class (ABC from dim_dfu.abc_vol)
  service_levels:
    A: 0.98
    B: 0.95
    C: 0.90
    default: 0.95                 # fallback if abc_vol is NULL or unrecognized

  # Precomputed Z-scores (standard normal, one-tail)
  z_table:
    0.85: 1.036
    0.90: 1.282
    0.95: 1.645
    0.97: 1.881
    0.98: 2.054
    0.99: 2.326

  # Guard rails
  min_ss_days: 3         # never compute SS below 3 days of supply
  max_ss_days: 120       # cap at 120 days (prevents outlier spikes)

  # Fallback: when LT std is unavailable (< min LT observations)
  lt_std_fallback_pct: 0.20   # assume LT std = 20% of LT mean
```

---

## API Endpoints

**Router:** `mvp/demand/api/routers/inv_planning.py` (extends IPfeature1/2 router)

```
GET /inv-planning/safety-stock/summary
  Query params: item, location, abc_vol, cluster_assignment, policy_version (default: v1)
  Response: {
    total_dfus: int,
    below_ss_count: int,
    below_ss_pct: float,
    total_ss_gap_units: float,        -- sum of negative ss_gap values
    avg_ss_coverage: float,
    by_class: {
      A: { count, below_ss_count, avg_ss_combined, avg_coverage },
      B: { ... },
      C: { ... }
    },
    top_gaps: [ {item_no, loc, ss_combined, current_qty, ss_gap, ss_coverage} × 10 ]
  }
  Cache: max-age=120s

GET /inv-planning/safety-stock/detail
  Query params: item, location, abc_vol, is_below_ss (bool), cluster_assignment,
                policy_version, limit, offset, sort_by (ss_gap | ss_coverage | ss_combined), sort_dir
  Response: {
    total: int,
    rows: [ {item_no, loc, abc_vol, service_level_target, z_score,
             ss_combined, reorder_point, current_qty_on_hand, current_dos,
             ss_gap, ss_coverage, is_below_ss, target_dos_min} ]
  }
  Cache: max-age=120s

GET /inv-planning/safety-stock/waterfall
  Query params: item (required), location (required), policy_version (default: v1)
  Response: {
    item_no: str,
    loc: str,
    demand_component: float,      -- SS_demand_only
    lt_component: float,          -- SS_lt_only
    combined_ss: float,           -- SS_combined
    reorder_point: float,         -- ROP
    current_on_hand: float,
    ss_gap: float,
    z_score: float,
    service_level_target: float,
    lt_mean_days: float,
    lt_std_days: float | null,
    demand_mean_monthly: float,
    demand_std_monthly: float
  }
  Cache: max-age=120s

POST /inv-planning/safety-stock/override
  Auth: require_api_key
  Body: {
    item_no: str,
    loc: str,
    ss_override_qty: float,
    reason: str
  }
  Response: { item_no, loc, ss_combined (updated), ss_method: 'manual', modified_ts }

GET /inv-planning/safety-stock/config
  Response: current safety_stock_config.yaml as JSON
  Cache: max-age=600s
```

---

## Frontend UI

### New Tab: `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx`

This is the primary new tab for the InventoryPlanning EPIC. IPfeature3 delivers the first panel.

**Sidebar entry:**
- Icon: `Brain` (lucide-react)
- Label: "Inv. Planning"
- Section: inventory
- Keyboard shortcut: `7`
- Lazy-loaded via `React.lazy()`

**Panel 1: Safety Stock Overview (landing section of tab)**

KPI Cards (row of 4):
| Card | Value | Severity |
|---|---|---|
| Items Below SS Target | count + % | red if >20%, amber if 10–20%, green if <10% |
| Total SS Gap | sum of negative ss_gap in units | red if negative |
| Portfolio SS Coverage | avg ss_coverage | red if <0.8, amber if 0.8–1.0, green if >1.0 |
| Avg Safety Stock (Days) | avg target_dos_min | contextual |

Heat Matrix "Below SS by Segment":
- Rows: ABC class (A / B / C)
- Columns: variability_class (low / medium / high / lumpy)
- Cell value: count of items below SS in that intersection
- Color: white (0) → deep red (many)

Detail Table:
- Columns: item, loc, abc_vol, service_level_target, ss_combined, ROP, current_qty, ss_gap, ss_coverage, target_dos_min
- Filter controls: item input, location input, abc_vol dropdown, is_below_ss checkbox
- Sort by ss_gap ascending (biggest shortfall first)
- Row colors: red if is_below_ss AND ss_coverage < 0.5, amber if is_below_ss, green if ss_coverage > 1.5

SS Waterfall Chart (side panel, opens on row click):
- Horizontal stacked bar decomposition:
  - Bar 1: Demand component (blue)
  - Bar 2: LT component (orange) stacked on Bar 1
  - Bar 3: Combined SS = sqrt(demand² + LT²) (purple reference line)
  - Bar 4: ROP = LT demand + SS (gray)
  - Bar 5: Current on-hand (green or red depending on vs. SS)
- Shows which driver (demand variance or LT variance) dominates

---

## Backend Script

### `mvp/demand/scripts/compute_safety_stock.py`

```python
# Algorithm:
# 1. Load config: safety_stock_config.yaml
# 2. Load dim_dfu: demand_mean, demand_std, abc_vol, variability_class, dmdunit, loc
#    (populated by IPfeature1)
# 3. Load dim_item_lead_time_profile: lt_mean_days, lt_std_days per item_no + loc
#    (populated by IPfeature2)
#    If lt_std_days is NULL: use lt_mean_days * lt_std_fallback_pct as fallback
# 4. Load agg_inventory_monthly: most recent eom_qty_on_hand, avg_daily_sls per item_no + loc
# 5. For each DFU:
#    a. Determine service_level from abc_vol → config → z_score lookup
#    b. Convert monthly to daily: σ_D_daily = demand_std / sqrt(30.44)
#    c. D_avg_daily = demand_mean / 30.44
#    d. SS_demand = z * sqrt(LT_mean * σ_D_daily²)
#    e. SS_lt = z * D_avg_daily * lt_std_days
#    f. SS_combined = z * sqrt(LT_mean * σ_D_daily² + D_avg_daily² * lt_std_days²)
#    g. ROP = D_avg_daily * LT_mean + SS_combined
#    h. Apply guards: if SS_combined < min_ss_days * D_avg_daily → SS_combined = min_ss_days * D_avg_daily
#                    if SS_combined > max_ss_days * D_avg_daily → SS_combined = max_ss_days * D_avg_daily
#    i. target_dos_min = SS_combined / D_avg_daily (NULL if D_avg_daily = 0)
#    j. current_qty = latest eom_qty_on_hand
#    k. ss_coverage = current_qty / SS_combined
#    l. ss_gap = current_qty - SS_combined
#    m. is_below_ss = current_qty < SS_combined
# 6. Batch upsert into fact_safety_stock_targets
#    ON CONFLICT (item_no, loc, policy_version) DO UPDATE

# Edge cases:
# - demand_mean = 0 AND demand_std = 0: SS_combined = 0, mark ss_method='demand_only'
#   (zero-demand items need no safety stock by formula; check with planner)
# - LT_mean = 0: skip (invalid data), log warning
# - D_avg_daily = 0 but demand_std > 0: compute SS_demand only (intermittent with rare spikes)
```

**CLI Usage:**
```bash
uv run python scripts/compute_safety_stock.py
uv run python scripts/compute_safety_stock.py --config config/safety_stock_config.yaml
uv run python scripts/compute_safety_stock.py --policy-version v2
```

---

## Makefile Targets

```makefile
ss-schema:
	# apply sql/024_create_safety_stock_targets.sql

ss-compute:
	uv run python scripts/compute_safety_stock.py

ss-all: ss-schema ss-compute
	# Note: requires variability-all and lt-profile-all to have run first
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `dim_dfu.demand_std`, `demand_mean`, `abc_vol` | IPfeature1 | Must run variability-compute first |
| `dim_item_lead_time_profile` | IPfeature2 | Must run lt-profile-compute first |
| `agg_inventory_monthly` | Existing | Latest EOM on-hand for current position |
| `config/safety_stock_config.yaml` | New | Service levels, Z-table, guard rails |

---

## Testing Requirements

### Backend Unit Tests: `mvp/demand/tests/unit/test_safety_stock.py`

Minimum 20 tests:

**Formula correctness:**
- Verified case: σ_D_daily=2.0, LT_mean=14, lt_std=3.0, Z=1.645
  - SS_demand = 1.645 × sqrt(14 × 4) = 1.645 × 7.483 = 12.31
  - SS_lt = 1.645 × (D_avg=10) × 3.0 = 49.35
  - SS_combined = 1.645 × sqrt(14×4 + 100×9) = 1.645 × sqrt(956) = 50.87
- Z=1.645 at 95% SL (verify table lookup)
- Z=2.054 at 98% SL
- Service level routing: abc_vol='A' → SL=0.98 → Z=2.054

**Guard rails:**
- min_ss_days cap enforced: if SS < 3 × D_avg_daily → SS = 3 × D_avg_daily
- max_ss_days cap enforced: if SS > 120 × D_avg_daily → SS = 120 × D_avg_daily

**ROP:**
- ROP = D_avg_daily × LT_mean + SS_combined (verified numerically)

**Edge cases:**
- demand_std = 0, lt_std > 0: SS = Z × D_avg × lt_std (LT component only)
- lt_std = 0, demand_std > 0: SS = Z × sqrt(LT × σ_D²) (demand component only)
- demand_mean = 0 AND demand_std = 0: SS_combined = 0 (zero demand item)
- lt_mean = 0: skipped, not inserted
- lt_std NULL → fallback: lt_std = lt_mean × lt_std_fallback_pct (0.20)

**ss_coverage / ss_gap:**
- current_qty = 50, ss_combined = 80 → ss_gap = -30, is_below_ss = TRUE
- current_qty = 100, ss_combined = 80 → ss_gap = 20, is_below_ss = FALSE
- ss_coverage = current_qty / ss_combined

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_safety_stock.py`

Minimum 12 tests:
- `GET /inv-planning/safety-stock/summary` → 200 OK, by_class has A/B/C keys
- `GET /inv-planning/safety-stock/summary?abc_vol=A` → filtered to A items only
- `GET /inv-planning/safety-stock/detail` → 200 OK, rows with ss_combined
- `GET /inv-planning/safety-stock/detail?is_below_ss=true` → all rows have is_below_ss=True
- `GET /inv-planning/safety-stock/waterfall?item=X&location=Y` → 200 OK with all components
- `GET /inv-planning/safety-stock/waterfall` (no item/loc) → 422
- `GET /inv-planning/safety-stock/config` → 200 OK, has service_levels key
- `POST /inv-planning/safety-stock/override` without auth → 403
- `POST /inv-planning/safety-stock/override` with auth → 200 OK, ss_method='manual'
- Pagination: limit=5 returns ≤5 rows with correct total
- Sort by ss_gap asc → first row has most negative gap (biggest shortfall)

### Frontend Tests: `mvp/demand/frontend/src/tabs/__tests__/InvPlanningTab.test.tsx`

Initial smoke tests:
- Tab renders without crash
- KPI cards render with mock data
- Heat matrix renders 3×4 grid (ABC × variability_class)
- Detail table shows item + ss_combined columns

---

## Acceptance Criteria

- [ ] Z=1.645 at 95% SL verified in unit tests
- [ ] `ss_combined = Z × sqrt(LT_mean × σ_D_daily² + D_avg_daily² × lt_std_days²)` verified numerically
- [ ] `reorder_point = D_avg_daily × LT_mean + ss_combined` correct
- [ ] All DFUs have `is_below_ss` populated (TRUE or FALSE) after `make ss-compute`
- [ ] Items with zero demand have `ss_combined = 0` and `is_below_ss = FALSE`
- [ ] Guard rails: no DFU has `ss_combined < min_ss_days × D_avg_daily` unless zero demand
- [ ] `GET /inv-planning/safety-stock/summary` shows non-zero `below_ss_count`
- [ ] Waterfall chart correctly shows demand vs. LT component split
- [ ] `make test-all` passes with no regressions

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/024_create_safety_stock_targets.sql` | Create |
| `mvp/demand/config/safety_stock_config.yaml` | Create |
| `mvp/demand/scripts/compute_safety_stock.py` | Create |
| `mvp/demand/api/routers/inv_planning.py` | Modify — add SS endpoints |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Create |
| `mvp/demand/frontend/src/App.tsx` | Modify — add lazy-loaded InvPlanningTab |
| `mvp/demand/frontend/src/components/AppSidebar.tsx` | Modify — add "Inv. Planning" nav item |
| `mvp/demand/frontend/src/hooks/useKeyboardShortcuts.ts` | Modify — add shortcut `7` |
| `mvp/demand/tests/unit/test_safety_stock.py` | Create |
| `mvp/demand/tests/api/test_inv_planning_safety_stock.py` | Create |
| `mvp/demand/frontend/src/tabs/__tests__/InvPlanningTab.test.tsx` | Create |
| `mvp/demand/Makefile` | Modify — add ss-* targets |
| `docs/design-specs/IPfeature3.md` | Create (this file) |
