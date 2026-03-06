# IPfeature10 — Safety Stock Monte Carlo Simulation

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P3 — Nice to Have

## Effort
XL (Extra Large)

## Expert Perspectives
- **Simulation Expert** (lead) — Monte Carlo methodology, vectorization, convergence
- **Statistical Analyst** — distribution fitting, empirical sampling, CSL computation
- **Inventory Planning Expert** — service level curve interpretation, simulation vs. formula

---

## Problem Statement

IPfeature3 computes safety stock using the analytical combined formula. This formula **assumes normally distributed demand and lead time**. For lumpy, intermittent, or heavily skewed demand (DFUs with `variability_class = 'lumpy'`), the normal assumption breaks down — the formula either over- or under-estimates required SS.

Monte Carlo simulation removes this assumption entirely. It samples directly from the **empirical demand and lead time distributions** observed in history, runs 10,000 simulated replenishment cycles, and counts how often stock runs out at each SS level. The result is a statistically grounded service-level curve.

Additionally, simulation provides a **validation check** on the analytical formula: if simulated SS ≈ analytical SS, the formula is trusted. If they diverge significantly, the planner knows to rely on simulation.

---

## User Story

> As an inventory planner, I want to run a Monte Carlo simulation for any item-location, see the resulting service level curve (SS level → achievable CSL%), and know the minimum safety stock that achieves my target service level — validated by simulation rather than formula alone — so that my SS decisions for lumpy-demand items are defensible.

---

## Business Value

- Provides simulation-backed SS for lumpy/intermittent items where formula is unreliable
- Creates a validation layer for the analytical formula (IPfeature3)
- Enables what-if analysis: "What CSL does our current SS actually provide?"
- Positions the platform for more advanced simulation features (multi-echelon, policy comparison)

---

## Monte Carlo Algorithm

```python
# For each item-loc:
# 1. Load demand history (last 24 months from fact_sales_monthly)
#    daily_demand_obs = [monthly_demand / 30.44 for each month] → approx daily series
#    (better: use daily MTD increments from fact_inventory_snapshot if available)

# 2. Load LT observations from dim_item_lead_time_profile
#    If lt_std available: use observed_lt_values list
#    Else: [lt_mean] × n_simulations (constant LT fallback)

# 3. For each SS_level in ss_levels_to_test:
#    stockouts = 0
#    for trial in range(n_simulations):
#        lt_sim = np.random.choice(observed_lt_values)         # empirical LT sample
#        demand_during_lt = np.sum(
#            np.random.choice(daily_demand_obs, size=int(lt_sim), replace=True)
#        )
#        if demand_during_lt > current_on_hand + SS_level:
#            stockouts += 1
#    csl[SS_level] = 1 - stockouts / n_simulations

# 4. Find recommended_ss = min(SS_level where csl[SS_level] >= target_csl)

# 5. Store results_by_ss_level as JSONB:
#    [ {ss_qty: 0, csl: 0.72}, {ss_qty: 10, csl: 0.81}, ... ]
```

**Convergence:** At n=10,000 trials, Monte Carlo standard error ≈ 0.5% for CSL estimates. Sufficient for inventory planning decisions (±1% precision).

---

## Data Requirements

### New DDL: `mvp/demand/sql/030_create_ss_simulation_results.sql`

New table `fact_ss_simulation_results`:

```sql
CREATE TABLE IF NOT EXISTS fact_ss_simulation_results (
    sim_sk                  BIGSERIAL PRIMARY KEY,
    sim_run_id              TEXT NOT NULL,
    item_no                 TEXT NOT NULL,
    loc                     TEXT NOT NULL,
    simulation_date         DATE NOT NULL,
    n_simulations           INTEGER NOT NULL,
    -- Distribution params used
    demand_distribution     TEXT,              -- 'empirical' | 'normal'
    demand_mean             NUMERIC(15,4),
    demand_std              NUMERIC(15,4),
    lt_distribution         TEXT,              -- 'empirical' | 'constant'
    lt_mean_days            NUMERIC(10,2),
    lt_std_days             NUMERIC(10,2),
    -- Results: JSONB array [{ss_qty, csl}]
    results_by_ss_level     JSONB NOT NULL,
    -- Recommendations
    target_csl              NUMERIC(6,4),
    recommended_ss          NUMERIC(15,4),     -- minimum SS achieving target_csl
    recommended_ss_days     NUMERIC(10,2),     -- recommended_ss / avg_daily_demand
    -- Comparison with analytical formula
    analytical_ss           NUMERIC(15,4),     -- from fact_safety_stock_targets.ss_combined
    sim_vs_analytical_pct   NUMERIC(10,2),     -- (sim - analytical) / analytical × 100
    -- Metadata
    run_duration_secs       NUMERIC(8,2),
    load_ts                 TIMESTAMPTZ DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ss_sim_run_item
    ON fact_ss_simulation_results (sim_run_id, item_no, loc);
CREATE INDEX IF NOT EXISTS idx_ss_sim_item_loc
    ON fact_ss_simulation_results (item_no, loc, simulation_date DESC);
CREATE INDEX IF NOT EXISTS idx_ss_sim_divergence
    ON fact_ss_simulation_results (sim_vs_analytical_pct)
    WHERE ABS(sim_vs_analytical_pct) > 20;
```

### New Config: `mvp/demand/config/simulation_config.yaml`

```yaml
simulation:
  n_simulations: 10000
  demand_distribution: empirical      # use actual demand obs as pool
  lt_distribution: empirical          # use observed LT change-points
  ss_levels_to_test: 20               # test 20 SS levels from 0 to 2×analytical_SS
  random_seed: 42
  min_demand_days: 90                 # need 90+ days of demand obs to use empirical
  max_dfus_per_batch: 500             # process in batches to manage memory
  target_csl: null                    # null = use per-DFU from safety_stock_config
```

---

## API Endpoints

**Router:** `mvp/demand/api/routers/inv_planning.py`

```
POST /inv-planning/simulation/run
  Auth: require_api_key
  Body: {
    item_no: str,
    loc: str,
    n_simulations: int (default: 10000),
    target_csl: float (optional, default: per-DFU from ss_config)
  }
  Response: { sim_run_id: str, status: 'queued' }
  -- Runs as background job via JobManager

GET /inv-planning/simulation/{sim_run_id}/status
  Response: {
    sim_run_id: str,
    status: 'queued' | 'running' | 'completed' | 'failed',
    progress_pct: int,
    item_no: str,
    loc: str,
    started_at: datetime | null,
    completed_at: datetime | null,
    error: str | null
  }

GET /inv-planning/simulation/results
  Query params: item (required), location (required)
  Response: {
    item_no: str,
    loc: str,
    simulation_date: date,
    n_simulations: int,
    service_level_curve: [ {ss_qty, csl} ],   -- from results_by_ss_level JSONB
    target_csl: float,
    recommended_ss: float,
    recommended_ss_days: float,
    analytical_ss: float,
    sim_vs_analytical_pct: float,
    -- Distribution info
    demand_distribution: str,
    lt_distribution: str,
    demand_mean: float,
    demand_std: float,
    lt_mean_days: float,
    lt_std_days: float
  }

GET /inv-planning/simulation/compare
  Query params: item (required), location (required)
  Response: {
    item_no: str,
    loc: str,
    analytical_ss: float,
    simulated_ss: float,
    difference_pct: float,
    service_level_curve: [ {ss_qty, csl} ],
    current_on_hand: float,
    current_csl: float     -- CSL at current_on_hand from simulation curve
  }
```

---

## Frontend UI

### Panel: "Monte Carlo Simulation" in `InvPlanningTab.tsx`

**Item-Location Selector:**
- Searchable item + location dropdowns (pre-populated from position table)
- "Run Simulation" button → calls POST /simulation/run
- Progress bar polling `/status` every 2s (same pattern as clustering scenarios)
- On completion: auto-loads results

**Service Level Curve Chart:**
- X-axis: SS quantity (units)
- Y-axis: Simulated CSL % (0–100%)
- Curve: Monotonically non-decreasing line from simulation results
- Reference lines:
  - Vertical blue dashed: Analytical SS (from IPfeature3)
  - Vertical green solid: Simulated recommended SS
  - Horizontal red dashed: Target CSL (e.g., 95%)
- Shaded region: area below target CSL = "insufficient SS zone"

**Comparison KPI Cards (row of 3):**
| Card | Value |
|---|---|
| Analytical SS | from IPfeature3 formula |
| Simulated SS | minimum SS achieving target CSL |
| Difference | (sim - analytical) / analytical % |

**Distribution Preview:**
- Demand: histogram of empirical demand observations + normal curve overlay
- Lead Time: histogram of observed LT values + normal curve overlay
- Shows why simulation may deviate from formula (non-normal demand)

---

## Backend Script

### `mvp/demand/scripts/run_ss_simulation.py`

```python
# Registered as job type 'inventory_simulation' in common/job_registry.py
# JOB_TYPE_REGISTRY entry:
#   'inventory_simulation': {
#     'group': 'simulation',
#     'callable': run_ss_simulation_job,
#     'description': 'Monte Carlo Safety Stock Simulation'
#   }

def run_ss_simulation_job(item_no: str, loc: str, n_simulations: int = 10000,
                          target_csl: float | None = None):
    # 1. Load demand history (24 months) from fact_sales_monthly
    # 2. Convert to daily observations (monthly / 30.44)
    # 3. Load LT observations from dim_item_lead_time_profile
    # 4. Load analytical_ss from fact_safety_stock_targets
    # 5. Determine SS levels to test: np.linspace(0, 2 * analytical_ss, n_levels)
    # 6. Vectorized Monte Carlo loop (numpy arrays, not Python loop)
    #    demand_pool = np.array(daily_demand_obs)
    #    lt_pool = np.array(observed_lt_values)
    #    for ss_level in ss_levels:
    #        lt_sim = np.random.choice(lt_pool, size=n_simulations)
    #        demand_sim = np.array([
    #            np.sum(np.random.choice(demand_pool, size=int(lt), replace=True))
    #            for lt in lt_sim
    #        ])
    #        stockout_rate = np.mean(demand_sim > current_on_hand + ss_level)
    #        csl = 1 - stockout_rate
    # 7. Find recommended_ss
    # 8. Insert into fact_ss_simulation_results
    # 9. Update job_history with result summary
```

**Job Registration:** Add to `common/job_registry.py`:
```python
JOB_TYPE_REGISTRY['inventory_simulation'] = {
    'group': 'simulation',
    'callable': run_ss_simulation_job,
    'description': 'Monte Carlo Safety Stock Simulation'
}
```

**CLI Usage:**
```bash
uv run python scripts/run_ss_simulation.py --item 100320 --loc 1401-BULK
uv run python scripts/run_ss_simulation.py --item 100320 --loc 1401-BULK --n-simulations 5000
```

**Makefile Targets:**
```makefile
sim-schema:
	# apply sql/030_create_ss_simulation_results.sql

sim-run:
	uv run python scripts/run_ss_simulation.py --item $(ITEM) --loc $(LOC)
	# Usage: make sim-run ITEM=100320 LOC=1401-BULK
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `dim_dfu.demand_std`, `demand_mean` | IPfeature1 | For distribution characterization |
| `dim_item_lead_time_profile` | IPfeature2 | Empirical LT observations |
| `fact_safety_stock_targets.ss_combined` | IPfeature3 | analytical_ss for comparison |
| `common/job_registry.py` | Existing | Background job execution |
| `numpy`, `scipy` | Existing deps | Monte Carlo vectorization |

---

## Testing Requirements

### Backend Unit Tests: `mvp/demand/tests/unit/test_ss_simulation.py`

Minimum 15 tests:
- CSL curve is monotonically non-decreasing: `csl[i+1] >= csl[i]` for all i
- CSL at SS=0 < 1.0 (always some risk at zero SS)
- CSL at SS=∞ → approaches 1.0 (but never exactly 1.0 with finite simulations)
- At n=1000 simulations, repeated runs with same seed → identical results
- Empirical sampling: demand pool of [10, 20, 30] → samples always in pool
- LT=1 (constant): demand_during_lt = 1 day of demand
- LT=0: demand_during_lt = 0 (edge case — no exposure)
- recommended_ss: verified as first SS level where csl >= target_csl
- sim_vs_analytical_pct = (sim - analytical) / analytical × 100 formula
- Normally distributed demand (large N): simulated SS ≈ analytical SS within 15%
- Results JSONB: valid list of {ss_qty, csl} dicts

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_simulation.py`

Minimum 10 tests:
- `POST /inv-planning/simulation/run` without auth → 403
- `POST /inv-planning/simulation/run` with auth → {sim_run_id, status='queued'}
- `GET /inv-planning/simulation/{id}/status` → {status, progress_pct}
- `GET /inv-planning/simulation/results?item=X&location=Y` → service_level_curve list
- `GET /inv-planning/simulation/results` (no params) → 422
- `GET /inv-planning/simulation/compare` → analytical_ss and simulated_ss both present
- CSL curve has ≥ 5 data points
- Status endpoint: unknown run_id → 404

---

## Acceptance Criteria

- [ ] Service level curve is monotonically non-decreasing (CSL ≥ CSL at previous SS level always)
- [ ] With n=10,000 and normal demand distribution: simulated SS within 15% of analytical SS
- [ ] Simulation runs as background job via JobManager (pollable via status endpoint)
- [ ] `results_by_ss_level` JSONB parses as valid list of {ss_qty, csl} objects
- [ ] `sim_vs_analytical_pct = (recommended_ss - analytical_ss) / analytical_ss × 100`
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/030_create_ss_simulation_results.sql` | Create |
| `mvp/demand/config/simulation_config.yaml` | Create |
| `mvp/demand/scripts/run_ss_simulation.py` | Create |
| `mvp/demand/common/job_registry.py` | Modify — add inventory_simulation job type |
| `mvp/demand/api/routers/inv_planning.py` | Modify — add simulation endpoints |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Modify — add Simulation panel |
| `mvp/demand/tests/unit/test_ss_simulation.py` | Create |
| `mvp/demand/tests/api/test_inv_planning_simulation.py` | Create |
| `mvp/demand/Makefile` | Modify — add sim-* targets |
| `docs/design-specs/IPfeature10.md` | Create (this file) |
