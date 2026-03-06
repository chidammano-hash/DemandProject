# IPfeature6 — Inventory Health Score Dashboard

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P2 — Should Have

## Effort
M (Medium)

## Expert Perspectives
- **Supply Chain Control Tower Expert** (lead) — composite scoring, triage hierarchy
- **UI/UX Expert** — health tier visualization, drill-down, segment heatmap
- **Inventory Planning Expert** — score component weighting and thresholds

---

## Problem Statement

With IPfeature3 computing safety stock targets and IPfeature6 extending them with EOQ, planners now have the numbers — but reviewing 10,000 DFUs to find which need attention is still impossible.

A planner needs a single number per SKU-location that says: "this item is healthy, monitor, at-risk, or critical." Without it, all items look equal until a stockout occurs.

---

## User Story

> As a supply chain director, I want a single 0–100 health score for each SKU-location combining SS coverage, DOS target adherence, stockout risk history, and forecast accuracy, so that I can instantly see which items need attention without reviewing every row manually.

---

## Business Value

- Converts raw SS/EOQ/stockout data into a single triage signal
- Feeds IPfeature7 (exception queue priority) and IPfeature15 (control tower)
- Enables "critical items first" drill-down and segment health heatmaps
- Provides managers a daily headline: "Portfolio health: 72/100 (↓3 vs last month)"

---

## Health Score Formula

**4 components, 25 points each, summed to 0–100:**

### Component 1: SS Coverage (25 pts)
```
ss_coverage = current_qty_on_hand / ss_combined
score_ss_coverage:
  ss_coverage ≥ 1.5  → 25 (well-stocked above SS)
  ss_coverage ≥ 1.0  → 18 (at SS target)
  ss_coverage ≥ 0.5  → 10 (below SS but some buffer)
  ss_coverage < 0.5  → 0  (critically low)
  SS not computed    → 12 (neutral — no target set yet)
```

### Component 2: DOS Target Adherence (25 pts)
```
current_dos = eom_qty_on_hand / avg_daily_sls
score_dos_target:
  current_dos BETWEEN target_dos_min AND target_dos_max → 25 (within target)
  current_dos > target_dos_max                          → 10 (excess inventory)
  current_dos < target_dos_min AND current_dos > 0      → 5  (below minimum)
  current_dos = 0 (stockout)                            → 0
  No target set                                         → 15 (neutral)
```

### Component 3: Stockout Risk History (25 pts)
```
From mv_inventory_forecast_monthly, count stockout months in last 3 months:
  0 stockout months in last 3  → 25 (no recent stockouts)
  1 stockout month             → 15 (one recent event)
  2 stockout months            → 8  (repeated stockouts)
  3 stockout months            → 0  (chronic stockout)
  No forecast data available   → 20 (assume OK, not critical)
```

### Component 4: Forecast Accuracy (25 pts)
```
recent_wape = 3-month avg WAPE from mv_inventory_forecast_monthly (champion or external)
score_forecast_accuracy:
  recent_wape < 15%  → 25 (excellent)
  recent_wape < 25%  → 20 (good)
  recent_wape < 40%  → 15 (fair)
  recent_wape < 60%  → 8  (poor)
  recent_wape ≥ 60%  → 0  (very poor)
  No data            → 15 (neutral)
```

### Composite Score & Tier
```
health_score = score_ss_coverage + score_dos_target + score_stockout_risk + score_forecast_accuracy
health_tier:
  health_score ≥ 80 → 'healthy'
  health_score ≥ 60 → 'monitor'
  health_score ≥ 40 → 'at_risk'
  health_score < 40 → 'critical'
```

---

## Data Requirements

### New DDL: `mvp/demand/sql/026_create_inventory_health_score.sql`

New materialized view `mv_inventory_health_score`:

```sql
CREATE MATERIALIZED VIEW mv_inventory_health_score AS
WITH latest_inv AS (
    SELECT DISTINCT ON (item_no, loc)
        item_no, loc, month_start,
        eom_qty_on_hand, monthly_sales, avg_daily_sls, latest_lead_time_days
    FROM agg_inventory_monthly
    ORDER BY item_no, loc, month_start DESC
),
recent_stockout AS (
    SELECT item_no, loc,
        SUM(CASE WHEN is_stockout THEN 1 ELSE 0 END) AS stockout_count_3m
    FROM mv_inventory_forecast_monthly
    WHERE month_start >= (SELECT MAX(month_start) FROM mv_inventory_forecast_monthly)
                          - INTERVAL '2 months'
      AND model_id IN ('champion', 'external')
    GROUP BY item_no, loc
),
recent_accuracy AS (
    SELECT item_no, loc,
        SUM(abs_error) / NULLIF(ABS(SUM(actual_demand)), 0) AS recent_wape
    FROM mv_inventory_forecast_monthly
    WHERE month_start >= (SELECT MAX(month_start) FROM mv_inventory_forecast_monthly)
                          - INTERVAL '2 months'
      AND model_id IN ('champion', 'external')
    GROUP BY item_no, loc
),
ss AS (
    SELECT item_no, loc, ss_combined, reorder_point, is_below_ss,
           ss_coverage, target_dos_min, target_dos_max
    FROM fact_safety_stock_targets
    WHERE policy_version = 'v1'
)
SELECT
    l.item_no, l.loc, l.month_start,
    d.cluster_assignment, d.abc_vol, d.variability_class,
    d.seasonality_profile, d.region, d.xyz_class, d.abc_xyz_segment,
    -- Current position
    l.eom_qty_on_hand, l.avg_daily_sls,
    CASE WHEN l.avg_daily_sls > 0
         THEN l.eom_qty_on_hand / l.avg_daily_sls ELSE NULL END AS current_dos,
    -- SS targets
    s.ss_combined, s.reorder_point, s.is_below_ss, s.ss_coverage,
    s.target_dos_min, s.target_dos_max,
    -- Forecast accuracy
    ra.recent_wape,
    -- Score components
    CASE WHEN s.ss_combined IS NULL THEN 12
         WHEN s.ss_coverage >= 1.5 THEN 25
         WHEN s.ss_coverage >= 1.0 THEN 18
         WHEN s.ss_coverage >= 0.5 THEN 10
         ELSE 0 END AS score_ss_coverage,
    CASE WHEN s.target_dos_min IS NULL THEN 15
         WHEN l.avg_daily_sls = 0 THEN 5
         WHEN (l.eom_qty_on_hand / l.avg_daily_sls)
              BETWEEN s.target_dos_min AND s.target_dos_max THEN 25
         WHEN (l.eom_qty_on_hand / l.avg_daily_sls) > s.target_dos_max THEN 10
         ELSE 5 END AS score_dos_target,
    CASE WHEN rs.stockout_count_3m IS NULL THEN 20
         WHEN rs.stockout_count_3m = 0 THEN 25
         WHEN rs.stockout_count_3m = 1 THEN 15
         WHEN rs.stockout_count_3m = 2 THEN 8
         ELSE 0 END AS score_stockout_risk,
    CASE WHEN ra.recent_wape IS NULL THEN 15
         WHEN ra.recent_wape < 0.15 THEN 25
         WHEN ra.recent_wape < 0.25 THEN 20
         WHEN ra.recent_wape < 0.40 THEN 15
         WHEN ra.recent_wape < 0.60 THEN 8
         ELSE 0 END AS score_forecast_accuracy,
    -- Composite
    (score_ss_coverage + score_dos_target + score_stockout_risk
     + score_forecast_accuracy)::INTEGER AS health_score,
    CASE WHEN (composite) >= 80 THEN 'healthy'
         WHEN (composite) >= 60 THEN 'monitor'
         WHEN (composite) >= 40 THEN 'at_risk'
         ELSE 'critical' END AS health_tier
FROM latest_inv l
LEFT JOIN dim_dfu d ON l.item_no = d.dmdunit AND l.loc = d.loc
LEFT JOIN ss s ON l.item_no = s.item_no AND l.loc = s.loc
LEFT JOIN recent_stockout rs ON l.item_no = rs.item_no AND l.loc = rs.loc
LEFT JOIN recent_accuracy ra ON l.item_no = ra.item_no AND l.loc = ra.loc
WITH NO DATA;

CREATE UNIQUE INDEX IF NOT EXISTS idx_health_score_pk
    ON mv_inventory_health_score (item_no, loc);
CREATE INDEX IF NOT EXISTS idx_health_score_tier
    ON mv_inventory_health_score (health_tier);
CREATE INDEX IF NOT EXISTS idx_health_score_critical
    ON mv_inventory_health_score (health_score)
    WHERE health_tier = 'critical';
CREATE INDEX IF NOT EXISTS idx_health_score_abc
    ON mv_inventory_health_score (abc_vol, health_tier);
```

---

## API Endpoints

**Router:** `mvp/demand/api/routers/inv_planning.py`

```
GET /inv-planning/health/summary
  Query params: abc_vol, cluster_assignment, region, variability_class
  Response: {
    total_dfus: int,
    by_tier: { healthy: int, monitor: int, at_risk: int, critical: int },
    avg_health_score: float,
    score_histogram: [ {bin_start, bin_end, count} × 10 bins ],
    component_avgs: {
      score_ss_coverage, score_dos_target, score_stockout_risk, score_forecast_accuracy
    }
  }
  Cache: max-age=120s

GET /inv-planning/health/detail
  Query params: item, location, health_tier, abc_vol, cluster_assignment, variability_class,
                limit, offset, sort_by (health_score | ss_coverage | current_dos), sort_dir
  Response: {
    total: int,
    rows: [ {item_no, loc, abc_vol, variability_class, health_score, health_tier,
             score_ss_coverage, score_dos_target, score_stockout_risk, score_forecast_accuracy,
             ss_coverage, current_dos, target_dos_min, target_dos_max, is_below_ss,
             recent_wape} ]
  }
  Cache: max-age=120s

GET /inv-planning/health/heatmap
  Query params: group_x (abc_vol | xyz_class), group_y (variability_class | cluster_assignment | region)
  Response: {
    x_labels: [ str ],
    y_labels: [ str ],
    cells: [ {x, y, avg_health_score, count, critical_count} ]
  }
  Cache: max-age=300s
```

---

## Frontend UI

### Panel: "Portfolio Health" in `InvPlanningTab.tsx` (landing section — shown first on tab load)

**4 Status KPI Cards:**
| Tier | Color | Card content |
|---|---|---|
| Healthy | Green | count + %, health_score ≥ 80 |
| Monitor | Yellow | count + %, health_score 60–79 |
| At Risk | Orange | count + %, health_score 40–59 |
| Critical | Red | count + %, health_score < 40 |

**Health Tier Donut Chart:**
- 4 segments, colored by tier
- Center label: "Avg Score: 72"

**ABC × Variability Class Heatmap:**
- Rows: ABC class (A / B / C)
- Columns: variability_class (low / medium / high / lumpy)
- Cell: avg_health_score + count
- Color scale: 80–100=green, 60–79=yellow, 40–59=orange, <40=red
- Click cell → filters detail table below

**Portfolio Health Detail Table:**
- Default sort: health_score ascending (worst first)
- Columns: item, loc, abc_vol, variability_class, health_score, health_tier badge, is_below_ss, current_dos, target_dos_min/max, recent_wape
- Row color: red if critical, orange if at_risk, yellow if monitor, white if healthy
- Filter bar: health_tier dropdown, item input, location input

---

## Backend Script

### `mvp/demand/scripts/refresh_health_scores.py`

Thin wrapper:
```python
# Executes: REFRESH MATERIALIZED VIEW CONCURRENTLY mv_inventory_health_score
# Logs: rows updated, timing
```

**Makefile Targets:**
```makefile
health-schema:
	# apply sql/026_create_inventory_health_score.sql (CREATE MAT VIEW WITH NO DATA)

health-refresh:
	uv run python scripts/refresh_health_scores.py
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `dim_dfu.variability_class` | IPfeature1 | Required for heatmap segmentation |
| `fact_safety_stock_targets` | IPfeature3 | SS coverage component |
| `mv_inventory_forecast_monthly` | Feature 37 | Stockout risk + forecast accuracy components |
| `agg_inventory_monthly` | Existing | Current DOS computation |

---

## Testing Requirements

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_health.py`

Minimum 8 tests:
- `GET /inv-planning/health/summary` → 200 OK, by_tier has 4 keys
- `GET /inv-planning/health/summary?abc_vol=A` → filtered (only A-class items)
- `GET /inv-planning/health/detail` → rows have health_score in 0–100 range
- `GET /inv-planning/health/detail?health_tier=critical` → all rows health_tier='critical'
- `GET /inv-planning/health/detail` sort by health_score asc → first row has lowest score
- `GET /inv-planning/health/heatmap` → cells list, each with x, y, avg_health_score
- Empty view → returns zeros, not 500
- health_score bounds: all rows 0 ≤ health_score ≤ 100

### Unit Test Coverage (within existing ss tests or new file)
- score formula: ss_coverage=1.6 → score_ss_coverage=25
- score formula: ss_coverage=0.9 → score_ss_coverage=18
- score formula: ss_coverage=0.4 → score_ss_coverage=0
- tier classification: 82 → 'healthy', 61 → 'monitor', 45 → 'at_risk', 38 → 'critical'
- wape score: 0.12 → 25, 0.38 → 15, 0.65 → 0

### Frontend Tests: extend `InvPlanningTab.test.tsx`
- Health panel renders 4 KPI cards
- Donut chart has 4 segments
- Heatmap renders grid cells
- Detail table renders with health_score column

---

## Acceptance Criteria

- [ ] Every DFU has a `health_score` 0–100 and `health_tier` after `make health-refresh`
- [ ] `health_score = sum of 4 components`, each 0–25
- [ ] Items with no SS target get neutral component scores (not 0)
- [ ] `health_tier = 'critical'` for health_score < 40
- [ ] Heatmap shows non-trivial distribution across ABC × variability segments
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/026_create_inventory_health_score.sql` | Create |
| `mvp/demand/scripts/refresh_health_scores.py` | Create |
| `mvp/demand/api/routers/inv_planning.py` | Modify — add health endpoints |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Modify — add Health panel |
| `mvp/demand/tests/api/test_inv_planning_health.py` | Create |
| `mvp/demand/Makefile` | Modify — add health-* targets |
| `docs/design-specs/IPfeature6.md` | Create (this file) |
