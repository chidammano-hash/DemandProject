# IPfeature11 — ABC-XYZ Policy Matrix & Portfolio Segmentation

## EPIC
InventoryPlanning

## Status
Planned

## Priority
P2 — Should Have

## Effort
M (Medium)

## Expert Perspectives
- **Demand Planning Expert** (lead) — XYZ variability classification, segment-driven strategy
- **Inventory Planning Expert** — 9-cell policy matrix, DOS targets per segment
- **Statistical Analyst** — CV thresholds, intermittency classification

---

## Problem Statement

The existing system has `abc_vol` (volume-based ABC classification from the source system). IPfeature1 added `variability_class` (CV-based). Neither alone tells the full story:

- An **AZ item** (high value, highly variable demand) needs very different treatment from an **AX item** (high value, stable demand)
- An **AX item** can safely use lean continuous review with low SS; an **AZ item** needs large SS buffers and frequent review
- Current system treats all A-class items identically regardless of variability

The ABC-XYZ matrix is the industry-standard segmentation framework for differentiating inventory strategies across 9 item types.

---

## User Story

> As an inventory planner, I want each SKU-location automatically classified into a 9-cell ABC-XYZ matrix with pre-configured DOS targets and service levels per cell, so I can apply differentiated inventory strategies across my portfolio instead of managing 10,000 SKUs with identical rules.

---

## Business Value

- Replaces one-size-fits-all inventory rules with segment-appropriate policies
- Identifies AZ items (most challenging: high-value + volatile) for special attention
- CZ items (low-value + volatile) are candidates for rationalization or phase-out
- Provides a vocabulary for communication between planners and managers: "our AZ items have 73 average health score vs 91 for AX items"

---

## XYZ Classification Rules

```
XYZ (demand variability):
  X → demand_cv < 0.3  (low variability, predictable)
  Y → demand_cv 0.3 – 0.8  (moderate variability)
  Z → demand_cv > 0.8  OR  intermittency_ratio > 0.30  (high variability / lumpy)

ABC (demand volume — from existing abc_vol in dim_dfu):
  A = high volume
  B = medium volume
  C = low volume
```

**Combined segment:** `abc_xyz_segment = abc_vol.upper() + xyz_class` → e.g., 'AX', 'BZ', 'CY'

---

## 9-Cell Policy Matrix

| Segment | Service Level | DOS Min | DOS Max | Policy Type | Description |
|---|---|---|---|---|---|
| **AX** | 99% | 14d | 28d | continuous_rop | Lean, high-frequency, tight SS |
| **AY** | 98% | 21d | 42d | continuous_rop | Moderate SS for variability |
| **AZ** | 97% | 28d | 56d | periodic_review | Large SS, close monitoring |
| **BX** | 97% | 21d | 45d | continuous_rop | Standard efficient management |
| **BY** | 95% | 30d | 60d | periodic_review | Balanced service vs. cost |
| **BZ** | 92% | 45d | 90d | periodic_review | Conservative buffer |
| **CX** | 95% | 30d | 60d | min_max | Infrequent, batch ordering |
| **CY** | 90% | 45d | 90d | min_max | Reduce excess focus |
| **CZ** | 85% | 60d | 120d | manual | Review for rationalization |

---

## Data Requirements

### New DDL: `mvp/demand/sql/031_add_xyz_classification.sql`

Adds to `dim_dfu`:
```sql
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS xyz_class             TEXT;
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS abc_xyz_segment       TEXT;     -- 'AX', 'BZ', etc.
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS abc_xyz_policy_id     TEXT;     -- FK to dim_replenishment_policy
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS abc_xyz_dos_min       NUMERIC(10,2);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS abc_xyz_dos_max       NUMERIC(10,2);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS abc_xyz_service_level NUMERIC(6,4);
ALTER TABLE dim_dfu ADD COLUMN IF NOT EXISTS abc_xyz_classified_ts TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_dim_dfu_xyz ON dim_dfu (xyz_class);
CREATE INDEX IF NOT EXISTS idx_dim_dfu_abc_xyz ON dim_dfu (abc_xyz_segment);
```

### Config Extension: `mvp/demand/config/replenishment_policy_config.yaml`

Add `abc_xyz_policies` section (extends IPfeature5):
```yaml
abc_xyz_policies:
  AX: {service_level: 0.99, dos_min: 14, dos_max: 28, policy_type: continuous_rop}
  AY: {service_level: 0.98, dos_min: 21, dos_max: 42, policy_type: continuous_rop}
  AZ: {service_level: 0.97, dos_min: 28, dos_max: 56, policy_type: periodic_review}
  BX: {service_level: 0.97, dos_min: 21, dos_max: 45, policy_type: continuous_rop}
  BY: {service_level: 0.95, dos_min: 30, dos_max: 60, policy_type: periodic_review}
  BZ: {service_level: 0.92, dos_min: 45, dos_max: 90, policy_type: periodic_review}
  CX: {service_level: 0.95, dos_min: 30, dos_max: 60, policy_type: min_max}
  CY: {service_level: 0.90, dos_min: 45, dos_max: 90, policy_type: min_max}
  CZ: {service_level: 0.85, dos_min: 60, dos_max: 120, policy_type: manual}
```

---

## API Endpoints

**Router:** `mvp/demand/api/routers/inv_planning.py`

```
GET /inv-planning/abc-xyz/matrix
  Response: {
    cells: [
      { segment: 'AX', count: int, avg_health_score: float, avg_dos: float,
        avg_ss_coverage: float, below_ss_pct: float, dos_min: float, dos_max: float }
      × 9 cells (AX through CZ)
    ]
  }
  Cache: max-age=300s

GET /inv-planning/abc-xyz/summary
  Response: {
    total_dfus: int,
    classified_count: int,
    unclassified_count: int,
    by_segment: {
      AX: { count, pct, avg_health_score, avg_ss_coverage },
      ...
    }
  }
  Cache: max-age=300s

GET /inv-planning/abc-xyz/detail
  Query params: segment (e.g. 'AZ'), item, location, limit, offset
  Response: {
    total: int,
    segment: str,
    policy_config: { service_level, dos_min, dos_max, policy_type },
    rows: [ {item_no, loc, abc_vol, xyz_class, demand_cv, demand_mean,
             variability_class, health_score, health_tier, ss_coverage,
             current_dos, abc_xyz_policy_id} ]
  }
  Cache: max-age=120s
```

---

## Frontend UI

### Panel: "ABC-XYZ Matrix" in `InvPlanningTab.tsx`

**Interactive 3×3 Heatmap Grid:**
```
        X (stable)    Y (moderate)    Z (lumpy)
A  |   [AX: n, h]  |  [AY: n, h]  |  [AZ: n, h]  |
B  |   [BX: n, h]  |  [BY: n, h]  |  [BZ: n, h]  |
C  |   [CX: n, h]  |  [CY: n, h]  |  [CZ: n, h]  |
```

Each cell shows:
- DFU count (large text)
- Avg health score (small text below)
- Background color: health score scale (green ≥ 80, yellow 60–79, orange 40–59, red < 40)
- Tooltip on hover: service_level target, dos_min/max, policy_type

**Click a cell → filters the detail table below to that segment**

**Detail Table (below matrix):**
- Active segment shown as pill badge (e.g., "AZ — 47 items")
- Columns: item, loc, demand_cv, health_score, health_tier, ss_coverage, current_dos, abc_xyz_policy_id
- Default sort: health_score ascending (worst first)

**Portfolio Distribution Bar:**
- Horizontal stacked bar showing distribution across 9 segments (% of total DFUs)
- CZ items highlighted (rationalization candidates)

---

## Backend Script

### `mvp/demand/scripts/classify_abc_xyz.py`

```python
# Algorithm:
# 1. Load replenishment_policy_config.yaml (abc_xyz_policies section)
# 2. Load dim_dfu: item_no (=dmdunit), loc, abc_vol, demand_cv, intermittency_ratio
#    (demand_cv populated by IPfeature1)
# 3. For each DFU:
#    a. Determine xyz_class:
#       IF demand_cv IS NULL: xyz_class = NULL (skip — no variability data)
#       ELIF demand_cv > 0.8 OR intermittency_ratio > 0.30: xyz_class = 'Z'
#       ELIF demand_cv > 0.3: xyz_class = 'Y'
#       ELSE: xyz_class = 'X'
#    b. abc_class = abc_vol (already 'A', 'B', or 'C')
#    c. IF abc_class IS NULL OR abc_class NOT IN ('A','B','C'): skip
#    d. abc_xyz_segment = abc_class + xyz_class   (e.g. 'AZ')
#    e. Look up policy config for segment → service_level, dos_min, dos_max
#    f. Look up matching policy_id from dim_replenishment_policy
#       (match by policy_type from abc_xyz_policies config)
# 4. UPDATE dim_dfu SET xyz_class=..., abc_xyz_segment=..., abc_xyz_policy_id=...,
#    abc_xyz_dos_min=..., abc_xyz_dos_max=..., abc_xyz_service_level=...,
#    abc_xyz_classified_ts=NOW()
#    WHERE dmdunit=... AND loc=...
# 5. Print summary: count per segment, unclassified count
```

**CLI Usage:**
```bash
uv run python scripts/classify_abc_xyz.py
uv run python scripts/classify_abc_xyz.py --dry-run
```

**Makefile Targets:**
```makefile
abc-xyz-schema:
	# apply sql/031_add_xyz_classification.sql

abc-xyz-classify:
	uv run python scripts/classify_abc_xyz.py

abc-xyz-all: abc-xyz-schema abc-xyz-classify
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `dim_dfu.demand_cv`, `intermittency_ratio` | IPfeature1 | XYZ classification input |
| `dim_dfu.abc_vol` | Existing | ABC class from source system |
| `dim_replenishment_policy` | IPfeature5 | policy_id lookup by policy_type |
| `mv_inventory_health_score` | IPfeature6 | avg_health_score per cell in matrix |

---

## Testing Requirements

### Backend Unit Tests: `mvp/demand/tests/unit/test_abc_xyz_classification.py`

Minimum 12 tests:
- CV=0.25 → xyz_class='X'
- CV=0.55 → xyz_class='Y'
- CV=1.2 → xyz_class='Z'
- Boundary: CV=0.30 → xyz_class='X' (< threshold → X)
- Boundary: CV=0.80 → xyz_class='Y' (< 0.8 threshold)
- Boundary: CV=0.80001 → xyz_class='Z'
- intermittency_ratio=0.35, CV=0.2 → xyz_class='Z' (intermittency overrides CV)
- abc_vol='A', xyz_class='Z' → abc_xyz_segment='AZ'
- abc_vol='C', xyz_class='X' → abc_xyz_segment='CX'
- abc_vol=NULL → skip (no segment assigned)
- demand_cv=NULL → xyz_class=NULL (no classification)
- All 9 segments correctly mapped to service_level from config

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_abc_xyz.py`

Minimum 8 tests:
- `GET /inv-planning/abc-xyz/matrix` → 9 cells in response
- All segments (AX through CZ) present in cells list
- `GET /inv-planning/abc-xyz/summary` → classified_count + unclassified_count = total_dfus
- `GET /inv-planning/abc-xyz/detail?segment=AZ` → all rows have abc_xyz_segment='AZ'
- `GET /inv-planning/abc-xyz/detail?segment=invalid` → 422 or empty result
- Cell avg_health_score is 0–100 range
- Pagination on detail endpoint

---

## Acceptance Criteria

- [ ] All 9 matrix cells populated with non-zero counts after `make abc-xyz-classify`
- [ ] `xyz_class` = 'Z' for all items with `demand_cv > 0.8` or `intermittency_ratio > 0.30`
- [ ] `abc_xyz_segment` always 2 characters (e.g., 'AX', 'CZ') — no NULL segments for classified items
- [ ] Matrix heatmap click filters detail table to selected segment
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/031_add_xyz_classification.sql` | Create |
| `mvp/demand/config/replenishment_policy_config.yaml` | Modify — add abc_xyz_policies section |
| `mvp/demand/scripts/classify_abc_xyz.py` | Create |
| `mvp/demand/api/routers/inv_planning.py` | Modify — add abc-xyz endpoints |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Modify — add ABC-XYZ Matrix panel |
| `mvp/demand/tests/unit/test_abc_xyz_classification.py` | Create |
| `mvp/demand/tests/api/test_inv_planning_abc_xyz.py` | Create |
| `mvp/demand/Makefile` | Modify — add abc-xyz-* targets |
| `docs/design-specs/IPfeature11.md` | Create (this file) |
