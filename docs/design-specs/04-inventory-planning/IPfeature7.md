# IPfeature7 — Exception Queue & Replenishment Recommendations

## EPIC
InventoryPlanning

## Status
Implemented — 749 backend tests, 253 frontend tests

## Priority
P2 — Should Have

## Effort
L (Large)

## Expert Perspectives
- **Supply Chain Control Tower Expert** (lead) — exception triage, severity routing
- **UI/UX Expert** — exception queue UX, inline acknowledge, action states
- **Inventory Planning Expert** — recommendation logic, urgency calculation

---

## Problem Statement

IPfeature3–6 compute SS targets, EOQ, and health scores. But health scores don't tell a planner *what to do*. A score of 25/100 is alarming — but the planner still needs to know:

- **What kind of problem is it?** (Below ROP? Below SS? Excess? Zero-velocity?)
- **What action is recommended?** (Order X units by date D)
- **How urgent is it?** (Can I wait, or does stock-out happen within lead time?)
- **Was it already acknowledged?** (Is someone already handling it?)

Without an exception queue, every planner has to manually scan the detail table to find problems — an impossible task at scale.

---

## User Story

> As an inventory planner, I want a prioritized exception queue that tells me exactly which items need action, what action to take ("order 219 units of item 100320 at location 1401-BULK by 2026-03-10"), and tracks whether the exception has been acknowledged and resolved — so I can manage 10,000 SKUs without missing critical replenishments.

---

## Business Value

- Transforms the system from **descriptive** to **prescriptive** — the defining upgrade of this EPIC
- Reduces time-to-action for planners from hours of manual triage to minutes of queue processing
- Creates an auditable record of exception detection → acknowledgment → resolution
- Feeds IPfeature15 (Control Tower) with live exception counts

---

## Exception Types

| Type | Trigger Condition | Severity |
|---|---|---|
| `below_rop` | `current_qty ≤ reorder_point` AND `current_qty > ss_combined` | `high` |
| `below_rop_critical` | `current_qty ≤ reorder_point` AND `current_qty ≤ ss_combined` | `critical` |
| `below_ss` | `current_qty ≤ ss_combined` AND `current_qty > 0` | `high` if coverage>0.5, else `critical` |
| `stockout` | `current_qty ≤ 0` | `critical` |
| `excess` | `current_dos > target_dos_max × 1.5` | `medium` if dos<180d, else `low` |
| `zero_velocity` | `avg_daily_sls = 0` AND `eom_qty_on_hand > 0` | `low` |

**Severity levels:** `critical` > `high` > `medium` > `low`

---

## Recommendation Logic

```
For below_rop / below_ss / stockout exceptions:
  recommended_order_qty = max(effective_eoq, ss_combined - current_qty + effective_eoq/2)
  Capped at: max_eoq_months_supply × demand_mean_monthly

  recommended_order_by:
    critical: TODAY (immediate)
    high:     TODAY + review_cycle_days (from policy assignment, default 7)
    medium:   TODAY + review_cycle_days × 2

  expected_receipt_date = recommended_order_by + lt_mean_days

For excess exceptions:
  recommended_order_qty = 0 (no order; suggest return or reallocate)
  recommended_order_by = NULL

For zero_velocity exceptions:
  recommended_order_qty = 0 (no order; flag for review / write-off)
  recommended_order_by = NULL
```

---

## Deduplication Rule

An exception is NOT generated if:
- Same item_no + loc + exception_type already has an open (`status='open'`) exception created within the last 7 days

This prevents flooding the queue with the same item re-detected on every daily run.

---

## Data Requirements

### New DDL: `mvp/demand/sql/027_create_replenishment_exceptions.sql`

```sql
CREATE TABLE IF NOT EXISTS fact_replenishment_exceptions (
    exception_sk              BIGSERIAL PRIMARY KEY,
    exception_id              TEXT UNIQUE NOT NULL DEFAULT gen_random_uuid()::TEXT,
    item_no                   TEXT NOT NULL,
    loc                       TEXT NOT NULL,
    exception_date            DATE NOT NULL,
    exception_type            TEXT NOT NULL,
    severity                  TEXT NOT NULL CHECK (severity IN ('critical','high','medium','low')),
    -- Current state snapshot (at detection time)
    current_qty_on_hand       NUMERIC(15,4),
    current_dos               NUMERIC(10,2),
    ss_combined               NUMERIC(15,4),
    reorder_point             NUMERIC(15,4),
    -- Recommendation
    recommended_order_qty     NUMERIC(15,4),
    recommended_order_by      DATE,
    expected_receipt_date     DATE,
    estimated_order_value     NUMERIC(12,2),   -- recommended_order_qty × unit_cost
    -- Context
    policy_id                 TEXT,
    lead_time_mean_days       NUMERIC(10,2),
    -- Workflow
    status                    TEXT NOT NULL DEFAULT 'open'
                              CHECK (status IN ('open','acknowledged','ordered','resolved')),
    acknowledged_by           TEXT,
    acknowledged_ts           TIMESTAMPTZ,
    ordered_ts                TIMESTAMPTZ,
    resolved_ts               TIMESTAMPTZ,
    notes                     TEXT,
    load_ts                   TIMESTAMPTZ DEFAULT NOW(),
    modified_ts               TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_exceptions_item_loc   ON fact_replenishment_exceptions (item_no, loc);
CREATE INDEX IF NOT EXISTS idx_exceptions_type       ON fact_replenishment_exceptions (exception_type);
CREATE INDEX IF NOT EXISTS idx_exceptions_severity   ON fact_replenishment_exceptions (severity);
CREATE INDEX IF NOT EXISTS idx_exceptions_status     ON fact_replenishment_exceptions (status);
CREATE INDEX IF NOT EXISTS idx_exceptions_open_crit  ON fact_replenishment_exceptions (severity, exception_date)
    WHERE status = 'open' AND severity = 'critical';
CREATE INDEX IF NOT EXISTS idx_exceptions_date       ON fact_replenishment_exceptions (exception_date DESC);
```

---

## API Endpoints

**Router:** `mvp/demand/api/routers/inv_planning.py`

```
GET /inv-planning/exceptions
  Query params: exception_type, severity, status (default: open), item, location,
                abc_vol, cluster_assignment, limit, offset,
                sort_by (severity | exception_date | recommended_order_by | ss_gap), sort_dir
  Response: {
    total: int,
    rows: [ {exception_id, item_no, loc, exception_date, exception_type, severity,
             current_qty_on_hand, current_dos, ss_combined, reorder_point,
             recommended_order_qty, recommended_order_by, expected_receipt_date,
             estimated_order_value, policy_id, status, acknowledged_by, notes} ]
  }
  Cache: max-age=60s (short — exception status changes frequently)

GET /inv-planning/exceptions/summary
  Query params: status (default: open)
  Response: {
    open_count: int,
    by_type: { below_rop: int, below_ss: int, excess: int, zero_velocity: int, stockout: int },
    by_severity: { critical: int, high: int, medium: int, low: int },
    total_recommended_order_value: float,
    oldest_open_days: int
  }
  Cache: max-age=60s

PUT /inv-planning/exceptions/{exception_id}/acknowledge
  Auth: require_api_key
  Body: { acknowledged_by: str, notes: str (optional) }
  Response: updated exception row

PUT /inv-planning/exceptions/{exception_id}/status
  Auth: require_api_key
  Body: { status: 'ordered' | 'resolved', notes: str (optional) }
  Response: updated exception row

POST /inv-planning/exceptions/generate
  Auth: require_api_key
  Body: {} (no params — runs full scan)
  Response: {
    generated_count: int,
    skipped_dedup: int,
    by_type: { below_rop: int, below_ss: int, excess: int, zero_velocity: int }
  }
```

---

## Frontend UI

### Panel: "Exception Queue" in `InvPlanningTab.tsx`

**KPI Cards (row of 4):**
| Card | Value | Color |
|---|---|---|
| Total Open Exceptions | count | red if >50, amber if >10 |
| Critical | count of severity='critical' | always red if >0 |
| High | count of severity='high' | amber |
| Recommended Order Value | sum(estimated_order_value) | blue |

**Filter Bar:**
- Exception type pills: all / below_rop / below_ss / excess / zero_velocity / stockout
- Severity pills: all / critical / high / medium / low
- Status toggle: open (default) / acknowledged / all
- Item + location text inputs

**Exception Table:**
- Default sort: severity (critical first), then exception_date ascending (oldest first)
- Columns: severity badge, item, loc, type, current_qty, ss_combined, recommended_order_qty, order_by_date, expected_receipt, status badge
- Row background: critical=red-50, high=amber-50, medium=yellow-50, low=neutral
- **Inline "Acknowledge" button** per open row → calls PUT acknowledge endpoint, status changes to acknowledged, row fades to gray
- **"Mark Ordered" button** for acknowledged rows
- **"Resolve" button** for ordered rows

**"Generate Exceptions" button** (admin):
- Calls POST /generate
- Shows spinner + result toast: "Generated 243 exceptions (47 critical, 128 high)"

---

## Backend Script

### `mvp/demand/scripts/generate_replenishment_exceptions.py`

```python
# Algorithm:
# 1. Load config: safety_stock_config.yaml, eoq_config.yaml
# 2. Load latest inventory position:
#    SELECT item_no, loc, eom_qty_on_hand, avg_daily_sls
#    FROM agg_inventory_monthly (most recent month per item-loc)
# 3. Load fact_safety_stock_targets: ss_combined, reorder_point, effective_eoq,
#    target_dos_max, unit_cost, demand_mean_monthly, avg_daily_demand
# 4. Load fact_dfu_policy_assignment: policy_id, review_cycle_days (from dim_replenishment_policy)
# 5. Load dim_item_lead_time_profile: lt_mean_days
# 6. Load existing open exceptions (last 7 days) for deduplication
#
# For each item-loc:
#   current_qty = eom_qty_on_hand
#   current_dos = current_qty / avg_daily_sls (NULL if zero sls)
#
#   # Detect exception type
#   IF current_qty <= 0:
#     exception_type = 'stockout', severity = 'critical'
#   ELIF current_qty <= ss_combined:
#     exception_type = 'below_ss'
#     severity = 'critical' if current_qty/ss_combined < 0.5 else 'high'
#   ELIF current_qty <= reorder_point:
#     exception_type = 'below_rop', severity = 'high'
#   ELIF current_dos > target_dos_max * 1.5 AND target_dos_max IS NOT NULL:
#     exception_type = 'excess'
#     severity = 'medium' if current_dos < 180 else 'low'
#   ELIF avg_daily_sls == 0 AND current_qty > 0:
#     exception_type = 'zero_velocity', severity = 'low'
#   ELSE:
#     CONTINUE (no exception)
#
#   # Deduplication check
#   IF (item_no, loc, exception_type) in existing_open_exceptions: CONTINUE
#
#   # Compute recommendation
#   IF exception_type in ('below_rop', 'below_ss', 'stockout'):
#     gap = max(0, ss_combined - current_qty)
#     recommended_order_qty = max(effective_eoq, gap + effective_eoq/2)
#     recommended_order_qty = min(recommended_order_qty, 6 * demand_mean_monthly)
#     IF severity == 'critical': order_by = TODAY
#     ELSE: order_by = TODAY + review_cycle_days (default 7 if no policy)
#     expected_receipt = order_by + lt_mean_days
#     estimated_order_value = recommended_order_qty * unit_cost
#   ELSE:
#     recommended_order_qty = 0, order_by = NULL, receipt = NULL, value = 0
#
#   INSERT INTO fact_replenishment_exceptions ...
#
# 7. Print summary by type and severity
```

**CLI Usage:**
```bash
uv run python scripts/generate_replenishment_exceptions.py
uv run python scripts/generate_replenishment_exceptions.py --dry-run
```

---

## Makefile Targets

```makefile
exceptions-schema:
	# apply sql/027_create_replenishment_exceptions.sql

exceptions-generate:
	uv run python scripts/generate_replenishment_exceptions.py
```

---

## Dependencies

| Dependency | Type | Notes |
|---|---|---|
| `fact_safety_stock_targets` | IPfeature3 | ss_combined, reorder_point, effective_eoq |
| `fact_safety_stock_targets.unit_cost` | IPfeature4 | For estimated_order_value |
| `fact_dfu_policy_assignment` | IPfeature5 | review_cycle_days for order_by date |
| `dim_item_lead_time_profile` | IPfeature2 | lt_mean_days for expected_receipt |
| `agg_inventory_monthly` | Existing | Current position |

---

## Testing Requirements

### Backend Unit Tests: `mvp/demand/tests/unit/test_exception_generation.py`

Minimum 15 tests:
- `current_qty=0` → exception_type='stockout', severity='critical'
- `current_qty < ss_combined × 0.5` → severity='critical'
- `current_qty < ss_combined, current_qty > ss_combined × 0.5` → severity='high'
- `ss_combined < current_qty ≤ reorder_point` → exception_type='below_rop', severity='high'
- `current_dos = 200, target_dos_max = 60` → exception_type='excess', severity='low'
- `current_dos = 100, target_dos_max = 60` → exception_type='excess', severity='medium'
- `avg_daily_sls=0, current_qty>0` → exception_type='zero_velocity', severity='low'
- No exception: `ss_combined < current_qty ≤ target_dos_max × daily` → skip
- Deduplication: same item-loc-type already open → NOT generated
- Order qty: max(effective_eoq, gap + eoq/2) formula verified
- Critical order_by = TODAY
- High order_by = TODAY + review_cycle_days
- expected_receipt = order_by + lt_mean_days

### Backend API Tests: `mvp/demand/tests/api/test_inv_planning_exceptions.py`

Minimum 12 tests:
- `GET /inv-planning/exceptions` → 200 OK, rows with exception_id
- `GET /inv-planning/exceptions?severity=critical` → all rows have severity='critical'
- `GET /inv-planning/exceptions?status=open` → all rows have status='open'
- `GET /inv-planning/exceptions/summary` → by_severity has 4 keys, open_count ≥ 0
- `PUT /inv-planning/exceptions/{id}/acknowledge` without auth → 403
- `PUT /inv-planning/exceptions/{id}/acknowledge` with auth → 200, status='acknowledged'
- `PUT /inv-planning/exceptions/{id}/status` body={status:'resolved'} → resolved_ts set
- `POST /inv-planning/exceptions/generate` without auth → 403
- `POST /inv-planning/exceptions/generate` with auth → {generated_count, by_type}
- Sort by severity → critical rows first
- Pagination: limit=5 → ≤5 rows with correct total

---

## Acceptance Criteria

- [ ] Items with `current_qty ≤ reorder_point` have `status='open'` exception of type `below_rop`
- [ ] Deduplication: no duplicate open exception for same item-loc-type within 7 days
- [ ] `recommended_order_qty ≥ effective_eoq` for replenishment exceptions always
- [ ] `expected_receipt_date = recommended_order_by + lt_mean_days`
- [ ] Inline Acknowledge button changes status and visually grays out row
- [ ] `make test-all` passes

---

## Files Created / Modified

| File | Action |
|---|---|
| `mvp/demand/sql/027_create_replenishment_exceptions.sql` | Create |
| `mvp/demand/scripts/generate_replenishment_exceptions.py` | Create |
| `mvp/demand/api/routers/inv_planning.py` | Modify — add exception endpoints |
| `mvp/demand/frontend/src/tabs/InvPlanningTab.tsx` | Modify — add Exception Queue panel |
| `mvp/demand/tests/unit/test_exception_generation.py` | Create |
| `mvp/demand/tests/api/test_inv_planning_exceptions.py` | Create |
| `mvp/demand/Makefile` | Modify — add exceptions-* targets |
| `docs/design-specs/IPfeature7.md` | Create (this file) |
