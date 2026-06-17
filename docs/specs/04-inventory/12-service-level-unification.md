# 12 — Service-Level Target Unification

Single source of truth for service-level (SL) targets consumed by safety
stock, fill rate, S&OP gap analysis, and replenishment planning.

**Status:** Implemented (2026-04-23). **Supersedes:** scattered `_SL_TARGETS`
dicts and YAML-only targets.

---

## Problem

Before unification, three independent stores drove SL calculations:

| Source | Owner | Scope |
|---|---|---|
| `_SL_TARGETS` dict in [api/routers/operations/fill_rate.py](../../api/routers/operations/fill_rate.py) | Fill-rate reporting | Portfolio-wide, hardcoded |
| `service_levels_by_abc` in [config/shared_constants.yaml](../../config/shared_constants.yaml) | SS + replenishment | Portfolio-wide, YAML |
| `fact_service_level_targets` table (DDL [sql/051](../../sql/051_create_service_level_tracking.sql)) | Per-SKU overrides | SKU/class, DB |

They held the same numeric values today (A=0.98, B=0.95, C=0.90,
default=0.95), but drift was inevitable because no resolver enforced
consistency.

## Resolution

**[common/core/service_levels.py](../../common/core/service_levels.py)** is the
sole SL resolver. Priority order:

1. `fact_service_level_targets` row matching exact `(item_id, loc)`
2. `fact_service_level_targets` row for the `abc_class` where item/loc are NULL
3. YAML `service_levels_by_abc` in `shared_constants.yaml`
4. Hardcoded last-resort constants (matches YAML values)

`fact_service_level_targets` is the production authority. YAML remains as
a bootstrap / test fallback; scripts that cannot easily open a DB cursor
(e.g. CI-only linters) still work.

## Public API

```python
from common.core.service_levels import (
    load_sl_targets_by_abc,   # returns {"A": 0.98, "B": 0.95, ..., "default": 0.95}
    resolve_sl_target,        # per-SKU resolver with priority chain
)
```

- `load_sl_targets_by_abc(cursor=None)` — portfolio class defaults, DB
  overrides applied when a cursor is provided.
- `resolve_sl_target(abc_class, *, item_id, loc, cursor, targets_by_abc)` —
  resolves for a single SKU, honoring DB overrides.

Both helpers degrade gracefully: a missing table or bad row shape falls
back to YAML, never raises.

## Consumers updated

| File | Before | After |
|---|---|---|
| [api/routers/operations/fill_rate.py](../../api/routers/operations/fill_rate.py) | `_SL_TARGETS = {...}` | `load_sl_targets_by_abc(cursor=cur)` |
| [scripts/compute_safety_stock.py](../../scripts/compute_safety_stock.py) | YAML only | YAML + DB override via helper |

## Tests

- [tests/unit/test_service_levels.py](../../tests/unit/test_service_levels.py)
  — 8 tests covering YAML fallback, DB overrides, SKU-level precedence,
  transient-failure resilience, and hardcoded-fallback parity.
- Existing [test_fill_rate.py](../../tests/api/test_fill_rate.py) and
  [test_inv_planning_safety_stock.py](../../tests/api/test_inv_planning_safety_stock.py)
  continue to pass without changes — the helper is behavior-preserving for
  callers that don't supply DB overrides.

## Operational notes

- **Reading live targets:** `fact_service_level_targets` is authoritative.
  Operators upsert new targets via `PUT /analytics/service-level/targets`.
- **Editing YAML:** Still valid for the default fleet, but DB rows always
  win. Use YAML to change the baseline for all SKUs; use DB rows to
  override a specific ABC class or SKU.
- **Bootstrap:** Empty `fact_service_level_targets` means every SKU gets
  the YAML value for its ABC class — matches the pre-unification behavior.

## Follow-ups (not in this change)

- Hook `compute_replenishment_plan.py` into the helper (currently still
  reads `cfg["replenishment_plan"]["service_levels"]`, but that value
  transitively mirrors the YAML anchor — no functional drift).
- Wire the S&OP gap-analysis endpoints once Section 5.x lands a dedicated
  SOP router that reads SL targets directly.
