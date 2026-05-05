# Storyboard

> An exception-based planner workflow that detects supply chain anomalies using configurable rules, presents them as causal-chain cards with root cause and recommended action, and tracks planner decisions through acknowledgment to resolution.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | StoryboardTab |
| **Key Files** | `StoryboardTab.tsx`, `api/routers/storyboard.py`, `common/exception_engine.py`, `scripts/generate_storyboard_exceptions.py`, `config/operations/exception_config.yaml`, `sql/038_create_storyboard.sql` |

---

## Problem

Planners managing thousands of DFUs (Demand Forecast Units -- item + location combinations) cannot review every item daily. They need a prioritized work queue that surfaces the most urgent anomalies with enough context to act immediately. Without structured exception management, critical issues (stockouts, forecast blowouts, policy violations) get buried in dashboards and discovered too late.

---

## Solution

An `ExceptionEngine` evaluates every DFU against configurable threshold rules across six exception types. When a threshold is breached, the engine scores severity (critical, high, medium, low), constructs a causal chain explaining the root cause path, and generates a recommended action. Exceptions are deduplicated within a 7-day window to prevent alert fatigue. The frontend renders exceptions as prioritized cards that planners acknowledge (accept) and resolve with a recorded decision.

---

## How It Works

### Exception Types

| Type | What It Detects | Key Threshold |
|---|---|---|
| `stockout_risk` | DOS below lead time coverage | `dos < lead_time_days` |
| `excess_inventory` | DOS significantly above target | `dos > target_dos x excess_multiplier` |
| `forecast_degradation` | WAPE rising above tolerance | `wape > wape_threshold` |
| `bias_alert` | Persistent over- or under-forecast | `abs(bias_pct) > bias_threshold` |
| `policy_violation` | Inventory outside policy bounds | Order qty outside policy min/max |
| `supplier_delay` | Lead time exceeding expected range | `actual_lt > expected_lt x delay_multiplier` |

### Severity Scoring

Each exception type has independent thresholds per severity level:

| Severity | Color | Typical Trigger |
|---|---|---|
| Critical | Red | Immediate stockout risk or financial impact above $50K |
| High | Orange | Near-term risk requiring action this week |
| Medium | Yellow | Monitoring required, action within the month |
| Low | Blue | Informational, no immediate action needed |

### Causal Chain

Each exception card includes a visual causal chain showing how the anomaly propagates:

`[Trigger] -> [Root Cause] -> [Impact] -> [Recommended Action]`

Example: `[WAPE 62%] -> [Model bias +35%] -> [Excess 45 DOS] -> [Switch to CatBoost model, reduce safety stock]`

### Deduplication

The engine checks for existing open exceptions on the same DFU + exception type within the last 7 days. If a matching open exception exists, no new record is created -- this prevents alert fatigue from daily batch runs.

### Resolution Workflow

| Status | Meaning | Who Sets It |
|---|---|---|
| `open` | New exception, not yet reviewed | System (on creation) |
| `acknowledged` | Planner has seen it and accepts the finding | Planner |
| `resolved` | Planner has taken corrective action | Planner |
| `expired` | Exception auto-closed after threshold returns to normal | System (batch) |

---

## Data Model

| Table | Purpose | Key Columns |
|---|---|---|
| `fact_storyboard_exceptions` | Exception records | `id`, `exception_type`, `severity`, `item_id`, `loc`, `trigger_value`, `threshold_value`, `causal_chain` (JSONB), `recommendation`, `status`, `created_at`, `resolved_at` |

Six indexes support filtering by: severity, status, exception type, item-location, creation date, and the composite (status + severity) for the default sorted view.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/storyboard/exceptions` | Paginated exception list with severity, status, type filters |
| GET | `/storyboard/summary` | Aggregate counts by severity and type |
| GET | `/storyboard/exceptions/{id}` | Full exception detail including causal chain |
| PUT | `/storyboard/exceptions/{id}/acknowledge` | Mark exception as acknowledged |
| PUT | `/storyboard/exceptions/{id}/resolve` | Mark exception as resolved with optional notes |
| POST | `/storyboard/generate` | Trigger exception detection batch (returns 202) |

---

## Pipeline

| Step | Command | What It Does |
|---|---|---|
| Schema | `make storyboard-schema` | Creates `fact_storyboard_exceptions` table + indexes |
| Generate | `make storyboard-generate` | Run exception detection for all DFUs |
| Dry Run | `make storyboard-generate-dry` | Preview exceptions without writing |

The generation script (`scripts/generate_storyboard_exceptions.py`) accepts `--item` and `--loc` flags for single-DFU testing.

---

## Configuration

File: `config/operations/exception_config.yaml`

| Key | Purpose | Default |
|---|---|---|
| `stockout_risk.critical_dos_days` | DOS threshold for critical stockout risk | `7` |
| `excess_inventory.excess_multiplier` | DOS-to-target ratio for excess flag | `2.0` |
| `forecast_degradation.wape_threshold` | WAPE above this triggers an exception | `40` |
| `bias_alert.bias_pct_threshold` | Absolute bias percent threshold | `20` |
| `dedup_window_days` | Days to suppress duplicate exceptions | `7` |
| `severity_rules` | Per-type mapping from threshold breach to severity | 4 levels per type |

---

## Dependencies

| Dependency | Reason |
|---|---|
| Inventory snapshot (03-01) | Current DOS for stockout and excess detection |
| Forecast accuracy (`agg_forecast_monthly`) | WAPE and bias for forecast-related exceptions |
| Safety stock targets (03-03) | Target DOS for excess computation |
| Replenishment policies (03-04) | Policy bounds for violation detection |
| Lead time profiles (03-02) | Expected lead time for delay detection |

---

## See Also

- `06-ai-platform/01-ai-planning-agent.md` -- AI agent queries storyboard exceptions and generates deeper causal analysis
- `06-ai-platform/03-control-tower.md` -- exception counts roll up to control tower KPIs
- `03-inventory-planning/05-exception-queue.md` -- replenishment-specific exceptions (complementary to storyboard's cross-domain scope)
