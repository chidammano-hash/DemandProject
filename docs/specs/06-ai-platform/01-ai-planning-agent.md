# AI Planning Agent

> A proactive exception work-queue powered by Claude that scans the portfolio for inventory and forecast anomalies, generates structured insights with causal reasoning, and presents them to planners for acceptance or resolution -- not a chatbot.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | AIPlannerTab |
| **Key Files** | `AIPlannerTab.tsx`, `api/routers/intelligence/ai_planner.py`, `common/ai/ai_planner.py`, `scripts/generate_ai_insights.py`, `config/ai/ai_planner_config.yaml`, `sql/036_create_ai_insights.sql` |

---

## Problem

With thousands of DFUs (Demand Forecast Units -- item + location combinations), planners cannot manually review every forecast error, inventory imbalance, or policy violation. The existing exception queue (Storyboard) catches rule-based anomalies, but it cannot explain why an anomaly occurred or recommend a specific corrective action. Planners need an AI system that proactively identifies problems, reasons about root causes across multiple data domains (forecast, inventory, policy, supplier), and suggests concrete next steps.

---

## Solution

An `AIPlannerAgent` class uses Claude's tool_use API to autonomously query the database through 10 registered tools (9 read-only SQL queries + 1 write tool). The agent loops through tool calls until it has enough context to generate a structured insight. A portfolio scan analyzes all DFUs exceeding configurable thresholds; a single-DFU mode provides deep analysis of one item-location. All insights are validated by a Pydantic model before database write. Circuit breakers (40 turns, 100K tokens) prevent runaway loops.

---

## How It Works

### Agent Loop

1. System prompt with few-shot examples establishes output quality expectations.
2. Agent receives either a single DFU context or a portfolio scan instruction.
3. Claude calls read-only SQL tools to gather data: forecast performance, inventory trends, similar DFUs, stockout history, EOQ context, policy assignments.
4. When the agent has sufficient evidence, it calls `create_insight` with a structured payload.
5. `CreateInsightInput` Pydantic validator enforces: valid `insight_type`, summary must contain at least one number, `financial_impact_estimate` capped at $10M.
6. Validated insights are written to `ai_insights`. Invalid payloads are rejected (logged, not written).
7. After a portfolio scan, the agent generates a portfolio memo summarizing key findings.

### Tools

| Tool | Type | Purpose |
|---|---|---|
| `get_dfu_full_context` | Read | Item + location + cluster + ABC class + policy + inventory position |
| `get_forecast_performance` | Read | WAPE, bias, accuracy for a DFU across models and lags |
| `get_portfolio_exceptions` | Read | Open exceptions from the storyboard |
| `compute_bias_trend` | Read | Rolling bias direction over recent months |
| `get_inventory_trend` | Read | Monthly on-hand, on-order, DOS trend |
| `get_eoq_context` | Read | EOQ parameters and current order quantities |
| `get_similar_dfus` | Read | DFUs in the same cluster with similar patterns |
| `check_stockout_history` | Read | Past stockout events for a DFU |
| `get_portfolio_health_summary` | Read | Aggregate portfolio KPIs (total DFUs, exception counts, avg WAPE) |
| `create_insight` | Write | Validated insight creation with Pydantic enforcement |

### Insight Types

| Type | Trigger Condition |
|---|---|
| `forecast_degradation` | WAPE exceeds threshold or rising bias trend |
| `stockout_risk` | DOS (Days of Supply) below lead time coverage |
| `excess_inventory` | DOS significantly above target, tying up capital |
| `policy_mismatch` | Replenishment policy misaligned with demand pattern |
| `supplier_risk` | Supplier lead time variability or delivery issues |

### Circuit Breakers

| Guard | Limit | On Breach |
|---|---|---|
| Max turns | 40 tool-call rounds | Loop terminates, partial results saved |
| Token budget | 100,000 tokens per run | Loop terminates, partial results saved |
| Financial cap | $10M per insight | Insight rejected by Pydantic validator |
| Summary validation | Must contain a digit | Insight rejected (prevents vague summaries) |

---

## Data Model

| Table | Purpose | Key Columns |
|---|---|---|
| `ai_insights` | Generated insights | `id`, `scan_run_id`, `insight_type`, `item_id`, `loc`, `severity`, `summary`, `recommendation`, `financial_impact_estimate`, `status`, `created_at` |
| `ai_planning_memos` | Portfolio narrative summaries | `id`, `scan_run_id`, `period`, `memo_text`, `model_version`, `created_at` |
| `ai_call_log` | Per-turn observability | `id`, `scan_run_id`, `turn`, `model`, `prompt_tokens`, `completion_tokens`, `tool_name`, `tool_latency_ms`, `created_at` |
| `ai_recommendation_outcomes` | Planner decision tracking | `id`, `insight_id`, `action_taken`, `metric_before_wape`, `metric_before_dos`, `outcome_check_due_at`, `outcome_label` |

---

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/ai-planner/portfolio-scan` | Trigger async portfolio scan (returns 202) |
| POST | `/ai-planner/analyze` | Analyze a single DFU (synchronous) |
| GET | `/ai-planner/insights` | List insights with filtering by severity, status, type |
| PUT | `/ai-planner/insights/{id}/status` | Accept or resolve an insight; optionally records `action_taken` |
| GET | `/ai-planner/memos` | List portfolio memos |
| GET | `/ai-planner/metrics` | Per-model token usage, latency, and error aggregates from `ai_call_log` |

Portfolio scan runs in a background thread via `_executor.submit()`. The frontend polls for new insights after triggering a scan.

---

## Pipeline

| Step | Command | What It Does |
|---|---|---|
| Schema | `make ai-insights-schema` | Creates all 4 tables (insights, memos, call_log, outcomes) |
| Scan | `make ai-insights-scan` | Run full portfolio AI scan |
| Single DFU | `make ai-insights-sku ITEM=100320 LOC=1401-BULK` | Analyze one DFU |
| Full | `make ai-insights-all` | Schema + scan |

---

## Configuration

File: `config/ai/ai_planner_config.yaml`

| Key | Purpose | Default |
|---|---|---|
| `model` | Claude model to use | `claude-opus-4-6` |
| `thresholds.dos_critical` | DOS below this triggers stockout risk | `14` |
| `thresholds.wape_high` | WAPE above this triggers forecast degradation | `40` |
| `thresholds.bias_pct` | Absolute bias above this triggers investigation | `20` |
| `severity_rules` | Mapping from threshold breaches to severity levels | 3 levels: critical, high, medium |
| `max_turns` | Circuit breaker: max tool-call rounds | `40` |
| `token_budget` | Circuit breaker: max tokens per run | `100000` |

---

## Dependencies

| Dependency | Reason |
|---|---|
| `anthropic>=0.40.0` | Claude tool_use API client |
| Forecast data (`fact_external_forecast_monthly`) | WAPE and bias computation |
| Inventory data (`agg_inventory_monthly`) | DOS, on-hand, trend analysis |
| Safety stock targets (03-03) | Excess/shortage detection |
| Storyboard exceptions (06-04) | Portfolio exception context |
| Cluster assignments (`dim_sku.cluster_assignment`) | Similar DFU lookup |

---

## See Also

- `06-ai-platform/04-storyboard.md` -- rule-based exception queue that complements AI insights
- `06-ai-platform/03-control-tower.md` -- aggregated KPIs that the agent queries
- `03-inventory-planning/03-safety-stock.md` -- safety stock targets used for stockout/excess detection
- `02-forecasting/07-champion-selection.md` -- model performance data queried by the agent
