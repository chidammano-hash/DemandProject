# Spec 02-27: AI Planner Forecast Value Add — 10-Month Walk-Forward Backtest

**Status:** Implemented (MVP) — 2026-05-18
**Domain:** Forecasting / AI Platform / Backtest Framework
**Source PRD:** [../PRD/PRD-ai-planner-fva-backtest.md](../PRD/PRD-ai-planner-fva-backtest.md) (design intent + open questions)
**Related specs:** [01-ai-planning-agent](../06-ai-platform/01-ai-planning-agent.md) · [03-backtest-framework](03-backtest-framework.md) · [05-advanced-backtest](05-advanced-backtest.md)

## Implemented artifacts

| Layer | Path |
|---|---|
| DDL | [sql/186_create_ai_fva_backtest.sql](../../../sql/186_create_ai_fva_backtest.sql) — 4 fact/audit tables + 4 MVs |
| Config | [config/ai/ai_planner_fva_backtest_config.yaml](../../../config/ai/ai_planner_fva_backtest_config.yaml) |
| LLM client | [common/ai/llm_client.py](../../../common/ai/llm_client.py) — Ollama / Anthropic / OpenAI / openai_compat |
| Recommender | [common/ai/fva_recommender.py](../../../common/ai/fva_recommender.py) — prompt, schema, guardrails, apply rules |
| Runner | [scripts/forecasting/run_ai_fva_backtest.py](../../../scripts/forecasting/run_ai_fva_backtest.py) — walk-forward CLI w/ `ThreadPoolExecutor` |
| API router | [api/routers/forecasting/ai_fva_backtest.py](../../../api/routers/forecasting/ai_fva_backtest.py) — `/ai-planner/fva-backtest/*` |
| FVA waterfall integration | [api/routers/forecasting/fva.py](../../../api/routers/forecasting/fva.py) — `ai_adjusted` stage populated from latest succeeded run |
| Frontend tab | [frontend/src/tabs/AiPlannerFvaTab.tsx](../../../frontend/src/tabs/AiPlannerFvaTab.tsx) |
| Frontend queries | [frontend/src/api/queries/ai-planner-fva-backtest.ts](../../../frontend/src/api/queries/ai-planner-fva-backtest.ts) |
| Tests | [tests/unit/test_fva_recommender.py](../../../tests/unit/test_fva_recommender.py) (17) · [frontend/src/tabs/__tests__/AiPlannerFvaTab.test.tsx](../../../frontend/src/tabs/__tests__/AiPlannerFvaTab.test.tsx) (3) |
| Make targets | `ai-fva-backtest-smoke`, `ai-fva-backtest`, `ai-fva-backtest-dry` |

The rest of this document is the original PRD content, retained as the design reference.

---

---

## 1. Problem Statement

The platform produces a champion forecast per DFU through the multi-model competition framework. The **AI Planner** ([common/ai/ai_planner.py](../../../common/ai/ai_planner.py), router at [api/routers/intelligence/ai_planner.py](../../../api/routers/intelligence/ai_planner.py)) reads context for a DFU — recent actuals, comparable products, customer/pricing/program shifts, anomalies — and emits **insights and recommendations** to demand planners.

Today there is no proof that those AI recommendations actually improve forecast accuracy. The existing FVA waterfall in [api/routers/forecasting/fva.py:11-17](../../../api/routers/forecasting/fva.py#L11-L17) explicitly reserves the `ai_adjusted` stage for "AI-assisted forecast interventions **once they are measured**" — currently `state="planned"`.

The commercial cost of this gap is significant:

- **Sales:** every prospect asks "does the AI actually move forecast accuracy?" — we cannot answer with a number.
- **Planner adoption:** planners distrust AI suggestions they cannot verify. Win-rate evidence accelerates trust.
- **Pricing defense:** the AI Planner is a paid add-on in the Enterprise edition (see [docs/monetization/02-pricing-and-packaging.md](../../monetization/02-pricing-and-packaging.md)). FVA is the value driver.
- **Model improvement:** without measured FVA per recommendation type, we cannot tell which AI behaviors to keep, tune, or kill.

We need a repeatable **10-month walk-forward backtest** that simulates the AI Planner making forecast adjustments month-by-month over a historical period, applies those adjustments, and measures the resulting accuracy vs. the baseline champion forecast. The output is a single defensible number: **AI Planner FVA = baseline-accuracy lift attributable to AI recommendations**, decomposed by DFU segment and recommendation type.

---

## 2. Goals & Non-Goals

### Goals

1. **Quantify AI Planner FVA** as a measurable accuracy lift over the champion baseline across a 10-month historical window.
2. **Walk-forward correctness** — at month T, the AI sees only data available at month T (no actuals for months > T, no models trained on data > T).
3. **Per-DFU and aggregate** reporting — overall accuracy lift, per-cluster, per-segment (intermittent/lumpy/smooth/erratic), per-recommendation-type, per-month.
4. **Win-rate visibility** — % of DFUs where the AI override improved accuracy, % where it degraded it, magnitude of each.
5. **Cost-aware execution** — the 10-month backtest should be runnable on a configurable DFU sample, not the full universe of 317K+ DFUs by default.
6. **Wire the result into the existing FVA waterfall** so the `ai_adjusted` stage becomes `state="actual"` with a real `accuracy_pct`.
7. **Reproducible** — re-running the backtest with the same inputs, same model snapshots, and same AI temperature settings yields the same numbers.

### Non-Goals

- Real-time / production AI override of forecasts. This is a backtest only. Production override is a separate feature (call it Phase 3 below) and requires planner approval workflow.
- Auto-apply rules in production. The backtest measures *what would have happened* if AI overrides had been applied; production policy is out of scope here.
- Re-training champion models inside the backtest loop. We use existing point-in-time champion forecasts already stored in `fact_candidate_forecast` / `fact_production_forecast` (see [02-forecasting/08-production-forecast.md](../02-forecasting/08-production-forecast.md)).
- Comparing AI vs. human planner overrides. The `planner_adjusted` stage in the FVA waterfall is a separate measurement effort; this PRD covers `ai_adjusted` only.

---

## 3. User Stories

| Persona | Story | Acceptance |
|---|---|---|
| VP Supply Chain | "Before we trust AI overrides in production, show me a 10-month backtest where the AI's recommendations would have improved forecast accuracy." | A single dashboard panel shows baseline vs. AI-adjusted WAPE/accuracy for the last 10 months, with the lift in bold. |
| Demand Planning Manager | "I want to see which DFU segments AI helps and which it hurts, so I know where to trust it." | Segment breakdown table: cluster × recommendation type, with win rate and average lift. |
| Data Scientist | "I need to know which recommendation types add value and which are noise so I can tune the AI prompt." | Per-recommendation-type rollup with sample size, win rate, mean lift, p-value vs. zero. |
| Skeptical Executive | "How do you know this isn't hindsight bias?" | Audit log per AI call shows the exact context window the AI saw (no future leakage), reproducible from a `scan_run_id`. |
| Sales Engineer | "I need a one-page accuracy-proof PDF to attach to the POV close-out." | Export endpoint that produces a PDF summarizing the 10-month FVA with the customer's own data. |

---

## 4. Functional Requirements

### 4.1 The 10-Month Walk-Forward Loop

For a configurable window of `N` months (default `N=10`), ending at a configurable `as_of_date` (default `current_month - 1`):

```
For each month T in [as_of - N + 1, as_of - N + 2, ..., as_of]:
    1. Snapshot DFU universe at month T
       — DFUs that existed at T (introduction_date <= T, not discontinued before T)
       — Sample down per backtest config (default: 5% stratified sample, min 500 DFUs)

    2. Restrict data to point-in-time at end of month T
       — sales actuals through month T (inclusive)
       — champion forecast for months T+1..T+H, generated using only data <= T
         (look up from fact_candidate_forecast where forecast_run_month = T)

    3. For each sampled DFU:
        a. Build AI context (recent actuals, comparables, anomalies, customer/price events)
           using only point-in-time data
        b. Call AIPlannerAgent.recommend_forecast_adjustment(item_id, loc, T, baseline_forecast)
        c. Persist recommendation to fact_ai_forecast_recommendation
        d. Apply recommendation deterministically to compute ai_adjusted_forecast
        e. Persist ai_adjusted_forecast to fact_ai_adjusted_forecast

    4. Mark month T as complete in ai_fva_backtest_run.month_progress
```

After all `N` months complete:

```
For each DFU and each evaluation horizon (h = 1..H):
    accuracy_baseline   = WAPE(champion_forecast,   actuals)  for forecast at lag h
    accuracy_ai         = WAPE(ai_adjusted_forecast, actuals) for forecast at lag h
    lift                = accuracy_ai - accuracy_baseline
```

**WAPE formula** (per [CLAUDE.md](../../../CLAUDE.md) "Formulas"): `100 - (100 * SUM(ABS(F-A)) / ABS(SUM(A)))`. Use the same formula consistently across baseline and AI-adjusted to avoid metric mismatch.

### 4.2 Recommendation Taxonomy

The AI Planner emits one of the following recommendations per DFU per month. The recommendation type is stored verbatim so we can rollup FVA by type.

| Code | Description | Apply rule |
|---|---|---|
| `KEEP` | No change recommended. AI explicitly endorses the baseline. | `ai_forecast = baseline_forecast` |
| `SCALE_UP` | Increase forecast by `pct` (5–50% guardrails) for the next `horizon` months. | `ai_forecast[t..t+horizon] = baseline * (1 + pct/100)` |
| `SCALE_DOWN` | Decrease forecast by `pct` (5–50% guardrails). | `ai_forecast[t..t+horizon] = baseline * (1 - pct/100)` |
| `REPLACE` | AI proposes a specific quantity per month. | `ai_forecast[t..t+horizon] = ai_proposed_qty` |
| `SHIFT_TIMING` | Move demand from one month to another (e.g. promotional pull-forward). | `ai_forecast[from] -= delta`, `ai_forecast[to] += delta` |
| `OVERRIDE_TO_BASELINE` | AI declines to recommend (low confidence). | Same as KEEP, but tagged separately for analysis. |

Each recommendation must include:
- `confidence` — 0..1 from the LLM
- `rationale` — short text explanation (stored for audit)
- `evidence_keys` — list of context elements the AI cited (comparable products, anomalies, customer events)
- `apply_horizon_months` — 1..6 (default 3)
- `model_version` — LLM model + prompt version

### 4.3 Point-in-Time Correctness (No Data Leakage)

Critical property. The backtest is worthless if the AI sees future data.

**Mechanisms:**

1. **Forecast snapshot lookup.** At month T, the baseline forecast must come from a row in `fact_candidate_forecast` whose `forecast_run_month <= T` (i.e. the forecast that *would have been* the production forecast at month T). If no such snapshot exists, regenerate by re-running the champion model on point-in-time training data — but the simpler path is to require historical snapshots exist.

2. **Actuals truncation.** All AI context queries (sales actuals, OOS, customer demand, etc.) must filter `WHERE startdate <= '{T}'`.

3. **Customer/program events.** Any "anomaly detected" or "comparable product" lookup the AI uses must be restricted to events known at month T. Events stored in `fact_event` need an `effective_date` and `discovery_date`; the latter is what the AI sees, not the former.

4. **No model retraining inside the loop.** The champion model snapshot used for the baseline forecast at month T must be the model that was promoted **at or before** month T. We do not re-train. We use the historical promotion record from `dim_model_promotion` (extend if needed).

5. **Audit log.** Every AI call writes its full context to `ai_planner_audit_log` keyed by `scan_run_id`. Reproducibility requires the same context produces the same recommendation (modulo LLM nondeterminism — see §10).

### 4.4 Sampling Strategy

Running AI on 317K DFUs × 10 months = 3.17M LLM calls. At ~$0.10/call this is $317K per backtest run. Unacceptable.

**Default:** stratified 5% sample, min 500 DFUs, max 10K DFUs. Strata:
- Cluster (per `dim_cluster_assignment`)
- Demand pattern (intermittent / lumpy / smooth / erratic — per `dim_sku.demand_pattern`)
- Volume tier (top 20% / middle 60% / bottom 20% by trailing 12-month revenue)

**Override modes:**
- `full` — every DFU. For tier-1 enterprise customers willing to pay for a full backtest.
- `targeted` — explicit DFU list (for investigating specific failures).
- `cluster:<id>` — every DFU in a given cluster.

Sample strategy is configurable per-run and stored on `ai_fva_backtest_run`.

### 4.5 Evaluation Horizon

For each forecast emitted at month T, we evaluate accuracy at lags 1..H months out. Default `H = 3` (next 3 months). This matches the planning execution lag (see [02-forecasting/14-execution-lag-filters.md](../02-forecasting/14-execution-lag-filters.md)).

Per-lag accuracy is reported separately because AI overrides may help short-horizon (1 month, recent demand correction) more than long-horizon (3 months, trend extrapolation).

### 4.6 Aggregation & Reporting

After the loop completes, materialize the following:

| Materialized view / table | Grain | Columns |
|---|---|---|
| `mv_ai_fva_overall` | run_id | baseline_wape, ai_wape, lift, n_dfus, n_recommendations, win_rate, avg_lift_winners, avg_loss_losers |
| `mv_ai_fva_by_segment` | run_id × cluster × demand_pattern × volume_tier | same metrics |
| `mv_ai_fva_by_recommendation_type` | run_id × recommendation_code | same metrics + avg_pct_change_applied, avg_confidence |
| `mv_ai_fva_by_month` | run_id × month | same metrics |
| `mv_ai_fva_by_dfu` | run_id × item_id × loc | per-DFU lift for drill-down |

These feed the FVA waterfall stage `ai_adjusted` in [api/routers/forecasting/fva.py](../../../api/routers/forecasting/fva.py).

---

## 5. Data Model

New tables. All under `sql/<next_seq>_ai_fva_backtest.sql`.

```sql
-- Run metadata
CREATE TABLE ai_fva_backtest_run (
    run_id              uuid PRIMARY KEY,
    started_at          timestamptz NOT NULL DEFAULT now(),
    completed_at        timestamptz,
    status              text NOT NULL CHECK (status IN ('running','succeeded','failed','cancelled')),
    window_months       int  NOT NULL DEFAULT 10,
    as_of_date          date NOT NULL,
    horizon_months      int  NOT NULL DEFAULT 3,
    sample_strategy     jsonb NOT NULL,         -- {mode: 'stratified', pct: 5, ...}
    ai_model            text NOT NULL,          -- 'claude-opus-4-7'
    prompt_version      text NOT NULL,          -- 'v1.2.0'
    apply_guardrails    jsonb NOT NULL,         -- {max_pct: 50, min_confidence: 0.6}
    n_dfus_sampled      int,
    n_recommendations   int,
    error_message       text,
    created_by          text,
    notes               text
);

-- One row per AI recommendation, keyed by run × DFU × month
CREATE TABLE fact_ai_forecast_recommendation (
    run_id              uuid REFERENCES ai_fva_backtest_run(run_id) ON DELETE CASCADE,
    item_id             text NOT NULL,
    loc                 text NOT NULL,
    forecast_run_month  date NOT NULL,
    recommendation_code text NOT NULL,         -- KEEP|SCALE_UP|SCALE_DOWN|REPLACE|SHIFT_TIMING|OVERRIDE_TO_BASELINE
    pct_change          numeric(6,2),          -- nullable
    proposed_qty        jsonb,                 -- nullable; for REPLACE / SHIFT_TIMING
    apply_horizon_months int NOT NULL DEFAULT 3,
    confidence          numeric(4,3) NOT NULL,
    rationale           text,
    evidence_keys       jsonb,
    ai_call_ms          int,
    ai_tokens_in        int,
    ai_tokens_out       int,
    PRIMARY KEY (run_id, item_id, loc, forecast_run_month)
);

-- One row per DFU × forecast month × lag, with both baseline and AI-adjusted qty
CREATE TABLE fact_ai_adjusted_forecast (
    run_id              uuid REFERENCES ai_fva_backtest_run(run_id) ON DELETE CASCADE,
    item_id             text NOT NULL,
    loc                 text NOT NULL,
    forecast_run_month  date NOT NULL,        -- month T when the recommendation was issued
    target_month        date NOT NULL,        -- month being forecast
    lag                 int  NOT NULL,        -- target_month - forecast_run_month, in months
    baseline_qty        numeric NOT NULL,
    ai_qty              numeric NOT NULL,
    actual_qty          numeric,              -- backfilled when actuals arrive (always available in backtest)
    PRIMARY KEY (run_id, item_id, loc, forecast_run_month, target_month)
);

-- Audit log of AI context (one row per AI call, separate from recommendation for size)
CREATE TABLE ai_planner_backtest_audit (
    run_id              uuid REFERENCES ai_fva_backtest_run(run_id) ON DELETE CASCADE,
    item_id             text NOT NULL,
    loc                 text NOT NULL,
    forecast_run_month  date NOT NULL,
    context_payload     jsonb NOT NULL,       -- everything the AI saw
    ai_response_raw     jsonb NOT NULL,       -- raw LLM JSON response
    PRIMARY KEY (run_id, item_id, loc, forecast_run_month)
);

CREATE INDEX ix_ai_adj_run_lag        ON fact_ai_adjusted_forecast (run_id, lag);
CREATE INDEX ix_ai_rec_run_code       ON fact_ai_forecast_recommendation (run_id, recommendation_code);
```

Materialized views (`mv_ai_fva_overall`, `mv_ai_fva_by_segment`, etc.) defined per §4.6.

---

## 6. API Surface

All new endpoints under the existing `/ai-planner` and `/fva` prefixes. Per [CLAUDE.md](../../../CLAUDE.md) router conventions: `APIRouter(prefix=...)`, `get_conn()` (no `Depends(_get_pool)`), `Depends(require_api_key)` on writes, `%s` placeholders only.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ai-planner/fva-backtest/runs` | Start a new 10-month backtest. Body: `{window_months, as_of_date, horizon_months, sample_strategy, apply_guardrails, notes}`. Returns 202 + `run_id`. Background job. |
| `GET` | `/ai-planner/fva-backtest/runs` | List runs (paginated). |
| `GET` | `/ai-planner/fva-backtest/runs/{run_id}` | Run metadata + status + month progress. |
| `GET` | `/ai-planner/fva-backtest/runs/{run_id}/summary` | Aggregate FVA: baseline vs. AI WAPE, lift, win rate. |
| `GET` | `/ai-planner/fva-backtest/runs/{run_id}/by-segment` | Segment-level rollup. |
| `GET` | `/ai-planner/fva-backtest/runs/{run_id}/by-recommendation-type` | Recommendation-type rollup. |
| `GET` | `/ai-planner/fva-backtest/runs/{run_id}/by-month` | Month-by-month breakdown over the 10-month window. |
| `GET` | `/ai-planner/fva-backtest/runs/{run_id}/dfus` | Per-DFU breakdown, paginated, filterable by lift sign / recommendation_code / cluster. |
| `GET` | `/ai-planner/fva-backtest/runs/{run_id}/dfus/{item_id}/{loc}` | Drill-down: per-DFU per-month, with recommendation rationale and evidence. |
| `GET` | `/ai-planner/fva-backtest/runs/{run_id}/export.pdf` | One-page accuracy-proof PDF for sales / customer use. |
| `DELETE` | `/ai-planner/fva-backtest/runs/{run_id}` | Cancel/delete (admin-only). |

The existing FVA waterfall (`GET /fva/waterfall`) is updated: when a `?run_id=<latest_succeeded>` is supplied (or by default, picking the most recent succeeded run), the `ai_adjusted` stage transitions from `state="planned"` to `state="actual"` with `accuracy_pct` populated from `mv_ai_fva_overall`.

---

## 7. UI Requirements

### 7.1 New Tab: AI Planner FVA Backtest

Sidebar location: under the existing AI Planner section. New tab `frontend/src/tabs/ai-planner-fva/`. Per [CLAUDE.md](../../../CLAUDE.md), tab files < 600 lines — split into sub-panels.

Sub-panels:

1. **Runs list** (`RunsListPanel.tsx`) — table of historical runs with status, window, lift, n_dfus.
2. **New run dialog** (`NewRunDialog.tsx`) — form to configure and launch a run. Window = 10 (editable), as_of_date, horizon, sample strategy, model, guardrails.
3. **Run summary** (`RunSummaryPanel.tsx`) — KPI cards: baseline WAPE, AI WAPE, lift, win rate. Lift number is the hero. Confidence interval if sample size allows.
4. **Segment heatmap** (`SegmentHeatmapPanel.tsx`) — cluster × demand_pattern matrix, cell colored by lift. Click → DFU drill-down.
5. **Recommendation type breakdown** (`RecommendationBreakdownPanel.tsx`) — bar chart per `recommendation_code` with win rate and lift, plus average confidence.
6. **Monthly trend** (`MonthlyTrendPanel.tsx`) — line chart of baseline vs. AI WAPE over the 10 months. Visualizes whether AI lift is consistent or noisy.
7. **DFU drill-down** (`DfuDrillPanel.tsx`) — per-DFU detail, including the AI's rationale text and evidence list. Shows the exact context the AI saw (audit-log payload).
8. **Export** — button to generate the one-page PDF.

All HTTP from frontend goes through `src/api/queries/aiPlannerFvaBacktest.ts` using `fetchJson` (per [CLAUDE.md](../../../CLAUDE.md) frontend rules). Add to the queries barrel and Vite proxy.

### 7.2 Update to FVA Waterfall Tab

Existing FVA waterfall tab gets a "Backtest Source" selector — pick a `run_id` to populate the `ai_adjusted` stage. Default = latest succeeded run.

---

## 8. Configuration

New file `config/ai/ai_planner_fva_backtest_config.yaml`. **Default provider is Ollama (local, $0)** — Anthropic and OpenAI are opt-in for higher-quality benchmark runs.

```yaml
# AI Planner FVA Backtest defaults — overridable per run via API.
defaults:
  window_months: 10                    # Walk-forward window length
  horizon_months: 3                    # Forecast lags to evaluate (1..H)
  as_of_offset_months: 1               # as_of_date = current_month - offset
  prompt_version: v1.0.0               # Prompt template version (audit / reproducibility)

# ─── Provider abstraction ───────────────────────────────────────────────────
# Provider selection drives where the LLM call goes and what credentials are
# required. The LLMClient (common/ai/llm_client.py) handles the routing.
#
# - ollama (default): local, $0, no network. Model must be pulled via `ollama pull`.
# - anthropic: paid API, best quality. Requires ANTHROPIC_API_KEY.
# - openai: paid API. Requires OPENAI_API_KEY.
# - openai_compat: any OpenAI-compatible endpoint (Together, Fireworks, DeepInfra,
#                  Groq). Requires LLM_BASE_URL and LLM_API_KEY env vars.
provider: ollama

# Per-provider model identifiers. Active one is selected by `provider` above.
models:
  ollama: qwen2.5:32b                  # Recommended primary for FVA on M-series Mac.
                                        # Fallbacks: llama3.1:8b (screening) or
                                        # llama3.3:70b (slower, slightly better quality).
  anthropic: claude-opus-4-7
  openai: gpt-4o
  openai_compat: meta-llama/Llama-3.3-70B-Instruct-Turbo  # e.g. on Together AI

# Per-provider API URLs (Ollama is loopback by default).
endpoints:
  ollama: http://localhost:11434/v1    # OpenAI-compatible Ollama endpoint
  openai_compat: ${LLM_BASE_URL}       # e.g. https://api.together.xyz/v1

# Optional two-stage routing: cheap screener filters likely-KEEP DFUs,
# expensive primary handles the rest. See PRD §4.4.
hybrid_routing:
  enabled: false                       # Off by default; turn on for cost optimization
  screener_provider: ollama
  screener_model: llama3.1:8b
  screener_keep_threshold: 0.70        # Confidence above which screener emits KEEP
                                        # without escalating to primary

sampling:
  default_mode: stratified             # stratified | full | targeted | cluster
  stratified:
    pct: 5                             # Target % of DFU universe
    min_dfus: 500                      # Floor — don't go below this even if pct says less
    max_dfus: 10000                    # Cap — protect against runaway cost
    strata:                            # Stratification keys
      - cluster
      - demand_pattern
      - volume_tier

apply_guardrails:
  max_abs_pct_change: 50               # Reject AI recommendations outside [-50%, +50%]
  min_confidence: 0.60                 # Below this → treat as KEEP, log as low_confidence
  require_evidence: true               # Reject if evidence_keys is empty
  reject_on_horizon_overflow: true     # If apply_horizon > backtest horizon, reject

cost_controls:
  max_concurrent_ai_calls: 4           # Bound LLM concurrency (per-provider rate limit)
  per_call_timeout_seconds: 60         # Ollama 32B can be slower than paid APIs
  per_run_max_cost_usd: 1500           # Hard stop for paid providers; ignored for ollama
  # Per-provider cost estimates (USD per call) — drives pre-flight estimate.
  per_call_estimated_cost_usd:
    ollama: 0.0                        # Local — no API spend
    anthropic: 0.11                    # Opus 4.7 with prompt caching
    openai: 0.05                       # GPT-4o
    openai_compat: 0.005               # Together / Fireworks Llama 70B

evaluation:
  metric: wape                         # wape | mape | bias — wape is canonical (CLAUDE.md)
  per_lag_breakdown: true              # Report accuracy at each lag separately
  exclude_zero_actuals: false          # Whether DFUs with zero actuals in window count

# ─── Ollama-specific runtime knobs ──────────────────────────────────────────
ollama:
  keep_alive: 24h                      # Keep model loaded across long backtest runs
                                        # (default Ollama unloads after 5 min idle)
  request_timeout_seconds: 120         # Per-request timeout for slow local inference
  num_ctx: 8192                        # Context window — fits ~5K input + room to spare
  temperature: 0.0                     # Deterministic for reproducibility
```

Per [CLAUDE.md](../../../CLAUDE.md), every key has an inline comment. Config loaded via `load_config("ai_planner_fva_backtest_config")` from `common.core.utils`.

**Required environment variables (only needed for paid providers):**

```bash
# Set in .env — only needed when provider != ollama
ANTHROPIC_API_KEY=sk-ant-...           # Required when provider=anthropic
OPENAI_API_KEY=sk-...                  # Required when provider=openai
LLM_BASE_URL=https://api.together.xyz/v1   # Required when provider=openai_compat
LLM_API_KEY=...                        # Required when provider=openai_compat
```

**Provider quick-switch examples:**

```yaml
# Local development (default — no API spend, runs on M-series Mac)
provider: ollama

# Quality benchmark for sales / case study
provider: anthropic

# Cheap hosted-OSS for production-scale runs
provider: openai_compat   # then export LLM_BASE_URL + LLM_API_KEY
```

---

## 9. Metrics & Success Criteria

The PRD itself is successful when:

| Metric | Target | How measured |
|---|---|---|
| Backtest completes for 10-month window on default 5% sample | < 4 hours | wall-clock per run |
| Point-in-time correctness | 0 leakage incidents | automated test: assert no actuals beyond month T appear in any AI context payload |
| Reproducibility | Same inputs → same recommendations (within LLM tolerance) | replay test with stored audit logs |
| FVA reporting | Baseline WAPE, AI WAPE, lift visible in UI within 30 seconds of run completion | UI smoke test |
| Cost control | Pre-flight estimate matches actual within ±20% | post-run cost vs. estimate audit |

The **product** is successful when:

| Metric | Bar | Why |
|---|---|---|
| Median DFU lift across 10 months | Positive (AI WAPE < baseline WAPE) | Otherwise the AI Planner has no measurable value |
| Win rate (DFUs improved) | > 55% | Otherwise the AI is no better than coin-flip |
| Aggregate accuracy lift | ≥ 1.5 percentage points WAPE | This is the threshold a planning manager treats as material; below it, noise |
| Win rate × lift on `SCALE_UP` and `SCALE_DOWN` recommendations | > 60% with > 2pp average lift | These are the highest-volume recommendation types and must defend themselves |
| Variance across the 10 months | Lift standard deviation < lift mean | Indicates consistent value, not one-month noise |

If the product fails the bar, that itself is a valuable signal — it tells us the AI Planner needs prompt or context tuning before commercial use.

---

## 10. Risks & Edge Cases

1. **Hindsight bias / data leakage.** The single biggest threat. Mitigation: §4.3 mechanisms + an automated leakage-detection test that injects future actuals into a context payload and asserts the recommendation does not change.
2. **LLM nondeterminism.** Even at temperature=0 the same prompt can yield different output across model versions or providers. Mitigation: pin `ai_model` and `prompt_version` per run; store both in `ai_fva_backtest_run`. Reproducibility test allows ±5% rationale difference.
3. **Cost overrun.** Mitigation: pre-flight cost estimate based on sample size; hard stop at `per_run_max_cost_usd`; sampling defaults conservative.
4. **DFUs with no historical champion forecast snapshot.** Some DFUs may not have a `fact_candidate_forecast` row for month T. Mitigation: skip and tag in `ai_fva_backtest_run.notes`. Do not regenerate forecasts inside the loop in v1 (deferred to phase 2).
5. **Confounding from concurrent model retraining.** If the production champion model changed mid-window (e.g. month T-5 we promoted a new model), the backtest must use the model **as it was at month T**. Requires `dim_model_promotion` extension if not already point-in-time queryable.
6. **AI declines to recommend** (low confidence, missing context). Treated as `KEEP` with `low_confidence` tag. Do not exclude — track separately so we can measure decline rate.
7. **Recommendations that violate guardrails** (e.g. SCALE_UP 200%). Per `apply_guardrails`, clipped or rejected. Rejection is logged so we can see how often the AI proposes out-of-bounds changes (a prompt-tuning signal).
8. **DFU launches mid-window.** A DFU that didn't exist at month T-9 should not be in the sample for that month. Sampling logic checks `introduction_date <= forecast_run_month`.
9. **Cluster reassignments mid-window.** Clusters can shift over time. Use the cluster assignment **as of `forecast_run_month`** for the per-DFU context the AI sees.
10. **Very short / new product histories.** Cold-start DFUs (< 3 months history per [CLAUDE.md](../../../CLAUDE.md) `cold_start_min_months`) may produce noisy AI recommendations. Excluded by default; configurable via sample strategy.

---

## 11. Open Questions

1. **Should the backtest evaluate against `production_forecast` (post-promotion) or `candidate_forecast` (champion at run time)?** Recommendation: candidate, because that's what the AI would have been adjusting in real time. Confirm with planning team.
2. **Per-customer analytics scope.** Should the backtest run per `customer_group` (current DFU grain) or aggregate to item × loc? Current default = customer_group, matching DFU grain in `fact_external_forecast_monthly`.
3. **Multi-tenant readiness.** Once multi-tenancy lands ([04-product-readiness.md](../../monetization/04-product-readiness.md)), backtest runs must be tenant-scoped. Add `tenant_id` to all new tables in v1 even though it defaults to `'default'` today, to avoid migration pain later.
4. **Comparison to planner overrides.** Once human override data exists, do we want a unified "AI vs. Planner vs. Baseline" three-way FVA? Likely yes — but out of scope for v1.
5. **Replay vs. live AI re-generation.** A "replay" mode that re-runs the LLM on stored context payloads (from `ai_planner_backtest_audit`) without re-computing context could save 70%+ cost on re-runs. Phase 2.
6. **Statistical significance.** Should we report a confidence interval / p-value on the lift number? Recommend yes if sample size > 1000 DFUs, using a paired test (each DFU is its own control).

---

## 12. Phasing & Milestones

### Phase 1 — Backtest Engine (target 4–6 weeks, 1 senior eng + founder)

- [ ] DDL: 4 new tables + 5 MVs (sql/<next_seq>_ai_fva_backtest.sql)
- [ ] Domain spec doc (this PRD becomes spec `02-27` after sign-off)
- [ ] Extend `AIPlannerAgent` with `recommend_forecast_adjustment(item_id, loc, as_of_date, baseline_forecast)` method that returns a structured recommendation
- [ ] Backtest runner script `scripts/forecasting/run_ai_fva_backtest.py` with month walk-forward loop, sampling, point-in-time guards
- [ ] Recommendation application engine — deterministic apply rules per `recommendation_code`
- [ ] APScheduler job registration for runs (long jobs go to pg-queue per [CLAUDE.md](../../../CLAUDE.md) memory; this is an ad-hoc trigger, not recurring)
- [ ] Unit tests in `tests/unit/test_ai_fva_backtest_*.py` (point-in-time correctness, recommendation application, aggregation math)
- [ ] API endpoints per §6
- [ ] API tests in `tests/api/test_ai_fva_backtest.py`

### Phase 2 — UI & Reporting (target 2–3 weeks, 1 frontend eng)

- [ ] New tab `frontend/src/tabs/ai-planner-fva/` per §7.1
- [ ] Update FVA waterfall tab to populate `ai_adjusted` stage from latest run
- [ ] Tests for new tab (vitest) and an E2E test in `frontend/e2e/tests/navigation.spec.ts`
- [ ] PDF export endpoint (`/runs/{id}/export.pdf`)

### Phase 3 — Production Adoption (out of scope for this PRD; future spec)

- Real-time AI recommendations on the live forecast cycle
- Planner-approval workflow before applying
- Continuous FVA tracking on the production stream
- Auto-apply policy for high-confidence high-evidence recommendations

---

## 13. Documentation Updates Required (per CLAUDE.md "Docs in same commit" rule)

When this PRD's implementation lands, the following docs update in the same commit as the code:

- `docs/specs/02-forecasting/` — promote this PRD to a numbered spec (e.g. `27-ai-fva-backtest.md`)
- `docs/ARCHITECTURE.md` — add the 4 new tables and 5 new MVs to the catalog
- `docs/PLATFORM_GUIDE.md` — add the AI FVA Backtest tab to the feature list
- `docs/RUNBOOK.md` — add the new Make target (e.g. `make ai-fva-backtest`) and DB cleanup entries
- `docs/specs/06-ai-platform/01-ai-planning-agent.md` — add the new `recommend_forecast_adjustment` method
- `frontend/vite.config.ts` — proxy entry for `/ai-planner/fva-backtest`
- `frontend/src/api/queries/index.ts` — barrel entry
- `Makefile` — `db-truncate-data` and `refresh-mvs-tiered` updates for the 4 new tables and 5 new MVs

---

## 14. Glossary

- **DFU** — Demand Forecast Unit. Item × location × customer_group grain.
- **FVA** — Forecast Value Add. The accuracy improvement (or degradation) attributable to a downstream override of the baseline forecast.
- **WAPE** — Weighted Absolute Percent Error. `100 - (100 * SUM(|F-A|) / |SUM(A)|)`. Project-canonical accuracy metric.
- **Walk-forward** — backtest mechanic where the model/AI sees data only up to time T, predicts T+1..T+H, and we step T forward one period at a time.
- **Point-in-time** — data filtering that ensures no information from after time T is used in decisions made at time T.
- **Win rate** — % of DFUs where AI-adjusted forecast had lower WAPE than baseline.
- **Lift** — `ai_wape - baseline_wape`. Because WAPE in this codebase is the **accuracy form** (`100 - error%`, higher = better), a positive lift means AI improved the forecast. (Common SC textbooks use WAPE as an error measure where lift would be `baseline - ai`; this codebase consistently uses the accuracy form.)
