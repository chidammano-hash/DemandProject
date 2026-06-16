# 27 — AI Champion Forecast (forward-only AI adjuster)

**Status:** Implemented — 2026-06-16
**Supersedes:** the removed *AI Planner — FVA Backtest* (walk-forward backtest + accuracy grading). This feature keeps the LLM recommendation brain but drops all backtesting.

---

## 1. Problem & Intent

The promoted **champion** production forecast is the best statistical/ML model per DFU, but it only knows history. We want an optional AI pass that nudges the champion's **forward** forecast using judgment a planner would apply — customer concentration shifts, recent trend breaks, event notes — and persist the result as a **new forecast model**, `ai_champion`.

This is **forward-only**: it adjusts the future, it does not replay history and it does not measure accuracy (there are no actuals for future months). Providers are switchable so it can run with **Ollama (local, $0)** or **Anthropic Opus 4.7** (metered API).

## 2. Goals & Non-Goals

**Goals**
- Read the promoted champion forward forecast and produce an AI-adjusted forward forecast (`model_id='ai_champion'`).
- Deterministic, guard-railed application of a structured LLM recommendation.
- Run with no API spend (Ollama) or against Opus 4.7 for quality.
- Persist the adjustment + rationale for review.

**Non-Goals**
- No historical walk-forward backtest, no point-in-time replay, no accuracy/lift grading (that was the removed feature).
- No automatic promotion into `fact_production_forecast` (ai_champion is its own model, surfaced from its own table).
- No web research (a future option; see §9).

## 3. How It Works

`scripts/forecasting/generate_ai_champion_forecast.py` → `run_pipeline()`:

1. Resolve the champion `plan_version` (latest, or `--plan-version`).
2. Load, per DFU, the **forward** champion forecast (`fact_production_forecast`, `model_id='champion'`, `forecast_month >= plan_month`), capped to `forecast_window_months`.
3. Build a `DfuContext` (`common/ai/champion_adjuster.py`): trailing actuals (`fact_sales_monthly`), the champion forward forecast, `dim_sku` metadata (cluster, ABC), and top-K customers.
4. For each DFU (thread pool), call the LLM (`recommend()`), apply guardrails (`apply_guardrails()`), and apply the recommendation deterministically (`apply_recommendation()`).
5. Write per-DFU-month rows to `fact_ai_champion_forecast` and a run header to `ai_champion_run`.

Recommendation codes (unchanged from the recommender brain): `KEEP`, `SCALE_UP`, `SCALE_DOWN`, `REPLACE`, `SHIFT_TIMING`, `OVERRIDE_TO_BASELINE`. Guardrails downgrade low-confidence / no-evidence recommendations to `OVERRIDE_TO_BASELINE` (= champion) and clip `SCALE_*` magnitude. Any LLM/parse failure falls back to `KEEP` so every DFU still gets a complete `ai_champion` forecast.

## 4. Data Model (`sql/190_create_ai_champion_forecast.sql`)

- **`ai_champion_run`** — one row per run: `run_id`, `plan_version`, `provider`, `ai_model`, `prompt_version`, `status`, `n_dfus`, `n_adjusted`, `est_cost_usd`, timestamps.
- **`fact_ai_champion_forecast`** — one row per (`plan_version`, `item_id`, `loc`, `forecast_month`): `champion_qty` (baseline), `ai_qty` (the AI Champion forecast), `model_id='ai_champion'`, `recommendation_code`, `pct_change`, `confidence`, `rationale`, `evidence_keys`. Unique index `(plan_version, item_id, loc, forecast_month)`.

A dedicated table (not `fact_production_forecast`) because that table's unique index `(plan_version, item_id, loc, forecast_month)` has no `model_id`, so champion and ai_champion rows would collide. The dedicated table also co-locates the AI rationale with the adjusted quantity.

## 5. API (`api/routers/forecasting/ai_champion.py`, prefix `/ai-champion`)

| Method | Path | Description |
|---|---|---|
| GET | `/ai-champion/latest` | Latest run metadata + recommendation-code rollup |
| GET | `/ai-champion/forecast` | Per-DFU-month champion-vs-ai rows (filter `item_id`, `adjusted_only`) |
| POST | `/ai-champion/generate` | Submit a `generate_ai_champion` background job (`require_api_key`) → 202 + job_id |

## 6. Jobs & Make

- Job type **`generate_ai_champion`** (`common/services/job_registry.py` + `job_state.py`), group `forecast`. Params: `provider`, `plan_version`, `limit_dfus`.
- Make targets: `ai-champion` (full, Ollama), `ai-champion-smoke` (50 DFUs), `ai-champion-opus` (Opus 4.7), `ai-champion-dry` (plan only).

## 7. Configuration (`config/ai/ai_champion_config.yaml`)

`provider` (default `ollama`), per-provider `models`/`endpoints`, `defaults` (source/target model_id, horizon, window, lookback, top_customers), `apply_guardrails`, `cost_controls`, and `ollama` runtime knobs. Loaded via `load_config("ai_champion_config")`; the `LLMClient` is built with `build_from_config()`.

## 8. Providers — running without API spend

- **Ollama** (default): local, free, no network. `make ai-champion-smoke` runs 50 DFUs locally.
- **Anthropic Opus 4.7**: `make ai-champion-opus` (or `--provider anthropic`). Requires `ANTHROPIC_API_KEY` — these are **metered Anthropic API** calls and are **not** covered by a Claude Code Pro/Max subscription (a subscription only covers the Claude Code CLI itself, not a script's programmatic API calls). Use Ollama for zero-cost runs; use Opus 4.7 when you want the quality benchmark and accept the per-token cost (pre-flight cost estimate + `per_run_max_cost_usd` cap guard paid runs).

## 9. Risks & Future

- **No leakage risk** (forward-only) — unlike a backtest, there is nothing historical to peek at.
- **LLM quality variance** — small local models occasionally emit off-schema JSON; the pipeline degrades that DFU to KEEP rather than failing the run.
- **FVA waterfall** — the `ai_adjusted` ladder stage stays *reserved/planned*: `ai_champion` is forward-only with no measurable actual overlap, so it cannot feed the accuracy waterfall.
- **Future**: feed forward exogenous signals (`fact_event_calendar`, `fact_external_signal`) into `notes`; optional web research for the live override (leakage-safe since it's forward).

## 10. Reusable modules

- `common/ai/champion_adjuster.py` — recommendation schema, prompt, guardrails, deterministic apply (pure, no DB).
- `common/ai/llm_client.py` — provider switch (Ollama / Anthropic / OpenAI / openai_compat).
