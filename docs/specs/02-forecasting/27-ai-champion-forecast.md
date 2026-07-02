# 27 — AI Champion Forecast (interactive forward adjuster)

**Status:** Implemented (service, API, SQL, chart overlay) — 2026-06-16. The interactive adjuster **panel** was removed 2026-06-24 in favor of the SKU Chat agentic champion adjustment (§6).
**Supersedes:** the removed *AI Planner — FVA Backtest* (walk-forward backtest) **and** the earlier batch AI Champion pipeline. This feature keeps the LLM recommendation brain but is now **interactive and single-DFU** — there is no batch job, script, or Make target.

---

## 1. Problem & Intent

The promoted **champion** production forecast is the best statistical/ML model per DFU, but it only knows history. We want an optional AI pass that nudges the champion's **forward** forecast using judgment a planner would apply — customer-concentration shifts, recent trend breaks, product/market context — and persist the result as a **new forecast model**, `ai_champion`.

This is **forward-only** (it adjusts the future, never replays history) and **on-demand**. The adjustment is now initiated through the **SKU Chat agent** (approval-gated) rather than a dedicated panel; the planner reviews the staged recommendation and approves it, and the saved `ai_champion` forecast renders on the Item Analysis chart (§6).

## 2. Goals & Non-Goals

**Goals**
- One-click, per-DFU adjustment of the champion forward forecast from the Item Analysis tab.
- Use the item **and location** attributes (plus actuals, the champion forecast, and top customers) as LLM context.
- Provider choice per call: **Ollama** (local, $0), **Google Gemini**, **Anthropic (Opus)**, **OpenAI** — keys read server-side from `.env`.
- Preview-then-save: nothing persists until the planner clicks Save. Quantities are re-derived server-side from the authoritative champion baseline.

**Non-Goals**
- **No batch pipeline** — no portfolio-wide generation job, script, or Make target.
- No historical backtest / accuracy grading (forward-only).
- No automatic promotion into `fact_production_forecast` (ai_champion is its own model in its own table).

## 3. How It Works

`common/ai/champion_adjust_service.py` is the DB-aware orchestrator around the pure `common/ai/champion_adjuster.py` brain.

**Preview** — `adjust_dfu(item_id, loc, provider)`:
1. Resolve the latest champion `plan_version`.
2. Load this DFU's **forward** champion forecast (`fact_production_forecast`, `model_id='champion'`, `forecast_month >= plan_month`, capped to `forecast_window_months`).
3. Build a `DfuContext`: trailing actuals (`fact_sales_monthly`), the champion forward forecast, **item attributes** (`dim_sku`: brand, category, size, premise, region, supplier, ABC, cluster), **location attributes** (`dim_location`: site, state, primary demand location), top-K customers, and an optional **planner comment** (free text the planner typed for this DFU — a strong steer on intent, but it never overrides the numeric guardrails; fed to the prompt, never persisted on its own).
4. Call the LLM once (`recommend()`), apply guardrails (`apply_guardrails()`), apply the recommendation deterministically (`apply_recommendation()`).
5. Return a **preview** (champion vs AI per month, recommendation code, %Δ, confidence, rationale, evidence) — **no DB write**.

**Save** — `save_adjustment(item_id, loc, provider, recommendation)`:
- Re-applies guardrails to the echoed recommendation and **re-derives the quantities from the DB champion baseline** (the client never supplies quantities), then upserts `fact_ai_champion_forecast` under a get-or-create **interactive run** (`status='interactive'`, one per `plan_version`+`provider`).

Recommendation codes: `KEEP`, `SCALE_UP`, `SCALE_DOWN`, `REPLACE`, `SHIFT_TIMING`, `OVERRIDE_TO_BASELINE`. Guardrails downgrade low-confidence / no-evidence recommendations to `OVERRIDE_TO_BASELINE` (= champion) and clip `SCALE_*` magnitude. Any LLM/parse failure degrades to `KEEP` so the preview is always complete.

## 4. Data Model (`sql/190_create_ai_champion_forecast.sql`)

- **`ai_champion_run`** — run header: `run_id`, `plan_version`, `provider`, `ai_model`, `prompt_version`, `status` (`interactive` for the on-demand flow), `n_dfus`, `n_adjusted`, timestamps.
- **`fact_ai_champion_forecast`** — one row per (`plan_version`, `item_id`, `loc`, `forecast_month`): `champion_qty`, `ai_qty`, `model_id='ai_champion'`, `recommendation_code`, `pct_change` (per-month derived), `confidence`, `rationale`, `evidence_keys`. Unique index `(plan_version, item_id, loc, forecast_month)` → Save is an upsert.

A dedicated table (not `fact_production_forecast`) because that table's unique index has no `model_id`, so champion and ai_champion rows would collide.

## 5. API (`api/routers/forecasting/ai_champion.py`, prefix `/ai-champion`)

| Method | Path | Description |
|---|---|---|
| GET | `/ai-champion/forecast` | Latest **saved** adjustment for a DFU (`item_id` + optional `loc`, filtered in SQL) |
| POST | `/ai-champion/adjust` | Preview an LLM adjustment for one DFU (`require_api_key`) — no DB write |
| POST | `/ai-champion/save` | Persist a previewed adjustment (`require_api_key`); quantities re-derived server-side |

## 6. UI

> **2026-06-24 — the dedicated interactive adjuster panel was removed.** The `AiChampionItemPanel` card (provider dropdown + planner-comment textarea + **AI Adjust**/**Save**) is superseded by the **SKU Chat agentic champion adjustment** (approval-gated) — see [06-ai-platform/07-sku-chatbot](../06-ai-platform/07-sku-chatbot.md). The service, API, SQL, and the read-only chart overlay below all remain. The old `POST /ai-champion/adjust` + `/ai-champion/save` write endpoints stay live but are no longer called from the UI (the SKU Chat agent reaches the same `champion_adjust_service` engine via its staging tool).

**Chart overlay** — the `ai_champion` forward forecast is merged into the main Item Analysis forecast chart (`UnifiedChart`) as an amber dashed line alongside the champion/production/staging lines, and the recommendation rationale renders as a caption above the chart. It is fed by `fetchAiChampionSaved` (`GET /ai-champion/forecast`, query module `frontend/src/api/queries/ai-champion.ts`) — now read-only — and shows the **latest saved adjustment** for the DFU regardless of source (the SKU Chat agent's approved adjustments write to the same `fact_ai_champion_forecast` table). The line has a toggle pill (`AI Champion`, on by default) and is hidden when no saved row exists for the DFU.

## 7. Configuration (`config/ai/ai_champion_config.yaml`)

`provider` (default `ollama`, overridable per call), per-provider `models`/`endpoints` (incl. `google: gemini-2.0-flash`), `defaults` (source/target model_id, horizon, window, lookback, top_customers), `apply_guardrails`, and `cost_controls.per_call_timeout_seconds`. Loaded via `load_config("ai_champion_config")`; the `LLMClient` is built with `build_from_config()`.

## 8. Providers — keys and cost

Keys are read **server-side** from `.env`; the UI sends only the provider name.

- **Ollama** (default): local, free, no network. Pull the model first (`ollama pull llama3.1:8b`).
- **Google Gemini**: `GOOGLE_API_KEY`, via Gemini's OpenAI-compatible endpoint. Metered.
- **Anthropic (Opus)**: `ANTHROPIC_API_KEY`. Metered. **Not** covered by a Claude Code subscription (that covers only the CLI, not programmatic API calls).
- **OpenAI (GPT-4o)**: `OPENAI_API_KEY`. Metered.

## 9. Risks & Future

- **No leakage risk** (forward-only) — nothing historical to peek at.
- **LLM quality variance** — small local models occasionally emit off-schema JSON; that degrades to `KEEP` rather than failing.
- **FVA waterfall** — the `ai_adjusted` ladder stage stays *reserved/planned*: `ai_champion` is forward-only with no measurable actual overlap.
- **Future**: feed forward exogenous signals (`fact_event_calendar`, `fact_external_signal`) into the prompt; optional web research (leakage-safe since it's forward).

## 10. Reusable modules

- `common/ai/champion_adjuster.py` — recommendation schema, prompt (now includes item/location attributes), guardrails, deterministic apply (pure, no DB).
- `common/ai/champion_adjust_service.py` — single-DFU context build, `adjust_dfu`, `save_adjustment` (DB-aware).
- `common/ai/llm_client.py` — provider switch (Ollama / Google / Anthropic / OpenAI / openai_compat).
