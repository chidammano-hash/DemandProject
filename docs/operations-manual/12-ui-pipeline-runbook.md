# Section 12 — Running the Pipeline from the UI (Phases 5–10)

This runbook covers driving the **modeling → forecast → inventory → ops** pipeline
(everything from backtests onward) **from the web UI** instead of `make` targets.
It is the UI counterpart to the Make-based reset runbook (`RESET_RELOAD_2026_06.md`,
Phases 5–10).

> Verified against `common/services/job_registry.py`, `api/routers/core/jobs.py`,
> `api/routers/forecasting/backtest_management.py`, `frontend/src/api/queries/jobs.ts`,
> and `frontend/src/tabs/ModelTuningTab.tsx`. Exact on-screen button text may differ
> slightly by version — each control below is tied to the endpoint it calls so it is
> unambiguous.

---

## 12.0 The two UI surfaces

Everything is triggered from one of two tabs:

| Surface | Sidebar tab | What it drives |
|---|---|---|
| **Guided modeling flow** | **Model Tuning** (3 stages: **Backtest → Champion → Forecast**) | Phases 5–8: run backtests, load results, select/promote champion, train + generate + promote production forecasts |
| **Generic job runner** | **Jobs** (panels: Job Groups, Active Jobs, Pipeline Builder, Schedules, Job History) | Phase 9 (inventory, demand planning, ops), running many jobs at once, monitoring, chaining, and scheduling |

**Key facts that apply to everything below:**

- **All long steps run as background jobs** (pg-queue). Firing a control returns immediately
  with a job id; watch it in **Jobs → Active Jobs** (status `queued → running → completed/failed`)
  and read output in **Jobs → Job History → logs**.
- **Write actions require the API key.** The UI sends `X-API-Key` automatically once you're
  authenticated; for direct `curl` you must pass `-H "X-API-Key: $API_KEY"`.
- **Prerequisite:** API on `:8000` and UI on `:5173` running (`make api`, `make ui`). Data must
  already be loaded through Phase 4 (features + clustering) per the reset runbook.

---

## 12.1 Phase 5 — Backtests (Model Tuning → Backtest stage)

The Backtest stage lists every model in the roster. Each model has a **Run** control →
`POST /backtest-management/{model_id}/run` (writes; `?parallel=true` to let different model
families run concurrently — same family stays serial).

| Model (model_id) | Job type | Notes |
|---|---|---|
| `lgbm_cluster`, `catboost_cluster`, `xgboost_cluster` | `backtest_lgbm` / `_catboost` / `_xgboost` | tree models |
| `chronos`, `chronos_bolt`, `chronos2`, `chronos2_enriched`, `bolt_hierarchical` | `backtest_chronos*` / `backtest_bolt_hierarchical` | foundation — slow (chronos2/enriched multi-hour) |
| `seasonal_naive`, `rolling_mean` | `backtest_seasonal_naive` / `_rolling_mean` | statistical baselines (champion fallback) |
| `mstl` | `backtest_mstl` | statistical decomposition |
| `nhits`, `nbeats` | `backtest_nhits` / `_nbeats` | deep learning (needs the `dl` extra) |

**Running the whole roster from the UI** (no single "Run all" button): use the
**Jobs → Pipeline Builder** and add one step per backtest job type (`backtest_lgbm`,
`backtest_catboost`, … `backtest_nbeats`) → submit as one pipeline (`POST /jobs/pipeline`).
Or click **Run** on each model card. The foundation/DL backtests are the multi-hour
wall — fire them and monitor in Active Jobs.

---

## 12.2 Phase 6 — Load results + accuracy (Model Tuning → Backtest stage)

- **Load a model's predictions into the DB:** the **Load** control on the selected model →
  `POST /backtest-management/{model_id}/load` (writes predictions → `fact_candidate_forecast`).
  Job type `backtest_load_model`. Do this for every model you backtested.
- **Accuracy** is shown per model card (`GET /backtest-management/summary`); run history via
  `GET /backtest-management/{model_id}/runs`.
- **External ML extracts** (`ext_lgbm/cat/xg/best`) and the **accuracy-MV refresh** have **no
  dedicated UI button** — run `make load-ext-all` and `make refresh-accuracy-mvs`, or submit a
  `refresh_forecast_views` job from the Jobs tab.

---

## 12.3 Phase 7 — Champion (Model Tuning → Champion stage)

The Champion stage (`ChampionExperimentsPanel`) creates and runs selection experiments.

- **Create / run an experiment** → job type `champion_experiment` (params `{experiment_id}`) or
  `champion_select`, submitted via `POST /jobs`. Compares loaded backtests by rolling WAPE and
  writes the per-DFU winners.
- **Load results** → `champion_results_load` (requires the experiment to be promoted).
- These three job types are in the `champion` group and are **managed only here**, not in the
  generic Jobs tab job-group list.

---

## 12.4 Phase 8 — Production forecast + promote (Model Tuning → Forecast stage)

The Forecast stage (`ForecastPanel`) is a guided Train → Generate → Promote flow.

| Step | UI control → endpoint | Job type | Notes |
|---|---|---|---|
| **Train** production models | Train (per model or all) → `POST /backtest-management/{model_id}/train` (or `/all/train`) | `train_production_model` (`{model_id, all_models}`) | **Tree models only** — foundation/DL are zero-shot, no training |
| **Generate** forecast | Generate → `POST /backtest-management/{model_id}/generate` | `generate_production_forecast` (`{horizon, model_id}`) | writes `fact_production_forecast_staging`; check counts via `GET /backtest-management/staging-summary` |
| **Promote** to production | Promote → `POST /backtest-management/{model_id}/promote` (single) or `POST /backtest-management/champion/promote` (per-DFU champion) | — (direct call) | single-model promotes pass a **WAPE + coverage gate**; `model_id=champion` **bypasses** that gate (experiment-level gating). `X-API-Key` required; gate decisions logged to the AI ledger |

Promotion status is shown on the panel (`GET /backtest-management/promotion-status`). Keep the
system date aligned with the planning month so staging and promote `plan_version` match
(`2026-06`).

---

## 12.5 Phase 9 — Inventory, demand planning, ops (Jobs tab)

These run as generic jobs from **Jobs → Job Groups** (pick a group → pick a job → Run →
`POST /jobs {job_type, params}`), or chained in **Pipeline Builder**.

**`inventory` group (13 jobs):**

| Step | Job type |
|---|---|
| Safety stock | `compute_safety_stock` |
| EOQ / cycle stock | `compute_eoq` |
| Replenishment policies | `assign_policies` |
| Exceptions | `generate_exceptions` |
| ABC-XYZ | `classify_abc_xyz` |
| Demand variability | `compute_variability` |
| Demand signals | `compute_demand_signals` |
| Investment plan | `compute_investment` |
| Health scores | `refresh_health_scores` |
| Intramonth stockout | `refresh_intramonth` |
| Monte-Carlo SS simulation | `run_ss_simulation` |
| Inventory backtest | `inventory_backtest` |
| Algorithm comparison | `compare_inventory_algorithms` |
| **End-to-end inventory pipeline** | `inventory_planning_pipeline` (chains SS → EOQ → policies → exceptions) |

**`replenishment`:** `compute_replenishment_plan` — forward plan from the **promoted** forecast
(run after Phase 8). **`ai`:** `generate_ai_insights`, `generate_storyboard`.

> The fastest UI path for "all of inventory planning": run the single
> **`inventory_planning_pipeline`** job (or build a pipeline of the individual steps in the
> right order). Inventory tab panels auto-refresh once these jobs complete.

Steps that still have **no UI job** (run via `make` per the reset runbook): `setup-demand-planning`'s
forward-planning suite (projection, planned-orders, quantile, consensus, bias, blended,
service-level, lead-time, echelon) beyond `compute_replenishment_plan`, and the full
`setup-ops` suite (S&OP, events, financial plan, scenarios) — only `generate_storyboard`,
`generate_ai_insights`, and `data_quality` (platform group) are exposed as jobs.

---

## 12.6 Phase 10 — MV refresh + verify (Jobs tab)

- **Refresh views:** `refresh_forecast_views` and `refresh_customer_analytics` jobs (Jobs tab),
  or `make refresh-mvs-tiered` for the full tiered pass.
- **Verify:** `GET /backtest-management/promotion-status` and `staging-summary` on the Forecast
  stage; row-count health via `make health`.

---

## 12.7 Monitoring, chaining, scheduling

- **Monitor:** Jobs → **Active Jobs** (live status), **Job History** (+ per-job logs via
  `GET /jobs/{id}/logs`). Cancel a stuck job → `POST /jobs/{id}/cancel`.
- **Chain (run many in order):** Jobs → **Pipeline Builder** → add steps → submit
  (`POST /jobs/pipeline {steps:[{job_type, params}]}`). This is the UI way to run all of
  Phase 5, or the full inventory sequence, in one shot.
- **Schedule (recurring):** Jobs → **Schedules** → `POST /jobs/schedule {job_type, cron|interval}`.
  Useful for nightly backtests / MV refreshes.

---

## 12.8 What has NO UI trigger (still CLI/API)

| Need | Why | Do this instead |
|---|---|---|
| "Run all backtests" single button | not implemented | Pipeline Builder with one step per `backtest_*` job type, or per-model Run |
| Load external ML extracts | no job | `make load-ext-all` |
| Accuracy-MV refresh after backtest load | no dedicated button | `refresh_forecast_views` job, or `make refresh-accuracy-mvs` |
| Full demand-planning suite (projection/quantile/consensus/…) | only `compute_replenishment_plan` exposed | `make setup-demand-planning` |
| Full ops suite (S&OP/events/financial/scenarios) | only storyboard/AI/DQ exposed | `make setup-ops` |
| Auto-promote champion | deliberate manual gate | review, then click **Promote** on the Forecast stage |

---

## Cross-references
- Make-based equivalent of every step: `RESET_RELOAD_2026_06.md` (Phases 5–10).
- Data ingestion / pre-filter: `docs/operations-manual/02-data-ingestion.md`.
- Backtest framework + champion gate internals: `docs/operations-manual/04-forecasting-backtest.md`,
  `05-tuning-champion-selection.md`, `06-production-forecasting.md`.
