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
| **AI-guided operations + generic runner** | **Workflows** (**Plan & Run**, **Workflow Library**, **Manual Load**) | Input readiness through clustering, forecasting, archive, and inventory; plus monitoring, chaining, scheduling, and advanced loading |

**Key facts that apply to everything below:**

- **All long steps run as background jobs** (pg-queue). Firing a control returns immediately
  with a job id; watch it in **Workflows → Workflow Library → Active Jobs**
  (status `queued → running → completed/failed`) and read output under **Job History → logs**.
- **Preferred operating path:** use **Workflows → Plan & Run → Analyze workflows**,
  run only the first safe recommendation, then analyze again after it completes.
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
| `chronos2_enriched` | `backtest_chronos2_enriched` | foundation - slow (multi-hour) |
| `seasonal_naive`, `rolling_mean` | `backtest_seasonal_naive` / `_rolling_mean` | statistical baselines (champion fallback) |
| `mstl` | `backtest_mstl` | statistical decomposition — **needs the `statistical` extra** (`uv sync --extra statistical` / `uv pip install statsforecast`); without `statsforecast` the run produces **zero predictions** and the model stays "No backtest" |
| `nhits`, `nbeats` | `backtest_nhits` / `_nbeats` | deep learning (needs the `dl` extra) |

> **statsforecast extra.** MSTL (and the expert-panel statistical models AutoCES,
> DynamicOptimizedTheta, IMAPA, TSB, ADIDA) depend on `statsforecast`, declared as the
> `statistical` optional extra in `pyproject.toml`. If it's not installed, the backtest logs
> `statsforecast not installed; returning empty DataFrame` for every timeframe and exits with
> `No predictions produced` — nothing loads, so the UI correctly shows "No backtest". Fix:
> `uv pip install statsforecast` (or `uv sync --extra statistical`), then re-run.

**Running the whole roster from the UI** (no single "Run all" button): use the
**Workflows → Workflow Library → Pipeline Builder** and add one step per backtest job type (`backtest_lgbm`,
`backtest_catboost`, … `backtest_nbeats`) → submit as one pipeline (`POST /jobs/pipeline`).
Or click **Run** on each model card. The foundation/DL backtests are the multi-hour
wall — fire them and monitor in Active Jobs.

---

## 12.2 Phase 6 — Results auto-load + accuracy (Model Tuning → Backtest stage)

- **Auto-load (no Load button):** a backtest's predictions are loaded into the DB
  **automatically when the run completes** (server-side, inside the same job) — there is no
  separate "Load" click. The load writes `fact_external_forecast_monthly` (+
  `backtest_lag_archive`) and refreshes the `agg_forecast_monthly` MV, so each model's
  historical backtest shows up as the `forecast_<model>` line in **Item Analysis** and feeds
  the accuracy views. The Backtest-stage table (rows grouped by model family) shows
  **"Loaded"** per model once this finishes.
- **Recovery:** if the best-effort auto-load ever fails (a run shows **"Completed"** instead
  of **"Loaded"**), a manual **Load to DB** button appears on that model's detail panel →
  `POST /backtest-management/{model_id}/load` (job `backtest_load_model`). Re-running the
  backtest also re-loads.
- **Run all per group:** each family section (Tree / Foundation / Statistical / Deep Learning)
  has a **Run all** action that queues a backtest for every model in that group.
- **Accuracy** is shown per row (`GET /backtest-management/summary`); run history via
  `GET /backtest-management/{model_id}/runs`.
- **External ML extracts** (`ext_lgbm/cat/xg/best`) and the **accuracy-MV refresh** have **no
  dedicated UI button** — run `make load-ext-all` and `make refresh-accuracy-mvs`, or submit a
  `refresh_forecast_views` job from **Workflows → Workflow Library**.

---

## 12.3 Phase 7 — Champion (Model Tuning → Champion stage)

The Champion stage (`ChampionExperimentsPanel`) creates and runs selection experiments.

- **Create / run an experiment** → job type `champion_experiment` (params `{experiment_id}`) or
  `champion_select`, submitted via `POST /jobs`. Compares loaded backtests by rolling WAPE and
  writes the per-DFU winners.
- **Load results** → `champion_results_load` (requires the experiment to be promoted).
- These three job types are in the `champion` group and are **managed only here**, not in the
  generic Workflow Library job-group list.

**Champion Strategy Sweep (tournament).** The **Run Sweep** button (next to "New Experiment")
launches a `champion_sweep` job (`POST /champion-sweeps`) that fans out a grid of candidate champion
configs — each a real `champion_experiment` — ranks them globally and within demand segments,
assembles a per-segment composite, and recommends a winner. Pick template chips + optional model-subset
variants in the **Sweep Builder**; the live counter shows how many candidates the grid expands to
(capped at `sweep.max_candidates`, default 24; per-segment scoring adds **no** extra runs). The
**Sweep Results** panel shows the global leaderboard (with gate-eligibility badges), the per-segment
winner map, and a "composite vs. best global" headline. **Promote winner** delegates to the same
Stage-1 promotion as a single experiment and is enabled only when the recommendation passes the
promote gate (`min_wape_improvement_pct` / coverage). Composite promotion is available for the
`demand_class` axis; `ml_cluster`/`abc_xyz` are diagnostic-only. See spec
`docs/specs/02-forecasting/30-champion-strategy-sweep.md`.

---

## 12.4 Phase 8 — Production forecast + promote (Model Tuning → Forecast stage)

The Forecast stage (`ForecastPanel`) is a guided Train → Generate → Promote flow.

| Step | UI control → endpoint | Job type | Notes |
|---|---|---|---|
| **Train** production models | Train (per model or all) → `POST /backtest-management/{model_id}/train` (or `/all/train`) | `train_production_model` (`{model_id, all_models}`) | **Tree models only** — foundation/DL are zero-shot, no training |
| **Generate** forecast | Generate (per model) **or Generate All ready** → `POST /backtest-management/{model_id}/generate?horizon=&confidence_intervals=` | `generate_production_forecast` (`{horizon, model_id, confidence_intervals}`) | writes `fact_production_forecast_staging`; check counts via `GET /backtest-management/staging-summary`. The panel's **Horizon** input + **Include Confidence Intervals** toggle now thread through to every generate path (single / champion / Generate All) |
| **Promote** to production | Promote → `POST /backtest-management/{model_id}/promote` (single) or `POST /backtest-management/champion/promote` (per-DFU champion) | — (direct call) | single-model promotes pass a **WAPE + coverage gate**; `model_id=champion` **bypasses** that gate (experiment-level gating). `X-API-Key` required; gate decisions logged to the AI ledger. **Gate rejections (409) and "no staged rows" (400) now surface as toasts** instead of failing silently |

**Generate All** (Step 1 header button) fires a generate job for every ready model —
non-tree models plus production-trained tree models — in one click, carrying the same
horizon + CI settings.

Promotion status is shown on the panel (`GET /backtest-management/promotion-status`). Keep the
system date aligned with the planning month so staging and promote `plan_version` match
(`2026-06`).

> **Visualizing the result (Item Analysis tab).** Once generated/promoted, open **Item Analysis**
> for a DFU to see each model's **future** forecast (`staging_<model>` lines + the promoted
> `production_forecast` line) and its **past** backtest fit (`backtest_<model>` lines, dotted)
> on one timeline. Toggle them via the **Staging** and **Backtest** pill rows. The backtest
> lines read from `GET /forecast/candidate` (`fact_candidate_forecast`); they appear only
> after a model's predictions have been **Loaded** (Phase 6).

---

## 12.5 Phase 9 — Inventory, demand planning, ops (Workflows)

These run as generic jobs from **Workflows → Workflow Library → Job Groups** (pick a group → pick a job → Run →
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

## 12.6 Phase 10 — MV refresh + verify (Workflows)

- **Refresh views:** `refresh_forecast_views` and `refresh_customer_analytics` jobs (Workflow Library),
  or `make refresh-mvs-tiered` for the full tiered pass.
- **Verify:** `GET /backtest-management/promotion-status` and `staging-summary` on the Forecast
  stage; row-count health via `make health`.

---

## 12.7 Monitoring, chaining, scheduling

- **Monitor:** Workflows → **Workflow Library → Active Jobs** (live status), **Job History** (+ per-job logs via
  `GET /jobs/{id}/logs`). Cancel a stuck job → `POST /jobs/{id}/cancel`.
- **Chain (run many in order):** Workflows → **Workflow Library → Pipeline Builder** → add steps → submit
  (`POST /jobs/pipeline {steps:[{job_type, params}]}`). This is the UI way to run all of
  Phase 5, or the full inventory sequence, in one shot.
- **Schedule (recurring):** Workflows → **Workflow Library → Schedules** → `POST /jobs/schedule {job_type, cron|interval}`.
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
