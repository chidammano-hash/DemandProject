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
- **Write actions require authentication.** The UI signs in through
  `POST /auth/login`, keeps access and refresh tokens in browser-session storage,
  and centrally sends `Authorization: Bearer ...`; it never receives or exposes
  the service API key. Planner-or-higher JWTs may run workflows. Automation via
  direct `curl` may continue to pass `-H "X-API-Key: $API_KEY"`.
- **Prerequisite:** API on `:8000` and UI on `:5173` running (`make api`, `make ui`). Data must
  already be loaded through Phase 4 (features + clustering) per the reset runbook.

---

## 12.1 Phase 5 — Backtests (Model Tuning → Backtest stage)

The Backtest stage lists every model in the roster. Each model has a **Run** control →
`POST /backtest-management/{model_id}/run` (writes; `?parallel=true` to let different model
families run concurrently — same family stays serial).

| Model (model_id) | Job type | Notes |
|---|---|---|
| `lgbm_cluster` | `backtest_lgbm` | per-cluster LightGBM |
| `chronos2_enriched` | `backtest_chronos2_enriched` | foundation - slow (multi-hour) |
| `mstl` | `backtest_mstl` | statistical decomposition — **needs the `statistical` extra** (`uv sync --extra statistical` / `uv pip install statsforecast`); without `statsforecast` the run produces **zero predictions** and the model stays "No backtest" |
| `nhits`, `nbeats` | `backtest_nhits` / `_nbeats` | deep learning (needs the `dl` extra) |

> **StatsForecast extra.** MSTL depends on `statsforecast`, declared as the
> `statistical` optional extra in `pyproject.toml`. If it's not installed, the backtest logs
> `statsforecast not installed; returning empty DataFrame` for every timeframe and exits with
> `No predictions produced` — nothing loads, so the UI correctly shows "No backtest". Fix:
> `uv pip install statsforecast` (or `uv sync --extra statistical`), then re-run.

**Running the whole roster from the UI:** use each model-family section's **Run all** action, or
build a workflow with `backtest_lgbm`, `backtest_chronos2_enriched`, `backtest_mstl`,
`backtest_nhits`, and `backtest_nbeats`. Monitor the multi-hour foundation/DL work in Active Jobs.

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
- The **accuracy-MV refresh** has no dedicated model-panel button — run
  `make refresh-accuracy-mvs`, or submit a `refresh_forecast_views` job from
  **Workflows → Workflow Library**.

---

## 12.3 Phase 7 — Champion (Model Tuning → Champion stage)

The Champion stage (`ChampionExperimentsPanel`) creates and runs read-only selection experiments.

- **Create / run an experiment** → job type `champion_experiment` (params `{experiment_id}`),
  submitted by `POST /champion-experiments`. It compares loaded backtests without replacing the
  active config or champion results.
- The old **Promote Config**, **Load Results**, and sweep **Promote winner** controls are retired.
  Their compatibility endpoints return `410 manual_champion_promotion_retired` with guidance to
  `POST /jobs/pipelines/named/champion-refresh`; they perform no DB/config mutation and submit no job.
  Generic job launch, scheduling, and ad-hoc pipeline APIs also reject and hide
  `champion_results_load`.
- The named **model-refresh** workflow stops after loading all five governed backtests.
- The separate **champion-refresh** workflow uses `governed_champion_refresh`: it verifies the
  exact current five-run lineage and atomically promotes the completed experiment and its results.
- Named pipeline submission is server-idempotent. Concurrent or repeated
  launches of the same preset return HTTP 200 with `status=already_running`
  and the existing pipeline id; only the first launch returns HTTP 202. This
  protection uses a PostgreSQL advisory lock across API workers, not browser
  component state. A just-completed intermediate step is held for five minutes
  to cover successor creation/restart recovery; an abandoned chain with no
  successor stops blocking a clean relaunch after that recovery window.
- The stable API job id `champion_select` invokes the identical governed callable and recovery
  finalizer. Its old hidden configuration card was removed, and the destructive unscoped script is
  no longer reachable from either the UI or job API.
- These champion job types are in the `champion` group and are **managed only here**, not in the
  generic Workflow Library job-group list.

**Champion Strategy Sweep (tournament).** The **Run Sweep** button (next to "New Experiment")
launches a `champion_sweep` job (`POST /champion-sweeps`) that fans out a grid of candidate champion
configs — each a real `champion_experiment` — ranks them globally and within demand segments,
assembles a per-segment composite, and recommends a winner. Pick template chips + optional model-subset
variants in the **Sweep Builder**; the live counter shows how many candidates the grid expands to
(capped at `sweep.max_candidates`, default 24; per-segment scoring adds **no** extra runs). The
**Sweep Results** panel shows the global leaderboard (with gate-eligibility badges), the per-segment
winner map, and a "composite vs. best global" headline. Every recommendation is analysis-only.
To adopt one, make a reviewed production-config change and run the named **champion-refresh** workflow;
the UI never writes production config or champion facts. See spec
`docs/specs/02-forecasting/30-champion-strategy-sweep.md`.

---

## 12.4 Phase 8 — Production forecast + promote (Model Tuning → Forecast stage)

The Forecast stage (`ForecastPanel`) is a guided Train → Generate → Promote flow.

| Step | UI control → endpoint | Job type | Notes |
|---|---|---|---|
| **Train** production models | Train one persisted model (or all persisted models) → `POST /backtest-management/{model_id}/train` (or `/all/train`) | `train_production_model` (`{model_id, all_models}`) | LightGBM, N-HiTS, and N-BEATS require immutable production final-fit artifacts. MSTL fits from history and Chronos 2E uses pinned pretrained weights with covariates. |
| **Generate** forecast | Generate (per model) **or Generate All ready** → `POST /backtest-management/{model_id}/generate?horizon=&confidence_intervals=` | `generate_production_forecast` (`{horizon, model_id, confidence_intervals}`) | writes `fact_production_forecast_staging`; check counts via `GET /backtest-management/staging-summary`. The panel's **Horizon** input + **Include Confidence Intervals** toggle now thread through to every generate path (single / champion / Generate All) |
| **Promote** to production | Promote → `POST /backtest-management/{model_id}/promote` (single) or `POST /backtest-management/champion/promote` (per-DFU champion) | — (direct call) | Both paths are fail-closed. Single-model promotion validates its candidate; champion promotion also validates promoted experiment, routing, cluster/tuning lineage, historical quality, forward structure, and the outgoing snapshot archive. `X-API-Key` is required. Gate rejections and missing staging rows surface as toasts. |

The Forecast stage does not infer readiness from an artifact's own metadata.
`GET /backtest-management/training-status` revalidates LightGBM, N-HiTS, and
N-BEATS against the current completed sales batch and checksum, latest closed
month, generator contract, fitted cohort, and (for LightGBM) the promoted
cluster assignment generation. `GET /backtest-management/snapshot-roster-readiness`
then validates the current champion plus exact rank-1 through rank-3 contender
evidence. Until both the champion candidate and this roster are current, every
**Promote** control stays disabled. Use the single **Prepare Release** action to
launch the canonical `forecast-publish` workflow; the panel polls that exact
pipeline and revalidates readiness once its final step completes. Payload
integrity failures require operator review and intentionally do not offer an
automatic rebuild action.

**Generate All** (Step 1 header button) fires a generate job for every ready model —
the three artifact-backed models plus direct MSTL and Chronos 2E — in one click, carrying the same
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
forward-planning suite (projection, planned-orders, consensus, bias, blended,
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
| Accuracy-MV refresh after backtest load | no dedicated button | `refresh_forecast_views` job, or `make refresh-accuracy-mvs` |
| Full demand-planning suite (projection/consensus/planned orders/…) | only `compute_replenishment_plan` exposed | `make setup-demand-planning` |
| Full ops suite (S&OP/events/financial/scenarios) | only storyboard/AI/DQ exposed | `make setup-ops` |
| Auto-promote champion | deliberate manual gate | review, then click **Promote** on the Forecast stage |

---

## Cross-references
- Make-based equivalent of every step: `RESET_RELOAD_2026_06.md` (Phases 5–10).
- Data ingestion / pre-filter: `docs/operations-manual/02-data-ingestion.md`.
- Backtest framework + champion gate internals: `docs/operations-manual/04-forecasting-backtest.md`,
  `05-tuning-champion-selection.md`, `06-production-forecasting.md`.
