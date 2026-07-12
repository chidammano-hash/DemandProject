# 30 — Champion Strategy Sweep (Tournament)

**Status:** Implemented
**Date:** 2026-06-18
**Related:** 07-champion-selection.md, 24-candidate-forecast-promotion.md, 12-dual-promotion.md, 19-forecast-pipeline-config.md

---

## 1. Problem

Today a "champion" is **one configuration** — a single strategy (`rolling`, window 6) over a
fixed model set (`lgbm_cluster`, `chronos`), chosen by hand and committed in
`forecast_pipeline_config.yaml` `champion:`. The Champion Experimentation Studio
(`champion_experiment` table + `/champion-experiments` API) lets a user run **one** experiment
at a time and compare **two** experiments pairwise (`/champion-experiments/compare`).

What's missing is the answer to the actual business question:

> *"Across the reasonable space of champion configurations — single-model-per-SKU vs.
> blend-per-SKU, different model subsets, windows, metrics — which one is best for **our**
> data; and would a **per-segment mix** (blend the smooth SKUs, single robust model for the
> intermittent ones) beat any one global configuration? Is the winner good enough to promote?"*

Answering that today means manually creating dozens of experiments, eyeballing the leaderboard,
and comparing them two at a time. This spec adds a **sweep** (tournament) that fans out a curated
grid of configurations, runs each as a normal `champion_experiment`, ranks them on a robust
objective **both globally and within demand segments**, assembles a per-segment composite champion,
gates everything against current production, and surfaces a one-click "promote winner".

### 1.1 Framing — "mix for the same SKU" vs. "one model per SKU"

These are **not new capabilities** — they already exist as champion strategies (spec 07):

| Business phrasing | Existing strategy family | Examples | Output `model_id` |
|---|---|---|---|
| **One model per SKU** (winner-take-all, re-picked monthly) | single-model winners (10) | `expanding`, `rolling`, `decay`, `seasonal`, `per_cluster`, bandits | the winning model (`lgbm_cluster`, `chronos`, …) |
| **Mix of models for the same SKU** (weighted blend) | ensemble/blend (13) | `ensemble`, `ensemble_rolling`, `learned_blend`, `ridge_blend`, `adaptive_ensemble`, `shrinkage_blend` | `ensemble` / blend label |
| **Different strategy for different SKUs** | segment/route (8) | `per_segment`, `dfu_strategy_router`, `cluster_regime_hybrid`, `stacked_strategies` | varies per DFU |

The selection grain is already **per-DFU (item + customer_group + loc), re-evaluated per month**
with strict causal lag (spec 07). So the platform can *already produce* any of these. The sweep's
job is purely **empirical model-selection over configurations** — it picks the best of these and
assembles the best per-segment mix; it does not invent new selection math. **No new champion
strategy code is written for this feature** (one small, optional extension to `per_segment` —
§8.3 — is the only touch to strategy code, and only if per-segment model subsets are desired).

---

## 2. Design principles

1. **Reuse, don't reinvent.** A sweep is a *parent* over the existing single-experiment machinery.
   Every candidate config becomes a real `champion_experiment` row, so it inherits — for free —
   per-lag/per-month breakdowns, log streaming, pairwise compare, and the two-stage promotion path.
2. **Per-segment scoring is post-hoc slicing, not re-running.** Each candidate config is run **once**
   over all SKUs (exactly as a normal experiment). Per-segment accuracy is then computed by
   **restricting that config's per-DFU-month results to each segment's DFUs** — so a sweep that
   evaluates both global and per-segment costs essentially the same as the global sweep alone, plus
   cheap post-processing (§6.1). This is what makes "both modes in one release" tractable.
3. **Sequential by default, opt-in parallel.** Mirror the backtest-run pattern
   (commit 552ac68a): children run serially inside one sweep job unless `parallel=true`, with a
   **duplicate guard** (skip a config whose config-hash already has a completed experiment).
4. **Rank on robustness, not just point accuracy.** The winning config must be the one that
   generalizes, not the one that ceiling-chases a single lag/month/segment (§5).
5. **The composite must be reproducible in production.** A per-segment mix is promoted **as a
   `per_segment` strategy config** whose params encode the discovered segment→strategy map — so the
   next production champion run reproduces it natively, with no one-off artifact (§8).
6. **Promotion stays gated.** A sweep never auto-promotes. The "winner" is a *recommendation*;
   promotion reuses the existing `champion.promote_gate` (+1% WAPE improvement, ≥80% coverage)
   and the existing two-stage promote endpoints.
7. **Reads the real evaluation table.** Like `run_champion_experiment.py`, the sweep scores
   configurations off `fact_external_forecast_monthly` (via `load_monthly_errors_df()`). It does
   **not** touch `fact_candidate_forecast` (inert — see spec 24 §4.1).

---

## 3. Data model

Three new tables. Children are ordinary `champion_experiment` rows linked by a join table, plus a
per-segment score table — so the sweep adds orchestration + segment-slice metadata only.

### 3.1 `champion_sweep` (NEW)

Parent record for one tournament.

| Column | Type | Category | Description |
|---|---|---|---|
| `sweep_id` | SERIAL PK | id | |
| `label` | TEXT NOT NULL | config | User label |
| `notes` | TEXT | config | Optional |
| `mode` | TEXT DEFAULT `'both'` | config | `'global'`, `'per_segment'`, or `'both'` (default — compute both and recommend the higher-scoring, gate-passing option) |
| `segment_axis` | TEXT DEFAULT `'demand_class'` | config | Segmentation for per-segment scoring: `'demand_class'` (Syntetos-Boylan; the only *promotable* composite axis in Phase 1), `'ml_cluster'` or `'abc_xyz'` (diagnostic breakdown only — §8.2) |
| `objective` | TEXT DEFAULT `'robust'` | config | Ranking objective: `'accuracy'`, `'gap_to_ceiling'`, `'robust'` (§5) |
| `grid_spec` | JSONB NOT NULL | config | The axes that expand into candidate configs (§4) |
| `parallel` | BOOLEAN DEFAULT FALSE | config | Run children concurrently across model families (else serial) |
| `baseline_experiment_id` | INTEGER FK → `champion_experiment` | config | Optional incumbent to gate/score against; defaults to current promoted experiment |
| `status` | TEXT | state | `queued` / `running` / `completed` / `failed` / `cancelled` |
| `candidate_count` | INTEGER | state | Configs after expansion + dedup (segmentation does **not** multiply this) |
| `completed_count` | INTEGER | state | Children finished |
| `job_id` | VARCHAR(100) | state | Link to `job_history.job_id` |
| `created_at` / `started_at` / `completed_at` | TIMESTAMPTZ | state | |
| `runtime_seconds` | NUMERIC(10,1) | results | |
| `best_global_experiment_id` | INTEGER FK → `champion_experiment` | results | Top-ranked single global config |
| `composite_experiment_id` | INTEGER FK → `champion_experiment` | results | The synthesized per-segment composite (NULL in `global` mode) |
| `recommended_experiment_id` | INTEGER FK → `champion_experiment` | results | The higher-scoring of {best global, composite} that also passes the gate |
| `recommended_score` | NUMERIC(10,4) | results | Recommended config's objective score |
| `recommended_gate_eligible` | BOOLEAN | results | Did the recommendation pass the promote gate vs baseline? |

Indexes: `(status)`, `(created_at DESC)`, partial on `(recommended_experiment_id) WHERE recommended_experiment_id IS NOT NULL`.

### 3.2 `champion_sweep_member` (NEW)

One row per candidate config in the sweep.

| Column | Type | Description |
|---|---|---|
| `sweep_id` | INTEGER FK → `champion_sweep` ON DELETE CASCADE | |
| `experiment_id` | INTEGER FK → `champion_experiment` ON DELETE CASCADE | The child experiment |
| `config_hash` | TEXT | Stable hash of (strategy, params, models, metric, lag) — duplicate guard |
| `is_composite` | BOOLEAN DEFAULT FALSE | TRUE for the synthesized per-segment composite member |
| `global_rank` | INTEGER | 1 = best on the global objective; NULL until sweep completes |
| `global_score` | NUMERIC(10,4) | Global objective score (§5) |
| `gate_eligible` | BOOLEAN | Passed promote gate vs baseline |
| `skipped_duplicate` | BOOLEAN DEFAULT FALSE | TRUE when a prior completed experiment with the same `config_hash` was reused instead of re-run |

`UNIQUE(sweep_id, experiment_id)`; index on `(sweep_id, global_rank)`.

### 3.3 `champion_sweep_segment_score` (NEW)

Per-segment accuracy for each candidate, computed by post-hoc slicing (§6.1). This is the data behind
the per-segment winner map and the composite.

| Column | Type | Description |
|---|---|---|
| `sweep_id` | INTEGER FK → `champion_sweep` ON DELETE CASCADE | |
| `experiment_id` | INTEGER FK → `champion_experiment` ON DELETE CASCADE | The candidate scored |
| `segment` | TEXT | Segment label on `segment_axis` (e.g. `smooth`, `intermittent`) |
| `n_dfus` | INTEGER | DFUs in this segment for this candidate (coverage guard, §5) |
| `accuracy` | NUMERIC(8,4) | Segment-restricted champion accuracy % |
| `score` | NUMERIC(10,4) | Segment-restricted objective score |
| `segment_rank` | INTEGER | 1 = best candidate **within** this segment |

`UNIQUE(sweep_id, experiment_id, segment)`; index on `(sweep_id, segment, segment_rank)`.

**DDL file:** `sql/192_champion_sweep.sql` (idempotent — `IF NOT EXISTS` on all objects).

---

## 4. Grid specification

`grid_spec` is a small JSON describing **axes**; the runner takes their Cartesian product, dedups by
`config_hash`, and caps the result. Crucially, **segmentation is not an axis** — it is post-hoc
slicing of each candidate's results (§6.1), so `candidate_count` is independent of `segment_axis`.

### 4.1 Template-based (recommended default)

Reference existing entries from `config/forecasting/champion_experiment_templates.yaml` (36 curated
strategies) and optionally cross them with a model-subset axis:

```jsonc
{
  "templates": ["production_baseline", "rolling_6m", "ensemble_top3_inverse",
                "learned_blend", "per_segment", "adaptive_ensemble"],
  "models_variants": [                       // optional second axis
    ["lgbm_cluster", "chronos"],
  ],
  "metric": ["wape"]                         // optional; default = champion config metric
}
```

Expansion = `templates × models_variants × metric`. With the example above: 6 × 2 × 1 = **12 candidates**.

### 4.2 Explicit configs

A raw list of full experiment bodies (same shape as `CreateChampionExperimentBody`) for power users:

```jsonc
{ "configs": [ { "strategy": "rolling", "strategy_params": {"window_months": 3}, "models": [...] }, ... ] }
```

### 4.3 Guards

- **Cap:** `grid_spec` expands to at most `sweep.max_candidates` (config, default **24**). Over-cap →
  the create call rejects with a clear message listing the expanded count. No silent truncation.
- **Dedup:** identical `config_hash` collapses to one candidate.
- **Duplicate reuse:** if a *completed* `champion_experiment` with the same `config_hash` already
  exists (from a prior sweep or manual run), the member links to it and sets `skipped_duplicate=true`
  rather than re-running — saving the most expensive part (foundation-model strategies).

---

## 5. Ranking objective

Point accuracy alone over-promotes configs that spike on one lag/month/segment. Three objectives,
default `robust`, applied identically to global scores and to segment-restricted scores:

| Objective | Formula | Use when |
|---|---|---|
| `accuracy` | `champion_accuracy` (higher wins) | Quick look; horizon is short and uniform |
| `gap_to_ceiling` | `−gap_bps` (smallest gap to the per-DFU-month oracle wins) | You care about headroom left on the table |
| **`robust`** (default) | `mean_lag_accuracy − λ · stdev(lag_accuracy) − μ · stdev(month_accuracy)` | Production promotion — rewards consistency across the full lag horizon and over time |

`mean_lag_accuracy` / `stdev(lag_accuracy)` come from the child's `champion_experiment_lag` rows;
`stdev(month_accuracy)` from `champion_experiment_month`. `λ`, `μ` live in config
(`sweep.robust_lambda`, `sweep.robust_mu`; defaults `0.5`, `0.25`) — **no magic numbers in code.**

**Per-segment coverage guard.** A candidate's per-segment score is trusted only if the segment has at
least `sweep.min_segment_dfus` DFUs (config, default **30**). Below that, the segment falls back to the
**global** winner for that segment — small, noisy segments don't get to pick a fragile specialist.

**Gate eligibility (orthogonal to ranking).** Each member, and the composite, is independently checked
against the promote gate vs the baseline experiment (reusing `_evaluate_promotion_gate` logic / config:
`min_wape_improvement_pct`, `min_coverage_frac`). `recommended_experiment_id` is the higher-scoring of
{best global config, per-segment composite} that **also** passes the gate; if neither passes, the sweep
still completes and reports the top option with `recommended_gate_eligible=false` and a note ("no
candidate or composite beat production by the gate margin").

---

## 6. Execution & API

### 6.1 Job type & runner

New job type `champion_sweep` (group `champion`, alongside `champion_experiment` /
`champion_results_load`), params `{"sweep_id": int}`, handler `_run_champion_sweep` in
`common/services/job_state.py`, invoking `scripts/ml/run_champion_sweep.py --sweep-id <id>`.

Runner flow:

1. Load `champion_sweep` row; set `status='running'`.
2. Expand `grid_spec` → candidate configs; dedup; enforce cap. Resolve `baseline_experiment_id`
   (explicit, else current promoted experiment).
3. For each config: insert a `champion_experiment` row (`status='queued'`) + a
   `champion_sweep_member` link with its `config_hash`. Reuse completed duplicates (§4.3).
4. **Run children** — serial by default, thread-pooled across model families when `parallel=true` —
   via the same path `run_champion_experiment.py` uses (load monthly errors → apply strategy → score
   accuracy/ceiling/gap → write lag/month breakdowns + cache `experiment_{id}_winners.csv`). Update
   `completed_count`/progress; honor the job `cancel_event`. A child that errors is marked `failed`
   and excluded from ranking; the sweep continues.
5. **Global ranking:** score each successful member with the objective; assign `global_rank`,
   `global_score`, `gate_eligible`; set `best_global_experiment_id`.
6. **Per-segment slicing** (skipped in `global` mode): label every DFU by `segment_axis` (reusing the
   Syntetos-Boylan classifier in `common/ml/champion/segment.py` for `demand_class`, or `dim_sku`
   columns for `ml_cluster`/`abc_xyz`). For each candidate × segment, recompute accuracy/score over
   that segment's DFUs from the cached winners + monthly errors → `champion_sweep_segment_score`. Pick
   the per-segment winner (with the `min_segment_dfus` fallback to the global winner).
7. **Composite assembly** (`per_segment`/`both` modes): build the per-segment winners by
   concatenating, for each segment, that segment's winning child's per-DFU-month rows restricted to
   the segment's DFUs. Materialize as a new `champion_experiment` row (`strategy='per_segment'`,
   `strategy_params` = the discovered segment→strategy/params map — §8) with its own cached
   `experiment_{id}_winners.csv`, scored + gated like any candidate; set `composite_experiment_id`
   and a `is_composite=true` member.
8. **Recommend & finish:** `recommended_experiment_id` = higher-scoring, gate-passing of {best global,
   composite}; write `recommended_*`; `status='completed'`.

Cost note: steps 6–7 are cheap arithmetic over already-computed child results — **the expensive part
(the N child runs) is shared between global and per-segment**, so `mode='both'` ≈ cost of `global`
plus the single composite evaluation.

### 6.2 Endpoints

Under a new `/champion-sweeps` prefix — added to `api/main.py` **before** `domains.py`,
`frontend/vite.config.ts` `API_PATH_PREFIXES`, and `frontend/src/api/queries/index.ts` (same change):

| Method | Path | Purpose |
|---|---|---|
| POST | `/champion-sweeps` | Create + launch. Body: `label`, `mode`, `segment_axis`, `objective`, `grid_spec`, `parallel`, `baseline_experiment_id?`. Returns expanded `candidate_count` + `sweep_id` + `job_id`. `Depends(require_api_key)`. |
| GET | `/champion-sweeps` | List (paginated, status filter). |
| GET | `/champion-sweeps/{sweep_id}` | Detail: sweep row + progress + the recommendation summary. |
| GET | `/champion-sweeps/{sweep_id}/leaderboard` | Global-ranked members joined to their `champion_experiment` rows (strategy, params, accuracy, ceiling, gap, score, gate_eligible, rank). |
| GET | `/champion-sweeps/{sweep_id}/segments` | Per-segment winner map + per-segment scores (from `champion_sweep_segment_score`), and the global-vs-composite head-to-head. |
| POST | `/champion-sweeps/{sweep_id}/cancel` | Cancel running/queued sweep + its queued children. `require_api_key`. |
| POST | `/champion-sweeps/{sweep_id}/promote-winner` | Convenience: Stage-1 promote on `recommended_experiment_id` (refuses if not gate-eligible unless `bypass_token`). `require_api_key`. |
| DELETE | `/champion-sweeps/{sweep_id}` | Delete sweep + members (children left intact unless orphaned). `require_api_key`. |

All reads use `get_conn()` / `get_async_read_only_conn()`, `%s` placeholders, `psycopg.sql.Identifier`
for any identifier interpolation.

---

## 7. Frontend

In the **Champion** stage (`frontend/src/tabs/champion/`):

- **`SweepBuilder`** (modal): template chips (from `/champion-experiments/templates`); a **model-subset
  picker** — a config-driven checklist of the enabled roster (from `/config/forecast_pipeline_config`,
  labelled via `model-labels`) plus preset buttons (current champion, all-tree, all-foundation,
  tree+foundation, all-competing), defaulting to the current `champion.models`; `mode` (global /
  per-segment / both), `segment_axis`, objective dropdown, `parallel` toggle, and a live "expands to N
  candidates" counter. The model subset does **not** multiply the candidate count (one subset competes
  over all selected models); candidate count = number of selected templates.
- **`SweepResultsPanel`** with two views:
  - **Global leaderboard** — Rank, Label, Strategy, Models, Accuracy %, Ceiling %, Gap (bps), Robust
    score, **Gate** badge (✓ eligible / ✗ below margin), Actions (View experiment → reuses existing
    experiment detail/compare; this drill-in is free reuse). Sortable.
  - **Per-segment map** — for each segment (smooth / erratic / intermittent / lumpy …): winning
    strategy, segment accuracy, DFU count, and a flag when it fell back to the global winner. Plus a
    headline **"Composite vs. best global"** card showing the delta and which one is recommended.
  - A progress strip shows `completed_count / candidate_count` while running.
- Entry point: a **"Run Sweep"** button in `ChampionExperimentsPanel` header, next to "New Experiment".

All HTTP via a new `src/api/queries/champion-sweeps.ts` (`fetchJson`, typed to mirror the Pydantic
models — no `any`). Charts/colors via theme context per house rules.

---

## 8. Per-segment composite — reproducibility detail

The composite is the heart of "blend for some SKUs, single model for others", so its production
representation matters.

### 8.1 Promote as a `per_segment` config (default, `demand_class` axis)

`per_segment` (`common/ml/champion/segment.py:118`) already maps each Syntetos-Boylan demand class to a
strategy + params (e.g. smooth→`expanding`, intermittent→`rolling(6)`). The sweep's discovered
segment→winner map is exactly this shape, so the composite is stored and promoted **as a `per_segment`
experiment whose `strategy_params` carry the discovered map**. Promotion is then the ordinary two-stage
path (§24/§12): Stage-1 writes a runnable `per_segment` config to `forecast_pipeline_config.yaml`;
Stage-2 loads the cached composite winners. Next production cycle reproduces it natively — **no one-off
artifact, no new strategy.**

### 8.2 Other axes are diagnostic-only in Phase 1

`ml_cluster` and `abc_xyz` segmentations are offered as **read-only breakdowns** (insight: "where does
each config win?") because the existing `per_cluster` strategy selects the best *model* per cluster, not
the best *strategy* per cluster — so an ml-cluster composite has no clean runnable representation yet.
Promotion of a composite is allowed only for `segment_axis='demand_class'`. The UI disables
"Promote winner" for the composite under other axes and shows them for analysis.

### 8.3 Optional extension (only if needed)

If a sweep wants **per-segment model subsets** (e.g. blend lgbm+chronos for smooth, single seasonal for
lumpy), and `per_segment` cannot carry per-segment model lists today, that is a small additive extension
to `per_segment` in `segment.py` (accept an optional per-segment `models` override) — **not** a new
strategy. Flag during implementation; otherwise the composite holds the model set constant across
segments (the sweep's union) and varies only strategy+params per segment.

---

## 9. Config additions

`forecast_pipeline_config.yaml` gains a `sweep:` block (inline comments per house rule):

```yaml
sweep:
  max_candidates: 24        # hard cap on expanded grid size; create call rejects above this
  default_objective: robust # accuracy | gap_to_ceiling | robust
  default_mode: both        # global | per_segment | both
  default_segment_axis: demand_class  # demand_class (promotable) | ml_cluster | abc_xyz (diagnostic)
  robust_lambda: 0.5        # penalty weight on stdev of accuracy across execution lags
  robust_mu: 0.25           # penalty weight on stdev of accuracy across months
  min_segment_dfus: 30      # below this, a segment falls back to the global winner (anti-overfit)
  parallel_default: false   # children run serially unless the sweep opts in
```

The promote gate is **not** duplicated here — the sweep reuses `champion.promote_gate`.

Winner artifacts serialize ensemble `source_mix` values as JSON. Stage-2 result loading also
accepts legacy safe Python-literal list values created by earlier pandas CSV exports, validates that
the decoded value is a list of model-weight objects, and rejects all other shapes.

---

## 10. Testing

- **Unit** (`tests/unit/test_champion_sweep.py`): grid expansion (templates × variants), `config_hash`
  stability + dedup, cap rejection, duplicate-reuse path, each ranking objective, per-segment slicing
  (incl. `min_segment_dfus` fallback), composite assembly → `per_segment` param map, gate filtering,
  "neither beats gate" path.
- **API** (`tests/api/test_champion_sweep.py`): create (returns candidate_count; segmentation does not
  inflate it), list, detail, leaderboard ordering, segments endpoint, cancel, promote-winner (allowed
  vs gate-refused vs non-`demand_class` composite refused), delete — all with `make_pool` /
  `make_async_pool` and `httpx.AsyncClient` + `ASGITransport`.
- **Frontend** (`SweepBuilder.test.tsx`, `SweepResultsPanel.test.tsx`): candidate-count preview, global
  leaderboard render + sort, per-segment map + composite-vs-global card, gate badges, promote-winner
  wiring, disabled-promote for diagnostic axes (`TestQueryWrapper`, mocked queries).
- `make test-all` green; `make audit-routers` parity for the new prefix.

---

## 11. Migration & rollout

- DDL: `sql/192_champion_sweep.sql`; add the three tables to the `db-truncate-data` +
  cleanup runbook (`docs/operations-manual/11-maintenance-troubleshooting.md`).
- Operations: extend `docs/operations-manual/12-ui-pipeline-runbook.md` Phase 7 (Champion) with the
  sweep flow (run sweep → review global + per-segment → promote winner).
- Docs: this spec; `docs/ARCHITECTURE.md` Feature Catalog; `docs/specs/07-champion-selection.md`
  cross-link. Update `CLAUDE.md` only if a new critical rule emerges (none expected — additive).
- Backward compatible: existing single experiments, compare, and promotion are untouched — the sweep
  is purely additive orchestration on top of them.

---

## 12. Implementation notes (as shipped)

| Piece | Location |
|---|---|
| DDL (3 tables) | `sql/192_champion_sweep.sql` |
| Runner (grid expand, rank, per-segment slice, composite, recommend) | `scripts/ml/run_champion_sweep.py` |
| Job type `champion_sweep` | `common/services/job_registry.py`, handler `_run_champion_sweep` in `common/services/job_state.py` |
| API router (`/champion-sweeps`) | `api/routers/forecasting/champion_sweeps.py`, mounted in `api/main.py` before `domains.py` |
| Config `sweep:` block | `config/forecasting/forecast_pipeline_config.yaml` |
| Frontend queries | `frontend/src/api/queries/champion-sweeps.ts` (barrel + vite proxy wired) |
| Frontend UI | `frontend/src/tabs/champion/SweepBuilder.tsx`, `SweepResultsPanel.tsx`; "Run Sweep" button in `ChampionExperimentsPanel.tsx` |
| Tests | `tests/unit/test_champion_sweep.py`, `tests/api/test_champion_sweep.py`, `frontend/src/tabs/champion/__tests__/Sweep{Builder,ResultsPanel}.test.tsx` |

Key implementation choices vs. the design above:
- Children run **in-process** (the runner imports `run_champion_experiment.run_experiment`) rather than
  fanning out subprocesses — sequential, deterministic, reuses the proven experiment path. The
  `parallel` flag is plumbed through config/DB/UI; thread-pooled execution is the follow-up if
  serial wall-clock becomes a constraint.
- **Per-segment scoring is post-hoc slicing** of each child's cached `experiment_{id}_winners.csv` —
  no re-runs, so `mode='both'` costs ≈ `global` + the single composite evaluation, as designed.
- The gate check used for **ranking** is a lightweight accuracy-based proxy of the WAPE gate
  (`gate_eligible` in the runner); the **promote** path delegates to the experiments router's
  `promote_experiment`, which applies the full `champion.promote_gate` (WAPE + coverage) at commit time.
- Composite promotion is restricted to `segment_axis='demand_class'` (maps to a runnable `per_segment`
  config); `ml_cluster`/`abc_xyz` produce diagnostic per-segment scores only.
- The composite member is scored + ranked against the global candidates after assembly (its
  `global_rank`/`global_score`/`gate_eligible` are written back), so it appears in the leaderboard
  rather than as an unranked row.
- Segments with fewer than `sweep.min_segment_dfus` DFUs are skipped (no per-segment score rows
  emitted) and fall back to the global winner — so near-empty classes (e.g. `lumpy`/`intermittent` on
  dense-demand datasets where almost all DFUs have ADI≈1) don't surface as phantom `acc=None` rows.
```
