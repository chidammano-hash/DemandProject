# 33 - Forecast Snapshot Archive & Live FVA

**Status:** Proposed
**Date:** 2026-07-08 (revised same day after 3-lens adversarial review: statistical, factual, completeness)
**Related:** 03-backtest-framework.md, 08-production-forecast.md, 24-candidate-forecast-promotion.md, 27-ai-champion-forecast.md, 32-lag-decomposed-accuracy-leaderboard.md, ../08-integration/07-fva.md, ../07-user-experience/06-backtest-cleanup.md, ../01-foundation/06-execution-lag.md

---

## 1. Problem

Every month the platform generates forward forecasts for all competing algorithms into `fact_production_forecast_staging` and promotes a champion-routed plan into `fact_production_forecast`.
Neither table preserves history:

- The next promote executes `DELETE FROM fact_production_forecast` (backtest_management.py promote, step 4), destroying the previously published plan.
- Staging is regenerated per model with DELETE-before-INSERT, so each generation overwrites the last.
- Staging accumulates the full horizon for every model (today: 18 models, 5.3M rows, 2 GB) with no retention story.

Consequently the platform can never answer "what did each algorithm predict for June 2026, as of June 2026, and how accurate was it once actuals landed?"
The existing `backtest_lag_archive` does not cover this: it holds backtest *simulations* (retrospective predictions from simulated training cutoffs), is capped at lag 0-4, and is keyed on the `(item_id, customer_group, loc)` backtest grain.
Production forecasts are real forward statements at the `(item_id, loc)` DFU grain and must not be mixed into backtest accuracy surfaces.

The concrete June 2026 case: actuals for June 2026 are loaded, and forward forecasts generated as-of June 2026 (staging `forecast_month_generated = 2026-06-01`, 18 models; promoted champion `plan_version = '2026-06'`) cover June through November 2026.
Those six months are snapshot lags 0 through 5 relative to a June record month.
Without an archive, the moment the July cycle regenerates staging and re-promotes, the June predictions are gone and real (non-backtest) accuracy can never be computed.

## 2. Design principles

1. **Snapshot what was believed, keyed by when it was believed.**
   Every archived row carries `record_month` (the planning cycle that produced it) and a derived `lag` (months from record month to forecast month).
2. **Plan-age lag and model horizon are different numbers; keep both.**
   Snapshot `lag` is the age of the plan of record ("as of June, what did we publish for August" = lag 2).
   `horizon_months` is the model's true forecast distance from its own per-DFU last-actual origin.
   On the live June generation, 22.6% of series (48,233 of 213,578) have a first forecast month *before* June because their history ends early, so a June-dated row can be true horizon 1 for one DFU and horizon 9 for another.
   The archive therefore carries `horizon_months` verbatim so the two are never conflated, and snapshot lag curves must not be presented as equivalent to the backtest natural-lag curves of `agg_accuracy_lag_archive` (see 6.4).
3. **Archive and cleanup are separate workflows.**
   Archiving is a non-destructive monthly snapshot.
   Cleanup is an independently triggered purge, gated on verifying the archive exists and reconciles, so an operator can archive without committing to a purge and can never delete an un-archived generation.
4. **Real accuracy, not simulated, and honestly labeled.**
   Snapshot accuracy joins archived forward forecasts to actuals as they land, at DFU grain, so cross-model comparisons can be computed on a common DFU set (coverage differs sharply by model: 17 candidates + champion cover 12,306 DFUs in the June window, `ensemble` covers 4,376).
5. **Reuse the platform rails.**
   JobManager job types, `get_planning_date()` (never `date.today()`), `refresh_for_tables()` with the MV registered in `MV_SOURCES`, config in YAML, psycopg `%s` parameters.

## 3. Data model

### 3.1 `fact_forecast_snapshot` (NEW, sql/200)

One row per algorithm's prediction for one DFU-month, as of one record month.

| Column | Type | Description |
|---|---|---|
| snapshot_sk | BIGSERIAL PK | Surrogate key |
| record_month | DATE NOT NULL | As-of planning month (month-start CHECK); the user-facing "recordMonth" |
| model_id | VARCHAR(100) NOT NULL | Algorithm id, `champion` for the promoted plan, or `ai_champion` |
| item_id | VARCHAR(50) NOT NULL | DFU item |
| loc | VARCHAR(50) NOT NULL | DFU location |
| forecast_month | DATE NOT NULL | Target month (month-start CHECK) |
| lag | SMALLINT generated, stored | Plan age in months; see formula below; CHECK `lag >= 0` |
| horizon_months | SMALLINT | The model's true horizon from its per-DFU origin, carried from the source row (NULL for sources that lack it) |
| forecast_qty | NUMERIC(12,2) NOT NULL | Point forecast (P50) |
| forecast_qty_lower | NUMERIC(12,2) | P10 |
| forecast_qty_upper | NUMERIC(12,2) | P90 |
| source_model_id | VARCHAR(100) | For `champion` rows: the routed underlying model |
| cluster_id | TEXT | Cluster at generation time |
| plan_version | VARCHAR(30) | For `champion`/`ai_champion` rows: the promotion's plan_version |
| run_id | UUID | Generation run id carried from the source row |
| generated_at | TIMESTAMPTZ | Generation timestamp carried from the source row |
| archived_at | TIMESTAMPTZ DEFAULT NOW() | Snapshot write time |

- **Generated lag column** - PG16 rejects `age()`-based expressions in generated columns (not immutable; verified).
  Use plain EXTRACT arithmetic, the same formula as sql/010's `chk_backtest_lag_archive_lag_matches_dates`:

  ```sql
  lag SMALLINT GENERATED ALWAYS AS (
    ((EXTRACT(YEAR FROM forecast_month) - EXTRACT(YEAR FROM record_month)) * 12
     + (EXTRACT(MONTH FROM forecast_month) - EXTRACT(MONTH FROM record_month)))::smallint
  ) STORED
  ```

  Lag 0 is the record month itself; lag 5 is five calendar months after it.
  A lag/date mismatch is impossible by construction.
- Unique key: `(record_month, model_id, item_id, loc, forecast_month)`.
- **Snapshot immutability:** the archive INSERT uses `ON CONFLICT DO NOTHING` - the first snapshot for a record month wins.
  Rationale: the planning date can be pinned (`config/planning_config.yaml`) while the wall clock advances, so a late re-generation can be produced *after* the record month's actuals have loaded; letting it overwrite the earlier snapshot would inject hindsight into "as-of" accuracy.
  A deliberate re-snapshot requires `--overwrite`, which is logged loudly and records the replaced rows' count.
- Indexes: the unique key; `(record_month, lag)`; `(model_id, record_month)`; `(item_id, loc, forecast_month)`.
- Grain note: production forecasts are `(item_id, loc)` DFU grain with **no customer_group**.
  Never join this table to `dim_sku` on only `(item_id, loc)` - dim_sku's grain is 3-key and a 2-key join fans out across customer groups (see 01-accuracy-kpis.md).
  Item-level attributes may be joined via `dim_item` safely.

### 3.2 `agg_accuracy_snapshot` (NEW MV, same sql/200 file)

DFU-grain accuracy components, one row per archived DFU-month whose forecast month is closed and in the DFU's active window - the same "preserve the DFU grain so the API can intersect" rationale as `agg_accuracy_by_dfu` (sql/193).

```
closed months   = SELECT DISTINCT startdate FROM fact_sales_monthly
actuals         = SELECT item_id, loc, startdate, SUM(qty) AS actual_qty
                  FROM fact_sales_monthly GROUP BY 1,2,3
                  (sums across customer_group to the DFU grain; the type = 1
                   CHECK on fact_sales_monthly makes a type filter redundant)
active window   = DFU has at least one sales row in the 12 months up to and
                  including forecast_month
row             = (record_month, model_id, lag, horizon_months, item_id, loc,
                   forecast_month, forecast_qty,
                   actual_qty = COALESCE(actuals.actual_qty, 0),
                   abs_error  = ABS(forecast_qty - actual_qty))
                  for archive rows with forecast_month IN closed months
                  AND the DFU in its active window
```

- **Active-window rule:** a DFU-month with no sales row in a closed month scores `actual = 0` only when the DFU is active per the window above; otherwise the row is excluded.
  This scores legitimate intermittent zeros while not punishing models on DFUs that exited the assortment (a horizon-9 forecast for a dead DFU must not be booked as a lag-0 miss).
  `agg_dfu_naive_scale` (sql/194) densifies zeros only within a DFU's active span for the same reason.
- **Closed-month rule:** a month is closed when any committed sales rows exist for it.
  This is safe against partial visibility because sales loads are whole-file transactional upserts (scripts/etl/load.py `_safe_upsert`) and MV refresh runs post-commit; a month therefore appears all at once.
  Accuracy cells can still revise if a month's source file is re-delivered - the API exposes the MV's last-refresh time so the panel can display it.
  If an intra-month incremental sales feed is ever introduced, this rule must be replaced with an explicit load-complete marker *before* that feed ships.
- KPIs are never stored; the API derives them from summed components via `compute_kpis` (api/core.py): `wape = 100 * sum_abs_error / ABS(sum_actual)`, `accuracy_pct = 100 - wape`, `bias = sum_forecast / sum_actual - 1`, null KPIs when `sum_actual = 0`.
- Unique index on `(record_month, model_id, item_id, loc, forecast_month)` backs `REFRESH ... CONCURRENTLY`.
- Registered in `common/core/mv_refresh.py` `MV_SOURCES` as `agg_accuracy_snapshot: {fact_forecast_snapshot, fact_sales_monthly}` - **Tier 1** (aggregates directly over base tables, like `agg_dfu_naive_scale`), in the same change as the DDL; enforced by `tests/unit/test_mv_refresh.py`.
  Sales loads and archive runs then refresh it automatically via `refresh_for_tables()`.
- Sizing: ~213k rows per closed (record_month, lag) slice across 19 models; one full record month at all six lags closed is ~1.35M rows.

## 4. Workflow 1 - archive (`archive_forecast_snapshot`)

New script `scripts/forecasting/archive_forecast_snapshot.py` (modern template per scripts/db/refresh_mvs.py: `logging`, module-run form, int exit codes, no `print()`, no module-level `parents[N]`).

### 4.1 Sources and predicates

| Source | Predicate | Yields |
|---|---|---|
| `fact_production_forecast_staging` | `forecast_month_generated = record_month AND forecast_month BETWEEN record_month AND record_month + (horizon-1) months` | All candidate algorithms' rows, lags 0..horizon-1, incl. `horizon_months` |
| `fact_production_forecast` | `model_id = 'champion' AND plan_version = to_char(record_month, 'YYYY-MM')`, same forecast_month window | The promoted plan with `source_model_id` routing |
| `fact_ai_champion_forecast` | `plan_version = to_char(record_month, 'YYYY-MM')`, same window; archived as `model_id = 'ai_champion'`, `forecast_qty = ai_qty` | Interactive AI adjustments (sparse, single-DFU; see 27-ai-champion-forecast.md) |

- `record_month` defaults to `MAX(forecast_month_generated)` present in staging, validated `<= get_planning_date()` month; `--record-month YYYY-MM` overrides.
  Deriving the record month from the data (not the wall clock) makes a July run correctly archive the June generation.
  **Empty staging** (post-cleanup, pre-generation window): the script logs "nothing to archive" and exits 0 - a scheduled run in that window is a documented no-op, not an error.
- `horizon` defaults from config (`forecast_snapshot.horizon_months: 6`, i.e. lags 0-5); `--horizon` overrides.
- **Prerequisite fix (same change):** `promote_model()` currently stamps `plan_version = datetime.now(UTC).strftime("%Y-%m")` (backtest_management.py:998) - wall clock - while generation stamps `forecast_month_generated` from `get_planning_date()`.
  With the planning date pinned (as it is today: planning 2026-06-17 vs wall clock 2026-07-08), a re-promote would write plan_version `2026-07` for the June generation and silently break the champion predicate.
  Change promote to derive plan_version from `get_planning_date()` so both stamps come from the same clock, per the platform rule.
- Champion rows **absent** for the record month (no promote that cycle) is a warning; candidates still archive.
  A champion present whose `plan_version` does **not** match the record month is a hard failure (exit 2) - it means the clocks diverged; `--allow-plan-mismatch` overrides after investigation.
- Inserts are single `INSERT INTO ... SELECT` statements with `ON CONFLICT ... DO NOTHING` (first snapshot wins; `--overwrite` switches to `DO UPDATE`, see 3.1).
- After commit: `refresh_for_tables(["fact_forecast_snapshot"])`, then a per-model archived-row summary in the job log.
- `--dry-run` prints per-model counts; `--models a,b` restricts scope.

### 4.2 Job, scheduling, and the archive-before-overwrite invariant

- Register JobTypeDef `archive_forecast_snapshot` (group `forecast`, subprocess style via `_run_subprocess`, params `{record_month: None, horizon: None, dry_run: False, overwrite: False}`).
- **The hard deadline:** a generation's candidates die at the next generation's DELETE-before-INSERT, and the champion dies at the next promote's `DELETE FROM fact_production_forecast`.
  A calendar cron alone can miss that window, losing exactly the history this feature exists to keep.
  Therefore the `forecast-publish` named pipeline (config/forecasting/pipelines.yaml) gains a **leading** `archive_forecast_snapshot` step: the outgoing generation is snapshotted before train/generate replace it.
  With `ON CONFLICT DO NOTHING` this step is an idempotent no-op when the month-close run already archived, and a no-op on empty staging.
- Belt-and-suspenders cron: `default_schedules` entry `forecast_snapshot_monthly` (cron `0 4 3 * *`, the 3rd at 04:00, after month-open actuals loads; `enabled: false` initially).
- Manual trigger: Jobs tab, or `make forecast-archive ARGS="--record-month 2026-06"`.

### 4.3 Sizing (June 2026, measured)

1,281,468 staging rows in the June-November window: 17 models at full 12,306-DFU coverage plus `ensemble` at 4,376 DFUs (12,306 x 6 x 17 + 4,376 x 6 = 1,281,468; per-model counts logged at run time), plus 73,836 champion rows (12,306 x 6).
This coverage asymmetry is exactly why cross-model FVA deltas must use a common-DFU intersection (6.2).
Single INSERT..SELECT statements at this scale complete well inside APScheduler comfort; pg-queue is not needed.

## 5. Workflow 2 - cleanup (`cleanup_forecast_staging`)

Separate script `scripts/forecasting/cleanup_forecast_staging.py`, separate job type, never chained automatically after the archive.

### 5.1 Scope

Deletes from `fact_production_forecast_staging` only.
`fact_production_forecast` is not touched: the promoted plan must remain live for planning consumers (inventory, ai_champion, UI), and the next promote already replaces it.

Default target: generations strictly older than the current planning month (`forecast_month_generated < date_trunc('month', get_planning_date())`).
`--generation YYYY-MM` targets one generation explicitly (required to delete the current month's generation, which also requires `--force`).

### 5.2 Safety gate (the archive-first invariant)

Before deleting a generation G, for every `model_id` with staging rows in G's horizon window:

```
archived(record_month = G, model_id) >= staging rows(G, model_id, forecast_month within horizon)
```

Any shortfall aborts with exit 2 and a per-model reconciliation report; `--force` overrides (logged loudly).
Rows beyond the horizon window (e.g. 2027-2028 months) are intentionally not archived and are deleted without compensation - they are regenerated every cycle and have no accuracy value.

- `--dry-run` reports what would be deleted and the gate verdict without deleting.
- No MV refresh is needed: no registered MV reads staging (verified against `MV_SOURCES`).
- Register JobTypeDef `cleanup_forecast_staging` in the same `forecast` group: group FIFO guarantees archive and cleanup never run concurrently.
- Make target `forecast-staging-clean ARGS=...`.

## 6. Accuracy & FVA

### 6.1 What becomes measurable

With record month 2026-06 archived and June actuals loaded, lag-0 accuracy per algorithm is computable immediately.
Each subsequent month's actuals close one more lag: July actuals close lag 1, ..., November actuals close lag 5.
When July 2026 is archived (record month 2026-07), the lag matrix starts filling diagonally; after six cycles every lag has a measured value per model per record month - real forward accuracy alongside (not merged with) the simulated backtest curves.

### 6.2 API

Extend `api/routers/forecasting/fva.py` (read-only, `get_conn()`):

| Method | Path | Description |
|---|---|---|
| GET | `/fva/snapshot-accuracy?record_month=&lag=` | Per-model accuracy aggregated from `agg_accuracy_snapshot`: `{model_id, lag, forecast_month, n_dfus, accuracy_pct, wape, bias, fva_vs_naive_pts, n_dfus_common}` rows; omit `lag` for the full model x lag matrix of one record month |
| GET | `/fva/snapshot-months` | Distinct record months available, per-month closed-lag count, and the MV's last-refresh time (drives the UI selector and the freshness caption) |

- `accuracy_pct`, `wape`, `bias` per model are aggregated over that model's own covered DFUs (with `n_dfus` always returned).
- `fva_vs_naive_pts` is computed **only on the DFU intersection** between the model and `seasonal_naive` at the same `(record_month, lag)` - both sides re-aggregated over the common set (`n_dfus_common`), mirroring the common-DFUs path of `/forecast/accuracy/slice`.
  Coverage differs materially by model (ensemble 4,376 vs 12,306), so a whole-universe delta would be coverage bias, not forecast skill.
- `seasonal_naive` and `rolling_mean` are archived like any other candidate, so the baseline comes from the archive itself.
  **Definitional note:** the production `seasonal_naive` uses the most recent same-calendar-month occurrence from any prior year, while the existing `/fva/waterfall` naive stage is strict same-month-last-year computed on the fly; the two baselines are related but not identical and the two panels must not be presented as sharing one baseline.
- `champion` rows surface first; `ai_champion` appears when archived rows exist for the record month (sparse coverage expected - the matrix shows its `n_dfus`).

### 6.3 Frontend

New sub-panel `frontend/src/tabs/fva/SnapshotAccuracyPanel.tsx`, mounted in `FVATab.tsx` below the existing waterfall ladder:

- Record-month selector (from `/fva/snapshot-months`) and a model x lag matrix; closed lags show `accuracy_pct` **with the bias sign** (over/under direction has asymmetric cost - stockout vs write-off) and the vs-naive delta as a +/- pts badge; open lags render as pending.
- Every cell/row exposes coverage: `n_dfus`, and `n_dfus_common` on the delta badge; the vs-naive badge is suppressed (rendered as n/a with a tooltip) when the intersection is empty.
  Measured context: 27% of the 12,306 June-forecast DFUs had zero June actuals, so a low lag-0 accuracy without coverage and bias context would mislead.
- A caption shows the MV last-refresh time from `/fva/snapshot-months`.
- `champion` pinned first; `seasonal_naive` rendered as the baseline row.
- Fetchers `fetchFVASnapshotAccuracy` / `fetchFVASnapshotMonths` added to the FVA group in `frontend/src/api/queries/platform.ts`.
- `/fva` is already proxied in vite.config.ts; regenerate types via `npm run gen:types`.
- The existing waterfall ladder (../08-integration/07-fva.md) is unchanged: it remains backtest-based at execution lag; this panel adds the live-forward view alongside it.

### 6.4 Snapshot lag vs backtest lag (do not conflate)

`backtest_lag_archive` lag = true horizon from a simulated training cutoff (every DFU re-anchored per timeframe).
Snapshot lag = plan age from a single calendar record month, while each DFU's true horizon varies (`horizon_months`, 22.6% of June series originate before June).
The two curves answer different questions ("how good is the model at N months out" vs "how good was the plan of record N months before the target") and are shown in different panels with different labels.
`horizon_months` is preserved in both the table and the MV so a horizon-conditioned analysis (e.g. restrict to `horizon_months = lag + 1`) remains possible later without re-archiving.

## 7. Configuration

`config/forecasting/forecast_pipeline_config.yaml`:

```yaml
forecast_snapshot:
  horizon_months: 6        # archive lags 0..5 per record month
  active_window_months: 12 # DFU must have an actual within this window to be scored
```

## 8. Testing

| Test | Covers |
|---|---|
| `tests/unit/test_archive_forecast_snapshot.py` | Record-month derivation incl. empty-staging no-op, window predicates, DO NOTHING vs --overwrite, champion-absent warning vs plan_version-mismatch hard failure, ai_champion source (mocked psycopg) |
| `tests/unit/test_cleanup_forecast_staging.py` | Safety-gate arithmetic (pass/shortfall/force), default generation targeting, dry-run |
| `tests/api/test_fva.py` (extend) | `/fva/snapshot-accuracy` (matrix, common-DFU delta, null-KPI, empty archive) and `/fva/snapshot-months` via `make_pool` |
| `tests/api/test_backtest_management.py` (extend) | promote_model plan_version now derived from `get_planning_date()` |
| `tests/unit/test_mv_refresh.py` (existing, automatic) | Fails unless `agg_accuracy_snapshot` is registered in `MV_SOURCES` with sources matching the DDL |
| `frontend/src/tabs/__tests__/FVATab.test.tsx` (extend) + `fva/SnapshotAccuracyPanel.test.tsx` | Matrix rendering, pending lags, bias sign, coverage badges, suppressed delta on empty intersection |
| `tests/unit/test_pipeline_presets.py` (existing, automatic) | forecast-publish's new leading step references a registered job type |

## 9. Migration & rollout

1. `sql/200_create_forecast_snapshot.sql`: table + MV + indexes (idempotent `IF NOT EXISTS` per repo convention); `MV_SOURCES` Tier-1 entry in the same commit.
2. `promote_model()` plan_version source fix (4.1 prerequisite) in the same change, with its test.
3. Add `fact_forecast_snapshot` to the `db-truncate-data` Makefile transaction and to the verbatim SQL block in `docs/operations-manual/11-maintenance-troubleshooting.md`; document `forecast-archive` / `forecast-staging-clean` in that doc's Data Cleanup section.
4. Register the two job types; add Make targets; prepend the archive step to `forecast-publish`; optional disabled `default_schedules` entry.
5. Docs: `docs/ARCHITECTURE.md` fact-table catalog; this spec is indexed in `docs/specs/README.md` (row flagged **Proposed** until shipped); add the Implemented Features row only when shipped.

### 9.1 Initial run (June 2026)

1. Apply sql/200 (and the pending sql/198/199 if not yet applied - apply in numeric order).
2. `make forecast-archive ARGS="--record-month 2026-06"` - expect 1,281,468 candidate rows + 73,836 champion rows (+ any ai_champion rows; measured 2026-07-08); verify the per-model counts in the job log against the reconciliation in 4.3.
3. `refresh_for_tables` runs in-script; confirm `agg_accuracy_snapshot` holds lag-0 rows only (June actuals are loaded; July is not).
4. Review lag-0 accuracy in the new FVA panel: coverage badges should show 12,306 for full-coverage models, 4,376 for ensemble, and the ~27% zero-actual DFU share will surface as over-bias - expected, not a defect.
5. In the July cycle (planning month 2026-07), run `cleanup_forecast_staging` as its own workflow; the gate verifies the June archive before deleting the June generation (~5.3M staging rows, ~2 GB reclaimed).

## 10. Explicitly out of scope

- Archiving `fact_external_forecast_monthly` / `backtest_lag_archive` (owned by 03-backtest-framework.md and cleaned per ../07-user-experience/06-backtest-cleanup.md).
- Dimension-sliced snapshot accuracy (needs a fan-out-safe dim strategy at the item+loc grain; `dim_item` joins are safe when it comes).
- Pre-delete archiving of the outgoing plan inside the promote transaction itself.
  The leading pipeline step in 4.2 provides the archive-before-overwrite guarantee at the orchestration layer; moving it into the promote transaction is a possible hardening later.
- Horizon-conditioned accuracy surfaces (`horizon_months = lag + 1` restriction) - the data is preserved for it (6.4), the UI is not built.

## 11. Adjacent defects noted during design (not addressed here)

- `fact_candidate_forecast` has no writer anywhere in the repo, yet `POST /backtest-management/{model_id}/load` reports it as the target table (the submitted job actually loads `fact_external_forecast_monthly` + `backtest_lag_archive`).
- `generate_production_forecasts.py` `write_forecast()` (direct production-table upsert) is dead code - never called.
- A champion-mode generate stages rows under each true producing model_id but deletes only `WHERE model_id = 'champion'` beforehand, which can strand rows when the champion roster shrinks between runs.
